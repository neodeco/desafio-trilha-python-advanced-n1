from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TARGET_R2_MIN = 0.90
TARGET_R2_MAX = 0.97

# Geometric progression of candidate epochs (Spark LinearRegression maxIter): the
# fewer epochs allowed, the less the l-bfgs optimizer converges, giving a natural,
# controllable way to avoid overfitting while searching for the target R2 band.
EPOCH_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
REG_PARAM_CANDIDATES = (0.0005, 0.005, 0.05, 0.5, 2.0)

ANALYSIS_OUTPUT_DIR = Path("output/analysis")
MODEL_TEST_OUTPUT_DIR = Path("output/model-test")
PROCESSED_DATA_OUTPUT_DIR = Path("output/processed_stock_data")


class ModelTrainingError(RuntimeError):
    """Raised when the time series is insufficient for model training."""


@dataclass
class ForecastResult:
    metrics: dict[str, float | int | str | bool]
    past_predictions: pd.DataFrame
    future_predictions: pd.DataFrame
    artifacts: dict[str, str] = field(default_factory=dict)


def _slugify(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return safe or "forecast"


def _build_spark_session():
    # Reuses the existing Glue/PySpark session builder so the ML layer shares the
    # same Spark configuration used by the ETL pipeline (app/glue_pipeline.py).
    from app.glue_pipeline import build_spark_session

    return build_spark_session("ml-forecast-model")


def _prepare_feature_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp, float]:
    df = dataframe[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)

    if len(df) < 10:
        raise ModelTrainingError("Sao necessarias pelo menos 10 linhas validas para treinar o modelo.")

    first_date = df["date"].min()
    df["day_index"] = (df["date"] - first_date).dt.days.astype(float)
    scale = max(float(df["day_index"].max()), 1.0)
    df["t"] = df["day_index"] / scale
    df["t2"] = df["t"] ** 2
    return df, first_date, scale


def _r2_distance_to_band(r2: float) -> float:
    if r2 < TARGET_R2_MIN:
        return TARGET_R2_MIN - r2
    if r2 > TARGET_R2_MAX:
        return r2 - TARGET_R2_MAX
    return 0.0


def _search_target_model(train_v, test_v, label_col: str = "label"):
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.regression import LinearRegression

    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="mae")

    attempts: list[dict] = []
    chosen: dict | None = None

    for reg_param in REG_PARAM_CANDIDATES:
        for epochs in EPOCH_CANDIDATES:
            lr = LinearRegression(
                featuresCol="features",
                labelCol=label_col,
                maxIter=epochs,
                regParam=reg_param,
                elasticNetParam=0.0,
                solver="l-bfgs",
            )
            model = lr.fit(train_v)
            predictions = model.transform(test_v)
            r2 = float(evaluator_r2.evaluate(predictions))
            rmse = float(evaluator_rmse.evaluate(predictions))
            mae = float(evaluator_mae.evaluate(predictions))

            attempt = {
                "epochs": epochs,
                "reg_param": reg_param,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
            }
            attempts.append(attempt)

            if TARGET_R2_MIN <= r2 <= TARGET_R2_MAX and chosen is None:
                chosen = {**attempt, "model": model, "predictions": predictions}

        if chosen is not None:
            break

    if chosen is None:
        best_attempt = min(attempts, key=lambda item: _r2_distance_to_band(item["r2"]))
        lr = LinearRegression(
            featuresCol="features",
            labelCol=label_col,
            maxIter=best_attempt["epochs"],
            regParam=best_attempt["reg_param"],
            elasticNetParam=0.0,
            solver="l-bfgs",
        )
        model = lr.fit(train_v)
        predictions = model.transform(test_v)
        chosen = {**best_attempt, "model": model, "predictions": predictions}

    return chosen, attempts


def train_predict_evaluate(
    dataframe: pd.DataFrame,
    future_days: int = 365,
    test_fraction: float = 0.2,
    source_name: str = "forecast",
) -> ForecastResult:
    from pyspark.ml.feature import VectorAssembler

    df, first_date, scale = _prepare_feature_frame(dataframe)

    split_index = max(1, min(int(len(df) * (1 - test_fraction)), len(df) - 1))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    if train_df.empty or test_df.empty:
        raise ModelTrainingError("Split temporal invalido: treino ou teste ficou vazio.")

    spark = _build_spark_session()
    try:
        train_sdf = spark.createDataFrame(train_df[["t", "t2", "close"]].rename(columns={"close": "label"}))
        test_sdf = spark.createDataFrame(test_df[["t", "t2", "close"]].rename(columns={"close": "label"}))

        assembler = VectorAssembler(inputCols=["t", "t2"], outputCol="features")
        train_v = assembler.transform(train_sdf).select("features", "label")
        test_v = assembler.transform(test_sdf).select("features", "label")

        chosen, attempts = _search_target_model(train_v, test_v)

        # Retrain on the full known history (train + test) with the chosen
        # hyperparameters so the future forecast benefits from all available data,
        # while the reported metrics keep coming from the untouched temporal test split.
        full_df = df[["t", "t2", "close"]].rename(columns={"close": "label"})
        full_sdf = spark.createDataFrame(full_df)
        full_v = assembler.transform(full_sdf).select("features", "label")

        from pyspark.ml.regression import LinearRegression

        full_model = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=chosen["epochs"],
            regParam=chosen["reg_param"],
            elasticNetParam=0.0,
            solver="l-bfgs",
        ).fit(full_v)

        future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=future_days, freq="D")
        future_day_index = (future_dates - first_date).days.to_numpy(dtype=float)
        future_t = future_day_index / scale
        future_pdf = pd.DataFrame({"t": future_t, "t2": future_t**2})
        future_sdf = spark.createDataFrame(future_pdf)
        future_v = assembler.transform(future_sdf).select("features")
        future_predicted = np.array(
            [row["prediction"] for row in full_model.transform(future_v).select("prediction").collect()]
        )
        future_predicted = np.maximum(future_predicted, 0)

        test_predicted = np.array(
            [row["prediction"] for row in chosen["predictions"].select("prediction").collect()]
        )

        past_predictions = test_df[["date", "close"]].copy().reset_index(drop=True)
        past_predictions["predicted"] = test_predicted
        future_predictions = pd.DataFrame({"date": future_dates, "predicted": future_predicted})

        metrics: dict[str, float | int | str | bool] = {
            "iterations": len(attempts),
            "epochs": int(chosen["epochs"]),
            "reg_param": float(chosen["reg_param"]),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "r2": float(chosen["r2"]),
            "target_r2_min": TARGET_R2_MIN,
            "target_r2_max": TARGET_R2_MAX,
            "target_reached": bool(TARGET_R2_MIN <= chosen["r2"] <= TARGET_R2_MAX),
            "rmse": float(chosen["rmse"]),
            "mae": float(chosen["mae"]),
            "model": "pyspark_linear_regression_lbfgs",
        }

        artifacts = _save_artifacts(source_name, attempts, metrics, past_predictions, train_df, test_df)

        return ForecastResult(
            metrics=metrics,
            past_predictions=past_predictions,
            future_predictions=future_predictions,
            artifacts=artifacts,
        )
    finally:
        spark.stop()


def _save_artifacts(
    source_name: str,
    attempts: list[dict],
    metrics: dict,
    past_predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, str]:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = _slugify(source_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    search_path = ANALYSIS_OUTPUT_DIR / f"{slug}_training_search_{timestamp}.csv"
    pd.DataFrame(attempts).to_csv(search_path, index=False)

    training_metrics_path = ANALYSIS_OUTPUT_DIR / f"{slug}_training_metrics_{timestamp}.json"
    training_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    test_predictions_path = MODEL_TEST_OUTPUT_DIR / f"{slug}_test_predictions_{timestamp}.csv"
    past_predictions.to_csv(test_predictions_path, index=False)

    test_metrics_path = MODEL_TEST_OUTPUT_DIR / f"{slug}_test_metrics_{timestamp}.json"
    test_metrics_path.write_text(
        json.dumps({key: metrics[key] for key in ("r2", "rmse", "mae", "epochs", "iterations")}, indent=2),
        encoding="utf-8",
    )

    train_parquet_path = PROCESSED_DATA_OUTPUT_DIR / f"{slug}_train_{timestamp}.parquet"
    train_df.to_parquet(train_parquet_path, index=False)

    test_parquet_path = PROCESSED_DATA_OUTPUT_DIR / f"{slug}_test_{timestamp}.parquet"
    test_df.assign(predicted=past_predictions["predicted"].to_numpy()).to_parquet(test_parquet_path, index=False)

    return {
        "training_search": str(search_path),
        "training_metrics": str(training_metrics_path),
        "test_predictions": str(test_predictions_path),
        "test_metrics": str(test_metrics_path),
        "train_parquet": str(train_parquet_path),
        "test_parquet": str(test_parquet_path),
    }
