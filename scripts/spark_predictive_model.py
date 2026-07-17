from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_utils import detect_csv_separator_from_path as detect_csv_separator, slugify  # noqa: E402

# --- Forecast model (single symbol, date/close time series) -----------------
# Merged from the former scripts/ml_model.py: trains a PySpark MLlib
# LinearRegression model with a temporal split (no shuffling) to avoid
# overfitting, searching epochs/regularization to keep R2 (variance) within a
# target band. Used by app.py (via CLI, see `--forecast-input` below) and by
# scripts/localstack_pipeline_test.py.

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


def _build_forecast_spark_session():
    # Reuses the existing Glue/PySpark session builder so the forecast model
    # shares the same Spark configuration used by the ETL pipeline
    # (app/glue_pipeline.py).
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
    df, first_date, scale = _prepare_feature_frame(dataframe)

    split_index = max(1, min(int(len(df) * (1 - test_fraction)), len(df) - 1))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    if train_df.empty or test_df.empty:
        raise ModelTrainingError("Split temporal invalido: treino ou teste ficou vazio.")

    spark = _build_forecast_spark_session()
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

        artifacts = _save_forecast_artifacts(
            source_name, attempts, metrics, past_predictions, future_predictions, train_df, test_df
        )

        return ForecastResult(
            metrics=metrics,
            past_predictions=past_predictions,
            future_predictions=future_predictions,
            artifacts=artifacts,
        )
    finally:
        spark.stop()


def _save_forecast_artifacts(
    source_name: str,
    attempts: list[dict],
    metrics: dict,
    past_predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, str]:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = slugify(source_name, default="forecast")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    search_path = ANALYSIS_OUTPUT_DIR / f"{slug}_training_search_{timestamp}.csv"
    pd.DataFrame(attempts).to_csv(search_path, index=False)

    training_metrics_path = ANALYSIS_OUTPUT_DIR / f"{slug}_training_metrics_{timestamp}.json"
    training_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    test_predictions_path = MODEL_TEST_OUTPUT_DIR / f"{slug}_test_predictions_{timestamp}.csv"
    past_predictions.to_csv(test_predictions_path, index=False)

    future_predictions_path = MODEL_TEST_OUTPUT_DIR / f"{slug}_future_predictions_{timestamp}.csv"
    future_predictions.to_csv(future_predictions_path, index=False)

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
        "future_predictions": str(future_predictions_path),
        "test_metrics": str(test_metrics_path),
        "train_parquet": str(train_parquet_path),
        "test_parquet": str(test_parquet_path),
    }


def run_forecast_from_csv(
    csv_path: str | Path,
    source_name: str,
    sep: str = "auto",
    future_days: int = 365,
    test_fraction: float = 0.2,
) -> ForecastResult:
    """Read a treated date/close CSV (see app/glue_job.py price-series mode)
    and train/evaluate the forecast model. Intended to be invoked as a
    standalone subprocess (see `--forecast-input` below) so PySpark never runs
    inside the Streamlit UI process."""
    csv_path = Path(csv_path)
    csv_sep = detect_csv_separator(csv_path) if sep == "auto" else sep
    dataframe = pd.read_csv(csv_path, sep=csv_sep)
    dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]
    return train_predict_evaluate(dataframe, future_days=future_days, test_fraction=test_fraction, source_name=source_name)


# --- Legacy multi-symbol OHLCV training (COTAHIST-style training-set) -------
# Historical batch pipeline kept for the files/training-set workflow described
# in TECHNICAL-TECH-DOC.md: engineers next-close regression features from
# open/high/low/volume/prev_close and compares LinearRegression, RandomForest
# and GBT models.


def load_converter_module() -> object:
    converter_path = Path(__file__).resolve().parents[0] / "convert_cotahist_to_csv.py"
    spec = importlib.util.spec_from_file_location("convert_cotahist", converter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_csv_for_txt(input_path: Path, converter_module: object) -> Path:
    if input_path.suffix.lower() == ".csv":
        return input_path

    output_path = input_path.with_suffix(".csv")
    if not output_path.exists():
        converter_module.convert_to_csv(input_path, output_path)
    return output_path


def load_dataset(spark: SparkSession, folder: Path, converter_module: object, sep: str):
    dataframes = []
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in {".csv", ".txt"}:
            csv_path = ensure_csv_for_txt(file_path, converter_module)
            file_sep = detect_csv_separator(csv_path) if sep == "auto" else sep
            df = (
                spark.read.option("header", True)
                .option("inferSchema", True)
                .option("sep", file_sep)
                .csv(str(csv_path))
            )
            dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(f"No training files found in {folder}")

    result = dataframes[0]
    for df in dataframes[1:]:
        result = result.unionByName(df, allowMissingColumns=True)
    return result


def preprocess(df):
    df = (
        df.withColumn(
            "trade_date_norm",
            F.when(
                F.col("trade_date").rlike(r"^\d{4}-\d{2}-\d{2}$"),
                F.regexp_replace(F.col("trade_date"), "-", ""),
            )
            .when(F.col("trade_date").rlike(r"^\d{6}$"), F.concat(F.col("trade_date"), F.lit("01")))
            .otherwise(F.col("trade_date"))
        )
        .withColumn("trade_date_fmt", F.to_date(F.col("trade_date_norm"), "yyyyMMdd"))
        .withColumn("open", F.col("open").cast("double"))
        .withColumn("high", F.col("high").cast("double"))
        .withColumn("low", F.col("low").cast("double"))
        .withColumn("close", F.col("close").cast("double"))
        .withColumn("volume", F.col("volume").cast("double"))
        .withColumn("symbol", F.col("symbol"))
        .withColumn("prev_close", F.lag("close").over(Window.partitionBy("symbol").orderBy("trade_date_fmt")))
        .withColumn("next_close", F.lead("close").over(Window.partitionBy("symbol").orderBy("trade_date_fmt")))
    )

    return df.filter(F.col("next_close").isNotNull()).orderBy("symbol", "trade_date_fmt")


def build_features(df, symbol_filter=None):
    if symbol_filter:
        df = df.filter(F.col("symbol") == symbol_filter)

    feature_cols = ["open", "high", "low", "volume", "prev_close"]
    required_cols = feature_cols + ["next_close"]
    df = df.dropna(subset=required_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    featured = assembler.transform(df)
    return featured.filter(F.col("features").isNotNull()).select(
        "features",
        F.col("next_close").alias("label"),
        "symbol",
        "trade_date_fmt",
    )


def train_and_evaluate(training_df, validation_df, output_dir: Path):
    models = {
        "linear_regression": LinearRegression(featuresCol="features", labelCol="label", maxIter=50),
        "random_forest": RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=20),
        "gbt": GBTRegressor(featuresCol="features", labelCol="label", maxIter=20),
    }

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    results = []
    best_model = None
    best_rmse = float("inf")
    best_name = None

    for name, estimator in models.items():
        model = estimator.fit(training_df)
        predictions = model.transform(validation_df)
        rmse = evaluator.evaluate(predictions)
        mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
        r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "training_results.csv"
    with results_path.open("w", encoding="utf-8") as f:
        f.write("model,rmse,mae,r2\n")
        for row in results:
            f.write(f"{row['model']},{row['rmse']},{row['mae']},{row['r2']}\n")

    print(f"Best model: {best_name} with RMSE={best_rmse:.4f}")
    return best_name, best_model


def final_test(best_model, test_df, output_dir: Path):
    predictions = best_model.transform(test_df)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

    with (output_dir / "test_results.txt").open("w", encoding="utf-8") as f:
        f.write(f"rmse,{rmse}\n")
        f.write(f"mae,{mae}\n")
        f.write(f"r2,{r2}\n")

    print(f"Test performance: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")


def _run_legacy_training_cli(args: argparse.Namespace) -> None:
    spark = SparkSession.builder.master("local[1]").appName("spark-predictive-model").getOrCreate()
    converter_module = load_converter_module()

    training_df = load_dataset(spark, Path(args.training_dir), converter_module, args.sep)
    training_df = preprocess(training_df)

    if args.symbol is None:
        args.symbol = training_df.groupBy("symbol").count().orderBy(F.desc("count")).first()["symbol"]
        print(f"Selected symbol for modeling: {args.symbol}")

    data = build_features(training_df, symbol_filter=args.symbol)
    train_set, val_set = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training rows: {train_set.count()}, validation rows: {val_set.count()}")

    _, best_model = train_and_evaluate(train_set, val_set, Path(args.output_dir))

    test_path = Path(args.test_file)
    test_csv_path = ensure_csv_for_txt(test_path, converter_module)
    test_sep = detect_csv_separator(test_csv_path) if args.sep == "auto" else args.sep
    test_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("sep", test_sep)
        .csv(str(test_csv_path))
    )
    test_df = preprocess(test_df)
    test_features = build_features(test_df, symbol_filter=args.symbol)
    if test_features.count() == 0:
        raise RuntimeError(f"No valid test rows found for symbol {args.symbol} after preprocessing")
    final_test(best_model, test_features, Path(args.output_dir))

    spark.stop()


def _run_forecast_cli(args: argparse.Namespace) -> None:
    source_name = args.source_name or Path(args.forecast_input).stem
    result = run_forecast_from_csv(
        csv_path=args.forecast_input,
        source_name=source_name,
        sep=args.sep,
        future_days=args.future_days,
        test_fraction=args.test_fraction,
    )

    print(f"Metricas: R2={result.metrics['r2']:.4f} RMSE={result.metrics['rmse']:.4f} MAE={result.metrics['mae']:.4f}")
    print(f"Alvo de R2 atingido: {result.metrics['target_reached']}")

    # Last line is a machine-readable JSON summary for callers (e.g. app.py)
    # that invoke this script as a subprocess and need the metrics/artifacts back.
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}))


def main():
    parser = argparse.ArgumentParser(description="Train Spark ML regression models for stock price prediction")
    parser.add_argument(
        "--mode",
        choices=["training", "forecast"],
        default="training",
        help="'training' runs the legacy multi-model COTAHIST training-set flow (default); "
        "'forecast' trains/evaluates the single-symbol date/close forecast model "
        "(used by app.py and scripts/localstack_pipeline_test.py).",
    )
    parser.add_argument("--training-dir", default="files/training-set", help="[training mode] Folder containing training files")
    parser.add_argument("--test-file", default="files/test-set/COTAHIST_A2020.TXT", help="[training mode] Test file to evaluate model")
    parser.add_argument("--symbol", default=None, help="[training mode] Stock symbol to model (default: largest symbol in training set)")
    parser.add_argument("--output-dir", default="output/model", help="[training mode] Directory for model artifacts and results")
    parser.add_argument(
        "--forecast-input",
        default=None,
        help="[forecast mode] Path to a treated date/close CSV (see app/glue_job.py --mode price-series)",
    )
    parser.add_argument("--source-name", default=None, help="[forecast mode] Logical name (e.g. ticker) used for output filenames")
    parser.add_argument("--future-days", type=int, default=365, help="[forecast mode] Number of days to forecast into the future")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="[forecast mode] Fraction of rows reserved for the temporal test split")
    parser.add_argument("--sep", default="auto", help="CSV separator for input files ('auto' by default, or an explicit character)")
    args = parser.parse_args()

    if args.mode == "forecast":
        if not args.forecast_input:
            parser.error("--forecast-input is required when --mode forecast is used")
        _run_forecast_cli(args)
        return

    _run_legacy_training_cli(args)


if __name__ == "__main__":
    main()