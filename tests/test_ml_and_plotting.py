from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import spark_predictive_model as ml_model
from scripts.spark_predictive_model import ModelTrainingError, _load_forecast_cache_index, _register_forecast_cache, train_predict_evaluate
from scripts.plotting import build_interactive_forecast_figure, plot_comparative_forecast


@pytest.fixture(autouse=True)
def _isolate_ml_output_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_model, "ANALYSIS_OUTPUT_DIR", tmp_path / "analysis")
    monkeypatch.setattr(ml_model, "MODEL_TEST_OUTPUT_DIR", tmp_path / "model-test")
    monkeypatch.setattr(ml_model, "PROCESSED_DATA_OUTPUT_DIR", tmp_path / "processed_stock_data")


def test_train_predict_evaluate_uses_temporal_split_and_targets_r2_band() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    dataframe = pd.DataFrame(
        {
            "ticker": ["TESTE"] * len(dates),
            "date": dates,
            "close": [100 + index * 0.5 for index in range(80)],
            "high": [101 + index * 0.5 for index in range(80)],
            "low": [99 + index * 0.5 for index in range(80)],
            "volume": [1000 + index * 10 for index in range(80)],
        }
    )

    result = train_predict_evaluate(dataframe, future_days=365, source_name="TESTE")

    assert result.metrics["train_rows"] == 64
    assert result.metrics["test_rows"] == 16
    assert result.metrics["iterations"] >= 1
    assert result.metrics["epochs"] >= 1
    assert len(result.past_predictions) == 16
    assert len(result.future_predictions) == 365
    assert result.metrics["future_days"] == 365
    assert list(result.past_predictions.columns) == ["date", "close", "predicted"]
    assert list(result.future_predictions.columns) == ["date", "predicted"]
    assert result.future_predictions["date"].min() > dataframe["date"].max()

    assert list(pd.read_parquet(result.artifacts["train_parquet"]).columns) == ["ticker", "data", "fechamento", "mínima", "máxima", "volume"]
    assert list(pd.read_parquet(result.artifacts["test_parquet"]).columns) == ["ticker", "data", "fechamento", "mínima", "máxima", "volume"]
    assert ml_model.TARGET_R2_MIN <= float(result.metrics["train_r2"]) <= ml_model.TARGET_R2_MAX
    assert ml_model.TARGET_R2_MIN <= float(result.metrics["test_r2"]) <= ml_model.TARGET_R2_MAX
    assert float(result.metrics["test_r2"]) == float(result.metrics["r2"])
    assert result.metrics["train_target_reached"] is True
    assert result.metrics["test_target_reached"] is True
    assert result.metrics["target_reached"] is True
    assert "baseline_naive_rmse" in result.metrics
    assert "model_beats_naive_rmse" in result.metrics
    assert "backtest_folds" in result.metrics
    assert "backtest_summary" in result.metrics
    assert isinstance(result.metrics["backtest_folds"], list)

    for artifact_path in result.artifacts.values():
        assert Path(artifact_path).exists()


def test_train_predict_evaluate_requires_minimum_rows() -> None:
    dataframe = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq="D"), "close": [1, 2, 3, 4, 5]})

    with pytest.raises(ModelTrainingError):
        train_predict_evaluate(dataframe)


def test_register_forecast_cache_keeps_up_to_seven_entries_per_ticker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_model, "FORECAST_CACHE_INDEX_PATH", tmp_path / "forecast_cache_index.json")

    for index in range(10):
        _register_forecast_cache(f"cache-{index}", {"training_metrics": f"metrics-{index}.json"}, source_name="AAPL")

    index = _load_forecast_cache_index()
    ticker_history = index["ticker_history"]["AAPL"]

    assert len(ticker_history) == 7
    assert [entry["cache_key"] for entry in ticker_history] == [
        "cache-3",
        "cache-4",
        "cache-5",
        "cache-6",
        "cache-7",
        "cache-8",
        "cache-9",
    ]


def test_plot_comparative_forecast_saves_png(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    actual = pd.DataFrame({"date": dates, "close": range(20)})
    past = pd.DataFrame({"date": dates[-5:], "close": range(15, 20), "predicted": range(15, 20)})
    future = pd.DataFrame(
        {
            "date": pd.date_range(dates.max() + pd.Timedelta(days=1), periods=365, freq="D"),
            "predicted": range(365),
        }
    )

    fig, path = plot_comparative_forecast(actual, past, future, output_dir=tmp_path, filename_prefix="AAPL")

    try:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
    finally:
        fig.clear()


def test_build_interactive_forecast_figure_saves_html(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    actual = pd.DataFrame({"date": dates, "close": range(20)})
    past = pd.DataFrame({"date": dates[-5:], "close": range(15, 20), "predicted": range(15, 20)})
    future = pd.DataFrame(
        {
            "date": pd.date_range(dates.max() + pd.Timedelta(days=1), periods=365, freq="D"),
            "predicted": range(365),
        }
    )

    figure, path = build_interactive_forecast_figure(actual, past, future, output_dir=tmp_path, filename_prefix="AAPL")

    assert path.exists()
    assert path.suffix == ".html"
    assert path.stat().st_size > 0
    assert len(figure.data) == 3
    assert figure.data[2].name == "Predicao futura (365 dias)"
    assert figure.layout.xaxis.dtick == 15 * 24 * 60 * 60 * 1000
    assert "%b/%Y" in str(figure.layout.xaxis.tickformat)


def test_run_forecast_from_csv_allows_up_to_seven_reprocessings_then_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ml_model, "ANALYSIS_OUTPUT_DIR", tmp_path / "analysis")
    monkeypatch.setattr(ml_model, "MODEL_TEST_OUTPUT_DIR", tmp_path / "model-test")
    monkeypatch.setattr(ml_model, "PROCESSED_DATA_OUTPUT_DIR", tmp_path / "processed_stock_data")
    monkeypatch.setattr(ml_model, "FORECAST_CACHE_INDEX_PATH", (tmp_path / "analysis" / "forecast_cache_index.json"))

    csv_path = tmp_path / "treated.csv"
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "close": [10 + index * 0.2 for index in range(12)],
        }
    )
    dataframe.to_csv(csv_path, index=False, sep=";")

    call_count = {"value": 0}

    def _fake_train_predict_evaluate(dataframe, future_days=365, test_fraction=0.2, source_name="forecast"):
        call_count["value"] += 1
        run_number = call_count["value"]
        run_dir = tmp_path / "fake-artifacts" / f"run-{run_number}"
        run_dir.mkdir(parents=True, exist_ok=True)

        training_metrics_path = run_dir / "training_metrics.json"
        test_predictions_path = run_dir / "test_predictions.csv"
        future_predictions_path = run_dir / "future_predictions.csv"

        training_metrics_path.write_text(
            json.dumps(
                {
                    "from_cache": False,
                    "test_r2": 0.7,
                    "train_r2": 0.7,
                    "iterations": 1,
                    "epochs": 2,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "date": ["2024-01-10"],
                "close": [12.0],
                "predicted": [12.1],
            }
        ).to_csv(test_predictions_path, index=False)
        pd.DataFrame(
            {
                "date": ["2024-01-11"],
                "predicted": [12.2],
            }
        ).to_csv(future_predictions_path, index=False)

        artifacts = {
            "training_metrics": str(training_metrics_path),
            "test_predictions": str(test_predictions_path),
            "future_predictions": str(future_predictions_path),
        }
        metrics = {
            "from_cache": False,
            "iterations": 1,
            "epochs": 2,
            "train_r2": 0.7,
            "test_r2": 0.7,
        }
        return ml_model.ForecastResult(
            metrics=metrics,
            past_predictions=pd.read_csv(test_predictions_path),
            future_predictions=pd.read_csv(future_predictions_path),
            artifacts=artifacts,
        )

    monkeypatch.setattr(ml_model, "train_predict_evaluate", _fake_train_predict_evaluate)

    first = ml_model.run_forecast_from_csv(csv_path, source_name="AAPL")
    assert first.metrics["from_cache"] is False
    assert first.metrics["action_reprocess_count"] == 1

    for expected_count in range(2, 8):
        result = ml_model.run_forecast_from_csv(csv_path, source_name="AAPL")
        assert result.metrics["from_cache"] is False
        assert result.metrics["action_reprocess_count"] == expected_count
        assert result.metrics["cache_limit_reached"] is False

    eighth = ml_model.run_forecast_from_csv(csv_path, source_name="AAPL")
    assert call_count["value"] == 7
    assert eighth.metrics["from_cache"] is True
    assert eighth.metrics["action_reprocess_count"] == 7
    assert eighth.metrics["cache_limit_reached"] is True
