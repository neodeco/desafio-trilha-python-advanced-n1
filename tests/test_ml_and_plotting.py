from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import spark_predictive_model as ml_model
from scripts.spark_predictive_model import ModelTrainingError, train_predict_evaluate
from scripts.plotting import build_interactive_forecast_figure, plot_comparative_forecast


@pytest.fixture(autouse=True)
def _isolate_ml_output_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml_model, "ANALYSIS_OUTPUT_DIR", tmp_path / "analysis")
    monkeypatch.setattr(ml_model, "MODEL_TEST_OUTPUT_DIR", tmp_path / "model-test")
    monkeypatch.setattr(ml_model, "PROCESSED_DATA_OUTPUT_DIR", tmp_path / "processed_stock_data")


def test_train_predict_evaluate_uses_temporal_split_and_targets_r2_band() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    dataframe = pd.DataFrame({"ticker": ["TESTE"] * len(dates), "date": dates, "close": [100 + index * 0.5 for index in range(80)]})

    result = train_predict_evaluate(dataframe, future_days=365, source_name="TESTE")

    assert result.metrics["train_rows"] == 64
    assert result.metrics["test_rows"] == 16
    assert result.metrics["iterations"] >= 1
    assert result.metrics["epochs"] >= 1
    assert len(result.past_predictions) == 80
    assert len(result.future_predictions) == 365
    assert list(result.past_predictions.columns) == ["date", "close", "predicted"]
    assert list(result.future_predictions.columns) == ["date", "predicted"]
    assert result.past_predictions["date"].min() == dataframe["date"].min()
    assert result.past_predictions["date"].max() == dataframe["date"].max()
    assert result.future_predictions["date"].min() > dataframe["date"].max()

    assert list(pd.read_parquet(result.artifacts["train_parquet"]).columns) == ["ticker", "date", "close"]
    assert list(pd.read_parquet(result.artifacts["test_parquet"]).columns) == ["ticker", "date", "close"]
    assert ml_model.TARGET_R2_MIN <= float(result.metrics["train_r2"]) <= ml_model.TARGET_R2_MAX
    assert ml_model.TARGET_R2_MIN <= float(result.metrics["test_r2"]) <= ml_model.TARGET_R2_MAX
    assert float(result.metrics["test_r2"]) == float(result.metrics["r2"])
    assert result.metrics["train_target_reached"] is True
    assert result.metrics["test_target_reached"] is True
    assert result.metrics["target_reached"] is True

    for artifact_path in result.artifacts.values():
        assert Path(artifact_path).exists()


def test_train_predict_evaluate_requires_minimum_rows() -> None:
    dataframe = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq="D"), "close": [1, 2, 3, 4, 5]})

    with pytest.raises(ModelTrainingError):
        train_predict_evaluate(dataframe)


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
    assert figure.layout.xaxis.dtick == 15 * 24 * 60 * 60 * 1000
    assert "%b/%Y" in str(figure.layout.xaxis.tickformat)


def test_run_forecast_from_csv_reuses_cached_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    first = ml_model.run_forecast_from_csv(csv_path, source_name="AAPL")
    assert first.metrics["from_cache"] is False

    def _should_not_train(*args, **kwargs):
        raise AssertionError("training should not run when cache entry exists")

    monkeypatch.setattr(ml_model, "train_predict_evaluate", _should_not_train)
    second = ml_model.run_forecast_from_csv(csv_path, source_name="AAPL")

    assert second.metrics["from_cache"] is True
    assert second.artifacts["test_predictions"] == first.artifacts["test_predictions"]
