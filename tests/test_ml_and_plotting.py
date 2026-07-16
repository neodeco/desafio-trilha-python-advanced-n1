from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.ml_model import train_predict_evaluate
from scripts.plotting import plot_comparative_forecast


def test_train_predict_evaluate_uses_temporal_split_and_future_horizon() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    dataframe = pd.DataFrame({"Date": dates, "Close": [100 + index * 0.5 for index in range(80)]})

    result = train_predict_evaluate(dataframe, future_days=365)

    assert result.metrics["train_rows"] == 64
    assert result.metrics["test_rows"] == 16
    assert result.metrics["iterations"] == 1
    assert len(result.past_predictions) == 16
    assert len(result.future_predictions) == 365
    assert result.future_predictions["Date"].min() > dataframe["Date"].max()
    assert float(result.metrics["r2"]) > 0.9


def test_plot_comparative_forecast_saves_png(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    actual = pd.DataFrame({"Date": dates, "Close": range(20)})
    past = pd.DataFrame({"Date": dates[-5:], "Close": range(15, 20), "Predicted": range(15, 20)})
    future = pd.DataFrame(
        {
            "Date": pd.date_range(dates.max() + pd.Timedelta(days=1), periods=365, freq="D"),
            "Predicted": range(365),
        }
    )

    fig, path = plot_comparative_forecast(actual, past, future, output_dir=tmp_path, filename_prefix="AAPL")

    try:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
    finally:
        fig.clear()
