from __future__ import annotations

import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def _safe_prefix(filename_prefix: str) -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in filename_prefix).strip("_")
    return safe_prefix or "forecast"


def plot_comparative_forecast(
    actual_df: pd.DataFrame,
    past_predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    output_dir: str | Path = "output/plots",
    filename_prefix: str = "forecast",
    ticker: str | None = None,
):
    """Static comparative chart: red = actual data, black = 365-day future forecast,
    with the past (test-period) prediction overlaid for comparison."""
    actual = actual_df.copy()
    past = past_predictions.copy()
    future = future_predictions.copy()

    for dataframe in (actual, past, future):
        dataframe["date"] = pd.to_datetime(dataframe["date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual["date"], actual["close"], color="red", linewidth=2, label="Dados reais")
    ax.plot(
        past["date"],
        past["predicted"],
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label="Predicao passada (teste)",
    )
    ax.plot(
        future["date"],
        future["predicted"],
        color="black",
        linewidth=2,
        label="Predicao futura (30 dias)",
    )

    ax.set_xlabel("Ano")
    ax.set_ylabel("Preco")
    title_suffix = f" - {ticker}" if ticker else ""
    ax.set_title(f"Comparativo de preco real e predicoes{title_suffix}")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_prefix = _safe_prefix(filename_prefix)
    file_path = output_path / f"{safe_prefix}_comparative.png"

    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    return fig, file_path


def build_interactive_forecast_figure(
    actual_df: pd.DataFrame,
    past_predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    output_dir: str | Path = "output/plots",
    filename_prefix: str = "forecast",
    ticker: str | None = None,
) -> tuple[go.Figure, Path]:
    """Interactive Plotly chart for economists to zoom/hover the final prediction:
    red = actual data, black = 365-day future forecast, dashed = past prediction."""
    actual = actual_df.copy()
    past = past_predictions.copy()
    future = future_predictions.copy()

    for dataframe in (actual, past, future):
        dataframe["date"] = pd.to_datetime(dataframe["date"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["close"],
            mode="lines",
            name="Dados reais",
            line={"color": "red", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=past["date"],
            y=past["predicted"],
            mode="lines",
            name="Predicao passada (teste)",
            line={"color": "#1f77b4", "width": 2, "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=future["date"],
            y=future["predicted"],
            mode="lines",
            name="Predicao futura (30 dias)",
            line={"color": "black", "width": 2},
        )
    )

    title_suffix = f" - {ticker}" if ticker else ""
    figure.update_layout(
        title=f"Predicao final de precos (analise interativa){title_suffix}",
        xaxis_title="Data (marcacao de 15 em 15 dias)",
        yaxis_title="Preco",
        xaxis={
            "tickformat": "%d/%m<br>%b/%Y",
            "dtick": 15 * 24 * 60 * 60 * 1000,
            "tickangle": -30,
            "rangeslider": {"visible": True},
        },
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        template="plotly_white",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_prefix = _safe_prefix(filename_prefix)
    file_path = output_path / f"{safe_prefix}_interactive.html"
    figure.write_html(str(file_path))

    return figure, file_path
