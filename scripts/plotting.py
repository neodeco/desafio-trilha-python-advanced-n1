from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_comparative_forecast(
    actual_df: pd.DataFrame,
    past_predictions: pd.DataFrame,
    future_predictions: pd.DataFrame,
    output_dir: str | Path = "output/plots",
    filename_prefix: str = "forecast",
):
    actual = actual_df.copy()
    past = past_predictions.copy()
    future = future_predictions.copy()

    for dataframe in (actual, past, future):
        dataframe["Date"] = pd.to_datetime(dataframe["Date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual["Date"], actual["Close"], color="red", linewidth=2, label="Dados reais")
    ax.plot(
        past["Date"],
        past["Predicted"],
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label="Predicao passada",
    )
    ax.plot(
        future["Date"],
        future["Predicted"],
        color="black",
        linewidth=2,
        label="Predicao futura (365 dias)",
    )

    ax.set_xlabel("Ano/Data")
    ax.set_ylabel("Preco")
    ax.set_title("Comparativo de preco real e predicoes")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in filename_prefix).strip("_")
    file_path = output_path / f"{safe_prefix or 'forecast'}_comparative.png"

    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    return fig, file_path
