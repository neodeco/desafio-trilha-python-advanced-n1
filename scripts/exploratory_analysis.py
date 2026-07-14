import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_processed_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trade_date_fmt"] = pd.to_datetime(df["trade_date_fmt"], errors="coerce")
    df = df.sort_values(["symbol", "trade_date_fmt"])
    return df


def create_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_price_trends(df: pd.DataFrame, output_dir: Path, top_n: int = 6) -> None:
    symbols = df["symbol"].value_counts().nlargest(top_n).index.tolist()
    subset = df[df["symbol"].isin(symbols)]

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=subset, x="trade_date_fmt", y="close", hue="symbol", marker="o")
    plt.title("Stock Closing Price Trends")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(title="Symbol", loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "stock_price_trends.png")
    plt.close()


def plot_volume_trends(df: pd.DataFrame, output_dir: Path, top_n: int = 6) -> None:
    symbols = df["symbol"].value_counts().nlargest(top_n).index.tolist()
    subset = df[df["symbol"].isin(symbols)]

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=subset, x="trade_date_fmt", y="volume", hue="symbol", marker="o")
    plt.title("Trading Volume Trends")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend(title="Symbol", loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "volume_trends.png")
    plt.close()


def plot_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols = ["open", "high", "low", "close", "volume", "daily_pct_change"]
    correlation = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix for Stock Variables")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()


def save_summary(df: pd.DataFrame, output_dir: Path) -> None:
    summary = df.describe(include="all")
    summary_path = output_dir / "data_summary.csv"
    summary.to_csv(summary_path)

    symbol_counts = df["symbol"].value_counts().rename_axis("symbol").reset_index(name="count")
    symbol_counts.to_csv(output_dir / "symbol_counts.csv", index=False)


def main() -> None:
    source_path = Path(os.getenv("PROCESSED_DATA_PATH", "output/processed_cotahist.parquet"))
    output_dir = Path(os.getenv("ANALYSIS_OUTPUT_PATH", "output/analysis"))
    create_output_dir(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Processed data not found at {source_path}")

    df = load_processed_data(source_path)
    print(f"Loaded {len(df)} rows from {source_path}")

    save_summary(df, output_dir)
    plot_price_trends(df, output_dir)
    plot_volume_trends(df, output_dir)
    plot_correlations(df, output_dir)

    print(f"Exploratory analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
