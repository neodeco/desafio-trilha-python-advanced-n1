"""Create comparative time-series plots for a symbol.

Reads CSV files (training-set) and plots open/high/low/close with volume on a
secondary axis. Saves PNG to `output/plots` by default.

Usage:
  python scripts/comparative_series.py --symbol PETR4T --input-dir files/training-set --output-dir output/plots
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_csvs(folder: Path) -> pd.DataFrame:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    dfs = []
    for p in csvs:
        try:
            df = pd.read_csv(p, parse_dates=["trade_date"], dayfirst=False, dtype={"symbol": str})
            dfs.append(df)
        except Exception:
            # Skip unreadable CSVs
            continue
    return pd.concat(dfs, ignore_index=True)


def plot_symbol(df: pd.DataFrame, symbol: str, out_dir: Path):
    df_symbol = df[df["symbol"] == symbol].copy()
    if df_symbol.empty:
        raise ValueError(f"No rows for symbol {symbol}")

    # normalize trade_date column to datetime
    if df_symbol["trade_date"].dtype == object:
        # some files use YYYYMM or YYYYMMDD formats
        df_symbol["trade_date"] = pd.to_datetime(df_symbol["trade_date"], errors="coerce")
    df_symbol = df_symbol.dropna(subset=["trade_date"]).sort_values("trade_date")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_symbol["trade_date"], df_symbol["open"], label="open")
    ax.plot(df_symbol["trade_date"], df_symbol["high"], label="high")
    ax.plot(df_symbol["trade_date"], df_symbol["low"], label="low")
    ax.plot(df_symbol["trade_date"], df_symbol["close"], label="close")
    ax.set_xlabel("trade_date")
    ax.set_ylabel("price")
    ax.set_title(f"Price series for {symbol}")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.bar(df_symbol["trade_date"], df_symbol["volume"].astype(float) / 1e6, alpha=0.2, color="gray", label="volume (M)")
    ax2.set_ylabel("volume (millions)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"comparative_{symbol}.png"
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--input-dir", default="files/training-set")
    p.add_argument("--output-dir", default="output/plots")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_csvs(Path(args.input_dir))
    plot_symbol(df, args.symbol, Path(args.output_dir))


if __name__ == "__main__":
    main()
