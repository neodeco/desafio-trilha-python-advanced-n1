"""Create comparative time-series plots for a symbol.

Reads CSV files (training-set) and plots open/high/low/close with volume on a
secondary axis. Saves PNG to `output/plots` by default.

Usage:
  python scripts/comparative_series.py --symbol PETR4T --input-dir files/training-set --output-dir output/plots
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def detect_csv_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        return ";" if sample.count(";") > sample.count(",") else ","


def normalize_trade_date(series: pd.Series) -> pd.Series:
    as_text = series.astype(str).str.strip()
    as_text = as_text.where(as_text.str.len() != 6, as_text + "01")
    return pd.to_datetime(as_text, errors="coerce")


def load_csvs(folder: Path, sep: str = ";") -> pd.DataFrame:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs = []
    for p in csvs:
        try:
            current_sep = detect_csv_separator(p) if sep == "auto" else sep
            df = pd.read_csv(p, sep=current_sep, dtype={"symbol": str})
            if "trade_date" in df.columns:
                df["trade_date"] = normalize_trade_date(df["trade_date"])
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        raise RuntimeError(f"CSV files were found in {folder}, but none could be parsed.")
    return pd.concat(dfs, ignore_index=True)


def plot_symbol(df: pd.DataFrame, symbol: str, out_dir: Path):
    df_symbol = df[df["symbol"] == symbol].copy()
    if df_symbol.empty:
        raise ValueError(f"No rows for symbol {symbol}")

    if df_symbol["trade_date"].dtype == object:
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
    ax2.bar(
        df_symbol["trade_date"],
        df_symbol["volume"].astype(float) / 1e6,
        alpha=0.2,
        color="gray",
        label="volume (M)",
    )
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
    p.add_argument("--sep", default=";", help="CSV separator (';' by default, or 'auto')")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_csvs(Path(args.input_dir), sep=args.sep)
    plot_symbol(df, args.symbol, Path(args.output_dir))


if __name__ == "__main__":
    main()