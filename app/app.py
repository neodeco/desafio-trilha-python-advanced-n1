import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock Data Dashboard", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(path_str: str) -> pd.DataFrame:
    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parent
    input_path = Path(path_str)

    candidates = []
    for base in (Path.cwd(), project_root):
        candidates.extend(
            [
                base / input_path,
                base / "output/processed_stock_data",
                base / "output/processed_stock_data.parquet",
                base / "output/processed_cotahist.parquet",
            ]
        )

    # remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_resolved = c.resolve()
        if c_resolved not in seen:
            seen.add(c_resolved)
            unique_candidates.append(c_resolved)

    existing_path = next((p for p in unique_candidates if p.exists()), None)
    if existing_path is None:
        searched = ", ".join(str(p) for p in unique_candidates)
        raise FileNotFoundError(f"Data file/path not found. Tried: {searched}")

    if existing_path.is_dir() or existing_path.suffix.lower() in {".parquet", ""}:
        return pd.read_parquet(existing_path)

    if existing_path.suffix.lower() == ".csv":
        return pd.read_csv(existing_path)

    raise ValueError("Unsupported file format. Use .csv or parquet path.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [c.strip() for c in renamed.columns]

    expected_numeric = ["open", "high", "low", "close", "volume", "daily_pct_change", "prev_close"]
    for col in expected_numeric:
        if col in renamed.columns:
            renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    if "trade_date_fmt" in renamed.columns:
        renamed["trade_date_fmt"] = pd.to_datetime(renamed["trade_date_fmt"], errors="coerce")
    elif "trade_date" in renamed.columns:
        renamed["trade_date_fmt"] = pd.to_datetime(renamed["trade_date"], errors="coerce")
    else:
        renamed["trade_date_fmt"] = pd.NaT

    if "symbol" not in renamed.columns:
        renamed["symbol"] = "UNKNOWN"

    return renamed


def apply_filters(df: pd.DataFrame, selected_symbols: list[str], date_min, date_max) -> pd.DataFrame:
    filtered = df.copy()

    if selected_symbols:
        filtered = filtered[filtered["symbol"].isin(selected_symbols)]

    if "trade_date_fmt" in filtered.columns and filtered["trade_date_fmt"].notna().any():
        filtered = filtered[
            (filtered["trade_date_fmt"] >= pd.to_datetime(date_min))
            & (filtered["trade_date_fmt"] <= pd.to_datetime(date_max))
        ]

    return filtered.sort_values(["symbol", "trade_date_fmt"])


st.title("📊 Interactive Stock Data Dashboard")

with st.sidebar:
    st.header("Data Source")
    default_path = os.getenv("PROCESSED_DATA_PATH", "output/processed_stock_data")
    data_path: str | None = st.text_input("Input data path (.csv or parquet)", value=default_path)
    refresh = st.button("Reload data")

if refresh:
    st.cache_data.clear()

try:
    raw_df = load_data(data_path)
    df = normalize_columns(raw_df)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if df.empty:
    st.warning("Loaded dataset is empty.")
    st.stop()

symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist())
default_symbols = symbols[: min(8, len(symbols))] if symbols else []

with st.container(border=True):
    st.subheader("Filters")
    col1, col2 = st.columns(2)

    with col1:
        selected_symbols = st.multiselect("Symbols", options=symbols, default=default_symbols)

    with col2:
        has_dates = df["trade_date_fmt"].notna().any()
        if has_dates:
            min_date = df["trade_date_fmt"].min().date()
            max_date = df["trade_date_fmt"].max().date()
            selected_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(selected_range, tuple) and len(selected_range) == 2:
                date_min, date_max = selected_range
            else:
                date_min, date_max = min_date, max_date
        else:
            today = pd.Timestamp.today().date()
            date_min, date_max = today, today
            st.info("No valid date column found. Date filter disabled.")

filtered_df = apply_filters(df, selected_symbols, date_min, date_max)

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{len(filtered_df):,}")
metric_cols[1].metric("Symbols", f"{filtered_df['symbol'].nunique():,}")
if "close" in filtered_df.columns:
    metric_cols[2].metric("Avg Close", f"{filtered_df['close'].mean():.2f}")
if "volume" in filtered_df.columns:
    metric_cols[3].metric("Total Volume", f"{filtered_df['volume'].sum():,.0f}")

tab1, tab2, tab3 = st.tabs(["📈 Trends", "📋 Data", "🧾 Summary"])

with tab1:
    st.subheader("Price Trend (Close)")
    if {"trade_date_fmt", "close"}.issubset(filtered_df.columns):
        close_ts = (
            filtered_df.dropna(subset=["trade_date_fmt", "close"])
            .groupby("trade_date_fmt", as_index=True)["close"]
            .mean()
            .sort_index()
        )
        st.line_chart(close_ts, height=320)
    else:
        st.info("Columns required for close trend not available.")

    st.subheader("Volume Trend")
    if {"trade_date_fmt", "volume"}.issubset(filtered_df.columns):
        volume_ts = (
            filtered_df.dropna(subset=["trade_date_fmt", "volume"])
            .groupby("trade_date_fmt", as_index=True)["volume"]
            .sum()
            .sort_index()
        )
        st.area_chart(volume_ts, height=320)
    else:
        st.info("Columns required for volume trend not available.")

with tab2:
    st.subheader("Filtered Dataset")
    show_cols = [c for c in ["symbol", "trade_date_fmt", "open", "high", "low", "close", "volume", "prev_close", "daily_pct_change"] if c in filtered_df.columns]
    st.dataframe(filtered_df[show_cols], use_container_width=True, height=420)

with tab3:
    st.subheader("Descriptive Summary")
    summary = filtered_df.describe(include="all").T
    st.dataframe(summary, use_container_width=True, height=420)

    st.subheader("Top Symbols by Count")
    symbol_counts = filtered_df["symbol"].value_counts().rename_axis("symbol").reset_index(name="count")
    st.dataframe(symbol_counts, use_container_width=True, height=280)