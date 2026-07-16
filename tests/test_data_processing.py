from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from scripts.data_processing import DataProcessingError, fetch_history_by_ticker, format_dates_for_display, process_csv_input


def test_process_csv_input_corrects_non_semicolon_separator_and_formats_dates() -> None:
    content = (
        "Date,Close\n"
        "2024-01-01,10.5\n"
        "02/01/2024,11.25\n"
    ).encode("utf-8")

    result = process_csv_input(content)
    display_df = format_dates_for_display(result.dataframe)

    assert list(result.dataframe.columns) == ["Date", "Close"]
    assert display_df["Date"].tolist() == ["01/01/2024", "02/01/2024"]
    assert result.dataframe["Close"].tolist() == [10.5, 11.25]
    assert any("parsing foi corrigido" in warning for warning in result.warnings)


def test_process_csv_input_limits_period_to_last_five_years() -> None:
    rows = ["Date;Close"]
    dates = pd.date_range("2018-01-01", "2025-01-01", freq="YS")
    rows.extend(f"{item.date()};{100 + index}" for index, item in enumerate(dates))

    result = process_csv_input("\n".join(rows).encode("utf-8"))

    assert result.dataframe["Date"].min() >= pd.Timestamp("2020-01-01")
    assert any("ultimos 5 anos" in warning for warning in result.warnings)


def test_fetch_history_by_ticker_uses_pandas_datareader_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("pandas_datareader")
    data_module = types.ModuleType("pandas_datareader.data")

    def fake_data_reader(ticker: str, source: str, start, end):
        assert ticker == "AAPL"
        assert source == "stooq"
        return pd.DataFrame(
            {"Close": [12.0, 11.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
        )

    data_module.DataReader = fake_data_reader
    module.data = data_module
    monkeypatch.setitem(sys.modules, "pandas_datareader", module)
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", data_module)

    result = fetch_history_by_ticker("aapl", "2024-01-01", "2024-01-02")

    assert result.dataframe["Date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert result.dataframe["Close"].tolist() == [11.0, 12.0]


def test_fetch_history_by_ticker_rejects_invalid_date_range(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DataProcessingError, match="data de inicio"):
        fetch_history_by_ticker("AAPL", "2024-02-01", "2024-01-01")
