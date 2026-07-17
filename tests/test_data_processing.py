from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from scripts import data_processing
from scripts.data_processing import (
    DataProcessingError,
    fetch_history_by_ticker,
    format_dates_for_display,
    process_csv_input,
)


def test_process_csv_input_corrects_non_semicolon_separator_and_formats_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_processing, "ANALYSIS_OUTPUT_DIR", tmp_path / "analysis")
    monkeypatch.setattr(data_processing, "FROM_INPUT_DIR", tmp_path / "from-input")

    content = (
        "Date,Close\n"
        "2024-01-01,10.5\n"
        "02/01/2024,11.25\n"
    ).encode("utf-8")

    result = process_csv_input(content, source_name="minha_planilha.csv")
    display_df = format_dates_for_display(result.dataframe)

    assert list(result.dataframe.columns) == ["date", "close"]
    assert display_df["date"].tolist() == ["01/01/2024", "02/01/2024"]
    assert result.dataframe["close"].tolist() == [10.5, 11.25]
    assert any("parsing foi corrigido" in warning for warning in result.warnings)
    assert result.saved_path is not None
    assert result.saved_path.exists()
    assert result.saved_path.suffix == ".csv"
    assert result.raw_csv_path is not None
    assert result.raw_csv_path.exists()
    assert result.raw_csv_path.read_bytes() == content


def test_process_csv_input_limits_period_to_last_365_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_processing, "ANALYSIS_OUTPUT_DIR", tmp_path / "analysis")
    monkeypatch.setattr(data_processing, "FROM_INPUT_DIR", tmp_path / "from-input")

    rows = ["Date;Close"]
    dates = pd.date_range("2018-01-01", "2025-01-01", freq="YS")
    rows.extend(f"{item.date()};{100 + index}" for index, item in enumerate(dates))

    result = process_csv_input("\n".join(rows).encode("utf-8"))

    assert (result.dataframe["date"].max() - result.dataframe["date"].min()).days <= 365
    assert any("365 dias" in warning for warning in result.warnings)


def test_process_csv_input_filters_seventh_ticker_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_processing, "ANALYSIS_OUTPUT_DIR", tmp_path / "analysis")
    monkeypatch.setattr(data_processing, "FROM_INPUT_DIR", tmp_path / "from-input")

    tickers = [f"TICK{i}" for i in range(1, 9)]
    rows = ["symbol;Date;Close"]
    base_date = pd.Timestamp("2024-01-01")
    for day_offset, ticker in enumerate(tickers):
        trade_date = (base_date + pd.Timedelta(days=day_offset)).date()
        rows.append(f"{ticker};{trade_date};{100 + day_offset}")

    result = process_csv_input("\n".join(rows).encode("utf-8"))

    seventh_ticker = tickers[6]
    assert any("setimo ticker" in warning.lower() for warning in result.warnings)
    assert any(seventh_ticker in warning for warning in result.warnings)
    # the seventh ticker's close value should no longer be present in the final series
    assert (100 + 6) not in result.dataframe["close"].tolist()


def test_process_csv_input_keeps_data_when_fewer_than_seven_tickers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_processing, "ANALYSIS_OUTPUT_DIR", tmp_path / "analysis")
    monkeypatch.setattr(data_processing, "FROM_INPUT_DIR", tmp_path / "from-input")

    rows = [
        "symbol;Date;Close",
        "AAA;2024-01-01;10",
        "BBB;2024-01-02;11",
        "CCC;2024-01-03;12",
    ]

    result = process_csv_input("\n".join(rows).encode("utf-8"))

    assert len(result.dataframe) == 3
    assert any("nao ha um setimo ticker" in warning.lower() for warning in result.warnings)


def test_fetch_history_by_ticker_uses_get_data_yahoo_and_saves_raw_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_processing, "FROM_INPUT_DIR", tmp_path / "from-input")

    module = types.ModuleType("pandas_datareader")
    data_module = types.ModuleType("pandas_datareader.data")

    def fake_get_data_yahoo(ticker: str, start=None, end=None):
        assert ticker == "AAPL"
        return pd.DataFrame(
            {"Close": [12.0, 11.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-01"]),
        )

    data_module.get_data_yahoo = fake_get_data_yahoo
    module.data = data_module
    monkeypatch.setitem(sys.modules, "pandas_datareader", module)
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", data_module)

    result = fetch_history_by_ticker("aapl", "2024-01-01", "2024-01-02")

    assert result.dataframe["date"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert result.dataframe["close"].tolist() == [11.0, 12.0]
    assert result.raw_csv_path is not None
    assert result.raw_csv_path.exists()
    assert result.raw_csv_path.name == "AAPL.csv"


def test_fetch_history_by_ticker_rejects_invalid_date_range() -> None:
    with pytest.raises(DataProcessingError, match="data de inicio"):
        fetch_history_by_ticker("AAPL", "2024-02-01", "2024-01-01")


def test_fetch_history_by_ticker_raises_on_empty_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_processing, "FROM_INPUT_DIR", tmp_path / "from-input")

    module = types.ModuleType("pandas_datareader")
    data_module = types.ModuleType("pandas_datareader.data")

    def fake_get_data_yahoo(ticker: str, start=None, end=None):
        return pd.DataFrame()

    data_module.get_data_yahoo = fake_get_data_yahoo
    module.data = data_module
    monkeypatch.setitem(sys.modules, "pandas_datareader", module)
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", data_module)

    with pytest.raises(DataProcessingError, match="Nenhum dado retornado"):
        fetch_history_by_ticker("INVALIDX", "2024-01-01", "2024-01-02")


def test_fetch_history_by_ticker_wraps_network_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data_processing, "FROM_INPUT_DIR", tmp_path / "from-input")

    module = types.ModuleType("pandas_datareader")
    data_module = types.ModuleType("pandas_datareader.data")

    def fake_get_data_yahoo(ticker: str, start=None, end=None):
        raise ConnectionError("network down")

    data_module.get_data_yahoo = fake_get_data_yahoo
    module.data = data_module
    monkeypatch.setitem(sys.modules, "pandas_datareader", module)
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", data_module)

    with pytest.raises(DataProcessingError, match="Erro de rede"):
        fetch_history_by_ticker("AAPL", "2024-01-01", "2024-01-02")
