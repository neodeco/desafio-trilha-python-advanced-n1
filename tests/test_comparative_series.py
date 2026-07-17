from pathlib import Path

from scripts.comparative_series import load_csvs


def test_load_csvs_supports_mixed_delimiters_and_normalizes_trade_date(tmp_path: Path) -> None:
    csv_semicolon = tmp_path / "a.csv"
    csv_comma = tmp_path / "b.csv"

    csv_semicolon.write_text(
        "symbol;trade_date;open;high;low;close;volume\n"
        "PETR4T;202507;10;11;9;10.5;1000\n",
        encoding="utf-8",
    )
    csv_comma.write_text(
        "symbol,trade_date,open,high,low,close,volume\n"
        "PETR4T,2025-07-02,10.1,11.2,9.2,10.7,1200\n",
        encoding="utf-8",
    )

    df = load_csvs(tmp_path, sep="auto")

    assert len(df) == 2
    assert df["trade_date"].notna().all()
    assert str(df["trade_date"].dt.date.min()) == "2025-07-01"