import csv
from pathlib import Path

from scripts.convert_cotahist_to_csv import convert_to_csv


def test_convert_to_csv_creates_pipeline_ready_rows(tmp_path: Path) -> None:
    sample_input = tmp_path / "sample.txt"
    sample_output = tmp_path / "sample.csv"

    sample_input.write_text(
        "00COTAHIST.2025BOVESPA 20250731\n"
        "012025071796EALT3F      020ACO ALTONA  ON           R$  000000000150100000000015370000000001493000000000150000000000014930000000001482000000000149300013000000000000000073000000000000109542000000000000009999123100000010000000000000BREALTACNOR4139\n",
        encoding="latin-1",
    )

    convert_to_csv(sample_input, sample_output)

    with sample_output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["symbol"] == "96EALT3F"
    assert rows[0]["trade_date"] == "20250717"
    assert rows[0]["open"] == "15.01"
    assert rows[0]["high"] == "15.37"
    assert rows[0]["low"] == "14.93"
    assert rows[0]["close"] == "14.82"
    assert rows[0]["volume"] == "1493"
