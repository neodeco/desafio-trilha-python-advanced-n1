import csv
import re
from pathlib import Path
from typing import List, Tuple


RECORD_LAYOUT = [
    ("date", 2, 8),
    ("symbol", 8, 12),
    ("name", 12, 24),
    ("market", 24, 27),
    ("currency", 27, 30),
    ("open", 30, 42),
    ("high", 42, 54),
    ("low", 54, 66),
    ("close", 66, 78),
    ("volume", 78, 90),
]

# The historical file uses a compact fixed-width layout where the
# numeric values are provided as 12-digit strings that represent
# prices and volume in cents/units. We extract them from the
# record text directly rather than relying on the fixed positions.


def _parse_fixed_width(line: str) -> dict:
    record = {}
    for name, start, end in RECORD_LAYOUT:
        raw_value = line[start:end].strip()
        record[name] = raw_value
    return record


def _normalize(record: dict, line: str) -> dict:
    symbol = line[8:24].strip()
    symbol = re.sub(r"[^A-Za-z0-9]", "", symbol)
    if symbol.startswith("20"):
        symbol = symbol[2:]
    if len(symbol) > 8 and symbol[:4].isdigit():
        symbol = symbol[4:]

    trade_date = record.get("date", "")
    if len(trade_date) == 8:
        trade_date = f"{trade_date[0:4]}-{trade_date[4:6]}-{trade_date[6:8]}"

    def parse_number(raw: str) -> str:
        raw = raw.replace(".", "").replace(",", "")
        if not raw:
            return ""
        try:
            return f"{int(raw) / 100:.2f}".replace("-0.00", "0.00")
        except ValueError:
            return "0.00"

    # The historical record stores the numeric values as fixed-size chunks after the currency marker.
    price_sequence = line[line.find("R$")+2:].strip()
    if len(price_sequence) >= 60:
        open_value = price_sequence[0:12]
        high_value = price_sequence[12:24]
        low_value = price_sequence[24:36]
        close_value = price_sequence[36:48]
        volume_value = price_sequence[48:60]
    else:
        open_value = high_value = low_value = close_value = volume_value = ""

    return {
        "symbol": symbol,
        "trade_date": trade_date,
        "open": parse_number(open_value),
        "high": parse_number(high_value),
        "low": parse_number(low_value),
        "close": parse_number(close_value),
        "volume": str(int(volume_value) if volume_value.isdigit() else 0),
    }


def convert_to_csv(input_path: Path | str, output_path: Path | str) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = input_path.read_text(encoding="latin-1", errors="replace").splitlines()
    rows = []
    for line in lines:
        if not line.startswith("01"):
            continue
        parsed = _normalize(_parse_fixed_width(line), line)
        if parsed["symbol"]:
            rows.append(parsed)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "trade_date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)

    return output_path


if __name__ == "__main__":
    convert_to_csv("files/COTAHIST_M072025.TXT", "files/cotahist_m072025.csv")
