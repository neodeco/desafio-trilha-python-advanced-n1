from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def _parse_price(field_text: str) -> str:
    value = int(field_text.strip() or "0")
    return f"{value / 100:.2f}"


def _parse_trade_date(field_text: str) -> str:
    return datetime.strptime(field_text, "%Y%m%d").strftime("%Y-%m-%d")


def _parse_volume(line: str) -> str:
    # The project expects the negotiated quantity column for "volume".
    return str(int((line[134:147] or "0").strip() or "0"))


def _parse_record(line: str) -> dict[str, str] | None:
    if not line.startswith("01") or len(line) < 147:
        return None

    return {
        "symbol": line[10:24].strip(),
        "trade_date": _parse_trade_date(line[2:10]),
        "open": _parse_price(line[56:69]),
        "high": _parse_price(line[69:82]),
        "low": _parse_price(line[82:95]),
        "close": _parse_price(line[121:134]),
        "volume": _parse_volume(line),
    }


def convert_to_csv(input_path: str | Path, output_path: str | Path) -> Path:
    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="latin-1") as input_file, destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["symbol", "trade_date", "open", "high", "low", "close", "volume"])
        writer.writeheader()

        for raw_line in input_file:
            line = raw_line.rstrip("\n")
            record = _parse_record(line)
            if record is not None:
                writer.writerow(record)

    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert COTAHIST fixed-width TXT to pipeline-ready CSV")
    parser.add_argument("input", help="Input COTAHIST TXT path")
    parser.add_argument("output", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    converted_path = convert_to_csv(args.input, args.output)
    print(f"Converted file written to {converted_path}")


if __name__ == "__main__":
    main()