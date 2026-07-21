from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


def _slice(line: str, start: int, end: int) -> str:
    """Return 1-based inclusive slice from a COTAHIST fixed-width line."""
    return line[start - 1 : end]


def _parse_price(raw: str) -> str:
    digits = "".join(ch for ch in raw.strip() if ch.isdigit())
    if not digits:
        return "0.00"
    return f"{int(digits) / 100:.2f}"


def _parse_int(raw: str) -> str:
    digits = "".join(ch for ch in raw.strip() if ch.isdigit())
    return str(int(digits)) if digits else "0"


def _parse_trade_date(raw: str) -> str:
    return datetime.strptime(raw.strip(), "%Y%m%d").date().isoformat()


def convert_to_csv(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="latin-1", errors="ignore") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=["symbol", "trade_date", "open", "high", "low", "close", "volume"])
        writer.writeheader()

        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if len(line) < 147 or _slice(line, 1, 2) != "01":
                continue

            row = {
                "symbol": f"{_slice(line, 11, 12).strip()}{_slice(line, 13, 24).strip()}",
                "trade_date": _parse_trade_date(_slice(line, 3, 10)),
                "open": _parse_price(_slice(line, 57, 69)),
                "high": _parse_price(_slice(line, 70, 82)),
                "low": _parse_price(_slice(line, 83, 95)),
                "close": _parse_price(_slice(line, 122, 134)),
                "volume": _parse_int(_slice(line, 135, 147)),
            }
            writer.writerow(row)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COTAHIST fixed-width TXT to CSV")
    parser.add_argument("input", type=Path, help="Input COTAHIST .TXT file")
    parser.add_argument("output", type=Path, nargs="?", default=None, help="Output .CSV path (optional)")
    args = parser.parse_args()

    output_path = args.output or args.input.with_suffix(".csv")
    converted = convert_to_csv(args.input, output_path)
    print(f"CSV gerado em: {converted}")


if __name__ == "__main__":
    main()
