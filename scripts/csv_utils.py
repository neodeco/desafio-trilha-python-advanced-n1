"""Shared CSV helpers used across the ingestion, ETL and modeling scripts.

Centralizes logic that was previously duplicated in ``scripts/data_processing.py``,
``scripts/comparative_series.py`` and ``scripts/spark_predictive_model.py``.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path


def detect_csv_separator(sample: str) -> str:
    """Detect whether a CSV sample uses ';' or ',' as the field separator."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except csv.Error:
        return ";" if sample.count(";") >= sample.count(",") else ","


def detect_csv_separator_from_bytes(content: bytes, sample_size: int = 4096) -> str:
    """Detect the CSV separator from raw file bytes (decoded with a lenient charset)."""
    sample = content[:sample_size].decode("utf-8-sig", errors="ignore")
    return detect_csv_separator(sample)


def detect_csv_separator_from_path(path: Path, sample_size: int = 4096) -> str:
    """Detect the CSV separator by reading the first bytes of a file on disk."""
    sample = Path(path).read_text(encoding="utf-8", errors="ignore")[:sample_size]
    return detect_csv_separator(sample)


def slugify(text: str, default: str = "dados") -> str:
    """Convert arbitrary text (tickers, filenames) into a filesystem-safe slug."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return safe or default
