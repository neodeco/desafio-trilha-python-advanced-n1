from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd


MAX_PERIOD_YEARS = 5
DATE_COLUMNS = ("Date", "date", "trade_date", "trade_date_fmt", "Data")
CLOSE_COLUMNS = ("Close", "close", "Adj Close", "adj_close", "Fechamento")


class DataProcessingError(RuntimeError):
    """Raised when the input data cannot produce a valid Date/Close series."""


@dataclass
class ProcessingResult:
    dataframe: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _read_bytes(source: str | Path | bytes | BinaryIO | TextIO) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()

    content = source.read()
    if isinstance(content, str):
        return content.encode("utf-8")
    return content


def detect_csv_separator(content: bytes) -> str:
    sample = content[:4096].decode("utf-8-sig", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except csv.Error:
        return ";" if sample.count(";") >= sample.count(",") else ","


def _read_csv_auto(content: bytes) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    separator = detect_csv_separator(content)
    if separator != ";":
        warnings.append(
            f"CSV enviado com separador '{separator}'. O parsing foi corrigido automaticamente para carregar os dados."
        )

    text = content.decode("utf-8-sig", errors="replace")
    try:
        dataframe = pd.read_csv(StringIO(text), sep=separator)
    except Exception as exc:
        raise DataProcessingError(f"Nao foi possivel ler o CSV: {exc}") from exc

    return dataframe, warnings


def _pick_column(dataframe: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    exact = [column for column in candidates if column in dataframe.columns]
    if exact:
        return exact[0]

    normalized = {str(column).strip().lower(): column for column in dataframe.columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found is not None:
            return found

    raise DataProcessingError(f"Coluna obrigatoria nao encontrada: {label}.")


def _normalize_dates(series: pd.Series) -> pd.Series:
    as_text = series.astype(str).str.strip()
    compact_month = as_text.str.fullmatch(r"\d{6}", na=False)

    normalized = as_text.copy()
    normalized = normalized.where(~compact_month, normalized + "01")
    return pd.to_datetime(normalized, format="mixed", dayfirst=True, errors="coerce")


def _normalize_close(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    values = series.astype(str).str.strip()
    has_comma_decimal = values.str.contains(",", regex=False, na=False)
    cleaned = values.where(~has_comma_decimal, values.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def finalize_price_dataframe(raw_df: pd.DataFrame, warnings: list[str] | None = None) -> ProcessingResult:
    notices = list(warnings or [])
    if raw_df.empty:
        raise DataProcessingError("Fonte de dados vazia.")

    dataframe = raw_df.copy()
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    date_column = _pick_column(dataframe, DATE_COLUMNS, "Date")
    close_column = _pick_column(dataframe, CLOSE_COLUMNS, "Close")

    final_df = pd.DataFrame(
        {
            "Date": _normalize_dates(dataframe[date_column]),
            "Close": _normalize_close(dataframe[close_column]),
        }
    )
    final_df = final_df.dropna(subset=["Date", "Close"]).sort_values("Date")
    final_df = final_df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    if final_df.empty:
        raise DataProcessingError("Nenhuma linha valida encontrada apos converter Date e Close.")

    min_rows = 30
    if len(final_df) < min_rows:
        notices.append(f"A serie possui apenas {len(final_df)} linhas validas; as metricas podem ficar instaveis.")

    latest_date = final_df["Date"].max()
    cutoff = latest_date - pd.DateOffset(years=MAX_PERIOD_YEARS)
    if final_df["Date"].min() < cutoff:
        final_df = final_df[final_df["Date"] >= cutoff].reset_index(drop=True)
        notices.append("Periodo maior que 5 anos. A serie foi limitada aos ultimos 5 anos disponiveis.")

    return ProcessingResult(dataframe=final_df, warnings=notices)


def process_csv_input(source: str | Path | bytes | BinaryIO | TextIO) -> ProcessingResult:
    content = _read_bytes(source)
    dataframe, warnings = _read_csv_auto(content)
    return finalize_price_dataframe(dataframe, warnings)


def fetch_history_by_ticker(ticker: str, start_date: date | datetime, end_date: date | datetime) -> ProcessingResult:
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataProcessingError("Informe um ticker valido.")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if pd.isna(start) or pd.isna(end):
        raise DataProcessingError("Datas de inicio e fim devem ser validas.")
    if start > end:
        raise DataProcessingError("A data de inicio deve ser anterior ou igual a data de fim.")

    try:
        from pandas_datareader import data as web
    except ImportError as exc:
        raise DataProcessingError(
            "pandas_datareader nao esta instalado. Instale a dependencia para buscar historico por ticker."
        ) from exc

    try:
        raw_df = web.DataReader(ticker, "stooq", start, end)
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise DataProcessingError(f"Erro de rede ao buscar historico para {ticker}: {exc}") from exc
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise DataProcessingError(f"Nao foi possivel buscar o ticker {ticker}: {message}") from exc

    if raw_df is None or raw_df.empty:
        raise DataProcessingError(f"Nenhum dado retornado para o ticker {ticker} no periodo informado.")

    raw_df = raw_df.copy()
    raw_df.index.name = raw_df.index.name or "Date"
    raw_df = raw_df.reset_index()
    return finalize_price_dataframe(raw_df)


def format_dates_for_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    display_df = dataframe.copy()
    display_df["Date"] = pd.to_datetime(display_df["Date"]).dt.strftime("%d/%m/%Y")
    return display_df
