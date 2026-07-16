from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from io import StringIO
from typing import BinaryIO, TextIO

import pandas as pd


MAX_PERIOD_DAYS = 365
SEVENTH_TICKER_POSITION = 7
ANALYSIS_OUTPUT_DIR = Path("files/analysis")

DATE_COLUMNS = ("Date", "date", "trade_date", "trade_date_fmt", "Data")
CLOSE_COLUMNS = ("Close", "close", "Adj Close", "adj_close", "Fechamento")
SYMBOL_COLUMNS = ("symbol", "Symbol", "ticker", "Ticker", "Ativo", "ativo", "codigo", "Codigo")


class DataProcessingError(RuntimeError):
    """Raised when the input data cannot produce a valid date/close series."""


@dataclass
class ProcessingResult:
    dataframe: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    saved_path: Path | None = None


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


def _pick_column(dataframe: pd.DataFrame, candidates: tuple[str, ...], label: str, required: bool = True) -> str | None:
    exact = [column for column in candidates if column in dataframe.columns]
    if exact:
        return exact[0]

    normalized = {str(column).strip().lower(): column for column in dataframe.columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found is not None:
            return found

    if required:
        raise DataProcessingError(f"Coluna obrigatoria nao encontrada: {label}.")
    return None


def _normalize_dates(series: pd.Series) -> pd.Series:
    """Convert a raw date-like series to datetime, robust to ISO and dd/mm/yyyy formats."""
    as_text = series.astype(str).str.strip()
    compact_month = as_text.str.fullmatch(r"\d{6}", na=False)
    as_text = as_text.where(~compact_month, as_text + "01")

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    iso_mask = as_text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    if iso_mask.any():
        result.loc[iso_mask] = pd.to_datetime(as_text[iso_mask], format="mixed", dayfirst=False, errors="coerce")

    remaining_mask = ~iso_mask
    if remaining_mask.any():
        result.loc[remaining_mask] = pd.to_datetime(as_text[remaining_mask], format="mixed", dayfirst=True, errors="coerce")

    return result


def _normalize_close(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    values = series.astype(str).str.strip()
    has_comma_decimal = values.str.contains(",", regex=False, na=False)
    cleaned = values.where(~has_comma_decimal, values.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(cleaned, errors="coerce")


def _filter_seventh_ticker(dataframe: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """If more than one ticker is present, drop all rows for the 7th distinct ticker found."""
    symbol_column = _pick_column(dataframe, SYMBOL_COLUMNS, "symbol", required=False)
    if symbol_column is None:
        return dataframe

    symbols = dataframe[symbol_column].astype(str).str.strip()
    unique_tickers = list(dict.fromkeys(symbols.tolist()))

    if len(unique_tickers) <= 1:
        return dataframe

    if len(unique_tickers) < SEVENTH_TICKER_POSITION:
        warnings.append(
            f"Foram encontrados {len(unique_tickers)} tickers diferentes no arquivo. "
            "Nenhum ticker foi removido pois nao ha um setimo ticker distinto."
        )
        return dataframe

    seventh_ticker = unique_tickers[SEVENTH_TICKER_POSITION - 1]
    mask = symbols == seventh_ticker
    removed_rows = int(mask.sum())

    warnings.append(
        f"Foram encontrados {len(unique_tickers)} tickers diferentes no arquivo. "
        f"O setimo ticker ('{seventh_ticker}') foi filtrado e {removed_rows} linha(s) removida(s)."
    )
    return dataframe[~mask].reset_index(drop=True)


def _slugify(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return safe or "dados"


def _save_to_analysis_folder(dataframe: pd.DataFrame, source_name: str) -> Path:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{_slugify(source_name)}_tratado_{timestamp}.csv"
    output_path = ANALYSIS_OUTPUT_DIR / filename

    export_df = dataframe.copy()
    export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%d/%m/%Y")
    export_df.to_csv(output_path, index=False, sep=";")
    return output_path


def finalize_price_dataframe(raw_df: pd.DataFrame, warnings: list[str] | None = None) -> ProcessingResult:
    notices = list(warnings or [])
    if raw_df.empty:
        raise DataProcessingError("Fonte de dados vazia.")

    dataframe = raw_df.copy()
    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    dataframe = _filter_seventh_ticker(dataframe, notices)

    date_column = _pick_column(dataframe, DATE_COLUMNS, "Date")
    close_column = _pick_column(dataframe, CLOSE_COLUMNS, "Close")

    final_df = pd.DataFrame(
        {
            "date": _normalize_dates(dataframe[date_column]),
            "close": _normalize_close(dataframe[close_column]),
        }
    )
    final_df = final_df.dropna(subset=["date", "close"]).sort_values("date")
    final_df = final_df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if final_df.empty:
        raise DataProcessingError("Nenhuma linha valida encontrada apos converter date e close.")

    min_rows = 30
    if len(final_df) < min_rows:
        notices.append(f"A serie possui apenas {len(final_df)} linhas validas; as metricas podem ficar instaveis.")

    latest_date = final_df["date"].max()
    cutoff = latest_date - pd.Timedelta(days=MAX_PERIOD_DAYS)
    if final_df["date"].min() < cutoff:
        final_df = final_df[final_df["date"] >= cutoff].reset_index(drop=True)
        notices.append("Periodo maior que 1 ano (365 dias). A serie foi limitada aos ultimos 365 dias disponiveis.")

    return ProcessingResult(dataframe=final_df, warnings=notices)


def process_csv_input(source: str | Path | bytes | BinaryIO | TextIO, source_name: str = "csv_enviado") -> ProcessingResult:
    content = _read_bytes(source)
    dataframe, warnings = _read_csv_auto(content)

    result = finalize_price_dataframe(dataframe, warnings)
    result.saved_path = _save_to_analysis_folder(result.dataframe, source_name)
    return result


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
        raise DataProcessingError(
            f"Nao foi possivel buscar o ticker {ticker}. Verifique se o simbolo e valido: {message}"
        ) from exc

    if raw_df is None or raw_df.empty:
        raise DataProcessingError(f"Nenhum dado retornado para o ticker {ticker} no periodo informado.")

    raw_df = raw_df.copy()
    raw_df.index.name = raw_df.index.name or "Date"
    raw_df = raw_df.reset_index()
    return finalize_price_dataframe(raw_df)


def format_dates_for_display(dataframe: pd.DataFrame) -> pd.DataFrame:
    display_df = dataframe.copy()
    display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%d/%m/%Y")
    return display_df
