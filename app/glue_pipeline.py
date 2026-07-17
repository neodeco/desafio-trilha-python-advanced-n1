import os
from datetime import datetime, timedelta
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

DEFAULT_DATE_COLUMNS = ("Date", "date", "trade_date", "trade_date_fmt", "Data")
DEFAULT_CLOSE_COLUMNS = ("Close", "close", "Adj Close", "adj_close", "Fechamento")
DEFAULT_SYMBOL_COLUMNS = ("symbol", "Symbol", "ticker", "Ticker", "Ativo", "ativo", "codigo", "Codigo")


def build_spark_session(app_name: str = "local-glue-pipeline") -> SparkSession:
    os.environ.setdefault("HADOOP_HOME", os.path.abspath("."))
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    return SparkSession.builder.master("local[1]").appName(app_name).getOrCreate()


def transform_stock_data(raw_df: DataFrame) -> DataFrame:
    schema = StructType(
        [
            StructField("symbol", StringType(), True),
            StructField("trade_date", StringType(), True),
            StructField("open", StringType(), True),
            StructField("high", StringType(), True),
            StructField("low", StringType(), True),
            StructField("close", StringType(), True),
            StructField("volume", StringType(), True),
        ]
    )
    if raw_df.schema != schema:
        raw_df = raw_df.select(*[F.col(f.name).cast(f.dataType) for f in schema.fields])

    window = Window.partitionBy("symbol").orderBy("trade_date_fmt")

    cleaned = (
        raw_df
        .withColumn("symbol", F.when(F.col("symbol").isNull(), "UNKNOWN").otherwise(F.col("symbol")))
        .withColumn("trade_date", F.when(F.col("trade_date").isNull(), F.lit("1970-01-01")).otherwise(F.col("trade_date")))
        .withColumn("open", F.col("open").cast(DoubleType()))
        .withColumn("high", F.col("high").cast(DoubleType()))
        .withColumn("low", F.col("low").cast(DoubleType()))
        .withColumn("close", F.col("close").cast(DoubleType()))
        .withColumn("volume", F.col("volume").cast(LongType()))
        .withColumn(
            "trade_date_fmt",
            F.when(F.col("trade_date").rlike(r"^\d{4}-\d{2}-\d{2}$"), F.to_date(F.col("trade_date"), "yyyy-MM-dd"))
            .when(F.col("trade_date").rlike(r"^\d{4}\d{2}$"), F.to_date(F.concat(F.col("trade_date"), F.lit("01")), "yyyyMMdd"))
            .otherwise(F.to_date(F.lit("1970-01-01"), "yyyy-MM-dd")),
        )
        .withColumn("trade_date_fmt", F.date_format(F.col("trade_date_fmt"), "yyyy-MM-dd"))
        .withColumn("prev_close", F.last(F.col("close"), ignorenulls=True).over(window.rowsBetween(Window.unboundedPreceding, Window.currentRow - 1)))
        .withColumn(
            "daily_pct_change",
            F.when(F.col("prev_close").isNull() | (F.col("prev_close") == 0), F.lit(None))
            .otherwise(((F.col("close") - F.col("prev_close")) / F.col("prev_close")) * 100),
        )
        .fillna({
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "volume": 0,
            "daily_pct_change": 0.0,
        })
    )

    return cleaned


def transform_price_series(
    raw_df: DataFrame,
    date_columns: tuple[str, ...] = DEFAULT_DATE_COLUMNS,
    close_columns: tuple[str, ...] = DEFAULT_CLOSE_COLUMNS,
    symbol_columns: tuple[str, ...] = DEFAULT_SYMBOL_COLUMNS,
    max_period_days: int = 365,
    seventh_ticker_position: int = 7,
    min_recommended_rows: int = 30,
) -> tuple[DataFrame, list[str]]:
    """Normalize an arbitrary price CSV (ticker download or user upload) into a
    clean ``date``/``close`` Spark DataFrame, entirely with ``pyspark.sql``.

    This mirrors the business rules that used to live in
    ``scripts/data_processing.py`` (pandas-based): auto-detects the date/close
    columns regardless of casing/naming, drops all rows belonging to the 7th
    distinct ticker when multiple symbols are present, normalizes ISO and
    dd/MM/yyyy dates plus comma-decimal prices, removes null/duplicate rows and
    limits the series to the last ``max_period_days`` days.

    Returns the cleaned DataFrame (columns ``date`` as ``yyyy-MM-dd`` string and
    ``close`` as double) together with a list of human-readable warnings.
    """
    warnings: list[str] = []
    columns_lookup = {column.strip().lower(): column for column in raw_df.columns}

    def _find_column(candidates: tuple[str, ...], label: str, required: bool) -> Optional[str]:
        for candidate in candidates:
            found = columns_lookup.get(candidate.strip().lower())
            if found is not None:
                return found
        if required:
            raise ValueError(f"Coluna obrigatoria nao encontrada: {label}.")
        return None

    date_column = _find_column(date_columns, "Date", required=True)
    close_column = _find_column(close_columns, "Close", required=True)
    symbol_column = _find_column(symbol_columns, "symbol", required=False)

    df = raw_df

    if symbol_column is not None:
        ordered_symbols = [
            row[symbol_column]
            for row in (
                df.withColumn("_row_id", F.monotonically_increasing_id())
                .groupBy(symbol_column)
                .agg(F.min("_row_id").alias("_first_seen"))
                .orderBy("_first_seen")
                .select(symbol_column)
                .collect()
            )
        ]

        if len(ordered_symbols) > 1:
            if len(ordered_symbols) < seventh_ticker_position:
                warnings.append(
                    f"Foram encontrados {len(ordered_symbols)} tickers diferentes no arquivo. "
                    "Nenhum ticker foi removido pois nao ha um setimo ticker distinto."
                )
            else:
                seventh_ticker = ordered_symbols[seventh_ticker_position - 1]
                removed_rows = df.filter(F.col(symbol_column) == F.lit(seventh_ticker)).count()
                df = df.filter(
                    F.col(symbol_column).isNull() | (F.col(symbol_column) != F.lit(seventh_ticker))
                )
                warnings.append(
                    f"Foram encontrados {len(ordered_symbols)} tickers diferentes no arquivo. "
                    f"O setimo ticker ('{seventh_ticker}') foi filtrado e {removed_rows} linha(s) removida(s)."
                )

    raw_date_text = F.trim(F.col(date_column).cast(StringType()))
    compact_month = raw_date_text.rlike(r"^\d{6}$")
    normalized_date_text = F.when(compact_month, F.concat(raw_date_text, F.lit("01"))).otherwise(raw_date_text)

    # `try_to_date` (instead of `to_date`) returns NULL on unparsable input
    # rather than raising, which is required for the coalesce-based fallback
    # to work under Spark's ANSI mode. Try ISO first (unambiguous), then fall
    # back to day-first formats, matching the previous pandas
    # `dayfirst=True` convention for non-ISO input.
    parsed_date = F.coalesce(
        F.try_to_date(normalized_date_text, "yyyy-MM-dd"),
        F.try_to_date(normalized_date_text, "yyyyMMdd"),
        F.try_to_date(normalized_date_text, "dd/MM/yyyy"),
        F.try_to_date(normalized_date_text, "MM/dd/yyyy"),
        F.try_to_date(normalized_date_text, "yyyy/MM/dd"),
    )

    raw_close_text = F.trim(F.col(close_column).cast(StringType()))
    has_comma_decimal = raw_close_text.contains(",")
    cleaned_close_text = F.when(
        has_comma_decimal,
        F.regexp_replace(F.regexp_replace(raw_close_text, r"\.", ""), ",", "."),
    ).otherwise(raw_close_text)
    parsed_close = cleaned_close_text.try_cast(DoubleType())

    normalized = (
        df.withColumn("date", parsed_date)
        .withColumn("close", parsed_close)
        .select("date", "close")
        .dropna(subset=["date", "close"])
    )

    # Deduplicate by date, keeping the row that appeared last in the original
    # input order (matches pandas' `drop_duplicates(keep="last")` semantics).
    deduped = (
        normalized.withColumn("_row_id", F.monotonically_increasing_id())
        .groupBy("date")
        .agg(F.max(F.struct("_row_id", "close")).alias("_last"))
        .select("date", F.col("_last.close").alias("close"))
    )

    total_rows = deduped.count()
    if total_rows == 0:
        raise ValueError("Nenhuma linha valida encontrada apos converter date e close.")

    if total_rows < min_recommended_rows:
        warnings.append(f"A serie possui apenas {total_rows} linhas validas; as metricas podem ficar instaveis.")

    bounds = deduped.agg(F.max("date").alias("max_date"), F.min("date").alias("min_date")).first()
    latest_date, earliest_date = bounds["max_date"], bounds["min_date"]

    if earliest_date is not None and latest_date is not None:
        cutoff = latest_date - timedelta(days=max_period_days)
        if earliest_date < cutoff:
            deduped = deduped.filter(F.col("date") >= F.lit(cutoff))
            warnings.append(
                f"Periodo maior que 1 ano ({max_period_days} dias). "
                f"A serie foi limitada aos ultimos {max_period_days} dias disponiveis."
            )

    result = deduped.orderBy("date").withColumn("date", F.date_format(F.col("date"), "yyyy-MM-dd"))
    return result, warnings


def run_pipeline(input_path: str, output_path: str, app_name: str = "local-glue-pipeline") -> DataFrame:
    spark = build_spark_session(app_name)
    try:
        raw_df = spark.read.option("header", True).csv(input_path)
        transformed_df = transform_stock_data(raw_df)
        transformed_df.write.mode("overwrite").parquet(output_path)
        return transformed_df
    finally:
        spark.stop()
