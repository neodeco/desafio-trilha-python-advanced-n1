import os
from datetime import datetime
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType


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


def run_pipeline(input_path: str, output_path: str, app_name: str = "local-glue-pipeline") -> DataFrame:
    spark = build_spark_session(app_name)
    try:
        raw_df = spark.read.option("header", True).csv(input_path)
        transformed_df = transform_stock_data(raw_df)
        transformed_df.write.mode("overwrite").parquet(output_path)
        return transformed_df
    finally:
        spark.stop()
