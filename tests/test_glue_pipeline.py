import pytest
from pyspark.sql import SparkSession

from app.glue_pipeline import transform_stock_data


def test_transform_stock_data_creates_daily_pct_change_and_cleaned_dates():
    spark = SparkSession.builder.master("local[1]").appName("test-glue-pipeline").getOrCreate()

    try:
        raw_rows = [
            ("AAPL", "2024-01-01", "100.0", "102.0", "99.0", "101.0", "1000"),
            ("AAPL", "2024-01-02", None, None, None, None, None),
            ("AAPL", "2024-01-03", "102.5", "104.0", "101.75", "103.0", "1200"),
        ]

        raw_df = spark.createDataFrame(
            raw_rows,
            schema="symbol string, trade_date string, open string, high string, low string, close string, volume string",
        )

        transformed_df = transform_stock_data(raw_df)
        rows = transformed_df.orderBy("trade_date").collect()

        assert rows[0]["trade_date"] == "2024-01-01"
        assert rows[1]["open"] == 0.0
        assert rows[1]["close"] == 0.0
        assert rows[1]["volume"] == 0
        assert rows[2]["daily_pct_change"] == pytest.approx(1.9801980198)
        assert rows[2]["trade_date_fmt"] == "2024-01-03"
    finally:
        spark.stop()
