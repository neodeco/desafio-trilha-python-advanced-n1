import pytest
from pyspark.sql import SparkSession

from app.glue_pipeline import transform_price_series, transform_stock_data


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[1]").appName("test-glue-pipeline").getOrCreate()
    yield session
    session.stop()


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


def test_transform_price_series_normalizes_mixed_dates_decimals_and_dedup(spark):
    raw_df = spark.createDataFrame(
        [
            ("2024-01-01", "10,5"),
            ("01/02/2024", "11,25"),
            ("2024-01-01", "10,9"),  # duplicate date; keep-last semantics
        ],
        schema="Date string, Close string",
    )

    transformed_df, warnings = transform_price_series(raw_df)
    rows = transformed_df.orderBy("date").collect()

    assert [row["date"] for row in rows] == ["2024-01-01", "2024-02-01"]
    assert rows[0]["close"] == pytest.approx(10.9)
    assert rows[1]["close"] == pytest.approx(11.25)
    assert any("apenas" in warning for warning in warnings)


def test_transform_price_series_requires_single_ticker(spark):
    rows = [("AAA", "2024-01-01", "100.0"), ("BBB", "2024-01-02", "101.0")]
    raw_df = spark.createDataFrame(rows, schema="symbol string, Date string, Close string")

    with pytest.raises(ValueError, match="apenas um ticker"):
        transform_price_series(raw_df)


def test_transform_price_series_trims_to_last_365_days(spark):
    raw_df = spark.createDataFrame(
        [("2018-01-01", "100.0"), ("2024-01-01", "200.0")],
        schema="Date string, Close string",
    )

    transformed_df, warnings = transform_price_series(raw_df)
    rows = transformed_df.orderBy("date").collect()

    assert len(rows) == 1
    assert rows[0]["date"] == "2024-01-01"
    assert any("365 dias" in warning for warning in warnings)


def test_transform_price_series_raises_when_no_valid_rows(spark):
    raw_df = spark.createDataFrame([("not-a-date", "not-a-number")], schema="Date string, Close string")

    with pytest.raises(ValueError, match="Nenhuma linha valida"):
        transform_price_series(raw_df)


def test_transform_price_series_rejects_dates_without_day_component(spark):
    raw_df = spark.createDataFrame([("202401", "10.0"), ("202402", "11.0")], schema="Date string, Close string")

    with pytest.raises(ValueError, match="Nenhuma linha valida"):
        transform_price_series(raw_df)
