"""Glue-style ETL entrypoint.

Supports two modes:
  - ``stock`` (default): legacy OHLCV transformation for COTAHIST-style
    multi-symbol training data (`transform_stock_data`).
  - ``price-series``: normalizes any raw price CSV (ticker download from
    `yfinance.download` or a user-uploaded CSV) into a
    clean ``date``/``close`` series using PySpark SQL, writes the treated CSV
    to `files/from-file/` (consumed by `scripts/localstack_pipeline_test.py`
    and `scripts/spark_predictive_model.py`), writes an optimized Parquet
    file and uploads it to a LocalStack S3 bucket.

Both modes are meant to run as standalone subprocesses so that the PySpark
JVM never runs inside the Streamlit UI process (see `app/app.py`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import boto3
from botocore.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.glue_pipeline import build_spark_session, transform_price_series, transform_stock_data  # noqa: E402
from scripts.csv_utils import detect_csv_separator_from_path, slugify  # noqa: E402

DEFAULT_ENDPOINT_URL = "http://localhost:4566"
DEFAULT_BUCKET = "processed-data"
TREATED_CSV_DIR = Path("files/from-file")
PROCESSED_DATA_DIR = Path("output/processed_stock_data")


@dataclass
class PriceSeriesJobResult:
    dataframe: "object"
    csv_path: Path
    parquet_path: Path
    bucket: str
    key: str
    warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict:
        return {
            "csv_path": str(self.csv_path),
            "parquet_path": str(self.parquet_path),
            "bucket": self.bucket,
            "key": self.key,
            "warnings": self.warnings,
            "rows": int(len(self.dataframe)),
        }


def upload_to_s3(local_path: Path, bucket: str, key: str, endpoint_url: str) -> None:
    """Upload a local file to a LocalStack/AWS S3 bucket, creating the bucket if needed."""
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    s3 = session.client("s3", endpoint_url=endpoint_url, config=Config(signature_version="s3v4"))

    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

    s3.upload_file(str(local_path), bucket, key)


def run_price_series_job(
    input_path: str | Path,
    source_name: str,
    sep: str = "auto",
    treated_dir: str | Path = TREATED_CSV_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    bucket: str = DEFAULT_BUCKET,
    key_prefix: str = "processed",
    endpoint_url: str = DEFAULT_ENDPOINT_URL,
    app_name: str = "glue-price-series-job",
) -> PriceSeriesJobResult:
    """Run the price-series ETL: raw CSV -> Spark treatment -> treated CSV +
    Parquet uploaded to S3 (LocalStack)."""
    input_path = Path(input_path)
    slug = slugify(source_name)
    csv_sep = detect_csv_separator_from_path(input_path) if sep == "auto" else sep

    treated_dir = Path(treated_dir)
    output_dir = Path(output_dir)
    treated_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    spark = build_spark_session(app_name)
    try:
        raw_df = spark.read.option("header", True).option("sep", csv_sep).csv(str(input_path))
        transformed_df, warnings = transform_price_series(raw_df)

        pandas_df = transformed_df.toPandas()

        csv_path = treated_dir / f"{slug}.csv"
        pandas_df.to_csv(csv_path, index=False, sep=";")

        parquet_path = output_dir / f"{slug}.parquet"
        pandas_df.to_parquet(parquet_path, index=False)

        key = f"{key_prefix}/{slug}.parquet"
        upload_to_s3(parquet_path, bucket, key, endpoint_url)

        return PriceSeriesJobResult(
            dataframe=pandas_df,
            csv_path=csv_path,
            parquet_path=parquet_path,
            bucket=bucket,
            key=key,
            warnings=warnings,
        )
    finally:
        spark.stop()


def run_stock_job(
    input_path: str,
    sep: str,
    output_path: str,
    bucket: str,
    key: str,
    endpoint_url: str,
    app_name: str = "glue-local-job",
) -> None:
    """Legacy OHLCV ETL job used for the COTAHIST-style multi-symbol training-set flow."""
    csv_sep = detect_csv_separator_from_path(Path(input_path)) if sep == "auto" else sep
    spark = build_spark_session(app_name)
    try:
        raw_df = spark.read.option("header", True).option("sep", csv_sep).csv(input_path)
        transformed_df = transform_stock_data(raw_df)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        transformed_df.toPandas().to_parquet(output_path, index=False)

        output_file = Path(output_path)
        if not output_file.exists():
            raise FileNotFoundError(f"Expected parquet output at {output_file}")

        upload_to_s3(output_file, bucket, key, endpoint_url)
        print(f"Processed data uploaded to s3://{bucket}/{key}")
        print(f"Processed data written to {output_path}")
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Glue-style stock data ETL job")
    parser.add_argument(
        "--mode",
        choices=["stock", "price-series"],
        default=os.getenv("GLUE_JOB_MODE", "stock"),
        help="'stock' runs the legacy OHLCV transformation; 'price-series' normalizes a "
        "ticker/CSV price series down to date/close, writes it to files/from-file and "
        "uploads the Parquet output to S3 via LocalStack.",
    )
    parser.add_argument(
        "--input",
        default=os.getenv("RAW_DATA_PATH", "files/training-set/traning-set.csv"),
        help="Path to the raw CSV data",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Logical name (e.g. ticker) used to derive output filenames in price-series mode",
    )
    parser.add_argument(
        "--sep",
        default=os.getenv("CSV_SEPARATOR", "auto"),
        help="CSV separator for input data ('auto' to detect automatically, default)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("PROCESSED_DATA_PATH", "output/processed_stock_data"),
        help="[stock mode] Destination path for the processed Parquet file",
    )
    parser.add_argument(
        "--treated-dir",
        default=os.getenv("TREATED_CSV_DIR", str(TREATED_CSV_DIR)),
        help="[price-series mode] Destination folder for the treated date/close CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("PROCESSED_DATA_DIR", str(PROCESSED_DATA_DIR)),
        help="[price-series mode] Destination folder for the Parquet file",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET", DEFAULT_BUCKET),
        help="Destination S3 bucket for the Parquet file",
    )
    parser.add_argument(
        "--key",
        default=os.getenv("S3_KEY", "processed/processed_stock_data.parquet"),
        help="[stock mode] Destination S3 key for the Parquet file",
    )
    parser.add_argument(
        "--key-prefix",
        default=os.getenv("S3_KEY_PREFIX", "processed"),
        help="[price-series mode] Destination S3 key prefix for the Parquet file",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.getenv("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT_URL),
        help="LocalStack S3 endpoint URL",
    )
    parser.add_argument("--app-name", default="glue-local-job", help="Spark application name")
    args = parser.parse_args()

    if args.mode == "price-series":
        source_name = args.source_name or Path(args.input).stem
        result = run_price_series_job(
            input_path=args.input,
            source_name=source_name,
            sep=args.sep,
            treated_dir=args.treated_dir,
            output_dir=args.output_dir,
            bucket=args.bucket,
            key_prefix=args.key_prefix,
            endpoint_url=args.endpoint_url,
            app_name=args.app_name,
        )
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        print(f"Treated CSV written to {result.csv_path}")
        print(f"Processed Parquet written to {result.parquet_path}")
        print(f"Processed data uploaded to s3://{result.bucket}/{result.key}")
        # Last line is a machine-readable JSON summary for callers (e.g. app.py) that
        # invoke this script as a subprocess and need the output paths back.
        print(json.dumps(result.to_summary()))
        return

    run_stock_job(
        input_path=args.input,
        sep=args.sep,
        output_path=args.output,
        bucket=args.bucket,
        key=args.key,
        endpoint_url=args.endpoint_url,
        app_name=args.app_name,
    )


if __name__ == "__main__":
    main()