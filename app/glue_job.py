import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config

from app.glue_pipeline import build_spark_session, transform_stock_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Glue-style stock data ETL job")
    parser.add_argument("--input", default=os.getenv("RAW_DATA_PATH", "files/training-set.csv"), help="Path to the raw CSV data")
    parser.add_argument("--output", default=os.getenv("PROCESSED_DATA_PATH", "output/processed_stock_data"), help="Destination for the processed Parquet data")
    parser.add_argument("--bucket", default=os.getenv("S3_BUCKET", "processed-data"), help="Destination S3 bucket for the Parquet file")
    parser.add_argument("--key", default=os.getenv("S3_KEY", "processed/processed_stock_data.parquet"), help="Destination S3 key for the Parquet file")
    parser.add_argument("--endpoint-url", default=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"), help="LocalStack S3 endpoint URL")
    parser.add_argument("--app-name", default="glue-local-job", help="Spark application name")
    args = parser.parse_args()

    spark = build_spark_session(args.app_name)
    try:
        raw_df = spark.read.option("header", True).csv(args.input)
        transformed_df = transform_stock_data(raw_df)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        transformed_df.toPandas().to_parquet(args.output, index=False)

        session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        s3 = session.client("s3", endpoint_url=args.endpoint_url, config=Config(signature_version="s3v4"))

        output_path = Path(args.output)
        if output_path.exists():
            s3.upload_file(str(output_path), args.bucket, args.key)
            print(f"Processed data uploaded to s3://{args.bucket}/{args.key}")
        else:
            raise FileNotFoundError(f"Expected parquet output at {output_path}")

        print(f"Processed data written to {args.output}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
