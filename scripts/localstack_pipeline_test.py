"""Run an automated pipeline test against LocalStack.

Steps performed:
 - ensure LocalStack S3/SQS/DynamoDB resources
 - run the ETL job (`app.glue_job`) which writes Parquet and uploads to S3
 - verify object exists in S3
 - run the predictive model script
 - collect and write a short run report to `output/localstack_test_results.txt`

Usage:
  python scripts/localstack_pipeline_test.py --endpoint-url http://localhost:4566
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import traceback

import boto3
from botocore.config import Config
import os


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    res.check_returncode()
    return res.stdout


def verify_s3_object(endpoint_url: str, bucket: str, key: str) -> bool:
    session = boto3.Session(aws_access_key_id="test", aws_secret_access_key="test", region_name="us-east-1")
    s3 = session.client("s3", endpoint_url=endpoint_url, config=Config(signature_version="s3v4"))
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        print("S3 verification failed:", e)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-url", default="http://localhost:4566")
    p.add_argument("--input", default="files/training-set/cotahist_m072025_training-set.csv")
    p.add_argument("--output", default="output/processed_stock_data.parquet")
    p.add_argument("--bucket", default="processed-data")
    p.add_argument("--key", default="processed/processed_stock_data.parquet")
    p.add_argument("--model-output", default="output/model-test")
    args = p.parse_args()

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "localstack_test_results.txt"

    report_lines = []
    try:
        report_lines.append("Ensuring LocalStack resources (running scripts/setup_localstack.py)...")
        env = dict(os.environ)
        env["AWS_ENDPOINT_URL"] = args.endpoint_url
        env["AWS_ACCESS_KEY_ID"] = env.get("AWS_ACCESS_KEY_ID", "test")
        env["AWS_SECRET_ACCESS_KEY"] = env.get("AWS_SECRET_ACCESS_KEY", "test")
        run_cmd([sys.executable, "scripts/setup_localstack.py"])
        report_lines.append("LocalStack resources ensured.")

        # Determine input CSV (allow auto-discovery if default path missing)
        input_path = Path(args.input)
        if not input_path.exists():
            candidates = sorted(Path('files/training-set').glob('*.csv'))
            if not candidates:
                raise FileNotFoundError(f"No CSV files found in files/training-set and input {args.input} not found")
            input_path = candidates[0]
            report_lines.append(f"Auto-selected input file: {input_path}")

        # Run ETL job
        report_lines.append("Running ETL job (app.glue_job)...")
        cmd = [sys.executable, "-m", "app.glue_job", "--input", str(input_path), "--output", args.output, "--bucket", args.bucket, "--key", args.key, "--endpoint-url", args.endpoint_url]
        etl_out = run_cmd(cmd)
        report_lines.append("ETL output:\n" + etl_out)

        # Verify S3 upload
        report_lines.append("Verifying S3 upload...")
        ok = verify_s3_object(args.endpoint_url, args.bucket, args.key)
        report_lines.append(f"S3 object present: {ok}")

        # Run predictive model
        report_lines.append("Running predictive model script...")
        cmd2 = [sys.executable, "scripts/spark_predictive_model.py", "--training-dir", "files/training-set", "--test-file", "files/test-set/COTAHIST_A2020.TXT", "--output-dir", args.model_output]
        model_out = run_cmd(cmd2)
        report_lines.append("Model output:\n" + model_out)

        report_lines.append("Test run completed successfully")
    except Exception:
        tb = traceback.format_exc()
        report_lines.append("Test run failed:\n" + tb)
    finally:
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Wrote test report to {report_path}")


if __name__ == "__main__":
    main()
