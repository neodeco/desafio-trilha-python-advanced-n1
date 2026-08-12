"""Run an automated end-to-end pipeline test against LocalStack.

Exercises the same flow used by ``app/app.py``:
 - ensure LocalStack S3/SQS/DynamoDB resources (``scripts/setup_localstack.py``)
 - obtain a raw price CSV under ``files/from-input`` (auto-discovered, or a
   synthetic sample is generated so the test never depends on network access)
 - run the PySpark ETL job (``app.glue_job --mode price-series``), which
   normalizes the CSV down to ``date``/``close``, writes it to
   ``files/from-file/`` and uploads an optimized Parquet copy to a LocalStack
   S3 bucket
 - verify the Parquet object exists in S3
 - run the PySpark forecast model (``scripts.spark_predictive_model --mode
   forecast``) against the treated CSV
 - collect and write a short run report to ``output/localstack_test_results.txt``

Usage:
  python scripts/localstack_pipeline_test.py --endpoint-url http://localhost:4566
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.csv_utils import slugify  # noqa: E402

FROM_INPUT_DIR = Path("files/from-input")
FROM_FILE_DIR = Path("files/from-file")
SYNTHETIC_SAMPLE_NAME = "SAMPLE_LOCALSTACK_TEST"


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> str:
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(res.stdout)
    res.check_returncode()
    return res.stdout


def parse_last_json_line(output: str) -> dict:
    for line in reversed([line for line in output.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON summary line found in subprocess output")


def verify_s3_object(endpoint_url: str, bucket: str, key: str) -> bool:
    session = boto3.Session(aws_access_key_id="test", aws_secret_access_key="test", region_name="us-east-1")
    s3 = session.client("s3", endpoint_url=endpoint_url, config=Config(signature_version="s3v4"))
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        print("S3 verification failed:", e)
        return False


def discover_or_create_sample_input() -> Path:
    """Return a raw CSV under ``files/from-input`` to feed the ETL job.

    Reuses the most recently modified CSV already present (e.g. produced by a
    real ticker fetch or CSV upload via ``app.py``). If none exists, generates
    a small deterministic synthetic OHLCV series so this test never depends on
    network access or a prior manual run.
    """
    FROM_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(FROM_INPUT_DIR.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    close = 10 + np.cumsum(rng.normal(0.05, 0.3, size=len(dates)))
    sample_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Adj Close": close,
            "Volume": 1000,
        }
    )
    sample_path = FROM_INPUT_DIR / f"{SYNTHETIC_SAMPLE_NAME}.csv"
    sample_df.to_csv(sample_path, index=False)
    return sample_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-url", default="http://localhost:4566")
    parser.add_argument("--input", default=None, help="Raw CSV path (default: auto-discover/synthesize under files/from-input)")
    parser.add_argument("--bucket", default="processed-data")
    parser.add_argument("--key-prefix", default="processed")
    args = parser.parse_args()

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "localstack_test_results.txt"

    report_lines: list[str] = []
    try:
        report_lines.append("Ensuring LocalStack resources (running scripts/setup_localstack.py)...")
        env = dict(os.environ)
        env["AWS_ENDPOINT_URL"] = args.endpoint_url
        env["AWS_ACCESS_KEY_ID"] = env.get("AWS_ACCESS_KEY_ID", "test")
        env["AWS_SECRET_ACCESS_KEY"] = env.get("AWS_SECRET_ACCESS_KEY", "test")
        run_cmd([sys.executable, "scripts/setup_localstack.py"], env=env)
        report_lines.append("LocalStack resources ensured.")

        input_path = Path(args.input) if args.input else discover_or_create_sample_input()
        report_lines.append(f"Using input CSV: {input_path}")
        source_name = slugify(input_path.stem)

        # Step 1: PySpark ETL (app.glue_job --mode price-series) -----------
        report_lines.append("Running PySpark ETL job (app.glue_job --mode price-series)...")
        etl_cmd = [
            sys.executable,
            "-m",
            "app.glue_job",
            "--mode",
            "price-series",
            "--input",
            str(input_path),
            "--source-name",
            source_name,
            "--bucket",
            args.bucket,
            "--key-prefix",
            args.key_prefix,
            "--endpoint-url",
            args.endpoint_url,
        ]
        etl_out = run_cmd(etl_cmd, env=env)
        report_lines.append("ETL output:\n" + etl_out)
        etl_summary = parse_last_json_line(etl_out)

        treated_csv_path = Path(etl_summary["csv_path"])
        if not treated_csv_path.exists():
            raise FileNotFoundError(
                f"Treated CSV not found at {treated_csv_path} (expected under {FROM_FILE_DIR})"
            )
        report_lines.append(f"Treated date/close CSV confirmed at {treated_csv_path}")

        # Step 2: verify Parquet upload in S3 (LocalStack) ------------------
        report_lines.append("Verifying S3 upload...")
        ok = verify_s3_object(args.endpoint_url, etl_summary["bucket"], etl_summary["key"])
        report_lines.append(f"S3 object present: {ok}")
        if not ok:
            raise RuntimeError(f"Expected S3 object s3://{etl_summary['bucket']}/{etl_summary['key']} not found")

        # Step 3: PySpark forecast model (scripts.spark_predictive_model) --
        report_lines.append("Running PySpark forecast model (scripts.spark_predictive_model --mode forecast)...")
        model_cmd = [
            sys.executable,
            "-m",
            "scripts.spark_predictive_model",
            "--mode",
            "forecast",
            "--forecast-input",
            str(treated_csv_path),
            "--source-name",
            source_name,
        ]
        model_out = run_cmd(model_cmd, env=env)
        report_lines.append("Model output:\n" + model_out)
        model_summary = parse_last_json_line(model_out)
        report_lines.append(f"Forecast metrics: {json.dumps(model_summary['metrics'], indent=2)}")

        for artifact_name, artifact_path in model_summary["artifacts"].items():
            if not Path(artifact_path).exists():
                raise FileNotFoundError(f"Expected forecast artifact '{artifact_name}' not found at {artifact_path}")
        report_lines.append("All forecast artifacts confirmed on disk.")

        report_lines.append("Test run completed successfully")
    except Exception:
        tb = traceback.format_exc()
        report_lines.append("Test run failed:\n" + tb)
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Wrote test report to {report_path}")
        raise
    else:
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"Wrote test report to {report_path}")


if __name__ == "__main__":
    main()
