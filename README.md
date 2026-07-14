# desafio-trilha-python-advanced-n1

This workspace now contains a small AWS Glue-style ingestion workflow that can be run locally and adapted to AWS Glue later.

## What is included

- A PySpark transformation job in app/glue_pipeline.py
- A runnable entrypoint in app/glue_job.py
- A sample raw CSV file in files/raw_stock_data.csv
- A LocalStack bootstrap script in scripts/setup_localstack.py
- A regression test in tests/test_glue_pipeline.py

## Run the transformation locally

```bash
source .venv/Scripts/activate
python -m app.glue_job --input files/raw_stock_data.csv --output output/processed_stock_data.parquet
```

## Run the LocalStack bootstrap

```bash
source .venv/Scripts/activate
python scripts/setup_localstack.py
```

## Run tests

```bash
source .venv/Scripts/activate
python -m pytest -q
```

## Notes

- The job handles null values, casts numeric columns, normalizes dates, and computes a daily percentage change column.
- The output is written as Parquet to a local folder, which can be redirected to S3 when deployed to AWS Glue.
- LocalStack is configured for S3, SQS, and DynamoDB to mimic a basic AWS data platform environment.
