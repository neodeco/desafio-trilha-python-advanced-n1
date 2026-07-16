# desafio-trilha-python-advanced-n1

This workspace now contains a small AWS Glue-style ingestion workflow that can be run locally and adapted to AWS Glue later.

## Architecture (3 layers)

- **Exibicao** (`app/app.py`): Streamlit UI. Accepts either a CSV upload OR a ticker + date range
  (mutually exclusive - a CSV upload always takes precedence and ignores the ticker/date fields).
  Renders warnings, metrics, an interactive Plotly chart of the final prediction (for economists'
  analysis) and a static comparative chart.
- **Tratamento de dados** (`scripts/data_processing.py`): reads/validates the CSV (auto-detects and
  corrects the separator if it is not `;`), fetches ticker history via `pandas_datareader`, filters
  out the 7th distinct ticker when multiple tickers are present, limits the period to the last 365
  days, converts/validates dates, and produces the final `date`/`close` dataframe. Treated CSV input
  is saved under `files/analysis/`.
- **Modelo de machine learning** (`scripts/ml_model.py` + `scripts/plotting.py`): trains a PySpark
  MLlib `LinearRegression` model with a temporal split (no shuffling) to avoid overfitting, searching
  epochs/regularization to keep R2 (variance) between 0.90 and 0.97. Reports epochs/iterations, R2,
  RMSE and MAE; produces a past prediction (compared against the test split) and a 365-day future
  forecast. Training artifacts go to `output/analysis`, test artifacts to `output/model-test`, and
  Parquet files from both stages go to `output/processed_stock_data`. Plots are saved to
  `output/plots`.

## What is included

- A PySpark transformation job in app/glue_pipeline.py
- A runnable entrypoint in app/glue_job.py
- A sample raw CSV file in files/raw_stock_data.csv
- A LocalStack bootstrap script in scripts/setup_localstack.py
- A regression test in tests/test_glue_pipeline.py

## Install dependencies

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run app/app.py
```

## Run the transformation locally

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
# then run the pipeline
python -m app.glue_job --input files/raw_stock_data.csv --output output/processed_stock_data.parquet
```

## Run the predictive model locally

```bash
# Activate the repository virtual environment first
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
python scripts/spark_predictive_model.py --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model
```

## Automation & Monitoring (Glue)

Scripts are provided to create Glue jobs/triggers (works with AWS or LocalStack) and to poll job runs.

Create a job (example):

```bash
python scripts/glue_automation.py --create-job --job-name glue-etl-job --script-location s3://my-bucket/scripts/glue_job.py
```

Create a scheduled trigger (cron example):

```bash
python scripts/glue_automation.py --create-trigger --trigger-name daily-trigger --job-name glue-etl-job --cron 'cron(0 2 * * ? *)'
```

Monitor runs:

```bash
python scripts/monitor_glue_jobs.py --job-name glue-etl-job --interval 30
```

## Comparative time-series plot

Generate a comparative price+volume plot for `PETR4T`:

```bash
python scripts/comparative_series.py --symbol PETR4T --input-dir files/training-set --output-dir output/plots
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
