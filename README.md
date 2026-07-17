# desafio-trilha-python-advanced-n1

This workspace contains an AWS Glue-style ingestion + ML forecasting workflow that runs locally
against LocalStack (S3) and can be adapted to real AWS Glue later.

## Architecture

The Streamlit UI never runs PySpark in-process. All heavy Spark/JVM work (ETL treatment and model
training) runs in isolated subprocesses invoked by `app/app.py`, which only reads the resulting
files back from disk. This keeps the UI responsive and avoids Spark's JVM (which writes temp files
under the project tree while Streamlit's file-watcher observes the same tree) from ever colliding
with Streamlit's widget/form state.

1. **Entrada de dados** (`app/app.py` + `scripts/data_processing.py`): accepts a CSV upload OR a
   ticker + date range (mutually exclusive - a CSV upload always takes precedence). Ticker history
   is fetched via `pandas_datareader.data.get_data_yahoo`. Either way, the *raw* CSV is saved to
   `files/from-input/{ticker-or-slug}.csv` - the input for the PySpark ETL step below. A quick
   pandas-only preview (`ProcessingResult.dataframe`) is used for warnings/UI feedback.
2. **Tratamento (PySpark ETL)** - subprocess `python -m app.glue_job --mode price-series` (see
   `app/glue_job.py` + `app/glue_pipeline.py::transform_price_series`): reads the raw CSV, auto-
   detects the date/close/symbol columns regardless of naming/casing, filters out the 7th distinct
   ticker when multiple symbols are present, normalizes ISO/dd-MM-yyyy dates and comma-decimal
   prices, deduplicates by date (keep-last) and limits the series to the last 365 days - entirely
   with `pyspark.sql`. Writes the treated `date;close` CSV to `files/from-file/{slug}.csv` (the
   same directory used by `scripts/localstack_pipeline_test.py`), an optimized Parquet copy to
   `output/processed_stock_data/{slug}.parquet`, and uploads that Parquet file to a LocalStack S3
   bucket (`processed-data` by default).
3. **Modelo de machine learning (PySpark)** - subprocess `python -m scripts.spark_predictive_model
   --mode forecast` (see `scripts/spark_predictive_model.py`): trains a PySpark MLlib
   `LinearRegression` model with a temporal split (no shuffling) to avoid overfitting, searching
   epochs/regularization to keep R2 (variance) between 0.90 and 0.97. Reports epochs/iterations, R2,
   RMSE and MAE; produces a past prediction (compared against the test split) and a 365-day future
   forecast. Training artifacts go to `output/analysis`, test/future predictions to
   `output/model-test`, and Parquet files from both stages go to `output/processed_stock_data`.
4. **Visualizacao** (`app/app.py` + `scripts/plotting.py`): reads the CSV/JSON artifacts produced by
   steps 2-3 back into pandas and renders warnings, metrics, an interactive Plotly chart (for
   economists' analysis) and a static comparative chart. Plots are saved to `output/plots`.

> **Nota sobre o Yahoo Finance**: `pandas_datareader.data.get_data_yahoo` depende de um endpoint
> HTML da Yahoo que foi descontinuado/alterado externamente; chamadas reais podem falhar com um erro
> de rede/HTTP mesmo com o ticker correto. Isso e uma limitacao externa do provedor de dados, nao do
> codigo deste projeto - o app trata essa falha como um `DataProcessingError` amigavel. Use um upload
> de CSV como alternativa quando o Yahoo estiver indisponivel.

## What is included

- A PySpark transformation job in `app/glue_pipeline.py` (legacy multi-symbol OHLCV +
  `transform_price_series` for single-symbol date/close series)
- Two runnable ETL entrypoints in `app/glue_job.py` (`--mode stock` legacy, `--mode price-series` new)
- A merged training/forecast CLI in `scripts/spark_predictive_model.py` (`--mode training` legacy,
  `--mode forecast` new)
- A LocalStack bootstrap script in `scripts/setup_localstack.py`
- An end-to-end LocalStack pipeline test in `scripts/localstack_pipeline_test.py`
- Regression tests in `tests/`

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

## Run the ETL (PySpark) transformation locally

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Ticker/CSV -> date/close CSV + Parquet + upload to LocalStack S3
python -m app.glue_job --mode price-series --input files/from-input/AAPL.csv --source-name AAPL

# Legacy multi-symbol OHLCV (COTAHIST-style training-set) transformation
python -m app.glue_job --mode stock --input files/training-set/sample.csv --output output/processed_stock_data.parquet
```

## Run the predictive model locally

```bash
# Activate the repository virtual environment first
.venv\Scripts\Activate.ps1

# Single-symbol date/close forecast (used by app.py and the LocalStack pipeline test)
python -m scripts.spark_predictive_model --mode forecast --forecast-input files/from-file/AAPL.csv --source-name AAPL

# Legacy multi-symbol COTAHIST training-set flow
python -m scripts.spark_predictive_model --mode training --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model
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

## Run the end-to-end LocalStack pipeline test

Exercises the full flow against a running LocalStack container: ETL (`app.glue_job --mode
price-series`) -> S3 upload verification -> forecast model (`scripts.spark_predictive_model --mode
forecast`) -> artifact verification. Auto-discovers a CSV under `files/from-input/` (produced by a
real ticker fetch or CSV upload) or synthesizes a deterministic sample so the test never depends on
network access.

```bash
source .venv/Scripts/activate
python scripts/localstack_pipeline_test.py --endpoint-url http://localhost:4566
```

## Run tests

```bash
source .venv/Scripts/activate
python -m pytest -q
```

## Notes

- The ETL jobs handle null values, cast numeric columns, normalize dates, and compute a daily
  percentage change column (legacy OHLCV mode) or a clean `date`/`close` series (price-series mode).
- Parquet output is written locally and uploaded to a LocalStack S3 bucket, which can be redirected
  to a real AWS S3 bucket when deployed to AWS Glue.
- LocalStack is configured for S3, SQS, and DynamoDB to mimic a basic AWS data platform environment.

