# Technical Tech Doc

## Overview

This document captures the full development history, steps, warnings, errors, and corrections applied during the project from the beginning through the current predictive modeling phase.

## Project Purpose

The repository implements a local AWS Glue-style PySpark ETL workflow that:
- ingests fixed-width stock market data from `.TXT` files,
- converts it to CSV,
- processes and cleans the data with Spark,
- writes Parquet output locally,
- uploads processed output to LocalStack S3,
- performs exploratory analysis,
- and builds a Spark MLlib regression pipeline to predict next-day closing prices.

## Initial Setup and Scope

### Environment

- Python 3.14 via `.venv`
- PySpark for ETL and modeling
- Boto3 with LocalStack for local AWS emulation
- Pandas/Matplotlib/Seaborn for analysis
- GitHub Actions CI workflow partially in place
#### Commands

```bash
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
source .venv/Scripts/activate
# or on Windows PowerShell:
# .venv\Scripts\Activate.ps1
```
### Initial Files

- `app/glue_pipeline.py`: core Spark ETL logic
- `app/glue_job.py`: CLI runner and output management
- `scripts/setup_localstack.py`: LocalStack resource setup
- `scripts/convert_cotahist_to_csv.py`: fixed-width converter
- `scripts/exploratory_analysis.py`: analysis workflow
- `README.md`: documentation and instructions
- `tests/test_glue_pipeline.py`: ETL unit test

## Data Conversion and ETL

### Fixed-width converter

- Implemented `scripts/convert_cotahist_to_csv.py` to parse COTAHIST `.TXT` records.
- The converter uses a record layout and extracts fields from fixed slices.
- Normalization included symbol cleanup, date formatting, numeric parsing, and volume extraction.

#### Commands

```bash
# Inspect training/test dataset folders
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
ls files/training-set && ls files/test-set

# Convert a TXT file to CSV using the converter module
python scripts/convert_cotahist_to_csv.py
# or directly from Python if using the helper
python - <<'PY'
from scripts.convert_cotahist_to_csv import convert_to_csv
from pathlib import Path
convert_to_csv(Path('files/test-set/COTAHIST_A2020.TXT'), Path('files/test-set/COTAHIST_A2020.csv'))
PY
```

### Key Issues and Fixes

- Issue: Spark `Window` import missing from `app/glue_pipeline.py`.
  - Fix: imported `Window` and used window functions for `lag`/`lead` computations.

- Issue: Spark on Windows required explicit Hadoop path / winutils warnings.
  - Fix: acknowledged local Windows warning with `HADOOP_HOME` not set; the workflow still runs.

- Issue: Date parsing in some records did not account for `YYYYMM` values.
  - Fix: added logic to normalize `trade_date` to `YYYYMMDD` when needed.

- Issue: volume cast overflow when using `IntegerType`.
  - Fix: changed to `LongType` for volume.

- Issue: fixed-width parsing had incorrect string slicing around symbol/currency fields.
  - Fix: refined the converter parsing logic and added robust `parse_number` handling.

## LocalStack Integration

### Setup script

- Provided `scripts/setup_localstack.py` to bootstrap LocalStack resources.
- Resources included S3 bucket creation and any required local services.

#### Commands

```bash
# Bootstrap LocalStack resources
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
python scripts/setup_localstack.py
```

### AWS upload path

- `app/glue_job.py` was implemented to write Parquet locally and use `boto3.upload_file` to upload to LocalStack S3 in a simulated data platform flow.

## Exploratory Analysis

- Implemented `scripts/exploratory_analysis.py` for charting and summary statistics.
- Used pandas and seaborn/matplotlib to inspect processed Parquet.

#### Commands

```bash
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
python scripts/exploratory_analysis.py
```

## Predictive Modeling Phase

### New Script

- Added `scripts/spark_predictive_model.py` to train regression models with Spark MLlib.
- The script:
  - loads training data from `files/training-set`
  - converts any `.TXT` files to CSV when needed
  - preprocesses data with Spark
  - builds feature vectors from `open`, `high`, `low`, `volume`, `prev_close`
  - trains Linear Regression, Random Forest, and GBT models
  - validates on a hold-out split
  - evaluates the best model on `files/test-set/COTAHIST_A2020.TXT`
  - stores results in `output/model`

### Data Inspection

- Verified training directory contents include both CSV and `.TXT` files.
- Confirmed existing training CSV `files/training-set/cotahist_m072025.csv`.
- Confirmed test source file `files/test-set/COTAHIST_A2020.TXT` exists.
- Inspected training sample rows and confirmed data columns.

#### Commands

```bash
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
python - <<'PY'
import pandas as pd
from pathlib import Path
train = pd.read_csv('files/training-set/cotahist_m072025.csv')
print(train.head())
print(train['symbol'].value_counts().head())
PY
```

### Preprocessing Corrections

- Added robust row filtering to remove rows with NaN features before Spark feature assembly.
- Configured `VectorAssembler` with `handleInvalid="skip"`.
- Added test dataset validation to raise an error if no valid rows remain after filtering.

### Model Run Results

- The model run successfully completed in local testing.
- Selected symbol for modeling: `PETR4T`.
- Training/validation counts: `1440` training rows, `298` validation rows.
- Validation metrics:
  - `LinearRegression`: RMSE ~ 1.72e9, R2 ~ 0.162
  - `RandomForest`: RMSE ~ 1.56e9, R2 ~ 0.311
  - `GBT`: RMSE ~ 1.25e9, R2 ~ 0.554
- Best model: `gbt`
- Test metrics on `COTAHIST_A2020.TXT`:
  - RMSE ~ 4.86e9
  - MAE ~ 3.81e9
  - R2 ~ -1.241

#### Commands

```bash
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
.venv/Scripts/python -u scripts/spark_predictive_model.py --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model-test

# Inspect generated output files
python - <<'PY'
from pathlib import Path
print(list(Path('output/model-test').glob('*')))
PY
```

### Warnings and Observations

- Spark on Windows issues with missing `winutils.exe` and native Hadoop library load were observed but did not prevent the run.
- The selected symbol suffered from test set generalization drift, as evidenced by negative R2.
- The dataset contains a large number of symbols, and the model currently uses only the most frequent one by default.

## Documentation Updates

- Updated `README.md` to include instructions for running the new predictive model script.

## Current Files Added/Modified

- Added: `scripts/spark_predictive_model.py`
- Added: `TECHNICAL-TECH-DOC.md`
- Modified: `README.md`
- Verified existing files: `scripts/convert_cotahist_to_csv.py`, `app/glue_pipeline.py`, `app/glue_job.py`, `scripts/setup_localstack.py`, `scripts/exploratory_analysis.py`

## How to Run End-to-End

1. Activate the Python virtual environment.
2. Obtain a raw price CSV: either fetch a ticker via `pandas_datareader.data.get_data_yahoo` (through
   `app/app.py`, which saves it to `files/from-input/{ticker}.csv`) or upload a CSV.
3. Run the PySpark ETL job: `python -m app.glue_job --mode price-series --input files/from-input/AAPL.csv --source-name AAPL`.
   This writes `files/from-file/AAPL.csv` (`date;close`), `output/processed_stock_data/AAPL.parquet`,
   and uploads the Parquet file to the LocalStack S3 bucket `processed-data`.
4. Run the forecast model: `python -m scripts.spark_predictive_model --mode forecast --forecast-input files/from-file/AAPL.csv --source-name AAPL`.
5. Inspect results in `output/analysis/` (training search/metrics) and `output/model-test/`
   (test/future predictions and metrics).
6. (Optional) run the legacy multi-symbol COTAHIST flow:
   `python -m scripts.spark_predictive_model --mode training --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model`
   and inspect `output/model/training_results.csv` / `output/model/test_results.txt`.

## Automation & Monitoring Commands

The automation scripts can be used against AWS or LocalStack (provide `--endpoint-url` to point to LocalStack).

```bash
# Create a Glue job (example)
python scripts/glue_automation.py --create-job --job-name glue-etl-job --script-location s3://my-bucket/scripts/glue_job.py

# Create a scheduled trigger
python scripts/glue_automation.py --create-trigger --trigger-name daily-trigger --job-name glue-etl-job --cron 'cron(0 2 * * ? *)'

# Start a job run immediately
python scripts/glue_automation.py --start-job --job-name glue-etl-job

# Monitor job runs
python scripts/monitor_glue_jobs.py --job-name glue-etl-job --interval 30
```

## Comparative Plot Command

```bash
python scripts/comparative_series.py --symbol PETR4T --input-dir files/training-set --output-dir output/plots
```

## Ticker/CSV -> Spark -> LocalStack Refactor (Streamlit `st.form` fix)

### Problem

The Streamlit app occasionally raised: `There are multiple identical forms with key='data_input'.`
This happened because PySpark (training/ETL) ran synchronously **inside** the Streamlit process.
Spark's JVM blocks the Python interpreter for extended periods and `build_spark_session()` points
`HADOOP_HOME` at the project's own working directory, so the JVM's temp files under that tree were
observed by Streamlit's file-watcher, triggering concurrent reruns that collided on the `st.form`
widget key.

### Fix: subprocess isolation

All PySpark work now runs in standalone subprocesses invoked from `app/app.py` via
`subprocess.run([sys.executable, "-m", ...])`; the Streamlit process itself never imports PySpark or
starts a JVM. Each subprocess prints a final JSON summary line to stdout, which `app.py` parses to
locate the output files and metrics to render.

- `python -m app.glue_job --mode price-series --input <raw.csv> --source-name <slug>` (PySpark ETL)
- `python -m scripts.spark_predictive_model --mode forecast --forecast-input <treated.csv> --source-name <slug>` (PySpark MLlib)

### `pandas_datareader.data.get_data_yahoo`

Per request, ticker history is fetched with `pandas_datareader.data.get_data_yahoo` instead of the
previous (already broken upstream) `DataReader(ticker, "stooq", ...)` call.

- `pandas-datareader` 0.11.1 (latest at the time) removed `get_data_yahoo` and the `"stooq"` source
  entirely, so the previous code was already non-functional with the installed version.
- `pandas-datareader==0.10.0` still has `get_data_yahoo`, but its import breaks under recent pandas
  releases because `pandas.util._decorators.deprecate_kwarg` gained a required leading `klass`
  argument. `scripts/data_processing.py::_ensure_pandas_datareader_compat()` shims this function
  (feature-detected, no-op on pandas versions that don't need it) before importing
  `pandas_datareader.data`.
- Yahoo Finance's HTML-scraped endpoint used internally by `get_data_yahoo` has been discontinued and
  currently returns a 404 for real network calls - a known, unfixable **external** limitation, not a
  bug in this codebase. Failures are wrapped as `DataProcessingError` with a clear message; tests
  mock the call so they remain deterministic regardless of Yahoo's live availability. CSV upload
  remains a fully working alternative when Yahoo is unreachable.
- The raw fetched (or uploaded) data is always saved to `files/from-input/{ticker-or-slug}.csv`,
  which becomes the input to the PySpark ETL step.

### `app/glue_pipeline.py::transform_price_series`

A new PySpark-SQL transform normalizes any raw price CSV (ticker download or upload) down to
`date`/`close`, replicating the business rules that used to live in pandas
(`scripts/data_processing.py::finalize_price_dataframe`): auto-detects date/close/symbol columns
case-insensitively, drops the 7th distinct ticker with a warning when multiple symbols are present,
parses ISO/dd-MM-yyyy/compact-`yyyyMM` dates and comma-decimal prices via `try_to_date`/`try_cast`
(required because Spark 4.x ANSI mode raises instead of returning null on `to_date`), deduplicates by
date keeping the last occurrence, and trims the series to the last 365 days. Covered by
`tests/test_glue_pipeline.py`.

### `app/glue_job.py --mode price-series`

New CLI mode wraps `transform_price_series`: reads the raw CSV (auto-detected separator), writes the
treated `date;close` CSV to `files/from-file/{slug}.csv`, writes a Parquet copy to
`output/processed_stock_data/{slug}.parquet`, and uploads the Parquet file to a LocalStack S3 bucket
(auto-created if missing). Prints a JSON summary line for subprocess callers. The legacy `--mode
stock` behavior (multi-symbol OHLCV) is preserved unchanged.

### Merging `scripts/ml_model.py` into `scripts/spark_predictive_model.py`

`scripts/ml_model.py` was merged into `scripts/spark_predictive_model.py` and removed. The merged
file now exposes both flows:

- `--mode forecast --forecast-input <csv>`: the single-symbol date/close forecast model (former
  `ml_model.py` content: `ForecastResult`, `ModelTrainingError`, `train_predict_evaluate`, R2-band
  hyperparameter search, artifact persistence). Persistence was extended so
  `future_predictions.csv` is now also written to `output/model-test/` (previously only
  `past_predictions`/metrics were saved) - required because `app.py` reads it back from disk after
  the subprocess exits, with no in-memory `ForecastResult` object available across the process
  boundary.
- `--mode training --training-dir ... --test-file ...`: the original multi-symbol COTAHIST
  LinearRegression/RandomForest/GBT comparison flow, unchanged.

### `scripts/localstack_pipeline_test.py`

Rewritten to exercise the new price-series flow end-to-end against a live LocalStack container:
ensures LocalStack resources, auto-discovers a CSV under `files/from-input/` (or synthesizes a
deterministic sample so the test never depends on network access), runs `app.glue_job --mode
price-series`, verifies the resulting Parquet object exists in S3 via `boto3`, runs
`scripts.spark_predictive_model --mode forecast` against the treated CSV, and verifies all forecast
artifacts exist on disk. Writes a report to `output/localstack_test_results.txt`.

### Repository cleanup

- Removed `scripts/ml_model.py` (merged into `scripts/spark_predictive_model.py`).
- Untracked accidentally-committed LocalStack runtime state (`docker/volume/**` - certs, license,
  machine id), stale Parquet binaries (`output/processed_stock_data/*.parquet`), and compiled
  bytecode (`scripts/__pycache__/*.pyc`) from git; added `docker/volume/`, `output/`, `*.parquet`,
  `*.pyc` to `.gitignore`.
- Extracted shared CSV helpers (`detect_csv_separator*`, `slugify`) that were duplicated across
  `scripts/data_processing.py`, `scripts/comparative_series.py` and `scripts/spark_predictive_model.py`
  into a new `scripts/csv_utils.py` module.

