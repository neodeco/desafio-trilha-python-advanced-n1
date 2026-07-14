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
2. Run ETL: `python -m app.glue_job --input files/raw_stock_data.csv --output output/processed_stock_data.parquet`
3. Run model training: `python scripts/spark_predictive_model.py --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model`
4. Inspect results in `output/model/training_results.csv` and `output/model/test_results.txt`.
