from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_processing import (  # noqa: E402
    DataProcessingError,
    fetch_history_by_ticker,
    format_dates_for_display,
    process_csv_input,
)
from scripts.plotting import build_interactive_forecast_figure, plot_comparative_forecast  # noqa: E402


st.set_page_config(page_title="Previsao de precos", layout="wide")


class SubprocessJobError(RuntimeError):
    """Raised when a PySpark subprocess (ETL or forecast) fails."""


def _run_json_subprocess(command: list[str]) -> dict:
    """Run ``command`` as a subprocess and parse the last stdout line as JSON.

    All PySpark work (ETL treatment and model training/forecast) runs in a
    separate process/JVM, never inside the Streamlit process. This is what
    prevents Spark's JVM (blocking calls, temp files under the project tree
    observed by Streamlit's file-watcher) from triggering concurrent reruns
    that collide on ``st.form`` widget keys.
    """
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SubprocessJobError(
            f"Comando '{' '.join(command)}' falhou (codigo {result.returncode}).\n{result.stderr[-2000:]}"
        )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise SubprocessJobError(
        f"Comando '{' '.join(command)}' nao retornou um resumo JSON valido.\n{result.stdout[-2000:]}"
    )


@st.cache_data(show_spinner=False, ttl="15m", max_entries=20)
def load_ticker_data(ticker: str, start_date: date, end_date: date):
    return fetch_history_by_ticker(ticker, start_date, end_date)


@st.cache_data(show_spinner=False, ttl="15m", max_entries=20)
def load_csv_data(content: bytes, source_name: str):
    return process_csv_input(content, source_name=source_name)


@st.cache_data(show_spinner=False, ttl="15m", max_entries=20)
def run_price_series_etl(raw_csv_path: str, source_name: str) -> dict:
    """Treat a raw CSV via PySpark (app/glue_job.py), producing a date/close CSV
    in files/from-file/, a Parquet copy, and an upload to S3 via LocalStack."""
    return _run_json_subprocess(
        [
            sys.executable,
            "-m",
            "app.glue_job",
            "--mode",
            "price-series",
            "--input",
            raw_csv_path,
            "--source-name",
            source_name,
        ]
    )


@st.cache_data(show_spinner=False, ttl="15m", max_entries=20)
def run_forecast_model(treated_csv_path: str, source_name: str) -> dict:
    """Train/evaluate the PySpark forecast model (scripts/spark_predictive_model.py)
    against a treated date/close CSV."""
    return _run_json_subprocess(
        [
            sys.executable,
            "-m",
            "scripts.spark_predictive_model",
            "--mode",
            "forecast",
            "--forecast-input",
            treated_csv_path,
            "--source-name",
            source_name,
        ]
    )


def render_metrics(metrics: dict[str, float | int | str | bool]) -> None:
    cols = st.columns(6)
    cols[0].metric("R2 treino", f"{float(metrics['train_r2']):.4f}")
    cols[1].metric("R2 teste", f"{float(metrics['test_r2']):.4f}")
    cols[2].metric("RMSE", f"{float(metrics['rmse']):.4f}")
    cols[3].metric("MAE", f"{float(metrics['mae']):.4f}")
    cols[4].metric("Iteracoes de busca", f"{int(metrics['iterations'])}")
    cols[5].metric("Epocas do modelo", f"{int(metrics['epochs'])}")

    train_target_status = "atingido" if metrics["train_target_reached"] else "nao atingido"
    test_target_status = "atingido" if metrics["test_target_reached"] else "nao atingido"
    st.caption(
        f"Modelo: {metrics['model']} | Split temporal (sem shuffle): "
        f"{metrics['train_rows']} treino / {metrics['test_rows']} teste | "
        f"Faixa de R2 alvo [{metrics['target_r2_min']:.2f} - {metrics['target_r2_max']:.2f}] | "
        f"treino {train_target_status} ({float(metrics['train_r2']):.4f}) | "
        f"teste {test_target_status} ({float(metrics['test_r2']):.4f})."
    )


st.title("Previsao de precos com serie temporal")
st.caption(
    "Envie um CSV OU informe ticker + periodo. As duas entradas sao mutuamente exclusivas. "
    "O tratamento (PySpark) e o treinamento do modelo rodam em subprocessos separados, "
    "integrados ao LocalStack (S3), para nao bloquear a interface do Streamlit."
)

with st.form("data_input", border=True):
    uploaded_csv = st.file_uploader("CSV com colunas date/close (separador ';' ou ',')", type=["csv"])

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="AAPL", disabled=uploaded_csv is not None)
    with col2:
        start_date = st.date_input(
            "Data inicio", value=date.today() - timedelta(days=365), disabled=uploaded_csv is not None
        )
    with col3:
        end_date = st.date_input("Data fim", value=date.today(), disabled=uploaded_csv is not None)

    submitted = st.form_submit_button("Processar", type="primary", icon=":material/play_arrow:")

if not submitted:
    st.info("Envie um CSV ou informe ticker e periodo para iniciar.")
    st.stop()

# Step 1: obtain the raw input CSV (pandas-only, no PySpark) -----------------
try:
    if uploaded_csv is not None:
        result = load_csv_data(uploaded_csv.getvalue(), uploaded_csv.name)
        source_label = uploaded_csv.name
        st.caption("CSV enviado; ticker e datas foram ignorados.")
    else:
        result = load_ticker_data(ticker, start_date, end_date)
        source_label = ticker.strip().upper()
except DataProcessingError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001 - surface any unexpected error to the user safely
    st.error(f"Erro inesperado ao obter os dados: {exc}")
    st.stop()

if result.raw_csv_path is None:
    st.error("Nao foi possivel localizar o CSV bruto gerado para o tratamento via PySpark.")
    st.stop()

source_slug = Path(result.raw_csv_path).stem

# Step 2: PySpark ETL (subprocess) -> files/from-file/{slug}.csv + S3 (LocalStack)
try:
    with st.spinner("Tratando os dados com PySpark (app/glue_job.py)..."):
        etl_summary = run_price_series_etl(str(result.raw_csv_path), source_slug)
except SubprocessJobError as exc:
    st.error(f"Erro ao tratar os dados com PySpark: {exc}")
    st.stop()

for warning in etl_summary.get("warnings", []):
    st.warning(warning)

st.caption(
    f"CSV tratado salvo em {etl_summary['csv_path']} | "
    f"Parquet em {etl_summary['parquet_path']} | "
    f"S3 (LocalStack): s3://{etl_summary['bucket']}/{etl_summary['key']}"
)

# Step 3: PySpark forecast model (subprocess) --------------------------------
try:
    with st.spinner("Executando treino/predicao com PySpark (scripts/spark_predictive_model.py)..."):
        forecast_summary = run_forecast_model(etl_summary["csv_path"], source_slug)
except SubprocessJobError as exc:
    st.error(f"Erro inesperado ao treinar o modelo: {exc}")
    st.stop()

metrics = forecast_summary["metrics"]
artifacts = forecast_summary["artifacts"]

# Step 4: read back results from disk (pandas-only, no PySpark) --------------
final_df = pd.read_csv(etl_summary["csv_path"], sep=";")
final_df["date"] = pd.to_datetime(final_df["date"])

past_predictions = pd.read_csv(artifacts["test_predictions"])
past_predictions["date"] = pd.to_datetime(past_predictions["date"])

future_predictions = pd.read_csv(artifacts["future_predictions"])
future_predictions["date"] = pd.to_datetime(future_predictions["date"])

interactive_fig, interactive_path = build_interactive_forecast_figure(
    actual_df=final_df,
    past_predictions=past_predictions,
    future_predictions=future_predictions,
    filename_prefix=source_label,
    ticker=source_label,
)
fig, plot_path = plot_comparative_forecast(
    actual_df=final_df,
    past_predictions=past_predictions,
    future_predictions=future_predictions,
    filename_prefix=source_label,
    ticker=source_label,
)

st.subheader("Metricas")
render_metrics(metrics)
if bool(metrics.get("from_cache", False)):
    st.info("Dados de ticker e periodo ja treinados anteriormente. Predicoes reaproveitadas do cache.")

st.subheader(f"Predicao final para {source_label} (analise interativa para economistas)")
st.plotly_chart(interactive_fig)
st.caption(f"Grafico interativo salvo em {interactive_path}")

st.subheader("Grafico comparativo")
st.pyplot(fig)
st.caption(f"Grafico comparativo salvo em {plot_path}")

st.subheader("Dataframe final (tratado via PySpark)")
display_df = format_dates_for_display(final_df)
st.dataframe(
    display_df,
    hide_index=True,
    column_config={
        "date": st.column_config.TextColumn("Data"),
        "close": st.column_config.NumberColumn("Fechamento", format="%.4f"),
    },
)
