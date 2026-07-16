from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

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
from scripts.ml_model import ModelTrainingError, train_predict_evaluate  # noqa: E402
from scripts.plotting import build_interactive_forecast_figure, plot_comparative_forecast  # noqa: E402


st.set_page_config(page_title="Previsao de precos", layout="wide")


@st.cache_data(show_spinner=False, ttl="15m", max_entries=20)
def load_ticker_data(ticker: str, start_date: date, end_date: date):
    return fetch_history_by_ticker(ticker, start_date, end_date)


@st.cache_data(show_spinner=False, ttl="15m", max_entries=20)
def load_csv_data(content: bytes, source_name: str):
    return process_csv_input(content, source_name=source_name)


def render_metrics(metrics: dict[str, float | int | str | bool]) -> None:
    cols = st.columns(5)
    cols[0].metric("R2 (variancia)", f"{float(metrics['r2']):.4f}")
    cols[1].metric("RMSE", f"{float(metrics['rmse']):.4f}")
    cols[2].metric("MAE", f"{float(metrics['mae']):.4f}")
    cols[3].metric("Iteracoes de busca", f"{int(metrics['iterations'])}")
    cols[4].metric("Epocas do modelo", f"{int(metrics['epochs'])}")

    target_status = "atingido" if metrics["target_reached"] else "nao atingido"
    st.caption(
        f"Modelo: {metrics['model']} | Split temporal (sem shuffle): "
        f"{metrics['train_rows']} treino / {metrics['test_rows']} teste | "
        f"Faixa de R2 alvo [{metrics['target_r2_min']:.2f} - {metrics['target_r2_max']:.2f}] {target_status}; "
        f"valor real {float(metrics['r2']):.4f}."
    )


st.title("Previsao de precos com serie temporal")
st.caption("Envie um CSV OU informe ticker + periodo. As duas entradas sao mutuamente exclusivas.")

with st.form("data_input", border=True):
    uploaded_csv = st.file_uploader("CSV com colunas date/close (separador ';')", type=["csv"])

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
    st.error(f"Erro inesperado ao processar os dados: {exc}")
    st.stop()

final_df = result.dataframe
warnings = result.warnings

try:
    forecast = train_predict_evaluate(final_df, future_days=365, source_name=source_label)
except ModelTrainingError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Erro inesperado ao treinar o modelo: {exc}")
    st.stop()

interactive_fig, interactive_path = build_interactive_forecast_figure(
    actual_df=final_df,
    past_predictions=forecast.past_predictions,
    future_predictions=forecast.future_predictions,
    filename_prefix=source_label,
)
fig, plot_path = plot_comparative_forecast(
    actual_df=final_df,
    past_predictions=forecast.past_predictions,
    future_predictions=forecast.future_predictions,
    filename_prefix=source_label,
)

for warning in warnings:
    st.warning(warning)

if result.saved_path is not None:
    st.caption(f"Dados tratados salvos em {result.saved_path}")

st.subheader("Metricas")
render_metrics(forecast.metrics)

st.subheader("Predicao final (analise interativa para economistas)")
st.plotly_chart(interactive_fig, use_container_width=True)
st.caption(f"Grafico interativo salvo em {interactive_path}")

st.subheader("Grafico comparativo")
st.pyplot(fig)
st.caption(f"Grafico comparativo salvo em {plot_path}")

st.subheader("Dataframe final")
display_df = format_dates_for_display(final_df)
st.dataframe(
    display_df,
    hide_index=True,
    column_config={
        "date": st.column_config.TextColumn("Data"),
        "close": st.column_config.NumberColumn("Fechamento", format="%.4f"),
    },
)
