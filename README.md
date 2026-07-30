# Desafio Trilha Python - ETL e Machine Learning para Previsão de Séries Temporais de Ações

Este repositório implementa um fluxo local de ETL e machine learning para analisar séries temporais de ações e prever o valor de fechamento futuro. A interface principal é um app Streamlit que aceita um CSV ou um ticker, processa os dados com PySpark, treina um modelo de regressão e exibe previsões para os próximos dias.

## O que o projeto faz hoje

1. Entrada de dados
   - Aceita um arquivo CSV com colunas de data/fechamento ou um ticker para buscar histórico via yfinance.
   - O CSV bruto é salvo em files/from-input/ e, em seguida, tratado pelo ETL.

2. ETL com PySpark
   - O job app.glue_job --mode price-series converte a entrada em uma série limpa com colunas date e close.
   - O resultado é salvo em files/from-file/, exportado para Parquet em output/processed_stock_data/ e enviado para um bucket S3 do LocalStack.

3. Previsão de fechamento com ML
   - O módulo scripts/spark_predictive_model.py treina um modelo de regressão do PySpark MLlib para prever o valor de fechamento.
   - O problema não é de classificação (subiu/baixou). A ideia é estimar o preço futuro e inferir a tendência a partir da direção da série prevista.
   - A modelagem usa uma divisão temporal sem shuffle, features t, t² e log(t+1) e alvos em escala log (log(close+1)), o que melhora a extrapolação para séries financeiras.

4. Visualização
   - O app gera gráficos interativos e comparativos para mostrar os preços reais, as previsões do período de teste e a projeção futura.

## Arquitetura importante

- O app Streamlit não executa PySpark diretamente no processo da interface.
- O ETL e o treinamento do modelo rodam em subprocessos separados para evitar travamentos da JVM do Spark e conflitos com os widgets do Streamlit.
- Os resultados são lidos de volta do disco após a conclusão de cada subprocesso.

## Métricas de aprendizado de máquina

As métricas atuais são de regressão, não de classificação. As principais métricas exportadas são:

- train_r2: desempenho no conjunto de treino.
- test_r2: desempenho na parte de teste preservada temporalmente.
- rmse: erro quadrático médio da previsão.
- mae: erro absoluto médio.
- target_reached: indica se o modelo ficou dentro da faixa alvo de R² esperada, atualmente definida entre 0.55 e 0.97.

Em execuções recentes, os valores observados podem ficar longe dessa faixa. Por exemplo, um run recente para AAPL registrou:

- train_r2 = 0.3938
- test_r2 = -3.1534
- rmse = 26.7115
- mae = 24.1953
- target_reached = false

## Estrutura principal

- app/app.py: interface Streamlit e orquestração do fluxo.
- app/glue_job.py: ponto de entrada do ETL PySpark.
- app/glue_pipeline.py: lógica de tratamento dos dados.
- scripts/spark_predictive_model.py: treinamento, avaliação e geração de previsões.
- scripts/plotting.py: criação de gráficos comparativos e interativos.
- output/analysis/: métricas de treino e arquivos de busca de hiperparâmetros.
- output/model-test/: previsões de teste e futuras.
- output/plots/: gráficos exportados.

## Instalação

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Executar o app

```bash
streamlit run app/app.py
```

## Executar o ETL localmente

```bash
.venv\Scripts\Activate.ps1
python -m app.glue_job --mode price-series --input files/from-input/AAPL.csv --source-name AAPL
```

## Executar o modelo preditivo localmente

```bash
.venv\Scripts\Activate.ps1
python -m scripts.spark_predictive_model --mode forecast --forecast-input files/from-file/AAPL.csv --source-name AAPL
```

## Artefatos gerados

- files/from-file/{slug}.csv: série tratada date;close.
- output/processed_stock_data/{slug}.parquet: cópia parquet do dado tratado.
- output/analysis/{slug}_training_metrics_*.json: métricas do modelo.
- output/model-test/{slug}_test_predictions_*.csv: previsões no período de teste.
- output/model-test/{slug}_future_predictions_*.csv: projeção futura.
- output/plots/: gráficos exportados.

## Observações

- O fluxo foi projetado para rodar localmente com PySpark e LocalStack, mas pode ser adaptado para um ambiente AWS real.
- Se o Yahoo Finance não estiver disponível, o app ainda aceita CSV para que o processo continue.
- O fluxo atual exige uma única ação por arquivo; arquivos com múltiplos tickers distintos são rejeitados na etapa de normalização.
- O foco atual é a previsão de valor de fechamento, e não a classificação binária de tendência.
