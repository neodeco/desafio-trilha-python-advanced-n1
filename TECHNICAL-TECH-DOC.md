Aqui está a tradução técnica do documento `.md` para o português brasileiro, mantendo toda a formatação original, blocos de código, comandos e terminologias padrões da área de engenharia de dados (como *pipeline*, *workflow*, *deploy*, etc.):

---

# Documento técnico do projeto

## Visão geral

Este repositório reúne um fluxo local de ETL e modelagem preditiva para séries temporais de ações. O objetivo atual é transformar dados brutos em uma série tratada de date/close, treinar um modelo de regressão com PySpark e gerar previsões de tendência para o fechamento futuro.

## Arquitetura atual

A aplicação Streamlit em app/app.py não executa PySpark diretamente. Em vez disso, ela chama subprocessos separados para:

1. tratar a entrada via ETL em app/glue_job.py;
2. treinar e avaliar o modelo em scripts/spark_predictive_model.py.

Essa separação evita conflitos com a JVM do Spark e mantém a interface responsiva.

## Fluxo de processamento

### 1. Entrada e ingestão

A aplicação aceita:

- um CSV com colunas de data e fechamento; ou
- um ticker para buscar o histórico via yfinance.

Os dados brutos são salvos em files/from-input/ e servem de entrada para o ETL.

### 2. ETL com PySpark

O job python -m app.glue_job --mode price-series produz:

- um CSV tratado em files/from-file/{slug}.csv;
- um Parquet em output/processed_stock_data/{slug}.parquet;
- um upload para o bucket S3 do LocalStack.

O formato tratado é simples: date e close.

### 3. Modelo de machine learning

O script scripts/spark_predictive_model.py implementa um fluxo de previsão para uma única ação. Ele:

- carrega a série tratada;
- realiza uma divisão temporal, sem shuffle;
- cria features baseadas em tempo (t, t², log(t+1));
- treina um LinearRegression do PySpark MLlib em escala log (log(close+1));
- gera previsões para o período de teste e para um horizonte futuro, definido pelo número de linhas de teste ou pelo parâmetro future-days.

## Natureza do problema

O projeto não faz classificação binária de tendência (subiu/baixou). O objetivo atual é prever o valor de fechamento futuro e, a partir disso, interpretar a tendência visualmente.

Portanto, as métricas relevantes são métricas de regressão:

- train_r2
- test_r2
- rmse
- mae
- target_reached

## Métricas atuais do modelo

O modelo exporta os resultados em arquivos JSON em output/analysis/ e output/model-test/.

A métrica principal de qualidade é o R², mas o projeto também reporta RMSE e MAE para dar contexto ao erro absoluto e ao erro quadrático.

A faixa alvo atualmente usada é:

- R² mínimo: 0.55
- R² máximo: 0.97

Se o valor ficar fora dessa janela, o indicador target_reached passa a ser false.

### Exemplo de resultado recente

Um exemplo real de execução para a ação da Apple AAPL registrou:

- train_r2 = 0.3938
- test_r2 = -3.1534
- rmse = 26.7115
- mae = 24.1953
- target_reached = false

## Regras de normalização implementadas

A transformação PySpark-SQL em app/glue_pipeline.py normaliza qualquer CSV bruto de preço (download de ticker ou upload) para apenas date/close. As regras atuais são:

- detecta colunas de data, fechamento e símbolo sem distinção de maiúsculas/minúsculas;
- exige uma única ação por arquivo; múltiplos tickers distintos são rejeitados;
- aceita datas em formato ISO, yyyyMMdd, dd/MM/yyyy e yyyy/MM/dd;
- converte preços com vírgula decimal para ponto decimal;
- remove linhas inválidas e duplicatas por data, mantendo a última ocorrência;
- limita a série ao período mais recente, com default de 365 dias (configurável em max_period_days).

## Artefatos produzidos

- output/analysis/{slug}_training_metrics_*.json: métricas de treino.
- output/analysis/{slug}_training_search_*.csv: tentativas de hiperparâmetros.
- output/model-test/{slug}_test_predictions_*.csv: previsões no período de teste.
- output/model-test/{slug}_future_predictions_*.csv: projeções futuras.
- output/plots/: gráficos gerados para análise.

## Execução local

### Ambiente

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### App Streamlit

```bash
streamlit run app/app.py
```

### ETL

```bash
python -m app.glue_job --mode price-series --input files/from-input/AAPL.csv --source-name AAPL
```

### Modelo preditivo

```bash
python -m scripts.spark_predictive_model --mode forecast --forecast-input files/from-file/AAPL.csv --source-name AAPL
```

## Pontos de atenção

- O código atual usa uma abordagem de regressão para prever o fechamento, e não um modelo de classificação.
- A extrapolação futura é mais sensível do que a previsão dentro do intervalo de treino.
- As métricas devem ser interpretadas como um sinal de desempenho do modelo, e não como uma garantia de precisão financeira.
- A execução com dados reais pode ter desempenho ruim se a série for curta, ruidosa ou não estacionária.
- O upload para o LocalStack depende de um container ativo em localhost:4566.

## Próximos passos possíveis

- testar outros algoritmos de regressão ou modelos baseados em séries temporais;
- incluir validação adicional por janela temporal;
- acrescentar métricas de direção de tendência, como acurácia de sinal, sem perder o foco principal de regressão do fechamento.

## Problema e solução de arquitetura

O aplicativo Streamlit apresentava o erro There are multiple identical forms with key='data_input' quando o PySpark era executado de forma síncrona dentro do processo da interface. A solução adotada foi isolar o trabalho do Spark em subprocessos independentes, invocados a partir de app/app.py via subprocess.run. Cada subprocesso imprime um resumo JSON final em stdout, que o app lê para localizar os artefatos de saída.

### Fluxos suportados

- python -m app.glue_job --mode price-series --input <raw.csv> --source-name <slug>
- python -m scripts.spark_predictive_model --mode forecast --forecast-input <treated.csv> --source-name <slug>

## Observações adicionais

- O histórico do ticker é buscado com yfinance.download.
- O fluxo de previsão persiste artefatos de treino, previsões de teste e previsões futuras em output/analysis/ e output/model-test/.
- O módulo scripts/csv_utils.py concentra utilidades compartilhadas para detecção de separador e geração de slug.