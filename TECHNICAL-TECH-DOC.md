Aqui está a tradução técnica do documento `.md` para o português brasileiro, mantendo toda a formatação original, blocos de código, comandos e terminologias padrões da área de engenharia de dados (como *pipeline*, *workflow*, *deploy*, etc.):

---

# Documento técnico do projeto

## Visão geral

Este repositório reúne um fluxo local de ETL e modelagem preditiva para séries temporais de ações. O objetivo atual é transformar dados brutos em uma série tratada de `date/close`, treinar um modelo de regressão com PySpark e gerar previsões para o fechamento futuro.

## Arquitetura atual

A aplicação Streamlit em `app/app.py` não executa PySpark diretamente. Em vez disso, ela chama subprocessos separados para:

1. tratar a entrada via ETL em `app.glue_job.py`;
2. treinar e avaliar o modelo em `scripts/spark_predictive_model.py`.

Essa separação evita conflitos com a JVM do Spark e mantém a interface responsiva.

## Fluxo de processamento

### 1. Entrada e ingestão

A aplicação aceita:

- um CSV com colunas de data e fechamento; ou
- um ticker para buscar o histórico via `yfinance`.

Os dados brutos são salvos em `files/from-input/` e servem de entrada para o ETL.

### 2. ETL com PySpark

O job `python -m app.glue_job --mode price-series` produz:

- um CSV tratado em `files/from-file/{slug}.csv`;
- um Parquet em `output/processed_stock_data/{slug}.parquet`;
- um upload para o bucket S3 do LocalStack.

O formato tratado é simples: `date` e `close`.

### 3. Modelo de machine learning

O script `scripts/spark_predictive_model.py` implementa um fluxo de previsão para uma única ação. Ele:

- carrega a série tratada;
- realiza uma divisão temporal, sem shuffle;
- cria features baseadas em tempo (`t`, `t²`, `log(t+1)`);
- treina um `LinearRegression` do PySpark MLlib em escala log (`log(close+1)`);
- gera previsões para o período de teste e para um horizonte futuro de 30 dias, na tentativa de minimizar erros.

## Natureza do problema

O projeto não está fazendo classificação binária de tendência (subiu/baixou). O objetivo atual é prever o valor de fechamento futuro e, a partir disso, interpretar a tendência visualmente.

Portanto, as métricas relevantes são métricas de regressão:

- `train_r2`
- `test_r2`
- `rmse`
- `mae`
- `target_reached`

## Métricas atuais do modelo

O modelo exporta os resultados em arquivos JSON em `output/analysis/` e `output/model-test/`.

A métrica principal de qualidade é o R², mas o projeto também reporta RMSE e MAE para dar contexto ao erro absoluto e ao erro quadrático.

A faixa alvo atualmente usada é:

- R² mínimo: `0.55`
- R² máximo: `0.97`

Se o valor ficar fora dessa janela, o indicador `target_reached` passa a ser `false`.

### Exemplo de resultado recente

Um exemplo real de execução para a ação da Apple `AAPL` registrou:

- `train_r2 = 0.3938`
- `test_r2 = -3.1534`
- `rmse = 26.7115`
- `mae = 24.1953`
- `target_reached = false`

Esse resultado indica que, para esse conjunto de dados e para essa configuração atual, a previsão ainda não atingiu a faixa de qualidade esperada.

## Artefatos produzidos

- `output/analysis/{slug}_training_metrics_*.json`: métricas de treino.
- `output/analysis/{slug}_training_search_*.csv`: tentativas de hiperparâmetros.
- `output/model-test/{slug}_test_predictions_*.csv`: previsões no período de teste.
- `output/model-test/{slug}_future_predictions_*.csv`: projeções futuras.
- `output/plots/`: gráficos gerados para análise.

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

## Próximos passos possíveis

- testar outros algoritmos de regressão ou modelos baseados em séries temporais;
- incluir validação adicional por janela temporal;
- acrescentar métricas de direção de tendência, como acurácia de sinal, sem perder o foco principal de regressão do fechamento.

### Problema

O aplicativo Streamlit ocasionalmente apresentava o erro: `There are multiple identical forms with key='data_input'.`
Isso acontecia porque o PySpark (treino/ETL) executava de forma síncrona **dentro** do processo do Streamlit.
A JVM do Spark bloqueia o interpretador Python por longos períodos e o `build_spark_session()` aponta o `HADOOP_HOME` para o próprio diretório de trabalho do projeto, fazendo com que os arquivos temporários da JVM sob essa árvore fossem detectados pelo observador de arquivos (file-watcher) do Streamlit, disparando execuções simultâneas que colidiam na chave do widget `st.form`.

### Solução: isolamento por subprocesso

Todo o trabalho do PySpark agora roda em subprocessos independentes invocados a partir de `app/app.py` via `subprocess.run([sys.executable, "-m", ...])`; o próprio processo do Streamlit nunca importa o PySpark ou inicia uma JVM. Cada subprocesso imprime uma linha final em formato JSON contendo o resumo na saída padrão (stdout), a qual o `app.py` analisa para localizar os arquivos de saída e métricas para renderização.

* `python -m app.glue_job --mode price-series --input <raw.csv> --source-name <slug>` (ETL do PySpark)
* `python -m scripts.spark_predictive_model --mode forecast --forecast-input <treated.csv> --source-name <slug>` (PySpark MLlib)

### `yfinance.download`

Conforme solicitado, o histórico do ticker é buscado com `yfinance.download` em vez da chamada anterior `pandas_datareader.data.get_data_yahoo`.

* A integração com `yfinance` elimina a dependência do *shim* de compatibilidade que antes era necessário para importar `pandas-datareader` em versões recentes do pandas.
* O download do Yahoo Finance continua sujeito a indisponibilidade de rede/API do provedor externo. As falhas são tratadas como `DataProcessingError` com mensagem clara; os testes simulam (mock) a chamada para permanecerem determinísticos, independentemente da disponibilidade do Yahoo. O envio de arquivo CSV continua sendo uma alternativa totalmente funcional caso o Yahoo esteja inacessível.
* Os dados brutos buscados (ou enviados) são sempre salvos em `files/from-input/{ticker-or-slug}.csv`, que se torna a entrada para a etapa de ETL do PySpark.

### `app/glue_pipeline.py::transform_price_series`

Uma nova transformação PySpark-SQL normaliza qualquer CSV bruto de preço (download de ticker ou upload) para apenas `date`/`close`, replicando as regras de negócio que antes ficavam no pandas (`scripts/data_processing.py::finalize_price_dataframe`): detecta colunas de data/fechamento/símbolo sem distinção de maiúsculas/minúsculas, descarta o 7º ticker distinto emitindo um aviso caso múltiplos símbolos estejam presentes, analisa datas no formato ISO/dd-MM-aaaa/compacto-`aaaaMM` e preços com vírgula decimal via `try_to_date`/`try_cast` (necessário pois o modo ANSI do Spark 4.x lança exceção em vez de retornar nulo em `to_date`), remove duplicatas por data mantendo a última ocorrência e limita a série temporal aos últimos 30 dias. Coberto por `tests/test_glue_pipeline.py`.

### `app/glue_job.py --mode price-series`

O novo modo de CLI encapsula o `transform_price_series`: lê o CSV bruto (separador auto-detectado), grava o CSV tratado `date;close` em `files/from-file/{slug}.csv`, grava uma cópia em Parquet em `output/processed_stock_data/{slug}.parquet` e faz o upload do arquivo Parquet para um bucket S3 do LocalStack (criado automaticamente caso não exista). Imprime uma linha de resumo em JSON para os chamadores de subprocesso. O comportamento legado `--mode stock` (OHLCV multi-símbolo) foi preservado sem alterações.

### Fusão de `scripts/ml_model.py` em `scripts/spark_predictive_model.py`

O arquivo `scripts/ml_model.py` foi mesclado ao `scripts/spark_predictive_model.py` e removido. O arquivo unificado agora expõe ambos os fluxos:

* `--mode forecast --forecast-input <csv>`: o modelo de previsão de data/fechamento de símbolo único (antigo conteúdo do `ml_model.py`: `ForecastResult`, `ModelTrainingError`, `train_predict_evaluate`, busca de hiperparâmetros por faixa de R2, persistência de artefatos). A persistência foi estendida de modo que o `future_predictions.csv` agora também é gravado em `output/model-test/` (anteriormente apenas as `past_predictions`/métricas eram salvas) — necessário pois o `app.py` realiza a leitura de volta do disco após a finalização do subprocesso, sem a existência de um objeto `ForecastResult` em memória compartilhado entre os limites do processo.
* `--mode training --training-dir ... --test-file ...`: o fluxo original de comparação multi-símbolo do COTAHIST (LinearRegression/RandomForest/GBT), sem alterações.

### `scripts/localstack_pipeline_test.py`

Reescrito para testar o novo fluxo de séries de preços de ponta a ponta contra um container do LocalStack ativo: garante a presença dos recursos do LocalStack, auto-detecta um CSV em `files/from-input/` (ou sintetiza uma amostra determinística para que o teste nunca dependa de acesso à rede), executa o `app.glue_job --mode price-series`, verifica se o objeto Parquet resultante existe no S3 via `boto3`, executa o `scripts.spark_predictive_model --mode forecast` contra o CSV tratado e verifica se todos os artefatos de previsão existem em disco. Grava um relatório em `output/localstack_test_results.txt`.

### Limpeza do Repositório

* Removido `scripts/ml_model.py` (mesclado em `scripts/spark_predictive_model.py`).
* Removido do rastreamento do git o estado de execução do LocalStack que havia sido commitado acidentalmente (`docker/volume/**` — certificados, licença, ID da máquina), binários Parquet desatualizados (`output/processed_stock_data/*.parquet`) e bytecode compilado (`scripts/__pycache__/*.pyc`); adicionados `docker/volume/`, `output/`, `*.parquet`, `*.pyc` ao `.gitignore`.
* Extraídos utilitários compartilhados de CSV (`detect_csv_separator*`, `slugify`) que estavam duplicados em `scripts/data_processing.py`, `scripts/comparative_series.py` e `scripts/spark_predictive_model.py` para um novo módulo `scripts/csv_utils.py`.