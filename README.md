# desafio-trilha-python-advanced-n1

Este workspace contém um fluxo de trabalho de ingestão no estilo AWS Glue + previsão com ML (Machine Learning) que roda localmente usando o LocalStack (S3) e pode ser adaptado para o AWS Glue real posteriormente.

## Arquitetura

A interface do Streamlit nunca executa o PySpark em processo ativo. Todo o trabalho pesado de Spark/JVM (tratamento de ETL e treinamento do modelo) roda em subprocessos isolados invocados por `app/app.py`, que apenas lê os arquivos resultantes de volta do disco. Isso mantém a interface responsiva e evita que a JVM do Spark (que grava arquivos temporários na árvore do projeto enquanto o observador de arquivos do Streamlit monitora essa mesma árvore) colida com o estado dos widgets/formulários do Streamlit.

1. **Entrada de dados** (`app/app.py` + `scripts/data_processing.py`): aceita o upload de um arquivo CSV OU um ticker + intervalo de datas (mutuamente exclusivos - o upload do CSV sempre tem precedência). O histórico do ticker é buscado via `pandas_datareader.data.get_data_yahoo`. De qualquer forma, o CSV *bruto* é salvo em `files/from-input/{ticker-or-slug}.csv` — que é a entrada para a etapa de ETL do PySpark abaixo. Uma pré-visualização rápida feita apenas com pandas (`ProcessingResult.dataframe`) é usada para avisos/retornos na interface de usuário.
2. **Tratamento (PySpark ETL)** — subprocesso `python -m app.glue_job --mode price-series` (veja `app/glue_job.py` + `app/glue_pipeline.py::transform_price_series`): lê o CSV bruto, detecta automaticamente as colunas de data/fechamento/símbolo independentemente da nomenclatura/capitalização (maiúsculas e minúsculas), descarta o 7º ticker distinto quando múltiplos símbolos estão presentes, normaliza datas ISO/dd-MM-yyyy e preços com vírgula decimal, remove duplicatas por data (mantendo o último registro) e limita a série aos últimos 365 dias — tudo feito inteiramente com `pyspark.sql`. Grava o CSV tratado `date;close` em `files/from-file/{slug}.csv` (o mesmo diretório usado por `scripts/localstack_pipeline_test.py`), exporta uma cópia otimizada em Parquet para `output/processed_stock_data/{slug}.parquet` e faz o upload desse arquivo Parquet para um bucket S3 do LocalStack (`processed-data` por padrão).
3. **Modelo de machine learning (PySpark)** — subprocesso `python -m scripts.spark_predictive_model --mode forecast` (veja `scripts/spark_predictive_model.py`): treina um modelo `LinearRegression` do PySpark MLlib com uma divisão temporal (sem embaralhamento/*shuffling*) para evitar overfitting, buscando épocas/regularização para manter o R2 (variância) entre 0,90 e 0,97. Reporta épocas/iterações, R2, RMSE e MAE; gera uma previsão passada (comparada com a divisão de teste) e uma previsão futura de 365 dias. Os artefatos de treinamento vão para `output/analysis`, as previsões de teste/futuras vão para `output/model-test`, e os arquivos Parquet de ambas as etapas vão para `output/processed_stock_data`.
4. **Visualização** (`app/app.py` + `scripts/plotting.py`): lê os artefatos CSV/JSON produzidos pelas etapas 2 e 3 de volta para o pandas e renderiza avisos, métricas, um gráfico interativo do Plotly (para análise de economistas) e um gráfico estático comparativo. Os gráficos são salvos em `output/plots`.

> **Nota sobre o Yahoo Finance**: O `pandas_datareader.data.get_data_yahoo` depende de um endpoint HTML do Yahoo que foi descontinuado/alterado externamente; chamadas reais podem falhar com um erro de rede/HTTP mesmo com o ticker correto. Isso é uma limitação externa do provedor de dados, não do código deste projeto — o app trata essa falha como um erro amigável do tipo `DataProcessingError`. Use o upload de CSV como alternativa quando o Yahoo estiver indisponível.

## O que está incluso

- Um job de transformação PySpark em `app/glue_pipeline.py` (legado multi-símbolo OHLCV + `transform_price_series` para séries de data/fechamento de símbolo único)
- Dois pontos de entrada (entrypoints) de ETL executáveis em `app/glue_job.py` (legado `--mode stock`, novo `--mode price-series`)
- Uma CLI integrada de treinamento/previsão em `scripts/spark_predictive_model.py` (legado `--mode training`, novo `--mode forecast`)
- Um script de inicialização (*bootstrap*) do LocalStack em `scripts/setup_localstack.py`
- Um teste de pipeline de ponta a ponta (*end-to-end*) do LocalStack em `scripts/localstack_pipeline_test.py`
- Testes de regressão em `tests/`

## Instalação das dependências

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

## Executar o app do Streamlit

```bash
streamlit run app/app.py

```

## Executar a transformação de ETL (PySpark) localmente

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Ticker/CSV -> CSV de data/fechamento + Parquet + upload para o S3 do LocalStack
python -m app.glue_job --mode price-series --input files/from-input/AAPL.csv --source-name AAPL

# Transformação legada multi-símbolo OHLCV (conjunto de treino estilo COTAHIST)
python -m app.glue_job --mode stock --input files/training-set/sample.csv --output output/processed_stock_data.parquet

```

## Executar o modelo preditivo localmente

```bash
# Ative o ambiente virtual do repositório primeiro
.venv\Scripts\Activate.ps1

# Previsão de símbolo único de data/fechamento (usado pelo app.py e pelo teste de pipeline do LocalStack)
python -m scripts.spark_predictive_model --mode forecast --forecast-input files/from-file/AAPL.csv --source-name AAPL

# Fluxo legado multi-símbolo de conjunto de treino COTAHIST
python -m scripts.spark_predictive_model --mode training --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model

```

## Automação e Monitoramento (Glue)

São fornecidos scripts para criar jobs/triggers do Glue (funciona com AWS real ou LocalStack) e para monitorar as execuções dos jobs.

Criar um job (exemplo):

```bash
python scripts/glue_automation.py --create-job --job-name glue-etl-job --script-location s3://meu-bucket/scripts/glue_job.py

```

Criar um gatilho (*trigger*) agendado (exemplo com cron):

```bash
python scripts/glue_automation.py --create-trigger --trigger-name daily-trigger --job-name glue-etl-job --cron 'cron(0 2 * * ? *)'

```

Monitorar execuções:

```bash
python scripts/monitor_glue_jobs.py --job-name glue-etl-job --interval 30

```

## Gráfico comparativo de séries temporais

Gerar um gráfico comparativo de preço + volume para `PETR4T`:

```bash
python scripts/comparative_series.py --symbol PETR4T --input-dir files/training-set --output-dir output/plots

```

## Executar a inicialização (bootstrap) do LocalStack

```bash
source .venv/Scripts/activate
python scripts/setup_localstack.py

```

## Executar o teste de pipeline de ponta a ponta do LocalStack

Executa o fluxo completo contra um container ativo do LocalStack: ETL (`app.glue_job --mode price-series`) -> verificação do upload no S3 -> modelo de previsão (`scripts.spark_predictive_model --mode forecast`) -> verificação de artefatos. Detecta automaticamente um CSV em `files/from-input/` (gerado por uma busca real de ticker ou upload de CSV) ou sintetiza uma amostra determinística para que o teste nunca dependa de acesso à rede.

```bash
source .venv/Scripts/activate
python scripts/localstack_pipeline_test.py --endpoint-url http://localhost:4566

```

## Executar os testes

```bash
source .venv/Scripts/activate
python -m pytest -q

```

## Notas

* Os jobs de ETL lidam com valores nulos, convertem tipos de colunas numéricas, normalizam datas e calculam uma coluna de variação percentual diária (modo legado OHLCV) ou uma série limpa de `date`/`close` (modo price-series).
* A saída em Parquet é gravada localmente e enviada para um bucket S3 do LocalStack, que pode ser redirecionado para um bucket S3 real da AWS quando implantado no AWS Glue.
* O LocalStack está configurado para S3, SQS e DynamoDB para simular um ambiente básico de plataforma de dados da AWS.

