Aqui está a tradução técnica do documento `.md` para o português brasileiro, mantendo toda a formatação original, blocos de código, comandos e terminologias padrões da área de engenharia de dados (como *pipeline*, *workflow*, *deploy*, etc.):

---

# Documento Técnico (Tech Doc)

## Visão Geral

Este documento registra todo o histórico de desenvolvimento, etapas, avisos, erros e correções aplicadas durante o projeto, desde o início até a fase atual de modelagem preditiva.

## Objetivo do Projeto

O repositório implementa um fluxo de trabalho (workflow) ETL PySpark local no estilo AWS Glue que:

* ingere dados de mercado de ações em formato de largura fixa a partir de arquivos `.TXT`,
* converte-os para CSV,
* processa e limpa os dados com Spark,
* grava a saída em formato Parquet localmente,
* faz o upload da saída processada para o S3 do LocalStack,
* realiza análise exploratória,
* e constrói um pipeline de regressão do Spark MLlib para prever os preços de fechamento do dia seguinte.

## Configuração Inicial e Escopo

### Ambiente

* Python 3.14 via `.venv`
* PySpark para ETL e modelagem
* Boto3 com LocalStack para emulação local de AWS
* Pandas/Matplotlib/Seaborn para análise
* Fluxo de trabalho de CI do GitHub Actions parcialmente implementado

#### Comandos

```bash
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
source .venv/Scripts/activate
# ou no Windows PowerShell:
# .venv\Scripts\Activate.ps1

```

### Arquivos Iniciais

* `app/glue_pipeline.py`: lógica central de ETL do Spark
* `app/glue_job.py`: executor de CLI e gerenciamento de saídas
* `scripts/setup_localstack.py`: configuração de recursos do LocalStack
* `scripts/convert_cotahist_to_csv.py`: conversor de arquivos de largura fixa
* `scripts/exploratory_analysis.py`: workflow de análise exploratória
* `README.md`: documentação e instruções
* `tests/test_glue_pipeline.py`: teste unitário do ETL

## Conversão de Dados e ETL

### Conversor de Largura Fixa

* Implementado o script `scripts/convert_cotahist_to_csv.py` para analisar os registros do COTAHIST `.TXT`.
* O conversor utiliza um layout de registro e extrai os campos a partir de fatias (slices) fixas.
* A normalização incluiu a limpeza de símbolos, formatação de datas, conversão de tipos numéricos e extração de volume.

#### Comandos

```bash
# Inspecionar as pastas de conjuntos de dados de treino/teste
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
ls files/training-set && ls files/test-set

# Converter um arquivo TXT para CSV usando o módulo conversor
python scripts/convert_cotahist_to_csv.py
# ou diretamente pelo Python usando o helper auxiliar
python - <<'PY'
from scripts.convert_cotahist_to_csv import convert_to_csv
from pathlib import Path
convert_to_csv(Path('files/test-set/COTAHIST_A2020.TXT'), Path('files/test-set/COTAHIST_A2020.csv'))
PY

```

### Principais Problemas e Correções

* **Problema:** Importação do Spark `Window` ausente em `app/glue_pipeline.py`.
* **Correção:** Importado o `Window` e utilizadas funções de janela para os cálculos de `lag`/`lead`.


* **Problema:** O Spark no Windows exigia um caminho explícito do Hadoop / avisos de winutils.
* **Correção:** Reconhecido o aviso local do Windows com o `HADOOP_HOME` não configurado; o workflow continua executando normalmente.


* **Problema:** A conversão de data em alguns registros não considerava valores no formato `AAAAMM`.
* **Correção:** Adicionada lógica para normalizar `trade_date` para `AAAAMMDD` quando necessário.


* **Problema:** Estouro de capacidade (overflow) na conversão de volume ao usar `IntegerType`.
* **Correção:** Alterado para `LongType` para o campo de volume.


* **Problema:** A análise de largura fixa apresentava fatiamento de string incorreto nos campos de símbolo/moeda.
* **Correção:** Refinada a lógica de análise do conversor e adicionado tratamento robusto em `parse_number`.



## Integração com LocalStack

### Script de Configuração

* Disponibilizado o script `scripts/setup_localstack.py` para inicializar os recursos do LocalStack.
* Os recursos incluíram a criação do bucket S3 e quaisquer serviços locais necessários.

#### Comandos

```bash
# Inicializar os recursos do LocalStack
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
python scripts/setup_localstack.py

```

### Caminho de Upload para AWS

* O script `app/glue_job.py` foi implementado para gravar o Parquet localmente e usar `boto3.upload_file` para enviar ao S3 do LocalStack, simulando um fluxo de plataforma de dados real.

## Análise Exploratória

* Implementado o script `scripts/exploratory_analysis.py` para geração de gráficos e estatísticas descritivas.
* Utilizados pandas e seaborn/matplotlib para inspecionar os arquivos Parquet processados.

#### Comandos

```bash
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
python scripts/exploratory_analysis.py

```

## Fase de Modelagem Preditiva

### Novo Script

* Adicionado o script `scripts/spark_predictive_model.py` para treinar modelos de regressão com Spark MLlib.
* O script realiza as seguintes etapas:
* carrega os dados de treino a partir de `files/training-set`
* converte quaisquer arquivos `.TXT` para CSV quando necessário
* pré-processa os dados com Spark
* constrói vetores de características (features) a partir de `open`, `high`, `low`, `volume`, `prev_close`
* treina modelos de Regressão Linear, Random Forest e GBT (Gradient-Boosted Trees)
* valida em uma divisão de dados separada (hold-out split)
* avalia o melhor modelo no arquivo `files/test-set/COTAHIST_A2020.TXT`
* armazena os resultados em `output/model`



### Inspeção de Dados

* Verificado se o conteúdo do diretório de treino inclui tanto arquivos CSV quanto `.TXT`.
* Confirmada a existência do CSV de treino existente `files/training-set/cotahist_m072025.csv`.
* Confirmada a existência do arquivo de teste de origem `files/test-set/COTAHIST_A2020.TXT`.
* Inspecionadas as linhas da amostra de treino e confirmadas as colunas de dados.

#### Comandos

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

### Correções no Pré-processamento

* Adicionado filtro robusto de linhas para remover registros com características nulas (NaN) antes da montagem de features do Spark.
* Configurado o `VectorAssembler` com `handleInvalid="skip"`.
* Adicionada validação de conjunto de dados de teste para lançar um erro caso nenhuma linha válida reste após a filtragem.

### Resultados da Execução do Modelo

* A execução do modelo foi concluída com sucesso nos testes locais.
* Símbolo (ticker) selecionado para modelagem: `PETR4T`.
* Quantidade de treino/validação: `1440` linhas de treino, `298` linhas de validação.
* Métricas de validação:
* `LinearRegression`: RMSE ~ 1.72e9, R2 ~ 0.162
* `RandomForest`: RMSE ~ 1.56e9, R2 ~ 0.311
* `GBT`: RMSE ~ 1.25e9, R2 ~ 0.554


* Melhor modelo: `gbt`
* Métricas de teste no arquivo `COTAHIST_A2020.TXT`:
* RMSE ~ 4.86e9
* MAE ~ 3.81e9
* R2 ~ -1.241



#### Comandos

```bash
cd /c/Users/andre/Projects/desafio-trilha-python-advanced-n1
.venv/Scripts/python -u scripts/spark_predictive_model.py --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model-test

# Inspecionar arquivos de saída gerados
python - <<'PY'
from pathlib import Path
print(list(Path('output/model-test').glob('*')))
PY

```

### Avisos e Observações

* Problemas do Spark no Windows com a ausência do `winutils.exe` e falha no carregamento da biblioteca nativa do Hadoop foram observados, mas não impediram a execução.
* O ticker selecionado sofreu com desvio de generalização (drift) no conjunto de teste, evidenciado pelo R2 negativo.
* O conjunto de dados contém uma grande quantidade de tickers, e o modelo atualmente utiliza apenas o mais frequente por padrão.

## Atualizações de Documentação

* Atualizado o `README.md` para incluir instruções de execução do novo script de modelo preditivo.

## Arquivos Atuais Adicionados/Modificados

* Adicionado: `scripts/spark_predictive_model.py`
* Adicionado: `TECHNICAL-TECH-DOC.md`
* Modificado: `README.md`
* Arquivos existentes verificados: `scripts/convert_cotahist_to_csv.py`, `app/glue_pipeline.py`, `app/glue_job.py`, `scripts/setup_localstack.py`, `scripts/exploratory_analysis.py`

## Como Executar de Ponta a Ponta (End-to-End)

1. Ative o ambiente virtual Python.
2. Obtenha um CSV de preços brutos: busque um ticker via `yfinance.download` (através de `app/app.py`, que o salva em `files/from-input/{ticker}.csv`) ou faça o upload de um CSV.
3. Execute o job de ETL do PySpark: `python -m app.glue_job --mode price-series --input files/from-input/AAPL.csv --source-name AAPL`.
Isso irá gravar `files/from-file/AAPL.csv` (`date;close`), `output/processed_stock_data/AAPL.parquet` e fazer o upload do arquivo Parquet para o bucket S3 do LocalStack chamado `processed-data`.
4. Execute o modelo de previsão: `python -m scripts.spark_predictive_model --mode forecast --forecast-input files/from-file/AAPL.csv --source-name AAPL`.
5. Inspecione os resultados em `output/analysis/` (busca/métricas de treino) e `output/model-test/` (métricas e previsões futuras/de teste).
6. (Opcional) execute o fluxo legado multi-símbolo do COTAHIST:
`python -m scripts.spark_predictive_model --mode training --training-dir files/training-set --test-file files/test-set/COTAHIST_A2020.TXT --output-dir output/model`
e inspecione `output/model/training_results.csv` / `output/model/test_results.txt`.

## Comandos de Automação & Monitoramento

Os scripts de automação podem ser utilizados contra a AWS ou LocalStack (forneça `--endpoint-url` para apontar para o LocalStack).

```bash
# Criar um Glue job (exemplo)
python scripts/glue_automation.py --create-job --job-name glue-etl-job --script-location s3://my-bucket/scripts/glue_job.py

# Criar um gatilho agendado (scheduled trigger)
python scripts/glue_automation.py --create-trigger --trigger-name daily-trigger --job-name glue-etl-job --cron 'cron(0 2 * * ? *)'

# Iniciar a execução de um job imediatamente
python scripts/glue_automation.py --start-job --job-name glue-etl-job

# Monitorar execuções de jobs
python scripts/monitor_glue_jobs.py --job-name glue-etl-job --interval 30

```

## Comando para Gráfico Comparativo

```bash
python scripts/comparative_series.py --symbol PETR4T --input-dir files/training-set --output-dir output/plots

```

## Refatoração Ticker/CSV -> Spark -> LocalStack (Correção de `st.form` do Streamlit)

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

Uma nova transformação PySpark-SQL normaliza qualquer CSV bruto de preço (download de ticker ou upload) para apenas `date`/`close`, replicando as regras de negócio que antes ficavam no pandas (`scripts/data_processing.py::finalize_price_dataframe`): detecta colunas de data/fechamento/símbolo sem distinção de maiúsculas/minúsculas, exige ticker único quando a coluna de símbolo/ticker existe, aceita apenas datas completas (ano+mês+dia) em formatos suportados e preços com vírgula decimal via `try_to_date`/`try_cast` (necessário pois o modo ANSI do Spark 4.x lança exceção em vez de retornar nulo em `to_date`), remove duplicatas por data mantendo a última ocorrência e limita a série temporal aos últimos 365 dias. Coberto por `tests/test_glue_pipeline.py`.

### `app/glue_job.py --mode price-series`

O novo modo de CLI encapsula o `transform_price_series`: lê o CSV bruto (separador auto-detectado), grava o CSV tratado `date;close` em `files/from-file/{slug}.csv`, grava uma cópia em Parquet em `output/processed_stock_data/{slug}.parquet` e faz o upload do arquivo Parquet para um bucket S3 do LocalStack (criado automaticamente caso não exista). Imprime uma linha de resumo em JSON para os chamadores de subprocesso. O comportamento legado `--mode stock` (OHLCV multi-símbolo) foi preservado sem alterações.

### Fusão de `scripts/ml_model.py` em `scripts/spark_predictive_model.py`

O arquivo `scripts/ml_model.py` foi mesclado ao `scripts/spark_predictive_model.py` e removido. O arquivo unificado agora expõe ambos os fluxos:

* `--mode forecast --forecast-input <csv>`: o modelo de previsão de data/fechamento de símbolo único (antigo conteúdo do `ml_model.py`: `ForecastResult`, `ModelTrainingError`, `train_predict_evaluate`, busca de hiperparâmetros por faixa de R2, persistência de artefatos). A persistência foi estendida de modo que o `future_predictions.csv` também é gravado em `output/model-test/` e a `past_predictions` cobre todo o intervalo real da série de entrada — necessário pois o `app.py` realiza a leitura de volta do disco após a finalização do subprocesso, sem a existência de um objeto `ForecastResult` em memória compartilhado entre os limites do processo.
* `--mode training --training-dir ... --test-file ...`: o fluxo original de comparação multi-símbolo do COTAHIST (LinearRegression/RandomForest/GBT), sem alterações.

### `scripts/localstack_pipeline_test.py`

Reescrito para testar o novo fluxo de séries de preços de ponta a ponta contra um container do LocalStack ativo: garante a presença dos recursos do LocalStack, auto-detecta um CSV em `files/from-input/` (ou sintetiza uma amostra determinística para que o teste nunca dependa de acesso à rede), executa o `app.glue_job --mode price-series`, verifica se o objeto Parquet resultante existe no S3 via `boto3`, executa o `scripts.spark_predictive_model --mode forecast` contra o CSV tratado e verifica se todos os artefatos de previsão existem em disco. Grava um relatório em `output/localstack_test_results.txt`.

### Limpeza do Repositório

* Removido `scripts/ml_model.py` (mesclado em `scripts/spark_predictive_model.py`).
* Removido do rastreamento do git o estado de execução do LocalStack que havia sido commitado acidentalmente (`docker/volume/**` — certificados, licença, ID da máquina), binários Parquet desatualizados (`output/processed_stock_data/*.parquet`) e bytecode compilado (`scripts/__pycache__/*.pyc`); adicionados `docker/volume/`, `output/`, `*.parquet`, `*.pyc` ao `.gitignore`.
* Extraídos utilitários compartilhados de CSV (`detect_csv_separator*`, `slugify`) que estavam duplicados em `scripts/data_processing.py`, `scripts/comparative_series.py` e `scripts/spark_predictive_model.py` para um novo módulo `scripts/csv_utils.py`.