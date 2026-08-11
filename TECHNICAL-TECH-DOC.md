Aqui está a tradução técnica do documento `.md` para o português brasileiro, mantendo toda a formatação original, blocos de código, comandos e terminologias padrões da área de engenharia de dados (como *pipeline*, *workflow*, *deploy*, etc.):

---

# Documento técnico do projeto

## Visão geral

Este repositório reúne um fluxo local de ETL e modelagem preditiva para séries temporais de ações. O objetivo atual é transformar dados brutos em uma série tratada de date/close, treinar um modelo de regressão com PySpark e gerar previsões de tendência para o fechamento futuro.

O fluxo foi refinado para: escrever o ETL com Spark nativo no caminho principal, controlar o horizonte futuro com `future_days`, comparar o modelo com um baseline ingênuo de persistência e registrar backtesting temporal em folds sequenciais. No Windows, o ETL possui fallback automático para evitar falhas de gravação ligadas ao `winutils`.

## Arquitetura atual

A aplicação Streamlit em app/app.py não executa PySpark diretamente. Em vez disso, ela chama subprocessos separados para:

1. tratar a entrada via ETL em app/glue_job.py;
2. treinar e avaliar o modelo em scripts/spark_predictive_model.py.

Essa separação evita conflitos com a JVM do Spark e mantém a interface responsiva.

O app também lê de volta do disco os artefatos produzidos pelos subprocessos, incluindo métricas de treino, previsões de teste, projeções futuras e sumários de baseline/backtesting.

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

Na implementação atual, a persistência é feita com Spark nativo no caminho principal. Quando o ambiente local do Windows não fornece suporte de Hadoop compatível, o job cai para um fallback com pandas apenas para concluir a gravação dos artefatos.

### 3. Modelo de machine learning

O script scripts/spark_predictive_model.py implementa um fluxo de previsão para uma única ação. Ele:

- carrega a série tratada;
- realiza uma divisão temporal, sem shuffle;
- cria features baseadas em tempo (t, t², log(t+1));
- treina um LinearRegression do PySpark MLlib em escala log (log(close+1));
- gera previsões para o período de teste e para um horizonte futuro definido por `future_days`;
- registra baseline ingênuo de persistência e backtesting temporal em folds sequenciais.

## Natureza do problema

O projeto não faz classificação binária de tendência (subiu/baixou). O objetivo atual é prever o valor de fechamento futuro e, a partir disso, interpretar a tendência visualmente.

Portanto, as métricas relevantes são métricas de regressão:

- train_r2
- test_r2
- rmse
- mae
- target_reached

Métricas complementares também são exportadas para contextualizar a previsão:

- baseline_naive_rmse, baseline_naive_mae, baseline_naive_r2
- model_beats_naive_rmse
- backtest_folds, backtest_summary

## Métricas atuais do modelo

O modelo exporta os resultados em arquivos JSON em output/analysis/ e output/model-test/.

A métrica principal de qualidade é o R², mas o projeto também reporta RMSE e MAE para dar contexto ao erro absoluto e ao erro quadrático.

A faixa alvo atualmente usada é:

- R² mínimo: 0.60
- R² máximo: 0.90

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

O ETL de stock/COTAHIST legado foi mantido para compatibilidade com os testes e usa o conversor scripts/convert_cotahist_to_csv.py, que reconstitui CSVs de séries históricas a partir do layout fixo do arquivo original.

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
python -m scripts.spark_predictive_model --mode forecast --forecast-input files/from-file/AAPL.csv --source-name AAPL --future-days 30
```

### Validação E2E LocalStack

```bash
python scripts/localstack_pipeline_test.py --endpoint-url http://localhost:4566
```

Esse teste executa o setup do LocalStack, o ETL, a verificação do upload no S3 e o forecast final, gravando um relatório em output/localstack_test_results.txt.

## Pontos de atenção

- O código atual usa uma abordagem de regressão para prever o fechamento, e não um modelo de classificação.
- A extrapolação futura é mais sensível do que a previsão dentro do intervalo de treino.
- As métricas devem ser interpretadas como um sinal de desempenho do modelo, e não como uma garantia de precisão financeira.
- A execução com dados reais pode ter desempenho ruim se a série for curta, ruidosa ou não estacionária.
- O upload para o LocalStack depende de um container ativo em localhost:4566.
- O compose foi fixado em `localstack/localstack:3.5.0` para evitar o shutdown por licença Pro na imagem mais recente.

## Próximos passos possíveis

- testar outros algoritmos de regressão ou modelos baseados em séries temporais;
- incluir validação adicional por janela temporal;
- acrescentar métricas de direção de tendência, como acurácia de sinal, sem perder o foco principal de regressão do fechamento.
- reduzir a dependência de fallback no Windows, se o ambiente local do Spark/Hadoop for estabilizado.

## Problema e solução de arquitetura

O aplicativo Streamlit apresentava o erro There are multiple identical forms with key='data_input' quando o PySpark era executado de forma síncrona dentro do processo da interface. A solução adotada foi isolar o trabalho do Spark em subprocessos independentes, invocados a partir de app/app.py via subprocess.run. Cada subprocesso imprime um resumo JSON final em stdout, que o app lê para localizar os artefatos de saída.

### Fluxos suportados

- python -m app.glue_job --mode price-series --input <raw.csv> --source-name <slug>
- python -m scripts.spark_predictive_model --mode forecast --forecast-input <treated.csv> --source-name <slug>
- python scripts/localstack_pipeline_test.py --endpoint-url http://localhost:4566

## Observações adicionais

- O histórico do ticker é buscado com yfinance.download.
- O fluxo de previsão persiste artefatos de treino, previsões de teste e previsões futuras em output/analysis/ e output/model-test/.
- O cache de treinamento foi atualizado para manter até 7 registros recentes por ticker, com rotação automática das entradas mais antigas ao registrar um novo treino para o mesmo ticker.
- O módulo scripts/csv_utils.py concentra utilidades compartilhadas para detecção de separador e geração de slug.
- O arquivo docker/docker-compose.yml usa a imagem comunitária `localstack/localstack:3.5.0`.