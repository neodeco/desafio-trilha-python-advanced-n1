import argparse
import csv
import importlib.util
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def detect_csv_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        return dialect.delimiter
    except Exception:
        return ";" if sample.count(";") > sample.count(",") else ","


def load_converter_module() -> object:
    converter_path = Path(__file__).resolve().parents[0] / "convert_cotahist_to_csv.py"
    spec = importlib.util.spec_from_file_location("convert_cotahist", converter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_csv_for_txt(input_path: Path, converter_module: object) -> Path:
    if input_path.suffix.lower() == ".csv":
        return input_path

    output_path = input_path.with_suffix(".csv")
    if not output_path.exists():
        converter_module.convert_to_csv(input_path, output_path)
    return output_path


def load_dataset(spark: SparkSession, folder: Path, converter_module: object, sep: str):
    dataframes = []
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() in {".csv", ".txt"}:
            csv_path = ensure_csv_for_txt(file_path, converter_module)
            file_sep = detect_csv_separator(csv_path) if sep == "auto" else sep
            df = (
                spark.read.option("header", True)
                .option("inferSchema", True)
                .option("sep", file_sep)
                .csv(str(csv_path))
            )
            dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError(f"No training files found in {folder}")

    result = dataframes[0]
    for df in dataframes[1:]:
        result = result.unionByName(df, allowMissingColumns=True)
    return result


def preprocess(df):
    df = (
        df.withColumn(
            "trade_date_norm",
            F.when(
                F.col("trade_date").rlike(r"^\d{4}-\d{2}-\d{2}$"),
                F.regexp_replace(F.col("trade_date"), "-", ""),
            )
            .when(F.col("trade_date").rlike(r"^\d{6}$"), F.concat(F.col("trade_date"), F.lit("01")))
            .otherwise(F.col("trade_date"))
        )
        .withColumn("trade_date_fmt", F.to_date(F.col("trade_date_norm"), "yyyyMMdd"))
        .withColumn("open", F.col("open").cast("double"))
        .withColumn("high", F.col("high").cast("double"))
        .withColumn("low", F.col("low").cast("double"))
        .withColumn("close", F.col("close").cast("double"))
        .withColumn("volume", F.col("volume").cast("double"))
        .withColumn("symbol", F.col("symbol"))
        .withColumn("prev_close", F.lag("close").over(Window.partitionBy("symbol").orderBy("trade_date_fmt")))
        .withColumn("next_close", F.lead("close").over(Window.partitionBy("symbol").orderBy("trade_date_fmt")))
    )

    return df.filter(F.col("next_close").isNotNull()).orderBy("symbol", "trade_date_fmt")


def build_features(df, symbol_filter=None):
    if symbol_filter:
        df = df.filter(F.col("symbol") == symbol_filter)

    feature_cols = ["open", "high", "low", "volume", "prev_close"]
    required_cols = feature_cols + ["next_close"]
    df = df.dropna(subset=required_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    featured = assembler.transform(df)
    return featured.filter(F.col("features").isNotNull()).select(
        "features",
        F.col("next_close").alias("label"),
        "symbol",
        "trade_date_fmt",
    )


def train_and_evaluate(training_df, validation_df, output_dir: Path):
    models = {
        "linear_regression": LinearRegression(featuresCol="features", labelCol="label", maxIter=50),
        "random_forest": RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=20),
        "gbt": GBTRegressor(featuresCol="features", labelCol="label", maxIter=20),
    }

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    results = []
    best_model = None
    best_rmse = float("inf")
    best_name = None

    for name, estimator in models.items():
        model = estimator.fit(training_df)
        predictions = model.transform(validation_df)
        rmse = evaluator.evaluate(predictions)
        mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
        r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

        results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "training_results.csv"
    with results_path.open("w", encoding="utf-8") as f:
        f.write("model,rmse,mae,r2\n")
        for row in results:
            f.write(f"{row['model']},{row['rmse']},{row['mae']},{row['r2']}\n")

    print(f"Best model: {best_name} with RMSE={best_rmse:.4f}")
    return best_name, best_model


def final_test(best_model, test_df, output_dir: Path):
    predictions = best_model.transform(test_df)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)

    with (output_dir / "test_results.txt").open("w", encoding="utf-8") as f:
        f.write(f"rmse,{rmse}\n")
        f.write(f"mae,{mae}\n")
        f.write(f"r2,{r2}\n")

    print(f"Test performance: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Spark ML regression models for next-close prediction")
    parser.add_argument("--training-dir", default="files/training-set", help="Folder containing training files")
    parser.add_argument("--test-file", default="files/test-set/COTAHIST_A2020.TXT", help="Test file to evaluate model")
    parser.add_argument("--sep", default=";", help="CSV separator for input files (';' by default, or 'auto')")
    parser.add_argument("--symbol", default=None, help="Stock symbol to model (default: largest symbol in training set)")
    parser.add_argument("--output-dir", default="output/model", help="Directory for model artifacts and results")
    args = parser.parse_args()

    spark = SparkSession.builder.master("local[1]").appName("spark-predictive-model").getOrCreate()
    converter_module = load_converter_module()

    training_df = load_dataset(spark, Path(args.training_dir), converter_module, args.sep)
    training_df = preprocess(training_df)

    if args.symbol is None:
        args.symbol = training_df.groupBy("symbol").count().orderBy(F.desc("count")).first()["symbol"]
        print(f"Selected symbol for modeling: {args.symbol}")

    data = build_features(training_df, symbol_filter=args.symbol)
    train_set, val_set = data.randomSplit([0.8, 0.2], seed=42)
    print(f"Training rows: {train_set.count()}, validation rows: {val_set.count()}")

    _, best_model = train_and_evaluate(train_set, val_set, Path(args.output_dir))

    test_path = Path(args.test_file)
    test_csv_path = ensure_csv_for_txt(test_path, converter_module)
    test_sep = detect_csv_separator(test_csv_path) if args.sep == "auto" else args.sep
    test_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("sep", test_sep)
        .csv(str(test_csv_path))
    )
    test_df = preprocess(test_df)
    test_features = build_features(test_df, symbol_filter=args.symbol)
    if test_features.count() == 0:
        raise RuntimeError(f"No valid test rows found for symbol {args.symbol} after preprocessing")
    final_test(best_model, test_features, Path(args.output_dir))

    spark.stop()


if __name__ == "__main__":
    main()