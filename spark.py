import os
import sys
from typing import List
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import from_json, col, format_number, when
from pyspark.sql.types import StructType, StructField, StringType

# Set up the environment variables for PySpark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


def create_spark_session(app_name: str) -> SparkSession:
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "20") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()


def define_schema() -> StructType:
    return StructType([
        StructField("created_at", StringType()),
        StructField("id_str", StringType()),
        StructField("text", StringType()),
        StructField("user", StructType([
            StructField("id_str", StringType()),
            StructField("screen_name", StringType())
        ]))
    ])


def load_models(model_paths: List[str]) -> List[PipelineModel]:
    return [PipelineModel.load(path) for path in model_paths]


def read_from_kafka(spark: SparkSession, topic: str, servers: str) -> DataFrame:
    return spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", servers) \
        .option("subscribe", topic) \
        .option("startingOffsets", "earliest") \
        .load() \
        .selectExpr("CAST(value AS STRING)")


def parse_json(df: DataFrame, schema: StructType) -> DataFrame:
    return df.withColumn("jsonData", from_json(col("value"), schema)).select("jsonData.*")


def preprocess_tweets(df: DataFrame) -> DataFrame:
    return df.select(
        col("id_str").alias("id"),
        col("user.screen_name").alias("user"),
        col("text")
    ).repartition(10, "id")


def make_predictions(df: DataFrame, model: PipelineModel, neutral_threshold: float, model_name: str) -> DataFrame:
    return model.transform(df) \
        .select("id", "user", "text",
                when(vector_to_array(col("probability")).getItem(1) > neutral_threshold, "positive")
                .when(vector_to_array(col("probability")).getItem(0) > neutral_threshold, "negative")
                .otherwise("neutral").alias(f"prediction_{model_name}"))


def combine_predictions(dfs: List[DataFrame]) -> DataFrame:
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.join(df, ["id", "user", "text"])
    return combined_df.coalesce(5)


def write_to_console(df: DataFrame) -> None:
    df.writeStream \
        .outputMode("append") \
        .format("console") \
        .start() \
        .awaitTermination()


def main():
    spark = create_spark_session("SentimentAnalysisProcessing")

    schema = define_schema()
    model_paths = ["models/GBTClassifier", "models/LogisticRegression", "models/NeuralNetwork",
                   "models/RandomForestClassifier"]
    models = load_models(model_paths)

    kafka_df = read_from_kafka(spark, "tweets", "localhost:9092")
    parsed_df = parse_json(kafka_df, schema)
    tweets_df = preprocess_tweets(parsed_df)

    neutral_threshold = 0.6
    model_names = ["GBT", "LR", "NN", "RF"]
    predictions_dfs = [make_predictions(tweets_df, model, neutral_threshold, name) for model, name in
                       zip(models, model_names)]

    final_df = combine_predictions(predictions_dfs)
    write_to_console(final_df)


if __name__ == "__main__":
    main()
