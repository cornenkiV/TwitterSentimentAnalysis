from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, \
    MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.sql import SparkSession
import os


def create_spark_session(app_name="SentimentAnalysisModelTraining"):
    """Create and configure a Spark session."""
    return SparkSession.builder.appName(app_name) \
        .config("spark.sql.shuffle.partitions", 200) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "12g") \
        .config("spark.memory.offHeap.enabled", True) \
        .config("spark.memory.offHeap.size", "32g") \
        .getOrCreate()


def load_data(spark, file_path):
    """Load and preprocess data from a CSV file."""
    columns = ["target", "id", "date", "flag", "user", "text"]
    return spark.read.csv(file_path, header=True, inferSchema=True).toDF(*columns)


def create_pipeline(algorithm):
    """Create a machine learning pipeline."""
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    label_stringIdx = StringIndexer(inputCol="target", outputCol="targetIndex")

    return Pipeline(stages=[tokenizer, remover, hashingTF, idf, label_stringIdx, algorithm])


def save_model(name, model, path="models"):
    """Save the trained model to the specified path."""
    model_path = os.path.join(path, name)
    model.write().overwrite().save(model_path)
    print(f"Model saved to {model_path}")


def evaluate_model(name, model, test_data):
    """Evaluate the model's performance on the test data."""
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                  labelCol="targetIndex",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"{name} model accuracy: {accuracy * 100:.2f}%")


def fit_and_evaluate_model(name, algorithm, train_data, test_data):
    """Fit the model, evaluate it, and save the trained model."""
    pipeline = create_pipeline(algorithm)
    model = pipeline.fit(train_data)
    evaluate_model(name, model, test_data)
    save_model(name, model)


def main():
    spark = create_spark_session()
    data = load_data(spark, "data/training.1600000.processed.noemoticon.csv")

    train_data, test_data = data.randomSplit([0.8, 0.2], seed=12345)

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="targetIndex", maxIter=1000),
        "RandomForestClassifier": RandomForestClassifier(featuresCol="features", labelCol="targetIndex", numTrees=100),
        "GBTClassifier": GBTClassifier(featuresCol="features", labelCol="targetIndex", maxIter=100),
        "NeuralNetwork": MultilayerPerceptronClassifier(featuresCol="features", labelCol="targetIndex",
                                                        layers=[1000, 128, 64, 2])
    }

    for name, algorithm in models.items():
        fit_and_evaluate_model(name, algorithm, train_data, test_data)


if __name__ == "__main__":
    main()
