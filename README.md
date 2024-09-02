# Twitter Sentiment Analysis with Spark and Kafka

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Apache Kafka](#apache-kafka)
  - [Apache Spark](#apache-spark)
- [Spark Pipeline](#spark-pipeline)
- [Machine Learning Models](#machine-learning-models)

## Overview

This project demonstrates a real-time sentiment analysis pipeline using Apache Kafka and Apache Spark. The system streams Twitter data from an API into Kafka, processes it with Spark Streaming, and applies various machine learning models to analyze sentiment.

## Architecture

### Apache Kafka

Apache Kafka is used for real-time data streaming and messaging. In this project, Kafka acts as a message broker that streams Twitter data from the producer script to the Spark Streaming application. Kafka provides a reliable and scalable way to handle large volumes of data in real-time, ensuring that the data pipeline can process tweets continuously.

### Apache Spark

Apache Spark is utilized for processing the streaming data. Spark Streaming enables real-time data processing by consuming data from Kafka, performing transformations, and applying machine learning models. Spark's distributed computing capabilities allow it to handle large datasets efficiently and perform complex processing tasks in parallel.

## Spark Pipeline

The Spark pipeline is a sequence of stages used to prepare and transform the data before applying machine learning models. Here's a breakdown of the pipeline components used in this project:

1. **Tokenizer**: Splits the tweet text into individual words (tokens). This step is crucial for converting raw text into a format that can be used for feature extraction.

2. **StopWordsRemover**: Removes common words (e.g., "the", "and") that do not contribute to the sentiment analysis. This step helps to focus on meaningful words and reduce noise in the data.

3. **HashingTF**: Converts the filtered words into a numerical feature vector using the term frequency approach. This transformation represents the presence of words in a fixed-size feature vector.

4. **IDF (Inverse Document Frequency)**: Weighs the term frequency features by their importance across the dataset. IDF helps to highlight terms that are unique and significant, improving the model's ability to differentiate between sentiments.

5. **StringIndexer**: Converts categorical labels (e.g., sentiment classes) into numerical indices. This is necessary for machine learning algorithms that require numerical input.

This pipeline processes the tweets and converts them into feature vectors suitable for the machine learning models.

## Machine Learning Models

1. **Logistic Regression**: This is a binary classification algorithm that estimates the probability of a tweet belonging to a particular sentiment class. It is effective for linearly separable data and is used here to classify tweets into positive or negative sentiments based on their features.

2. **Random Forest**: An ensemble learning method that combines multiple decision trees to improve classification accuracy. By averaging the results of individual trees, Random Forest provides robust and reliable predictions and is less prone to overfitting compared to a single decision tree.

3. **GBTClassifier (Gradient Boosted Trees)**: This boosting method builds models sequentially, with each new model correcting the errors made by the previous ones. GBTClassifier focuses on difficult-to-classify examples, leading to enhanced performance on complex datasets where traditional methods might struggle.

4. **Neural Network (Multilayer Perceptron)**: A deep learning model consisting of multiple layers that can capture complex patterns and non-linear relationships in the data. The Multilayer Perceptron is used here to leverage its capability to model intricate dependencies and achieve high accuracy in sentiment classification.
