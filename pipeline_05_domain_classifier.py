from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
from sentence_transformers import SentenceTransformer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np
import re

KEEP_PREFIX = False # remove prefix like "speaker:" in the dataset
# Function to extract only real conversation turns
def extract_conversations(example):
    text = example['text']
    # Split by both '|' and '\t'
    turns = []
    for part in text.replace('\t', '|').split('|'):
        part = part.strip()
        # Remove any persona lines
        if 'persona' in part.lower():
            continue
        if part:  # skip empty
            turns.append(part)
    return {'turns': turns}

# Function to split dialogue into turns in domain dataset
def split_turns(dialogue, keep_prefix=KEEP_PREFIX):
    # Split on "Doctor:" or "Patient:"
    turns = re.split(r'(Doctor:|Patient:)', dialogue)

    clean_turns = []
    for i in range(1, len(turns), 2):
        speaker = turns[i].strip()
        text = turns[i+1].strip()
        if keep_prefix:
            clean_turns.append(f"{speaker}: {text}")
        else:
            clean_turns.append(text)
    return clean_turns

def non_domain_dataset_prepear():
    # Load dataset
    dataset = load_dataset("awsaf49/persona-chat", split="train")
    # Apply to dataset
    clean_dataset = dataset.map(extract_conversations, batched=False)

    # Flatten all turns into a single list
    all_messages = [turn for ex in clean_dataset for turn in ex['turns']]

    print("Sample cleaned messages:", all_messages[:10])
    print("Total cleaned messages:", len(all_messages))
    non_domain_messages = all_messages[:3088]
    return non_domain_messages

def domain_dataset_prepear():
    file_path = "data/data_csv/MTS-Dialog-Automatic-Summaries-ValidationSet.csv"
    try:
        df = pd.read_csv(file_path)
        print("File loaded successfully!")
        print(df.columns)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please double-check the path and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        # Option: keep speaker labels or not
        KEEP_PREFIX = False  # True = keep "Doctor:" / "Patient:", False = remove
        # Apply to all dialogues
        df['turns'] = df['Dialogue'].apply(split_turns)

        # Flatten all turns into a single list
        domain_messages = [turn for conv in df['turns'] for turn in conv]

        print("Sample domain messages:", domain_messages[:10])
        print("Total domain messages:", len(domain_messages))

        # Optional: save to CSV
        # pd.DataFrame({"message": domain_messages}).to_csv("domain_messages_flat.csv", index=False)
        return domain_messages
def regression_model_creation():
    domain_messages = domain_dataset_prepear()
    non_domain_messages = non_domain_dataset_prepear()
    # Assign labels: 1 = domain-specific, 0 = non-domain
    domain_labels = [1] * len(domain_messages)
    non_domain_labels = [0] * len(non_domain_messages)

    # Combine messages and labels|
    all_messages = domain_messages + non_domain_messages
    all_labels = domain_labels + non_domain_labels

    # Split into train and test (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
    all_messages, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    # --- Start Spark ---
    spark = SparkSession.builder.appName("DomainClassifier").getOrCreate()

    # --- Assume you already have these lists from your train/test split ---
    # X_train, y_train, X_test, y_test

    # --- Encode sentences ---
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    train_emb = embedder.encode(X_train)
    test_emb = embedder.encode(X_test)

    # --- Prepare Spark DataFrames ---
    train_rows = [Row(features=Vectors.dense(vec), label=float(label))
                for vec, label in zip(train_emb, y_train)]
    train_df = spark.createDataFrame(train_rows)

    test_rows = [Row(features=Vectors.dense(vec), label=float(label))
                for vec, label in zip(test_emb, y_test)]
    test_df = spark.createDataFrame(test_rows)

    # --- Train Logistic Regression ---
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
    lr_model = lr.fit(train_df)
    lr_model.save("data/output/lr_model")
    print("Model saved to data/output/lr_model with :- ")
    # --- Predict on test set ---
    pred = lr_model.transform(test_df)
    # --- Evaluation ---
    # Accuracy
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = acc_eval.evaluate(pred)

    # Precision
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    precision = precision_eval.evaluate(pred)

    # Recall
    recall_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    recall = recall_eval.evaluate(pred)

    # F1 Score
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = f1_eval.evaluate(pred)

    # Confusion Matrix
    pred_rdd = pred.select("prediction", "label").rdd.map(tuple)
    metrics = MulticlassMetrics(pred_rdd)
    confusion_matrix = np.array(metrics.confusionMatrix().toArray())

    # --- Print Results ---
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_matrix)


regression_model_creation()