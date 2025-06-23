import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
from src.utils.logging import logger


PROCESSED_TRAIN_PATH = os.path.join("data", "processed", "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join("data", "processed", "test_processed.csv")
MLFLOW_EXPERIMENT_NAME = "TF-IDF N-gram Tuning"


def evaluate_model(y_true, y_pred):
    """Calculates and returns a dictionary of metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def run_experiment():
    """Runs the n-gram experiment and logs results to MLflow."""
    logger.info("Starting N-gram tuning experiment.")

    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow experiment set to: '{MLFLOW_EXPERIMENT_NAME}'")

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    train_df["processed_text"] = train_df["processed_text"].fillna("")
    test_df["processed_text"] = test_df["processed_text"].fillna("")
    X_train, y_train = train_df["processed_text"], train_df["category"]
    X_test, y_test = test_df["processed_text"], test_df["category"]
    logger.info("Data loaded successfully.")

    ngram_ranges = [(1, 1), (1, 2), (1, 3)]

    for ngram_range in ngram_ranges:
        run_name = f"LR_ngram_{ngram_range[0]}-{ngram_range[1]}"
        with mlflow.start_run(run_name=run_name):
            logger.info(f"--- Starting run: {run_name} ---")

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("ngram_range", str(ngram_range))

            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=ngram_range)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            logger.info(f"Vectorized with ngram_range: {ngram_range}")

            model = LogisticRegression(random_state=42)
            model.fit(X_train_tfidf, y_train)

            y_pred = model.predict(X_test_tfidf)
            metrics = evaluate_model(y_test, y_pred)
            logger.info(f"Metrics for {run_name}: {metrics}")

            mlflow.log_metrics(metrics)

    logger.info("N-gram tuning experiment finished.")
    print(
        f"Experiment complete. Check the '{MLFLOW_EXPERIMENT_NAME}' experiment in the MLflow UI."
    )


if __name__ == "__main__":
    run_experiment()
