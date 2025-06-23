import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn


sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
from src.utils.logging import logger


PROCESSED_TRAIN_PATH = os.path.join("data", "processed", "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join("data", "processed", "test_processed.csv")
MLFLOW_EXPERIMENT_NAME = "Class Imbalance Handling"


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
    """Runs the class imbalance experiment and logs results to MLflow."""
    logger.info("Starting Class Imbalance handling experiment.")

    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow experiment set to: '{MLFLOW_EXPERIMENT_NAME}'")

    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    train_df["processed_text"] = train_df["processed_text"].fillna("")
    test_df["processed_text"] = test_df["processed_text"].fillna("")
    X_train_raw, y_train_raw = train_df["processed_text"], train_df["category"]
    X_test_raw, y_test = test_df["processed_text"], test_df["category"]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    logger.info("Data vectorized.")

    with mlflow.start_run(run_name="Baseline_No_Imbalance_Handling"):
        logger.info("--- Starting run: Baseline (No SMOTE) ---")
        mlflow.log_param("imbalance_technique", "None")

        model = LogisticRegression(random_state=42)
        model.fit(X_train_tfidf, y_train_raw)

        y_pred = model.predict(X_test_tfidf)
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics for Baseline: {metrics}")

    with mlflow.start_run(run_name="With_SMOTE"):
        logger.info("--- Starting run: With SMOTE ---")
        mlflow.log_param("imbalance_technique", "SMOTE")

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_tfidf, y_train_raw
        )
        logger.info(
            f"SMOTE applied. Original shape: {X_train_tfidf.shape}, Resampled shape: {X_train_resampled.shape}"
        )

        model_smote = LogisticRegression(random_state=42)
        model_smote.fit(X_train_resampled, y_train_resampled)

        y_pred_smote = model_smote.predict(X_test_tfidf)
        metrics_smote = evaluate_model(y_test, y_pred_smote)
        mlflow.log_metrics(metrics_smote)
        logger.info(f"Metrics for SMOTE: {metrics_smote}")

    logger.info("Class Imbalance experiment finished.")


if __name__ == "__main__":
    run_experiment()
