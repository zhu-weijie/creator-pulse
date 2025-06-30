import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import pickle


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.logging import logger


MLFLOW_EXPERIMENT_NAME = "Production_Training_Pipeline"
BEST_PARAMS = {
    "n_estimators": 951,
    "learning_rate": 0.13186508156408952,
    "num_leaves": 156,
    "max_depth": 5,
    "min_child_samples": 12,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "random_state": 42,
    "verbose": -1,
}
VECTORIZER_PATH = "artifacts/tfidf_vectorizer.pkl"
MODEL_PATH = "artifacts/lgbm_model.pkl"


def run_training():
    """Runs the production model training and logging pipeline."""
    logger.info("--- Starting Production Training Pipeline ---")

    os.makedirs("artifacts", exist_ok=True)

    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df = pd.read_csv("data/processed/train_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")

    train_df["processed_text"] = train_df["processed_text"].fillna("")
    test_df["processed_text"] = test_df["processed_text"].fillna("")

    X_train, y_train = train_df["processed_text"], train_df["category"]
    X_test, y_test = test_df["processed_text"], test_df["category"]
    logger.info("Loaded preprocessed data.")

    with mlflow.start_run(run_name="Production_LGBM_Training"):
        mlflow.log_params(BEST_PARAMS)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logger.info("Data vectorized.")

        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        mlflow.log_artifact(VECTORIZER_PATH)
        logger.info("Vectorizer saved and logged to MLflow.")

        model = lgb.LGBMClassifier(**BEST_PARAMS)
        model.fit(X_train_tfidf, y_train)
        logger.info("Model trained.")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(MODEL_PATH)
        logger.info("Model saved and logged to MLflow.")

        logger.info("Evaluating model on the test set.")
        y_pred = model.predict(X_test_tfidf)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, average='weighted'),
            "test_recall": recall_score(y_test, y_pred, average='weighted'),
            "test_f1_score": f1_score(y_test, y_pred, average='weighted')
        }

        mlflow.log_metrics(metrics)
        logger.info(f"Model evaluation metrics logged: {metrics}")

    logger.info("--- Production Training Pipeline Finished ---")


if __name__ == "__main__":
    run_training()
