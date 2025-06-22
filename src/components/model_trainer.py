import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.logging import logger


class ModelTrainer:
    def __init__(self):
        load_dotenv()
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.processed_train_path = os.path.join(
            "data", "processed", "train_processed.csv"
        )
        self.processed_test_path = os.path.join(
            "data", "processed", "test_processed.csv"
        )

    def initiate_model_training(self):
        logger.info("Starting model training process.")

        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.mlflow_tracking_uri}")

            train_df = pd.read_csv(self.processed_train_path)
            test_df = pd.read_csv(self.processed_test_path)
            logger.info("Loaded preprocessed train and test data.")

            train_df["processed_text"] = train_df["processed_text"].fillna("")
            test_df["processed_text"] = test_df["processed_text"].fillna("")

            X_train = train_df["processed_text"]
            y_train = train_df["category"]
            X_test = test_df["processed_text"]
            y_test = test_df["category"]

            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            logger.info("Text data vectorized using TF-IDF.")

            with mlflow.start_run(run_name="Baseline_LogisticRegression"):
                logger.info("MLflow run started.")

                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("vectorizer", "TfidfVectorizer")
                mlflow.log_param("max_features", 5000)

                model = LogisticRegression(random_state=42)
                model.fit(X_train_tfidf, y_train)
                logger.info("Model training complete.")

                y_pred = model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"Baseline model accuracy: {accuracy}")

                mlflow.log_metric("accuracy", accuracy)

                input_schema = Schema(
                    [
                        TensorSpec(
                            type=np.dtype(np.float32), shape=(-1, X_train_tfidf.shape)
                        )
                    ]
                )
                output_schema = Schema(
                    [TensorSpec(type=np.dtype(np.int64), shape=(-1,))]
                )
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                input_example = X_train_tfidf[:5]

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )
                logger.info("Model logged to MLflow with signature.")

                print(
                    f"Run complete. Check the MLflow UI at: {self.mlflow_tracking_uri}"
                )

        except Exception as e:
            logger.error(f"An error occurred during model training: {e}")
            raise


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()
