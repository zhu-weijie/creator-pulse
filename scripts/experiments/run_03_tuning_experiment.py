import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import lightgbm as lgb
from dotenv import load_dotenv
import mlflow
import optuna

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)
from src.utils.logging import logger

PROCESSED_TRAIN_PATH = os.path.join("data", "processed", "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join("data", "processed", "test_processed.csv")
MLFLOW_EXPERIMENT_NAME = "Model Selection and Tuning"
OPTUNA_TRIALS = 10

train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
test_df = pd.read_csv(PROCESSED_TEST_PATH)
train_df["processed_text"] = train_df["processed_text"].fillna("")
test_df["processed_text"] = test_df["processed_text"].fillna("")
X_train_raw, y_train = train_df["processed_text"], train_df["category"]
X_test_raw, y_test = test_df["processed_text"], test_df["category"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_train_tfidf = vectorizer.fit_transform(X_train_raw)
X_test_tfidf = vectorizer.transform(X_test_raw)


def objective(trial):
    """The Optuna objective function to be minimized/maximized."""
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "verbose": -1,
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("f1_score", f1)

    return f1


def run_experiment():
    """Runs model selection and tuning experiment."""
    logger.info("Starting Model Selection and Tuning experiment.")
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Baseline_LogisticRegression"):
        model_lr = LogisticRegression(random_state=42)
        model_lr.fit(X_train_tfidf, y_train)
        y_pred_lr = model_lr.predict(X_test_tfidf)
        f1_lr = f1_score(y_test, y_pred_lr, average="weighted")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("f1_score", f1_lr)
        logger.info(f"Baseline Logistic Regression F1 Score: {f1_lr}")

    with mlflow.start_run(run_name="LGBM_Optuna_Tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_TRIALS)

        logger.info(f"Optuna study complete. Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")

        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_f1_score", study.best_trial.value)
        mlflow.set_tag("best_trial_number", study.best_trial.number)


if __name__ == "__main__":
    run_experiment()
