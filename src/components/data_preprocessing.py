import os
import sys
import re
import pandas as pd
from dataclasses import dataclass
from nltk.stem import WordNetLemmatizer
import nltk

from src.utils.preprocessing import preprocess_text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.logging import logger

logger.info("Ensuring NLTK data (wordnet, punkt, punkt_tab) is up-to-date...")
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
logger.info("NLTK check complete.")


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""

    processed_train_path: str = os.path.join("data", "processed", "train_processed.csv")
    processed_test_path: str = os.path.join("data", "processed", "test_processed.csv")


class DataTransformation:
    """Handles text preprocessing and transformation."""

    def __init__(self):
        self.config = DataTransformationConfig()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """Applies a series of cleaning steps to the text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = " ".join(text.split())
        return text

    def lemmatize_text(self, text):
        """Lemmatizes the text."""
        words = nltk.word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)

    def initiate_data_transformation(self, train_path, test_path):
        """Loads raw data, applies transformations, and saves the processed data."""
        logger.info("Starting data transformation.")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Loaded raw train and test data.")
        except FileNotFoundError as e:
            logger.error(f"Data not found: {e}. Please run data ingestion first.")
            raise

        text_column = "clean_comment"

        logger.info("Applying shared preprocessing function...")
        train_df["processed_text"] = train_df[text_column].apply(preprocess_text)
        test_df["processed_text"] = test_df[text_column].apply(preprocess_text)

        os.makedirs(os.path.dirname(self.config.processed_train_path), exist_ok=True)

        train_df.to_csv(self.config.processed_train_path, index=False)
        test_df.to_csv(self.config.processed_test_path, index=False)

        logger.info(
            f"Processed train data saved to: {self.config.processed_train_path}"
        )
        logger.info(f"Processed test data saved to: {self.config.processed_test_path}")

        return self.config.processed_train_path, self.config.processed_test_path


if __name__ == "__main__":
    raw_train_path = os.path.join("data", "raw", "train.csv")
    raw_test_path = os.path.join("data", "raw", "test.csv")

    transformer = DataTransformation()
    transformer.initiate_data_transformation(raw_train_path, raw_test_path)
