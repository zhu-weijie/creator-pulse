import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.logging import logger


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion paths."""

    train_data_path: str = os.path.join("data", "raw", "train.csv")
    test_data_path: str = os.path.join("data", "raw", "test.csv")
    raw_data_path: str = os.path.join("data", "raw", "data.csv")


class DataIngestion:
    """Handles downloading and splitting the dataset."""

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.dataset_url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"

    def initiate_data_ingestion(self):
        """Main method to perform data ingestion."""
        logger.info("Entered the data ingestion method or component")

        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

        try:
            logger.info("Downloading dataset...")
            df = pd.read_csv(self.dataset_url, on_bad_lines="skip")
            logger.info("Dataset downloaded successfully")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            logger.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logger.info("Ingestion of the data is completed")
            logger.info(f"Train data saved at: {self.ingestion_config.train_data_path}")
            logger.info(f"Test data saved at: {self.ingestion_config.test_data_path}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logger.error(f"An error occurred during data ingestion: {e}")
            raise e


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
