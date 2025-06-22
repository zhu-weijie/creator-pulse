import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_DATA_PATH = os.path.join("data", "raw", "train.csv")
REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)


def run_analysis():
    """Loads data and runs exploratory data analysis."""
    print("--- Starting Exploratory Data Analysis ---")

    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"Data loaded successfully from {TRAIN_DATA_PATH}")
    except FileNotFoundError:
        print(
            f"Error: Training data not found at {TRAIN_DATA_PATH}. Please run the data ingestion script first."
        )
        return

    print("\n--- Basic Information ---")
    print("Shape of the dataset:", df.shape)
    print("\nData Info:")
    df.info()

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Target Variable Distribution ---")
    print(df["category"].value_counts())

    plt.figure(figsize=(8, 6))
    sns.countplot(x="category", data=df)
    plt.title("Distribution of Sentiment Categories")
    plt.xlabel("Category (-1: Negative, 0: Neutral, 1: Positive)")
    plt.ylabel("Count")

    plot_path = os.path.join(REPORTS_DIR, "category_distribution.png")
    plt.savefig(plot_path)
    print(f"\nSaved category distribution plot to {plot_path}")

    print("\n--- EDA Completed ---")


if __name__ == "__main__":
    run_analysis()
