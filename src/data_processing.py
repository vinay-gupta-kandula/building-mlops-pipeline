import pandas as pd
import os

RAW_PATH = "data/raw/data.csv"
PROCESSED_PATH = "data/processed/processed.csv"

def process_data():
    # Load data
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning (dummy example)
    df = df.dropna()

    # Feature engineering (example)
    df["feature_sum"] = df["feature1"] + df["feature2"]

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Data processed successfully!")

if __name__ == "__main__":
    process_data()