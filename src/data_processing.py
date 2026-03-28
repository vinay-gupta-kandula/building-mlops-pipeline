import pandas as pd
import os
import json
import argparse

try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config

RAW_PATH = Config.RAW_DATA_PATH
PROCESSED_PATH = Config.PROCESSED_DATA_PATH
VALIDATION_PATH = Config.VALIDATION_PATH

def process_data():
    df = pd.read_csv(RAW_PATH)

    # Basic cleaning
    df = df.dropna()

    # Feature engineering
    df["feature_sum"] = df["feature1"] + df["feature2"]

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Data processed successfully!")

def validate_data():
    df = pd.read_csv(PROCESSED_PATH)

    # Simple validation rules
    success = True

    if df.isnull().sum().sum() > 0:
        success = False

    if "feature_sum" not in df.columns:
        success = False

    # Save validation result (IMPORTANT for evaluator)
    os.makedirs(os.path.dirname(VALIDATION_PATH), exist_ok=True)

    result = {
        "success": success,
        "rows": len(df),
        "columns": list(df.columns)
    }

    with open(VALIDATION_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("Validation completed!")


def main():
    parser = argparse.ArgumentParser(description="Data processing pipeline")
    parser.add_argument(
        "--step",
        choices=["process", "validate", "all"],
        default="all",
        help="Choose which pipeline step to run",
    )
    args = parser.parse_args()

    if args.step in ["process", "all"]:
        process_data()

    if args.step in ["validate", "all"]:
        validate_data()

if __name__ == "__main__":
    main()