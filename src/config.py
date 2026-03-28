import os


class Config:
    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/data.csv")
    PROCESSED_DATA_PATH = os.getenv("DATA_PATH", "data/processed/processed.csv")
    VALIDATION_PATH = os.getenv("VALIDATION_PATH", "reports/validation_result.json")
    REPORT_PATH = os.getenv("REPORT_PATH", "reports/data_drift_report.html")
    FLAG_FILE = os.getenv("DRIFT_FLAG_FILE", "drift_detected.flag")
    MODEL_NAME = os.getenv("MODEL_NAME", "production-model")
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "churn-prediction")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))