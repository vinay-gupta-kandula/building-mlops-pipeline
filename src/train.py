import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config

DATA_PATH = Config.PROCESSED_DATA_PATH


def train_model(C):
    df = pd.read_csv(DATA_PATH)

    X = df[["feature1", "feature2", "feature_sum"]]
    y = df["target"]

    model = LogisticRegression(C=C)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    return model, acc


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", Config.MLFLOW_TRACKING_URI)
    )
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    client = MlflowClient()

    best_acc = -1
    best_run_id = None

    # Train multiple runs
    for C in [0.1, 1.0, 10.0]:
        with mlflow.start_run() as run:
            model, acc = train_model(C)

            mlflow.log_param("C", C)
            mlflow.log_metric("accuracy", acc)

            mlflow.sklearn.log_model(model, "model")

            print(f"Run completed with C={C}, accuracy={acc}")

            # Track best model
            if acc > best_acc:
                best_acc = acc
                best_run_id = run.info.run_id

    # Register best model
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=Config.MODEL_NAME
    )

    # Move model to Production stage
    client.transition_model_version_stage(
        name=Config.MODEL_NAME,
        version=result.version,
        stage="Production"
    )

    print("Model registered and moved to Production stage")