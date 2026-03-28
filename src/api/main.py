from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from src.config import Config

app = FastAPI()

model = None
model_load_attempted = False


def _try_load_model(timeout_seconds: int = 10):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            mlflow.pyfunc.load_model,
            f"models:/{Config.MODEL_NAME}/Production",
        )
        return future.result(timeout=timeout_seconds)


def get_model():
    global model
    global model_load_attempted

    if model is not None or model_load_attempted:
        return model

    model_load_attempted = True
    try:
        model = _try_load_model(timeout_seconds=10)
        print("Model loaded")
    except TimeoutError:
        print("Model load timed out, continuing without it")
    except Exception as e:
        print("Model not found, continuing without it:", e)

    return model


class InputData(BaseModel):
    feature1: float
    feature2: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    active_model = get_model()

    if active_model is None:
        return {"error": "Model not available"}

    df = pd.DataFrame([{
        "feature1": data.feature1,
        "feature2": data.feature2,
        "feature_sum": data.feature1 + data.feature2
    }])

    prediction = active_model.predict(df)[0]
    return {"prediction": int(prediction)}