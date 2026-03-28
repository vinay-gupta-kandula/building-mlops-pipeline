from fastapi.testclient import TestClient

from src.api.main import app
import src.api.main as api_main


client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_predict_invalid_payload_returns_422():
    res = client.post("/predict", json={"feature1": "wrong"})
    assert res.status_code == 422


def test_predict_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(api_main, "get_model", lambda: None)

    res = client.post("/predict", json={"feature1": 1.0, "feature2": 2.0})
    assert res.status_code == 200
    assert res.json() == {"error": "Model not available"}


def test_predict_success_with_mock_model(monkeypatch):
    class MockModel:
        def predict(self, df):
            return [1]

    monkeypatch.setattr(api_main, "get_model", lambda: MockModel())

    res = client.post("/predict", json={"feature1": 1.0, "feature2": 2.0})
    assert res.status_code == 200
    assert res.json() == {"prediction": 1}