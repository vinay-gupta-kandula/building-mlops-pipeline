import pandas as pd

import src.train as train


def test_train_model_returns_model_and_accuracy(tmp_path, monkeypatch):
    data_path = tmp_path / "processed.csv"
    df = pd.DataFrame(
        {
            "feature1": [0.0, 1.0, 0.5, 1.5],
            "feature2": [1.0, 0.0, 1.5, 0.5],
            "feature_sum": [1.0, 1.0, 2.0, 2.0],
            "target": [0, 1, 0, 1],
        }
    )
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(train, "DATA_PATH", str(data_path))

    model, acc = train.train_model(1.0)

    assert model is not None
    assert 0.0 <= acc <= 1.0
