import json
from pathlib import Path

import pandas as pd

import src.data_processing as dp


def test_process_data_creates_processed_file(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    processed_path = tmp_path / "processed.csv"

    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0],
        }
    )
    df.to_csv(raw_path, index=False)

    monkeypatch.setattr(dp, "RAW_PATH", str(raw_path))
    monkeypatch.setattr(dp, "PROCESSED_PATH", str(processed_path))

    dp.process_data()

    assert processed_path.exists()
    processed_df = pd.read_csv(processed_path)
    assert "feature_sum" in processed_df.columns
    assert processed_df["feature_sum"].iloc[0] == 5.0


def test_validate_data_writes_success_boolean(tmp_path, monkeypatch):
    processed_path = tmp_path / "processed.csv"
    validation_path = tmp_path / "validation_result.json"

    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0],
            "feature2": [3.0, 4.0],
            "feature_sum": [4.0, 6.0],
            "target": [0, 1],
        }
    )
    df.to_csv(processed_path, index=False)

    monkeypatch.setattr(dp, "PROCESSED_PATH", str(processed_path))
    monkeypatch.setattr(dp, "VALIDATION_PATH", str(validation_path))

    dp.validate_data()

    assert validation_path.exists()
    payload = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    assert isinstance(payload["success"], bool)
    assert payload["success"] is True
