from pathlib import Path

import pandas as pd

import src.monitoring as monitoring
import src.retrain as retrain


def _write_dataset(path: Path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_monitoring_creates_report_and_flag_on_drift(tmp_path, monkeypatch):
    ref_path = tmp_path / "reference.csv"
    curr_path = tmp_path / "current.csv"
    report_path = tmp_path / "drift_report.html"
    flag_path = tmp_path / "drift_detected.flag"

    _write_dataset(
        ref_path,
        [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 1.1, "feature2": 2.2},
        ],
    )
    _write_dataset(
        curr_path,
        [
            {"feature1": 10.0, "feature2": 2.0},
            {"feature1": 11.0, "feature2": 2.2},
        ],
    )

    monkeypatch.setattr(monitoring, "REFERENCE_DATA", str(ref_path))
    monkeypatch.setattr(monitoring, "CURRENT_DATA", str(curr_path))
    monkeypatch.setattr(monitoring, "REPORT_PATH", str(report_path))
    monkeypatch.setattr(monitoring, "FLAG_FILE", str(flag_path))
    monkeypatch.setattr(monitoring, "DRIFT_THRESHOLD", 0.5)

    monitoring.detect_drift()

    assert report_path.exists()
    assert "Data Drift Report" in report_path.read_text(encoding="utf-8")
    assert flag_path.exists()


def test_retrain_skips_without_flag(monkeypatch, capsys, tmp_path):
    flag_path = tmp_path / "missing.flag"
    monkeypatch.setattr(retrain, "FLAG_FILE", str(flag_path))

    called = {"run": False}

    def fake_run(*args, **kwargs):
        called["run"] = True

    monkeypatch.setattr(retrain.subprocess, "run", fake_run)

    retrain.retrain()

    assert called["run"] is False
    assert "No drift detected" in capsys.readouterr().out


def test_retrain_runs_when_flag_exists(monkeypatch, tmp_path):
    flag_path = tmp_path / "drift_detected.flag"
    flag_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(retrain, "FLAG_FILE", str(flag_path))

    called = {"run": False, "check": False}

    def fake_run(args, check=False):
        called["run"] = True
        called["check"] = check
        return None

    monkeypatch.setattr(retrain.subprocess, "run", fake_run)

    retrain.retrain()

    assert called["run"] is True
    assert called["check"] is True
    assert not flag_path.exists()
