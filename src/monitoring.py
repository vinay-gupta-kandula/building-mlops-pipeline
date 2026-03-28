import pandas as pd
import os

try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config

REFERENCE_DATA = Config.PROCESSED_DATA_PATH
CURRENT_DATA = Config.PROCESSED_DATA_PATH
REPORT_PATH = Config.REPORT_PATH
FLAG_FILE = Config.FLAG_FILE
DRIFT_THRESHOLD = Config.DRIFT_THRESHOLD


def detect_drift():
    ref = pd.read_csv(REFERENCE_DATA)
    curr = pd.read_csv(CURRENT_DATA)

    # Simple drift check (mean difference)
    drift_detected = False
    details = []

    for col in ["feature1", "feature2"]:
        mean_diff = abs(ref[col].mean() - curr[col].mean())
        details.append((col, mean_diff))
        if mean_diff > DRIFT_THRESHOLD:
            drift_detected = True

    # Generate HTML report
    os.makedirs("reports", exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        f.write("<html><body>")
        f.write("<h1>Data Drift Report</h1>")
        f.write(f"<p>Drift Detected: {drift_detected}</p>")
        f.write("<ul>")
        for col, diff in details:
            f.write(f"<li>{col}: mean diff={diff:.6f}</li>")
        f.write("</ul>")
        f.write("</body></html>")

    print("Drift report generated!")

    # Create flag if drift detected
    if drift_detected:
        open(FLAG_FILE, "w").close()
        print("Drift detected → flag created!")


if __name__ == "__main__":
    detect_drift()