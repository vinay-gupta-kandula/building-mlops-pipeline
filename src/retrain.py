import os
import subprocess

try:
    from src.config import Config
except ModuleNotFoundError:
    from config import Config

FLAG_FILE = Config.FLAG_FILE


def _training_command():
    run_in_docker = os.getenv("RETRAIN_IN_DOCKER", "1") == "1"
    if run_in_docker:
        return ["docker", "compose", "exec", "-T", "api", "python", "src/train.py"]
    return ["python", "src/train.py"]


def retrain():
    if os.path.exists(FLAG_FILE):
        print("Drift detected. Retraining model...")

        subprocess.run(_training_command(), check=True)

        os.remove(FLAG_FILE)
        print("Retraining complete and flag removed!")
    else:
        print("No drift detected. Skipping retraining.")


if __name__ == "__main__":
    retrain()