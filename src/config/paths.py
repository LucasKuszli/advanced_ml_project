from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]

DATA_DIR = ROOT_DIR / "data"
IMG_DIR = ROOT_DIR / "img"

LOG_DIR = ROOT_DIR / "logs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
