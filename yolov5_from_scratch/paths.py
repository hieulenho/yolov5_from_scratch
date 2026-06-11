import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("YOLO_PROJECT_ROOT", PACKAGE_ROOT.parent)
).resolve()

CONFIGS_DIR = PACKAGE_ROOT / "configs"
DATASETS_DIR = Path(
    os.environ.get("YOLO_DATASETS_DIR", PROJECT_ROOT / "datasets")
).resolve()
ARTIFACTS_DIR = Path(
    os.environ.get("YOLO_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
).resolve()
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
TRAINING_DIR = ARTIFACTS_DIR / "training"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

DEFAULT_DATA_CONFIG = CONFIGS_DIR / "traffic5.yaml"
DEFAULT_WEIGHTS = CHECKPOINTS_DIR / "traffic5" / "best.pt"
