"""Central configuration for paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

IMG_SIZE = (224, 224)          # (height, width) used by CNN / MobileNetV2
SEED = 42

# Class-folder name treated as the "positive" class when reporting precision/recall.
POSITIVE_CLASS = "counterfeit"

DEFAULT_MODEL_PATH = MODELS_DIR / "authenticity_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
