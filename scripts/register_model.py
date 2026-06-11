"""Register the notebook-trained model into the API model registry."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.api import ModelRegistry
from src.model_loader import bootstrap_registry


def main():
    registry = ModelRegistry()
    version = bootstrap_registry(registry)
    if version:
        print(f"Registered model version: {version}")
        print(f"Current: {registry.manifest['current']}")
    else:
        print("No saved models found. Train models in notebooks/preprocessing.ipynb first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
