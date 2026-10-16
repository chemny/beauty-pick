#!/usr/bin/env python3
"""Validation and smoke test script for BeautyPick models and dependencies."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

REQUIRED_MODELS = [
    "blaze_face_short_range.tflite",
    "face_landmarker.task",
    "pose_landmarker_full.task",
    "c2pa-trust-anchors.pem",
]


def validate() -> int:
    missing = []
    for model_name in REQUIRED_MODELS:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            missing.append(model_name)
        elif model_path.stat().st_size == 0:
            missing.append(f"{model_name} (empty file)")

    if missing:
        print(f"[FAIL] Missing required model files: {', '.join(missing)}")
        print("Run `python3 scripts/download_models.py` or `./setup.sh` to download.")
        return 1

    print("[PASS] All required vision models and trust anchors are verified.")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
