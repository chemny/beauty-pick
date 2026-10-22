"""Execution identity shared by reports, model loaders and cache keys."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import ROOT

ENGINE_ID = "fast_objective_engine"
MODEL_NAME = "native_apple_mps_resnet50_mediapipe_cv"
ALGORITHM_VERSION = "local-reliability-1"
PREPROCESSING_VERSION = "oriented-rgb-thumbnail1800-jpeg90-optimize-v1"


@lru_cache(maxsize=128)
def _digest(path: str, inode: int, size: int, mtime: int, ctime: int) -> str:
    with open(path, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def file_fingerprint(path: Path) -> str | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return _digest(str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def dependency_versions() -> dict:
    result = {}
    for name in ("torch", "timm", "torchvision", "mediapipe", "opencv-python", "numpy", "Pillow"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


SOURCE_FILES = ("engine.py", "provider.py", "scoring.py", "pipeline.py", "images.py",
                "beauty_model.py", "pose_model.py", "photo_model.py", "standard_face.py", "face_preprocessing.py")
_IMPLEMENTATION = {name: file_fingerprint(ROOT / "app" / name) for name in SOURCE_FILES}


def identity(module: str) -> dict:
    from .beauty_model import MODEL_PATH, FACE_LANDMARKER_PATH
    from .pose_model import MODEL_PATH as POSE_PATH

    detector = ROOT / "models/face_detection_full_range.tflite"
    if not detector.exists():
        detector = ROOT / "models/blaze_face_short_range.tflite"
    weights = {"face": [MODEL_PATH, FACE_LANDMARKER_PATH, detector], "body": [POSE_PATH],
               "photography": [FACE_LANDMARKER_PATH, detector]}[module]
    implementation = {name: file_fingerprint(ROOT / "app" / name) for name in SOURCE_FILES}
    if implementation != _IMPLEMENTATION:
        raise RuntimeError("评分代码已变化，请重启服务后重新分析")
    result = {"provider": ENGINE_ID, "model": MODEL_NAME, "reasoning_effort": None,
            "algorithm_version": ALGORITHM_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
            "weights": {p.name: file_fingerprint(p) for p in weights},
            "implementation": implementation,
            "dependencies": dependency_versions()}
    if module == "face":
        result.update(model="SCUT-FBP5500-ResNet50", face_scoring_method="standard-face-v1",
                      algorithm_version="standard-face-production-20261022-1",
                      preprocessing_version="standard-face-margin12-square224-bilinear-png-v1")
    return result
