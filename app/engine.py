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
    for name in ("torch", "timm", "torchvision", "mediapipe", "opencv-python", "numpy", "Pillow", "pillow-heif", "einops", "ftfy", "regex", "wcwidth"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


SOURCE_FILES = ("engine.py", "provider.py", "scoring.py", "pipeline.py", "images.py", "image_formats.py",
                "beauty_model.py", "pose_model.py", "photo_model.py", "standard_face.py", "face_preprocessing.py", "fpem_model.py", "topiq_model.py")
SOURCE_FILES += tuple(str(p.relative_to(ROOT / "app")) for p in sorted((ROOT / "app/fpem_vendor").rglob("*")) if p.is_file() and p.suffix in (".py", ".gz", ".json"))
SOURCE_FILES += tuple(str(p.relative_to(ROOT / "app")) for p in sorted((ROOT / "app/topiq_vendor").rglob("*")) if p.is_file() and p.suffix in (".py", ".json"))
_IMPLEMENTATION = {name: file_fingerprint(ROOT / "app" / name) for name in SOURCE_FILES}


def identity(module: str) -> dict:
    if module == "body":
        from .body_model import identity as body_identity
        return body_identity()
    from .beauty_model import MODEL_PATH, FACE_LANDMARKER_PATH
    from .pose_model import MODEL_PATH as POSE_PATH
    from . import fpem_model, topiq_model
    face_model = fpem_model.MODEL_PATH if fpem_model.selected_engine() == "fpem" else MODEL_PATH

    detector = ROOT / "models/face_detection_full_range.tflite"
    if not detector.exists():
        detector = ROOT / "models/blaze_face_short_range.tflite"
    weights = {"face": [face_model, FACE_LANDMARKER_PATH, detector], "body": [POSE_PATH],
               "photography": [topiq_model.MODEL_PATH] if topiq_model.selected_engine() == "topiq" else [FACE_LANDMARKER_PATH, detector]}[module]
    implementation = {name: file_fingerprint(ROOT / "app" / name) for name in SOURCE_FILES}
    if implementation != _IMPLEMENTATION:
        raise RuntimeError("评分代码已变化，请重启服务后重新分析")
    result = {"provider": ENGINE_ID, "model": MODEL_NAME, "reasoning_effort": None,
            "algorithm_version": ALGORITHM_VERSION, "preprocessing_version": PREPROCESSING_VERSION,
            "weights": {p.name: file_fingerprint(p) for p in weights},
            "implementation": implementation,
            "dependencies": dependency_versions()}
    if module == "face" and fpem_model.selected_engine() == "fpem":
        result.update(model="FPEM", face_scoring_method=fpem_model.METHOD,
                      rubric_version=fpem_model.RUBRIC_VERSION, prompt_version=fpem_model.PROMPT_VERSION,
                      device=fpem_model.device_name(), algorithm_version="fpem-production-20261023-1",
                      preprocessing_version="standard-face-edge-v2-fpem-224-112-160-rgb01-fp32",
                      display_mapping="raw_1_to_5_times_20")
    elif module == "face":
        result.update(model="SCUT-FBP5500-ResNet50", face_scoring_method="standard-face-v1",
                      algorithm_version="standard-face-production-20261023-2",
                      preprocessing_version="standard-face-margin12-square224-bilinear-png-edge-v2")
    if module == "photography" and topiq_model.selected_engine() == "topiq":
        result.update(model="TOPIQ-IAA-ResNet50", scoring_method=topiq_model.METHOD,
                      rubric_version=topiq_model.RUBRIC_VERSION, prompt_version=topiq_model.PROMPT_VERSION,
                      device="cpu", algorithm_version="topiq-production-20261023-1",
                      preprocessing_version="oriented-rgb-lossless-png-shortedge384-antialias-imagenet-v1",
                      display_mapping="raw_1_to_10_times_10")
    return result
