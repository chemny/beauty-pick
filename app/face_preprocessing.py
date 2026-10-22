"""Frozen face-only candidate: existing weights, square alignment, monotone scale.

No fitting to the review images, reference labels, component scores or rankings.
It does not change the production provider or any previous candidate.
"""
from __future__ import annotations

import hashlib
import math
from decimal import Decimal, ROUND_HALF_UP

import cv2
import numpy as np
from PIL import Image

from .config import ROOT
from .engine import dependency_versions, file_fingerprint, signature

VERSION = "face-square-monotone-20261022-1"
PRIMARY = "aligned_square"
VARIANTS = ("context", "mesh_rectangle", "square", PRIMARY)
MARGIN = .12
INPUT_SIZE = 224
# Retain the legacy raw-score scale for a controlled preprocessing comparison.
# These are display anchors, not fitted human rating norms or percentiles.
SCALE = {"raw_low": 2.0, "score_low": 40.0, "raw_high": 5.3,
         "score_high": 98.0, "clip_low": 35.0, "clip_high": 99.0}


def identity():
    return {"provider": "local_face_preprocessing_candidate", "model": "SCUT-FBP5500-ResNet50",
            "algorithm_version": VERSION, "primary": PRIMARY, "margin": MARGIN,
            "input_size": INPUT_SIZE, "scale": dict(SCALE), "reference_used_for_scoring": False,
            "weights": {p: file_fingerprint(ROOT/"models"/p) for p in
                        ("scut_fbp5500_beauty.pt", "face_landmarker.task")},
            "implementation": {p: file_fingerprint(ROOT/"app"/p) for p in
                               ("face_preprocessing.py", "face_preprocessing_trial.py", "beauty_model.py", "images.py")},
            "dependencies": dependency_versions()}


def scale_score(raw: float) -> dict:
    if type(raw) not in (int, float) or not math.isfinite(raw):
        raise ValueError("模型原值必须为有限实数")
    value = SCALE["score_low"] + (raw-SCALE["raw_low"])/(SCALE["raw_high"]-SCALE["raw_low"])*(SCALE["score_high"]-SCALE["score_low"])
    bounded = max(SCALE["clip_low"], min(SCALE["clip_high"], value))
    return {"total": float(Decimal(str(bounded)).quantize(Decimal(".1"), rounding=ROUND_HALF_UP)),
            "unclipped_total": value, "clipped": value != bounded,
            "raw_outside_nominal_1_to_5": not 1 <= raw <= 5}


def square_transform(points: np.ndarray, *, align: bool) -> dict:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("面部关键点缺失或无效")
    eyes = sorted([(points[33]+points[133])/2, (points[362]+points[263])/2], key=lambda p: p[0])
    direction = eyes[1]-eyes[0]
    if np.linalg.norm(direction) < 8:
        raise ValueError("眼部像素不足以稳定对齐")
    angle = math.atan2(direction[1], direction[0]) if align else 0.0
    c,s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c,s],[-s,c]])
    rotated = points[:468] @ rotation.T
    low,high = rotated.min(axis=0), rotated.max(axis=0)
    side = float(max(high-low)*(1+2*MARGIN))
    if side < 60:
        raise ValueError("面部裁脸像素不足")
    center = (low+high)/2
    factor = (INPUT_SIZE-1)/side
    matrix = np.column_stack((factor*rotation, (INPUT_SIZE-1)/2-factor*center))
    return {"matrix": matrix.tolist(), "rotation_degrees": math.degrees(angle), "side_pixels": side,
            "center_in_rotated_image": center.tolist(), "margin": MARGIN, "aligned": align}


def warp_square(image: Image.Image, transform: dict) -> tuple[Image.Image, dict]:
    rgb = np.asarray(image.convert("RGB"))
    matrix = np.asarray(transform["matrix"], dtype=np.float64)
    result = cv2.warpAffine(rgb, matrix, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    coverage = cv2.warpAffine(np.ones(rgb.shape[:2], np.uint8), matrix, (INPUT_SIZE, INPUT_SIZE),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    padding = float(1-coverage.mean())
    if padding > .5:
        raise ValueError("裁脸超过一半在原图范围之外")
    return Image.fromarray(result), {**transform, "padding_fraction": round(padding, 6),
                                     "warning": "部分边缘补齐，需复核" if padding > .05 else None}


def landmarks(image: Image.Image, target: dict) -> tuple[np.ndarray, Image.Image, dict]:
    from .images import crop_face
    from .beauty_model import _get_face_landmarker, _LANDMARKER_LOCK
    import mediapipe as mp

    if not target or any(type(target.get(k)) not in (int,float) or not math.isfinite(target[k])
                         for k in ("x","y","width","height")):
        raise ValueError("没有有效的已选目标面部")
    if min(target["width"]*image.width, target["height"]*image.height) < 90:
        raise ValueError("目标面部像素不足90，未补造评分")
    if target["width"] <= 0 or target["height"] <= 0 or min(target["x"], target["y"]) < 0 or max(target["x"]+target["width"],target["y"]+target["height"]) > 1.001:
        raise ValueError("目标面部超出原图")
    crop, box = crop_face(image, target)
    detector = _get_face_landmarker()
    if detector is None:
        raise ValueError("本地面部关键点模型缺失")
    rgb = np.asarray(crop.convert("RGB"))
    with _LANDMARKER_LOCK:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        raise ValueError("目标区域没有检出面部关键点")
    # crop_face may thumbnail a large target; map points back using crop's box.
    pts = np.array([(box["left"]+p.x*(box["right"]-box["left"]),
                     box["top"]+p.y*(box["bottom"]-box["top"])) for p in result.face_landmarks[0]])
    return pts, crop, box


def preprocess(image: Image.Image, target: dict) -> tuple[dict, dict]:
    points, context, box = landmarks(image, target)
    low,high = points[:468].min(axis=0),points[:468].max(axis=0)
    margin = (high-low)*MARGIN
    left,top = np.maximum(0, np.floor(low-margin)).astype(int)
    right,bottom = np.minimum(image.size, np.ceil(high+margin)).astype(int)
    rect = image.crop((left,top,right,bottom))
    images = {"context": context, "mesh_rectangle": rect}
    diagnostics = {"context": {"source_box": box}, "mesh_rectangle": {"source_box": [int(left),int(top),int(right),int(bottom)]}}
    for key, align in (("square",False), (PRIMARY,True)):
        images[key], diagnostics[key] = warp_square(image, square_transform(points,align=align))
    for key in ("context", "mesh_rectangle"):
        diagnostics[key]["aspect_ratio_before_resize"] = images[key].width/images[key].height
    return images,diagnostics


def infer(image: Image.Image) -> dict:
    from .beauty_model import evaluate_face_beauty
    # Save/hash this exact tensor input; model's Resize(224,224) becomes a no-op.
    if image.size != (INPUT_SIZE,INPUT_SIZE):
        raise ValueError("评分输入必须已经规范为224×224")
    output = evaluate_face_beauty(image)
    if not output.get("available"):
        raise ValueError(output.get("error", "本地颜值模型未返回分数"))
    raw = output["raw_score"]
    return {"raw_score": raw, **scale_score(raw), "status": "completed",
            "model": output["model"], "input_pixel_hash": signature({"mode":"RGB","size":list(image.size),
                 "pixels_sha256": hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()})}


def tensor_input(image: Image.Image) -> Image.Image:
    # Match torchvision's existing PIL Resize default for controlled ablations.
    return image.convert("RGB").resize((INPUT_SIZE,INPUT_SIZE),Image.Resampling.BILINEAR)


def perturbations(image: Image.Image, target: dict):
    resized = image.resize((round(image.width*.9), round(image.height*.9)), Image.Resampling.LANCZOS)
    yield "resize90", resized, dict(target)
    trim_x,trim_y = round(image.width*.02),round(image.height*.02)
    cropped = image.crop((trim_x,trim_y,image.width-trim_x,image.height-trim_y))
    box = dict(target)
    box.update(x=(target["x"]*image.width-trim_x)/cropped.width,
               y=(target["y"]*image.height-trim_y)/cropped.height,
               width=target["width"]*image.width/cropped.width,
               height=target["height"]*image.height/cropped.height)
    yield "outer_crop2", cropped, box
