from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_PIXELS, ROOT

FORMAT_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class InvalidImage(ValueError):
    pass


def inspect_image(raw: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in FORMAT_MIME:
                raise InvalidImage("仅支持 JPEG、PNG 和静态 WebP")
            if getattr(image, "n_frames", 1) != 1:
                raise InvalidImage("不支持动画图片")
            if image.width * image.height > MAX_PIXELS:
                raise InvalidImage("图片像素数超过上限")
            image.load()
            return {"mime": FORMAT_MIME[image.format], "width": ImageOps.exif_transpose(image).width,
                    "height": ImageOps.exif_transpose(image).height, "sha256": hashlib.sha256(raw).hexdigest()}
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage("图片无法安全解码") from exc


def oriented_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def preview_bytes(path: Path, max_side: int = 1800) -> bytes:
    image = oriented_rgb(path)
    image.thumbnail((max_side, max_side))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=86, optimize=True)
    return output.getvalue()


def quality_metrics(image: Image.Image) -> dict:
    rgb = np.asarray(image)
    if rgb.shape[0] > 1200 or rgb.shape[1] > 1200:
        scale = min(1200 / rgb.shape[0], 1200 / rgb.shape[1])
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "luminance": round(float(gray.mean()), 1),
        "blur_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "dark_fraction": round(float(np.mean(gray < 12)), 3),
        "bright_fraction": round(float(np.mean(gray > 243)), 3),
        "note": "这些是图像测量值，不是美感分或真实性概率。",
    }


def detect_faces(image: Image.Image) -> list[dict]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    full_model = ROOT / "models" / "face_detection_full_range.tflite"
    short_model = ROOT / "models" / "blaze_face_short_range.tflite"
    model = full_model if full_model.exists() else short_model
    if not model.exists():
        raise RuntimeError("本地人脸检测模型尚未安装")
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    is_full_range = (model == full_model)
    min_conf = 0.35 if is_full_range else 0.5
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        min_detection_confidence=min_conf,
    )
    with vision.FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    found = []
    detector_name = "MediaPipe FullRange" if is_full_range else "MediaPipe BlazeFace"
    for i, detection in enumerate(result.detections):
        box = detection.bounding_box
        x, y = max(0, box.origin_x), max(0, box.origin_y)
        bw, bh = min(box.width, w - x), min(box.height, h - y)
        conf = float(detection.categories[0].score)
        # Filter obvious bottom-ground false positives in full-body shots when using full-range
        if is_full_range and y > 0.75 * h and conf < 0.7:
            continue
        found.append({"index": i, "x": round(x / w, 5), "y": round(y / h, 5),
                      "width": round(bw / w, 5), "height": round(bh / h, 5),
                      "pixel_width": bw, "pixel_height": bh,
                      "detection_confidence": round(conf, 3),
                      "detector": detector_name})
    # Sort primarily by confidence and size, prioritizing true faces over accidental background boxes
    return sorted(found, key=lambda item: (item["detection_confidence"] >= 0.6, item["detection_confidence"], item["width"] * item["height"]), reverse=True)


def crop_face(image: Image.Image, box: dict, margin: float = 0.45) -> tuple[Image.Image, dict]:
    w, h = image.size
    x0, y0 = box["x"] * w, box["y"] * h
    bw, bh = box["width"] * w, box["height"] * h
    side = max(bw, bh) * (1 + 2 * margin)
    center_x, center_y = x0 + bw / 2, y0 + bh / 2
    left, top = max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2))
    right, bottom = min(w, int(center_x + side / 2)), min(h, int(center_y + side / 2))
    crop = image.crop((left, top, right, bottom))
    crop.thumbnail((1024, 1024))
    return crop, {"left": left, "top": top, "right": right, "bottom": bottom}


def face_color_profile(image: Image.Image) -> dict | None:
    """Approximate face-region pixel color for model QA, never a beauty score."""
    faces = detect_faces(image)
    if not faces or faces[0]["detection_confidence"] < 0.7:
        return None
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    face = faces[0]
    left = max(0, int(face["x"] * width))
    top = max(0, int(face["y"] * height))
    right = min(width, left + face["pixel_width"])
    bottom = min(height, top + face["pixel_height"])
    region = rgb[top:bottom, left:right]
    if region.size == 0:
        return None
    lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
    ycrcb = cv2.cvtColor(region, cv2.COLOR_RGB2YCrCb)
    # Broad skin-like pixel gate removes most hair, eyes, and background. It is
    # intentionally descriptive and cannot identify complexion or skin health.
    sample = ((ycrcb[:, :, 1] > 130) & (ycrcb[:, :, 1] < 180)
              & (ycrcb[:, :, 2] > 75) & (ycrcb[:, :, 2] < 140)
              & (lab[:, :, 0] > 70))
    if int(sample.sum()) < 500 or float(sample.mean()) < 0.2:
        return None
    pixels = lab[sample]
    return {
        "sample_pixels": int(sample.sum()),
        "median_lightness_L_star": round(float(np.median(pixels[:, 0])) * 100 / 255, 1),
        "median_red_green_a_star": round(float(np.median(pixels[:, 1])) - 128, 1),
        "median_yellow_blue_b_star": round(float(np.median(pixels[:, 2])) - 128, 1),
    }


def save_crop(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=90, optimize=True)


def face_assessability(box: dict, crop: Image.Image) -> dict:
    metrics = quality_metrics(crop)
    face_px = min(box["pixel_width"], box["pixel_height"])
    limitations = []
    if crop.width < 160 or crop.height < 160 or face_px < 90:
        limitations.append("可见面部像素不足，无法可靠评分")
    if metrics["blur_variance"] < 25:
        limitations.append("标准裁脸较模糊，可能影响细节判断")
    return {"assessable": not limitations, "limitations": limitations, "metrics": metrics,
            "note": "自动门槛仅作初筛；遮挡、角度和表情仍需在语义评估中逐项判断。"}
