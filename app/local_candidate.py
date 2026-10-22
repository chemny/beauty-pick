"""Offline candidate measurements. No Codex, network, database or default-provider use.

Body outlines and image statistics are proxies, not trained aesthetic judgements.
The trial fits separate regularized overall/facet heads to saved reference labels.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import ROOT
from .engine import dependency_versions, file_fingerprint, signature

VERSION = "local-reference-calibration-20261022-1"
ALPHA = 10.0  # Fixed before viewing errors; never select it on the test photo.
MODULES = ("face", "body", "photography")

# These are measurement inputs, not aesthetic weights or universal ideal ratios.
FEATURES = {
    "face": {
        "overall": ("beauty_raw", "eye_aspect", "nose_width", "jaw_cheek", "skin_lightness", "skin_redness"),
        "eyes": ("eye_aspect", "eye_spacing"),
        "nose_lips": ("nose_width", "mouth_width", "lip_aspect"),
        "harmony": ("eye_spacing", "middle_lower", "upper_middle"),
        "face_shape": ("face_aspect", "jaw_cheek"),
        "skin": ("skin_lightness", "skin_redness", "skin_yellowness", "skin_spread"),
    },
    "body": {
        "overall": ("chest_shoulder", "waist_chest", "hip_waist", "shoulder_hip", "torso_shoulder", "leg_torso"),
        "chest": ("chest_shoulder",),
        "waist": ("waist_chest",),
        "hips": ("hip_waist", "shoulder_hip"),
        "legs": ("leg_torso",),
        "proportion": ("shoulder_hip", "torso_shoulder", "leg_torso"),
        "harmony": ("outline_variation",),
    },
    "photography": {
        "overall": ("subject_clarity", "subject_clipping", "subject_range", "background_edges", "headroom", "saturation"),
        "light": ("subject_clipping", "subject_range", "subject_luminance"),
        "clarity": ("subject_clarity",),
        "composition": ("headroom", "horizontal_center"),
        "color": ("saturation", "chroma_spread"),
        "background": ("background_edges", "subject_background_detail"),
    },
}


def identity() -> dict:
    files = ("local_candidate.py", "local_calibration_trial.py", "beauty_model.py", "images.py")
    weights = ("scut_fbp5500_beauty.pt", "face_landmarker.task", "pose_landmarker_full.task")
    return {"provider": "local_reference_candidate", "model": "resnet50+mediapipe+cv+ridge",
            "algorithm_version": VERSION, "network_required": False, "alpha": ALPHA,
            "weights": {p: file_fingerprint(ROOT / "models" / p) for p in weights},
            "implementation": {p: file_fingerprint(ROOT / "app" / p) for p in files},
            "dependencies": dependency_versions(), "preprocessing": "trial-jpeg90-face-tight12-body-pad32-photo1024-v1"}


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def ratio(a, b):
    return float(a / b) if finite(a) and finite(b) and b > 1e-8 else None


def aligned_body_input(image: Image.Image):
    """Pad rather than stretch: MediaPipe float-mask copy aborts on unaligned rows.

    Padding is outside original coordinates and never treated as visible body.
    """
    image = image.copy(); image.thumbnail((1024, 1024))
    rgb = np.asarray(image.convert("RGB"))
    h,w = rgb.shape[:2]
    return cv2.copyMakeBorder(rgb,0,(-h)%32,0,(-w)%32,cv2.BORDER_REPLICATE), (w,h)


def segmentation_array(mask, width: int, height: int) -> np.ndarray:
    """Reject unsafe native float copies before entering MediaPipe's C API.

    Some installed MediaPipe builds abort (rather than raise) when numpy_view
    copies a padded float image. Aligned input normally avoids that path, but
    validate the returned mask too so a changed output cannot kill the process.
    Keep the native image alive until the owned NumPy copy is complete.
    """
    if (mask.width, mask.height, mask.channels) != (width, height, 1):
        raise ValueError("人物分割图尺寸或通道不匹配，已跳过不安全的读取")
    if not mask.is_contiguous():
        raise ValueError("人物分割图内存不连续，已跳过可能导致崩溃的读取")
    values = np.array(mask.numpy_view(), copy=True)
    if values.dtype != np.float32 or values.size != width * height:
        raise ValueError("人物分割图数据格式不匹配")
    return values.reshape(height, width)


def fit_head(rows: list[dict], features: tuple[str, ...], *, upper: float) -> dict:
    """Training-only median imputation/normalization. At least four real labels.

    The intercept baseline is stored to expose models that merely learn the mean.
    Constant/missing features have exactly zero coefficients; no fake defaults.
    """
    valid = [r for r in rows if finite(r.get("target")) and 0 <= r["target"] <= upper]
    if len(valid) < 4:
        raise ValueError("有效参考样本少于4张，不能校准")
    x = np.array([[r["features"].get(k, np.nan) if finite(r["features"].get(k)) else np.nan
                   for k in features] for r in valid], dtype=float)
    center = np.array([float(np.median(c[np.isfinite(c)])) if np.isfinite(c).any() else 0.0 for c in x.T])
    present = np.sum(np.isfinite(x), axis=0)
    x = np.where(np.isfinite(x), x, center)
    scale = np.std(x, axis=0)
    usable = (scale > 1e-6) & (present >= 4)
    if not usable.any():
        raise ValueError("没有足够的可变测量特征，不能用平均分冒充模型评分")
    scale = np.where(usable, scale, 1.0)
    z = (x - center) / scale
    z[:, ~usable] = 0
    y = np.array([r["target"] for r in valid])
    intercept = float(y.mean())
    centered = z - z.mean(axis=0)
    coefficients = np.linalg.solve(centered.T @ centered + ALPHA * np.eye(len(features)), centered.T @ (y - intercept))
    # Median imputation means z need not have zero mean. Preserve the fitted mean.
    intercept -= float(z.mean(axis=0) @ coefficients)
    return {"features": list(features), "center": center.tolist(), "scale": scale.tolist(),
            "coefficients": coefficients.tolist(), "usable": usable.tolist(), "intercept": intercept,
            "mean_baseline": float(y.mean()), "training_ids": [r["id"] for r in valid],
            "upper": upper, "alpha": ALPHA}


def predict(head: dict, values: dict) -> dict:
    z, missing, outliers = [], [], []
    for k, center, scale, use in zip(head["features"], head["center"], head["scale"], head["usable"]):
        value = values.get(k)
        if not finite(value):
            missing.append(k)
            standardized = 0.0
        else:
            standardized = (value - center) / scale
        if use and abs(standardized) > 3:
            outliers.append(k)
        z.append(standardized if use else 0.0)
    raw = float(head["intercept"] + np.dot(z, head["coefficients"]))
    return {"score": round(max(0.0, min(head["upper"], raw)), 1), "unclipped": raw,
            "missing_features": missing, "out_of_distribution": outliers}


class Measurements:
    def __init__(self):
        self._pose = None

    def close(self):
        if self._pose is not None:
            self._pose.close()

    def face(self, image: Image.Image, *, include_beauty: bool = True) -> dict:
        from .beauty_model import _get_face_landmarker, _LANDMARKER_LOCK, evaluate_face_beauty
        import mediapipe as mp

        detector = _get_face_landmarker()
        if detector is None:
            raise ValueError("缺少本地面部关键点模型")
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        with _LANDMARKER_LOCK:
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            raise ValueError("本地面部关键点未检出")
        pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
        dist = lambda a, b: float(np.linalg.norm(pts[a] - pts[b]))
        cheek = dist(234, 454)
        eye_w = (dist(33, 133) + dist(362, 263)) / 2
        values = {
            "eye_aspect": ratio((dist(159, 145) + dist(386, 374)) / 2, eye_w),
            "eye_spacing": ratio(dist(133, 362), eye_w),
            "nose_width": ratio(dist(102, 331), cheek),
            "mouth_width": ratio(dist(61, 291), cheek),
            "lip_aspect": ratio(dist(0, 17), dist(61, 291)),
            "face_aspect": ratio(dist(10, 152), cheek),
            "jaw_cheek": ratio(dist(136, 365), cheek),
            "middle_lower": ratio(dist(168, 2), dist(2, 152)),
            "upper_middle": ratio(dist(10, 168), dist(168, 2)),
        }
        # Tight face crop suppresses clothing/background without inventing an ideal face.
        x0, y0 = pts[:468].min(axis=0); x1, y1 = pts[:468].max(axis=0)
        mx, my = (x1 - x0) * .12, (y1 - y0) * .12
        tight = image.crop((max(0, int(x0-mx)), max(0, int(y0-my)), min(w, int(x1+mx)), min(h, int(y1+my))))
        beauty = None
        if include_beauty:
            beauty = evaluate_face_beauty(tight)
            if not beauty.get("available"):
                raise ValueError(beauty.get("error", "本地颜值模型失败"))
            values["beauty_raw"] = beauty["raw_score"]
        mask = np.zeros((h, w), np.uint8)
        for indexes in ((50, 101, 205, 187, 123), (280, 330, 425, 411, 352)):
            cv2.fillConvexPoly(mask, cv2.convexHull(pts[list(indexes)].astype(np.int32)), 1)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(float)
        pixels = lab[mask > 0]
        if len(pixels) >= 80:
            med = np.median(pixels, axis=0)
            values.update(skin_lightness=float(med[0] * 100 / 255), skin_redness=float(med[1]-128),
                          skin_yellowness=float(med[2]-128),
                          skin_spread=float(np.median(np.linalg.norm(pixels - med, axis=1))))
        return {"features": values, "scope": "face", "available": True,
                "limitations": ["局部几何与双颊色彩是测量代理；并非公认美学标准，也不能识别真实肤质。"],
                "diagnostics": {"cheek_pixels": int(len(pixels)), "beauty_model": beauty["model"] if beauty else None}}

    def body(self, image: Image.Image) -> dict:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if self._pose is None:
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(ROOT / "models/pose_landmarker_full.task")),
                output_segmentation_masks=True, min_pose_detection_confidence=.5, min_pose_presence_confidence=.5))
        rgb, (original_w,original_h) = aligned_body_input(image)
        h,w = rgb.shape[:2]
        result = self._pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.pose_landmarks or not result.segmentation_masks:
            raise ValueError("本地身体关键点或人物分割未检出")
        points = result.pose_landmarks[0]
        xy = np.array([(p.x*w, p.y*h) for p in points])
        visible = lambda k: points[k].visibility >= .6 and points[k].presence >= .6 and .01*original_w <= xy[k,0] <= .99*original_w and .01*original_h <= xy[k,1] <= .99*original_h
        if not all(visible(k) for k in (11, 12)):
            raise ValueError("双肩缺少可靠可见测量，不能计算身材总分")
        shoulder = (xy[11]+xy[12])/2
        shoulder_width = float(np.linalg.norm(xy[11]-xy[12]))
        if shoulder_width < 30:
            raise ValueError("可见肩部像素不足")
        hip_visible = all(visible(k) for k in (23, 24))
        hip = (xy[23]+xy[24])/2
        mask = segmentation_array(result.segmentation_masks[0], w, h)[:original_h,:original_w] > .65
        w,h = original_w,original_h

        def width_at(point):
            x, y = int(point[0]), int(point[1])
            if not 3 <= y < h-3 or not 1 <= x < w-1:
                return None
            line = np.mean(mask[y-2:y+3], axis=0) >= .6
            if not line[x]:
                return None
            left, right = x, x
            while left > 0 and line[left-1]: left -= 1
            while right < w-1 and line[right+1]: right += 1
            if left <= 1 or right >= w-2:
                return None  # A clipped contour is unknown, never a narrow waist.
            width = right-left+1
            return float(width) if width > .15*shoulder_width else None

        chest_point = shoulder + np.array([0, shoulder_width*.30])
        chest = width_at(chest_point)
        values = {"chest_shoulder": ratio(chest, shoulder_width)}
        if hip_visible:
            torso = float(np.linalg.norm(hip-shoulder))
            hip_width = float(np.linalg.norm(xy[23]-xy[24]))
            waist = width_at(shoulder*.35 + hip*.65)
            hips = width_at(hip)
            widths = [width_at(shoulder*(1-t)+hip*t) for t in np.linspace(.25, .9, 12)]
            widths = [v for v in widths if v is not None]
            values.update(waist_chest=ratio(waist, chest), hip_waist=ratio(hips, waist),
                          shoulder_hip=ratio(shoulder_width, hip_width), torso_shoulder=ratio(torso, shoulder_width),
                          outline_variation=ratio(float(np.std(np.diff(widths))), float(np.mean(widths))) if len(widths) >= 8 else None)
            legs = []
            for hip_k, knee_k, ankle_k in ((23,25,27), (24,26,28)):
                if not all(visible(k) for k in (hip_k,knee_k,ankle_k)):
                    continue
                a, b = xy[hip_k]-xy[knee_k], xy[ankle_k]-xy[knee_k]
                cosine = float(np.dot(a,b) / max(1e-6,np.linalg.norm(a)*np.linalg.norm(b)))
                angle = math.degrees(math.acos(max(-1., min(1., cosine))))
                if angle >= 150:
                    legs.append(float(np.linalg.norm(a)+np.linalg.norm(b)))
            values["leg_torso"] = ratio(float(np.mean(legs)), torso) if legs else None
        available = chest is not None or hip_visible
        return {"features": values, "available": available,
                "error": None if available else "肩胸外轮廓被画框截断，且腰胯没有可靠可见测量",
                "scope": "full_body" if finite(values.get("leg_torso")) else "half_body" if hip_visible else "upper_body",
                "limitations": ["测量人物及衣物外轮廓，手臂、宽松衣物和透视可能干扰。",
                                 "关键点与分割不等于胸部、臀部形态识别；局部预测仅为低置信代理。"],
                "diagnostics": {"hips_visible": hip_visible, "shoulder_pixels": round(shoulder_width, 1)}}

    def photography(self, image: Image.Image, box: dict | None) -> dict:
        im = image.copy(); im.thumbnail((1024,1024))
        rgb = np.asarray(im.convert("RGB")); h,w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        values = {}
        if box:
            x,y,bw,bh = (box[k] for k in ("x","y","width","height"))
            x0,y0,x1,y1 = max(0,int(x*w)),max(0,int(y*h)),min(w,int((x+bw)*w)),min(h,int((y+bh)*h))
            subject = gray[y0:y1,x0:x1]
            if subject.size < 100:
                raise ValueError("人物测光区域太小")
            values.update(headroom=float(y), horizontal_center=float(abs(x+bw/2-.5)))
            bg_mask = np.ones((h,w), bool)
            bg_mask[max(0,y0-int(bh*h)):h,max(0,x0-int(bw*w)):min(w,x1+int(bw*w))] = False
        else:
            subject = gray
            bg_mask = np.zeros((h,w), bool)
        subject_standard = cv2.resize(subject,(256,256),interpolation=cv2.INTER_AREA)
        clarity = float(cv2.Laplacian(subject_standard,cv2.CV_64F).var())
        edges = cv2.Canny(gray,80,160)>0
        bg_edges = float(edges[bg_mask].mean()) if bg_mask.any() else None
        hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(float)
        values.update(subject_clarity=math.log1p(clarity),
                      subject_clipping=float(np.mean((subject < 10)|(subject > 245))),
                      subject_range=float(np.percentile(subject,95)-np.percentile(subject,5)),
                      subject_luminance=float(subject.mean()), background_edges=bg_edges,
                      saturation=float(hsv[:,:,1].mean()/255),
                      chroma_spread=float(np.std(lab[:,:,1:3],axis=(0,1)).mean()),
                      subject_background_detail=ratio(float(cv2.Canny(subject_standard,80,160).mean()/255),bg_edges))
        return {"features": values, "available": True, "scope": "whole_photo",
                "limitations": ["曝光、焦点、构图位置与背景边缘为技术代理，不能完整理解画面意图。"],
                "diagnostics": {"subject": "face" if box else "whole_photo", "normalized_long_edge": max(w,h)}}


def cache_key(input_hashes: dict, engine: dict) -> str:
    return signature({"input_hashes": input_hashes, "engine": engine})
