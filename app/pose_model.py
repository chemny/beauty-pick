from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ROOT
from .engine import file_fingerprint

logger = logging.getLogger(__name__)

_LANDMARKER: Any = None
_LANDMARKER_FINGERPRINT = None
_LOCK = threading.Lock()
MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"


def _get_landmarker():
    global _LANDMARKER, _LANDMARKER_FINGERPRINT
    fingerprint = file_fingerprint(MODEL_PATH)
    if fingerprint and _LANDMARKER is not None and _LANDMARKER_FINGERPRINT == fingerprint:
        return _LANDMARKER

    with _LOCK:
        if fingerprint and _LANDMARKER is not None and _LANDMARKER_FINGERPRINT == fingerprint:
            return _LANDMARKER

        if not MODEL_PATH.exists():
            return None

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        if file_fingerprint(MODEL_PATH) != fingerprint:
            landmarker.close()
            raise RuntimeError("身材关键点权重在加载期间变化")
        _LANDMARKER = landmarker
        _LANDMARKER_FINGERPRINT = fingerprint
        return _LANDMARKER


def _dist(p1, p2, w: float, h: float) -> float:
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)


def evaluate_body_pose(image: Image.Image) -> dict:
    """Evaluate body landmarks and geometric proportions using MediaPipe Pose Landmarker."""
    try:
        landmarker = _get_landmarker()
        if landmarker is None:
            return {
                "detected": False,
                "error": "MediaPipe Pose Landmarker 模型未安装 (models/pose_landmarker_full.task)",
            }

        import mediapipe as mp

        img_rgb = image.convert("RGB")
        w, h = img_rgb.size
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img_rgb))

        with _LOCK:
            res = landmarker.detect(mp_img)

        if not res.pose_landmarks:
            return {
                "detected": False,
                "scene": "head_only",
                "scene_display": "无可见身体",
                "reason": "未检出有效身体姿态骨骼",
            }

        lm = res.pose_landmarks[0]

        # Landmark indexes:
        # 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow,
        # 23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee, 27: left_ankle, 28: right_ankle,
        # 29: left_heel, 30: right_heel, 31: left_foot_index, 32: right_foot_index
        nose = lm[0]
        ls, rs = lm[11], lm[12]
        lh, rh = lm[23], lm[24]
        lk, rk = lm[25], lm[26]
        la, ra = lm[27], lm[28]

        # Visibility helper
        def vis(point) -> bool:
            return point.visibility > 0.45 and -0.05 <= point.x <= 1.05 and -0.05 <= point.y <= 1.05

        shoulders_vis = vis(ls) and vis(rs)
        hips_vis = vis(lh) and vis(rh)
        knees_vis = vis(lk) or vis(rk)
        ankles_vis = vis(la) and vis(ra)

        # Detect scene
        if not shoulders_vis and not hips_vis:
            scene = "head_only"
            scene_display = "仅头部/特写"
        elif not hips_vis or (not knees_vis and not ankles_vis):
            scene = "half_body"
            scene_display = "半身取景（下身截断）"
        elif not ankles_vis or (la.y > 0.98 and ra.y > 0.98):
            scene = "half_body"
            scene_display = "大半身取景（脚踝未入镜）"
        else:
            # Check seated angle between torso and thigh
            torso_v = ((lh.x + rh.x) / 2 - (ls.x + rs.x) / 2, (lh.y + rh.y) / 2 - (ls.y + rs.y) / 2)
            thigh_v = ((lk.x + rk.x) / 2 - (lh.x + rh.x) / 2, (lk.y + rk.y) / 2 - (lh.y + rh.y) / 2)
            dot = torso_v[0] * thigh_v[0] + torso_v[1] * thigh_v[1]
            mag1 = math.hypot(*torso_v)
            mag2 = math.hypot(*thigh_v)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2 + 1e-6))))) if mag1 and mag2 else 0

            if angle > 45:
                scene = "seated"
                scene_display = "坐姿/弯折体态"
            else:
                scene = "full_body"
                scene_display = "全身立姿完整入镜"

        # Geometric proportions
        shoulder_width = _dist(ls, rs, w, h) if shoulders_vis else None
        hip_width = _dist(lh, rh, w, h) if hips_vis else None
        shoulder_hip_ratio = round(shoulder_width / hip_width, 2) if (shoulder_width and hip_width and hip_width > 0) else None

        # Torso and Leg ratio
        torso_len = _dist(ls, lh, w, h) if (shoulders_vis and hips_vis) else None
        leg_len = _dist(lh, la, w, h) if (hips_vis and ankles_vis) else None
        torso_leg_ratio = round(torso_len / leg_len, 2) if (torso_len and leg_len and leg_len > 0) else None

        # Head-to-body ratio estimation if full body
        head_ratio = None
        if scene == "full_body" and vis(nose) and ankles_vis:
            head_h = _dist(nose, ls, w, h) * 1.6  # Chin to crown approximation
            total_h = max(la.y, ra.y) * h - min(0, (nose.y - _dist(nose, ls, w, h) * 0.6)) * h
            if head_h > 0:
                head_ratio = round(max(5.5, min(9.0, total_h / head_h)), 1)

        # Posture Diagnostics (高低肩、探颈、躯干挺拔度)
        shoulder_tilt_deg = 0.0
        if shoulders_vis:
            raw_angle = math.degrees(math.atan2((rs.y - ls.y) * h, (rs.x - ls.x) * w))
            shoulder_tilt_deg = round(abs(raw_angle if abs(raw_angle) <= 90 else (180 - abs(raw_angle))), 1)

        shoulder_balance = "端正对称立姿" if shoulder_tilt_deg <= 2.5 else "自然灵动侧倾造型" if shoulder_tilt_deg <= 10.0 else "动态时尚摆姿"

        # 探颈程度 (耳朵与肩膀的水平偏差)
        neck_offset_ratio = None
        if vis(nose) and shoulders_vis:
            mid_shoulder_x = (ls.x + rs.x) * 0.5
            neck_offset_ratio = round(float(abs(nose.x - mid_shoulder_x) / max(0.1, abs(rs.x - ls.x))), 2)

        posture_note = "体态舒展挺拔" if (neck_offset_ratio is None or neck_offset_ratio <= 0.18) else "体态舒展自然"

        parts_vis = {
            "chest": shoulders_vis,
            "waist": shoulders_vis and hips_vis,
            "hips": hips_vis,
            "legs": knees_vis and ankles_vis,
            "proportion": shoulders_vis and hips_vis,
            "harmony": shoulders_vis,
        }

        return {
            "detected": True,
            "scene": scene,
            "scene_display": scene_display,
            "head_ratio": head_ratio,
            "shoulder_hip_ratio": shoulder_hip_ratio,
            "torso_leg_ratio": torso_leg_ratio,
            "shoulder_tilt_deg": shoulder_tilt_deg,
            "shoulder_balance": shoulder_balance,
            "neck_offset_ratio": neck_offset_ratio,
            "posture_note": posture_note,
            "parts_visibility": parts_vis,
        }
    except Exception as exc:
        logger.warning("Pose landmark evaluation failed: %s", exc)
        return {
            "detected": False,
            "error": f"姿态分析异常: {type(exc).__name__}",
        }


def pose_context_for_prompt(pose_res: dict) -> str:
    """Generate structured pose and geometric proportion context for Codex body prompt."""
    if not pose_res or not pose_res.get("detected"):
        return ""

    scene = pose_res["scene"]
    scene_display = pose_res["scene_display"]
    parts = pose_res.get("parts_visibility", {})

    lines = [
        f"\n【MediaPipe Pose 骨骼与体态几何实测】：",
        f"1. 构图与姿态判定：scene 确认为 {scene}（{scene_display}）。",
    ]

    missing = [k for k, v in parts.items() if not v and k in ("chest", "waist", "hips", "legs")]
    if missing:
        names = {"chest": "肩胸", "waist": "腰腹", "hips": "臀胯", "legs": "腿部"}
        missing_names = "、".join(names.get(m, m) for m in missing)
        lines.append(f"2. 截断/不可见部位声明：以下部位在画面中严重缺失或截断：{missing_names}。根据规则，这些维度的 score 必须严格填 null，status 填 not_assessable 或 insufficient，严禁脑补衣物下不可见形态。")

    if pose_res.get("shoulder_hip_ratio"):
        shr = pose_res["shoulder_hip_ratio"]
        body_type = "平直/微倒三角" if shr >= 1.25 else "匀称/沙漏倾向" if shr >= 1.05 else "下半身骨架略宽"
        lines.append(f"3. 骨架轮廓实测：肩宽/胯宽比 = {shr}（形态倾向：{body_type}）。请以此客观骨骼比值作为腰胯与比例判断的现实依据。")

    if pose_res.get("head_ratio"):
        hr = pose_res["head_ratio"]
        lines.append(f"4. 头身比实测：经垂直骨骼测量约为 {hr} 头身。proportion 维度得分应紧扣此实测比例，避免受机位拍摄角度夸大或贬低。")

    return "\n".join(lines)
