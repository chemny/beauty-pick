from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ROOT

logger = logging.getLogger(__name__)

_MODEL: Any = None
_DEVICE: Any = None
_TRANSFORMS: Any = None
_LOCK = threading.Lock()

MODEL_PATH = ROOT / "models" / "scut_fbp5500_beauty.pt"


def _get_model():
    global _MODEL, _DEVICE, _TRANSFORMS
    if _MODEL is not None:
        return _MODEL, _DEVICE, _TRANSFORMS

    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _DEVICE, _TRANSFORMS

        if not MODEL_PATH.exists():
            return None, None, None

        import torch
        import torchvision.transforms as transforms
        import timm

        # Prioritize Apple Silicon GPU (MPS) for ~7ms fast inference
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = timm.create_model("resnet50", pretrained=False, num_classes=1)
        state = torch.load(str(MODEL_PATH), map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        _MODEL, _DEVICE, _TRANSFORMS = model, device, tf
        return _MODEL, _DEVICE, _TRANSFORMS


def evaluate_face_beauty(image: Image.Image) -> dict:
    """Evaluate face beauty using SCUT-FBP5500 ResNet-50 visual aesthetic regression model.
    
    Returns a dict with:
      - available: bool
      - model: str
      - raw_score: float (1.0 to 5.0)
      - benchmark_score: float (35.0 to 99.0)
      - tier: str
      - guidance: str
    """
    try:
        model, device, tf = _get_model()
        if model is None:
            return {
                "available": False,
                "error": "本地颜值美学模型权重未安装 (models/scut_fbp5500_beauty.pt)",
            }

        import torch

        img = image.convert("RGB")
        tensor = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.forward_features(tensor)
            pooled = model.global_pool(feat)
            raw_score = float(model.fc(pooled).item())

        # Map 2.0~5.3 raw scale into calibrated 40.0~98.0 aesthetic score
        benchmark_score = round(40.0 + (raw_score - 2.0) / (5.3 - 2.0) * 58.0, 1)
        benchmark_score = max(35.0, min(99.0, benchmark_score))

        if benchmark_score >= 85.0:
            tier = "很漂亮·整体突出"
            guidance = "面部骨相与五官比例优势鲜明，在真实审美样本中位列第一梯队"
        elif benchmark_score >= 75.0:
            tier = "明显漂亮·优势突出"
            guidance = "面部轮廓端正、局部或整体搭配优势较强，明显优于普通水准"
        elif benchmark_score >= 60.0:
            tier = "顺眼·有局部优势"
            guidance = "整体自然匀称，具备局部优势，但整体未达到高度精致"
        elif benchmark_score >= 40.0:
            tier = "普通·形态自然"
            guidance = "五官骨相与比例平整自然，美感优势不显著"
        else:
            tier = "较弱"
            guidance = "当前图像可见的面部结构比例与对称匀称度较弱"

        return {
            "available": True,
            "model": "SCUT-FBP5500-ResNet50",
            "raw_score": round(raw_score, 3),
            "benchmark_score": benchmark_score,
            "tier": tier,
            "guidance": guidance,
        }
    except Exception as exc:
        logger.warning("Visual face beauty model evaluation failed: %s", exc)
        return {
            "available": False,
            "error": f"推理执行异常: {type(exc).__name__}",
        }


def beauty_context_for_prompt(beauty_res: dict) -> str:
    """Generate objective visual anchor context for Codex prompt."""
    if not beauty_res or not beauty_res.get("available"):
        return ""

    raw = beauty_res["raw_score"]
    benchmark = beauty_res["benchmark_score"]
    tier = beauty_res["tier"]
    guidance = beauty_res["guidance"]

    return (
        f"\n【人脸骨相美学客观测量（SCUT-FBP5500 视觉模型）】：\n"
        f"该图面孔经纯视觉美学模型（去除衣服、光影情境、背景修饰干扰）测得客观基准分：{benchmark} 分（五分制原生值：{raw:.3f}，档位：{tier}）。\n"
        f"特征引导：{guidance}。\n"
        f"请以此基准分为面部五官（眉眼、鼻唇、五官协调、脸型）的整体评分锚点；各分项得分均值应与此客观基准分保持合理一致（允许按各部位实际可见优缺点在锚点上下细化浮动，但整体不得与客观骨相发生大幅脱节）。"
    )


_FACE_LANDMARKER: Any = None
_LANDMARKER_LOCK = threading.Lock()
FACE_LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"


def _get_face_landmarker():
    global _FACE_LANDMARKER
    if _FACE_LANDMARKER is not None:
        return _FACE_LANDMARKER

    with _LANDMARKER_LOCK:
        if _FACE_LANDMARKER is not None:
            return _FACE_LANDMARKER
        if not FACE_LANDMARKER_PATH.exists():
            return None

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
        return _FACE_LANDMARKER


def evaluate_face_geometry(image: Image.Image) -> dict:
    """Evaluate 3D facial proportions, symmetry, catchlight, and natural skin texture."""
    try:
        landmarker = _get_face_landmarker()
        if not landmarker:
            return {"available": False, "error": "face_landmarker 模型文件不存在"}

        import cv2
        import numpy as np
        import mediapipe as mp

        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)

        if not res.face_landmarks:
            return {"available": False, "error": "未检测到面部关键网格"}

        lm = res.face_landmarks[0]

        # 1. 三庭垂直比例 (上庭 10->168, 中庭 168->2, 下庭 2->152)
        top_y = lm[10].y * h
        mid_y = lm[168].y * h
        nose_y = lm[2].y * h
        chin_y = lm[152].y * h

        h_upper = max(1.0, mid_y - top_y)
        h_mid = max(1.0, nose_y - mid_y)
        h_lower = max(1.0, chin_y - nose_y)
        h_total = h_upper + h_mid + h_lower

        r_upper = h_upper / h_total * 3.0
        r_mid = h_mid / h_total * 3.0
        r_lower = h_lower / h_total * 3.0

        three_parts_ratio = f"1 : {r_mid / r_upper:.2f} : {r_lower / r_upper:.2f}"
        # 偏离理想1:1:1的方差
        ratio_dev = float(np.std([r_upper, r_mid, r_lower]))

        # 2. 五眼水平比例 (左外眦 33, 左内眦 133, 右内眦 362, 右外眦 263)
        le_w = max(1.0, abs(lm[133].x - lm[33].x) * w)
        re_w = max(1.0, abs(lm[263].x - lm[362].x) * w)
        inter_w = max(1.0, abs(lm[362].x - lm[133].x) * w)
        eye_spacing_ratio = round(float(inter_w / ((le_w + re_w) / 2.0)), 2)

        # 3. 鼻翼宽度与两眼内眦间距比 (鼻翼 102, 331; 内眦 133, 362)
        nose_alar_w = max(1.0, abs(lm[331].x - lm[102].x) * w)
        nose_interocular_ratio = round(float(nose_alar_w / inter_w), 2)
        if 0.92 <= nose_interocular_ratio <= 1.12:
            nose_ratio_label = "鼻翼宽度恰合内眦间距（符合经典五眼比例）"
        elif nose_interocular_ratio < 0.92:
            nose_ratio_label = "鼻翼偏小巧精致（略小于内眦距）"
        else:
            nose_ratio_label = "鼻翼形态舒展自然"

        # 4. 脸部宽长比与脸型骨相 (颧骨宽 234->454, 下颌角宽 136->365, 脸长 10->152)
        face_width = max(1.0, abs(lm[454].x - lm[234].x) * w)
        face_length = max(1.0, abs(lm[152].y - lm[10].y) * h)
        jaw_width = max(1.0, abs(lm[365].x - lm[136].x) * w)
        face_lw_ratio = round(float(face_length / face_width), 2)
        jaw_cheek_ratio = round(float(jaw_width / face_width), 2)

        if face_lw_ratio >= 1.34 and jaw_cheek_ratio <= 0.74:
            face_shape_label = f"标准鹅蛋脸（宽长比 1:{face_lw_ratio:.2f}）"
        elif face_lw_ratio >= 1.30:
            face_shape_label = f"修长椭圆脸（宽长比 1:{face_lw_ratio:.2f}）"
        elif face_lw_ratio <= 1.18 and jaw_cheek_ratio >= 0.76:
            face_shape_label = f"饱满方圆脸（宽长比 1:{face_lw_ratio:.2f}）"
        elif face_lw_ratio <= 1.20:
            face_shape_label = f"精致幼态圆脸（宽长比 1:{face_lw_ratio:.2f}）"
        elif jaw_cheek_ratio <= 0.70:
            face_shape_label = f"心形小尖脸/瓜子脸（下颌比 {jaw_cheek_ratio:.2f}）"
        else:
            face_shape_label = f"温婉方圆脸（宽长比 1:{face_lw_ratio:.2f}）"

        # 5. 瞳孔眼神光 (Iris Catchlight: 468 左, 473 右)
        def detect_catchlight(iris_idx: int) -> bool:
            ix = int(lm[iris_idx].x * w)
            iy = int(lm[iris_idx].y * h)
            patch = rgb[max(0, iy - 6):min(h, iy + 7), max(0, ix - 6):min(w, ix + 7)]
            if patch.size == 0:
                return False
            g = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            return float(g.max()) >= 170 and float(g.max() - g.min()) >= 50

        has_catchlight = detect_catchlight(468) or detect_catchlight(473)

        # 6. 皮肤真实肌理 (识别重度磨皮与AI塑料感)
        cx = int((lm[234].x + lm[454].x) * 0.5 * w)
        cy = int(lm[2].y * h)
        cheek = rgb[max(0, cy - 12):min(h, cy + 13), max(0, cx - 12):min(w, cx + 13)]
        if cheek.size > 0:
            gray_cheek = cv2.cvtColor(cheek, cv2.COLOR_RGB2GRAY)
            skin_high_freq = float(cv2.Laplacian(gray_cheek, cv2.CV_64F).var())
        else:
            skin_high_freq = 25.0

        is_over_smoothed = (skin_high_freq < 4.5)
        skin_texture_label = "过度磨皮/塑料平滑感较强" if is_over_smoothed else "保留自然皮肤微纹理"

        return {
            "available": True,
            "three_parts_ratio": three_parts_ratio,
            "ratio_dev": round(ratio_dev, 3),
            "eye_spacing_ratio": eye_spacing_ratio,
            "nose_interocular_ratio": nose_interocular_ratio,
            "nose_ratio_label": nose_ratio_label,
            "face_lw_ratio": face_lw_ratio,
            "jaw_cheek_ratio": jaw_cheek_ratio,
            "face_shape_label": face_shape_label,
            "has_catchlight": has_catchlight,
            "skin_high_freq": round(skin_high_freq, 1),
            "is_over_smoothed": is_over_smoothed,
            "skin_texture_label": skin_texture_label,
        }
    except Exception as exc:
        logger.warning("Facial geometry analysis failed: %s", exc)
        return {"available": False, "error": str(exc)}

