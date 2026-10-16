#!/usr/bin/env python3
"""Check and download required vision models and assets for BeautyPick."""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

MODELS = [
    {
        "filename": "blaze_face_short_range.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "required": True,
        "description": "MediaPipe BlazeFace 短距离人脸检测模型",
    },
    {
        "filename": "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "required": True,
        "description": "MediaPipe 468 人脸关键点与面部几何模型",
    },
    {
        "filename": "pose_landmarker_full.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "required": True,
        "description": "MediaPipe 33 姿态骨骼关键点模型",
    },
    {
        "filename": "c2pa-trust-anchors.pem",
        "url": "https://verify.contentauthenticity.org/trust/anchors.pem",
        "required": True,
        "description": "C2PA / CAI 数字内容凭证根证书信任列表",
    },
    {
        "filename": "scut_fbp5500_beauty.pt",
        "url": None,  # Provided in release assets or local copy
        "required": False,
        "description": "SCUT-FBP5500 ResNet50 骨相客观美学基准模型 (约 90MB)",
    },
]


def download_file(url: str, dest: Path) -> None:
    print(f"  正在下载 {dest.name} 从 {url} ...")
    tmp = dest.with_suffix(".tmp")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BeautyPick-Model-Downloader/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(tmp, "wb") as out_file:
            total_size = response.getheader("Content-Length")
            total = int(total_size) if total_size else 0
            downloaded = 0
            block_size = 65536
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total:
                    pct = downloaded * 100 // total
                    sys.stdout.write(f"\r  进度: {pct}% ({downloaded // 1024} KB / {total // 1024} KB)")
                    sys.stdout.flush()
            print()
        tmp.replace(dest)
        print(f"  ✓ {dest.name} 下载完成")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"  ✗ 下载失败: {e}")
        raise


def ensure_models() -> bool:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    print("正在检查本地模型与信任资料...")
    for item in MODELS:
        fn = item["filename"]
        target = MODELS_DIR / fn
        desc = item["description"]
        if target.exists() and target.stat().st_size > 0:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  ✓ [{size_mb:.1f}MB] {fn} ({desc})")
        else:
            if item["url"]:
                try:
                    download_file(item["url"], target)
                except Exception:
                    all_ok = False
            else:
                if item["required"]:
                    print(f"  ✗ 缺少必要模型: {fn} ({desc})")
                    all_ok = False
                else:
                    print(f"  ℹ 可选模型暂未放入: {fn} ({desc}) - 系统将自动降级使用几何与微观肌理特征")

    return all_ok


if __name__ == "__main__":
    success = ensure_models()
    sys.exit(0 if success else 1)
