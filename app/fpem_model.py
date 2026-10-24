"""Offline overall facial attractiveness scoring with the verified FPEM checkpoint."""
from __future__ import annotations
import math
import os
import threading
from pathlib import Path
from .config import ROOT

MODEL_PATH = ROOT / 'models/fpem.pth'
CHECKPOINT_SHA256 = '5e18fb2a807366bd1f4bf8498493437059c60d4862a2f5b54ab664cc92af9615'
METHOD = 'fpem-face-v1'
RUBRIC_VERSION = 'fpem-face-1.0'
PROMPT_VERSION = 'local-fpem-no-prompt'
_MODEL = None
_FINGERPRINT = None
_LOCK = threading.RLock()


def selected_engine() -> str:
    name = os.environ.get('BEAUTYPICK_FACE_MODEL', 'fpem')
    if name not in ('fpem', 'resnet50'):
        raise ValueError('BEAUTYPICK_FACE_MODEL 仅支持 fpem 或 resnet50')
    return name


def device_name() -> str:
    import torch
    requested = os.environ.get('BEAUTYPICK_FPEM_DEVICE', 'auto')
    if requested not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('不支持的 FPEM 设备')
    if requested == 'auto':
        return 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('当前机器不支持 MPS')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前机器不支持 CUDA')
    return requested


def _get_model():
    global _MODEL, _FINGERPRINT
    from .engine import file_fingerprint
    fingerprint = file_fingerprint(MODEL_PATH)
    if fingerprint != CHECKPOINT_SHA256:
        raise ValueError('FPEM 权重缺失或校验失败，请检查 models/fpem.pth')
    device = device_name()
    cache = (fingerprint, device)
    with _LOCK:
        if _MODEL is not None and _FINGERPRINT == cache:
            return _MODEL, device
        import torch
        from .fpem_vendor.clip.model import build_model
        from .fpem_vendor.Clips import FPEM
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        state = {k.removeprefix('FPEM_add.'): v for k, v in checkpoint['state_dict'].items()}
        clip_state = {k.removeprefix('FACLIP.clip.'): v for k, v in state.items() if k.startswith('FACLIP.clip.')}
        clip_model = build_model(clip_state).float()
        model = FPEM(clip_model=clip_model, device='cpu').float()
        matched = model.load_state_dict(state, strict=True)
        if matched.missing_keys or matched.unexpected_keys or len(state) != 1242:
            raise ValueError('FPEM 权重与模型结构不匹配')
        model.eval().to(device)
        model.device = torch.device(device)
        model.FACLIP.device = torch.device(device)
        model.texts = model.texts.to(device)
        model.FACLIP.texts = model.FACLIP.texts.to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if file_fingerprint(MODEL_PATH) != fingerprint:
            raise ValueError('FPEM 权重在加载期间变化')
        _MODEL, _FINGERPRINT = model, cache
        return model, device


def infer(image) -> dict:
    if image.size != (224, 224):
        raise ValueError('FPEM 需要统一的224×224标准面部裁脸')
    import torch
    from torchvision.transforms.functional import to_tensor, resize
    # Serialize shared device work, including cold-start shader initialization.
    with _LOCK:
        model, device = _get_model()
        tensor = to_tensor(image.convert('RGB')).unsqueeze(0)
        inputs = tuple(resize(tensor, [size, size], antialias=True).to(device) for size in (224, 112, 160))
        with torch.inference_mode():
            raw = float(model(inputs)[0].item())
    if not math.isfinite(raw) or not 1 <= raw <= 5:
        raise ValueError('FPEM 没有返回有效颜值分')
    return {'status': 'completed', 'raw_score': raw, 'total': round(raw * 20, 1)}


def assessment(raw: dict) -> dict:
    value = raw.get('raw_score')
    if (raw.get('status') != 'completed' or type(value) not in (int, float)
            or not math.isfinite(value) or not 1 <= value <= 5):
        raise ValueError('FPEM 评分未完成或原始分无效')
    total = round(value * 20, 1)
    if raw.get('total') != total:
        raise ValueError('FPEM 总分与原始分不一致')
    return {'module': 'face', 'status': 'completed', 'total': total, 'raw_score': value,
            'dimensions': [], 'weights': {}, 'scoring_method': METHOD,
            'rubric_version': RUBRIC_VERSION, 'prompt_version': PROMPT_VERSION,
            'summary': '', 'strengths': [], 'limitations': [], 'shooting_tips': [], 'editing_tips': [],
            'display_mapping': 'raw_1_to_5_times_20', 'explanations_only': False}
