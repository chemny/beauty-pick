"""Offline whole-photo aesthetics with the complete TOPIQ-IAA AVA ResNet50 checkpoint."""
from __future__ import annotations
import math
import os
import threading
from .config import ROOT

MODEL_PATH = ROOT / 'models/topiq-iaa-res50.pth'
CHECKPOINT_SHA256 = '3cd62bb33f9933ed7c6e3d5e79129e81c898eba78b7a2af516a0b0b974616975'
METHOD = 'topiq-iaa-v1'
RUBRIC_VERSION = 'topiq-iaa-1.0'
PROMPT_VERSION = 'local-topiq-no-prompt'
_MODEL = None
_FINGERPRINT = None
_LOCK = threading.RLock()


def selected_engine():
    name = os.environ.get('BEAUTYPICK_PHOTO_MODEL', 'topiq')
    if name not in ('topiq', 'cv'):
        raise ValueError('BEAUTYPICK_PHOTO_MODEL 仅支持 topiq 或 cv')
    return name


def _get_model():
    global _MODEL, _FINGERPRINT
    from .engine import file_fingerprint
    fingerprint = file_fingerprint(MODEL_PATH)
    if fingerprint != CHECKPOINT_SHA256:
        raise ValueError('TOPIQ-IAA 权重缺失或校验失败')
    with _LOCK:
        if _MODEL is not None and _FINGERPRINT == fingerprint:
            return _MODEL
        import torch
        from .topiq_vendor.topiq import CFANet
        model = CFANet(semantic_model_name='resnet50', model_name='cfanet_iaa_ava_res50',
                       use_ref=False, inter_dim=512, num_heads=8, num_class=10,
                       test_img_size=384, backbone_pretrain=False, pretrained=False)
        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)['params']
        model.load_state_dict(state, strict=True)
        model.eval().to('cpu')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if file_fingerprint(MODEL_PATH) != fingerprint:
            raise ValueError('TOPIQ-IAA 权重在加载期间变化')
        _MODEL, _FINGERPRINT = model, fingerprint
        return model


def infer(image):
    import torch
    from torchvision.transforms.functional import to_tensor
    with _LOCK:
        model = _get_model()
        tensor = to_tensor(image.convert('RGB')).unsqueeze(0)
        with torch.inference_mode():
            raw = float(model(tensor).item())
    if not math.isfinite(raw) or not 1 <= raw <= 10:
        raise ValueError('TOPIQ-IAA 没有返回有效摄影分')
    return {'status': 'completed', 'raw_score': raw, 'total': round(raw * 10, 1)}


def assessment(raw):
    value = raw.get('raw_score')
    if (raw.get('status') != 'completed' or type(value) not in (int, float)
            or not math.isfinite(value) or not 1 <= value <= 10
            or raw.get('total') != round(value * 10, 1)):
        raise ValueError('TOPIQ-IAA 评分未完成或原始分无效')
    return {'module': 'photography', 'status': 'completed', 'total': round(value * 10, 1),
            'raw_score': value, 'dimensions': [], 'weights': {}, 'scoring_method': METHOD,
            'rubric_version': RUBRIC_VERSION, 'prompt_version': PROMPT_VERSION,
            'summary': '', 'strengths': [], 'limitations': [], 'shooting_tips': [], 'editing_tips': [],
            'display_mapping': 'raw_1_to_10_times_10', 'explanations_only': False}
