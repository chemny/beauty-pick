# 本地模型与信任资料

获取日期：2026-10-14。

- `blaze_face_short_range.tflite`：https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
  - SHA-256：`b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`
- `c2pa-trust-anchors.pem`：https://contentcredentials.org/trust/anchors.pem
  - SHA-256：`db57e50b9bd4e4c786cbb91f38fd85e2af6a9d5412438a8cddcc42688df07ae5`

BlazeFace随MediaPipe使用；发布或再分发前应检查Google模型页和对应许可。信任锚来自Content Credentials公开信任列表，仅作有日期的离线快照；更新需重新记录校验值，不自动联网更新。

## FPEM (default overall face scorer)

- File: `models/fpem.pth`, 582,528,073 bytes.
- Upstream: https://github.com/Estella-LH/FPEM (commit c2965425247d7bf8b764d27e4483a06fc7a061e5).
- Download the actual Git LFS payload, not its text pointer: https://media.githubusercontent.com/media/Estella-LH/FPEM/c2965425247d7bf8b764d27e4483a06fc7a061e5/pretrained/fpem_srcc_0.9243.pth
- SHA-256: `5e18fb2a807366bd1f4bf8498493437059c60d4862a2f5b54ab664cc92af9615`.
- Complete checkpoint contains CLIP and all face/fusion branches. Inference never downloads additional weights; missing/mismatched files fail explicitly.
- FP32 runs on Mac MPS or CPU (also CUDA); `BEAUTYPICK_FPEM_DEVICE=cpu` explicitly selects CPU.
- Vendored inference architecture/tokenizer and license notices: `app/fpem_vendor/`. No trial-directory dependencies are used by the backend.
- Keep `scut_fbp5500_beauty.pt` for an explicit rollback: restart with `BEAUTYPICK_FACE_MODEL=resnet50`. No automatic fallback or rewriting historical scores.

## TOPIQ-IAA 摄影评分（默认）

`topiq-iaa-res50.pth` 为已完整加载验证的 AVA ResNet50 检查点，293,931,049 bytes，SHA-256 `3cd62bb33f9933ed7c6e3d5e79129e81c898eba78b7a2af516a0b0b974616975`。来源：IQA-PyTorch `cfanet_iaa_ava_res50-3cd62bb3.pth`，正式路径独立于试验目录。CPU离线运行，不额外下载骨干。权重未纳入Git。实现许可见 `app/topiq_vendor/LICENSE`，适用边界见 `docs/runtime-install.md`。

### 当前身材引擎（2026-10-24）

FASHN本地权重位于`models/fashn-human-parser/`，MediaPipe Heavy位于`models/pose_landmarker_heavy.task`；隔离依赖位于`models/body-runtime/deps/`。评分使用本机`qwen3.5:2b-mlx`，不依赖Codex。模型、隔离依赖不提交Git。详见[正式接入说明](../docs/runtime-install.md)，不要把隔离版transformers/huggingface-hub装进主应用环境。
