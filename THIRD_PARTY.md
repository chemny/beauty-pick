# Third-party components and model sources

The root MIT license covers BeautyPick's own code only. The notices shipped with each vendored component remain authoritative; model weights are downloaded separately and are not relicensed by BeautyPick.

| Component | Source and shipped notice | Distribution boundary |
| --- | --- | --- |
| FPEM | https://github.com/Estella-LH/FPEM ; `app/fpem_vendor/LICENSE-FPEM`, `PROVENANCE.json` | Preserve the upstream MIT notice; checkpoint source/revision/hash are in `models/manifest.json`. |
| OpenAI CLIP | https://github.com/openai/CLIP ; `app/fpem_vendor/LICENSE-CLIP` | Preserve upstream MIT notice. |
| facenet-pytorch | https://github.com/timesler/facenet-pytorch ; `app/fpem_vendor/LICENSE-FACENET` | Preserve upstream MIT notice. |
| TOPIQ / IQA-PyTorch | https://github.com/chaofengc/IQA-PyTorch ; `app/topiq_vendor/LICENSE` | Bundled implementation notice is PolyForm Noncommercial 1.0.0. Root MIT does not override it; do not represent this component as unrestricted commercial software. |
| FASHN Human Parser / SegFormer | https://huggingface.co/fashn-ai/fashn-human-parser ; `app/body_vendor/FASHN_LICENSE`, `NVIDIA_LICENSE` | Preserve upstream FASHN and NVIDIA source license conditions; model card specifies nvidia-segformer. |
| MediaPipe models | https://ai.google.dev/edge/mediapipe/solutions/guide | Upstream Google model assets and their applicable terms, not BeautyPick-owned weights. Fixed source URLs and hashes are listed in the manifest. |
| Qwen 3.5 via Ollama | https://ollama.com/library/qwen3.5:2b-mlx | Download separately through Ollama. Upstream model page includes Apache 2.0 notice; no model blobs bundled. |
| Content Credentials trust anchors | https://contentcredentials.org/ | Existing bundled dated certificate snapshot is checksum-verified, not silently replaced with a newer trust store. |

This inventory describes the delivery, not a grant of rights to datasets, photos, likenesses, or model training material. Full license texts take precedence over this table. Keep notices when distributing the vendored sources. Experimental InsightFace and rejected HumanAesExpert models are not required by the production workflow.
