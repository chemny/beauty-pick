from __future__ import annotations

import json
from pathlib import Path

from PIL import ExifTags, Image
from .config import ROOT


SAFE_EXIF = {"Make", "Model", "DateTime", "DateTimeOriginal", "Software", "Orientation", "LensModel", "FNumber", "ExposureTime", "ISOSpeedRatings", "PhotographicSensitivity", "FocalLength", "DateTimeDigitized"}


def metadata(path: Path) -> dict:
    result: dict = {"exif": {}, "limitations": []}
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            values = dict(exif)
            try:
                values.update(exif.get_ifd(34665))
            except (KeyError, TypeError, ValueError):
                pass
            for key, value in values.items():
                name = ExifTags.TAGS.get(key, str(key))
                if name in SAFE_EXIF and (isinstance(value, (str, int, float)) or hasattr(value, "numerator")):
                    result["exif"][name] = str(value)[:200]
    except Exception:
        result["limitations"].append("元数据读取失败")
    if not result["exif"]:
        result["limitations"].append("未发现可展示的 EXIF；这不能证明图片由 AI 生成")
    return result


def inspect_c2pa(path: Path) -> dict:
    result = {"manifest_status": "indeterminate", "signature_status": "unknown",
              "asset_binding_status": "unknown", "signer_trust": "unknown",
              "platform": None, "evidence": [], "limitations": []}
    try:
        import c2pa
    except ImportError:
        result["limitations"].append("C2PA 验证组件未安装")
        return result
    try:
        trust = ROOT / "models" / "c2pa-trust-anchors.pem"
        settings = c2pa.Settings.from_dict({
            "trust": {"trust_anchors": trust.read_text()} if trust.is_file() else {},
            "verify": {"remote_manifest_fetch": False, "ocsp_fetch": False},
            "core": {"allowed_network_hosts": []},
        })
        with c2pa.Context(settings=settings) as context, c2pa.Reader(str(path), context=context) as reader:
            raw = json.loads(reader.json())
    except Exception as exc:
        message = str(exc).lower()
        if "manifestnotfound" in message or "no manifest" in message or "no jumbf data" in message or "manifest does not exist" in message:
            result["manifest_status"] = "absent"
            result["limitations"].append("未发现 C2PA 凭据；不能据此判定图片真实")
        else:
            result["limitations"].append("C2PA 读取失败，无法完成验证")
        return result

    return interpret_manifest(raw, result)


def interpret_manifest(raw: dict, result: dict | None = None) -> dict:
    result = result or {"platform": None, "evidence": [], "limitations": []}
    validation = raw.get("validation_status") or []
    active = raw.get("active_manifest")
    manifest = (raw.get("manifests") or {}).get(active, {})
    checks = (raw.get("validation_results") or {}).get("activeManifest") or {}
    successes = {v.get("code") for v in checks.get("success", [])}
    failures = {v.get("code") for v in checks.get("failure", [])}
    issues = sorted(failures or {v.get("code", "unknown") for v in validation if isinstance(v, dict)})
    result["manifest_status"] = "invalid" if failures else ("valid" if raw.get("validation_state") in ("Valid", "Trusted") else "indeterminate")
    result["signature_status"] = ("invalid" if "claimSignature.mismatch" in failures else
                                  "valid" if "claimSignature.validated" in successes else "unknown")
    result["asset_binding_status"] = ("invalid" if any("hash.mismatch" in s.lower() for s in failures) else
                                      "valid" if "assertion.dataHash.match" in successes else "unknown")
    result["signer_trust"] = ("untrusted" if "signingCredential.untrusted" in failures or "signingCredential.invalid" in failures else "trusted" if "signingCredential.trusted" in successes else "unknown")
    result["limitations"].append("仅离线验证内嵌凭据；不获取远程凭据或在线吊销状态")
    result["evidence"].append({"type": "c2pa_manifest", "level": "parsed",
                               "title": str(manifest.get("title", "内容凭据"))[:160],
                               "claim_generator": str(manifest.get("claim_generator", ""))[:160],
                               "validation_issues": issues[:20]})
    signature = manifest.get("signature_info") or {}
    result["evidence"].append({"type": "signer", "level": "parsed", "title": "声明签发者",
                               "value": str(signature.get("issuer", "未知"))[:200]})
    source_types = []
    software_names = []
    for assertion in manifest.get("assertions", []):
        if not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        data = assertion.get("data") or {}
        result["limitations"].append("动作链可能不完整；不能据此排除未声明的编辑")
        for action in data.get("actions", [])[:40]:
            source = str(action.get("digitalSourceType", ""))
            agent = action.get("softwareAgent") or {}
            software = agent.get("name", "") if isinstance(agent, dict) else str(agent)
            source_types.append(source.rsplit("/", 1)[-1]); software_names.append(software)
            result["evidence"].append({"type": "c2pa_action", "level": "parsed", "title": str(action.get("action", "动作"))[:100],
                                       "value": f"软件声明：{software}；数字来源类型：{source}"[:500]})
    verified = (not failures and result["signer_trust"] == "trusted" and
                result["signature_status"] == "valid" and result["asset_binding_status"] == "valid")
    result["category"] = "unknown"
    if verified:
        generated = any(s in ("trainedAlgorithmicMedia", "algorithmicMedia") for s in source_types)
        edited = any(s in ("compositeWithTrainedAlgorithmicMedia", "algorithmicallyEnhanced") for s in source_types)
        camera = "digitalCapture" in source_types
        result["category"] = "mixed" if camera and (generated or edited) else "ai_generated" if generated else "ai_edited" if edited else "camera" if camera else "unknown"
        if "OpenAI" in str(signature.get("issuer", "")) and any("ChatGPT" in n for n in software_names):
            result["platform"] = "OpenAI / ChatGPT（签名声明）"
        for e in result["evidence"]:
            e["level"] = "verified_claim"
    if issues:
        result["limitations"].append("内容凭据存在验证问题；不能据此进行平台归因")
    else:
        result["limitations"].append("签名、绑定及签发者信任分开验证；可信声明也不证明人物或场景真实")
    return result


def provenance_report(path: Path, user_statement: str | None = None) -> dict:
    meta = metadata(path)
    c2pa_data = inspect_c2pa(path)
    evidence = list(c2pa_data["evidence"])
    for name, value in meta["exif"].items():
        evidence.append({"type": "exif", "level": "unverified", "title": name,
                         "value": value})
    if user_statement:
        evidence.append({"type": "user_statement", "level": "unverified",
                         "title": "用户声明", "value": user_statement[:300]})
    return {
        "category": c2pa_data.get("category", "unknown"), "platform": c2pa_data.get("platform"), "evidence_level": "verified_claim" if any(e["level"] == "verified_claim" for e in evidence) else "unverified" if evidence else "none",
        "manifest_status": c2pa_data["manifest_status"],
        "signature_status": c2pa_data["signature_status"],
        "asset_binding_status": c2pa_data["asset_binding_status"],
        "signer_trust": c2pa_data["signer_trust"],
        "evidence": evidence,
        "limitations": meta["limitations"] + c2pa_data["limitations"] +
                       (["没有可信的生成或拍摄来源证据，具体平台与内容类别无法确认"] if c2pa_data.get("category", "unknown") == "unknown" else ["分类来自验证过的来源声明，不是对图像内容的真实性保证"]),
    }


def source_display(provenance: dict, user_source: str = 'unknown') -> dict:
    """User-facing source summary; preserve cryptographic verification separately."""
    names = {'ai_generated': 'AI 生成', 'ai_edited': 'AI 编辑', 'camera': '相机拍摄', 'mixed': '混合来源'}
    category = provenance.get('category', 'unknown')
    verified = (provenance.get('signature_status') == 'valid' and
                provenance.get('asset_binding_status') == 'valid' and
                provenance.get('signer_trust') == 'trusted')
    actions = [e.get('value', '') for e in provenance.get('evidence', []) if e.get('type') == 'c2pa_action']
    marked_ai = any('trainedAlgorithmicMedia' in a or 'algorithmicMedia' in a or 'compositeWithTrainedAlgorithmicMedia' in a or 'algorithmicallyEnhanced' in a for a in actions)
    marked_camera = any('digitalCapture' in a for a in actions)
    platform = provenance.get('platform') if verified else None
    platform_basis = '已验证' if platform else None
    if not platform and any('软件声明：ChatGPT；' in a for a in actions):
        platform, platform_basis = 'ChatGPT', '文件标记'
    if verified and category in names:
        label, basis = names[category], '来源已验证'
    elif user_source in ('ai', 'camera'):
        label, basis = ('AI 图片' if user_source == 'ai' else '相机拍摄'), '用户确认'
    elif marked_ai:
        label, basis = '含 AI 生成标记', '文件标记'
    elif marked_camera:
        label, basis = '含拍摄标记', '文件标记'
    else:
        label, basis = '无法确认', None
    return {'label': label, 'basis': basis, 'platform': platform, 'platform_basis': platform_basis}
