from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import engine, provider, storage as s
from .images import oriented_rgb, detect_faces, quality_metrics, crop_face, save_crop, face_assessability
from .provenance import provenance_report
from .scoring import finalize, skipped, prompt_version, rubric_version

EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="portrait-job")
MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="portrait-model")
DAILY_LIMIT = int(os.environ.get("PORTRAIT_DAILY_CALL_LIMIT", "0"))


def usage_day() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def usage_summary() -> dict:
    rows = s.many("SELECT status,usage_json FROM model_calls WHERE day=?", (usage_day(),))
    tokens = sum((json.loads(r["usage_json"]).get("input_tokens", 0) +
                  json.loads(r["usage_json"]).get("output_tokens", 0))
                 for r in rows if r["usage_json"])
    return {"calls": len(rows), "limit": DAILY_LIMIT if DAILY_LIMIT > 0 else None, "tokens": tokens,
            "day": usage_day(), "timezone": "America/Los_Angeles", "cost": None}


def reserve_call(module: str) -> str:
    with s.LOCK, s.db() as con:
        count = con.execute("SELECT COUNT(*) FROM model_calls WHERE day=?", (usage_day(),)).fetchone()[0]
        if DAILY_LIMIT > 0 and count >= DAILY_LIMIT:
            raise provider.ProviderError("今日调用次数已达上限，明日恢复或由本机管理员调整上限")
        call_id = s.new_id()
        con.execute("INSERT INTO model_calls VALUES(?,?,?,?,?,?)", (call_id, usage_day(), module, s.now(), "running", None))
    return call_id


def local_report(asset: dict, statement: str | None = None) -> dict:
    path = s.asset_path(asset["id"])
    image = oriented_rgb(path)
    faces, detector_error = [], None
    try:
        faces = detect_faces(image)
    except Exception:
        detector_error = "本地人脸检测失败，可稍后重试；不会伪造人脸数量"
    target = faces[0] if len(faces) == 1 else None
    gate = None
    if target:
        crop, transform = crop_face(image, target)
        save_crop(crop, s.crop_path(asset["id"]))
        target = {**target, "transform": transform}
        gate = face_assessability(target, crop)
        try:
            from .beauty_model import evaluate_face_beauty
            from .fpem_model import selected_engine
            b_eval = evaluate_face_beauty(crop) if selected_engine() == "resnet50" else {}
            if b_eval.get("available"):
                target["visual_benchmark"] = b_eval
        except Exception:
            pass
    report = {"schema_version": "1.0", "id": s.new_id(), "asset_id": asset["id"],
              "created_at": s.now(), "input_hash": asset["sha256"],
              "filename": asset["filename"], "width": image.width, "height": image.height,
              "provenance": provenance_report(path, statement), "quality": quality_metrics(image),
              "faces": faces, "detector_error": detector_error, "target": target, "face_gate": gate,
              "face": skipped("face", "尚未启用模型评估"),
              "photography": skipped("photography", "尚未启用模型评估"),
              "body": skipped("body", "体态默认关闭；可主动启用可见姿态分析，必要信息不足时总分留空"), "usage": [],
              "subjective_notice": "分数是指定主观规则下的图像呈现评价，不是客观颜值或真人实际长相。"}
    s.run("UPDATE assets SET target_json=?,report_json=?,status='local_ready' WHERE id=? AND deleted_at IS NULL",
          (json.dumps(target) if target else None, json.dumps(report, ensure_ascii=False), asset["id"]))
    return report


def set_target(asset: dict, target: dict) -> dict:
    with s.LOCK:
        if not s.get_asset(asset["id"]):
            raise ValueError("图片已删除")
        existing = json.loads(asset["target_json"] or "null")
        if existing and all(abs(existing[k] - target[k]) < 0.000001 for k in ("x", "y", "width", "height")):
            return json.loads(asset["report_json"])
        image = oriented_rgb(s.asset_path(asset["id"]))
        target = {**target, "pixel_width": int(target["width"] * image.width),
                  "pixel_height": int(target["height"] * image.height), "selection": "user"}
        crop, transform = crop_face(image, target)
        target["transform"] = transform
        save_crop(crop, s.crop_path(asset["id"]))
        try:
            from .beauty_model import evaluate_face_beauty
            from .fpem_model import selected_engine
            b_eval = evaluate_face_beauty(crop) if selected_engine() == "resnet50" else {}
            if b_eval.get("available"):
                target["visual_benchmark"] = b_eval
        except Exception:
            pass
        report = json.loads(asset["report_json"])
        report.update({"target": target, "face_gate": face_assessability(target, crop),
                       "face": skipped("face", "目标裁剪已改变，请重新评估")})
        s.run("UPDATE jobs SET cancelled=1,status='cancelled' WHERE asset_id=? AND status IN ('queued','running')", (asset["id"],))
        s.run("UPDATE assets SET target_json=?,report_json=? WHERE id=? AND deleted_at IS NULL",
              (json.dumps(target), json.dumps(report, ensure_ascii=False), asset["id"]))
        return report

def submit(asset: dict, modules: list[str], body_scoring: bool = True, force: bool = False) -> str:
    with s.LOCK, s.db() as con:
        if con.execute("SELECT id FROM jobs WHERE asset_id=? AND status IN ('queued','running')", (asset["id"],)).fetchone():
            raise ValueError("这张图片已有分析任务")
        job_id, created = s.new_id(), s.now()
        con.execute("INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (job_id, asset["id"], "queued", json.dumps({"modules": modules, "body_scoring": body_scoring, "force": force}), created, created))
    EXECUTOR.submit(run_job, job_id)
    return job_id


def evaluation_context(asset: dict, module: str, body_scoring: bool) -> tuple[dict, bytes]:
    original_hash = engine.file_fingerprint(s.asset_path(asset["id"]))
    if original_hash != asset["sha256"]:
        raise provider.ProviderError("原图已变化或缺失，请重新上传")
    identity = provider.engine_identity(module)
    body = module == "body" and identity.get("scoring_method") == "visible-body-fixed-zero-v1"
    standard = module == "face" and identity.get("face_scoring_method") in ("standard-face-v1", "fpem-face-v1")
    topiq = module == "photography" and identity.get("scoring_method") == "topiq-iaa-v1"
    if standard:
        from .standard_face import prepare
        image, transform = prepare(oriented_rgb(s.asset_path(asset["id"])), json.loads(asset["target_json"] or "null"))
    else:
        source = s.crop_path(asset["id"]) if module == "face" else s.asset_path(asset["id"])
        image = oriented_rgb(source)
        if not topiq and not body:
            image.thumbnail((1800, 1800))
    output = io.BytesIO()
    if standard or topiq or body:
        image.save(output, "PNG")
    else:
        image.save(output, "JPEG", quality=90, optimize=True)
    data = output.getvalue()
    context = {"cache_schema": 2, "original_sha256": original_hash,
               "input_sha256": hashlib.sha256(data).hexdigest(),
               "target": json.loads(asset["target_json"] or "null"), "module": module,
               "body_scoring": body_scoring if module == "body" else None,
               "engine": identity,
               "prompt_version": identity.get("prompt_version", prompt_version(module)),
               "rubric_version": identity.get("rubric_version", rubric_version(module))}
    if standard:
        context.update(input_format="PNG", standard_face_transform=transform)
    if topiq or body:
        context.update(input_format="PNG")
    return context, data


def evaluate_module(asset: dict, module: str, body_scoring: bool, cache_key: str,
                    cancelled, context: dict, input_bytes: bytes) -> tuple[dict | None, dict | None]:
    if cancelled():
        return None, None
    call_id = None
    started = time.monotonic()
    try:
        call_id = reserve_call(module)
        if provider.engine_identity(module) != context["engine"]:
            raise provider.ProviderError("排队期间模型或算法已变化，请重新分析")
        with tempfile.TemporaryDirectory(prefix="portrait-input-") as directory:
            path = Path(directory) / ("image.png" if context.get("input_format") == "PNG" else "image.jpg")
            path.write_bytes(input_bytes)
            raw, usage = provider.evaluate(module, path, cancelled, body_scoring=body_scoring) if module == "body" else provider.evaluate(module, path, cancelled)
        if cancelled():
            raise provider.ProviderError("用户取消分析")
        if usage.get("engine") != context["engine"] or provider.engine_identity(module) != context["engine"]:
            raise provider.ProviderError("本次执行的引擎版本与请求不一致，请重新分析")
        raw = dict(raw)
        measurements = raw.pop("_measurements", {})
        if module == "face" and context["engine"].get("face_scoring_method") == "fpem-face-v1":
            from .fpem_model import assessment
            assessed = assessment(raw)
        elif module == "face" and context["engine"].get("face_scoring_method") == "standard-face-v1":
            from .standard_face import assessment
            assessed = assessment(raw)
        elif module == "photography" and context["engine"].get("scoring_method") == "topiq-iaa-v1":
            from .topiq_model import assessment
            assessed = assessment(raw)
        elif module == "body" and context["engine"].get("scoring_method") == "visible-body-fixed-zero-v1":
            from .body_model import assessment
            assessed = assessment(raw)
        else:
            assessed = finalize(module, raw)
        for key in ("visual_benchmark", "pose_metrics", "photo_metrics"):
            if key in measurements:
                assessed[key] = measurements[key]
        if module == "body" and not body_scoring:
            assessed["total"] = None
            assessed["status"] = "suggestions_only"
            assessed.update(assessed_count=0, assessed_dimensions=[], effective_weights={}, excluded_dimensions=[d['key'] for d in assessed['dimensions']])
            for dim in assessed["dimensions"]:
                dim["score"] = None
                dim["status"] = "not_assessable"
        assessed.update(cache_key=cache_key, cache_reused=False, evaluation_context=context,
                        engine=context["engine"], provider=context["engine"]["provider"],
                        model=context["engine"]["model"], reasoning_effort=context["engine"]["reasoning_effort"],
                        evaluated_at=s.now(), input_sha256=context["input_sha256"])
        usage = {**usage, "elapsed_seconds": round(time.monotonic() - started, 3)}
        assessed["elapsed_seconds"] = usage["elapsed_seconds"]
        s.run("UPDATE model_calls SET status='completed',usage_json=? WHERE id=?", (json.dumps(usage), call_id))
        return assessed, usage
    except Exception as exc:
        reason = str(exc)[:300] if isinstance(exc, (provider.ProviderError, ValueError)) else "处理失败（" + type(exc).__name__ + "）"
        usage = {"engine": context["engine"], "elapsed_seconds": round(time.monotonic() - started, 3), "error": reason}
        if call_id:
            s.run("UPDATE model_calls SET status=?,usage_json=? WHERE id=?",
                  ("cancelled" if cancelled() else "failed", json.dumps(usage), call_id))
        result = skipped(module, reason, "failed")
        result.update(attempted_engine=context["engine"], input_sha256=context["input_sha256"])
        return result, usage


def run_job(job_id: str) -> None:
    job = s.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job or job["cancelled"]:
        return
    asset = s.get_asset(job["asset_id"])
    if not asset:
        return
    s.run("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (s.now(), job_id))
    options = json.loads(job["modules_json"])
    modules = options if isinstance(options, list) else options["modules"]
    body_scoring = True if isinstance(options, list) else options.get("body_scoring", True)
    force = False if isinstance(options, list) else options.get("force", False)
    previous_report = json.loads(asset["report_json"])
    report = copy.deepcopy(previous_report)
    report["requested_modules"] = modules
    report["job_id"] = job_id
    report["base_report_id"] = previous_report["id"]
    for module in modules:
        report[module] = skipped(module, "本次分析尚未完成")
    report["id"], report["created_at"] = s.new_id(), s.now()
    report["usage"] = []

    def cancelled() -> bool:
        current = s.get_asset(asset["id"])
        state = s.one("SELECT cancelled FROM jobs WHERE id=?", (job_id,))
        return (not current or not state or state["cancelled"] or not current["external_consent"]
                or current["target_json"] != asset["target_json"] or current["expires_at"] < s.now() and not current["retained"])

    try:
        pending = {}
        usages = {}
        for module in modules:
            if cancelled():
                return
            if module in ("face", "body") and not asset["adult_confirmed"]:
                report[module] = skipped(module, "未确认成年人物，不开放外貌评分")
                continue
            if module == "face":
                if not asset["adult_confirmed"]:
                    report[module] = skipped(module, "未确认成年人物，不开放外貌评分")
                    continue
                if not report["target"]:
                    report[module] = skipped(module, "请先选择目标面孔", "not_assessable")
                    continue
            try:
                if module == "face":
                    original = oriented_rgb(s.asset_path(asset["id"]))
                    target = dict(report["target"])
                    target.update(pixel_width=target["width"]*original.width,
                                  pixel_height=target["height"]*original.height)
                    crop, _ = crop_face(original, target)
                    report["face_gate"] = face_assessability(target, crop)
                    if not report["face_gate"]["assessable"]:
                        report[module] = skipped(module, "；".join(report["face_gate"]["limitations"]), "not_assessable")
                        continue
                context, input_bytes = evaluation_context(asset, module, body_scoring)
                cache_key = engine.signature(context)
            except Exception as exc:
                reason = str(exc)[:300] if isinstance(exc, (provider.ProviderError, RuntimeError, ValueError)) else "准备分析失败（" + type(exc).__name__ + "）"
                report[module] = skipped(module, reason, "failed")
                continue
            cached = previous_report.get(module, {})
            if (not force and cached.get("cache_key") == cache_key
                    and cached.get("status") in ("completed", "suggestions_only")
                    and cached.get("engine") == context["engine"]):
                report[module] = copy.deepcopy(cached)
                report[module].update(cache_reused=True, source_report_id=previous_report["id"])
                continue
            pending[MODEL_EXECUTOR.submit(evaluate_module, asset, module, body_scoring, cache_key,
                                          cancelled, context, input_bytes)] = module
        for future in as_completed(pending):
            module = pending[future]
            report[module], usage = future.result()
            if usage:
                usages[module] = usage
        report["usage"] = [usages[module] for module in modules if module in usages]
        if cancelled():
            return
        states = [report[m]["status"] for m in modules]
        status = "failed" if all(state == "failed" for state in states) else "completed" if all(state in ("completed", "suggestions_only") for state in states) else "partial"
        report["attempt_status"] = status
        report["carried_modules"] = [m for m in ("face", "body", "photography") if m not in modules]
        failures = [report[m]["summary"] for m in modules if report[m]["status"] == "failed"]
        payload = json.dumps(report, ensure_ascii=False)
        with s.LOCK, s.db() as con:
            if cancelled():
                return
            # Preserve the previous snapshot, including legacy reports that were only on the asset.
            if any(previous_report.get(m, {}).get("total") is not None for m in ("face", "body", "photography")):
                con.execute("INSERT OR IGNORE INTO assessments VALUES(?,?,?,?)",
                            (previous_report["id"], asset["id"], previous_report["created_at"], json.dumps(previous_report, ensure_ascii=False)))
            con.execute("INSERT INTO assessments VALUES(?,?,?,?)", (report["id"], asset["id"], report["created_at"], payload))
            if status == "failed":
                con.execute("UPDATE assets SET status=? WHERE id=? AND deleted_at IS NULL", (status, asset["id"]))
            else:
                con.execute("UPDATE assets SET report_json=?,status=? WHERE id=? AND deleted_at IS NULL", (payload, status, asset["id"]))
            con.execute("UPDATE jobs SET status=?,updated_at=?,result_json=?,error=? WHERE id=?",
                        (status, s.now(), json.dumps({"report_id": report["id"], "previous_report_id": previous_report["id"]}),
                         "；".join(failures) or None, job_id))
    except Exception as exc:
        s.run("UPDATE jobs SET status='failed',updated_at=?,error=? WHERE id=? AND cancelled=0",
              (s.now(), "处理失败，请稍后重试（" + type(exc).__name__ + "）", job_id))
    finally:
        s.run("UPDATE jobs SET status='cancelled',updated_at=? WHERE id=? AND status='running'", (s.now(), job_id))
