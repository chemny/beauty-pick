from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import provider, storage as s
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
            b_eval = evaluate_face_beauty(crop)
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
            b_eval = evaluate_face_beauty(crop)
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


def evaluate_module(asset: dict, module: str, body_scoring: bool, cache_key: str,
                    cancelled) -> tuple[dict | None, dict | None]:
    if cancelled():
        return None, None
    call_id = None
    try:
        call_id = reserve_call(module)
        with tempfile.TemporaryDirectory(prefix="portrait-input-") as directory:
            path = Path(directory) / "image.jpg"
            if module == "face":
                image = oriented_rgb(s.crop_path(asset["id"]))
            else:
                image = oriented_rgb(s.asset_path(asset["id"]))
            image.thumbnail((1800, 1800))
            save_crop(image, path)
            raw, usage = provider.evaluate(module, path, cancelled, body_scoring=body_scoring) if module == "body" else provider.evaluate(module, path, cancelled)
        assessed = finalize(module, raw)
        if module == "face":
            try:
                from .beauty_model import evaluate_face_beauty, evaluate_face_geometry
                crop_img = oriented_rgb(s.crop_path(asset["id"]))
                b_eval = evaluate_face_beauty(crop_img)
                geom = evaluate_face_geometry(crop_img)
                if b_eval.get("available"):
                    assessed["visual_benchmark"] = {
                        **b_eval,
                        "three_parts_ratio": geom.get("three_parts_ratio"),
                        "eye_spacing_ratio": geom.get("eye_spacing_ratio"),
                        "face_shape_label": geom.get("face_shape_label"),
                        "nose_ratio_label": geom.get("nose_ratio_label"),
                        "nose_interocular_ratio": geom.get("nose_interocular_ratio"),
                        "face_lw_ratio": geom.get("face_lw_ratio"),
                        "has_catchlight": geom.get("has_catchlight"),
                        "is_over_smoothed": geom.get("is_over_smoothed"),
                        "skin_texture_label": geom.get("skin_texture_label"),
                    }
            except Exception:
                pass
        if module == "body":
            try:
                from .pose_model import evaluate_body_pose
                body_img = oriented_rgb(s.asset_path(asset["id"]))
                pose_eval = evaluate_body_pose(body_img)
                if pose_eval.get("detected"):
                    assessed["pose_metrics"] = {
                        "scene": pose_eval["scene"],
                        "scene_display": pose_eval["scene_display"],
                        "head_ratio": pose_eval.get("head_ratio"),
                        "shoulder_hip_ratio": pose_eval.get("shoulder_hip_ratio"),
                        "torso_leg_ratio": pose_eval.get("torso_leg_ratio"),
                        "shoulder_tilt_deg": pose_eval.get("shoulder_tilt_deg"),
                        "shoulder_balance": pose_eval.get("shoulder_balance"),
                    }
                    # Enforce strict not_assessable on physically missing/truncated body parts
                    missing_keys = [k for k, v in pose_eval.get("parts_visibility", {}).items() if not v and k in ("chest", "waist", "hips", "legs")]
                    if missing_keys and body_scoring:
                        changed = False
                        for dim in assessed.get("dimensions", []):
                            if dim["key"] in missing_keys and dim["score"] is not None:
                                dim["score"] = None
                                dim["status"] = "insufficient"
                                dim["limitations"].append("骨骼实测表明该部位未入镜或被严重截断")
                                changed = True
                        if changed:
                            from .scoring import body_total_score
                            scores = {d["key"]: d["score"] for d in assessed["dimensions"]}
                            w = assessed["weights"]
                            assessed["total"] = body_total_score(scores, w)
                            assessed_keys = [k for k, v in scores.items() if v is not None]
                            w_sum = sum(w[k] for k in assessed_keys)
                            assessed.update(
                                assessed_count=len(assessed_keys),
                                assessed_dimensions=assessed_keys,
                                excluded_dimensions=[k for k, v in scores.items() if v is None],
                                effective_weights={k: round(w[k] * 100 / w_sum, 4) for k in assessed_keys} if w_sum else {},
                                scoring_scope="full" if len(assessed_keys) == len(scores) else "visible_only",
                            )
            except Exception:
                pass
        if module == "photography":
            try:
                from .photo_model import evaluate_photo_metrics
                photo_img = oriented_rgb(s.asset_path(asset["id"]))
                target = json.loads(asset["target_json"]) if asset.get("target_json") else None
                p_eval = evaluate_photo_metrics(photo_img, target)
                if p_eval.get("available"):
                    assessed["photo_metrics"] = {
                        "benchmark_score": p_eval["benchmark_score"],
                        "exposure_label": p_eval["exposure"]["label"],
                        "face_exposure_label": p_eval["exposure"].get("face_exposure_label"),
                        "light_softness": p_eval["exposure"].get("light_softness"),
                        "clarity_label": p_eval["clarity"]["label"],
                        "composition_note": p_eval["composition"]["note"],
                        "headroom_label": p_eval["composition"].get("headroom_label"),
                        "dof_label": p_eval["composition"]["dof_label"],
                        "catchlight_label": p_eval["composition"].get("catchlight_label"),
                    }
            except Exception:
                pass
        if module == "body" and not body_scoring:
            assessed["total"] = None
            assessed["status"] = "suggestions_only"
            assessed.update(assessed_count=0, assessed_dimensions=[], effective_weights={}, excluded_dimensions=[d['key'] for d in assessed['dimensions']])
            for dim in assessed["dimensions"]:
                dim["score"] = None
                dim["status"] = "not_assessable"
        assessed["cache_key"] = cache_key
        assessed["model"] = provider.OPENAI_MODEL
        assessed["reasoning_effort"] = provider.OPENAI_REASONING_EFFORT
        s.run("UPDATE model_calls SET status='completed',usage_json=? WHERE id=?", (json.dumps(usage), call_id))
        return assessed, usage
    except Exception as exc:
        reason = str(exc)[:300] if isinstance(exc, (provider.ProviderError, ValueError)) else "处理失败（" + type(exc).__name__ + "）"
        if call_id:
            s.run("UPDATE model_calls SET status='failed' WHERE id=?", (call_id,))
        return skipped(module, reason, "failed"), None


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
    report = json.loads(asset["report_json"])
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
                if not report["face_gate"]["assessable"]:
                    report[module] = skipped(module, "；".join(report["face_gate"]["limitations"]), "not_assessable")
                    continue
            cache_input = f"{asset['sha256']}:{asset['target_json'] if module == 'face' else ''}:{module}:{prompt_version(module)}:{rubric_version(module)}:{provider.OPENAI_MODEL}:{provider.OPENAI_REASONING_EFFORT}"
            if module == "body":
                cache_input += f":scoring={body_scoring}"
            cache_key = hashlib.sha256(cache_input.encode()).hexdigest()
            if not force and report.get(module, {}).get("cache_key") == cache_key:
                report[module]["cache_reused"] = True
                continue
            pending[MODEL_EXECUTOR.submit(evaluate_module, asset, module, body_scoring, cache_key, cancelled)] = module
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
        payload = json.dumps(report, ensure_ascii=False)
        with s.LOCK, s.db() as con:
            if cancelled():
                return
            con.execute("INSERT INTO assessments VALUES(?,?,?,?)", (report["id"], asset["id"], report["created_at"], payload))
            con.execute("UPDATE assets SET report_json=?,status=? WHERE id=? AND deleted_at IS NULL", (payload, status, asset["id"]))
            con.execute("UPDATE jobs SET status=?,updated_at=?,result_json=? WHERE id=?", (status, s.now(), json.dumps({"report_id": report["id"]}), job_id))
    except Exception as exc:
        s.run("UPDATE jobs SET status='failed',updated_at=?,error=? WHERE id=? AND cancelled=0",
              (s.now(), "处理失败，请稍后重试（" + type(exc).__name__ + "）", job_id))
    finally:
        s.run("UPDATE jobs SET status='cancelled',updated_at=? WHERE id=? AND status='running'", (s.now(), job_id))
