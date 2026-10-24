"""Round-one regressions; all persistence uses temporary databases and images."""
import copy
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import beauty_model, config, engine, pipeline, pose_model, provider, storage
from app.main import app
from app.scoring import RULES, finalize


def output(module, score=7.2):
    raw = dict(dimensions=[dict(key=k, status="assessable", score=score,
                               evidence="测试可见依据", limitations=[]) for k, *_ in RULES[module]],
               summary="测试结果", strengths=[], limitations=[], shooting_tips=[], editing_tips=[])
    if module == "body":
        raw["scene"] = "half_body"
    return raw


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "detect_faces", lambda image: [])
    monkeypatch.setattr(beauty_model, "evaluate_face_beauty", lambda image: {"available": False})
    storage.init_db()
    image = Image.new("RGB", (400, 500), (160, 120, 95))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    aid = storage.new_id()
    storage.insert_asset(asset_id=aid, project_id=None, filename="test.png", mime="image/png",
                         width=400, height=500, size=len(buf.getvalue()),
                         sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
                         adult_confirmed=True, rights_confirmed=True)
    storage.asset_path(aid).write_bytes(buf.getvalue())
    report = pipeline.local_report(storage.get_asset(aid))
    target = dict(x=.1, y=.1, width=.7, height=.6)
    report.update(target=target, face_gate={"assessable": True, "limitations": []})
    image.save(storage.crop_path(aid), "JPEG")
    storage.run("UPDATE assets SET target_json=?,report_json=?,external_consent=1 WHERE id=?",
                (json.dumps(target), json.dumps(report), aid))
    identity = dict(provider="test-local", model="test-weights", reasoning_effort=None,
                    weights={"test.pt": "hash1"}, algorithm_version="v1", preprocessing_version="v1")
    monkeypatch.setattr(provider, "engine_identity", lambda module: copy.deepcopy(identity))
    monkeypatch.setattr(pipeline, "face_assessability", lambda *a: {"assessable": True, "limitations": []})
    calls = []

    def evaluate(module, path, cancelled, **kwargs):
        calls.append(module)
        return output(module), {"engine": copy.deepcopy(identity), "total_tokens": 0}

    monkeypatch.setattr(provider, "evaluate", evaluate)

    def run(modules=("photography",), force=False):
        jid = storage.new_id()
        storage.run("INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (jid, aid, "queued", json.dumps(dict(modules=list(modules), force=force)), storage.now(), storage.now()))
        pipeline.run_job(jid)
        return storage.one("SELECT * FROM jobs WHERE id=?", (jid,))

    return SimpleNamespace(aid=aid, path=storage.asset_path(aid), identity=identity, calls=calls, run=run)


def current(sample):
    return json.loads(storage.get_asset(sample.aid)["report_json"])


def attempt(job):
    rid = json.loads(job["result_json"])["report_id"]
    return json.loads(storage.one("SELECT payload FROM assessments WHERE id=?", (rid,))["payload"])


@pytest.mark.parametrize("failure", ["weights_missing", "inference_error", "landmarks_missing", "skin_missing"])
def test_face_never_substitutes_75(sample, monkeypatch, failure):
    import app.images as images
    beauty = {"available": True, "benchmark_score": 82.0}
    geom = {"available": True}
    profile = {"median_lightness_L_star": 60, "median_yellow_blue_b_star": 18}
    if failure in ("weights_missing", "inference_error"):
        beauty = {"available": False, "error": failure}
    elif failure == "landmarks_missing":
        geom = {"available": False, "error": "没有面部关键点"}
    else:
        profile = None
    monkeypatch.setattr(beauty_model, "evaluate_face_beauty", lambda image: beauty)
    monkeypatch.setattr(beauty_model, "evaluate_face_geometry", lambda image: geom)
    monkeypatch.setattr(images, "face_color_profile", lambda image: profile)
    with pytest.raises(provider.ProviderError):
        provider.fast_objective_evaluate("face", sample.path)


@pytest.mark.parametrize("module", ["body", "photography"])
def test_failed_measurement_cannot_generate_scores(sample, monkeypatch, module):
    from app import photo_model
    if module == "body":
        monkeypatch.setattr(pose_model, "evaluate_body_pose", lambda im: {"detected": False, "error": "模型加载失败"})
    else:
        monkeypatch.setattr(photo_model, "evaluate_photo_metrics", lambda im: {"available": False, "error": "测量失败"})
    with pytest.raises(provider.ProviderError, match="失败"):
        provider.fast_objective_evaluate(module, sample.path)


def test_no_body_detected_is_zero_not_model_failure(sample, monkeypatch):
    monkeypatch.setattr(pose_model, "evaluate_body_pose", lambda im: {"detected": False, "scene": "head_only"})
    raw, _ = provider.fast_objective_evaluate("body", sample.path)
    raw.pop("_measurements")
    result = finalize("body", raw)
    assert result["total"] == 0
    assert result["status"] == "completed"
    assert result["score_basis"] == "no_visible_body"
    assert result["summary"] == ""
    assert all(d["score"] is None for d in result["dimensions"])


def test_reports_usage_and_status_record_real_engine(sample):
    job = sample.run()
    result = attempt(job)["photography"]
    assert result["engine"] == sample.identity
    assert result["provider"] == "test-local"
    assert result["model"] == "test-weights"
    assert result["reasoning_effort"] is None
    assert result["input_sha256"] == result["evaluation_context"]["input_sha256"]
    assert len(result["input_sha256"]) == 64
    assert result["elapsed_seconds"] >= 0
    usage = json.loads(storage.one("SELECT usage_json FROM model_calls")["usage_json"])
    assert usage["engine"] == sample.identity
    with TestClient(app) as client:
        client.get("/api/session")
        status = client.get("/api/status").json()
        assert status["provider"] == engine.ENGINE_ID
        assert status["model"] == engine.MODEL_NAME
        assert status["reasoning_effort"] is None


@pytest.mark.parametrize("change", ["model", "weights", "algorithm_version", "preprocessing_version", "implementation", "provider"])
def test_engine_changes_invalidate_cache(sample, change):
    first = attempt(sample.run())["photography"]
    sample.run()
    assert len(sample.calls) == 1
    sample.identity[change] = {"new": "hash2"} if change in ("weights", "implementation") else "new-version"
    new = attempt(sample.run())["photography"]
    assert len(sample.calls) == 2
    assert first["cache_key"] != new["cache_key"]
    assert new["cache_reused"] is False


def test_force_bypasses_cache_and_cache_preserves_original_timestamp(sample):
    first = attempt(sample.run())["photography"]
    cached_report = attempt(sample.run())
    cached = cached_report["photography"]
    assert cached["cache_reused"] is True
    assert cached["evaluated_at"] == first["evaluated_at"]
    assert cached_report["usage"] == []
    new = attempt(sample.run(force=True))["photography"]
    assert sample.calls == ["photography", "photography"]
    assert new["cache_reused"] is False


def test_crop_and_input_changes_invalidate_cache(sample):
    first = attempt(sample.run(("face",)))["face"]
    Image.new("RGB", (200, 240), "blue").save(storage.crop_path(sample.aid), "JPEG")
    new = attempt(sample.run(("face",)))["face"]
    assert first["cache_key"] != new["cache_key"]
    assert first["input_sha256"] != new["input_sha256"]
    assert sample.calls == ["face", "face"]


def test_changed_original_rejected_without_overwriting_report(sample):
    sample.run()
    previous = storage.get_asset(sample.aid)["report_json"]
    Image.new("RGB", (400, 500), "blue").save(sample.path, "PNG")
    job = sample.run(force=True)
    assert job["status"] == "failed" and "原图已变化" in job["error"]
    assert storage.get_asset(sample.aid)["report_json"] == previous
    assert len(sample.calls) == 1


def test_failure_keeps_old_report_and_saves_separate_failed_attempt(sample, monkeypatch):
    first_job = sample.run()
    previous = storage.get_asset(sample.aid)["report_json"]
    def fail(*args, **kwargs):
        raise provider.ProviderError("测试模型超时")
    monkeypatch.setattr(provider, "evaluate", fail)
    failed_job = sample.run(force=True)
    assert failed_job["status"] == "failed"
    assert "测试模型超时" in failed_job["error"]
    assert storage.get_asset(sample.aid)["report_json"] == previous
    failed = attempt(failed_job)
    assert failed["id"] != attempt(first_job)["id"]
    assert failed["photography"]["total"] is None
    assert failed["photography"]["dimensions"] == []
    assert failed["photography"]["attempted_engine"] == sample.identity
    assert len(storage.many("SELECT * FROM assessments WHERE asset_id=?", (sample.aid,))) == 2
    public = storage.public_asset(storage.get_asset(sample.aid))
    assert public["latest_attempt"]["status"] == "failed"
    assert public["latest_attempt"]["result"]["report_id"] == failed["id"]
    with TestClient(app) as client:
        client.get("/api/session")
        response = client.get(f"/api/assets/{sample.aid}/export.json", params={"report_id": failed["id"]})
        assert response.json()["photography"]["total"] is None
        assert client.get(f"/api/assets/{sample.aid}/export.json").json() == json.loads(previous)


def test_partial_failure_never_copies_previous_score_into_new_attempt(sample, monkeypatch):
    sample.run(("face", "body", "photography"))
    previous = storage.get_asset(sample.aid)["report_json"]
    def evaluate(module, path, cancelled, **kwargs):
        if module == "body":
            raise provider.ProviderError("身材模型异常")
        return output(module, 8.1), {"engine": copy.deepcopy(sample.identity)}
    monkeypatch.setattr(provider, "evaluate", evaluate)
    job = sample.run(("face", "body", "photography"), force=True)
    new = attempt(job)
    assert job["status"] == "partial"
    assert new["body"]["status"] == "failed" and new["body"]["total"] is None
    assert new["face"]["total"] == new["photography"]["total"] == 81.0
    old = storage.one("SELECT payload FROM assessments WHERE id=?", (json.loads(previous)["id"],))["payload"]
    assert old == previous


def test_legacy_current_report_archived_before_replacement(sample):
    report = current(sample)
    report["photography"] = finalize("photography", output("photography"))
    previous = json.dumps(report)
    storage.run("UPDATE assets SET report_json=? WHERE id=?", (previous, sample.aid))
    sample.run(force=True)
    archived = storage.one("SELECT payload FROM assessments WHERE id=?", (report["id"],))
    assert json.loads(archived["payload"]) == json.loads(previous)


def test_cancelled_late_result_cannot_replace_previous_report(sample, monkeypatch):
    sample.run()
    previous = storage.get_asset(sample.aid)["report_json"]
    count = len(storage.many("SELECT * FROM assessments"))
    def late_result(module, path, cancelled, **kwargs):
        storage.run("UPDATE jobs SET cancelled=1,status='cancelled' WHERE asset_id=? AND status='running'", (sample.aid,))
        return output(module, 9), {"engine": copy.deepcopy(sample.identity)}
    monkeypatch.setattr(provider, "evaluate", late_result)
    job = sample.run(force=True)
    assert job["status"] == "cancelled"
    assert storage.get_asset(sample.aid)["report_json"] == previous
    assert len(storage.many("SELECT * FROM assessments")) == count
    assert storage.many("SELECT * FROM model_calls WHERE status='cancelled'")


def test_mismatched_execution_identity_rejected(sample, monkeypatch):
    def wrong_engine(module, path, cancelled, **kwargs):
        return output(module), {"engine": {**sample.identity, "model": "wrong-model"}}
    monkeypatch.setattr(provider, "evaluate", wrong_engine)
    job = sample.run()
    assert job["status"] == "failed"
    assert attempt(job)["photography"]["total"] is None


def test_failed_result_is_not_reused_as_cache(sample, monkeypatch):
    sample.run()
    report = current(sample)
    report["photography"]["status"] = "failed"
    storage.run("UPDATE assets SET report_json=? WHERE id=?", (json.dumps(report), sample.aid))
    sample.run()
    assert len(sample.calls) == 2


@pytest.mark.parametrize("kind", ["beauty", "face_landmarker", "pose"])
def test_missing_weights_cannot_reuse_loaded_model(tmp_path, monkeypatch, kind):
    missing = tmp_path / "missing.weights"
    if kind == "beauty":
        monkeypatch.setattr(beauty_model, "MODEL_PATH", missing)
        monkeypatch.setattr(beauty_model, "_MODEL", object())
        assert beauty_model._get_model() == (None, None, None)
    elif kind == "face_landmarker":
        monkeypatch.setattr(beauty_model, "FACE_LANDMARKER_PATH", missing)
        monkeypatch.setattr(beauty_model, "_FACE_LANDMARKER", object())
        assert beauty_model._get_face_landmarker() is None
    else:
        monkeypatch.setattr(pose_model, "MODEL_PATH", missing)
        monkeypatch.setattr(pose_model, "_LANDMARKER", object())
        assert pose_model._get_landmarker() is None


def test_real_weight_content_is_part_of_identity(tmp_path, monkeypatch):
    weights = tmp_path / "test.pt"
    weights.write_bytes(b"first weights")
    from app import fpem_model
    monkeypatch.setattr(fpem_model, "MODEL_PATH", weights)
    before = engine.identity("face")
    weights.write_bytes(b"other weights")
    after = engine.identity("face")
    assert before["weights"] != after["weights"]
    assert engine.signature(before) != engine.signature(after)


def test_unimplemented_provider_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(provider, "PREFERRED_PROVIDER", "codex")
    assert provider.available() is False
    with pytest.raises(provider.ProviderError):
        provider.engine_identity("face")


def test_new_local_reports_can_be_ranked_without_reasoning_effort(sample):
    from app.comparison import assessment
    sample.run(("photography",))
    result, error = assessment(storage.get_asset(sample.aid), "photography", {k: w for k, _, w, _ in RULES["photography"]})
    assert error is None
    assert result["score"] == 72.0


def test_changed_loaded_code_requires_restart(monkeypatch):
    monkeypatch.setattr(engine, "_IMPLEMENTATION", {})
    with pytest.raises(RuntimeError, match="重启"):
        engine.identity("photography")


def test_prompt_and_rubric_changes_invalidate_cache(sample, monkeypatch):
    sample.run()
    monkeypatch.setattr(pipeline, "prompt_version", lambda module: "new-prompt")
    sample.run()
    monkeypatch.setattr(pipeline, "rubric_version", lambda module: "new-rubric")
    sample.run()
    assert len(sample.calls) == 3


def test_body_scoring_mode_is_part_of_cache(sample):
    asset = storage.get_asset(sample.aid)
    scored, _ = pipeline.evaluation_context(asset, "body", True)
    suggestions, _ = pipeline.evaluation_context(asset, "body", False)
    assert engine.signature(scored) != engine.signature(suggestions)


def test_engine_change_during_evaluation_rejects_result(sample, monkeypatch):
    def changed_engine(module, path, cancelled, **kwargs):
        original = copy.deepcopy(sample.identity)
        sample.identity["weights"] = {"test.pt": "replaced"}
        return output(module), {"engine": original}
    monkeypatch.setattr(provider, "evaluate", changed_engine)
    job = sample.run()
    assert job["status"] == "failed"
    assert attempt(job)["photography"]["total"] is None


def test_invalid_schema_never_overwrites_previous_report(sample, monkeypatch):
    sample.run()
    previous = storage.get_asset(sample.aid)["report_json"]
    monkeypatch.setattr(provider, "evaluate", lambda *args, **kwargs: ({"dimensions": []}, {"engine": sample.identity}))
    job = sample.run(force=True)
    assert job["status"] == "failed"
    assert storage.get_asset(sample.aid)["report_json"] == previous
    assert attempt(job)["photography"]["total"] is None


def test_restart_interrupts_job_without_replacing_previous_scores(sample):
    sample.run()
    previous = storage.get_asset(sample.aid)["report_json"]
    storage.run("INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                ("interrupted", sample.aid, "running", '["photography"]', storage.now(), storage.now()))
    storage.init_db()
    assert storage.one("SELECT status FROM jobs WHERE id='interrupted'")["status"] == "interrupted"
    assert storage.get_asset(sample.aid)["report_json"] == previous


def test_local_photography_smoke_records_real_identity(tmp_path, monkeypatch):
    """Exercise actual local CV, schema and identity without external calls or user images."""
    monkeypatch.setenv("BEAUTYPICK_PHOTO_MODEL", "cv")
    import numpy as np
    path = tmp_path / "synthetic.jpg"
    rng = np.random.default_rng(42)
    Image.fromarray(rng.integers(40, 220, (256, 192, 3), dtype=np.uint8)).save(path)
    raw, usage = provider.evaluate("photography", path, lambda: False)
    measurements = raw.pop("_measurements")
    result = finalize("photography", raw)
    assert result["total"] is not None
    assert measurements["photo_metrics"]["benchmark_score"] > 0
    assert usage["provider"] == engine.ENGINE_ID
    assert usage["engine"] == engine.identity("photography")
    assert usage["total_tokens"] == 0


def test_reanalysis_refreshes_saved_gate(sample, monkeypatch):
    report = current(sample)
    report["face_gate"] = {"assessable": False, "limitations": ["old gate"]}
    storage.run("UPDATE assets SET report_json=? WHERE id=?", (json.dumps(report), sample.aid))
    job = sample.run(("face",), force=True)
    assert attempt(job)["face"]["status"] == "completed"
    assert current(sample)["face_gate"]["assessable"] is True


def test_face_gate_shared_minimum_preserves_blur_check():
    import numpy as np
    from app.images import face_assessability, MIN_FACE_PIXELS
    rng = np.random.default_rng(42)
    crop = Image.fromarray(rng.integers(0, 256, (170,170,3), dtype=np.uint8))
    assert face_assessability({"pixel_width":89,"pixel_height":89}, crop)["assessable"]
    # An 83px face produces a 158px intermediate crop; FPEM later resizes to224.
    small = Image.fromarray(rng.integers(0, 256, (158,158,3), dtype=np.uint8))
    assert face_assessability({"pixel_width":83,"pixel_height":83}, small)["assessable"]
    assert not face_assessability({"pixel_width":MIN_FACE_PIXELS-1,"pixel_height":89}, crop)["assessable"]
    assert not face_assessability({"pixel_width":89,"pixel_height":89}, Image.new("RGB",(170,170)))["assessable"]


def test_no_body_zero_is_rankable(sample, monkeypatch):
    from app.comparison import assessment
    monkeypatch.setattr(pose_model, "evaluate_body_pose", lambda im: {"detected": False, "scene": "head_only"})
    raw, _ = provider.fast_objective_evaluate("body", sample.path)
    raw.pop("_measurements")
    result = finalize("body", raw)
    result["engine"] = sample.identity
    asset = {"id": sample.aid, "filename": "test.png", "adult_confirmed": True,
             "report_json": json.dumps({"id": "test", "body": result})}
    item, error = assessment(asset, "body", {r[0]: r[2] for r in RULES["body"]})
    assert error is None and item["score"] == 0


def test_face_device_inference_serialized(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    import torch
    active = peak = 0
    guard = threading.Lock()

    def forward(tensor):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(.02)
        with guard:
            active -= 1
        return tensor

    model = SimpleNamespace(forward_features=forward, global_pool=lambda x: x,
                            fc=lambda x: torch.tensor(4.0))
    monkeypatch.setattr(beauty_model, "_get_model", lambda: (model, "cpu", lambda im: torch.zeros(3,224,224)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: beauty_model.evaluate_face_beauty(Image.new("RGB",(224,224))), range(4)))
    assert all(r["available"] for r in results)
    assert peak == 1
