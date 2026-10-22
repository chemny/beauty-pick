import copy
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from PIL import Image

from app import holistic_provider as provider
from app.holistic_scoring import MODULES, PARTS, finalize, prompt, schema, validate_overall
from app.holistic_trial import compare, select_samples, snapshot, reports, write_image
from app.provider import ProviderError


def overall():
    return {m: {"status": "assessable", "score": 83.1, "confidence": "medium",
                "scope": {"face": "face", "body": "half_body", "photography": "whole_photo"}[m],
                "evidence": "整体搭配有可见优势"} for m in MODULES} | {
        "observation": {"framing": "半身", "scene": "half_body"}}


def explanations():
    return {m: {"dimensions": [{"key": key, "status": "assessable", "score": 7.0,
                                 "evidence": "具体可见局部特征", "limitations": []} for key, *_ in PARTS[m]],
                "strengths": ["有具体优势"], "improvements": [], "limitations": []} for m in MODULES}


def test_overall_is_preserved_and_not_computed_from_facets():
    raw = overall()
    result = finalize(raw, explanations())
    for module in MODULES:
        assert result[module]["total"] == 83.1
        assert result[module]["weights"] == {}
        assert result[module]["explanations_only"] is True
    changed = explanations()
    for m in MODULES:
        for d in changed[m]["dimensions"]:
            d["score"] = 6.1
    assert finalize(raw, changed)["face"]["total"] == 83.1
    assert raw == overall()


def test_incomplete_facet_does_not_erase_assessable_overall():
    parts = explanations()
    for module in ("face", "body"):
        parts[module]["dimensions"][0].update(status="insufficient", score=None, evidence="该局部被遮挡")
    result = finalize(overall(), parts)
    assert result["face"]["total"] == result["body"]["total"] == 83.1
    assert result["body"]["assessed_count"] == 5
    assert result["body"]["excluded_dimensions"] == ["chest"]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 100.1, -1, 8.31])
def test_overall_rejects_invalid_numbers(value):
    raw = overall(); raw["face"]["score"] = value
    with pytest.raises(ValueError):
        validate_overall(raw)


def test_status_and_scope_must_match_total():
    raw = overall(); raw["face"]["score"] = None
    with pytest.raises(ValueError):
        validate_overall(raw)
    raw = overall(); raw["face"]["scope"] = "whole_photo"
    with pytest.raises(ValueError):
        validate_overall(raw)


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "null_with_score", "precision"])
def test_explanation_contract(change):
    parts = explanations()
    dims = parts["face"]["dimensions"]
    if change == "duplicate":
        dims[0]["key"] = dims[1]["key"]
    elif change == "missing":
        dims.pop()
    elif change == "extra":
        dims.append({**dims[0], "key": "expression"})
    elif change == "null_with_score":
        dims[0]["status"] = "insufficient"
    else:
        dims[0]["score"] = 7.01
    with pytest.raises(ValueError):
        finalize(overall(), parts)


def test_head_only_cannot_generate_body_score():
    raw = overall(); raw["observation"]["scene"] = "head_only"
    with pytest.raises(ValueError):
        validate_overall(raw)
    raw["body"].update(status="not_assessable", score=None, scope="unavailable")
    parts = explanations()
    for dim in parts["body"]["dimensions"]:
        dim.update(status="not_assessable", score=None, evidence="未入镜")
    result = finalize(raw, parts)
    assert result["body"]["total"] is None
    assert result["body"]["status"] == "not_assessable"


def test_contradiction_is_flagged_without_repairing_scores():
    raw = overall(); raw["face"]["score"] = 40.1
    parts = explanations()
    for d in parts["face"]["dimensions"]:
        d["score"] = 9.0
    result = finalize(raw, parts)["face"]
    assert result["status"] == "review_required"
    assert result["total"] == 40.1
    assert all(d["score"] == 9 for d in result["dimensions"])


def test_prompts_have_no_old_scores_and_explanation_has_no_overall_input():
    first, second = prompt("overall", True), prompt("explanations", True)
    assert "不要计算、生成或平均任何分项" in first
    assert "你不知道第一阶段总分" in second
    assert "weight" not in second
    for generated_schema in (schema("overall"), schema("explanations")):
        assert generated_schema["additionalProperties"] is False
        assert set(generated_schema["required"]) == set(generated_schema["properties"])
    assert "score" not in schema("explanations")["properties"]


def test_runtime_is_ephemeral_and_restricts_tools(tmp_path):
    args = provider.command("codex", tmp_path, "explanations", [tmp_path / "image.jpg"])
    assert "--ignore-user-config" in args and "--ephemeral" in args
    assert "--no-daemon" in args
    assert "resume" not in args and "fork" not in args
    assert "read-only" in args and "never" in args
    assert "shell_tool" in args and "skip_host_skill_discovery" in args


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (20, 20), "gray").save(path)
    monkeypatch.setattr(provider, "identity", lambda: {"model": "test-model"})
    script = tmp_path / "child.py"
    def configure(code):
        script.write_text(code)
        monkeypatch.setattr(provider, "command", lambda executable, directory, phase, images:
                            [sys.executable, str(script), str(directory)])
    return path, configure


def test_runtime_reads_its_own_result_and_usage(fake_runtime):
    path, configure = fake_runtime
    configure("import sys,pathlib,json\nsys.stdin.read()\np=pathlib.Path(sys.argv[1]); (p/'result.json').write_text('{\"own_result\": true}')\nprint(json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':4}}))\n")
    raw, usage = provider.evaluate("overall", path, None)
    assert raw == {"own_result": True}
    assert usage["input_tokens"] == 12 and usage["engine"]["model"] == "test-model"
    assert len(usage["input_hashes"]) == 1


def test_runtime_timeout_terminates_without_result(fake_runtime):
    path, configure = fake_runtime
    configure("import sys,time\nsys.stdin.read()\ntime.sleep(5)\n")
    with pytest.raises(ProviderError, match="超过"):
        provider.evaluate("overall", path, None, timeout=.1)


def test_runtime_rejects_tool_execution(fake_runtime):
    path, configure = fake_runtime
    configure("import sys,pathlib,json\nsys.stdin.read()\n(pathlib.Path(sys.argv[1])/'result.json').write_text('{}')\nprint(json.dumps({'type':'item.completed','item':{'type':'command_execution'}}))\n")
    with pytest.raises(ProviderError, match="工具调用"):
        provider.evaluate("overall", path, None)


def test_runtime_failure_never_uses_existing_report(fake_runtime):
    path, configure = fake_runtime
    configure("import sys,json\nsys.stdin.read()\nprint(json.dumps({'type':'error','message':'test unavailable'}))\nsys.exit(1)\n")
    with pytest.raises(ProviderError, match="test unavailable"):
        provider.evaluate("overall", path, None)


def test_comparison_uses_raw_overall_and_sequential_subset_ranks():
    def sample(aid, score):
        return {"id": aid, "label": aid, "filename": aid + '.png', "created_at": aid,
                "report_json": json.dumps({m: {"total": score} for m in MODULES})}
    samples = [sample("a", 90), sample("b", 80), sample("c", 80)]
    outcomes = {aid: {"status": "completed", "results": {m: {"total": value, "status": "completed"} for m in MODULES}}
                for aid, value in (("a", 70), ("b", 88), ("c", 88))}
    before = copy.deepcopy(samples)
    rows = compare(samples, outcomes)["rows"]
    assert [r["scores"]["face"]["new_rank"] for r in rows] == [3, 1, 2]
    assert rows[0]["scores"]["face"]["delta"] == -20
    assert rows[1]["scores"]["face"]["new"] == rows[2]["scores"]["face"]["new"] == 88
    assert samples == before


def test_trial_snapshot_is_read_only_and_detects_result_changes(tmp_path):
    database = tmp_path / "production.db"
    with sqlite3.connect(database) as con:
        for table in ("assets", "assessments", "jobs", "model_calls"):
            con.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, payload TEXT)")
            con.execute(f"INSERT INTO {table} VALUES ('old', 'preserve')")
    before = database.read_bytes()
    baseline = snapshot(database)
    assert database.read_bytes() == before
    assert snapshot(database)["fingerprints"] == baseline["fingerprints"]
    with sqlite3.connect(database) as con:
        con.execute("UPDATE assessments SET payload='changed'")
    after = snapshot(database)
    assert after["fingerprints"]["assets"] == baseline["fingerprints"]["assets"]
    assert after["fingerprints"]["assessments"] != baseline["fingerprints"]["assessments"]


def test_sample_selection_requires_existing_adult_and_full_image_authorization():
    from app.config import ROOT
    from app.holistic_trial import EXTRA_SAMPLES
    fixtures = json.loads((ROOT / "tests/fixtures/representatives.json").read_text())["samples"]
    assets = [{"id": item["id"], "sha256": item["sha256"], "deleted_at": None,
               "adult_confirmed": 1, "external_consent": 1, "external_scope": "full_image"} for item in fixtures]
    assets.extend({"id": aid, "sha256": aid, "deleted_at": None, "adult_confirmed": 1,
                   "external_consent": 1, "external_scope": "full_image"} for aid, _ in EXTRA_SAMPLES)
    assert len(select_samples(assets)) == 12
    assets[0]["external_scope"] = "face_only"
    with pytest.raises(ValueError, match="授权"):
        select_samples(assets)


def test_report_does_not_read_incomplete_pending_preview(tmp_path):
    sample = {"id": "pending", "label": "等待评分", "filename": "test.png", "created_at": "now",
              "report_json": json.dumps({m: {"total": 80} for m in MODULES})}
    directory = tmp_path / "pending"; directory.mkdir()
    (directory / "whole.jpg").write_bytes(b"")  # Other worker has not published a complete image yet.
    baseline = {"samples": [sample], "engine": {"model": "test", "reasoning_effort": "medium"}}
    reports(tmp_path, baseline, {}, {"unchanged": True})
    assert "等待评分" in (tmp_path / "comparison.html").read_text()
    assert '<img alt=' not in (tmp_path / "comparison.html").read_text()


def test_trial_preview_is_written_as_complete_jpeg(tmp_path):
    path = tmp_path / "whole.jpg"
    write_image(path, Image.new("RGB", (80, 100), "gray"))
    with Image.open(path) as im:
        assert im.size == (80, 100)
    assert not path.with_suffix('.jpg.tmp').exists()
