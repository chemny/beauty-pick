import io
import json

from PIL import Image
import pytest

from app import pipeline, provider
from app.standard_face import assessment


def test_selected_total_is_not_reconstructed_from_facets():
    r = assessment({"status": "completed", "raw_score": 5.564, "total": 99.0})
    assert r["total"] == 99 and r["display_clipped"]
    assert r["dimensions"] == [] and r["weights"] == {}
    assert assessment({"status": "completed", "raw_score": 3.764, "total": 71.0})["total"] == 71
    with pytest.raises(ValueError): assessment({"status": "completed", "raw_score": 3.764, "total": 99})


def test_provider_selected_face_bypasses_legacy_facets(tmp_path, monkeypatch):
    identity = {"face_scoring_method": "standard-face-v1", "provider": "local", "model": "ResNet50"}
    monkeypatch.setattr(provider, "engine_identity", lambda _: identity)
    monkeypatch.setattr(provider, "fast_objective_evaluate", lambda *a, **kw: pytest.fail("legacy facets called"))
    from app import standard_face
    monkeypatch.setattr(standard_face, "infer", lambda im: {"status": "completed", "raw_score": 3.764, "total": 71.0})
    p = tmp_path / "face.png"; Image.new("RGB", (224, 224)).save(p)
    raw, usage = provider.evaluate("face", p, lambda: False)
    assert assessment(raw)["total"] == 71 and usage["engine"] == identity


def test_context_uses_original_and_lossless_standard_input(tmp_path, monkeypatch):
    from app import standard_face
    source = tmp_path / "original.png"; Image.new("RGB", (400, 600)).save(source)
    from app.engine import file_fingerprint
    monkeypatch.setattr(pipeline.s, "asset_path", lambda _: source)
    monkeypatch.setattr(pipeline.s, "crop_path", lambda _: pytest.fail("old crop used"))
    monkeypatch.setattr(provider, "engine_identity", lambda _: {"face_scoring_method": "standard-face-v1"})
    pixels = Image.new("RGB", (224, 224), (21, 73, 119))
    monkeypatch.setattr(standard_face, "prepare", lambda im, target: (pixels, {"margin": .12}))
    asset = {"id": "test", "sha256": file_fingerprint(source), "target_json": json.dumps({"x": .2})}
    context, data = pipeline.evaluation_context(asset, "face", True)
    assert data.startswith(b"\x89PNG") and context["input_format"] == "PNG"
    with Image.open(io.BytesIO(data)) as actual:
        assert actual.tobytes() == pixels.tobytes()


def test_overall_only_assessment_is_rankable_and_not_reweighted():
    from app.comparison import assessment as comparable
    from app.scoring import RULES
    result = assessment({'status':'completed','raw_score':3.764,'total':71.0})
    result.update(engine={'provider':'local','model':'SCUT-FBP5500-ResNet50'})
    asset = {'id':'test','filename':'test.png','adult_confirmed':True,
             'report_json':json.dumps({'id':'report','face':result})}
    weights = {r[0]:r[2] for r in RULES['face']}
    item, error = comparable(asset, 'face', weights)
    assert error is None and item['score']==71.0
    weights.update(eyes=30, skin=0)
    assert comparable(asset, 'face', weights)[0] is None
