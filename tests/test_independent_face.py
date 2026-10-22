import copy
import json

from app.independent_face_trial import FACETS, evaluate, render_section, statistics


def fixture(count=8):
    samples = []; measured = {}; refs = {}
    for i in range(count):
        aid = str(i)
        samples.append({"id": aid, "label": f"图{i}", "report_json": json.dumps({"face": {"dimensions": []}})})
        measured[aid] = {"available": True, "features": {k: 1.0 + i * .1 + j * .03
            for j, k in enumerate(dict.fromkeys(k for keys in FACETS.values() for k in keys))}}
        refs[aid] = {"results": {"face": {"status": "completed", "total": 20.0 + i,
            "dimensions": [{"key": k, "score": 5.0 + i * .4, "status": "assessable"} for k in FACETS]}}}
    return samples, measured, refs


def test_total_raw_and_rank_never_affect_any_facet():
    samples, measured, refs = fixture()
    first = evaluate(samples, measured, refs)
    for aid in measured:
        measured[aid]["features"]["beauty_raw"] = 999.0
        measured[aid]["total"] = -999.0
        refs[aid]["results"]["face"]["total"] = 100.0
        refs[aid]["rank"] = 100
    assert evaluate(samples, measured, refs) == first
    assert all(o["total"] is None for o in first.values())
    assert all("beauty_raw" not in d["head"]["features"] for o in first.values() for d in o["dimensions"])


def test_held_out_label_does_not_leak_and_other_facets_do_not_drive_eyes():
    samples, measured, refs = fixture()
    first = evaluate(samples, measured, refs)
    changed = copy.deepcopy(refs)
    for d in changed["0"]["results"]["face"]["dimensions"]:
        d["score"] = .1
    assert evaluate(samples, measured, changed)["0"] == first["0"]
    for ref in refs.values():
        for d in ref["results"]["face"]["dimensions"]:
            if d["key"] != "eyes":
                d["score"] = .2
    second = evaluate(samples, measured, refs)
    for aid in first:
        assert second[aid]["dimensions"][0] == first[aid]["dimensions"][0]
        assert all(aid not in d["head"]["training_ids"] for d in first[aid]["dimensions"])


def test_missing_partial_inputs_and_detection_failures_never_get_imputed_scores():
    samples, measured, refs = fixture()
    measured["0"]["features"]["eye_spacing"] = None
    measured["1"] = {"available": False, "features": {}, "error": "未检出"}
    result = evaluate(samples, measured, refs)
    assert result["0"]["dimensions"][0]["score"] is None
    assert result["0"]["dimensions"][1]["score"] is not None
    assert all(d["score"] is None for d in result["1"]["dimensions"])


def test_insufficient_and_constant_features_do_not_return_default_scores():
    samples, measured, refs = fixture(4)
    assert all(d["score"] is None for o in evaluate(samples, measured, refs).values() for d in o["dimensions"])
    samples, measured, refs = fixture()
    for m in measured.values():
        m["features"] = {k: 1.0 for k in m["features"]}
    assert all(d["score"] is None for o in evaluate(samples, measured, refs).values() for d in o["dimensions"])


def test_report_keeps_existing_total_rank_crop_and_only_uses_factual_evidence():
    samples, measured, refs = fixture(1)
    outcomes = evaluate(samples, measured, refs)
    source = "<section id='face-previous'><table><tbody><tr><td>photo</td><td>图0<small>原值 5</small></td><td>80</td><td>81</td><td>82</td><td class='revised'>99.0<small>#1</small></td><td><img src='original-face.png'></td></tr></tbody></table></section>"
    page = render_section(source, samples, outcomes, refs)
    assert "99.0<small>#1</small>" in page and "original-face.png" in page
    assert page.count("<b>—</b>") == 5
    assert "独立分项" in page
    assert "黄金对称" not in page and "唇弓饱满" not in page and "毛孔细腻" not in page
    assert "mean_baseline_mae" in statistics(outcomes, refs)["eyes"]
