import copy
import json

import numpy as np
import pytest

from app.local_candidate import ALPHA, FEATURES, MODULES, aligned_body_input, cache_key, fit_head, predict
from app.local_calibration_trial import comparison, cross_validate, render
from app.holistic_scoring import PARTS


def data(count=7):
    samples=[]; measurements={}; refs={}
    for i in range(count):
        aid=f"photo-{i}"
        old={m:{"total":float(65+i)} for m in MODULES}
        samples.append({"id":aid,"created_at":str(i),"label":aid,"report_json":json.dumps(old)})
        measurements[aid]={m:{"available":True,"features":{k:float(i+j*.2) for j,k in enumerate(dict.fromkeys(k for keys in FEATURES[m].values() for k in keys))},
                              "scope":"face" if m=="face" else "half_body" if m=="body" else "whole_photo"} for m in MODULES}
        refs[aid]={"results":{m:{"status":"completed","total":float(75+i),"dimensions":[{"key":k,"score":float(6+i*.1),"status":"assessable"} for k,*_ in PARTS[m]]} for m in MODULES}}
    return samples,measurements,refs


def test_held_out_labels_never_affect_own_predictions_or_normalization():
    samples,measured,refs=data()
    original=cross_validate(samples,measured,refs)
    changed=copy.deepcopy(refs)
    for module in MODULES:
        changed["photo-0"]["results"][module]["total"]=1.0
        for d in changed["photo-0"]["results"][module]["dimensions"]:
            d["score"]=.2
    again=cross_validate(samples,measured,changed)
    assert again["photo-0"]==original["photo-0"]
    for outcome in original.values():
        for module in MODULES:
            for head in outcome["results"][module]["heads"].values():
                assert outcome["held_out_id"] not in head["training_ids"]


def test_overall_does_not_depend_on_explanatory_facet_scores():
    samples,measured,refs=data()
    first=cross_validate(samples,measured,refs)
    for ref in refs.values():
        for module in MODULES:
            for d in ref["results"][module]["dimensions"]:
                d["score"]*=.1
    second=cross_validate(samples,measured,refs)
    for aid in first:
        for module in MODULES:
            assert first[aid]["results"][module]["total"]==second[aid]["results"][module]["total"]
            assert first[aid]["results"][module]["weights"]=={}


def test_no_measurement_failure_fallback_or_invisible_part_score():
    samples,measured,refs=data()
    measured["photo-0"]["face"]={"available":False,"features":{},"error":"模型缺失"}
    measured["photo-0"]["body"]["features"]["leg_torso"]=None
    result=cross_validate(samples,measured,refs)["photo-0"]["results"]
    assert result["face"]["total"] is None
    assert all(d["score"] is None for d in result["face"]["dimensions"])
    assert next(d for d in result["body"]["dimensions"] if d["key"]=="legs")["score"] is None
    assert result["body"]["total"] is not None


def test_all_constant_or_missing_features_cannot_masquerade_as_scores():
    for value in (None,2.0):
        rows=[{"id":str(i),"features":{"x":value},"target":float(i+5)} for i in range(6)]
        with pytest.raises(ValueError,match="可变"):
            fit_head(rows,("x",),upper=10)


def test_too_few_reference_labels_leave_score_empty():
    samples,measured,refs=data(4)
    result=cross_validate(samples,measured,refs)
    assert all(o["results"][m]["total"] is None for o in result.values() for m in MODULES)


def test_training_median_imputation_is_neutral_and_scale_is_not_test_derived():
    rows=[{"id":str(i),"features":{"x":float(i)},"target":float(i+1)} for i in range(8)]
    head=fit_head(rows,("x",),upper=10)
    before=copy.deepcopy(head)
    missing=predict(head,{"x":None})
    center=predict(head,{"x":head["center"][0]})
    assert missing["score"]==center["score"]
    assert missing["missing_features"]==["x"]
    assert head==before and head["alpha"]==ALPHA
    assert predict(head,{"x":1000.})["out_of_distribution"]==["x"]


@pytest.mark.parametrize("value",[True,float("nan"),float("inf"),-1.,101.])
def test_invalid_reference_values_are_not_training_samples(value):
    rows=[{"id":"bad","features":{"x":1.},"target":value}]
    with pytest.raises(ValueError):
        fit_head(rows,("x",),upper=100)


def test_schema_version_input_and_weights_all_invalidate_measurement_cache():
    base=cache_key({"prepared":["image1"],"target":"box1"},{"version":"v1","weights":"hash1"})
    assert base!=cache_key({"prepared":["image2"],"target":"box1"},{"version":"v1","weights":"hash1"})
    assert base!=cache_key({"prepared":["image1"],"target":"box2"},{"version":"v1","weights":"hash1"})
    assert base!=cache_key({"prepared":["image1"],"target":"box1"},{"version":"v2","weights":"hash1"})
    assert base!=cache_key({"prepared":["image1"],"target":"box1"},{"version":"v1","weights":"hash2"})


def test_same_scores_keep_continuous_stable_ranks_without_score_changes():
    samples,measured,refs=data()
    outcomes=cross_validate(samples,measured,refs)
    for outcome in outcomes.values():
        for m in MODULES: outcome["results"][m]["total"]=82.0
    result=comparison(samples,refs,outcomes)
    assert [r["scores"]["face"]["local_rank"] for r in result["rows"]]==list(range(1,8))
    assert all(r["scores"]["face"]["local"]==82.0 for r in result["rows"])
    assert result["reference_is_ground_truth"] is False


def test_report_adds_three_columns_preserves_original_and_no_model_call(tmp_path,monkeypatch):
    from PIL import Image
    import subprocess
    monkeypatch.setattr(subprocess,"Popen",lambda *a,**kw:pytest.fail("报告生成不得调用引擎"))
    samples,measured,refs=data()
    outcomes=cross_validate(samples,measured,refs)
    result=comparison(samples,refs,outcomes)
    result.update(samples=samples,references=refs)
    for s in samples:
        p=tmp_path/s["id"];p.mkdir();Image.new("RGB",(80,100)).save(p/"whole.jpg")
    original="<html>原始两列报告</html>"
    (tmp_path/"comparison.html").write_text(original)
    render(tmp_path,result,outcomes)
    assert (tmp_path/"comparison.codex-only.html").read_text()==original
    page=(tmp_path/"comparison.html").read_text()
    assert page.count("本地新版 / 排名")==3
    assert page.count("Codex 参考 / 排名")==3
    assert page.count("<section id=")==3
    render(tmp_path,result,outcomes)
    assert (tmp_path/"comparison.codex-only.html").read_text()==original


def test_measurement_features_do_not_include_pose_completeness_or_reference_scores():
    body_features=set(k for keys in FEATURES["body"].values() for k in keys)
    assert not body_features & {"visibility","scene","confidence","old_score","codex_score","face_score"}


def test_mean_baseline_is_visible_for_detecting_score_collapse():
    samples,measured,refs=data()
    outcomes=cross_validate(samples,measured,refs)
    result=comparison(samples,refs,outcomes)
    assert all(result["statistics"][m]["leave_one_out_mean_mae"] is not None for m in MODULES)


def test_float_segmentation_input_has_aligned_rows_without_stretching():
    from PIL import Image
    rng=np.random.default_rng(1)
    original=rng.integers(0,256,(512,306,3),dtype=np.uint8)
    padded,(w,h)=aligned_body_input(Image.fromarray(original))
    assert (w,h)==(306,512)
    assert padded.shape[0]%32==padded.shape[1]%32==0
    assert np.array_equal(padded[:h,:w],original)


def test_segmentation_rejects_padded_native_mask_before_unsafe_copy():
    from app.local_candidate import segmentation_array

    class PaddedMask:
        width, height, channels = 306, 512, 1
        def is_contiguous(self):
            return False
        def numpy_view(self):
            pytest.fail("must not enter the native float-copy path")

    with pytest.raises(ValueError, match="内存不连续"):
        segmentation_array(PaddedMask(), 306, 512)


def test_segmentation_copy_owns_memory_after_native_result_changes():
    from app.local_candidate import segmentation_array

    class Mask:
        width, height, channels = 32, 16, 1
        values = np.ones((16, 32, 1), dtype=np.float32)
        def is_contiguous(self):
            return True
        def numpy_view(self):
            return self.values

    mask = Mask()
    values = segmentation_array(mask, 32, 16)
    mask.values[:] = 0
    assert values.shape == (16, 32)
    assert np.all(values == 1)


def test_full_local_run_uses_existing_reference_without_external_call_or_database_writes(tmp_path,monkeypatch):
    import subprocess
    from PIL import Image
    from app import local_calibration_trial as trial
    from app.engine import file_fingerprint

    monkeypatch.setattr(subprocess,"Popen",lambda *a,**kw:pytest.fail("本地试评不得调用Codex或其他进程"))
    samples,measured,refs=data()
    assets=tmp_path/"assets";assets.mkdir()
    for sample in samples:
        aid=sample["id"]; directory=tmp_path/aid;directory.mkdir()
        Image.new("RGB",(80,100)).save(directory/"whole.jpg")
        (assets/aid).write_bytes((directory/"whole.jpg").read_bytes())
        sample["sha256"]=file_fingerprint(assets/aid)
        sample["target_json"]="null"
        for module in MODULES:
            refs[aid]["results"][module].update(engine={"model":"saved-reference"},input_hashes=[file_fingerprint(directory/"whole.jpg")])
        refs[aid]["status"]="completed"
        (directory/"result.json").write_text(json.dumps(refs[aid]))
    baseline={"samples":samples,"engine":{"model":"saved-reference"},"fingerprints":{"all":"before"}}
    (tmp_path/"baseline.json").write_text(json.dumps(baseline))
    monkeypatch.setattr(trial,"DATA_DIR",tmp_path)
    monkeypatch.setattr(trial,"identity",lambda:{"weights":{"model":"hash"},"version":"test"})
    monkeypatch.setattr(trial,"snapshot",lambda path:{"fingerprints":{"all":"before"},"counts":{"assets":7}})
    class FakeModels:
        def close(self): pass
        def face(self,image): return measured["photo-1"]["face"]
        def body(self,image): return measured["photo-1"]["body"]
        def photography(self,image,box): return measured["photo-1"]["photography"]
    monkeypatch.setattr(trial,"Measurements",FakeModels)
    trial.run(tmp_path)
    result=json.loads((tmp_path/"local-candidate/comparison.json").read_text())
    assert result["preservation"]["unchanged"] is True
    assert len(result["rows"])==7
    assert (tmp_path/"comparison.html").exists()
    for sample in samples:
        assert file_fingerprint(assets/sample["id"])==sample["sha256"]
    # A changed original must stop the trial before producing any new scores.
    (assets/"photo-0").write_bytes(b"changed original")
    with pytest.raises(ValueError,match="原图变化"):
        trial.run(tmp_path)
