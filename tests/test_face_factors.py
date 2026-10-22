import copy
import json

import numpy as np
import pytest
from PIL import Image

from app import face_factors as factors
from app import face_factor_trial as trial
from app.face_preprocessing import scale_score


def fixture():
    rng=np.random.default_rng(5)
    image=Image.fromarray(rng.integers(0,256,(400,400,3),dtype=np.uint8))
    base=dict(left=60,top=60,right=340,bottom=340)
    points=rng.uniform(140,260,(478,2))
    points[[33,133]]=[[150,170],[170,175]]
    points[[362,263]]=[[230,190],[250,195]]
    return image,base,points


def test_each_experiment_changes_only_its_declared_factor():
    crop=[factors.SPECS[v] for v in factors.GROUPS["crop"]["variants"]]
    assert len({v["resize"] for v in crop})==len({v["rotation"] for v in crop})==1
    resized=[factors.SPECS[v] for v in factors.GROUPS["resize"]["variants"]]
    assert len({v["factor"] for v in resized})==len({v["rotation"] for v in resized})==1
    rotated=[factors.SPECS[v] for v in factors.GROUPS["rotation"]["variants"]]
    assert len({v["factor"] for v in rotated})==len({v["resize"] for v in rotated})==1


def test_resize_group_uses_identical_source_pixels_and_geometry():
    image,base,points=fixture();prepared=factors.prepare_inputs(image,base,points)
    for name in factors.GROUPS["resize"]["variants"]:
        record=prepared[name]
        assert record["source_box"]==[60,60,340,340]
        assert record["image"].size==(224,224)
        assert record["rotation_degrees"]==0
    expected=image.crop((60,60,340,340)).resize((224,224),Image.Resampling.BILINEAR)
    assert prepared["baseline"]["image"].tobytes()==expected.tobytes()


def test_crop_group_keeps_same_center_with_only_integer_rounding_error():
    _,base,_=fixture()
    for factor in (.65,.85,1.,1.15):
        left,top,right,bottom=factors.source_box(base,factor)
        assert abs((left+right)/2-200)<=.5
        assert abs((top+bottom)/2-200)<=.5
        assert right-left==bottom-top==round(280*factor)


def test_rotation_cannot_recenter_or_refit_the_crop():
    image,base,points=fixture();prepared=factors.prepare_inputs(image,base,points)
    a,b=prepared["rotation0"],prepared["rotation_eye"]
    assert a["source_box"]==b["source_box"]
    ma,mb=np.array(a["matrix"]),np.array(b["matrix"])
    assert np.linalg.det(ma[:,:2])==pytest.approx(np.linalg.det(mb[:,:2]))
    center=np.array([199.5,199.5])
    assert np.allclose(ma[:,:2]@center+ma[:,2],[111.5,111.5])
    assert np.allclose(mb[:,:2]@center+mb[:,2],[111.5,111.5])


def test_outside_photo_and_truncated_face_are_excluded_without_synthetic_padding():
    image,base,points=fixture()
    base=dict(left=0,top=0,right=300,bottom=300)
    prepared=factors.prepare_inputs(image,base,points)
    assert prepared["crop115"]["status"]=="not_comparable"
    assert "image" not in prepared["crop115"]
    assert prepared["crop065"]["status"]=="not_comparable"
    assert prepared["baseline"]["status"]=="ready"


def test_non_rotation_tests_survive_missing_keypoints():
    image,base,_=fixture();prepared=factors.prepare_inputs(image,base,None,"关键点失败")
    assert prepared["baseline"]["status"]==prepared["bicubic"]["status"]=="ready"
    assert prepared["rotation0"]["status"]==prepared["rotation_eye"]["status"]=="not_comparable"


def outcomes():
    samples=[];results={};refs={}
    for i in range(5):
        aid=str(i);samples.append(dict(id=aid,label=aid,created_at=aid));refs[aid]=float(60+i*5)
        results[aid]={"variants":{k:{"status":"completed","raw_score":3+i*.2,**scale_score(3+i*.2)} for k in factors.SPECS}}
    return samples,results,refs


def test_all_within_group_metrics_use_the_identical_common_subset():
    samples,results,refs=outcomes()
    results["0"]["variants"]["crop115"]={"status":"not_comparable","error":"超出原图"}
    result=trial.comparison(samples,results,refs)
    assert result["groups"]["crop"]["common_count"]==4
    assert all(v["count"]==4 for v in result["groups"]["crop"]["comparisons"].values())
    assert result["groups"]["resize"]["common_count"]==5
    assert result["groups"]["crop"]["excluded"][0]["id"]=="0"


def test_unclipped_values_drive_change_metrics_even_when_both_display_99():
    samples,results,refs=outcomes()
    for r in results.values():
        r["variants"]["baseline"].update(raw_score=5.8,**scale_score(5.8))
        r["variants"]["lanczos"].update(raw_score=5.5,**scale_score(5.5))
    stats=trial.comparison(samples,results,refs)["groups"]["resize"]["comparisons"]["lanczos"]
    assert stats["median_absolute_delta"]>5
    assert stats["over_2_points"]==5


def test_reference_changes_never_change_new_scores_or_ranks():
    samples,results,refs=outcomes();before=copy.deepcopy(results)
    a=trial.comparison(samples,results,refs)
    b=trial.comparison(samples,results,{k:100-v for k,v in refs.items()})
    assert results==before
    assert [[r[k] for k in factors.SPECS] for r in a["rows"]]==[[r[k] for k in factors.SPECS] for r in b["rows"]]


def test_model_failure_does_not_copy_reference_or_other_input_score(tmp_path,monkeypatch):
    image,base,points=fixture()
    monkeypatch.setattr(trial,"geometry",lambda *args:(base,points,None))
    def failing(image):raise ValueError("模型推理失败")
    monkeypatch.setattr(trial,"infer",failing)
    result=trial.evaluate(image,{},tmp_path)
    assert result["status"]=="failed"
    assert all(v.get("total") is None for v in result["variants"].values())


def test_original_page_sections_are_preserved_on_report_generation(tmp_path):
    source="<html><style>body{color:black}</style><section id='face'>old face</section><section id='body'>old body</section><section id='photography'>old photo</section></html>"
    (tmp_path/"comparison.html").write_text(source);(tmp_path/trial.FOLDER).mkdir()
    result=trial.comparison([],{},{});trial.render(tmp_path,result,{})
    page=(tmp_path/"comparison.html").read_text()
    assert "<section id='body'>old body</section>" in page
    assert "<section id='face-previous'>old face</section>" in page
    assert (tmp_path/"comparison.face-v2.html").read_text()==source
    assert page.count("id='factor-")==3
    trial.render(tmp_path,result,{})
    assert (tmp_path/"comparison.face-v2.html").read_text()==source
