import copy
import json
import math

import numpy as np
import pytest
from PIL import Image

from app import face_preprocessing as face
from app import face_preprocessing_trial as trial


def points(angle=0.):
    rng=np.random.default_rng(42)
    p=rng.uniform((75,55),(175,205),(478,2))
    p[[33,133]]=[[95,100],[110,100]]
    p[[362,263]]=[[140,100],[155,100]]
    theta=math.radians(angle)
    r=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    return (p-[125,125])@r.T+[125,125]


def test_square_transform_preserves_ratios_and_horizontally_aligns_eyes():
    p=points(17)
    t=face.square_transform(p,align=True)
    matrix=np.array(t["matrix"])
    gram=matrix[:,:2].T@matrix[:,:2]
    assert gram[0,1]==pytest.approx(0.,abs=1e-9)
    assert gram[0,0]==pytest.approx(gram[1,1])
    changed=p@matrix[:,:2].T+matrix[:,2]
    assert changed[[33,133],1].mean()==pytest.approx(changed[[362,263],1].mean())
    assert np.all((changed[:468]>=0)&(changed[:468]<=223))
    assert t["rotation_degrees"]==pytest.approx(17)


def test_square_crop_uses_same_scale_for_both_axes_without_stretch():
    output,diagnostics=face.warp_square(Image.new("RGB",(250,250)),face.square_transform(points(),align=False))
    assert output.size==(224,224)
    assert diagnostics["padding_fraction"]==0


@pytest.mark.parametrize("raw",[True,float("nan"),float("inf"),"4"])
def test_invalid_model_output_never_becomes_default_score(raw):
    with pytest.raises(ValueError):face.scale_score(raw)


def test_fixed_scale_never_reverses_raw_order_and_uses_no_fitted_parameters():
    values=[face.scale_score(float(x))["total"] for x in np.linspace(-1,8,1000)]
    assert values==sorted(values)
    assert face.scale_score(2.)["total"]==40.
    assert face.scale_score(5.3)["total"]==98.
    assert face.scale_score(3.969)["total"]>face.scale_score(3.233)["total"]


def test_output_limits_are_visible_not_silently_claimed_as_raw_model_predictions():
    assert face.scale_score(10.)["clipped"] is True
    assert face.scale_score(10.)["raw_outside_nominal_1_to_5"] is True
    assert face.scale_score(4.)["clipped"] is False


def test_alignment_is_invariant_to_input_scale_and_translation():
    p=points(12)
    a=np.array(face.square_transform(p,align=True)["matrix"])
    b=np.array(face.square_transform(p*1.3+[35,20],align=True)["matrix"])
    x=p@a[:,:2].T+a[:,2]
    y=(p*1.3+[35,20])@b[:,:2].T+b[:,2]
    assert np.allclose(x,y)


def test_missing_landmarks_do_not_produce_crop():
    with pytest.raises(ValueError):face.square_transform(np.zeros((3,2)),align=True)
    p=points();p[0]=float("nan")
    with pytest.raises(ValueError):face.square_transform(p,align=True)


def test_outer_crop_updates_target_without_changing_facial_location():
    im=Image.new("RGB",(1000,800));box=dict(x=.3,y=.2,width=.2,height=.25)
    variants=list(face.perturbations(im,box))
    _,cropped,target=variants[1]
    assert target["x"]*cropped.width==pytest.approx(280)
    assert target["width"]*cropped.width==pytest.approx(200)
    assert box==dict(x=.3,y=.2,width=.2,height=.25)


def test_scoring_failure_does_not_fill_from_other_crop_or_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,"preprocess",lambda *args:({k:Image.new("RGB",(224,224)) for k in face.VARIANTS},{k:{} for k in face.VARIANTS}))
    monkeypatch.setattr(trial,"infer",lambda image: (_ for _ in ()).throw(ValueError("推理失败")))
    with pytest.raises(ValueError,match="推理失败"):
        trial.evaluate(Image.new("RGB",(400,400)),{},tmp_path)


def test_reference_scores_only_affect_report_comparison_not_new_scores():
    samples=[dict(id="a",label="a",created_at="a",report_json='{"face":{"total":80}}'),
             dict(id="b",label="b",created_at="b",report_json='{"face":{"total":70}}')]
    references={k:{"results":{"face":{"total":v}}} for k,v in (("a",90),("b",60))}
    prior={"rows":[{"id":k,"scores":{"face":{"local":80.}}} for k in ("a","b")]}
    outputs={k:{"status":"completed","primary":{"total":face.scale_score(v)["total"],"raw_score":v},"variants":{}} for k,v in (("a",4.),("b",3.))}
    first=trial.compare(samples,references,prior,outputs)
    references["a"]["results"]["face"]["total"]=1.
    second=trial.compare(samples,references,prior,outputs)
    assert [(r["revised"],r["revised_rank"]) for r in first["rows"]]==[(r["revised"],r["revised_rank"]) for r in second["rows"]]


def test_report_preserves_body_photography_and_previous_html(tmp_path):
    (tmp_path/trial.FOLDER).mkdir()
    source="<section id='face'>old face</section><section id='body'>unchanged body</section><section id='photography'>unchanged photo</section>"
    (tmp_path/"comparison.html").write_text(source)
    result={"rows":[],"statistics":{k:{"order_agreement":.5} for k in ("old","local_v1","revised")},"stability_pass_count":0,"sample_count":0}
    trial.render(tmp_path,result,{})
    page=(tmp_path/"comparison.html").read_text()
    assert "<section id='body'>unchanged body</section>" in page
    assert "<section id='photography'>unchanged photo</section>" in page
    assert "颜值修正版 / 排名" in page
    assert (tmp_path/"comparison.local-v1.html").read_text()==source
    trial.render(tmp_path,result,{})
    assert (tmp_path/"comparison.local-v1.html").read_text()==source


def test_clipping_cannot_hide_model_instability():
    primary={"raw_score":5.7,**face.scale_score(5.7)}
    outcome={"primary":primary,"stability":{key:{"status":"completed","raw_score":5.4,**face.scale_score(5.4)} for key in ("a","b")}}
    result=trial.stability_summary(outcome)
    assert result["max_display_delta"]==0
    assert result["max_stability_delta"]>5
    assert result["stability_pass"] is False


def test_report_rebuild_may_change_report_code_but_never_weights_or_preprocessing():
    old={"weights":{"face":"one"},"implementation":{"face_preprocessing.py":"crop1","face_preprocessing_trial.py":"report1"}}
    changed=copy.deepcopy(old);changed["implementation"]["face_preprocessing_trial.py"]="report2"
    assert trial.scoring_identity(old)==trial.scoring_identity(changed)
    changed["weights"]["face"]="two"
    assert trial.scoring_identity(old)!=trial.scoring_identity(changed)


def test_report_only_reuses_real_saved_values_and_never_invokes_model(tmp_path,monkeypatch):
    from app.engine import file_fingerprint,signature
    data=tmp_path/"data";(data/"assets").mkdir(parents=True)
    review=tmp_path/"review";review.mkdir()
    aid="photo-a";(review/aid).mkdir()
    Image.new("RGB",(200,200)).save(review/aid/"whole.jpg")
    original=data/"assets"/aid;original.write_bytes((review/aid/"whole.jpg").read_bytes())
    sample=dict(id=aid,label=aid,created_at="a",sha256=file_fingerprint(original),adult_confirmed=True,
                target_json='{"x":0.2,"y":0.2,"width":0.5,"height":0.5}',report_json='{"face":{"total":80}}')
    baseline=dict(samples=[sample],fingerprints={"all":"before"},engine={"model":"saved-reference"})
    (review/"baseline.json").write_text(json.dumps(baseline))
    (review/aid/"result.json").write_text(json.dumps({"status":"completed","results":{"face":{"engine":baseline["engine"],"total":85.}}}))
    (review/"local-candidate").mkdir()
    (review/"local-candidate/comparison.json").write_text(json.dumps({"rows":[{"id":aid,"scores":{"face":{"local":83.}}}]}))
    (review/"comparison.html").write_text("<section id='body'>body</section><section id='photography'>photo</section>")
    old_engine={"weights":{"model":"same"},"implementation":{"face_preprocessing.py":"crop1","face_preprocessing_trial.py":"old-report"}}
    current=copy.deepcopy(old_engine);current["implementation"]["face_preprocessing_trial.py"]="new-report"
    manifest={"engine":old_engine,"samples":[{k:sample[k] for k in ("id","sha256","target_json")}],"primary_frozen_before_reference":face.PRIMARY,"scale_source":"legacy fixed mapping; no fitting"}
    output=review/trial.FOLDER;(output/aid).mkdir(parents=True)
    (output/"manifest.json").write_text(json.dumps(manifest))
    outcome={"status":"completed","primary":{"raw_score":4.,**face.scale_score(4.)},"variants":{},"stability":{},"cache_key":signature({"manifest":manifest,"id":aid})}
    (output/aid/"result.json").write_text(json.dumps(outcome))
    monkeypatch.setattr(trial,"identity",lambda:copy.deepcopy(current))
    monkeypatch.setattr(trial,"DATA_DIR",data)
    monkeypatch.setattr(trial,"snapshot",lambda path:{"fingerprints":{"all":"before"},"counts":{"assets":1}})
    monkeypatch.setattr(trial,"evaluate",lambda *a,**kw:pytest.fail("重建报告不得重新调用模型"))
    trial.run(review,rebuild_report=True)
    result=json.loads((output/"comparison.json").read_text())
    assert result["rows"][0]["revised"]==face.scale_score(4.)["total"]
    assert result["rebuilt_without_inference"] is True
    assert result["engine"]==old_engine
    assert json.loads((output/"manifest.json").read_text())==manifest
