import io
import json
import math
from PIL import Image
import pytest
from app import fpem_model, engine, provider, pipeline, storage
from app.scoring import RULES, is_overall_face


def test_fpem_identity_and_explicit_rollback(monkeypatch):
    monkeypatch.delenv('BEAUTYPICK_FACE_MODEL',raising=False)
    current=engine.identity('face')
    assert current['model']=='FPEM' and current['face_scoring_method']=='fpem-face-v1'
    assert 'fpem.pth' in current['weights'] and 'scut_fbp5500_beauty.pt' not in current['weights']
    monkeypatch.setenv('BEAUTYPICK_FACE_MODEL','resnet50')
    old=engine.identity('face')
    assert old['model']=='SCUT-FBP5500-ResNet50'
    assert engine.signature(current)!=engine.signature(old)


@pytest.mark.parametrize('raw',[None,True,float('nan'),float('inf'),.9,5.1])
def test_fpem_invalid_raw_never_substitutes_default(raw):
    with pytest.raises(ValueError):fpem_model.assessment({'status':'completed','raw_score':raw,'total':80})


def test_fpem_overall_contract_and_version():
    result=fpem_model.assessment({'status':'completed','raw_score':4.15663,'total':83.1})
    assert result['dimensions']==[] and result['weights']=={} and result['total']==83.1
    assert is_overall_face(result)
    assert not is_overall_face({**result,'rubric_version':'fpem-face-unknown'})
    with pytest.raises(ValueError):fpem_model.assessment({'status':'completed','raw_score':4.15663,'total':99})


def test_fpem_provider_uses_selected_model_only(tmp_path,monkeypatch):
    identity={'provider':'local','model':'FPEM','face_scoring_method':'fpem-face-v1'}
    monkeypatch.setattr(provider,'engine_identity',lambda _:identity)
    monkeypatch.setattr(provider,'fast_objective_evaluate',lambda *a,**k:pytest.fail('legacy scoring used'))
    monkeypatch.setattr(fpem_model,'infer',lambda image:{'status':'completed','raw_score':4.15663,'total':83.1})
    path=tmp_path/'face.png';Image.new('RGB',(224,224)).save(path)
    raw,usage=provider.evaluate('face',path,lambda:False)
    assert fpem_model.assessment(raw)['total']==83.1 and usage['engine']==identity


def test_fpem_missing_or_changed_weight_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(fpem_model,'MODEL_PATH',tmp_path/'missing.pth')
    with pytest.raises(ValueError,match='权重'):fpem_model._get_model()
    (tmp_path/'missing.pth').write_text('not a valid checkpoint')
    with pytest.raises(ValueError,match='校验'):fpem_model._get_model()


def test_fpem_context_uses_standard_lossless_crop(tmp_path,monkeypatch):
    from app import standard_face
    source=tmp_path/'image.png';Image.new('RGB',(400,600)).save(source)
    monkeypatch.setattr(pipeline.s,'asset_path',lambda _:source)
    monkeypatch.setattr(pipeline.s,'crop_path',lambda _:pytest.fail('legacy crop used'))
    identity={'face_scoring_method':'fpem-face-v1','rubric_version':fpem_model.RUBRIC_VERSION,'prompt_version':fpem_model.PROMPT_VERSION}
    monkeypatch.setattr(provider,'engine_identity',lambda _:identity)
    pixel=Image.new('RGB',(224,224),(55,88,111))
    monkeypatch.setattr(standard_face,'prepare',lambda *a:(pixel,{'margin':.12}))
    context,data=pipeline.evaluation_context({'id':'test','sha256':engine.file_fingerprint(source),'target_json':'{}'},'face',True)
    assert context['rubric_version']==fpem_model.RUBRIC_VERSION
    assert context['prompt_version']==fpem_model.PROMPT_VERSION
    assert context['input_format']=='PNG'
    assert Image.open(io.BytesIO(data)).tobytes()==pixel.tobytes()


def test_fpem_is_rankable_and_cannot_reweight_anatomy():
    from app.comparison import assessment
    face=fpem_model.assessment({'status':'completed','raw_score':4.15663,'total':83.1})
    face['engine']={'provider':'local','model':'FPEM'}
    asset={'id':'test','filename':'test.png','adult_confirmed':True,'report_json':json.dumps({'id':'report','face':face})}
    weights={r[0]:r[2] for r in RULES['face']}
    item,error=assessment(asset,'face',weights)
    assert error is None and item['score']==83.1
    weights.update(eyes=30,skin=0)
    assert assessment(asset,'face',weights)[0] is None


def test_fpem_shared_model_device_calls_are_serialized(monkeypatch):
    import threading
    import time
    import torch
    from concurrent.futures import ThreadPoolExecutor
    active = peak = 0
    guard=threading.Lock()
    def model(inputs):
        nonlocal active,peak
        assert [tuple(v.shape) for v in inputs]==[(1,3,224,224),(1,3,112,112),(1,3,160,160)]
        with guard:
            active+=1;peak=max(peak,active)
        time.sleep(.02)
        with guard:active-=1
        return (torch.tensor([[4.0]]),None)
    monkeypatch.setattr(fpem_model,'_get_model',lambda:(model,'cpu'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:fpem_model.infer(Image.new('RGB',(224,224))),range(4)))
    assert peak==1 and all(r['total']==80.0 for r in results)
