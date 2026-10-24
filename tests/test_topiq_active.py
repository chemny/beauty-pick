import io
import json
import pytest
from PIL import Image
from app import topiq_model, engine, provider, pipeline
from app.scoring import RULES, is_overall_photo


def test_identity_and_rollback(monkeypatch):
    monkeypatch.delenv('BEAUTYPICK_PHOTO_MODEL',raising=False)
    current=engine.identity('photography')
    assert current['model']=='TOPIQ-IAA-ResNet50'
    assert current['weights']=={'topiq-iaa-res50.pth':topiq_model.CHECKPOINT_SHA256}
    assert current['device']=='cpu'
    monkeypatch.setenv('BEAUTYPICK_PHOTO_MODEL','cv')
    assert engine.signature(current)!=engine.signature(engine.identity('photography'))


@pytest.mark.parametrize('value',[None,True,float('nan'),float('inf'),0,10.1])
def test_invalid_output_fails_closed(value):
    with pytest.raises(ValueError):topiq_model.assessment({'status':'completed','raw_score':value,'total':50})


def test_overall_contract_and_ranking():
    from app.comparison import assessment
    result=topiq_model.assessment({'status':'completed','raw_score':5.33117,'total':53.3})
    assert is_overall_photo(result) and result['dimensions']==[] and result['weights']=={}
    result['engine']={'model':'TOPIQ-IAA-ResNet50','provider':'local'}
    asset={'id':'test','filename':'test.png','adult_confirmed':False,
           'report_json':json.dumps({'id':'report','photography':result})}
    weights={r[0]:r[2] for r in RULES['photography']}
    item,error=assessment(asset,'photography',weights)
    assert error is None and item['score']==53.3
    weights['light']+=1
    assert assessment(asset,'photography',weights)[0] is None
    with pytest.raises(ValueError):topiq_model.assessment({'status':'completed','raw_score':5.33117,'total':99})


def test_missing_weight(tmp_path,monkeypatch):
    monkeypatch.setattr(topiq_model,'MODEL_PATH',tmp_path/'missing.pth')
    with pytest.raises(ValueError,match='权重'):topiq_model._get_model()


def test_full_photo_lossless_context(tmp_path,monkeypatch):
    source=tmp_path/'original.png';image=Image.new('RGB',(1901,2101),(55,88,111));image.save(source)
    monkeypatch.setattr(pipeline.s,'asset_path',lambda _:source)
    monkeypatch.setattr(pipeline.s,'crop_path',lambda _:pytest.fail('Face crop used for photography'))
    context,data=pipeline.evaluation_context({'id':'test','sha256':engine.file_fingerprint(source),'target_json':'null'},'photography',True)
    decoded=Image.open(io.BytesIO(data))
    assert decoded.size==image.size and decoded.tobytes()==image.tobytes()
    assert context['input_format']=='PNG' and context['rubric_version']==topiq_model.RUBRIC_VERSION


def test_provider_never_uses_legacy_score(tmp_path,monkeypatch):
    identity=engine.identity('photography')
    monkeypatch.setattr(provider,'engine_identity',lambda _:identity)
    monkeypatch.setattr(provider,'fast_objective_evaluate',lambda *a,**k:pytest.fail('Legacy CV score used'))
    monkeypatch.setattr(topiq_model,'infer',lambda im:{'status':'completed','raw_score':5.1,'total':51.0})
    path=tmp_path/'whole.png';Image.new('RGB',(300,500)).save(path)
    raw,usage=provider.evaluate('photography',path,lambda:False)
    assert topiq_model.assessment(raw)['total']==51.0 and usage['engine']==identity


def test_resize_preserves_aspect_ratio_and_uses_cpu(monkeypatch):
    import torch
    from app.topiq_vendor.topiq import CFANet
    seen=[]
    class Model:
        def __call__(self,x):
            seen.append((tuple(x.shape),x.device.type))
            return torch.tensor([[5.0]])
    monkeypatch.setattr(topiq_model,'_get_model',lambda:Model())
    assert topiq_model.infer(Image.new('RGB',(400,700)))['total']==50.0
    assert seen==[((1,3,700,400),'cpu')]
    # Official tensor resize takes a short edge size, not a square crop.
    import torchvision.transforms.functional as TF
    x=TF.resize(torch.zeros(1,3,700,400),384,antialias=True)
    assert x.shape[-2:]==(672,384)
