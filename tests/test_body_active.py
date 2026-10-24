import copy,json,io
from pathlib import Path
import pytest
from PIL import Image
from app import body_model as b,provider,pipeline,engine

def raw(values=(80,None,None,80)):
 assessment={'scope':'测试',**{k:{'score':v,'visibility':'not_visible' if v is None else 'partial','confidence':'medium','evidence':'原始依据'} for k,v in zip(b.WEIGHTS,values)}}
 visibility={'dimensions':[200,400],'parts':{k:{'allowed':v is not None,'state':'partial' if v is not None else 'not_visible','reason':'测试权限'} for k,v in zip(b.WEIGHTS,values)}}
 return {'assessment':assessment,'visibility':visibility}

def test_fixed_weight_missing_zero_and_raw_preservation():
 r=raw();before=copy.deepcopy(r);out=b.assessment(r);assert out['total']==36 and r==before
 assert out['dimensions'][1]['score']==0 and out['dimensions'][1]['observed_score'] is None and out['dimension_scale']==100
 assert b.assessment(raw((None,)*4))['total']==0
 assert b.assessment(raw((80,None,None,None)))['total']==20

def test_actual_zero_and_invalid_observation():
 assert b.assessment(raw((0,None,None,None)))['dimensions'][0]['status']=='assessable'
 for v in [True,101,-1,float('nan'),70.123]:
  with pytest.raises(ValueError):b.assessment(raw((v,None,None,None)))
 r=raw();r['visibility']['parts']['chest']['allowed']=False
 with pytest.raises(ValueError):b.assessment(r)

def test_whitelist_cannot_be_reopened_by_model():
 r=raw();o=r['assessment'];o['hips'].update(score=90,visibility='visible');clean,changes=b.enforce(o,r['visibility']);assert clean['hips']['score'] is None and any(c['part']=='hips' for c in changes)

def test_identity_records_real_local_models():
 ident=engine.identity('body');assert ident['model']==b.MODEL and ident['model_digest']==b.DIGEST and ident['scoring_method']==b.METHOD
 assert all(ident['weights'].values()) and ident['reasoning_effort'] is None

def test_body_input_lossless_and_full_frame(tmp_path,monkeypatch):
 p=tmp_path/'image.png';im=Image.new('RGB',(1901,2101),(33,55,77));im.save(p);monkeypatch.setattr(pipeline.s,'asset_path',lambda _:p)
 context,data=pipeline.evaluation_context({'id':'test','sha256':engine.file_fingerprint(p),'target_json':'null'},'body',True)
 image=Image.open(io.BytesIO(data));assert image.size==im.size and image.tobytes()==im.tobytes() and context['input_format']=='PNG'

def test_body_provider_uses_new_model_not_legacy(tmp_path,monkeypatch):
 p=tmp_path/'image.png';Image.new('RGB',(200,400)).save(p);monkeypatch.setattr(b,'infer',lambda *a:(raw(),{'local':True}));monkeypatch.setattr(provider,'fast_objective_evaluate',lambda *a,**kw:pytest.fail('legacy scoring'))
 result,usage=provider.evaluate('body',p,lambda:False);assert b.assessment(result)['total']==36 and usage['engine']['model']==b.MODEL

def test_body_comparison_and_public_report_contract():
 from app.comparison import assessment
 result=b.assessment(raw());result.update(engine=b.identity());asset={'id':'test','filename':'image.png','adult_confirmed':True,'report_json':json.dumps({'id':'test-report','body':result,'target':None})}
 ranked,error=assessment(asset,'body',b.WEIGHTS);assert error is None and ranked['score']==36
 assert assessment(asset,'body',{'chest':100,'waist':0,'hips':0,'legs':0})[0] is None

def test_reviewed_catalogue_matches_approved_preview():
 p=Path(__file__).resolve().parents[1]/'data/trials/body-zero-fill-v28-20261024/result.json'
 if not p.exists():pytest.skip('private trial not distributed')
 for r in json.loads(p.read_text())['samples']:
  assert b.assessment({'assessment':r['assessment'],'visibility':r['visibility']})['total']==r['total_display']
