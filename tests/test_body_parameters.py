import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from app import body_parameters as p,body_visibility_base as base,body_visibility as side,body_model
ROOT=Path(__file__).resolve().parents[1]

def test_current_identity_and_fixed_zero():
 ident=body_model.identity();assert ident['prompt_version']=='body-candidate-parameters-v31-production-1'
 assert all(k in ident['implementation'] for k in ['body_parameters.py','body_parameter_geometry.py','body_regional.py'])
 assert body_model.total({'chest':None,'waist':80,'hips':None,'legs':None})==24

def test_runtime_replays_all_reviewed_candidates():
 folder=ROOT/'data/trials/body-parameters-v31-20261024'
 if not folder.exists():pytest.skip('private review fixtures not distributed')
 rows=json.loads((folder/'baseline.json').read_text())['samples'];manifest=json.loads((folder/'input-manifest.json').read_text())
 for row in rows:
  paths=manifest[row['sample']['id']];read=lambda suffix:next(ROOT/x for x in paths if any(x.endswith(s) for s in suffix))
  record=json.loads(read(['/record.json','/measurement-record.json']).read_text());mask=np.array(Image.open(read(['/fused-labels.png','/roi-labels.png'])));whole=np.array(Image.open(read(['/fashn_labels.png','/whole-labels.png'])))
  prior=side.evaluate(record['models'],mask,whole,base.evaluate(record['models'],mask,whole));out=p.evaluate(record['models'],mask,whole,prior)
  for key in ['chest','legs']:assert out['parts'][key]==prior['parts'][key]
  for key in ['waist','hips']:
   actual=out['parts'][key];expected=row['visibility']['parts'][key]
   for field in ['allowed','box','candidate_parameters','width_parameters','scoring_mode']:assert actual[field]==expected[field],(row['sample']['label'],key,field)
   b=actual['box']
   for f in actual['candidate_parameters'].values():
    for x,y in f['observed_points']:assert b[0]<=x<b[2] and b[1]<=y<b[3]
  text=json.dumps(p.summary(out));assert 'observed_points' not in text and 'matched_rows' in text if out['parts']['hips']['width_parameters'] else True

def test_single_side_never_paired():
 from app.body_parameter_geometry import extract
 t={'shoulder':[0,0],'T':100,'S':50,'rows':{90:{'sides':{'left':{'point':[10,90],'external':False,'scan_clipped':False}}}}}
 features,width=extract(t,'hips');assert features and not width
