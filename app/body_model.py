"""Local visible-body scoring with enforced permissions and fixed missing-zero totals."""
from __future__ import annotations
import base64,copy,hashlib,io,json,math,os,subprocess,sys,tempfile,threading,time
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from fractions import Fraction
import httpx
from PIL import Image,ImageOps
from .config import ROOT
from .body_prompts import PROMPT,TORSO_PROMPT,SCHEMA,OPTIONS,WEIGHTS
METHOD='visible-body-fixed-zero-v1';RUBRIC_VERSION='body-visible-fixed-zero-1.0';PROMPT_VERSION='body-candidate-parameters-v31-production-1'
MODEL='qwen3.5:2b-mlx';DIGEST='9dd89c688f18f01cb3636d01f9512d8daae2a531b3e726f59d68cf7b91c8358e'
NAMES={'chest':'肩胸','waist':'腰腹','hips':'臀胯','legs':'腿部'}
RULES=[(k,NAMES[k],w,'仅评价程序放行的可见范围；缺项按0计入') for k,w in WEIGHTS.items()]
LOCK=threading.Semaphore(1)
URL='http://127.0.0.1:11434'

def identity():
 from .engine import file_fingerprint,signature
 sources=['body_model.py','body_prompts.py','body_visibility.py','body_visibility_base.py','body_measure_worker.py','body_vendor/parser.py','body_vendor/labels.py','image_formats.py','images.py','body_parameters.py','body_regional.py','body_parameter_geometry.py']
 files=[ROOT/'models/fashn-human-parser/model.safetensors',ROOT/'models/fashn-human-parser/config.json',ROOT/'models/pose_landmarker_full.task',ROOT/'models/pose_landmarker_heavy.task']
 implementation={p:file_fingerprint(ROOT/'app'/p) for p in sources};weights={str(p.relative_to(ROOT)):file_fingerprint(p) for p in files}
 if any(v is None for v in [*weights.values(),*implementation.values()]):raise ValueError('身材模型文件缺失')
 return {'provider':'local_body_whitelist','model':MODEL,'model_digest':DIGEST,'reasoning_effort':None,'scoring_method':METHOD,'algorithm_version':'v26-legs-v31-torso-parameters-zero','preprocessing_version':'oriented-rgb1672-pose-pad16-fashn-whole-roi-v1','rubric_version':RUBRIC_VERSION,'prompt_version':PROMPT_VERSION,'weights':weights,'implementation':implementation,'dependencies':{'transformers':'4.57.6','tokenizers':'0.22.2','huggingface-hub':'0.36.0'},'prompt_sha256':signature({'full':PROMPT,'torso':TORSO_PROMPT,'schema':SCHEMA,'options':OPTIONS})}

def total(scores,weights=None):
 weights=WEIGHTS if weights is None else weights
 if set(scores)!=set(WEIGHTS) or set(weights)!=set(WEIGHTS) or sum(weights.values())!=100 or any(type(w) is not int or w<0 for w in weights.values()):raise ValueError('身材分项/权重不匹配')
 exact=Fraction()
 for k,v in scores.items():
  if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=100 or abs(v*10-round(v*10))>1e-6):raise ValueError('身材分项应为0–100且最多一位小数')
  exact+=Fraction(str(0 if v is None else v))*Fraction(weights[k],100)
 return float(Decimal(str(float(exact))).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))

def assessment(raw):
 observed=raw['assessment'];scores={k:observed[k]['score'] for k in WEIGHTS};score=total(scores)
 for k,p in raw['visibility']['parts'].items():
  if not p['allowed'] and scores[k] is not None:raise ValueError('关闭的部位不得评分')
 dims=[{'key':k,'score':0. if scores[k] is None else float(scores[k]),'observed_score':scores[k],'status':'missing_zero' if scores[k] is None else 'assessable','evidence':'' if scores[k] is None else observed[k]['evidence'],'limitations':[]} for k in WEIGHTS]
 return {'module':'body','status':'completed','total':score,'dimensions':dims,'dimension_scale':100,'weights':dict(WEIGHTS),'rubric_version':RUBRIC_VERSION,'prompt_version':PROMPT_VERSION,'scoring_method':METHOD,'score_basis':'fixed_weight_missing_zero','summary':'','strengths':[],'limitations':[],'shooting_tips':[],'editing_tips':[],'assessed_count':sum(v is not None for v in scores.values()),'dimension_count':4,'assessed_dimensions':[k for k,v in scores.items() if v is not None],'excluded_dimensions':[k for k,v in scores.items() if v is None],'observed_scores':scores,'scoring_input_modes':{k:raw['visibility']['parts'][k].get('scoring_mode','visual_only') for k in ['waist','hips']},'body_visibility':raw['visibility'],'raw_assessment':observed,'enforcement':raw.get('enforcement',[]),'effective_weights':dict(WEIGHTS),'model':MODEL,'provider':'local_body_whitelist','reasoning_effort':None}

def validate(obj):
 if not isinstance(obj,dict) or set(obj)!=set(['scope',*WEIGHTS]):raise ValueError('身材结果字段不完整')
 for k in WEIGHTS:
  p=obj[k]
  if set(p)!=set(['visibility','score','confidence','evidence']) or p['visibility'] not in ['visible','partial','not_visible'] or p['confidence'] not in ['high','medium','low'] or not isinstance(p['evidence'],str):raise ValueError('身材分项格式不正确')
 total({k:obj[k]['score'] for k in WEIGHTS})

def enforce(obj,v):
 clean=copy.deepcopy(obj);events=[]
 for k,p in v['parts'].items():
  q=clean[k]
  if not p['allowed']:
   if q.get('score') is not None:events.append({'part':k,'kind':'blocked_score_removed','raw':copy.deepcopy(q)})
   clean[k]={'visibility':'not_visible','score':None,'confidence':'high','evidence':p['reason']}
  elif q.get('score') is None:q['visibility']='not_visible'
  elif p['state']=='partial':q['visibility']='partial'
 leg=clean['legs'];p=v['parts']['legs']
 if leg.get('score') is not None:
  ankle=any(s.get('allowed') and s.get('end_joint_good') for name,s in p.get('segments',{}).items() if name.endswith('calf'));text=leg['evidence']
  if any(s in text for s in ['脚踝','踝部','脚部','双脚']) and not ankle or p['state']=='partial' and any(s in text for s in ['完整腿长','全腿比例','双腿完整','完整双腿']):
   events.append({'part':'legs','kind':'unsupported_named_anatomy_or_full_scope','raw':copy.deepcopy(leg)});clean['legs']={'visibility':'not_visible','score':None,'confidence':'low','evidence':'超出已验证范围'}
 validate(clean);return clean,events

def measure(path,cancelled):
 with tempfile.TemporaryDirectory(prefix='beautypick-body-measure-') as temp:
  output=Path(temp)/'result.json';log=open(Path(temp)/'worker.log','w+')
  process=subprocess.Popen([sys.executable,'-m','app.body_measure_worker',str(path),str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
  try:
   while process.poll() is None:
    if cancelled():raise ValueError('用户取消分析')
    if time.monotonic()-start>180:raise ValueError('身材分割超时，请重试')
    time.sleep(.1)
   if process.returncode!=0:
    log.seek(0);message=log.read()[-1200:]
    import logging
    logging.getLogger(__name__).error('Body measurement worker failed: %s',message)
    raise ValueError('身材本地测量失败，请重试')
   return json.loads(output.read_text())['visibility']
  finally:
   if process.poll() is None:
    import signal
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   log.close()

def model_identity():
 with httpx.Client(timeout=10,trust_env=False) as client:
  response=client.get(URL+'/api/tags');response.raise_for_status();tags=response.json();m=next((x for x in tags['models'] if x['name']==MODEL),None)
 if not m or m['digest']!=DIGEST or 'vision' not in m.get('capabilities',[]):raise ValueError('本地千问模型不存在或版本已变化')
 return m

def infer(image_path,cancelled):
 from .config import DATA_DIR
 while not LOCK.acquire(timeout=.1):
  if cancelled():raise ValueError('用户取消分析')
 try:
  identity_before=identity();v=measure(image_path,cancelled);allowed={k:{'permission':'allowed' if p['allowed'] else 'blocked','state':p['state'],'reason':p['reason'],'allowed_image_sides':p.get('allowed_sides',[]),'allowed_segments':p.get('allowed_segments',[]),'restrictions':p.get('restrictions',[])} for k,p in v['parts'].items()};enforcement=[]
  obj={'scope':'无获准可见部位',**{k:{'visibility':'not_visible','score':None,'confidence':'high','evidence':p['reason']} for k,p in v['parts'].items()}}
  raw_output=None;usage={};input_hashes=[]
  if any(p['allowed'] for p in v['parts'].values()):
   model_identity();image=ImageOps.exif_transpose(Image.open(image_path)).convert('RGB');whole=image.copy();whole.thumbnail((1280,1280));views=[('整图',whole)];w,h=v['dimensions']
   for k,p in v['parts'].items():
    if not p['allowed']:continue
    box=p.get('box')
    if k=='legs':
     pts=[q['point'] for s in p.get('segments',{}).values() if s['allowed'] for q in s.get('points',[]) if q['supported']]
     if pts:
      margin=.1*v['shoulder_span_px'];box=[max(0,min(q[0] for q in pts)-margin),max(0,min(q[1] for q in pts)-margin),min(w,max(q[0] for q in pts)+margin),min(h,max(q[1] for q in pts)+margin)]
    if box and box[2]>box[0] and box[3]>box[1]:
     crop=image.crop((box[0]/w*image.width,box[1]/h*image.height,box[2]/w*image.width,box[3]/h*image.height));crop.thumbnail((640,640));views.append((NAMES[k]+'获准候选局部',crop))
   images=[]
   for name,im in views:
    b=io.BytesIO();im.save(b,'JPEG',quality=92);images.append(base64.b64encode(b.getvalue()).decode());input_hashes.append(hashlib.sha256(b.getvalue()).hexdigest())
   from .body_parameters import summary as parameter_summary
   prompt=PROMPT+'\n腰腹臀胯候选参数：'+json.dumps(parameter_summary(v),ensure_ascii=False)+'\n肩胸腿部权限及候选范围：'+json.dumps(allowed,ensure_ascii=False)+'\n只评获准可见范围；图像左右按照片坐标。禁止把单侧轮廓说成完整宽度或背面臀部。\n图片顺序：'+json.dumps([name for name,im in views],ensure_ascii=False)
   # Run HTTP in a bounded helper thread so cancellation remains responsive while the model generates.
   import concurrent.futures
   executor=concurrent.futures.ThreadPoolExecutor(max_workers=1)
   def request():
    with httpx.Client(timeout=httpx.Timeout(180,connect=10),trust_env=False) as client:
     r=client.post(URL+'/api/chat',json={'model':MODEL,'stream':False,'think':False,'format':SCHEMA,'options':OPTIONS,'keep_alive':'5m','messages':[{'role':'user','content':prompt,'images':images}]});r.raise_for_status();return r.json()
   future=executor.submit(request)
   try:
    while not future.done():
     if cancelled():raise ValueError('用户取消分析')
     time.sleep(.1)
    raw_output=future.result()
   finally:executor.shutdown(wait=False,cancel_futures=True)
   if not raw_output.get('done') or raw_output.get('message',{}).get('tool_calls') or raw_output.get('model')!=MODEL:raise ValueError('身材模型返回不完整')
   obj=json.loads(raw_output['message']['content']);obj,enforcement=enforce(obj,v);model_identity();usage={k:raw_output.get(k) for k in ['total_duration','prompt_eval_count','eval_count']}
  if cancelled():raise ValueError('用户取消分析')
  if identity()!=identity_before:raise ValueError('分析期间身材引擎发生变化')
  audit={'assessment':obj,'visibility':v,'enforcement':enforcement,'raw_model_response':raw_output,'input_view_sha256':input_hashes};folder=DATA_DIR/'body-audit';folder.mkdir(exist_ok=True);key=hashlib.sha256(Path(image_path).read_bytes()).hexdigest();(folder/(key+'-'+str(time.time_ns())+'.json')).write_text(json.dumps(audit,ensure_ascii=False))
  return audit,{'provider':'local_body_whitelist','model':MODEL,'local':True,**usage}
 finally:LOCK.release()
