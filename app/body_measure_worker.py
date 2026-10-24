"""Isolated offline FASHN/MediaPipe process; protects face/photo dependency versions."""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'models/body-runtime/deps'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

def main():
 import numpy as np,cv2,torch,mediapipe as mp
 from PIL import Image,ImageOps
 from mediapipe.tasks.python import vision,BaseOptions
 from .body_vendor.parser import FashnHumanParser
 from . import body_visibility_base as geo
 from .body_visibility import evaluate
 from .body_parameters import evaluate as parameters
 from .images import oriented_rgb
 source,output=sys.argv[1:3];image=oriented_rgb(Path(source));image.thumbnail((1672,1672));w,h=image.size
 torch.set_num_threads(4)
 models={};pw=((w+15)//16)*16;pixels=cv2.copyMakeBorder(np.asarray(image),0,0,0,pw-w,cv2.BORDER_REPLICATE);mi=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(pixels))
 for key,path in [('pose_full',ROOT/'models/pose_landmarker_full.task'),('pose_heavy',ROOT/'models/pose_landmarker_heavy.task')]:
  with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)))) as pose:
   res=pose.detect(mi);models[key]={'landmarks':[{'x':float(p.x)*pw/w,'y':float(p.y),'z':float(p.z),'visibility':float(p.visibility),'presence':float(p.presence)} for p in res.pose_landmarks[0]] if res.pose_landmarks else []}
 # Local weights only; the private worker never makes network connections.
 parser=FashnHumanParser(model_id=str(ROOT/'models/fashn-human-parser'),device='cpu')
 def infer(crop,pad=False):
  cw,ch=crop.size
  if pad:
   pw=max(cw,int(np.ceil(ch*2/3)));ph=max(ch,int(np.ceil(pw*1.5)));canvas=Image.new('RGB',(pw,ph));canvas.paste(crop,(0,0))
  else:canvas=crop;pw,ph=canvas.size
  with torch.inference_mode():native=parser.model(pixel_values=parser._preprocess_single(np.asarray(canvas)).unsqueeze(0)).logits[0].numpy()
  a=cv2.resize(native.transpose(1,2,0),(pw,ph),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)[:,:ch,:cw];a-=a.max(0,keepdims=True);a=np.exp(a);a/=a.sum(0,keepdims=True);return a
 whole=infer(image);whole_mask=whole.argmax(0).astype(np.uint8);fused=whole.copy();count=np.ones((h,w),np.float32);name,p=geo.select(models)
 if p and all(geo.good(p[i]) for i in [23,24]):
  xy=lambda i:np.array([p[i]['x']*w,p[i]['y']*h]);sh=(xy(11)+xy(12))/2;hip=(xy(23)+xy(24))/2;T=hip[1]-sh[1];S=float(np.linalg.norm(xy(11)-xy(12)))
  if T>.25*S:
   for factor in [1.,1.2]:
    cx=(sh[0]+hip[0])/2;cy=sh[1]+.62*T;rh=max(1.55*T,1.5*S)*factor;rw=max(1.65*S,rh*2/3);rh=max(rh,rw*1.5);rw=rh*2/3;x0=max(0,int(cx-rw/2));y0=max(0,int(cy-rh/2));x1=min(w,int(cx+rw/2));y1=min(h,int(cy+rh/2))
    if x1>x0 and y1>y0:fused[:,y0:y1,x0:x1]+=infer(image.crop((x0,y0,x1,y1)),True);count[y0:y1,x0:x1]+=1
 mask=(fused/count[None,:,:]).argmax(0).astype(np.uint8);prior=geo.evaluate(models,mask,whole_mask);visibility=evaluate(models,mask,whole_mask,prior);visibility=parameters(models,mask,whole_mask,visibility)
 Path(output).write_text(json.dumps({'visibility':visibility,'dimensions':[w,h]},ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
