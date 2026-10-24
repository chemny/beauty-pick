"""Pixel-supported visibility permissions, not anatomy or attractiveness inference."""
import numpy as np
import cv2
CONFIG={'version':'pixel-pose-whitelist-26.0','pose_confidence':.6,'core_support_min':.35,'max_arm_hand_core':.40,'min_visible_region_fraction':.65,'leg_disk_radius_shoulder':.07,'leg_pixel_support_min':.4,'leg_continuous_length_shoulder':.15,'leg_area_shoulder_squared':.015,'leg_samples':61,'flared_garment_width_ratio':1.18}
def good(p):return p.get('visibility',0)>=.6 and p.get('presence',0)>=.6 and .01<p.get('x',0)<.99 and .01<p.get('y',0)<.99
def select(models):
 candidates=[]
 for name in ['pose_full','pose_heavy']:
  pts=models.get(name,{}).get('landmarks',[])
  if len(pts)==33 and all(good(pts[i]) for i in [11,12]):candidates.append((sum(float(pts[i].get('visibility',0))*float(pts[i].get('presence',0)) for i in [11,12,23,24]),name,pts))
 return max(candidates,key=lambda c:(c[0],c[1]))[1:] if candidates else (None,None)
def result(allowed,state,reason,**kwargs):return {'allowed':bool(allowed),'state':state,'reason':reason,**kwargs}
def longest_run(flags):
 best=0;start=None;best_start=None;run=0
 for i,v in enumerate(flags):
  if v:
   if run==0:start=i
   run+=1
   if run>best:best=run;best_start=start
  else:run=0
 return best,best_start

def leg_segment(mask,whole,a,b,S,start_good,end_good):
 h,w=mask.shape;a=np.array(a,float);b=np.array(b,float);length=float(np.linalg.norm(b-a));radius=max(3,round(.07*S));rows=[];flags=[]
 if not start_good or length<.15*S or length>4*max(h,w):return {'allowed':False,'reason':'近端关键点不足或候选段长度异常','points':[]}
 for t in np.linspace(0,1,61):
  p=a+(b-a)*t;x,y=map(lambda z:int(round(z)),p);inside=radius<=x<w-radius and radius<=y<h-radius
  skin=pants=0.
  if inside:
   r=mask[y-radius:y+radius+1,x-radius:x+radius+1];q=whole[y-radius:y+radius+1,x-radius:x+radius+1]
   # Both independent saved masks must support the same semantic class.
   skin=float(((r==14)&(q==14)).mean());pants=float(((r==6)&(q==6)).mean())
  supported=inside and max(skin,pants)>=.4;flags.append(supported);rows.append({'point':p.tolist(),'inside':inside,'skin_support':skin,'pants_support':pants,'supported':supported})
 n,start=longest_run(flags);span=(n-1)*length/60 if n else 0;area=int(((mask==14)&(whole==14)).sum());enough_area=area>=max(30,.015*S*S) or any(r['pants_support']>=.4 for r in rows)
 allowed=span>=.15*S and enough_area
 return {'allowed':bool(allowed),'reason':'点位走向内有连续腿部/裤装像素支持' if allowed else '未找到足够连续的腿部像素支持','continuous_length_px':span,'minimum_length_px':.15*S,'supported_fraction':sum(flags)/61,'skin_area_px':area,'start_joint_good':start_good,'end_joint_good':end_good,'full_segment_supported':bool(allowed and end_good and sum(flags)/61>=.7),'points':rows,'longest_run_start':start,'longest_run_samples':n}

def evaluate(models,mask,whole):
 h,w=mask.shape;name,p=select(models);parts={k:result(False,'uncertain','可靠肩点或身体区域不足') for k in ['chest','waist','hips','legs']};out={'config':CONFIG,'pose_source':name,'parts':parts,'dimensions':[w,h],'source':'saved MediaPipe pose plus FASHN whole/ROI agreement; no new inference','scope':'visible appearance permission, not exact anatomy or circumference'}
 if p is None:return out
 xy=lambda i:np.array([p[i]['x']*w,p[i]['y']*h]);shoulder=(xy(11)+xy(12))/2;S=float(np.linalg.norm(xy(11)-xy(12)));out['shoulder_span_px']=S
 if S<40:return out
 hips_good=all(good(p[i]) for i in [23,24]);hip=(xy(23)+xy(24))/2 if hips_good else None;T=float(hip[1]-shoulder[1]) if hips_good else S;axis_valid=bool(hips_good and T>=.25*S)
 # Shoulder-only local candidate scope does not manufacture hip joints.
 bands={'chest':(.12,.50),'waist':(.50,.85),'hips':(.88,1.18)} if axis_valid else {'chest':(.15,.65)}
 for part,(lo,hi) in bands.items():
  scale=T if axis_valid else S;y0=int(round(shoulder[1]+lo*scale));y1=int(round(shoulder[1]+hi*scale));visible=max(0,min(h,y1)-max(0,y0))/max(1,y1-y0);y0c=max(0,y0);y1c=min(h,y1)
  center=float(shoulder[0]+((hip[0]-shoulder[0])*((lo+hi)/2)) if axis_valid else shoulder[0]);x0=max(0,round(center-.38*S));x1=min(w,round(center+.38*S));crop=mask[y0c:y1c,x0:x1];q=whole[y0c:y1c,x0:x1]
  if visible<.65 or crop.size==0:parts[part]=result(False,'not_visible','候选部位主要在画面外或范围不足',box=[x0,y0c,x1,y1c],visible_region_fraction=visible);continue
  usable=np.isin(crop,[3,4,5,6,16])&np.isin(q,[3,4,5,6,16]);support=float(usable.mean());arms=float(np.isin(crop,[12,13]).mean());garment=float(np.isin(crop,[3,4,5,6]).mean());skin=float((crop==16).mean())
  widths=[];edge_visible=0;validrows=0
  for y in range(y0c,y1c):
   l=max(0,round(center-.70*S));r=min(w,round(center+.70*S));v=mask[y,l:r];indices=np.flatnonzero(np.isin(v,[3,4,5,6,16]))
   if not len(indices):continue
   validrows+=1;a=l+indices[0];b=l+indices[-1];widths.append([y,b-a])
   # Count actual external/background edges, not arm contact or clipped corridor edges.
   if a>l and b<r-1 and mask[y,a-1]==0 and mask[y,b+1]==0:edge_visible+=1
  edgefrac=edge_visible/max(1,validrows);flare=None
  if len(widths)>=6:
   n=max(2,len(widths)//3);upper=float(np.median([v for _,v in widths[:n]]));lower=float(np.median([v for _,v in widths[-n:]]));flare=lower/max(1,upper)
  reason='';allow=support>=.35 and arms<=.4
  if not allow:reason='躯干像素不足或手臂/手遮挡占比过高'
  if part=='hips' and garment>.5 and (flare is not None and flare>1.18):allow=False;reason='下部衣物明显外扩，裙摆不能替代臀胯轮廓'
  if part in ['waist','hips'] and edgefrac<.20 and skin<.20:allow=False;reason='缺少可核对的部位外缘，衣物/接触边界身份不明'
  state='partial' if allow else 'uncertain';reason=reason or '仅放行可见躯干/着装局部；不确认真实围度、贴合度或隐藏臀部'
  parts[part]=result(allow,state,reason,box=[x0,y0c,x1,y1c],visible_region_fraction=visible,torso_support_fraction=support,arm_hand_fraction=arms,garment_fraction=garment,skin_fraction=skin,external_edge_fraction=edgefrac,lower_upper_width_ratio=flare,restrictions=['只能评可见着装轮廓','不得把装饰、袖子当身体','不得推断背面臀形或真实围度'])
 # Leg permissions choose an observed pose per segment; never substitute face landmarks.
 legs={}
 for side,a,b,c in [('left',23,25,27),('right',24,26,28)]:
  for label,start,end in [(side+'_thigh',a,b),(side+'_calf',b,c)]:
   candidates=[]
   for source in ['pose_full','pose_heavy']:
    pts=models.get(source,{}).get('landmarks',[])
    if len(pts)!=33:continue
    point=lambda i:[pts[i]['x']*w,pts[i]['y']*h]
    q=leg_segment(mask,whole,point(start),point(end),S,good(pts[start]),good(pts[end]));q['pose_source']=source;candidates.append(q)
   legs[label]=max(candidates,key=lambda q:(q['allowed'],q.get('full_segment_supported',False),q.get('supported_fraction',0))) if candidates else {'allowed':False,'reason':'腿点缺失','points':[]}
 # Skirt-only hips without observable waist/hip transition remain uncertain.
 if parts['hips']['allowed'] and not parts['waist']['allowed'] and parts['hips'].get('garment_fraction',0)>.5:
  box=parts['hips'].get('box');region=mask[box[1]:box[3],box[0]:box[2]] if box else np.array([])
  if region.size and float(np.isin(region,[4,5]).mean())>.5:
   parts['hips'].update(allowed=False,state='uncertain',reason='裙装覆盖且没有获准的腰胯衔接证据，不能证明臀胯可见')
 allowed=[k for k,v in legs.items() if v['allowed']];full=all(v.get('full_segment_supported',False) for v in legs.values());parts['legs']=result(bool(allowed),'visible' if full else 'partial' if allowed else 'not_visible','连续腿段有点位与分割双重支持' if allowed else '没有通过连续腿段像素验证；预测关键点不能证明腿可见',segments=legs,allowed_segments=allowed,restrictions=['只评获准腿段','不得从大腿局部推断完整腿长','不得用裙装/脚部替代腿段'])
 return out
