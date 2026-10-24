"""Regional waist/abdomen and independent local hip permissions, trial only."""
import copy
import numpy as np
from app import body_visibility as old
from app import body_visibility_base as pose
CONFIG={'version':'regional-waist-independent-hip-29.0','hip_boundary_distance_shoulder_max':.40,'abdomen_skin_min':.50,'abdomen_arm_max':.12,'lower_garment_flare_ratio':1.08,'minimum_flow_height_axis':.18}

def anatomy(models,w,h):
 source,p=pose.select(models)
 if p is None:return None
 xy=lambda i:np.array([p[i]['x']*w,p[i]['y']*h]);S=float(np.linalg.norm(xy(11)-xy(12)));sh=(xy(11)+xy(12))/2
 hips=[xy(i) for i in [23,24] if pose.good(p[i])]
 if S<40 or not hips:return None
 hip=np.mean(hips,axis=0);T=hip[1]-sh[1]
 if T<.25*S:return None
 return {'source':source,'p':p,'S':S,'T':float(T),'sh':sh,'hip':hip,'hip_sides':{'left':min(hips,key=lambda q:q[0]).tolist(),'right':max(hips,key=lambda q:q[0]).tolist()},'mode':'full_hip' if len(hips)==2 else 'one_observed_hip'}

def abdomen(mask,whole,a):
 h,w=mask.shape;S,T=a['S'],a['T'];sh,hip=a['sh'],a['hip'];cx=float(sh[0]+.7*(hip[0]-sh[0]));box=[max(0,round(cx-.20*S)),max(0,round(sh[1]+.55*T)),min(w,round(cx+.20*S)),min(h,round(sh[1]+.85*T))];x0,y0,x1,y1=box
 if x1<=x0 or y1-y0<max(12,.1*S):return {'allowed':False,'reason':'正面腹部候选范围不足','box':box}
 r=mask[y0:y1,x0:x1];q=whole[y0:y1,x0:x1];skin=float(((r==16)&(q==16)).mean());arms=float(np.isin(r,[12,13]).mean());allow=skin>=.5 and arms<=.12
 return {'allowed':bool(allow),'box':box,'skin_agreement_fraction':skin,'arm_fraction':arms,'reason':'腹部候选中央有连续可见躯干皮肤，仅可评这块可见表面，不评价侧腰或真实腹围' if allow else '中央被衣物或手臂覆盖，不能据衣褶推断腹部形态'}

def lower_garment(mask,a):
 h,w=mask.shape;S,T=a['S'],a['T'];sh,hip=a['sh'],a['hip'];ys=range(max(0,round(sh[1]+1.12*T)),min(h,round(sh[1]+1.62*T)));widths=[];garment=0;visiblelegs=0
 for y in ys:
  cx=float(sh[0]+(hip[0]-sh[0])*(y-sh[1])/T);l=max(0,round(cx-S));r=min(w,round(cx+S));row=mask[y,l:r];ix=np.flatnonzero(np.isin(row,[4,5]));visiblelegs+=int(np.count_nonzero(row==14))
  if len(ix)>10:widths.append((y,float(ix[-1]-ix[0])));garment+=len(ix)
 if len(widths)<6:return {'risk':False,'reason':'没有足够长的连续裙装流向证据'}
 n=max(2,len(widths)//3);ratio=float(np.median([x[1] for x in widths[-n:]])/max(1,np.median([x[1] for x in widths[:n]])));span=widths[-1][0]-widths[0][0];covered=visiblelegs/max(1,garment)<.03
 return {'risk':bool(ratio>1.08 and span>.18*T and covered),'lower_upper_ratio':ratio,'span_y_px':span,'visible_leg_to_garment_pixels':visiblelegs/max(1,garment),'reason':'长裙持续外展且下方无可见腿段，存在悬垂布料身份风险；不是单凭胯部展开否决'}

def shoulder_abdomen(mask,whole,models):
 import cv2
 h,w=mask.shape;name,p=pose.select(models)
 if p is None:return {'allowed':False,'reason':'可靠肩点缺失，不能以分割标签猜腹部位置'}
 xy=lambda i:np.array([p[i]['x']*w,p[i]['y']*h]);sh=(xy(11)+xy(12))/2;S=float(np.linalg.norm(xy(11)-xy(12)))
 if S<40:return {'allowed':False,'reason':'肩点尺度不足'}
 x0=max(0,round(sh[0]-.35*S));x1=min(w,round(sh[0]+.35*S));y0=max(0,round(sh[1]+.55*S));y1=min(h,round(sh[1]+1.75*S))
 if x1<=x0 or y1<=y0:return {'allowed':False,'reason':'候选腹部不在画面内'}
 skin=((mask[y0:y1,x0:x1]==16)&(whole[y0:y1,x0:x1]==16)).astype(np.uint8);n,labels,stats,_=cv2.connectedComponentsWithStats(skin,8)
 for left,top,width,height,area in stats[1:]:
  if area<.04*S*S or height<.10*S:continue
  xx=x0+left;yy=y0+top;hh=max(3,round(.05*S));above=mask[max(0,yy-hh):yy,xx:xx+width];below=mask[yy+height:min(h,yy+height+hh),xx:xx+width]
  upper=float(np.isin(above,[3,4]).mean()) if above.size else 0;lower=float(np.isin(below,[3,4,5,6,7]).mean()) if below.size else 0
  # Skin between an observed upper garment and lower garment is a candidate midriff; no hip joint is synthesized.
  if upper>=.30 and lower>=.30:return {'allowed':True,'box':[int(xx),int(yy),int(xx+width),int(yy+height)],'skin_area_px':int(area),'upper_garment_fraction':upper,'lower_garment_fraction':lower,'reason':'肩点以下、上下衣物之间有连续裸露躯干皮肤，仅放行这块腹部候选表面','hip_synthesized':False}
 return {'allowed':False,'reason':'未找到上下衣物夹持的连续腹部皮肤，不把胸前皮肤或衣物皱褶当腹部'}

def evaluate(models,mask,whole,prior):
 out=copy.deepcopy(prior);out['regional_trial_config']=CONFIG;h,w=mask.shape;a=anatomy(models,w,h)
 if a is None:
  fallback=shoulder_abdomen(mask,whole,models);out['shoulder_abdomen_fallback']=fallback
  if fallback['allowed']:
   out['parts']['waist'].update(allowed=True,state='partial',reason=fallback['reason'],box=fallback['box'],allowed_regions=['front_abdomen_skin'],allowed_sides=[],restrictions=['只能评价上下衣物之间的可见腹部皮肤，不评价侧腰','不合成髋点或推断完整腰围'])
  for k in ['waist','hips']:out['parts'][k]['regional_reason']='缺少可核对的腹/胯定位锚点，不仅凭衣物猜位置'
  return out
 # Existing tracing requires both hips. A single observed hip supports only central abdominal observation.
 x=old.trace(whole,models);y=old.trace(mask,models);abd=abdomen(mask,whole,a)
 if x is None or y is None:
  p=copy.deepcopy(prior['parts']['waist']);p.update(allowed=abd['allowed'],state='partial' if abd['allowed'] else 'uncertain',reason=abd['reason'],box=abd['box'],allowed_regions=['front_abdomen_skin'] if abd['allowed'] else [],abdomen_check=abd,localization_mode=a['mode'],restrictions=['只评价指定中央可见腹部皮肤','不得推断侧腰、真实围度或衣物下面的腹部']);out['parts']['waist']=p;return out
 for part in ['waist','hips']:
  p=copy.deepcopy(prior['parts'][part]);sides={s:old.side_support(x,y,part,s) for s in ['left','right']};allowed_sides=[];side_gates={}
  for side,c in sides.items():
   hip_positions=[]
   for name in ['pose_full','pose_heavy']:
    pts=models.get(name,{}).get('landmarks',[])
    if len(pts)==33 and all(pose.good(pts[i]) for i in [23,24]):
     coords=sorted([[pts[i]['x']*w,pts[i]['y']*h] for i in [23,24]])
     hip_positions.append(coords[0 if side=='left' else 1])
   pose_agreement=float(np.linalg.norm(np.array(hip_positions[0])-np.array(hip_positions[1]))/a['S']) if len(hip_positions)==2 else None
   # Local occlusion replaces the whole central arm-fraction veto. Contacts remain excluded by tracing.
   anchor=a['hip_sides'][side];points=c['support_points'];distances=[abs(q[0]-anchor[0])/a['S'] for q in points];d=float(np.median(distances)) if distances else None;anchor_ok=part!='hips' or d is not None and d<=.40
   existing_route=bool(part=='hips' and prior['parts']['hips']['allowed'] and side in prior['parts']['hips'].get('allowed_sides',[]))
   independent_route=bool(c['allowed'] and anchor_ok and (part!='hips' or pose_agreement is not None and pose_agreement<=.08))
   allowed=bool(existing_route or independent_route);side_gates[side]={'allowed':allowed,'side_continuity':c,'hip_anchor_horizontal_distance_shoulder':d,'hip_anchor_ok':anchor_ok,'pose_hip_agreement_shoulder':pose_agreement,'existing_verified_transition_route':existing_route,'independent_local_route':independent_route}
   if allowed:allowed_sides.append(side)
  lap=prior['parts'][part].get('risks',{}).get('folded_lap_pose_risk',False);flow=lower_garment(mask,a) if part=='hips' else {'risk':False};garment_risk=prior['parts'][part].get('risks',{}).get('broad_cloth_with_additional_risk',False)
  # Independent hips: no requirement for waist permission/bridge. Keep distinct local anatomical/garment evidence.
  allow=(bool(allowed_sides) or part=='waist' and abd['allowed']) and not lap and not garment_risk and not flow['risk']
  regions=[s+'_side' for s in allowed_sides] + (['front_abdomen_skin'] if part=='waist' and abd['allowed'] else [])
  if lap:reason='坐姿低位折叠投影未解决，保留衣物误测否决'
  elif garment_risk or flow['risk']:reason='衣物过宽/跳变或持续悬垂展开，局部身份仍不确定'
  elif allow:reason='独立局部证据放行：'+('、'.join(regions))+'；不要求另一侧或同侧腰线全程可见'
  else:reason='没有通过连续外缘与局部定位的证据；不猜遮挡部位'
  pts=[q for s in allowed_sides for q in sides[s]['support_points']];box=p.get('box')
  if part=='waist' and abd['allowed']:
   z=abd['box'];pts += [[z[0],z[1]],[z[2],z[3]]]
  if pts:
   cx=float(a['hip'][0]);margin=.08*a['S'];box=[max(0,round(min(min(q[0] for q in pts),cx)-margin)),max(0,round(min(q[1] for q in pts)-margin)),min(w,round(max(max(q[0] for q in pts),cx)+margin)),min(h,round(max(q[1] for q in pts)+margin))]
  p.update(allowed=bool(allow),state='partial' if allow else 'uncertain',reason=reason,box=box,allowed_sides=allowed_sides if allow else [],allowed_regions=regions if allow else [],regional_side_checks=side_gates,abdomen_check=abd if part=='waist' else None,independent_hip=True if part=='hips' else None,lower_garment_check=flow,restrictions=['只评获准局部，不能将着装外缘称为真实身体尺寸','腰腹：侧腰与中央腹部互不替代，腹部未获准不能根据衣褶评价平坦度','臀胯：只评可见侧胯，不推断背面臀形、饱满度或腰臀关系','不得补齐遮挡或推断完整宽度'])
  out['parts'][part]=p
 return out
