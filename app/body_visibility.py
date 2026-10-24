"""Independent side tracking for visible torso appearance, without hidden completion."""
import copy
import numpy as np
from . import body_visibility_base as old
CONFIG={'version':'independent-side-torso-27.0','max_search_shoulder':1.35,'old_search_shoulder':.70,'max_arm_boundary_distance_shoulder':.12,'min_side_continuity_axis':.13,'max_row_step_shoulder':.055,'max_cross_view_delta_shoulder':.04,'max_broad_width_shoulder':1.55,'part_bands':{'waist':[.50,.85],'hips':[.88,1.18]}}
TARGET=[3,4,5,6,16]
def intervals(v):
 dif=np.diff(np.r_[False,v,False].astype(int));return list(zip(np.flatnonzero(dif==1),np.flatnonzero(dif==-1)-1))
def distance(p,a,b):
 d=b-a;t=np.clip(np.dot(p-a,d)/max(1e-8,np.dot(d,d)),0,1);return float(np.linalg.norm(p-(a+t*d)))
def trace(mask,models):
 h,w=mask.shape;source,p=old.select(models)
 if p is None or not all(old.good(p[i]) for i in [23,24]):return None
 xy=lambda i:np.array([p[i]['x']*w,p[i]['y']*h]);sh=(xy(11)+xy(12))/2;hip=(xy(23)+xy(24))/2;S=float(np.linalg.norm(xy(11)-xy(12)));T=hip[1]-sh[1]
 if S<40 or T<.25*S:return None
 arms=[(xy(a),xy(b)) for a,b in [(11,13),(13,15),(12,14),(14,16)] if old.good(p[a]) and old.good(p[b])];rows={}
 for y in range(max(0,round(sh[1]+.32*T)),min(h,round(sh[1]+1.3*T))):
  t=(y-sh[1])/T;cx=float(sh[0]+(hip[0]-sh[0])*t);l=max(0,round(cx-1.35*S));r=min(w,round(cx+1.35*S)+1);v=mask[y,l:r];runs=[(l+a,l+b) for a,b in intervals(np.isin(v,TARGET)) if b-a>=max(4,.015*S)]
  candidates=[(a,b) for a,b in runs if a<=cx+.75*S and b>=cx-.75*S]
  row={'y':y,'center':cx,'sides':{},'selection':'torso corridor intersects component; do not merge across arms'}
  if candidates:
   # Select a single torso-connected interval, never take extrema across all clothing islands.
   central=min(candidates,key=lambda ab:(0 if ab[0]<=cx<=ab[1] else min(abs(ab[0]-cx),abs(ab[1]-cx)),abs((ab[0]+ab[1])/2-cx)))
   row['central_interval']=list(central)
   for side in ['left','right']:
    # Trace each observed component side separately; never infer the intervening gap as body width.
    options=[]
    for a,b in candidates:
     x=a if side=='left' else b
     if side=='left' and x>cx+.1*S or side=='right' and x<cx-.1*S:continue
     adj=x-1 if side=='left' else x+1;label=int(mask[y,x]);outer=int(mask[y,adj]) if 0<=adj<w else -1;arm_dist=min((distance(np.array([x,y]),u,v) for u,v in arms),default=float('inf'));clipped=x in [l,r-1,0,w-1]
     # Background at the actual edge is observed; skeleton proximity alone cannot veto a nearby body side.
     visible=outer==0 and not clipped
     item={'point':[float(x),float(y)],'label':label,'adjacent':outer,'external':bool(visible),'arm_risk':bool(arm_dist<.12*S),'arm_distance_px':arm_dist if np.isfinite(arm_dist) else None,'scan_clipped':bool(clipped),'old_window_would_clip':bool(abs(x-cx)>=.70*S),'reason':'external' if visible else 'scan_or_image_edge' if clipped else 'contact_or_nonbackground','component_interval':[a,b]}
     options.append(item)
    if options:
     # Prefer directly visible side; choose nearest torso-side radial band, not best aesthetic curvature.
     row['sides'][side]=min(options,key=lambda item:(not item['external'],abs(abs(item['point'][0]-cx)-.45*S)))
   if len(row['sides'])==2:row['width']=row['sides']['right']['point'][0]-row['sides']['left']['point'][0]
  rows[y]=row
 return {'source':source,'shoulder':sh.tolist(),'hip':hip.tolist(),'S':S,'T':float(T),'rows':rows}

def side_support(a,b,part,side):
 S,T=a['S'],a['T'];lo,hi=CONFIG['part_bands'][part];ys=list(range(max(0,round(a['shoulder'][1]+lo*T)),round(a['shoulder'][1]+hi*T)));runs=[];current=[];rejected={};corrected=0;allpoints=[]
 for y in ys:
  ea=a['rows'].get(y,{}).get('sides',{}).get(side);eb=b['rows'].get(y,{}).get('sides',{}).get(side)
  reason=None
  if not ea or not eb:reason='missing_or_cropped'
  elif not ea['external'] or not eb['external']:reason=eb['reason'] if not eb['external'] else ea['reason']
  elif abs(ea['point'][0]-eb['point'][0])>.04*S:reason='whole_roi_disagree'
  elif current and abs(eb['point'][0]-current[-1][0])>.055*S:reason='row_boundary_jump'
  if reason:
   rejected[reason]=rejected.get(reason,0)+1
   if current:runs.append(current);current=[]
  else:
   current.append(eb['point']);allpoints.append(eb['point']);corrected+=eb['old_window_would_clip']
 if current:runs.append(current)
 best=max(runs,key=len) if runs else [];length=best[-1][1]-best[0][1] if best else 0;minimum=max(12,.13*T)
 return {'allowed':bool(length>=minimum),'continuous_y_px':float(length),'minimum_px':float(minimum),'support_points':best,'all_observed_points':allpoints,'rejected_rows':rejected,'recovered_old_window_clips':corrected,'scope':'single external side only, no opposite-side completion','total_candidate_rows':len(ys)}

def bridge(a,b,side,waist,hips):
 if not waist['allowed'] or not hips['allowed']:return {'passed':False,'reason':'同侧腰或胯支持不足'}
 start=waist['support_points'][-1];end=hips['support_points'][0];prev=start;fail=[]
 for y in range(round(start[1]),round(end[1])+1):
  x=a['rows'].get(y,{}).get('sides',{}).get(side);z=b['rows'].get(y,{}).get('sides',{}).get(side)
  if not x or not z or not x['external'] or not z['external']:fail.append([y,'遮挡或接触中断']);continue
  if abs(x['point'][0]-z['point'][0])>.04*b['S'] or abs(z['point'][0]-prev[0])>.055*b['S']:fail.append([y,'跳变或跨取景不一致'])
  prev=z['point']
 return {'passed':not fail,'from_y':start[1],'to_y':end[1],'failed_rows':fail,'reason':'同侧腰至胯真实连续外缘，无跨遮挡补线' if not fail else '腰至胯之间中断，不能证明是同一躯干边界'}

def evaluate(models,mask,whole,prior):
 out=copy.deepcopy(prior);out['torso_update_config']=CONFIG;out['legs_frozen_from']='v26 exact visibility record';a=trace(whole,models);b=trace(mask,models)
 if a is None or b is None:
  out['torso_update_reason']='可靠肩髋轴不足，保留原躯干权限';return out
 # A low, folded sitting-pose projection can put lap fabric into nominal waist/hip bands.
 lap_risk=False
 for name in ['pose_full','pose_heavy']:
  p=models.get(name,{}).get('landmarks',[])
  if len(p)==33 and all(old.good(p[i]) for i in [23,24]):
   hy=(p[23]['y']+p[24]['y'])/2
   lap_risk=lap_risk or (hy>.85 and any(p[k].get('visibility',0)>=.3 and p[k].get('presence',0)>=.3 and p[k]['y']<hy-.03 for k in [25,26]))
 for part in ['waist','hips']:
  previous=copy.deepcopy(prior['parts'][part]);sides={side:side_support(a,b,part,side) for side in ['left','right']};lo,hi=CONFIG['part_bands'][part];selected=[r for y,r in b['rows'].items() if lo<=(y-b['shoulder'][1])/b['T']<=hi];widths=[r['width'] for r in selected if 'width' in r];garment_labels=[e['label'] for r in selected for e in r['sides'].values()];broad=float(np.median(widths)/b['S']) if widths else None
  garment_fraction=sum(lab in [3,4,5,6] for lab in garment_labels)/max(1,len(garment_labels));jumps=0
  for side in sides:
   pts=sides[side]['all_observed_points'];jumps+=sum(abs(pts[i][0]-pts[i-1][0])>.12*b['S'] for i in range(1,len(pts)))
  # Width expansion alone is NOT a veto. Broad cloth plus discontinuity/absent preceding-side support is ambiguous.
  preceding={side:side_support(a,b,'waist',side)['allowed'] for side in sides} if part=='hips' else None
  garment_risk=bool(garment_fraction>.7 and broad is not None and broad>1.55 and (jumps>0 or part=='hips' and not any(preceding.values())))
  allowed_sides=[side for side,o in sides.items() if o['allowed']]
  # Reject hanging skirt boundaries without any observed waist-to-hip side continuity.
  bridges={}
  if part=='hips' and garment_fraction>.7:
   for side in allowed_sides:bridges[side]=bridge(a,b,side,side_support(a,b,'waist',side),sides[side])
   allowed_sides=[side for side in allowed_sides if bridges[side]['passed'] and out['parts']['waist']['allowed'] and side in out['parts']['waist'].get('allowed_sides',[])]
  no_transition=bool(part=='hips' and garment_fraction>.7 and not allowed_sides)

  high_occlusion=previous.get('arm_hand_fraction',0)>.4
  allow=bool(allowed_sides) and not garment_risk and not no_transition and not lap_risk and not high_occlusion
  reason='坐姿投影与画面边缘使腰胯候选区可能落在腿上覆盖的衣物上，保持不确定' if lap_risk else '手臂/手遮挡核心区域过多，不能确定边界属于躯干' if high_occlusion else '单侧外缘连续可见，仅评对应侧局部着装轮廓' if allow else '宽衣物与跳变/腰胯衔接缺失共同构成风险' if garment_risk else '胯侧缺少同侧可见腰部衔接，不认定为身体轮廓' if no_transition else '没有足够连续、身份可核对的单侧外缘'
  pts=[p for side in allowed_sides for p in sides[side]['support_points']];box=previous.get('box')
  if pts:
   xs=[p[0] for p in pts];ys=[p[1] for p in pts];center=float(np.mean([r['center'] for r in selected])) if selected else np.mean(xs);margin=.1*b['S'];box=[max(0,round(min(min(xs),center)-margin)),max(0,round(min(ys)-.04*b['T'])),min(mask.shape[1],round(max(max(xs),center)+margin)),min(mask.shape[0],round(max(ys)+.04*b['T']))]
  out['parts'][part]={**previous,'allowed':allow,'state':'partial' if allow else 'uncertain','reason':reason,'box':box,'side_checks':sides,'allowed_sides':allowed_sides if allow else [],'risks':{'median_surface_width_over_shoulder':broad,'garment_edge_fraction':garment_fraction,'boundary_jump_count':jumps,'broad_cloth_with_additional_risk':garment_risk,'missing_same_side_waist_transition':no_transition,'same_side_waist_support':preceding,'folded_lap_pose_risk':bool(lap_risk),'high_core_arm_occlusion':high_occlusion,'waist_hip_bridges':bridges},'previous_permission':{'allowed':previous['allowed'],'state':previous['state'],'reason':previous['reason']},'restrictions':['只能评价'+('、'.join(allowed_sides) if allow else '未获准')+'侧局部轮廓','不得推断完整宽度、真实围度、完整腰臀比','不得推断背面臀形、饱满度或隐藏部分','可见衣物曲线不是已确认的解剖身体边缘']}
 return out
