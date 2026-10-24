"""Candidate parameters retained independently from visibility quality gates."""
import copy
from .body_regional import evaluate as regional
from .body_visibility import trace
from .body_parameter_geometry import extract,enclose
VERSION='body-candidate-parameters-v31-production-1'

def evaluate(models,mask,whole,prior):
 out=regional(models,mask,whole,prior);t=trace(mask,models);h,w=mask.shape
 for k in ['waist','hips']:
  p=out['parts'][k];p['previous_gate_allowed']=p['allowed'];p['previous_gate_reason']=p['reason'];p['old_box']=copy.deepcopy(p.get('box'));features,widths=extract(t,k);flags=[]
  for side,g in p.get('regional_side_checks',{}).items():
   if not g['hip_anchor_ok']:flags.append(side+':髋点距离不稳定')
   if g.get('pose_hip_agreement_shoulder') is None or g['pose_hip_agreement_shoulder']>.08:flags.append(side+':双姿态位置缺失或不一致')
   if not g['side_continuity']['allowed']:flags.append(side+':连续边界不足')
  if not p.get('regional_side_checks'):flags.append('无完整肩髋轴：候选框来自旧定位，不视为解剖实测')
  flags += [name for name,value in p.get('risks',{}).items() if value is True]
  if p.get('lower_garment_check',{}).get('risk'):flags.append('长裙悬垂风险')
  if any(f['boundary_type']=='contact_candidate' for f in features.values()):flags.append('接触边界候选：可能属于衣物/手臂接触，不是已确认身体外缘')
  box=enclose(p.get('box'),features,w,h);valid=bool(box and box[2]>box[0] and box[3]>box[1])
  p.update(allowed=valid,state='candidate' if valid else 'missing',box=box,candidate_parameters=features,width_parameters=widths,scoring_mode='parameter_assisted' if features else 'visual_only',uncertainty_flags=flags,allowed_regions=['candidate_visible_contour'] if valid else [],reason='保留外缘及接触候选并分别标记；同高度配对宽度仅为照片投影候选' if features else '没有可提取边界参数，仅能视觉试评',restrictions=['参数为照片投影候选，不是真实身体尺寸','边界缺口不补齐，模型不得伪造比例','可评价贴身衣物呈现；袖子裙摆和完全遮挡不作为身体证据'])
 out['section_relations']={};out['parameter_version']=VERSION
 for kind in ['external_pair','contact_candidate_pair']:
  wa=out['parts']['waist']['width_parameters'].get(kind);hi=out['parts']['hips']['width_parameters'].get(kind)
  if wa and hi:out['section_relations'][kind]={'waist_section_to_hip_section_width':wa['width_median_px']/hi['width_median_px'],'scope':'候选分区中位投影宽之比，不是真实腰围臀围比','source':kind}
 assert all(out['parts'][k]==prior['parts'][k] for k in ['chest','legs'])
 return out

def summary(v):
 return {k:{'candidate_available':p['allowed'],'box':p.get('box'),'scoring_mode':p['scoring_mode'],'parameters':{name:{key:value for key,value in f.items() if key!='observed_points'} for name,f in p['candidate_parameters'].items()},'width_parameters':{name:{key:value for key,value in f.items() if key!='rows'} for name,f in p['width_parameters'].items()},'uncertainty_flags':p['uncertainty_flags']} for k in ['waist','hips'] for p in [v['parts'][k]]}
