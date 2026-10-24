"""Deterministic comparisons of saved scores, never a second visual model judgment."""
from __future__ import annotations
import json
from itertools import combinations
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt, model_validator
from . import storage as s
from .scoring import photo_total_score, RULES, total_score, body_total_score, face_total_score, BODY_RUBRIC_VERSION, FACE_RUBRIC_VERSION, is_overall_face, is_overall_photo

router=APIRouter(prefix='/api')
Module=Literal['face','photography','body']

class Preference(BaseModel):
    name: str = Field(min_length=1,max_length=100)
    module: Module
    weights: dict[str,StrictInt]

    @model_validator(mode='after')
    def valid(self):
        if set(self.weights)!={r[0] for r in RULES[self.module]} or any(type(w) is not int or w<0 for w in self.weights.values()) or sum(self.weights.values())!=100:
            raise ValueError('权重必须覆盖全部维度，总和为100')
        return self

@router.get('/preferences')
def preferences():
    return [{**r,'weights':json.loads(r['weights_json'])} for r in s.many('SELECT * FROM preferences ORDER BY created_at')]

@router.post('/preferences')
def save_preference(data:Preference):
    pid=s.new_id()
    s.run('INSERT INTO preferences VALUES(?,?,?,?,?)',(pid,data.name,data.module,json.dumps(data.weights),s.now()))
    return {'id':pid}

@router.delete('/preferences/{preference_id}')
def delete_preference(preference_id:str):
    s.run('DELETE FROM preferences WHERE id=?',(preference_id,))
    return {'status':'deleted'}

def assessment(asset:dict,module:str,weights:dict):
    if module in ("face","body") and not asset["adult_confirmed"]:
        return None,"未确认成年状态"
    report=json.loads(asset['report_json'] or '{}')
    result=report.get(module,{})
    standard_face = module=='face' and is_overall_face(result)
    if module=='face' and not standard_face and result.get('rubric_version') not in (FACE_RUBRIC_VERSION, 'portrait-face-0.8'):
        return None,'颜值评分待更新'
    current_body = module=='body' and result.get('scoring_method')=='visible-body-fixed-zero-v1'
    if module=='body' and not current_body and result.get('rubric_version')!=BODY_RUBRIC_VERSION:
        return None,'身材评分待更新'
    dims=result.get('dimensions',[])
    if result.get('total') is None or result.get('status')!='completed':
        return None,'缺少完整可评结果'
    if result.get('engine'):
        from .engine import signature as engine_signature
        signature=(result.get('rubric_version'), result.get('prompt_version'), engine_signature(result['engine']))
    else:
        signature=tuple(result.get(k) for k in ('rubric_version','prompt_version','model','reasoning_effort'))
    if any(not v for v in signature):return None,'缺少评估版本信息'
    overall_photo = module == 'photography' and is_overall_photo(result)
    if overall_photo and weights != {r[0]:r[2] for r in RULES['photography']}:
        return None,'整体摄影模型不支持分项权重重算'
    if standard_face and weights != {r[0]:r[2] for r in RULES['face']}:
        return None,'整体颜值模型不支持分项权重重算'
    if current_body:
        from .body_model import WEIGHTS as BODY_WEIGHTS
        if weights not in (BODY_WEIGHTS, {r[0]:r[2] for r in RULES['body']}):return None,'当前身材规则使用固定权重'
    try: score=result['total'] if current_body else result['total'] if standard_face or overall_photo or (module=='body' and result.get('score_basis')=='no_visible_body') else (body_total_score if module=='body' else face_total_score if module=='face' else photo_total_score)({d['key']:d['score'] for d in dims},weights)
    except (ValueError,KeyError):return None,'分项数据不兼容'
    if score is None:return None,'当前权重下可评项目不足'
    return {'asset_id':asset['id'],'filename':asset['filename'],'report_id':report['id'],
            'score':score,'original_score':result['total'],'signature':signature,
            'dimensions':dims,'limitations':result.get('limitations',[]),
            'preview_url':f"/api/assets/{asset['id']}/preview",'crop_url':f"/api/assets/{asset['id']}/crop" if report.get('target') else None,'target':report.get('target')},None

def options(module,preference_id):
    weights={r[0]:r[2] for r in RULES[module]}
    name='默认规则'
    if preference_id:
        p=s.one('SELECT * FROM preferences WHERE id=?',(preference_id,))
        if not p or p['module']!=module:raise HTTPException(400,'偏好不存在或模块不一致')
        weights=json.loads(p['weights_json']);name=p['name']
    return weights,name

@router.get('/projects/{project_id}/ranking')
def ranking(project_id:str,module:Module='face',preference_id:str|None=None):
    if not s.get_project(project_id):raise HTTPException(404,'项目不存在')
    weights,name=options(module,preference_id)
    entries=[];excluded=[];signature=None
    groups=s.many('SELECT * FROM candidate_groups WHERE project_id=? ORDER BY created_at',(project_id,))
    if len(groups)>32:raise HTTPException(400,'单次对照最多32组，请缩小项目范围')
    for group in groups:
        asset=s.get_asset(group['representative_id']) if group['representative_id'] else None
        if not asset or (not asset['retained'] and asset['expires_at']<s.now()):
            excluded.append({'group':group['name'],'reason':'未选择有效代表照'});continue
        item,error=assessment(asset,module,weights)
        if error:excluded.append({'group':group['name'],'reason':error});continue
        if signature and signature!=item['signature']:
            excluded.append({'group':group['name'],'reason':'模型或规则版本不一致，需要重新评估'});continue
        signature=item['signature'];item.update(group_id=group['id'],group=group['name']);entries.append(item)
    if len(entries)>32:raise HTTPException(400,'单次排行最多32组，请缩小项目范围')
    pairs=[]
    for a,b in combinations(entries,2):
        difference=round(a['score']-b['score'],1)
        bd={d['key']:d for d in b['dimensions']}
        pairs.append({'a':a['group'],'b':b['group'],'a_id':a['asset_id'],'b_id':b['asset_id'],
                      'difference':difference,'outcome':'near_tie' if abs(difference)<=2 else 'a' if difference>0 else 'b',
                      'dimension_differences':[{'key':d['key'],'difference':round(d['score']-bd[d['key']]['score'],1) if d['score'] is not None and bd[d['key']]['score'] is not None else None,
                                                'a_evidence':d['evidence'],'b_evidence':bd[d['key']]['evidence']} for d in a['dimensions']]})
    eligible_ids={e['group_id'] for e in entries}
    for a,b in combinations(groups,2):
        if a['id'] in eligible_ids and b['id'] in eligible_ids:continue
        pairs.append({'a':a['name'],'b':b['name'],'a_id':a['representative_id'],'b_id':b['representative_id'],
                      'difference':None,'outcome':'incomparable','dimension_differences':[],
                      'reason':'至少一方缺少有效代表照、完整可评结果或兼容版本，不能据此判定胜负'})
    entries.sort(key=lambda r:(-r['score'],r['group']))
    baseline=sorted(entries,key=lambda r:(-r['original_score'],r['group']))
    for i,e in enumerate(entries):
        e['position']=i+1;e['baseline_position']=next(j+1 for j,b in enumerate(baseline) if b['asset_id']==e['asset_id'])
        e['near_tie']=any(p['outcome']=='near_tie' and e['asset_id'] in (p['a_id'],p['b_id']) for p in pairs)
    return {'module':module,'preference':name,'weights':weights,'entries':entries,'excluded':excluded,'pairs':pairs,
            'method':'已保存分项的确定性对照，不是独立的模型两图复评',
            'notice':'差距≤2/100标为近似分；该阈值是实验规则，并非统计置信区间。角度、光线和遮挡仍影响可比性。位置仅便于浏览，不表示确定优劣。',
            'missing_comparisons':len(excluded)>0,'independent_visual_comparisons':False}

@router.get('/projects/{project_id}/groups/{group_id}/recommendation')
def recommendation(project_id:str,group_id:str,module:Module='face'):
    group=s.one('SELECT * FROM candidate_groups WHERE id=? AND project_id=?',(group_id,project_id))
    if not group:raise HTTPException(404,'分组不存在')
    weights,_=options(module,None);eligible=[];excluded=[];signature=None
    for row in s.many('SELECT a.* FROM assets a JOIN group_members m ON m.asset_id=a.id WHERE m.group_id=? AND a.deleted_at IS NULL',(group_id,)):
        if not row['retained'] and row['expires_at']<s.now():continue
        item,error=assessment(row,module,weights)
        if error:excluded.append({'asset_id':row['id'],'reason':error});continue
        if signature and item['signature']!=signature:
            excluded.append({'asset_id':row['id'],'reason':'模型或规则版本不兼容'});continue
        signature=item['signature'];eligible.append(item)
    eligible.sort(key=lambda r:-r['score'])
    return {'recommended_id':eligible[0]['asset_id'] if eligible else None,'target':module,'candidates':eligible,'excluded':excluded,
            'reason':'同组兼容且完整可评的结果中，当前目标分数最高；请结合角度、遮挡与分项依据确认。' if eligible else '没有条件充分的可评结果，不强行推荐。'}

@router.get('/projects/{project_id}/ranking.html')
def ranking_html(project_id:str,module:Module='face',preference_id:str|None=None):
    import html
    from fastapi.responses import HTMLResponse
    result=ranking(project_id,module,preference_id)
    rows=''.join(f"<tr><td>{e['position']}</td><td>{html.escape(e['group'])}</td><td>{e['score']}</td><td>{'近似分' if e['near_tie'] else ''}</td></tr>" for e in result['entries'])
    excluded=''.join(f"<li>{html.escape(e['group'])}：{html.escape(e['reason'])}</li>" for e in result['excluded'])
    return HTMLResponse(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>照片排行</title><style>body{{max-width:960px;margin:40px auto;padding:20px;font:16px/1.8 system-ui}}td,th{{padding:8px 20px}}</style><h1>照片排行</h1><table><tr><th>排名</th><th>分组</th><th>分数</th><th>提示</th></tr>{rows}</table><ul>{excluded}</ul></html>',headers={'Content-Disposition':'attachment; filename="portrait-comparison.html"'})
