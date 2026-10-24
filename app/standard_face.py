"""Standard face-only input and isolated review; existing weights, no batch fitting.

python -m app.standard_face --trial data/trials/holistic-20261022-review
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DATA_DIR
from .engine import file_fingerprint,signature
from .face_preprocessing import identity as base_identity,infer,landmarks,perturbations,scale_score
from .face_preprocessing_trial import agreement,ranks
from .holistic_trial import snapshot,write_json
from .images import oriented_rgb

MARGIN=.12
FOLDER="standard-face-v1"


def standard_crop(image: Image.Image, points: np.ndarray):
    """Equal pixel scales on both axes; fixed face extent, not source framing.

    Floating source coordinates avoid integer crop-boundary jumps. PIL bilinear
    resampling matches the existing model adapter, including downsampling.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[0]<468 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError("面部关键点无效，不使用原取景代替标准裁脸")
    low,high=points[:468].min(axis=0),points[:468].max(axis=0)
    side=float(max(high-low)*(1+2*MARGIN));center=(low+high)/2
    if min(high-low)<40 or side<60:raise ValueError("可见面部像素不足")
    left,top=center-side/2;right,bottom=center+side/2
    if min(low)<0 or high[0]>image.width or high[1]>image.height:
        raise ValueError("面部关键点超出原图，无法用补边恢复不可见的面部")
    box=(float(left),float(top),float(right),float(bottom))
    # Local coordinates avoid PIL's absolute float-coordinate rounding varying
    # with the surrounding canvas. Keep interpolation support outside the ROI.
    guard=int(np.ceil(side/224))+2
    padding={"left":max(0,int(np.ceil(-left))+guard),"top":max(0,int(np.ceil(-top))+guard),
             "right":max(0,int(np.ceil(right-image.width))+guard),"bottom":max(0,int(np.ceil(bottom-image.height))+guard)}
    # Only add a border when the requested square actually leaves the image.
    padded=bool(min(left,top)<0 or right>image.width or bottom>image.height)
    if not padded:padding={key:0 for key in padding}
    canvas=image.convert("RGB")
    if padded:
        if (image.width+padding['left']+padding['right'])*(image.height+padding['top']+padding['bottom'])>100_000_000:
            raise ValueError("标准裁脸补边范围过大，请使用更紧凑的原图")
        pixels=np.pad(np.asarray(canvas),((padding['top'],padding['bottom']),(padding['left'],padding['right']),(0,0)),mode='edge')
        canvas=Image.fromarray(pixels)
    shift_x,shift_y=padding['left'],padding['top']
    actual_box=(left+shift_x,top+shift_y,right+shift_x,bottom+shift_y)
    origin_x=max(0,int(np.floor(actual_box[0]))-guard);origin_y=max(0,int(np.floor(actual_box[1]))-guard)
    patch=canvas.crop((origin_x,origin_y,min(canvas.width,int(np.ceil(actual_box[2]))+guard),min(canvas.height,int(np.ceil(actual_box[3]))+guard)))
    relative=tuple(round(v-o,6) for v,o in zip(actual_box,(origin_x,origin_y,origin_x,origin_y)))
    output=patch.resize((224,224),Image.Resampling.BILINEAR,box=relative)
    return output,{"source_box":list(box),"face_extent":(high-low).tolist(),"margin":MARGIN,
                   "long_face_extent_fraction":1/(1+2*MARGIN),"rotation_degrees":0.,
                   "resize":"PIL BILINEAR float source box","padding":padded,
                   "padding_pixels":padding,"padding_method":"edge" if padded else None}


def prepare(image: Image.Image,target: dict):
    points,_,_=landmarks(image,target)
    return standard_crop(image,points)


def assessment(raw: dict) -> dict:
    """Publish the selected overall-only method without inventing facet scores."""
    if raw.get("status") != "completed":
        raise ValueError("标准面部评分未完成")
    scaled = scale_score(raw.get("raw_score"))
    if raw.get("total") != scaled["total"]:
        raise ValueError("标准面部总分与固定换算不一致")
    return {"module": "face", "status": "completed", "total": scaled["total"],
            "dimensions": [], "weights": {}, "scoring_method": "standard-face-v1",
            "rubric_version": "standard-face-1.0", "prompt_version": "local-standard-face-no-prompt",
            "summary": "", "strengths": [], "limitations": [], "shooting_tips": [], "editing_tips": [],
            "raw_score": raw["raw_score"], "unclipped_total": scaled["unclipped_total"],
            "display_clipped": scaled["clipped"], "explanations_only": False}


def identity():
    base=base_identity()
    return {"version":"standard-face-20261022-1","model":base["model"],"weights":base["weights"],
            "dependencies":base["dependencies"],"scale":base["scale"],"margin":MARGIN,
            "implementation":{p:file_fingerprint(Path(__file__).parent/p) for p in
                              ("standard_face.py","face_preprocessing.py","beauty_model.py","images.py")},
            "reference_used_for_scoring":False}


def score(image,target,directory):
    actual,diagnostics=prepare(image,target)
    path=directory/"face.png";actual.save(path,"PNG")
    main={**infer(actual),"crop":diagnostics,"input_file_hash":file_fingerprint(path)}
    checks={}
    for name,changed,box in perturbations(image,target):
        try:
            crop,diag=prepare(changed,box);file=directory/f"{name}.png";crop.save(file,"PNG")
            value={**infer(crop),"crop":diag,"input_file_hash":file_fingerprint(file)}
            value["delta"]=value["unclipped_total"]-main["unclipped_total"]
            checks[name]=value
        except Exception as exc:checks[name]={"status":"failed","error":str(exc)}
    changes=[abs(v["delta"]) for v in checks.values() if v["status"]=="completed"]
    return {"status":"completed","primary":main,"checks":checks,"stability_pass":len(changes)==2 and max(changes)<=2,
            "maximum_delta":max(changes) if changes else None}


def replace_column(page: str, rows: list[dict], outcomes: dict):
    found=re.search(r"<section id='face-previous'>.*?</section>",page,re.S)
    if not found:raise ValueError("没有找到当前颜值对比列")
    section=found.group();body=re.search(r"<tbody>(.*?)</tbody>",section,re.S)
    source_rows=re.findall(r"<tr>.*?</tr>",body.group(1),re.S)
    if len(source_rows)!=len(rows):raise ValueError("照片数量不匹配")
    rendered=[]
    for row in rows:
        matches=[r for r in source_rows if '<td>'+html.escape(row['label'])+'<small>' in r]
        if len(matches)!=1:raise ValueError("照片行不唯一")
        cells=re.findall(r"<td(?:\s[^>]*)?>.*?</td>",matches[0],re.S)
        if len(cells)!=7:raise ValueError("对比列结构不匹配")
        result=outcomes[row['id']];primary=result.get('primary',{})
        raw=primary.get('raw_score');raw_text=f"{raw:.3f}" if raw is not None else '—'
        cells[1]=f"<td>{html.escape(row['label'])}<small>原值 {raw_text}</small></td>"
        score_text=f"{row['standard']:.1f}" if row['standard'] is not None else '—'
        cells[5]=f"<td class='revised'>{score_text}<small>#{row.get('standard_rank') or '—'}</small></td>"
        if result['status']=='completed':
            warning='；显示截顶' if primary['clipped'] else ''
            check='波动≤2分' if result['stability_pass'] else '稳定性需复核'
            cells[6]=f"<td><small>{check}{warning}</small><details><summary>标准面部输入</summary><img class='crop' src='{FOLDER}/{row['id']}/face.png' alt='标准面部裁脸'></details></td>"
        else:cells[6]='<td>'+html.escape(result['error'])+'</td>'
        rendered.append((row.get('standard_rank') or 999,'<tr>'+''.join(cells)+'</tr>'))
    table=section[section.index('<table>'):body.start(1)]+''.join(r for _,r in sorted(rendered))+section[body.end(1):]
    table=re.sub(r"<th>(?:原取景基线版|颜值修正版|标准面部版) / 排名</th>",'<th>标准面部版 / 排名</th>',table)
    note="<section id='face-previous'><h2>颜值对比</h2><p>当前列：标准面部版。按关键点统一面部范围与边距，PIL双线性缩放，不旋转、不根据单张分数选择裁脸。</p>"
    note+="<p>本轮重新评估颜值；五个分项未重算，身材和摄影保持原结果。正式图库未批量更新。</p>"
    return page[:found.start()]+note+table+page[found.end():]


def run(trial: Path):
    baseline=json.loads((trial/'baseline.json').read_text());before=snapshot(DATA_DIR/'portrait.db')
    if before['fingerprints']!=baseline['fingerprints']:raise ValueError("生产数据与试评基线不同")
    engine=identity();output=trial/FOLDER;output.mkdir(exist_ok=True)
    manifest={'engine':engine,'samples':[{k:s[k] for k in ('id','sha256','target_json')} for s in baseline['samples']]}
    file=output/'manifest.json'
    if file.exists() and json.loads(file.read_text())!=manifest:raise ValueError("标准裁脸版本变化，不能混用结果")
    if not file.exists():write_json(file,manifest)
    outcomes={};rows=[]
    for sample in baseline['samples']:
        aid=sample['id'];source=DATA_DIR/'assets'/aid;directory=output/aid;directory.mkdir(exist_ok=True)
        if not sample['adult_confirmed'] or file_fingerprint(source)!=sample['sha256']:raise ValueError("样本确认或原图哈希不匹配")
        key=signature({'manifest':manifest,'id':aid});checkpoint=directory/'result.json'
        cached=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        if cached.get('status')=='completed':
            if cached.get('cache_key')!=key:raise ValueError("缓存身份不匹配")
            for name,value in [('face',cached['primary']),*cached['checks'].items()]:
                if value.get('input_file_hash') and file_fingerprint(directory/f'{name}.png')!=value['input_file_hash']:raise ValueError("评分输入发生变化")
            result=cached
        else:
            try:result=score(oriented_rgb(source),json.loads(sample['target_json']),directory)
            except Exception as exc:result={'status':'failed','error':str(exc)}
            result.update(cache_key=key,engine=engine);write_json(checkpoint,result)
        outcomes[aid]=result
        rows.append({'id':aid,'label':sample['label'],'created_at':sample['created_at'],'standard':result.get('primary',{}).get('total')})
        print(sample['label'],rows[-1]['standard'],result.get('maximum_delta'),flush=True)
    positions=ranks(rows,'standard')
    for row in rows:
        row['standard_rank']=positions.get(row['id'])
        row['codex']=json.loads((trial/row['id']/'result.json').read_text())['results']['face']['total']
    after=snapshot(DATA_DIR/'portrait.db')
    if before['fingerprints']!=after['fingerprints'] or identity()!=engine or any(file_fingerprint(DATA_DIR/'assets'/s['id'])!=s['sha256'] for s in baseline['samples']):raise ValueError("原图、引擎或生产数据变化")
    report={'rows':rows,'engine':engine,'statistics':agreement(rows,'standard'),'stability_pass_count':sum(o.get('stability_pass',False) for o in outcomes.values()),'preservation':{'unchanged':True,'fingerprints':after['fingerprints']}}
    write_json(output/'comparison.json',report)
    page=trial/'comparison.html';original=page.read_text();changed=replace_column(original,rows,outcomes)
    backup=trial/'comparison.before-standard-face.html'
    if not backup.exists():backup.write_text(original)
    temp=trial/'comparison.html.tmp';temp.write_text(changed);temp.replace(page)
    write_json(trial/'display-selection.json',{'face_column':'标准面部版','source':FOLDER+'/comparison.json','score_key':'standard','rank_key':'standard_rank','production_updated':False})
    print(json.dumps({'statistics':report['statistics'],'stable':report['stability_pass_count'],'production_unchanged':True},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trial',type=Path,required=True)
    run(parser.parse_args().trial.resolve())
