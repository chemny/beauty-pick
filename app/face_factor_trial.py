"""Controlled local input tests; review only, never write to production SQLite.

python -m app.face_factor_trial --trial data/trials/holistic-20261022-review
"""
from __future__ import annotations

import argparse
import base64
import copy
import html
import io
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DATA_DIR
from .engine import file_fingerprint,signature
from .face_factors import GROUPS,LABELS,SPECS,distance,geometry,identity,infer,prepare_inputs
from .face_preprocessing_trial import agreement,ranks
from .holistic_trial import snapshot,write_json
from .images import oriented_rgb

FOLDER="face-single-factor-v1"


def evaluate(image,target,directory):
    directory.mkdir(parents=True,exist_ok=True)
    base,points,error=geometry(image,target)
    variants={}
    for name,prepared in prepare_inputs(image,base,points,error).items():
        record={k:v for k,v in prepared.items() if k!="image"}
        if prepared["status"]=="ready":
            path=directory/f"{name}.png";prepared["image"].save(path,"PNG")
            try:
                record.update(infer(prepared["image"]),input_file_hash=file_fingerprint(path))
            except Exception as exc:
                record.update(status="failed",total=None,error=str(exc))
        variants[name]=record
    return {"status":"completed" if variants["baseline"]["status"]=="completed" else "failed",
            "base_box":base,"points":points.tolist() if points is not None else None,
            "geometry_error":error,"variants":variants}


def comparison(samples,outcomes,references):
    rows=[]
    for s in samples:
        variants=outcomes[s["id"]]["variants"]
        rows.append({"id":s["id"],"label":s["label"],"created_at":s["created_at"],
                     "codex":references[s["id"]],**{name:v.get("total") for name,v in variants.items()}})
    for name in SPECS:
        positions=ranks(rows,name)
        for row in rows:row[name+"_rank"]=positions.get(row["id"])
    groups={}
    for key,group in GROUPS.items():
        common=[r for r in rows if all(r.get(v) is not None for v in group["variants"])]
        comparisons={}
        control=group["control"]
        for variant in group["variants"]:
            differences=[distance(outcomes[r["id"]]["variants"][variant]["raw_score"],outcomes[r["id"]]["variants"][control]["raw_score"]) for r in common]
            pair_count=inversions=0
            for i,a in enumerate(common):
                for b in common[i+1:]:
                    baseline_gap=distance(outcomes[a["id"]]["variants"][control]["raw_score"],outcomes[b["id"]]["variants"][control]["raw_score"])
                    if abs(baseline_gap)<2:continue
                    variant_gap=distance(outcomes[a["id"]]["variants"][variant]["raw_score"],outcomes[b["id"]]["variants"][variant]["raw_score"])
                    pair_count+=1;inversions+=int(variant_gap*baseline_gap<0)
            stats=agreement(common,variant)
            # Rank evaluation uses raw outputs so display clipping cannot alter metrics.
            raw_rows=[{**r,variant:outcomes[r["id"]]["variants"][variant]["raw_score"]} for r in common]
            stats["raw_order_agreement"]=agreement(raw_rows,variant)["order_agreement"]
            comparisons[variant]={**stats,"median_absolute_delta":float(np.median(np.abs(differences))) if differences else None,
                                  "maximum_absolute_delta":max(map(abs,differences)) if differences else None,
                                  "over_2_points":sum(abs(d)>2 for d in differences),
                                  "strong_pair_count":pair_count,"strong_pair_inversions":inversions}
        groups[key]={**group,"common_ids":[r["id"] for r in common],"common_count":len(common),
                     "excluded":[{"id":r["id"],"label":r["label"],"reasons":{v:outcomes[r["id"]]["variants"][v].get("error") for v in group["variants"] if r[v] is None}} for r in rows if r not in common],
                     "comparisons":comparisons}
    return {"rows":rows,"groups":groups,"baseline_statistics":agreement(rows,"baseline"),
            "reference_is_ground_truth":False,"baseline":"baseline","selected_winner":None}


def fmt(value):return "—" if value is None else f"{value:.1f}"
def pct(value):return "—" if value is None else f"{value*100:.1f}%"


def render(trial,result,outcomes):
    backup=trial/"comparison.face-v2.html"
    source=(backup if backup.exists() else trial/"comparison.html").read_text()
    start=source.find("<section id='face'>")
    if start<0:raise ValueError("原对比页没有颜值区域")
    previous=source[start:]
    previous=previous.replace("<section id='face'>","<section id='face-previous'>",1)
    previous=re.sub(r"</html>\s*$","",previous)
    style=re.search(r"<style>(.*?)</style>",source,re.S).group(1)
    style+=" .factor-summary td,.factor-summary th{white-space:nowrap} .baseline{background:#edf4ed;font-weight:700}.test-grid{display:flex;gap:16px;flex-wrap:wrap}.test-grid figure{width:160px}.test-grid img{width:144px;height:144px;object-fit:contain}.sample-link{font-size:12px}"
    sections=[];md=["# 原取景基线与单因素试验","","不拟合参考，不挑选逐图最优。各组仅用组内共同有效样本比较。",""]
    for key,group in result["groups"].items():
        summary=[]
        md += [f"## {group['name']}","",f"共同有效样本：{group['common_count']}/12。","","|输入|参考排序一致率（原值）|相对对照绝对分差中位数|最大分差|超过2分|明确分差排序反转|","|---|---:|---:|---:|---:|---:|"]
        for name,stats in group["comparisons"].items():
            changes=f"{stats['strong_pair_inversions']}/{stats['strong_pair_count']}"
            summary.append(f"<tr><td>{LABELS[name]}</td><td>{pct(stats['raw_order_agreement'])}</td><td>{fmt(stats['median_absolute_delta'])}</td><td>{fmt(stats['maximum_absolute_delta'])}</td><td>{stats['over_2_points']}/{group['common_count']}</td><td>{changes}</td></tr>")
            md.append(f"|{LABELS[name]}|{pct(stats['raw_order_agreement'])}|{fmt(stats['median_absolute_delta'])}|{fmt(stats['maximum_absolute_delta'])}|{stats['over_2_points']}/{group['common_count']}|{changes}|")
        cases=[]
        for row in sorted(result["rows"],key=lambda r:r["baseline_rank"] or 999):
            aid=row["id"];v=outcomes[aid]["variants"]
            cells=[];images=[]
            for name in group["variants"]:
                record=v[name];control=v[group["control"]]
                delta=distance(record["raw_score"],control["raw_score"]) if record["status"]==control["status"]=="completed" else None
                cell=f"{fmt(record.get('total'))}<small>{('变化 '+format(delta,'+.1f')) if delta is not None else html.escape(record.get('error',''))}</small>"
                if record.get("clipped"):cell+="<small>显示截顶</small>"
                cells.append(f"<td class='{'baseline' if name==group['control'] else ''}'>{cell}</td>")
                if record["status"]=="completed":
                    url=f"{FOLDER}/{aid}/{name}.png"
                    images.append(f"<figure><img src='{url}' alt='{LABELS[name]}'><figcaption>{LABELS[name]}<br>原值 {record['raw_score']:.3f}</figcaption></figure>")
            with Image.open(trial/aid/"whole.jpg") as photo:
                photo.thumbnail((120,160));buf=io.BytesIO();photo.save(buf,"JPEG",quality=75)
            image="data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()
            common=row["id"] in group["common_ids"]
            detail=f"<details><summary>实际输入</summary><div class='test-grid'>{''.join(images)}</div></details>"
            cases.append(f"<tr><td><img src='{image}' alt='{html.escape(row['label'],quote=True)}'></td><td>{html.escape(row['label'])}<small>{'纳入本组统计' if common else '部分输入不可比，未纳入本组统计'}</small></td><td>{fmt(row['codex'])}</td>{''.join(cells)}<td>{detail}</td></tr>")
        notes={"crop":"同一中心、同一PIL双线性缩放，不旋转，只改变原取景边长。超出原图或截断面部关键点的输入不参与比较。",
               "resize":"完全相同的原取景像素与裁脸边界，不旋转，只更换缩放算法。",
               "rotation":"两个输入共用原取景85%边长、同一中心、同一OpenCV双线性仿射采样，只改变旋转角度；不重新拟合画幅。"}
        header="".join(f"<th>{LABELS[v]}</th>" for v in group["variants"])
        sections.append(f"<section id='factor-{key}'><h2>{group['name']}</h2><p>{notes[key]} 共同有效 {group['common_count']}/12 张。</p><table class='factor-summary'><tr><th>输入</th><th>参考排序一致率</th><th>绝对分差中位数</th><th>最大分差</th><th>超过2分</th><th>相对对照顺序反转</th></tr>{''.join(summary)}</table><details><summary>查看12张逐图结果与实际输入</summary><table><thead><tr><th>照片</th><th>样本</th><th>Codex参考</th>{header}<th>裁脸</th></tr></thead><tbody>{''.join(cases)}</tbody></table></details></section>")
    baseline=result["baseline_statistics"]
    page=f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>BeautyPick · 裁脸单因素对照</title><style>{style}</style><h1 id='face'>原取景基线 · 单因素对照</h1><p>本地模型与固定换算不变。原取景基线参考排序一致率 {pct(baseline['order_agreement'])}；这不是人工准确率。各组共同有效样本可能不同，不能把不同组的百分比直接比较。</p><p>分差按未截顶值计算；顺序统计使用模型原值。只统计对照原值换算差至少2分的照片对发生的反转。每种输入都保留，不逐图挑分、不修排名。正式图库未更新。</p><nav><a href='#factor-crop'>裁脸松紧</a><a href='#factor-resize'>缩放方式</a><a href='#factor-rotation'>旋转</a><a href='#face-previous'>此前评分对比</a><a href='#body'>身材</a><a href='#photography'>摄影</a></nav>{''.join(sections)}<p><a href='comparison.face-v2.html'>上一轮完整对比页</a></p>{previous}</html>"
    if not backup.exists():backup.write_text(source,encoding="utf-8")
    temp=trial/"comparison.html.tmp";temp.write_text(page,encoding="utf-8");temp.replace(trial/"comparison.html")
    (trial/FOLDER/"comparison.md").write_text("\n".join(md),encoding="utf-8")


def run(trial: Path, *, report_only=False):
    baseline=json.loads((trial/"baseline.json").read_text());before=snapshot(DATA_DIR/"portrait.db")
    if before["fingerprints"]!=baseline["fingerprints"]:raise ValueError("生产数据与试评基线不同")
    engine=identity();samples=baseline["samples"]
    if any(v is None for v in engine["weights"].values()):raise ValueError("现有模型缺失，不下载或替换")
    manifest={"engine":engine,"samples":[{k:s[k] for k in ("id","sha256","target_json")} for s in samples]}
    output=trial/FOLDER;output.mkdir(exist_ok=True);manifest_path=output/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:raise ValueError("试验实现或输入变化，不能混用结果")
    elif report_only:raise ValueError("尚无试评结果")
    else:write_json(manifest_path,manifest)
    outcomes={}
    for s in samples:
        aid=s["id"];source=DATA_DIR/"assets"/aid
        if not s["adult_confirmed"] or file_fingerprint(source)!=s["sha256"]:raise ValueError("样本确认或原图哈希不匹配")
        directory=output/aid;checkpoint=directory/"result.json";cached=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        key=signature({"manifest":manifest,"asset_id":aid})
        if cached.get("status")=="completed":
            if cached.get("cache_key")!=key:raise ValueError("缓存身份不匹配")
            for name,record in cached["variants"].items():
                if record.get("input_file_hash") and file_fingerprint(directory/f"{name}.png")!=record["input_file_hash"]:raise ValueError("评分输入发生变化")
            result=cached
        elif report_only:raise ValueError("报告重建缺少完成结果，不补跑模型")
        else:
            started=time.monotonic()
            result=evaluate(oriented_rgb(source),json.loads(s["target_json"]),directory)
            result.update(cache_key=key,engine=engine,elapsed_seconds=round(time.monotonic()-started,3))
            write_json(checkpoint,result)
        outcomes[aid]=result
        print(json.dumps({"photo":s["label"],"scores":{k:v.get("total",v.get("status")) for k,v in result["variants"].items()}},ensure_ascii=False),flush=True)
    refs={}
    for s in samples:
        reference=json.loads((trial/s["id"]/"result.json").read_text())
        if reference.get("status")!="completed" or reference["results"]["face"]["engine"]!=baseline["engine"]:
            raise ValueError("参考引擎或完成状态不匹配")
        refs[s["id"]]=reference["results"]["face"]["total"]
    after=snapshot(DATA_DIR/"portrait.db")
    if engine!=identity() or before["fingerprints"]!=after["fingerprints"] or any(file_fingerprint(DATA_DIR/"assets"/s["id"])!=s["sha256"] for s in samples):raise ValueError("试验期间引擎、原图或生产数据发生变化")
    result=comparison(samples,outcomes,refs)
    result.update(engine=engine,preservation={"unchanged":True,"counts":after["counts"],"fingerprints":after["fingerprints"]},reference_digest=signature(refs))
    write_json(output/"comparison.json",result);render(trial,result,outcomes)
    print(json.dumps({"baseline":result["baseline_statistics"],"groups":{k:{"common_count":v["common_count"],"comparisons":v["comparisons"]} for k,v in result["groups"].items()},"preservation":result["preservation"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--trial",type=Path,required=True);parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args();run(args.trial.resolve(),report_only=args.report_only)
