"""Rerun the existing 12-photo review locally without production updates.

python -m app.face_preprocessing_trial --trial data/trials/holistic-20261022-review
The primary crop and monotone scale are frozen before any reference is read.
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

from PIL import Image

from .config import DATA_DIR
from .engine import file_fingerprint, signature
from .face_preprocessing import PRIMARY, VARIANTS, identity, infer, perturbations, preprocess, scale_score, tensor_input
from .holistic_trial import snapshot, write_json
from .images import oriented_rgb

FOLDER = "face-preprocessing-v2"
LABELS = {"context":"原取景", "mesh_rectangle":"面部矩形", "square":"正方形", "aligned_square":"对齐正方形"}


def stability_summary(outcome: dict) -> dict:
    """Do not let display clipping hide changes in the actual model output."""
    primary=outcome["primary"]
    changes=[]; display=[]
    for check in outcome.get("stability",{}).values():
        if check["status"]!="completed": continue
        check["delta"]=round(check["total"]-primary["total"],1)
        raw_change=scale_score(check["raw_score"])["unclipped_total"]-scale_score(primary["raw_score"])["unclipped_total"]
        check["unclipped_delta"]=round(raw_change,3)
        changes.append(abs(raw_change));display.append(abs(check["delta"]))
    return {"max_stability_delta":round(max(changes),3) if changes else None,
            "max_display_delta":max(display) if display else None,
            "stability_pass":len(changes)==2 and max(changes)<=2.0}


def evaluate(image: Image.Image, target: dict, directory: Path) -> dict:
    directory.mkdir(exist_ok=True,parents=True)
    images,diagnostics = preprocess(image,target)
    variants = {}
    for key in VARIANTS:
        actual = tensor_input(images[key])
        path = directory/f"{key}.png"
        actual.save(path,"PNG")
        variants[key] = {**infer(actual), "diagnostics":diagnostics[key], "input_file_hash":file_fingerprint(path)}
    primary = copy.deepcopy(variants[PRIMARY])
    primary["method"] = PRIMARY
    checks = {}
    for name,changed,box in perturbations(image,target):
        try:
            inputs,diagnostics = preprocess(changed,box)
            actual = tensor_input(inputs[PRIMARY])
            path = directory/f"{name}.png"; actual.save(path,"PNG")
            checks[name] = {**infer(actual), "diagnostics":diagnostics[PRIMARY], "input_file_hash":file_fingerprint(path)}
            checks[name]["delta"] = round(checks[name]["total"]-primary["total"],1)
        except Exception as exc:
            checks[name] = {"status":"failed", "total":None, "error":str(exc)}
    result={"status":"completed", "primary":primary, "variants":variants, "stability":checks}
    return {**result,**stability_summary(result)}


def ranks(rows,key):
    valid = [r for r in rows if r.get(key) is not None]
    valid.sort(key=lambda r:(-r[key],r["created_at"],r["id"]))
    return {r["id"]:i for i,r in enumerate(valid,1)}


def agreement(rows,key):
    eligible = [r for r in rows if r.get(key) is not None and r.get("codex") is not None]
    pairs = [(a,b) for i,a in enumerate(eligible) for b in eligible[i+1:] if abs(a["codex"]-b["codex"])>=1]
    matched = sum((a[key]-b[key])*(a["codex"]-b["codex"])>0 for a,b in pairs)
    return {"count":len(eligible),"pairs":len(pairs),"matched":matched,
            "order_agreement":matched/len(pairs) if pairs else None,
            "mae":sum(abs(r[key]-r["codex"]) for r in eligible)/len(eligible) if eligible else None}


def compare(samples, references, prior, outcomes):
    rows = []
    for sample in samples:
        aid=sample["id"]; outcome=outcomes[aid]
        old=json.loads(sample["report_json"]).get("face",{}).get("total")
        previous=next(r for r in prior["rows"] if r["id"]==aid)["scores"]["face"]
        rows.append({"id":aid,"label":sample["label"],"created_at":sample["created_at"],
                     "old":old,"codex":references[aid]["results"]["face"]["total"],"local_v1":previous["local"],
                     "revised":outcome.get("primary",{}).get("total"),"status":outcome["status"],
                     "raw":outcome.get("primary",{}).get("raw_score"),"error":outcome.get("error"),
                     "stability_pass":outcome.get("stability_pass",False),
                     "max_stability_delta":outcome.get("max_stability_delta")})
    for key in ("old","codex","local_v1","revised"):
        positions=ranks(rows,key)
        for row in rows: row[key+"_rank"]=positions.get(row["id"])
    statistics={key:agreement(rows,key) for key in ("old","local_v1","revised")}
    variant_statistics={}
    for method in VARIANTS:
        temporary=[{**r,"variant":outcomes[r["id"]].get("variants",{}).get(method,{}).get("total")} for r in rows]
        variant_statistics[method]=agreement(temporary,"variant")
    return {"rows":rows,"statistics":statistics,"variant_statistics":variant_statistics,
            "primary":PRIMARY,"reference_used_for_scoring":False,"fitted_on_review_photos":False,
            "human_accuracy_validated":False,
            "stability_pass_count":sum(r["stability_pass"] for r in rows),"sample_count":len(rows)}


def fmt(value):
    return "—" if value is None else f"{value:.1f}"


def render(trial: Path, result: dict, outcomes: dict):
    backup=trial/"comparison.local-v1.html"
    source=(backup if backup.exists() else trial/"comparison.html").read_text()
    sections=[]
    for module in ("body","photography"):
        found=re.search(rf"<section id='{module}'>.*?</section>",source,re.S)
        if not found: raise ValueError("原对比页缺少身材或摄影区，未替换页面")
        sections.append(found.group(0))
    md=["# 颜值裁脸与固定换算试评", "", "固定对齐正方形为主方案；同一单调标尺，不对本批参考拟合。分项未重算，身材摄影未重跑。", "",
        "|照片|旧版|Codex参考|本地上一版|颜值修正版|修正版排名|最大扰动分差|", "|---|---:|---:|---:|---:|---:|---:|"]
    table=[]
    for row in sorted(result["rows"],key=lambda r:r["revised_rank"] or 999):
        aid=row["id"]; outcome=outcomes[aid]
        with Image.open(trial/aid/"whole.jpg") as photo:
            photo.thumbnail((160,200)); buf=io.BytesIO(); photo.save(buf,"JPEG",quality=75)
        thumb=base64.b64encode(buf.getvalue()).decode()
        cells="".join(f"<td class='{key}'>{fmt(row[key])}<small>#{row[key+'_rank'] or '—'}</small></td>" for key in ("old","codex","local_v1","revised"))
        diagnostics=[]
        for key,value in outcome.get("variants",{}).items():
            url=f"{FOLDER}/{aid}/{key}.png"
            diagnostics.append(f"<figure><a href='{url}'><img class='crop' src='{url}' alt='{LABELS[key]}'></a><figcaption>{LABELS[key]}<br>原值 {value['raw_score']:.3f} · {fmt(value['total'])}分</figcaption></figure>")
        status="缩放/裁剪波动≤2分" if row["stability_pass"] else "稳定性需复核"
        if outcome.get("primary",{}).get("clipped"): status+="；显示分已截顶"
        stability="；".join(f"{name}：{fmt(v.get('total'))}"+(f"（未截顶变化{v['unclipped_delta']:+.1f}）" if 'unclipped_delta' in v else "（失败）") for name,v in outcome.get("stability",{}).items())
        note=html.escape(outcome.get("error") or "")
        details=f"<small>{status}</small><details><summary>裁脸对照</summary><div class='crops'>{''.join(diagnostics)}</div><p>{stability}</p><p>{note}</p></details>"
        table.append(f"<tr><td><img alt='{html.escape(row['label'],quote=True)}' src='data:image/jpeg;base64,{thumb}'></td><td>{html.escape(row['label'])}<small>原值 {row['raw'] if row['raw'] is not None else '—'}</small></td>{cells}<td>{details}</td></tr>")
        md.append("|"+row["label"]+"|"+"|".join(fmt(row[k]) for k in ("old","codex","local_v1","revised"))+f"|{row['revised_rank'] or '—'}|{fmt(row['max_stability_delta'])}|")
    def percentage(key):
        x=result["statistics"][key]["order_agreement"]
        return "—" if x is None else f"{x*100:.1f}%"
    summary=f"与Codex参考的顺序一致率：旧版 {percentage('old')}，本地上一版 {percentage('local_v1')}，颜值修正版 {percentage('revised')}。这是参考一致性，不是人工准确率。"
    face=f"<section id='face'><h2>颜值</h2><p>{summary}</p><p>轻微缩放和边缘裁剪检查：{result['stability_pass_count']}/{result['sample_count']}张在两项扰动中均不超过2分，按未截顶分数检查。固定换算只是统一显示标尺，未做人工分档校准；99分可能是显示截顶，不代表已确认极高颜值。</p><table><thead><tr><th>照片</th><th>样本</th><th>旧版 / 排名</th><th>Codex参考 / 排名</th><th>本地上一版 / 排名</th><th>颜值修正版 / 排名</th><th>检查</th></tr></thead><tbody>{''.join(table)}</tbody></table></section>"
    style="""body{max-width:1560px;margin:auto;padding:24px;background:#f5f5ef;color:#254539;font:16px/1.65 system-ui}nav{position:sticky;top:0;background:#f5f5ef;padding:14px 0;z-index:1}a{color:inherit;margin-right:20px}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;border-bottom:1px solid #ddd;padding:10px}img{width:90px;height:126px;object-fit:cover;object-position:top;border-radius:8px}small{display:block;color:#708278}.revised{font-size:22px;font-weight:700;background:#edf4ed}.codex{color:#66718c}section{scroll-margin-top:65px;overflow-x:auto}details{font-size:13px;max-width:550px}details td,details th{padding:5px}.crops{display:flex;flex-wrap:wrap;gap:8px}figure{margin:8px 0}.crop{width:112px;height:112px;object-fit:contain}figcaption{font-size:12px}@media(max-width:800px){body{padding:12px}td,th{padding:6px;font-size:13px}img{width:55px;height:80px}}"""
    page=f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>BeautyPick · 颜值裁脸修正版对比</title><style>{style}</style><h1>颜值 · 裁脸与校准修正版</h1><p>保留原有本地模型。统一使用眼部水平对齐的正方形裁脸，再用同一个固定公式换算分数；本轮不使用Codex分数拟合，不用几何或肤色特征额外加减分。</p><p>本次只重跑颜值，身材和摄影保留上一轮结果。整体分独立输出，五个分项本轮未重算。正式图库与历史评分未更新。</p><p><a href='comparison.local-v1.html'>上一轮三列对比</a><a href='comparison.codex-only.html'>原始两列对比</a></p><nav><a href='#face'>颜值</a><a href='#body'>身材（上一轮）</a><a href='#photography'>摄影（上一轮）</a></nav>{face}{''.join(sections)}</html>"
    if not backup.exists(): backup.write_text(source)
    temporary=trial/"comparison.html.tmp";temporary.write_text(page);temporary.replace(trial/"comparison.html")
    (trial/FOLDER/"comparison.md").write_text("\n".join(md)+"\n\n"+summary)


def scoring_identity(engine):
    result=copy.deepcopy(engine)
    result["implementation"].pop("face_preprocessing_trial.py",None)
    return result


def run(trial: Path, *, rebuild_report: bool=False):
    baseline=json.loads((trial/"baseline.json").read_text())
    before=snapshot(DATA_DIR/"portrait.db")
    if before["fingerprints"]!=baseline["fingerprints"]: raise ValueError("生产库与试评基线不同，未试跑")
    engine=identity();started_identity=copy.deepcopy(engine)
    if any(v is None for v in engine["weights"].values()): raise ValueError("缺少现有模型文件，不替换或下载")
    prior=json.loads((trial/"local-candidate/comparison.json").read_text())
    samples=baseline["samples"]; outputs={}; refs={}
    output=trial/FOLDER;output.mkdir(exist_ok=True)
    manifest={"engine":engine,"samples":[{"id":s["id"],"sha256":s["sha256"],"target_json":s["target_json"]} for s in samples],"primary_frozen_before_reference":PRIMARY,"scale_source":"legacy fixed mapping; no fitting"}
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous!=manifest:
            same_inputs=previous["samples"]==manifest["samples"]
            if not rebuild_report or not same_inputs or scoring_identity(previous["engine"])!=scoring_identity(engine):
                raise ValueError("裁脸试评引擎或输入已变化，不覆盖旧试评目录；请为新版本更改FOLDER")
            manifest=previous;engine=previous["engine"]
    elif rebuild_report:
        raise ValueError("缺少已完成试评，不能只重建报告")
    if not rebuild_report: write_json(manifest_path,manifest)
    for sample in samples:
        aid=sample["id"];source=DATA_DIR/"assets"/aid
        if file_fingerprint(source)!=sample["sha256"]: raise ValueError("原图哈希不匹配")
        if not sample["adult_confirmed"]: raise ValueError("样本缺少已有成年确认")
        cache=output/aid/"result.json"
        cached=json.loads(cache.read_text()) if cache.exists() else {}
        if cached.get("status")=="completed":
            outcome=cached
            expected=signature({"manifest":manifest,"id":aid})
            if outcome.get("cache_key")!=expected: raise ValueError("缓存身份不匹配")
            for group in ("variants","stability"):
                for name,v in outcome.get(group,{}).items():
                    if v.get("input_file_hash") and file_fingerprint(output/aid/f"{name}.png")!=v["input_file_hash"]:
                        raise ValueError("实际评分输入变化，不复用缓存")
        else:
            if rebuild_report: raise ValueError("缺少完成的试评结果，报告重建不会发起推理")
            started=time.monotonic()
            try:
                outcome=evaluate(oriented_rgb(source),json.loads(sample["target_json"]),output/aid)
            except Exception as exc:
                outcome={"status":"failed","error":str(exc),"primary":{"total":None}}
            outcome.update(asset_id=aid,cache_key=signature({"manifest":manifest,"id":aid}),engine=engine,
                           original_sha256=sample["sha256"],elapsed_seconds=round(time.monotonic()-started,3))
            cache.parent.mkdir(exist_ok=True);write_json(cache,outcome)
        if outcome["status"]=="completed":outcome.update(stability_summary(outcome))
        outputs[aid]=outcome
        print(json.dumps({"photo":sample["label"],"total":outcome.get("primary",{}).get("total"),"raw":outcome.get("primary",{}).get("raw_score"),"status":outcome["status"],"stability_delta":outcome.get("max_stability_delta")},ensure_ascii=False),flush=True)
    # References are only loaded after all inference/checkpoints are complete.
    for sample in samples:
        ref=json.loads((trial/sample["id"]/"result.json").read_text())
        if ref.get("status")!="completed" or ref["results"]["face"]["engine"]!=baseline["engine"]:
            raise ValueError("参考试评身份不匹配")
        refs[sample["id"]]=ref
    after=snapshot(DATA_DIR/"portrait.db")
    if started_identity!=identity() or before["fingerprints"]!=after["fingerprints"]: raise ValueError("引擎或生产数据变化，未发布报告")
    if any(file_fingerprint(DATA_DIR/"assets"/s["id"])!=s["sha256"] for s in samples):
        raise ValueError("原图在试评过程中发生变化，未发布报告")
    result=compare(samples,refs,prior,outputs)
    result.update(engine=engine,preservation={"unchanged":True,"counts":after["counts"],"fingerprints":after["fingerprints"]},
                  reference_digest=signature(refs),prior_comparison_digest=signature(prior),
                  report_implementation=file_fingerprint(Path(__file__)),rebuilt_without_inference=rebuild_report)
    write_json(output/"comparison.json",result)
    render(trial,result,outputs)
    print(json.dumps({"statistics":result["statistics"],"variant_statistics":result["variant_statistics"],"stability_pass_count":result["stability_pass_count"],"preservation":result["preservation"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",type=Path,required=True)
    parser.add_argument("--rebuild-report",action="store_true",help="仅复用已保存原值和输入，重建报告，不运行模型")
    args=parser.parse_args()
    run(args.trial.resolve(),rebuild_report=args.rebuild_report)
