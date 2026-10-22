"""Add an offline, leave-one-photo-out local candidate to an existing Codex trial.

python -m app.local_calibration_trial --trial data/trials/holistic-20261022-review
No external calls, production writes or changes to the active provider.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DATA_DIR
from .engine import file_fingerprint, signature
from .holistic_scoring import PARTS
from .holistic_trial import snapshot, write_json
from .local_candidate import FEATURES, MODULES, Measurements, cache_key, finite, fit_head, identity, predict

NAMES = {"face": "颜值", "body": "身材", "photography": "摄影"}
FEATURE_LABELS = {
    "beauty_raw": "本地颜值模型原值", "eye_aspect": "眼部宽高比", "eye_spacing": "眼距比",
    "nose_width": "鼻翼/脸宽", "mouth_width": "口宽/脸宽", "lip_aspect": "唇部宽高比",
    "face_aspect": "脸部长宽比", "jaw_cheek": "下颌/颧宽", "middle_lower": "中下段比",
    "upper_middle": "上中段比", "skin_lightness": "双颊亮度", "skin_redness": "双颊红绿分量",
    "skin_yellowness": "双颊黄蓝分量", "skin_spread": "双颊色彩离散",
    "chest_shoulder": "肩胸外轮廓/肩宽", "waist_chest": "腰部/胸部外轮廓宽",
    "hip_waist": "胯腰外轮廓比", "shoulder_hip": "肩胯关键点宽比", "torso_shoulder": "躯干长/肩宽",
    "leg_torso": "可见直腿/躯干长", "outline_variation": "躯干外轮廓变化",
    "subject_clarity": "人物区域清晰度", "subject_clipping": "人物区域明暗截断率",
    "subject_range": "人物区域亮度范围", "subject_luminance": "人物区域亮度",
    "background_edges": "背景边缘密度", "headroom": "面部框上方留白", "horizontal_center": "主体横向位置",
    "saturation": "整体饱和度", "chroma_spread": "色度离散", "subject_background_detail": "人物背景边缘比",
}


def eligible(measurement: dict, module: str, target: str) -> bool:
    if not measurement.get("available"):
        return False
    keys = FEATURES[module][target]
    return any(finite(measurement.get("features", {}).get(k)) for k in keys)


def label(reference: dict, module: str, target: str):
    result = reference.get("results", {}).get(module, {})
    if result.get("status") != "completed":
        return None
    if target == "overall":
        return result.get("total")
    return next((d["score"] for d in result.get("dimensions", [])
                 if d["key"] == target and d["status"] == "assessable"), None)


def cross_validate(samples: list[dict], measurements: dict, references: dict) -> dict:
    """Every displayed prediction excludes the photo and its labels from its fit."""
    outcomes = {}
    for sample in samples:
        aid = sample["id"]
        modules = {}
        for module in MODULES:
            observed = measurements[aid][module]
            predictions, heads = {}, {}
            for target, features in FEATURES[module].items():
                if not eligible(observed, module, target):
                    continue
                training = []
                for other in samples:
                    oid = other["id"]
                    if oid == aid or not eligible(measurements[oid][module], module, target):
                        continue
                    y = label(references[oid], module, target)
                    if finite(y):
                        training.append({"id": oid, "target": y, "features": measurements[oid][module]["features"]})
                try:
                    head = fit_head(training, features, upper=100 if target == "overall" else 10)
                except ValueError:
                    continue
                if not any(use and finite(observed["features"].get(k))
                           for k,use in zip(head["features"],head["usable"])):
                    continue
                predictions[target] = predict(head, observed["features"])
                heads[target] = head
            dimensions = []
            for key, name, _ in PARTS[module]:
                p = predictions.get(key)
                facts = "; ".join(f"{FEATURE_LABELS[k]}={observed['features'][k]:.3f}"
                                  for k in FEATURES[module][key] if finite(observed.get("features", {}).get(k)))
                dimensions.append({"key": key, "name": name, "score": p["score"] if p else None,
                                   "status": "assessable" if p else "insufficient",
                                   "evidence": facts if p else "缺少可靠局部测量或独立参考样本",
                                   "proxy_only": True})
            overall = predictions.get("overall")
            modules[module] = {"total": overall["score"] if overall else None,
                               "status": "completed" if overall else "insufficient",
                               "dimensions": dimensions, "weights": {}, "explanations_only": True,
                               "scope": observed.get("scope", "unavailable"), "confidence": "low",
                               "limitations": observed.get("limitations", []) + ([observed["error"]] if observed.get("error") else []),
                               "predictions": predictions, "heads": heads}
        outcomes[aid] = {"results": modules, "validation": "leave_one_photo_out", "held_out_id": aid}
    return outcomes


def rank_rows(rows, module, version):
    valid = [r for r in rows if finite(r["scores"][module].get(version))]
    valid.sort(key=lambda r: (-r["scores"][module][version], r["created_at"], r["id"]))
    return {r["id"]: i for i,r in enumerate(valid,1)}


def metrics(rows: list[dict], module: str, local: dict) -> dict:
    pairs = [r for r in rows if all(finite(r["scores"][module].get(k)) for k in ("old","codex","local"))]
    result = {"comparable_count": len(pairs)}
    for version in ("old", "local"):
        differences = [r["scores"][module][version]-r["scores"][module]["codex"] for r in pairs]
        good = total = ties = 0
        for i,a in enumerate(pairs):
            for b in pairs[i+1:]:
                ref_diff = a["scores"][module]["codex"]-b["scores"][module]["codex"]
                if abs(ref_diff) < 1:  # Fixed tolerance; do not count tiny reference gaps as truth.
                    continue
                predicted = a["scores"][module][version]-b["scores"][module][version]
                total += 1
                if predicted == 0: ties += 1
                elif predicted * ref_diff > 0: good += 1
        result[version] = {"mae_to_reference": round(float(np.mean(np.abs(differences))),2) if differences else None,
                           "pair_order_agreement": round(good/total,3) if total else None,
                           "pair_count": total, "score_ties": ties}
    baseline_errors = []
    for row in pairs:
        head = local[row["id"]]["results"][module]["heads"].get("overall")
        if head:
            baseline_errors.append(abs(head["mean_baseline"]-row["scores"][module]["codex"]))
    result["leave_one_out_mean_mae"] = round(float(np.mean(baseline_errors)),2) if baseline_errors else None
    return result


def comparison(samples, references, local):
    rows = []
    for sample in samples:
        aid = sample["id"]
        old = json.loads(sample["report_json"])
        scores = {m: {"old": old.get(m,{}).get("total"), "codex": label(references[aid],m,"overall"),
                      "local": local[aid]["results"][m]["total"]} for m in MODULES}
        rows.append({"id": aid, "label": sample["label"], "created_at": sample["created_at"], "scores": scores})
    for module in MODULES:
        for version in ("old","codex","local"):
            ranks = rank_rows(rows,module,version)
            for row in rows:
                row["scores"][module][version+"_rank"] = ranks.get(row["id"])
    return {"rows": rows, "statistics": {m: metrics(rows,m,local) for m in MODULES},
            "validation": "leave_one_photo_out", "reference_is_ground_truth": False}


def number(value):
    return "—" if value is None else f"{value:.1f}"


def render(trial: Path, result: dict, outcomes: dict):
    sections, md = [], ["# BeautyPick 三方案试评", "", "本地新版：本地测量＋留一校准，Codex为参考标签而非真值。正式图库未更新。", ""]
    for module,title in NAMES.items():
        stats = result["statistics"][module]
        mae = lambda version: stats[version]["mae_to_reference"]
        agreement = lambda version: (f"{stats[version]['pair_order_agreement']*100:.1f}%" if stats[version]["pair_order_agreement"] is not None else "—")
        summary = f"与 Codex 参考的平均分差：旧版 {number(mae('old'))} → 本地新版 {number(mae('local'))}；顺序一致率：{agreement('old')} → {agreement('local')}。只预测训练均值的分差为 {number(stats['leave_one_out_mean_mae'])}。"
        summary += f"本地有分 {sum(r['scores'][module]['local'] is not None for r in result['rows'])}/{len(result['rows'])} 张；误差与顺序指标仅比较三个方案都有分的同一批照片。"
        table = []
        md += [f"## {title}", "", summary, "", "|照片|旧版 / 排名|Codex参考 / 排名|本地新版 / 排名|", "|---|---:|---:|---:|"]
        for row in sorted(result["rows"],key=lambda r:r["scores"][module].get("local_rank") or 999):
            aid = row["id"]; s=row["scores"][module]; current=outcomes[aid]["results"][module]
            photo = Image.open(trial/aid/"whole.jpg"); photo.thumbnail((180,220)); buf=io.BytesIO(); photo.save(buf,"JPEG",quality=75)
            thumb = base64.b64encode(buf.getvalue()).decode()
            cells = "".join(f"<td class='{v}'>{number(s[v])}<small>#{s[v+'_rank'] or '—'}</small></td>" for v in ("old","codex","local"))
            facets=[]
            old=json.loads(next(x for x in result["samples"] if x["id"]==aid)["report_json"])
            ref=result["references"][aid]["results"][module]
            for d in current["dimensions"]:
                old_d=next((x.get("score") for x in old.get(module,{}).get("dimensions",[]) if x["key"]==d["key"]),None)
                ref_d=next((x.get("score") for x in ref.get("dimensions",[]) if x["key"]==d["key"]),None)
                facets.append(f"<tr><td>{html.escape(d['name'])}</td><td>{number(old_d)}</td><td>{number(ref_d)}</td><td>{number(d['score'])}</td></tr>")
            details = "<details><summary>分项对照</summary><table><tr><th>分项 /10</th><th>旧版</th><th>Codex</th><th>本地新版</th></tr>"+"".join(facets)+"</table>"
            details += "<p>分项独立预测，不加权生成整体分。身材局部是外轮廓代理，不代表已识别胸臀形态。</p>"
            details += "<ul>"+"".join(f"<li>{html.escape(s)}</li>" for s in current["limitations"])+"</ul></details>"
            overall=current["predictions"].get("overall",{})
            warning=""
            if overall.get("out_of_distribution"):
                warning="部分特征超出校准样本范围，需复核"
            elif current["total"] is None:
                warning="测量或参考样本不足"
            scope={"face":"面部","upper_body":"上身","half_body":"半身","full_body":"全身","whole_photo":"整张照片"}.get(current["scope"],"不可评")
            table.append(f"<tr><td><img alt='{html.escape(row['label'],quote=True)}' src='data:image/jpeg;base64,{thumb}'></td><td>{html.escape(row['label'])}<small>{scope} · 留一验证</small><small>{warning}</small></td>{cells}<td>{details}</td></tr>")
            md.append("|"+row["label"]+"|"+"|".join(f"{number(s[v])} / #{s[v+'_rank'] or '—'}" for v in ("old","codex","local"))+"|")
        sections.append(f"<section id='{module}'><h2>{title}</h2><p>{summary}</p><table><thead><tr><th>照片</th><th>样本</th><th>旧版 / 排名</th><th>Codex 参考 / 排名</th><th>本地新版 / 排名</th><th>分项</th></tr></thead><tbody>{''.join(table)}</tbody></table></section>")
        md.append("")
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BeautyPick · 三方案评分对比</title><style>
body{max-width:1450px;margin:auto;padding:28px;background:#f5f5ef;color:#254539;font:16px/1.65 system-ui}nav{position:sticky;top:0;background:#f5f5ef;padding:14px 0;z-index:1}a{color:inherit;margin-right:24px}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;border-bottom:1px solid #ddd;padding:12px}img{width:100px;height:140px;object-fit:cover;object-position:top;border-radius:8px}small{display:block;color:#708278}.local{font-weight:700;font-size:22px;background:#edf4ed}.codex{color:#66718c}section{scroll-margin-top:65px;overflow-x:auto}details{font-size:13px;max-width:420px}details td,details th{padding:5px}p{max-width:1200px}@media(max-width:700px){body{padding:12px}td,th{padding:6px;font-size:13px}img{width:55px;height:85px}}
</style><h1>旧版 · Codex 参考 · 本地新版</h1><p>本地新版使用现有本地模型与测量特征，Codex 只作离线校准参考。每张照片均用另外 11 张校准后预测，不把本张参考分用于训练；运行时不调用 Codex。</p><p>总分与分项分别预测，没有分项权重回算。身材、摄影尚无经过验证的专用审美模型，本轮为低置信代理试验。12 张单轮参考无法证明哪个更符合人的审美；半身与全身也需分范围看。</p><p>排名只在本批照片内连续编号，同分按上传时间和资产编号排序。顺序一致率只统计 Codex 相差至少 1 分的照片对；不修分、不额外两两调用。旧分历史字段可能误标引擎。</p>"""
    page += "<p>正式图库、历史结果、调用记录均未更新。此前两列对比已保留为 <a href='comparison.codex-only.html'>原始对比</a>。</p><nav><a href='#face'>颜值</a><a href='#body'>身材</a><a href='#photography'>摄影</a></nav>"+"".join(sections)+"</html>"
    existing=trial/"comparison.html"
    backup=trial/"comparison.codex-only.html"
    if existing.exists() and not backup.exists():
        backup.write_bytes(existing.read_bytes())
    temp=trial/"comparison.html.tmp"; temp.write_text(page,encoding="utf-8"); temp.replace(existing)
    (trial/"local-threeway.md").write_text("\n".join(md),encoding="utf-8")


def run(trial: Path):
    baseline=json.loads((trial/"baseline.json").read_text())
    current=snapshot(DATA_DIR/"portrait.db")
    if current["fingerprints"] != baseline["fingerprints"]:
        raise ValueError("生产库已变化，不能混用旧试评基线")
    initial_engine=identity()
    if any(v is None for v in initial_engine["weights"].values()):
        raise ValueError("缺少必要本地模型，不自动下载或降级补分")
    samples=baseline["samples"]; refs={}; measured={}
    model=Measurements()
    local_dir=trial/"local-candidate"; local_dir.mkdir(exist_ok=True)
    try:
        for sample in samples:
            aid=sample["id"]; directory=trial/aid
            source=DATA_DIR/"assets"/aid
            if file_fingerprint(source) != sample["sha256"]:
                raise ValueError(f"原图变化：{sample['label']}")
            reference=json.loads((directory/"result.json").read_text())
            if reference.get("status") != "completed":
                raise ValueError("缺少完成的Codex参考试评")
            whole=directory/"whole.jpg"; face=directory/"face.jpg"
            hashes=[file_fingerprint(whole)]+([file_fingerprint(face)] if face.exists() else [])
            for module in MODULES:
                if reference["results"][module]["engine"] != baseline["engine"] or reference["results"][module]["input_hashes"] != hashes:
                    raise ValueError("参考引擎或输入图像不匹配")
            refs[aid]=reference
            key=cache_key({"original":sample["sha256"],"prepared":hashes,"target":sample.get("target_json")},initial_engine)
            cache=local_dir/f"{aid}.measurements.json"
            prior=json.loads(cache.read_text()) if cache.exists() else {}
            if prior.get("cache_key")==key:
                measured[aid]=prior["modules"]
                continue
            start=time.monotonic(); modules={}
            for module in MODULES:
                try:
                    with Image.open(face if module=="face" and face.exists() else whole) as im:
                        if module=="face": value=model.face(im)
                        elif module=="body": value=model.body(im)
                        else: value=model.photography(im,json.loads(sample.get("target_json") or "null"))
                    modules[module]=value
                except Exception as exc:
                    modules[module]={"available":False,"features":{},"scope":"unavailable","error":str(exc)}
            measured[aid]=modules
            write_json(cache,{"cache_key":key,"engine":initial_engine,"input_hashes":hashes,"modules":modules,
                              "elapsed_seconds":round(time.monotonic()-start,3)})
            print(json.dumps({"sample":sample["label"],"elapsed":round(time.monotonic()-start,2),
                              "available":{m:v["available"] for m,v in modules.items()}},ensure_ascii=False),flush=True)
    finally:
        model.close()
    if initial_engine!=identity():
        raise ValueError("试评期间本地引擎变化，结果未发布")
    outcomes=cross_validate(samples,measured,refs)
    result=comparison(samples,refs,outcomes)
    after=snapshot(DATA_DIR/"portrait.db")
    result.update(engine=initial_engine,reference_engine=baseline["engine"],
                  preservation={"unchanged":after["fingerprints"]==current["fingerprints"],"counts":after["counts"],"fingerprints":after["fingerprints"]},
                  samples=samples,references=refs,
                  reference_digest=signature(refs),measurement_digest=signature(measured))
    if not result["preservation"]["unchanged"]:
        raise ValueError("生产数据发生变化，未替换报告")
    write_json(local_dir/"predictions.json",outcomes)
    write_json(local_dir/"comparison.json",{k:v for k,v in result.items() if k not in ("samples","references")})
    render(trial,result,outcomes)
    print(json.dumps({"statistics":result["statistics"],"preservation":result["preservation"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",type=Path,required=True)
    run(parser.parse_args().trial.resolve())
