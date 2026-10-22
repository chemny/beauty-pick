"""Read-only production sampling and checkpointed, separate holistic trial results.

python -m app.holistic_trial --output data/trials/<new-run>
python -m app.holistic_trial --output data/trials/<existing-run> --resume
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import holistic_provider as provider
from .config import DATA_DIR
from .engine import signature
from .holistic_scoring import MODULES, METHOD, PROMPT_VERSION, PROFILE, PARTS, finalize, validate_overall, ExplanationPass
from .images import oriented_rgb, crop_face

EXTRA_SAMPLES = [
    ("3f7b8654-e406-4374-a845-7f769b6b4bd9", "09 室内白上衣镜面自拍"),
    ("0e0b04d2-c293-4011-b62b-62013089085b", "10 雪山草地白裙"),
    ("0427beb1-1f13-4d31-9a0e-1fa26659bf82", "11 蓝天白裙全身"),
    ("4f540389-0fdf-4797-886c-d6264032ee48", "12 黑裙棚拍全身"),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_image(path: Path, image):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90, optimize=True)
    temp = path.with_suffix(".jpg.tmp")
    temp.write_bytes(buffer.getvalue())
    temp.replace(path)


def snapshot(database: Path) -> dict:
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        assets = [dict(r) for r in con.execute("SELECT * FROM assets ORDER BY id")]
        tables = {"assets": assets}
        for name in ("assessments", "jobs", "model_calls"):
            tables[name] = [dict(r) for r in con.execute(f"SELECT * FROM {name} ORDER BY id")]
    return {"assets": assets, "fingerprints": {name: signature({"rows": rows}) for name, rows in tables.items()},
            "counts": {name: len(rows) for name, rows in tables.items()}}


def select_samples(assets: list[dict]) -> list[dict]:
    from .config import ROOT
    representatives = json.loads((ROOT / "tests/fixtures/representatives.json").read_text())["samples"]
    rows = []
    for sample in representatives:
        matches = [a for a in assets if a["sha256"] == sample["sha256"] and not a["deleted_at"]]
        if len(matches) != 1:
            raise ValueError(f"代表照{sample['id']}不存在或不唯一")
        rows.append({**matches[0], "label": sample["label"].replace("_", " ")})
    for aid, label in EXTRA_SAMPLES:
        row = next((a for a in assets if a["id"] == aid and not a["deleted_at"]), None)
        if not row:
            raise ValueError(f"额外样本不存在：{label}")
        rows.append({**row, "label": label})
    for row in rows:
        if not row["adult_confirmed"] or not row["external_consent"] or row.get("external_scope") != "full_image":
            raise ValueError(f"样本缺少已有成年/整图分析授权：{row['label']}")
    return rows


def prepare(row: dict, directory: Path, source_dir: Path) -> tuple[Path, Path | None]:
    original = source_dir / "assets" / row["id"]
    if hashlib.sha256(original.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("原图哈希不一致，未评分")
    image = oriented_rgb(original)
    target = json.loads(row.get("target_json") or "null")
    crop = crop_face(image, target)[0] if target else None
    image.thumbnail((1800, 1800))
    path = directory / "whole.jpg"
    write_image(path, image)
    crop_path = directory / "face.jpg" if crop else None
    if crop:
        write_image(crop_path, crop)
    return path, crop_path


def compare(samples: list[dict], outcomes: dict[str, dict]) -> dict:
    rows = []
    for sample in samples:
        old = json.loads(sample["report_json"])
        outcome = outcomes.get(sample["id"], {})
        results = outcome.get("results", {})
        rows.append({"id": sample["id"], "label": sample["label"], "filename": sample["filename"],
                     "created_at": sample["created_at"], "status": outcome.get("status", "pending"),
                     "error": outcome.get("error"), "scores": {
                         m: {"old": old.get(m, {}).get("total"), "new": results.get(m, {}).get("total"),
                             "new_status": results.get(m, {}).get("status"),
                             "scope": results.get(m, {}).get("scoring_scope"),
                             "old_model_recorded": old.get(m, {}).get("model"),
                             "old_engine_recorded": old.get(m, {}).get("engine"),
                             "old_rubric": old.get(m, {}).get("rubric_version")} for m in MODULES}})
    for module in MODULES:
        for version in ("old", "new"):
            eligible = [r for r in rows if r["scores"][module][version] is not None
                        and (version == "old" or r["scores"][module]["new_status"] == "completed")]
            eligible.sort(key=lambda r: (-r["scores"][module][version], r["created_at"], r["id"]))
            for position, row in enumerate(eligible, 1):
                row["scores"][module][version + "_rank"] = position
        for row in rows:
            item = row["scores"][module]
            item["delta"] = round(item["new"] - item["old"], 1) if item["new"] is not None and item["old"] is not None else None
            item["rank_change"] = item.get("old_rank", 0) - item["new_rank"] if item.get("old_rank") and item.get("new_rank") else None
    statistics = {}
    for module in MODULES:
        statistics[module] = {}
        for key in ("old", "new"):
            values = [r["scores"][module][key] for r in rows if r["scores"][module][key] is not None]
            statistics[module][key] = {"count": len(values), "min": min(values) if values else None,
                                       "max": max(values) if values else None,
                                       "range": round(max(values) - min(values), 1) if values else None}
    return {"rows": rows, "sample_count": len(samples), "method": METHOD, "statistics": statistics,
            "notice": "排名仅限本次样本。旧分取运行前数据库快照，旧model字段可能存在历史误标；新分由Codex整体判断。引擎与规则同时变化，不能将差异全部归因于B评分方式。单轮试评不证明准确性或稳定性。"}


def reports(output: Path, baseline: dict, outcomes: dict[str, dict], preservation: dict):
    samples = baseline["samples"]
    comparison = compare(samples, outcomes)
    comparison["preservation"] = preservation
    comparison["engine"] = baseline["engine"]
    comparison["generated_at"] = now()
    write_json(output / "comparison.json", comparison)
    headings = {"face": "颜值", "body": "身材", "photography": "摄影"}
    md = ["# BeautyPick 整体评分试评对比", "", comparison["notice"], "",
          f"样本 {len(samples)} 张；模型 {baseline['engine']['model']} / {baseline['engine']['reasoning_effort']}。整体分与分项解释采用两次独立调用，解释阶段看不到整体分。", "",
          "生产数据校验：" + ("未变化。" if preservation.get("unchanged") else "检测到变化，请检查comparison.json。"), ""]
    sections = []
    for module, title in headings.items():
        md += [f"## {title}", "", "| 照片 | 旧分 | 新分 | 变化 | 旧排名 | 新排名 |", "|---|---:|---:|---:|---:|---:|"]
        table = []
        ordered = sorted(comparison["rows"], key=lambda r: r["scores"][module].get("new_rank", 999))
        for row in ordered:
            s = row["scores"][module]
            number = lambda n: "—" if n is None else f"{n:.1f}"
            md.append(f"| {row['label']} | {number(s['old'])} | {number(s['new'])} | {number(s['delta'])} | {s.get('old_rank', '—')} | {s.get('new_rank', '—')} |")
            image = output / row["id"] / "whole.jpg"
            thumbnail = ""
            if image.exists() and row["id"] in outcomes:
                im = oriented_rgb(image); im.thumbnail((180, 220)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=75)
                thumbnail = '<img alt="试评照片" src="data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'
            new = outcomes.get(row["id"], {}).get("results", {}).get(module, {})
            names = {key: name for key, name, _ in PARTS[module]}
            details = html.escape(new.get("summary") or row.get("error") or "尚未完成")
            scopes = {"face": "面部", "upper_body": "上身", "half_body": "半身", "full_body": "全身", "visible_body": "可见身材", "whole_photo": "整张照片", "unavailable": "不可评"}
            if new.get("scoring_scope"):
                details += "<small>评价范围：" + scopes.get(new["scoring_scope"], "可见范围") + "</small>"
            if new.get("consistency_flags"):
                details += " · 需复核：" + html.escape("；".join(new["consistency_flags"]))
            facets = "".join(f"<li><b>{html.escape(names.get(d['key'], d['key']))} {number(d['score'])}</b> {html.escape(d['evidence'])}</li>" for d in new.get("dimensions", []))
            table.append(f"<tr><td>{thumbnail}</td><td>{html.escape(row['label'])}</td><td>{number(s['old'])}<small>#{s.get('old_rank','—')}</small></td><td class='new'>{number(s['new'])}<small>#{s.get('new_rank','—')}</small></td><td>{number(s['delta'])}</td><td>{details}<details><summary>分项解释</summary><ul>{facets}</ul></details></td></tr>")
        sections.append(f"<section id='{module}'><h2>{title}</h2><table><thead><tr><th>照片</th><th>样本</th><th>旧分 / 排名</th><th>新分 / 排名</th><th>新－旧</th><th>整体依据与分项</th></tr></thead><tbody>{''.join(table)}</tbody></table></section>")
        md.append("")
    for row in comparison["rows"]:
        if row["error"]:
            md += [f"- {row['label']}：{row['error']}"]
    (output / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BeautyPick 新旧评分对比</title><style>
body{max-width:1300px;margin:auto;padding:28px;background:#f5f5ef;color:#254539;font:16px/1.65 system-ui}h1{margin-bottom:8px}nav{position:sticky;top:0;background:#f5f5ef;padding:14px 0}a{color:inherit;margin-right:24px}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;border-bottom:1px solid #ddd;padding:12px}td:first-child{width:110px}img{width:100px;height:140px;object-fit:cover;object-position:top;border-radius:8px}small{display:block;color:#708278}.new{font-weight:700;font-size:22px}section{scroll-margin-top:65px}details{font-size:13px;max-width:420px}p{max-width:1000px}@media(max-width:700px){body{padding:12px}td,th{padding:6px;font-size:13px}img{width:55px;height:85px}}
</style><h1>整体评分 · 新旧对比</h1>"""
    page += f"<p>{html.escape(comparison['notice'])}</p><p>独立试评 {len(samples)} 张；当前库内评分未被替换。整体与分项分两次独立调用。</p><nav><a href='#face'>颜值</a><a href='#body'>身材</a><a href='#photography'>摄影</a></nav>{''.join(sections)}</html>"
    (output / "comparison.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=12, choices=range(1, 13))
    args = parser.parse_args()
    output = args.output.resolve()
    database = DATA_DIR / "portrait.db"
    current = snapshot(database)
    if args.resume:
        baseline = json.loads((output / "baseline.json").read_text())
        if baseline["engine"] != provider.identity():
            raise ValueError("引擎或规则已变化，请创建新试评而不是混用结果")
        if baseline["fingerprints"] != current["fingerprints"]:
            raise ValueError("生产数据已变化，请创建新试评")
    else:
        output.mkdir(parents=True, exist_ok=False)
        baseline = {"created_at": now(), "engine": provider.identity(), "method": METHOD,
                    "prompt_version": PROMPT_VERSION, "profile": PROFILE,
                    "samples": select_samples(current["assets"])[:args.limit],
                    "fingerprints": current["fingerprints"], "counts": current["counts"]}
        write_json(output / "baseline.json", baseline)
    outcomes = {}
    stop = threading.Event()

    def evaluate_sample(row):
        directory = output / row["id"]
        directory.mkdir(exist_ok=True)
        saved = directory / "result.json"
        if args.resume and saved.exists():
            existing = json.loads(saved.read_text())
            if existing.get("status") == "completed":
                return existing
        try:
            whole, crop = prepare(row, directory, DATA_DIR)
            passes = {}
            for phase in ("overall", "explanations"):
                checkpoint = directory / f"{phase}.json"
                if args.resume and checkpoint.exists():
                    value = json.loads(checkpoint.read_text())
                else:
                    print(json.dumps({"label": row["label"], "phase": phase, "status": "running"}, ensure_ascii=False), flush=True)
                    raw, usage = provider.evaluate(phase, whole, crop, stop.is_set)
                    value = {"raw": raw, "usage": usage, "created_at": now()}
                    if usage["engine"] != baseline["engine"]:
                        raise ValueError("本次推理引擎不匹配试评基线")
                    if phase == "overall":
                        validate_overall(raw)
                    else:
                        ExplanationPass.model_validate(raw)
                    write_json(checkpoint, value)
                passes[phase] = value
            results = finalize(passes["overall"]["raw"], passes["explanations"]["raw"])
            for module in MODULES:
                results[module].update(engine=baseline["engine"], model=baseline["engine"]["model"],
                                       reasoning_effort=baseline["engine"]["reasoning_effort"], evaluated_at=passes["overall"]["created_at"],
                                       input_hashes=passes["overall"]["usage"]["input_hashes"])
            outcome = {"asset_id": row["id"], "label": row["label"], "status": "completed",
                       "results": results, "usage": [passes[p]["usage"] for p in ("overall", "explanations")]}
        except Exception as exc:
            outcome = {"asset_id": row["id"], "label": row["label"], "status": "failed", "error": str(exc)[:500]}
        write_json(saved, outcome)
        print(json.dumps({"label": row["label"], "status": outcome["status"], "error": outcome.get("error")}, ensure_ascii=False), flush=True)
        return outcome

    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="holistic-trial") as executor:
            pending = {executor.submit(evaluate_sample, row): row for row in baseline["samples"]}
            for future in as_completed(pending):
                outcome = future.result()
                outcomes[outcome["asset_id"]] = outcome
                after = snapshot(database)
                reports(output, baseline, outcomes, {"unchanged": after["fingerprints"] == baseline["fingerprints"], "after": after["fingerprints"]})
    except KeyboardInterrupt:
        stop.set()
        raise
    after = snapshot(database)
    unchanged = after["fingerprints"] == baseline["fingerprints"]
    reports(output, baseline, outcomes, {"unchanged": unchanged, "after": after["fingerprints"]})
    print(json.dumps({"output": str(output), "production_unchanged": unchanged,
                      "completed": sum(r["status"] == "completed" for r in outcomes.values()), "samples": len(baseline["samples"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
