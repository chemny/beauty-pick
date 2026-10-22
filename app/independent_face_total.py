"""Aggregate saved independent facets; never rerun models or update production."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import html
import json
from pathlib import Path
import re

from .config import DATA_DIR
from .engine import file_fingerprint, signature
from .holistic_trial import snapshot, write_json
from .independent_face_trial import FOLDER, render_section

WEIGHTS = {"eyes": 20, "nose_lips": 20, "harmony": 25, "face_shape": 25, "skin": 10}


def total(dimensions):
    keys = [d["key"] for d in dimensions]
    if len(keys) != len(WEIGHTS) or set(keys) != set(WEIGHTS):
        raise ValueError("五项必须完整且无重复")
    value = Decimal(0)
    for d in dimensions:
        score = d.get("score")
        if d.get("status") != "assessable" or type(score) not in (int, float):
            return None
        number = Decimal(str(score))
        if not number.is_finite() or not 0 <= number <= 10:
            return None
        value += number * WEIGHTS[d["key"]] / 10
    return float(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def aggregate(samples, outcomes):
    rows = [{"id": s["id"], "label": s["label"], "created_at": s["created_at"],
             "total": total(outcomes[s["id"]]["dimensions"])} for s in samples]
    rows.sort(key=lambda r: (r["total"] is None, -(r["total"] or 0), r["created_at"], r["id"]))
    rank = 0
    for row in rows:
        if row["total"] is not None:
            rank += 1
            row["rank"] = rank
        else:
            row["rank"] = None
    return rows


def replace_totals(section, rows):
    body = re.search(r"<tbody>(.*?)</tbody>", section, re.S)
    source_rows = re.findall(r"<tr>.*?</tr>", body.group(1), re.S)
    if len(source_rows) != len(rows):
        raise ValueError("原始对比行数量不匹配")
    rendered = []
    for row in rows:
        matches = [r for r in source_rows if "<td>" + html.escape(row["label"]) + "<small>" in r]
        if len(matches) != 1:
            raise ValueError("样本匹配不唯一")
        cells = re.findall(r"<td(?:\s[^>]*)?>.*?</td>", matches[0], re.S)
        if len(cells) != 7:
            raise ValueError("原始对比表列数不匹配")
        previous = re.sub(r"<[^>]+>", " ", cells[5]).strip()
        value = "—" if row["total"] is None else f"{row['total']:.1f}"
        cells[5] = f"<td class='revised'>{value}<small>#{row['rank'] or '—'}</small><small>原模型 {html.escape(previous)}</small></td>"
        rendered.append("<tr>" + "".join(cells) + "</tr>")
    section = section[:body.start(1)] + "".join(rendered) + section[body.end(1):]
    return section.replace("标准面部版 / 排名", "独立分项合成 / 排名")


def run(trial):
    directory = trial / FOLDER
    source = directory / "results.json"
    source_hash = file_fingerprint(source)
    report = json.loads(source.read_text())
    baseline = json.loads((trial / "baseline.json").read_text())
    before = snapshot(DATA_DIR / "portrait.db")
    if before["fingerprints"] != report["preservation"]["fingerprints"]:
        raise ValueError("生产库与独立分项试评基线不匹配")
    for s in baseline["samples"]:
        if file_fingerprint(DATA_DIR / "assets" / s["id"]) != s["sha256"]:
            raise ValueError("原图变化")
    rows = aggregate(baseline["samples"], report["outcomes"])
    page = trial / "comparison.html"
    current = page.read_text()
    template = (directory / "comparison.before.html").read_text()
    pattern = r"<section id='face-previous'>.*?</section>"
    found = re.search(pattern, template, re.S)
    if not found or not re.search(pattern, current, re.S):
        raise ValueError("缺少颜值对比区段")
    refs = {s["id"]: json.loads((trial / s["id"] / "result.json").read_text()) for s in baseline["samples"]}
    if signature(refs) != report["reference_digest"]:
        raise ValueError("分项参考发生变化")
    section = replace_totals(found.group(), rows)
    section = render_section(section, baseline["samples"], report["outcomes"], refs)
    section = section.replace("当前列：标准面部版。按关键点统一面部范围与边距，PIL双线性缩放，不旋转、不根据单张分数选择裁脸。",
                              "当前列：独立分项合成总分。眼部20%、鼻唇20%、协调25%、轮廓25%、皮肤10%；加权后转百分制，保留一位小数。")
    section = section.replace("五个独立分项已试评；标准面部版总分与排名、身材和摄影保持原结果。正式图库未更新。",
                              "已按合成总分重新排序，连续编号；同分按上传时间、资产编号排序。原模型分数保留在下方，身材、摄影和正式图库未更新。")
    section = section.replace("12张小样本，暂不合成新总分。", "仅为12张小样本的权重合成试验，权重沿用旧方案，不代表审美准确性已通过验证。")
    changed = re.sub(pattern, lambda _: section, current, count=1)
    after = snapshot(DATA_DIR / "portrait.db")
    if before["fingerprints"] != after["fingerprints"] or file_fingerprint(source) != source_hash:
        raise ValueError("生产数据或独立分项变化，不发布结果")
    result = {"weights": WEIGHTS, "formula": "sum(facet_score * percent_weight / 10)",
              "rounding": "decimal_half_up_one_place_then_rank", "rows": rows,
              "source_hash": source_hash, "source": str(source), "reference_used_for_aggregation": False,
              "preservation": {"unchanged": True, "fingerprints": after["fingerprints"]}}
    write_json(directory / "totals.json", result)
    backup = directory / "comparison.before-totals.html"
    if not backup.exists():
        backup.write_text(current)
    temp = page.with_suffix(".html.tmp"); temp.write_text(changed); temp.replace(page)
    write_json(trial / "display-selection.json", {"face_column": "独立分项合成", "source": FOLDER + "/totals.json",
                                                 "score_key": "total", "rank_key": "rank", "production_updated": False})
    print(json.dumps({"rows": rows, "production_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    run(parser.parse_args().trial.resolve())
