"""Isolated face-facet trial; saved references only, no production activation.

Each facet uses only its own measured inputs and other photos' facet labels.
No overall beauty output, label, rank or derived facet is an input.
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
from .engine import dependency_versions, file_fingerprint, signature
from .holistic_trial import snapshot, write_json
from .local_calibration_trial import FEATURE_LABELS, label, number
from .local_candidate import FEATURES, Measurements, finite, fit_head, predict

FOLDER = "independent-face-v1"
FACETS = {k: v for k, v in FEATURES["face"].items() if k != "overall"}
NAMES = {"eyes": "眼部", "nose_lips": "鼻唇", "harmony": "协调", "face_shape": "轮廓", "skin": "皮肤"}
LIMITS = {
    "eyes": "仅测眼形与眼距，未识别眉形、眼部精致度。",
    "nose_lips": "仅测可见宽高比例，未判断鼻部立体感和唇形细节。",
    "harmony": "仅测局部布局，比例接近某个值不等于更漂亮。",
    "face_shape": "仅测二维宽长关系，不能推断真实骨相。",
    "skin": "仅测双颊色彩，不能判断毛孔、健康或真实肤质。",
}


def complete(observed, keys):
    return observed.get("available", False) and all(finite(observed.get("features", {}).get(k)) for k in keys)


def evaluate(samples, measured, references):
    outcomes = {}
    for sample in samples:
        aid = sample["id"]
        observed = measured[aid]
        dimensions = []
        for key, keys in FACETS.items():
            training = []
            for other in samples:
                oid = other["id"]
                if oid == aid or not complete(measured[oid], keys):
                    continue
                target = label(references[oid], "face", key)
                if finite(target):
                    training.append({"id": oid, "features": measured[oid]["features"], "target": target})
            head = prediction = None
            error = None
            if complete(observed, keys):
                try:
                    head = fit_head(training, keys, upper=10)
                    prediction = predict(head, observed["features"])
                except ValueError as exc:
                    error = str(exc)
            else:
                error = observed.get("error") or "缺少本分项必要测量"
            facts = "；".join(f"{FEATURE_LABELS[k]} {observed['features'][k]:.3f}"
                             for k in keys if finite(observed.get("features", {}).get(k)))
            dimensions.append({"key": key, "name": NAMES[key],
                               "score": prediction["score"] if prediction else None,
                               "status": "assessable" if prediction else "insufficient",
                               "evidence": facts, "limitations": [LIMITS[key]], "error": error,
                               "head": head, "prediction": prediction})
        outcomes[aid] = {"dimensions": dimensions, "total": None, "confidence": "low",
                         "validation": "leave_one_photo_out", "held_out_id": aid,
                         "method": "independent_local_measurement_ridge", "overall_used": False}
    return outcomes


def statistics(outcomes, references):
    result = {}
    for key in FACETS:
        errors, mean_errors, scores = [], [], []
        for aid, outcome in outcomes.items():
            dim = next(d for d in outcome["dimensions"] if d["key"] == key)
            ref = label(references[aid], "face", key)
            if finite(dim["score"]) and finite(ref):
                errors.append(abs(dim["score"] - ref))
                mean_errors.append(abs(dim["head"]["mean_baseline"] - ref))
                scores.append(dim["score"])
        result[key] = {"count": len(errors), "mae": float(np.mean(errors)) if errors else None,
                       "mean_baseline_mae": float(np.mean(mean_errors)) if errors else None,
                       "range": [min(scores), max(scores)] if scores else None}
    return result


def render_section(section, samples, outcomes, references):
    body = re.search(r"<tbody>(.*?)</tbody>", section, re.S)
    if not body:
        raise ValueError("对比表缺少主体")
    source_rows = re.findall(r"<tr>.*?</tr>", body.group(1), re.S)
    if len(source_rows) != len(samples):
        raise ValueError("照片数量不匹配")
    rendered = []
    for source_row in source_rows:
        matches = [s for s in samples if "<td>" + html.escape(s["label"]) + "<small>" in source_row]
        if len(matches) != 1:
            raise ValueError("照片行不唯一")
        sample = matches[0]; aid = sample["id"]
        cells = re.findall(r"<td(?:\s[^>]*)?>.*?</td>", source_row, re.S)
        if len(cells) != 7:
            raise ValueError("当前对比表列数不匹配")
        old = json.loads(sample["report_json"]).get("face", {})
        rows = []
        for dim in outcomes[aid]["dimensions"]:
            key = dim["key"]
            old_score = next((d.get("score") for d in old.get("dimensions", []) if d["key"] == key), None)
            reference = label(references[aid], "face", key)
            note = dim["error"] or LIMITS[key]
            rows.append(f"<tr><td title='{html.escape(note, quote=True)}'>{NAMES[key]}</td>"
                        f"<td>{number(old_score)}</td><td>{number(reference)}</td>"
                        f"<td><b>{number(dim['score'])}</b></td></tr>")
        table = "<table class='facet-comparison'><thead><tr><th>分项 /10</th><th>旧分项</th><th>Codex</th><th>独立分项</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        facts = "".join("<li>" + html.escape(d["name"] + "：" + (d["evidence"] or "测量不足") + "。" + LIMITS[d["key"]]) + "</li>"
                        for d in outcomes[aid]["dimensions"])
        # Preserve existing score, rank, input preview and stability information.
        cells[6] = cells[6][:-5] + table + "<details><summary>测量依据</summary><ul>" + facts + "</ul></details></td>"
        rendered.append("<tr>" + "".join(cells) + "</tr>")
    section = section[:body.start(1)] + "".join(rendered) + section[body.end(1):]
    section = section.replace("本轮重新评估颜值；五个分项未重算，身材和摄影保持原结果。正式图库未批量更新。",
                              "五个独立分项已试评；标准面部版总分与排名、身材和摄影保持原结果。正式图库未更新。")
    note = "<p>独立分项仅用各自的局部测量及其他照片的已保存分项参考校准，不读取整体分；旧分项来自历史记录。12张小样本，暂不合成新总分。</p>"
    return section.replace("<table>", note + "<table>", 1)


def identity():
    return {"version": FOLDER, "features": {k: list(v) for k, v in FACETS.items()},
            "implementation": {p: file_fingerprint(Path(__file__).parent / p) for p in
                               ("independent_face_trial.py", "local_candidate.py", "beauty_model.py", "local_calibration_trial.py")},
            "weights": {"landmarker": file_fingerprint(Path(__file__).parent.parent / "models/face_landmarker.task")},
            "dependencies": dependency_versions(), "overall_used": False, "new_cloud_calls": 0}


def run(trial: Path):
    baseline = json.loads((trial / "baseline.json").read_text())
    before = snapshot(DATA_DIR / "portrait.db")
    if before["fingerprints"] != baseline["fingerprints"]:
        raise ValueError("生产库已变化，不能混用试评基线")
    engine = identity(); samples = baseline["samples"]
    output = trial / FOLDER; output.mkdir(exist_ok=True)
    measured = {}; refs = {}; input_hashes = {}
    model = Measurements()
    try:
        for sample in samples:
            aid = sample["id"]
            if not sample["adult_confirmed"] or file_fingerprint(DATA_DIR / "assets" / aid) != sample["sha256"]:
                raise ValueError("原图或成年确认不匹配")
            standard = trial / "standard-face-v1" / aid
            record = json.loads((standard / "result.json").read_text())
            input_file = standard / "face.png"
            digest = file_fingerprint(input_file)
            if record.get("status") != "completed" or digest != record["primary"]["input_file_hash"]:
                raise ValueError("标准面部输入不匹配")
            refs[aid] = json.loads((trial / aid / "result.json").read_text())
            if refs[aid]["results"]["face"]["engine"] != baseline["engine"]:
                raise ValueError("参考引擎不匹配")
            input_hashes[aid] = digest
            try:
                with Image.open(input_file) as im:
                    measured[aid] = model.face(im, include_beauty=False)
            except Exception as exc:
                measured[aid] = {"available": False, "features": {}, "error": str(exc)}
            print(sample["label"], measured[aid]["available"], flush=True)
    finally:
        model.close()
    outcomes = evaluate(samples, measured, refs)
    stats = statistics(outcomes, refs)
    after = snapshot(DATA_DIR / "portrait.db")
    if before["fingerprints"] != after["fingerprints"] or identity() != engine:
        raise ValueError("生产库或试评引擎变化，不发布页面")
    for sample in samples:
        aid = sample["id"]
        if file_fingerprint(DATA_DIR / "assets" / aid) != sample["sha256"] or file_fingerprint(trial / "standard-face-v1" / aid / "face.png") != input_hashes[aid]:
            raise ValueError("试评期间原图或面部输入变化")
    report = {"engine": engine, "input_hashes": input_hashes, "reference_digest": signature(refs),
              "statistics": stats, "measurements": measured, "outcomes": outcomes,
              "preservation": {"unchanged": True, "fingerprints": after["fingerprints"]}}
    write_json(output / "results.json", report)
    page = trial / "comparison.html"; current = page.read_text()
    backup = output / "comparison.before.html"
    if not backup.exists():
        backup.write_text(current)
    pattern = r"<section id='face-previous'>.*?</section>"
    original = re.search(pattern, backup.read_text(), re.S)
    if not original or not re.search(pattern, current, re.S):
        raise ValueError("缺少当前颜值区段")
    section = render_section(original.group(), samples, outcomes, refs)
    changed = re.sub(pattern, lambda _: section, current, count=1)
    temp = page.with_suffix(".html.tmp"); temp.write_text(changed); temp.replace(page)
    print(json.dumps({"statistics": stats, "production_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    run(parser.parse_args().trial.resolve())
