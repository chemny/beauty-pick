from __future__ import annotations

import asyncio
import base64
import html
import io
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import pipeline, provider, storage as s
from .config import MAX_UPLOAD_BYTES, MAX_PIXELS, ROOT
from .images import inspect_image, InvalidImage, preview_bytes, oriented_rgb, crop_face
from .scoring import RULES, ANCHORS, RUBRIC_VERSION, FACE_RUBRIC_VERSION, total_score

SESSION = secrets.token_urlsafe(32)


async def sweeper():
    while True:
        await asyncio.sleep(60)
        try:
            await asyncio.to_thread(s.cleanup_expired)
        except OSError:
            pass  # Tombstones remain for retry; access checks already deny deleted assets.


@asynccontextmanager
async def lifespan(app):
    s.init_db()
    s.cleanup_expired()
    task = asyncio.create_task(sweeper())
    yield
    task.cancel()
    s.run("UPDATE jobs SET cancelled=1,status='interrupted' WHERE status IN ('queued','running')")


app = FastAPI(title="颜选 · BeautyPick", lifespan=lifespan,
              docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def protection(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.headers.get('host')}"
        if origin and origin != expected:
            return JSONResponse({"detail": "只允许同源访问"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "拒绝跨站访问"}, status_code=403)
        if request.url.path != "/api/session":
            if request.cookies.get("portrait_session") != SESSION:
                return JSONResponse({"detail": "请刷新页面建立本机会话"}, status_code=401)
            if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("x-portrait-token") != SESSION:
                return JSONResponse({"detail": "请求校验失败"}, status_code=403)
        if request.url.path == "/api/assets" and request.method == "POST" and "content-length" not in request.headers:
            return JSONResponse({"detail": "上传需要声明文件请求长度"}, status_code=411)
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "无效请求长度"}, status_code=400)
        if length > MAX_UPLOAD_BYTES + 1_048_576:
            return JSONResponse({"detail": "文件超过上传大小限制"}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def require_asset(asset_id: str) -> dict:
    row = s.get_asset(asset_id)
    if not row or (not row["retained"] and row["expires_at"] < s.now()):
        raise HTTPException(404, "图片不存在、已删除或已过期")
    return row


def require_project(project_id: str) -> dict:
    row = s.get_project(project_id)
    if not row:
        raise HTTPException(404, "项目不存在")
    return row


@app.get("/api/session")
def session():
    response = JSONResponse({"token": SESSION})
    response.set_cookie("portrait_session", SESSION, httponly=True, samesite="strict")
    return response


@app.get("/api/status")
def status():
    from .beauty_model import MODEL_PATH as BEAUTY_MODEL_PATH
    from .pose_model import MODEL_PATH as POSE_MODEL_PATH
    return {"model": provider.OPENAI_MODEL, "reasoning_effort": provider.OPENAI_REASONING_EFFORT,
            "provider": "Codex CLI · 本机登录", "provider_available": provider.available(),
            "max_upload_mb": MAX_UPLOAD_BYTES // 1_048_576, "max_pixels": MAX_PIXELS,
            "usage": pipeline.usage_summary(), "rubric_version": RUBRIC_VERSION,
            "face_rubric_version": FACE_RUBRIC_VERSION,
            "beauty_model_available": BEAUTY_MODEL_PATH.exists(),
            "beauty_model_name": "SCUT-FBP5500-ResNet50",
            "pose_model_available": POSE_MODEL_PATH.exists(),
            "pose_model_name": "MediaPipe Pose Landmarker Full",
            "photo_metrics_available": True,
            "photo_model_name": "OpenCV Multi-spectrum Photographic Analyzer",
            "rules": {m: [{"key": k, "name": n, "weight": w, "observe": o} for k,n,w,o in r]
                      for m,r in RULES.items()}, "anchors": ANCHORS,
            "body_available": True, "retention_hours": 24}


@app.get("/api/assets")
def assets(project_id: str | None = None):
    rows = s.many("SELECT * FROM assets WHERE deleted_at IS NULL AND (retained=1 OR expires_at>?) ORDER BY created_at DESC", (s.now(),))
    return [s.public_asset(r) for r in rows if not project_id or r["project_id"] == project_id]


@app.post("/api/assets")
async def upload(file: UploadFile = File(...), rights_confirmed: bool = Form(False),
                 adult_confirmed: bool = Form(False), project_id: str | None = Form(None),
                 source_statement: str = Form("")):
    if not rights_confirmed:
        raise HTTPException(400, "请先确认你有权处理这张图片")
    if project_id:
        require_project(project_id)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "文件超过上传大小限制")
    try:
        info = await asyncio.to_thread(inspect_image, raw)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc
    duplicates = s.many("SELECT id FROM assets WHERE sha256=? AND deleted_at IS NULL AND (retained=1 OR expires_at>?)", (info["sha256"], s.now()))
    asset_id = s.new_id()
    s.prepare_dirs()
    path = s.asset_path(asset_id)
    path.write_bytes(raw)
    path.chmod(0o600)
    name = Path((file.filename or "image").replace("\\", "/")).name[:180]
    try:
        s.insert_asset(asset_id=asset_id, project_id=project_id, filename=name,
                       size=len(raw), adult_confirmed=adult_confirmed,
                       rights_confirmed=True, **info)
        await asyncio.to_thread(pipeline.local_report, require_asset(asset_id), source_statement[:300])
    except Exception:
        s.delete_asset(asset_id)
        path.unlink(missing_ok=True)
        raise HTTPException(500, "本地图片处理失败，请检查图片格式后重试")
    result = s.public_asset(require_asset(asset_id))
    result["duplicate_ids"] = [d["id"] for d in duplicates]
    return result


class VoteRequest(BaseModel):
    voted: bool


@app.put('/api/assets/{asset_id}/vote')
def vote(asset_id: str, data: VoteRequest):
    require_asset(asset_id)
    s.run('UPDATE assets SET voted=? WHERE id=? AND deleted_at IS NULL', (int(data.voted), asset_id))
    return s.public_asset(require_asset(asset_id))


class SourceConfirmation(BaseModel):
    source: Literal['unknown', 'ai', 'camera']


@app.post('/api/assets/{asset_id}/source')
def confirm_source(asset_id: str, data: SourceConfirmation):
    require_asset(asset_id)
    s.run('UPDATE assets SET user_source=? WHERE id=? AND deleted_at IS NULL', (data.source, asset_id))
    return s.public_asset(require_asset(asset_id))


@app.post("/api/assets/{asset_id}/source/refresh")
def refresh_source(asset_id: str):
    from .provenance import provenance_report
    asset = require_asset(asset_id)
    report = json.loads(asset["report_json"])
    statements = [e.get("value", "") for e in report["provenance"]["evidence"] if e["type"] == "user_statement"]
    provenance = provenance_report(s.asset_path(asset_id), statements[0] if statements else None)
    with s.LOCK, s.db() as con:
        current = require_asset(asset_id)
        report = json.loads(current["report_json"])
        report.update(id=s.new_id(), created_at=s.now(), provenance=provenance)
        payload = json.dumps(report, ensure_ascii=False)
        con.execute("INSERT INTO assessments VALUES(?,?,?,?)", (report["id"], asset_id, report["created_at"], payload))
        con.execute("UPDATE assets SET report_json=? WHERE id=? AND deleted_at IS NULL", (payload, asset_id))
    return s.public_asset(require_asset(asset_id))


@app.get("/api/assets/{asset_id}/preview")
def preview(asset_id: str):
    require_asset(asset_id)
    return Response(preview_bytes(s.asset_path(asset_id)), media_type="image/jpeg")


@app.get("/api/assets/{asset_id}/crop")
def face_crop(asset_id: str, report_id: str | None = None):
    asset = require_asset(asset_id)
    if report_id:
        report = selected_report(asset, report_id)
        if not report.get("target"):
            raise HTTPException(404, "历史报告没有目标面孔")
        image, _ = crop_face(oriented_rgb(s.asset_path(asset_id)), report["target"])
        output = io.BytesIO(); image.save(output, "JPEG", quality=90)
        return Response(output.getvalue(), media_type="image/jpeg")
    if not asset["target_json"] or not s.crop_path(asset_id).exists():
        raise HTTPException(404, "尚未选择面孔")
    return FileResponse(s.crop_path(asset_id), media_type="image/jpeg")


@app.delete("/api/assets/{asset_id}")
def delete(asset_id: str):
    require_asset(asset_id)
    try:
        s.delete_asset(asset_id)
        s.cleanup_expired()
    except OSError:
        return JSONResponse({"status": "pending_cleanup", "detail": "访问已关闭，文件清理待重试"}, status_code=202)
    return {"status": "deleted"}


class Target(BaseModel):
    x: float = Field(ge=0, lt=1)
    y: float = Field(ge=0, lt=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def bounds(self):
        if self.x + self.width > 1.00001 or self.y + self.height > 1.00001:
            raise ValueError("裁剪范围越界")
        return self


@app.post("/api/assets/{asset_id}/target")
def target(asset_id: str, box: Target):
    pipeline.set_target(require_asset(asset_id), box.model_dump())
    return s.public_asset(require_asset(asset_id))


class Consent(BaseModel):
    adult_confirmed: bool
    external_consent: bool
    external_scope: Literal["face_only", "full_image"] = "full_image"


@app.post("/api/assets/{asset_id}/consent")
def consent(asset_id: str, data: Consent):
    require_asset(asset_id)
    s.run("UPDATE jobs SET cancelled=1,status='cancelled' WHERE asset_id=? AND status IN ('queued','running')", (asset_id,))
    s.run("UPDATE assets SET adult_confirmed=?,external_consent=?,external_scope=? WHERE id=?", (int(data.adult_confirmed), int(data.external_consent), data.external_scope, asset_id))
    return s.public_asset(require_asset(asset_id))


class Analyze(BaseModel):
    body_scoring: bool = True
    force: bool = False
    modules: list[Literal["face", "photography", "body"]] = Field(min_length=1, max_length=3)


@app.post("/api/assets/{asset_id}/analyze")
def analyze(asset_id: str, data: Analyze):
    asset = require_asset(asset_id)
    if not asset["external_consent"]:
        raise HTTPException(400, "请先授权通过Codex账号发送本图及必要裁脸进行评估")
    if any(m in data.modules for m in ("face", "body")) and not asset["adult_confirmed"]:
        raise HTTPException(400, "未确认成年人物，不能启用面部评分")
    if asset["external_scope"] == "face_only" and any(m != "face" for m in data.modules):
        raise HTTPException(400, "当前只授权裁脸，请授权整图后再分析摄影或体态")
    if not provider.available():
        raise HTTPException(503, "Codex CLI不可用")
    try:
        return {"job_id": pipeline.submit(asset, list(dict.fromkeys(data.modules)), body_scoring=data.body_scoring, force=data.force)}
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/jobs")
def jobs():
    return s.many("SELECT id,asset_id,status,created_at,updated_at,error FROM jobs ORDER BY created_at DESC LIMIT 100")


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    if not s.run("UPDATE jobs SET cancelled=1,status='cancelled',updated_at=? WHERE id=? AND status IN ('queued','running')", (s.now(), job_id)):
        raise HTTPException(409, "任务不存在或已经结束")
    return {"status": "cancelled"}


@app.get("/api/assets/{asset_id}/history")
def history(asset_id: str):
    require_asset(asset_id)
    return [json.loads(r["payload"]) for r in s.many("SELECT payload FROM assessments WHERE asset_id=? ORDER BY created_at DESC", (asset_id,))]


class Retention(BaseModel):
    retained: bool


@app.post("/api/assets/{asset_id}/retention")
def retention(asset_id: str, data: Retention):
    require_asset(asset_id)
    s.run("UPDATE assets SET retained=? WHERE id=?", (int(data.retained), asset_id))
    return {"retained": data.retained}


def selected_report(asset: dict, report_id: str | None = None) -> dict:
    if not report_id:
        return json.loads(asset["report_json"])
    row = s.one("SELECT payload FROM assessments WHERE id=? AND asset_id=?", (report_id, asset["id"]))
    if not row:
        raise HTTPException(404, "历史报告不存在")
    return json.loads(row["payload"])


@app.get("/api/assets/{asset_id}/export.json")
def export_json(asset_id: str, report_id: str | None = None):
    asset = require_asset(asset_id)
    return Response(json.dumps(selected_report(asset, report_id), ensure_ascii=False), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="portrait-report.json"'})


@app.get("/api/assets/{asset_id}/export.html")
def export_html(asset_id: str, report_id: str | None = None):
    asset = require_asset(asset_id)
    report = selected_report(asset, report_id)
    image = base64.b64encode(preview_bytes(s.asset_path(asset_id), 1000)).decode()
    sections = []
    for module in ("face", "photography", "body"):
        result = report[module]
        if result['status'] == 'not_enabled':
            continue
        title = {"face": "颜值评分", "photography": "摄影技巧", "body": "身材评分"}[module]
        if module == 'body' and result.get('rubric_version') != 'body-appearance-3.0':
            title = '形体评分（旧版）'
        names = {r[0]: r[1] for r in RULES[module]}
        details = "".join(f"<li>{html.escape(names.get(d['key'], d['key']))}：{d['score'] if d['score'] is not None else '—'} / 10</li>" for d in result["dimensions"])
        benchmark_note = ""
        if module == "face" and result.get("visual_benchmark"):
            vb = result["visual_benchmark"]
            cl_txt = " · 瞳孔眼神光" if vb.get("has_catchlight") else ""
            tp_txt = f" · 三庭实测 {vb['three_parts_ratio']}" if vb.get("three_parts_ratio") else ""
            smooth_txt = " · 疑似过度磨皮" if vb.get("is_over_smoothed") else ""
            benchmark_note = f"<p><em>客观视觉骨相基准（SCUT-FBP5500）：{vb.get('benchmark_score', '—')} 分（{html.escape(vb.get('tier', ''))}）{cl_txt}{tp_txt}{smooth_txt}</em></p>"
        elif module == "body" and result.get("pose_metrics"):
            pm = result["pose_metrics"]
            hr_txt = f" · 头身比：{pm['head_ratio']}" if pm.get("head_ratio") else ""
            shr_txt = f" · 肩髋比：{pm['shoulder_hip_ratio']}" if pm.get("shoulder_hip_ratio") else ""
            sh_bal = f" · {pm['shoulder_balance']}" if pm.get("shoulder_balance") else ""
            benchmark_note = f"<p><em>客观骨骼几何实测（MediaPipe Pose）：{html.escape(pm.get('scene_display', ''))}{sh_bal}{hr_txt}{shr_txt}</em></p>"
        elif module == "photography" and result.get("photo_metrics"):
            p_m = result["photo_metrics"]
            f_exp = f" · 面部{p_m['face_exposure_label']}" if p_m.get("face_exposure_label") else f" · {p_m.get('exposure_label', '')}"
            cl_lbl = f" · {p_m['catchlight_label']}" if p_m.get("catchlight_label") else ""
            benchmark_note = f"<p><em>客观摄影技术测量：{html.escape(p_m.get('clarity_label', ''))}{f_exp}{cl_lbl} · {html.escape(p_m.get('dof_label', ''))}（基准分：{p_m.get('benchmark_score', '—')}）</em></p>"
        tips = f"<p>已评 {result.get('assessed_count', 0)}/6 项</p>" if module == 'body' and result.get('rubric_version') == 'body-appearance-3.0' else '' 
        if module == 'photography':
            for heading, items in [('优点', result.get('strengths', [])), ('不足与改进', result.get('shooting_tips', []) + result.get('editing_tips', []))]:
                short = ''.join('<li>' + html.escape(t.split('。')[0][:65]) + '</li>' for t in items[:2])
                if short:
                    tips += f'<h3>{heading}</h3><ul>{short}</ul>'
        sections.append(f"<section><h2>{title} · {result.get('total') if result.get('total') is not None else '—'} / 100</h2>{benchmark_note}<ul>{details}</ul>{tips}</section>")
    source = report['provenance']
    from .provenance import source_display
    display = source_display(source, asset.get('user_source', 'unknown'))
    exif = {e['title']: e.get('value', '') for e in source['evidence'] if e['type'] == 'exif'}
    categories = {'ai_generated':'AI生成','ai_edited':'AI编辑','camera':'相机拍摄','mixed':'混合来源'}
    fields = [('文件名', asset['filename']), ('大小', f"{asset['size']/1024/1024:.2f} MB"), ('尺寸', f"{asset['width']} × {asset['height']}"), ('来源', display['label'] + ('（' + display['basis'] + '）' if display['basis'] else '')), ('AI平台', (display['platform_basis'] + '：' + display['platform']) if display['platform'] else '无法确认'), ('设备', ' '.join(exif.get(k, '') for k in ('Make','Model')).strip() or '未记录'), ('拍摄日期', exif.get('DateTimeOriginal','未记录'))]
    archive = ''.join(f'<li>{k}：{html.escape(v)}</li>' for k,v in fields)
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>照片评分报告</title><style>body{{max-width:900px;margin:40px auto;padding:24px;background:#f5f3ed;color:#28342f;font:16px/1.8 system-ui}}img{{max-height:480px;max-width:100%;border-radius:12px}}section{{padding:20px 0;border-bottom:1px solid #cdd4cd}}</style><h1>照片评分报告</h1><img src="data:image/jpeg;base64,{image}" alt="照片"><section><h2>图片档案</h2><ul>{archive}</ul></section>{''.join(sections)}</html>"""
    return HTMLResponse(document, headers={"Content-Disposition": 'attachment; filename="portrait-report.html"'})


class Named(BaseModel):
    name: str = Field(min_length=1, max_length=100)


@app.post("/api/projects")
def create_project(data: Named):
    return s.create_project(data.name)


@app.get("/api/projects")
def projects():
    return s.many("SELECT * FROM projects WHERE deleted_at IS NULL ORDER BY created_at DESC")


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    require_project(project_id)
    for asset in s.many("SELECT id FROM assets WHERE project_id=? AND deleted_at IS NULL", (project_id,)):
        s.delete_asset(asset["id"])
    s.run("DELETE FROM candidate_groups WHERE project_id=?", (project_id,))
    s.run("UPDATE projects SET deleted_at=? WHERE id=?", (s.now(), project_id))
    s.cleanup_expired()
    return {"status": "deleted"}


class GroupInput(Named):
    asset_ids: list[str] = []
    representative_id: str | None = None
    target: Literal["face", "photography", "body"] = "face"


@app.get("/api/projects/{project_id}/groups")
def groups(project_id: str):
    require_project(project_id)
    result = s.many("SELECT * FROM candidate_groups WHERE project_id=? ORDER BY created_at", (project_id,))
    for group in result:
        members = s.many("SELECT asset_id,relation FROM group_members WHERE group_id=?", (group["id"],))
        group["asset_ids"] = [r["asset_id"] for r in members]
        group["relations"] = {r["asset_id"]: r["relation"] for r in members}
    return result


def save_group(project_id: str, data: GroupInput, group_id: str | None = None):
    require_project(project_id)
    for asset_id in data.asset_ids:
        if require_asset(asset_id)["project_id"] != project_id:
            raise HTTPException(400, "分组只能使用本项目图片")
    if data.representative_id and data.representative_id not in data.asset_ids:
        raise HTTPException(400, "代表照必须属于该分组")
    group_id = group_id or s.new_id()
    with s.LOCK, s.db() as con:
        con.execute("INSERT INTO candidate_groups VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,target=excluded.target,representative_id=excluded.representative_id",
                    (group_id, project_id, data.name, data.target, data.representative_id, s.now()))
        relations = {r["asset_id"]: r["relation"] for r in con.execute("SELECT asset_id,relation FROM group_members")}
        con.execute("DELETE FROM group_members WHERE group_id=?", (group_id,))
        for asset_id in dict.fromkeys(data.asset_ids):
            con.execute("DELETE FROM group_members WHERE asset_id=?", (asset_id,))
            con.execute("UPDATE candidate_groups SET representative_id=NULL WHERE id<>? AND representative_id=?", (group_id, asset_id))
            con.execute("INSERT INTO group_members(group_id,asset_id,relation) VALUES(?,?,?)", (group_id, asset_id, relations.get(asset_id,"unspecified")))
    return {"id": group_id}


@app.post("/api/projects/{project_id}/groups")
def create_group(project_id: str, data: GroupInput):
    return save_group(project_id, data)


@app.put("/api/projects/{project_id}/groups/{group_id}")
def update_group(project_id: str, group_id: str, data: GroupInput):
    if not s.one("SELECT id FROM candidate_groups WHERE id=? AND project_id=?", (group_id, project_id)):
        raise HTTPException(404, "分组不存在")
    return save_group(project_id, data, group_id)


@app.delete("/api/projects/{project_id}/groups/{group_id}")
def remove_group(project_id: str, group_id: str):
    s.run("DELETE FROM candidate_groups WHERE id=? AND project_id=?", (group_id, project_id))
    return {"status": "deleted"}


class Relation(BaseModel):
    relation: Literal["unspecified", "original", "edited", "generated"]


@app.post("/api/projects/{project_id}/groups/{group_id}/members/{asset_id}/relation")
def set_relation(project_id: str, group_id: str, asset_id: str, data: Relation):
    require_asset(asset_id)
    if not s.one("SELECT id FROM candidate_groups WHERE id=? AND project_id=?", (group_id, project_id)):
        raise HTTPException(404, "分组不存在")
    if not s.run("UPDATE group_members SET relation=? WHERE group_id=? AND asset_id=?", (data.relation, group_id, asset_id)):
        raise HTTPException(404, "分组成员不存在")
    return {"relation": data.relation, "evidence_level": "user_statement"}


@app.get("/api/projects/{project_id}/grid.jpg")
def grid(project_id: str):
    from PIL import Image, ImageOps, ImageDraw
    selection = groups(project_id)
    selected = [(g, require_asset(g["representative_id"])) for g in selection if g["representative_id"]]
    if not selected:
        raise HTTPException(400, "先为分组选择代表照")
    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 300, rows * 430), "#f5f3ed")
    draw = ImageDraw.Draw(canvas)
    for i,(group,asset) in enumerate(selected):
        picture = ImageOps.contain(oriented_rgb(s.asset_path(asset["id"])), (280, 380))
        x,y = (i % columns) * 300, (i // columns) * 430
        canvas.paste(picture, (x + (300-picture.width)//2, y+10))
        draw.text((x+16, y+400), f"{i+1:02}", fill="#16382f")
    output = io.BytesIO()
    canvas.save(output, "JPEG", quality=92)
    return Response(output.getvalue(), media_type="image/jpeg", headers={"Content-Disposition": 'attachment; filename="portrait-grid.jpg"'})


from .comparison import router as comparison_router
app.include_router(comparison_router)


if (ROOT / "web" / "dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web" / "dist", html=True), name="web")
