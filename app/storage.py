from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import DATA_DIR, TEMP_TTL_HOURS, prepare_dirs

LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


@contextmanager
def db():
    prepare_dirs()
    con = sqlite3.connect(DATA_DIR / "portrait.db", timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA secure_delete=ON")
    try:
        with con:
            yield con
    finally:
        con.close()


def init_db() -> None:
    with LOCK, db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
          expires_at TEXT NOT NULL, retained INTEGER NOT NULL DEFAULT 0,
          deleted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS assets (
          id TEXT PRIMARY KEY, project_id TEXT REFERENCES projects(id),
          filename TEXT NOT NULL, mime TEXT NOT NULL, width INTEGER NOT NULL,
          height INTEGER NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
          created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
          adult_confirmed INTEGER NOT NULL DEFAULT 0,
          rights_confirmed INTEGER NOT NULL DEFAULT 0,
          external_consent INTEGER NOT NULL DEFAULT 0,
          target_json TEXT, report_json TEXT, status TEXT NOT NULL DEFAULT 'uploaded',
          retained INTEGER NOT NULL DEFAULT 0, deleted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id),
          status TEXT NOT NULL, modules_json TEXT NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          error TEXT, result_json TEXT, cancelled INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS candidate_groups (
          id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
          name TEXT NOT NULL, target TEXT NOT NULL DEFAULT 'face',
          representative_id TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS group_members (
          group_id TEXT NOT NULL REFERENCES candidate_groups(id) ON DELETE CASCADE,
          asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          relation TEXT NOT NULL DEFAULT 'unspecified',
          PRIMARY KEY(group_id, asset_id)
        );
        CREATE INDEX IF NOT EXISTS idx_assets_expiry ON assets(expires_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_asset ON jobs(asset_id);
        CREATE TABLE IF NOT EXISTS assessments (
          id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id),
          created_at TEXT NOT NULL, payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_calls (
          id TEXT PRIMARY KEY, day TEXT NOT NULL, module TEXT NOT NULL,
          created_at TEXT NOT NULL, status TEXT NOT NULL, usage_json TEXT
        );
        CREATE TABLE IF NOT EXISTS preferences (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, module TEXT NOT NULL,
          weights_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)
        if "external_scope" not in {r[1] for r in con.execute("PRAGMA table_info(assets)")}:
            con.execute("ALTER TABLE assets ADD COLUMN external_scope TEXT NOT NULL DEFAULT 'full_image'")
        if "user_source" not in {r[1] for r in con.execute("PRAGMA table_info(assets)")}:
            con.execute("ALTER TABLE assets ADD COLUMN user_source TEXT NOT NULL DEFAULT 'unknown'")
        columns = {r[1] for r in con.execute("PRAGMA table_info(assets)")}
        if 'voted' not in columns:
            con.execute("ALTER TABLE assets ADD COLUMN voted INTEGER NOT NULL DEFAULT 0")
        if 'photo_tags_json' not in columns:
            con.execute("ALTER TABLE assets ADD COLUMN photo_tags_json TEXT")
        con.execute("CREATE TABLE IF NOT EXISTS app_migrations (name TEXT PRIMARY KEY)")
        if not con.execute("SELECT 1 FROM app_migrations WHERE name='retain_photos_default'").fetchone():
            con.execute("UPDATE assets SET retained=1 WHERE deleted_at IS NULL")
            con.execute("INSERT INTO app_migrations VALUES ('retain_photos_default')")
        con.execute("UPDATE model_calls SET status='interrupted' WHERE status='running'")
        con.execute("UPDATE jobs SET status='interrupted', updated_at=? WHERE status IN ('queued','running')", (now(),))


def one(query: str, values: tuple[Any, ...] = ()) -> dict | None:
    with LOCK, db() as con:
        row = con.execute(query, values).fetchone()
    return dict(row) if row else None


def many(query: str, values: tuple[Any, ...] = ()) -> list[dict]:
    with LOCK, db() as con:
        rows = con.execute(query, values).fetchall()
    return [dict(r) for r in rows]


def run(query: str, values: tuple[Any, ...] = ()) -> int:
    with LOCK, db() as con:
        cur = con.execute(query, values)
        return cur.rowcount


def insert_asset(*, asset_id: str, project_id: str | None, filename: str,
                 mime: str, width: int, height: int, size: int, sha256: str,
                 adult_confirmed: bool, rights_confirmed: bool) -> None:
    created = now()
    expiry = (datetime.now(timezone.utc) + timedelta(hours=TEMP_TTL_HOURS)).isoformat()
    run("""INSERT INTO assets
        (id,project_id,filename,mime,width,height,size,sha256,created_at,expires_at,adult_confirmed,rights_confirmed,retained)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)""",
        (asset_id, project_id, filename, mime, width, height, size, sha256,
         created, expiry, int(adult_confirmed), int(rights_confirmed)))


def get_asset(asset_id: str) -> dict | None:
    return one("SELECT * FROM assets WHERE id=? AND deleted_at IS NULL", (asset_id,))


def asset_path(asset_id: str) -> Path:
    return DATA_DIR / "assets" / asset_id


def crop_path(asset_id: str) -> Path:
    return DATA_DIR / "crops" / f"{asset_id}.jpg"


def public_asset(row: dict) -> dict:
    keys = ("id", "project_id", "filename", "mime", "width", "height", "size",
            "sha256", "created_at", "expires_at", "status", "retained",
            "adult_confirmed", "rights_confirmed", "external_consent", "external_scope")
    obj = {k: row[k] for k in keys}
    for key in ("retained", "adult_confirmed", "rights_confirmed", "external_consent"):
        obj[key] = bool(obj[key])
    obj["target"] = json.loads(row["target_json"]) if row["target_json"] else None
    obj["report"] = json.loads(row["report_json"]) if row["report_json"] else None
    from .provenance import source_display
    from .scoring import BODY_RUBRIC_VERSION, FACE_RUBRIC_VERSION
    face = (obj['report'] or {}).get('face', {})
    if face.get('status') not in ('not_enabled', 'failed', 'not_assessable') and face.get('rubric_version') != FACE_RUBRIC_VERSION:
        face.update(status='outdated', total=None, dimensions=[], summary='颜值评分待更新', strengths=[], shooting_tips=[])
    body = (obj['report'] or {}).get('body', {})
    if body.get('status') != 'not_enabled' and body.get('rubric_version') != BODY_RUBRIC_VERSION:
        body.update(status='outdated', total=None, dimensions=[], summary='身材评分待更新', strengths=[], shooting_tips=[])
    obj['user_source'] = row.get('user_source', 'unknown')
    obj['source_display'] = source_display((obj['report'] or {}).get('provenance', {}), obj['user_source'])
    obj['voted'] = bool(row.get('voted', 0))
    obj['vote_count'] = int(obj['voted'])
    photography = (obj['report'] or {}).get('photography', {})
    obj['photo_tags'] = json.loads(row['photo_tags_json']) if row.get('photo_tags_json') else {
        'framing': photography.get('framing'), 'style': photography.get('style')}
    obj["preview_url"] = f"/api/assets/{row['id']}/preview"
    obj["crop_url"] = f"/api/assets/{row['id']}/crop" if obj["target"] else None
    return obj


def create_project(name: str) -> dict:
    project_id = new_id()
    created = now()
    expiry = (datetime.now(timezone.utc) + timedelta(hours=TEMP_TTL_HOURS)).isoformat()
    run("INSERT INTO projects(id,name,created_at,expires_at) VALUES (?,?,?,?)",
        (project_id, name[:100].strip() or "未命名项目", created, expiry))
    return one("SELECT * FROM projects WHERE id=?", (project_id,)) or {}


def get_project(project_id: str) -> dict | None:
    return one("SELECT * FROM projects WHERE id=? AND deleted_at IS NULL", (project_id,))


def delete_asset(asset_id: str) -> bool:
    with LOCK, db() as con:
        row = con.execute("SELECT id FROM assets WHERE id=? AND deleted_at IS NULL", (asset_id,)).fetchone()
        if not row:
            return False
        con.execute("UPDATE assets SET deleted_at=?,status='deleted',report_json=NULL,target_json=NULL WHERE id=?", (now(), asset_id))
        con.execute("UPDATE jobs SET cancelled=1,status='cancelled',result_json=NULL WHERE asset_id=?", (asset_id,))
        con.execute("DELETE FROM assessments WHERE asset_id=?", (asset_id,))
        con.execute("DELETE FROM group_members WHERE asset_id=?", (asset_id,))
        con.execute("UPDATE candidate_groups SET representative_id=NULL WHERE representative_id=?", (asset_id,))
    for path in (asset_path(asset_id), crop_path(asset_id)):
        path.unlink(missing_ok=True)
    return True


def cleanup_expired() -> int:
    expired = many("SELECT id FROM assets WHERE deleted_at IS NULL AND retained=0 AND expires_at<?", (now(),))
    count = sum(delete_asset(row["id"]) for row in expired)
    # Retry file removal for tombstones before discarding the database references.
    for row in many("SELECT id FROM assets WHERE deleted_at IS NOT NULL"):
        try:
            asset_path(row["id"]).unlink(missing_ok=True)
            crop_path(row["id"]).unlink(missing_ok=True)
        except OSError:
            continue
        run("DELETE FROM jobs WHERE asset_id=?", (row["id"],))
        run("DELETE FROM assets WHERE id=?", (row["id"],))
    # Empty projects remain as labels; they contain no images or reports.
    with LOCK, db() as con:
        con.execute("PRAGMA secure_delete=ON")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return count
