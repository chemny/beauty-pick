from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PORTRAIT_DATA_DIR", ROOT / "data")).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get("PORTRAIT_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
MAX_PIXELS = int(os.environ.get("PORTRAIT_MAX_PIXELS", 25_000_000))
TEMP_TTL_HOURS = 24
OPENAI_MODEL = "gpt-6.1-sol"
OPENAI_REASONING_EFFORT = "medium"


def prepare_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    (DATA_DIR / "assets").mkdir(exist_ok=True, mode=0o700)
    (DATA_DIR / "crops").mkdir(exist_ok=True, mode=0o700)
