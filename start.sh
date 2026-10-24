#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo '请先运行 python3.13 setup.py --pull-body，详见 docs/runtime-install.md。' >&2
  exit 1
fi
.venv/bin/python scripts/doctor.py
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORTRAIT_PORT:-8765}"
