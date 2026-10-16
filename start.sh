#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ ! -f web/dist/index.html ]; then
  echo '请先按 README.md 安装本项目依赖并构建前端。' >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORTRAIT_PORT:-8765}"
