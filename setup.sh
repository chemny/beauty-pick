#!/bin/sh
set -eu
cd "$(dirname "$0")"
if ! command -v python3.13 >/dev/null 2>&1; then
  echo '需要 Python 3.13。安装后运行 python3.13 setup.py --pull-body。详见 docs/runtime-install.md。' >&2
  exit 1
fi
exec python3.13 setup.py "$@"
