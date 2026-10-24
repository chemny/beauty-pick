#!/usr/bin/env python3
"""Read-only checksum check of all current production assets."""
from download_models import main
import sys
if __name__ == '__main__':
    if '--check' not in sys.argv:
        sys.argv.append('--check')
    raise SystemExit(main())
