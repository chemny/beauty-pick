#!/usr/bin/env python3
"""Download the exact production assets; never substitute a model or update trust roots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent.parent

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']

def ensure(item, folder, opener=urllib.request.urlopen):
    target = folder / item['path']
    if target.exists():
        if not verify(target, item):
            raise ValueError(f'{target}: existing file differs; preserve it and resolve explicitly')
        return 'verified'
    if not item['url']:
        raise ValueError(f'{target}: required bundled asset is missing; restore the release file')
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.part', dir=target.parent)
    temp = Path(name)
    try:
        request = urllib.request.Request(item['url'], headers={'User-Agent':'BeautyPick-installer/0.3'})
        with os.fdopen(fd, 'wb') as output, opener(request, timeout=120) as source:
            while block := source.read(1024 * 1024):
                output.write(block)
        if not verify(temp, item):
            raise ValueError(f'{target.name}: size/SHA-256 mismatch; download not installed')
        # Exclusive create prevents replacing an asset created concurrently.
        try:
            os.link(temp, target)
        except FileExistsError:
            if not verify(target, item):
                raise ValueError(f'{target}: appeared during download and differs')
        return 'downloaded'
    finally:
        temp.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, help='optional preverified asset cache; never modified')
    parser.add_argument('--check', action='store_true', help='read-only check; no downloads')
    parser.add_argument('--models-dir', type=Path, default=ROOT/'models')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'models/manifest.json').read_text())
    errors = []
    for item in manifest['assets']:
        try:
            if args.check:
                if not verify(args.models_dir/item['path'], item):
                    raise ValueError('missing or checksum mismatch')
                status = 'verified'
            else:
                target=args.models_dir/item['path']
                cached=args.cache_dir/item['path'] if args.cache_dir else None
                if not target.exists() and cached and verify(cached,item):
                    ensure(item,args.models_dir,lambda *a,**kw:cached.open('rb'))
                    status='verified cache copy'
                else:
                    status = ensure(item, args.models_dir)
            print(f"[OK] {item['path']}: {status}", flush=True)
        except Exception as exc:
            errors.append(item['path']); print(f"[FAIL] {item['path']}: {exc}", flush=True)
    return 1 if errors else 0

if __name__ == '__main__':
    raise SystemExit(main())
