#!/usr/bin/env python3
"""Read-only production runtime checks. Does not create the application database."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
ROOT = Path(__file__).resolve().parent.parent

def check_service(opener=urllib.request.urlopen):
    spec = json.loads((ROOT/'models/manifest.json').read_text())['body_service']
    with opener(spec['url']+'/api/tags', timeout=10) as response:
        models = json.load(response)['models']
    match = next((m for m in models if m.get('name') == spec['model']), None)
    if not match or match.get('digest') != spec['digest'] or 'vision' not in match.get('capabilities', []):
        raise ValueError('指定千问模型缺失、摘要不同或不具备vision能力；不可替换同名模型')
    return spec

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-service', action='store_true', help='asset/dependency checks only, not a full readiness result')
    args=p.parse_args(); errors=[]
    def run(label, fn):
        try: fn(); print('[OK]',label,flush=True)
        except Exception as exc: errors.append(label); print('[FAIL]',label,':',exc,flush=True)
    def python():
        if sys.version_info[:2] != (3,13): raise ValueError('完整运行环境需要Python 3.13；请用python3.13 setup.py')
    run('Python 3.13',python)
    def primary():
        from packaging.requirements import Requirement
        # requirements.lock is the tested runtime contract, excluding pytest-only tooling.
        for line in (ROOT/'requirements.lock').read_text().splitlines():
            if not line or line.lstrip().startswith('#') or line.startswith(' '): continue
            req=Requirement(line)
            if req.marker and not req.marker.evaluate():continue
            actual=metadata.version(req.name)
            if actual not in req.specifier:raise ValueError(f'{req.name}: {actual}, expected {req.specifier}; rerun setup.py')
        import torch,torchvision,mediapipe,cv2,c2pa,pillow_heif,einops,ftfy,timm
    run('primary runtime versions/imports',primary)
    def assets():
        subprocess.run([sys.executable,str(ROOT/'scripts/download_models.py'),'--check'],check=True,cwd=ROOT)
    run('production asset SHA-256',assets)
    def isolated():
        code="""import sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'models/body-runtime/deps'))
import transformers,tokenizers,huggingface_hub
assert (transformers.__version__,tokenizers.__version__,huggingface_hub.__version__)==('4.57.6','0.22.2','0.36.0')
assert Path(transformers.__file__).resolve().is_relative_to((root/'models/body-runtime/deps').resolve())
from app.body_vendor.parser import FashnHumanParser
"""
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True,timeout=90)
    run('isolated body runtime imports',isolated)
    run('prebuilt frontend',lambda: (_ for _ in ()).throw(ValueError('web/dist/index.html missing')) if not (ROOT/'web/dist/index.html').is_file() else None)
    if not args.skip_service: run('exact local Qwen vision service',check_service)
    print('READY' if not errors and not args.skip_service else 'PARTIAL (service not checked)' if not errors else 'NOT READY',flush=True)
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
