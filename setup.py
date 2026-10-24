#!/usr/bin/env python3
"""Explicit local runtime installation. No automatic Agent discovery or global config writes."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
import time
import venv
ROOT=Path(__file__).resolve().parent

def python_in(folder):
    return folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')

def register_skill(target):
    """Copy only the explicitly requested skill; never overwrite an installation."""
    target=Path(target).expanduser().resolve()
    if target==ROOT or target.exists():raise ValueError(f'{target}: target already exists; no overwrite. Choose a new explicit --skill-dir')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.beautypick-skill-',dir=target.parent) as temp:
        staged=Path(temp)/'skill';staged.mkdir()
        (staged/'SKILL.md').write_text((ROOT/'SKILL.md').read_text().replace('{{BEAUTYPICK_DIR}}',str(ROOT)),encoding='utf-8')
        shutil.copytree(ROOT/'references',staged/'references')
        # rename refuses an existing nonempty destination; preexisting paths are checked above.
        if target.exists():raise ValueError('Target appeared during installation')
        staged.rename(target)
    print('Skill installed only at',target)

def pull_body(python):
    executable=shutil.which('ollama')
    if not executable:raise ValueError('请先从 https://ollama.com/download 安装Ollama并启动，然后重跑 python3.13 setup.py --pull-body；不会自动安装全局软件')
    spec=json.loads((ROOT/'models/manifest.json').read_text())['body_service']
    # Refuse upstream tag drift before pulling gigabytes or replacing the model tag.
    url='https://registry.ollama.ai/v2/library/qwen3.5/manifests/2b-mlx'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=30) as res: manifest=res.read()
            break
        except (OSError,http.client.HTTPException) as exc:
            if attempt==2:raise ValueError('无法完整读取Ollama模型清单，请检查网络后重试') from exc
            time.sleep(1)
    if hashlib.sha256(manifest).hexdigest()!=spec['digest']:raise ValueError('官方标签已变化；停止，不能静默换权重')
    subprocess.run([executable,'pull',spec['model']],check=True)
    subprocess.run([str(python),str(ROOT/'scripts/doctor.py')],check=True,cwd=ROOT)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv',type=Path,default=ROOT/'.venv')
    p.add_argument('--model-cache',type=Path,help='optional local cache of checksum-verified production assets')
    p.add_argument('--skill-dir',type=Path,help='explicit new Agent skill directory; never overwrites existing files')
    p.add_argument('--client-only',action='store_true',help='install only Skill entry for an SSH-forwarded Mac backend; no model/runtime installation')
    p.add_argument('--pull-body',action='store_true',help='download exact public Qwen model using already-installed Ollama')
    p.add_argument('--check',action='store_true',help='read-only full runtime check; no installation')
    args=p.parse_args()
    if args.check:
        return subprocess.call([str(python_in(args.venv.resolve())),str(ROOT/'scripts/doctor.py')],cwd=ROOT)
    if args.client_only:
        if not args.skill_dir or args.pull_body: p.error('--client-only requires --skill-dir and cannot pull models')
        register_skill(args.skill_dir);print('Connect to your own Mac backend using docs/runtime-install.md. No runtime installed on this client.');return 0
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise ValueError('完整四项身材引擎需要Apple Silicon Mac；Windows请使用 --client-only 和文档中的SSH连接，不替换模型')
    if sys.version_info[:2]!=(3,13):raise ValueError('请使用Python 3.13：python3.13 setup.py')
    folder=args.venv.expanduser().resolve()
    if not folder.exists():venv.EnvBuilder(with_pip=True).create(folder)
    python=python_in(folder)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements.lock')],check=True)
    deps=ROOT/'models/body-runtime/deps'
    # An isolated complete dependency set, never installed into the primary environment.
    subprocess.run([str(python),'-m','pip','install','--upgrade','--target',str(deps),'-r',str(ROOT/'requirements-body-runtime.lock')],check=True)
    download=[str(python),str(ROOT/'scripts/download_models.py')]
    if args.model_cache:download += ['--cache-dir',str(args.model_cache.resolve())]
    subprocess.run(download,check=True,cwd=ROOT)
    subprocess.run([str(python),str(ROOT/'scripts/doctor.py'),'--skip-service'],check=True,cwd=ROOT)
    if args.pull_body:pull_body(python)
    else:
        # Missing service is an incomplete setup, never a false "ready".
        subprocess.run([str(python),str(ROOT/'scripts/doctor.py')],check=True,cwd=ROOT)
    if args.skill_dir:register_skill(args.skill_dir)
    print('READY. Run ./start.sh; skill registration occurs only when --skill-dir is explicitly given.')
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,subprocess.CalledProcessError,OSError,http.client.HTTPException) as exc:print('安装未完成:',exc,file=sys.stderr);raise SystemExit(1)
