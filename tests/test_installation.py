"""Installation checks use isolated destinations and fake network payloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parent.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
installer=load('beautypick_setup',ROOT/'setup.py')
models=load('beautypick_downloader',ROOT/'scripts/download_models.py')
doctor=load('beautypick_doctor',ROOT/'scripts/doctor.py')

def item(payload):return {'path':'sub/model.bin','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'url':'https://example.invalid/model'}

def test_download_verified_atomic_and_resume(tmp_path):
    data=b'complete';spec=item(data);calls=[]
    def opener(*a,**kw):calls.append(a);return io.BytesIO(data)
    assert models.ensure(spec,tmp_path,opener)=='downloaded'
    assert models.ensure(spec,tmp_path,opener)=='verified'
    assert len(calls)==1 and not list(tmp_path.rglob('*.part'))

def test_bad_download_not_installed(tmp_path):
    spec=item(b'good')
    with pytest.raises(ValueError,match='mismatch'):models.ensure(spec,tmp_path,lambda *a,**kw:io.BytesIO(b'bad'))
    assert not (tmp_path/spec['path']).exists() and not list(tmp_path.rglob('*.part'))

def test_existing_corrupt_asset_is_preserved(tmp_path):
    spec=item(b'good');p=tmp_path/spec['path'];p.parent.mkdir();p.write_bytes(b'old')
    with pytest.raises(ValueError,match='existing'):models.ensure(spec,tmp_path,lambda *a,**kw:pytest.fail('network must not run'))
    assert p.read_bytes()==b'old'

def test_single_skill_target_no_overwrite(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();(root/'SKILL.md').write_text('path {{BEAUTYPICK_DIR}}');(root/'references').mkdir();(root/'references/r.md').write_text('rules');monkeypatch.setattr(installer,'ROOT',root)
    target=tmp_path/'one-agent/beauty-pick';installer.register_skill(target)
    assert str(root) in (target/'SKILL.md').read_text()
    with pytest.raises(ValueError,match='exists'):installer.register_skill(target)
    assert not (tmp_path/'other-agent').exists()

def test_client_mode_does_not_install_deps(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['setup.py','--client-only','--skill-dir',str(tmp_path/'skill')]);seen=[]
    monkeypatch.setattr(installer,'register_skill',lambda p:seen.append(p))
    monkeypatch.setattr(installer.subprocess,'run',lambda *a,**kw:pytest.fail('client mode must not install or start anything'))
    assert installer.main()==0 and len(seen)==1

def test_check_model_digest_and_vision():
    spec=json.loads((ROOT/'models/manifest.json').read_text())['body_service']
    def opener(data):return lambda *a,**kw:io.BytesIO(json.dumps({'models':[data]}).encode())
    good={'name':spec['model'],'digest':spec['digest'],'capabilities':['vision']}
    assert doctor.check_service(opener(good))==spec
    for bad in [{**good,'digest':'wrong'},{**good,'capabilities':[]}]:
        with pytest.raises(ValueError):doctor.check_service(opener(bad))

def test_check_mode_no_registration_or_download(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['setup.py','--check','--venv',str(tmp_path)])
    monkeypatch.setattr(installer.subprocess,'call',lambda *a,**kw:0)
    monkeypatch.setattr(installer,'register_skill',lambda *a:pytest.fail('no registration'))
    assert installer.main()==0

def test_manifest_matches_production_constants():
    from app import fpem_model,topiq_model,body_model
    manifest=json.loads((ROOT/'models/manifest.json').read_text());entries={x['path']:x for x in manifest['assets']}
    assert entries['fpem.pth']['sha256']==fpem_model.CHECKPOINT_SHA256
    assert entries['topiq-iaa-res50.pth']['sha256']==topiq_model.CHECKPOINT_SHA256
    assert manifest['body_service']['digest']==body_model.DIGEST
    assert manifest['body_service']['model']==body_model.MODEL

def test_pull_retries_incomplete_manifest_without_changing_model(monkeypatch):
    import http.client
    spec=json.loads((ROOT/'models/manifest.json').read_text())['body_service'];calls=[]
    def fail(*a,**kw):calls.append(a);raise http.client.IncompleteRead(b'partial')
    monkeypatch.setattr(installer.shutil,'which',lambda _: '/not-executed/ollama')
    monkeypatch.setattr(installer.urllib.request,'urlopen',fail)
    monkeypatch.setattr(installer.time,'sleep',lambda _:None)
    monkeypatch.setattr(installer.subprocess,'run',lambda *a,**kw:pytest.fail('must not pull incomplete manifest'))
    with pytest.raises(ValueError,match='清单'):installer.pull_body(Path('python'))
    assert len(calls)==3
