import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config, storage, pipeline, provider, images
from app.main import app
from app.scoring import total_score, face_total_score, finalize, RULES, FACE_RUBRIC_VERSION, model_prompt


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(storage, 'DATA_DIR', tmp_path)
    with TestClient(app) as client:
        token = client.get('/api/session').json()['token']
        client.headers['x-portrait-token'] = token
        yield client


def upload(client, adult=False, project=None):
    buf=io.BytesIO(); Image.new('RGB',(300,400),'gray').save(buf,'PNG')
    data={'rights_confirmed':'true','adult_confirmed':str(adult).lower()}
    if project: data['project_id']=project
    response=client.post('/api/assets',data=data,files={'file':('test.png',buf.getvalue(),'image/png')})
    assert response.status_code==200,response.text
    return response.json()


def test_rounding_null_and_strict_types():
    assert total_score({'a':7,'b':8},{'a':55,'b':45})==75
    assert total_score({'a':None,'b':8},{'a':55,'b':45}) is None
    for scores in ({'a':True,'b':8},{'a':7.1,'b':8},{'a':11,'b':8}):
        with pytest.raises(ValueError):total_score(scores,{'a':55,'b':45})
    with pytest.raises(ValueError):total_score({'a':7},{'a':90})


def model_output():
    return dict(dimensions=[dict(key=k,status='assessable',score=7,evidence='可见依据',limitations=[]) for k,*_ in RULES['face']],summary='测试',strengths=[],limitations=[],shooting_tips=[],editing_tips=[])


def test_schema_rejects_missing_duplicate_and_inconsistent():
    raw=model_output();assert finalize('face',raw)['total']==70
    raw['dimensions'][0]['key']='expression'
    with pytest.raises(ValueError):finalize('face',raw)
    raw=model_output();raw['dimensions'][0]['score']=None
    with pytest.raises(ValueError):finalize('face',raw)


def test_face_skin_replaces_expression_and_uses_new_version():
    assert [row[0] for row in RULES['face']] == ['eyes', 'nose_lips', 'harmony', 'face_shape', 'skin']
    assert [row[2] for row in RULES['face']] == [20, 20, 25, 25, 10]
    assessed = finalize('face', model_output())
    assert assessed['rubric_version'] == FACE_RUBRIC_VERSION
    assert assessed['total'] == 70
    assert '神态不参与颜值分' in model_prompt('face')
    assert '气色与色彩观感' in model_prompt('face')
    assert '白里透红' in model_prompt('face')
    assert '皮肤项专用评分分档' in model_prompt('face')
    assert '7.5–8.4' in model_prompt('face')
    assert '8.5–9.4' in model_prompt('face')
    assert '不按优势条数或描述字数给分' in model_prompt('face')
    assert '两种呈现分属不同档时不压缩成相邻分值' in model_prompt('face')
    assert '神态不参与颜值分' not in model_prompt('photography')


def test_face_decimal_scores_rounding_missing_and_ranking():
    from app.comparison import assessment
    from app.scoring import model_schema
    weights={r[0]:r[2] for r in RULES['face']}
    raw=model_output()
    for d in raw['dimensions']:
        d['score']=7.8 if d['key']=='harmony' else 7.7
    result=finalize('face',raw)
    assert result['total']==77.3
    assert face_total_score({'a':7.1,'b':7.2},{'a':55,'b':45})==71.5
    assert face_total_score({'a':None,'b':7.2},{'a':55,'b':45}) is None
    for invalid in (7.71, float('nan'), float('inf'), True, -0.1, 10.1):
        bad={**raw,'dimensions':[dict(d,score=invalid) if d['key']=='eyes' else d for d in raw['dimensions']]}
        with pytest.raises(ValueError):finalize('face',bad)
    # Existing photography remains integer-only.
    photo={**raw,'dimensions':[dict(key=k,status='assessable',score=7.7,evidence='可见依据',limitations=[]) for k,*_ in RULES['photography']]}
    with pytest.raises(ValueError):finalize('photography',photo)
    assert 'FaceDimension' in model_schema('face')['$defs']
    result.update(model='test-model',reasoning_effort='medium')
    asset={'id':'decimal-face','filename':'test.png','adult_confirmed':1,
           'report_json':json.dumps({'id':'report','face':result,'target':None})}
    ranked,error=assessment(asset,'face',weights)
    assert error is None and ranked['score']==77.3
    custom={key:100 if key=='harmony' else 0 for key in weights}
    ranked,error=assessment(asset,'face',custom)
    assert error is None and ranked['score']==78.0
    assert json.loads(asset['report_json'])['face']['total']==77.3


def test_face_color_profile_distinguishes_brighter_less_yellow_pixels(monkeypatch):
    monkeypatch.setattr(images, 'detect_faces', lambda _: [dict(
        x=0, y=0, pixel_width=100, pixel_height=100,
        detection_confidence=0.99)])
    warm = images.face_color_profile(Image.new('RGB', (100, 100), (180, 130, 90)))
    clearer = images.face_color_profile(Image.new('RGB', (100, 100), (190, 150, 130)))
    assert warm and clearer
    assert clearer['median_lightness_L_star'] > warm['median_lightness_L_star']
    assert clearer['median_yellow_blue_b_star'] < warm['median_yellow_blue_b_star']


def test_security_and_upload_validation(client):
    assert client.get('/api/assets',headers={'origin':'https://evil.example'}).status_code==403
    assert client.post('/api/projects',json={'name':'x'},headers={'x-portrait-token':'wrong'}).status_code==403
    assert client.get('/api/assets',headers={'host':'evil.example'}).status_code==400
    assert client.post('/api/assets',data={'rights_confirmed':'true'},files={'file':('bad.png',b'broken')}).status_code==400
    assert client.post('/api/assets',files={'file':('bad.png',b'broken')}).status_code==400


def test_local_report_consent_expiry_and_delete(client):
    a=upload(client);aid=a['id']
    assert a['report']['face']['total'] is None
    assert client.post(f'/api/assets/{aid}/analyze',json={'modules':['face']}).status_code==400
    client.post(f'/api/assets/{aid}/consent',json={'adult_confirmed':False,'external_consent':True})
    assert client.post(f'/api/assets/{aid}/analyze',json={'modules':['face']}).status_code==400
    assert client.get(f'/api/assets/{aid}/preview').status_code==200
    assert client.get(f'/api/assets/{aid}/export.html').status_code==200
    assert a['retained'] is True
    storage.run('UPDATE assets SET expires_at=?,retained=0 WHERE id=?',('2000-01-01',aid))
    assert client.get(f'/api/assets/{aid}/preview').status_code==404
    storage.cleanup_expired()
    assert not storage.asset_path(aid).exists()
    assert storage.get_asset(aid) is None


def test_groups_and_project_delete(client):
    pid=client.post('/api/projects',json={'name':'八图测试'}).json()['id']
    a=upload(client,project=pid); aid=a['id']
    body={'name':'01','asset_ids':[aid],'representative_id':aid}
    r=client.post(f'/api/projects/{pid}/groups',json=body); assert r.status_code==200
    assert client.get(f'/api/projects/{pid}/grid.jpg').status_code==200
    client.delete(f'/api/projects/{pid}')
    assert client.get(f'/api/assets/{aid}/preview').status_code==404
    assert not storage.asset_path(aid).exists()
    assert not storage.many('SELECT * FROM group_members')


def test_cancelled_job_cannot_write_late_result(client,monkeypatch):
    a=upload(client,adult=True);aid=a['id']
    storage.run('UPDATE assets SET external_consent=1 WHERE id=?',(aid,))
    def evaluate(module,path,cancelled):
        storage.delete_asset(aid)
        return model_output(),{}
    monkeypatch.setattr(provider,'evaluate',evaluate)
    # photography uses different keys; late result must never persist regardless of parsing.
    jid=storage.new_id()
    storage.run('INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)',(jid,aid,'queued',json.dumps(['photography']),storage.now(),storage.now()))
    pipeline.run_job(jid)
    assert storage.get_asset(aid) is None
    assert not storage.many('SELECT * FROM assessments WHERE asset_id=?',(aid,))
    assert storage.one('SELECT status FROM jobs WHERE id=?',(jid,))['status']=='cancelled'


def test_body_seated_keeps_visible_proportion():
    from app.scoring import BodyAssessment
    raw=model_output();raw['scene']='seated'
    raw['dimensions']=[dict(key=k,status='assessable',score=8,evidence='可见姿态',limitations=[]) for k,*_ in RULES['body']]
    result=finalize('body',raw)
    assert result['total'] == 80.0
    assert result['status']=='completed'
    assert result['assessed_count']==6
    assert next(d for d in result['dimensions'] if d['key']=='proportion')['score'] == 8


def test_ranking_compatible_scores_and_preferences(client):
    pid=client.post('/api/projects',json={'name':'排行'}).json()['id']
    ids=[]
    for i in range(3):
        a=upload(client,adult=True,project=pid);ids.append(a['id'])
        report=a['report'];raw=model_output();raw['dimensions'][0]['score']=7+i
        result=finalize('face',raw);result.update(model='test-model',reasoning_effort='medium')
        if i==2:result['model']='incompatible-model'
        report['face']=result
        storage.run('UPDATE assets SET report_json=? WHERE id=?',(json.dumps(report),a['id']))
        client.post(f'/api/projects/{pid}/groups',json={'name':str(i),'asset_ids':[a['id']],'representative_id':a['id']})
    result=client.get(f'/api/projects/{pid}/ranking').json()
    assert len(result['entries'])==2 and len(result['excluded'])==1
    assert result['pairs'][0]['outcome']=='near_tie'
    assert result['independent_visual_comparisons'] is False
    before=storage.get_asset(ids[0])['report_json']
    weights={k:0 for k,*_ in RULES['face']};weights['eyes']=100
    pref=client.post('/api/preferences',json={'name':'眼部','module':'face','weights':weights})
    assert pref.status_code==200
    changed=client.get(f"/api/projects/{pid}/ranking?preference_id={pref.json()['id']}").json()
    assert changed['entries'][0]['score']==80
    assert storage.get_asset(ids[0])['report_json']==before
    assert client.post('/api/preferences',json={'name':'错误','module':'face','weights':{'eyes':99}}).status_code==422


def test_absent_c2pa_is_not_generated(tmp_path):
    from app.provenance import provenance_report
    path=tmp_path/'plain.png';Image.new('RGB',(40,40)).save(path)
    result=provenance_report(path)
    assert result['category']=='unknown'
    assert result['manifest_status']=='absent'


def test_face_only_consent_blocks_full_image(client):
    a=upload(client,adult=True);aid=a['id']
    r=client.post(f'/api/assets/{aid}/consent',json={'adult_confirmed':True,'external_consent':True,'external_scope':'face_only'})
    assert r.json()['external_scope']=='face_only'
    assert client.post(f'/api/assets/{aid}/analyze',json={'modules':['photography']}).status_code==400
    assert client.post(f'/api/assets/{aid}/analyze',json={'modules':['body']}).status_code==400


def test_verified_claim_requires_trust_signature_and_binding():
    from app.provenance import interpret_manifest
    raw={'active_manifest':'m','validation_state':'Trusted','manifests':{'m':{'signature_info':{'issuer':'OpenAI OpCo, LLC'},'assertions':[{'label':'c2pa.actions.v2','data':{'actions':[{'action':'c2pa.created','softwareAgent':{'name':'ChatGPT'},'digitalSourceType':'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'}]}}]}},'validation_results':{'activeManifest':{'success':[{'code':c} for c in ['claimSignature.validated','assertion.dataHash.match','signingCredential.trusted']],'failure':[]}}}
    result=interpret_manifest(raw)
    assert result['category']=='ai_generated' and result['platform']
    raw['validation_results']['activeManifest']['failure']=[{'code':'signingCredential.untrusted'}]
    result=interpret_manifest(raw)
    assert result['category']=='unknown' and result['platform'] is None
    assert result['signature_status']=='valid' and result['asset_binding_status']=='valid'
    raw['validation_results']['activeManifest']['failure']=[{'code':'assertion.dataHash.mismatch'}]
    assert interpret_manifest(raw)['asset_binding_status']=='invalid'


def test_formats_orientation_and_metadata(tmp_path):
    from app.images import inspect_image,preview_bytes
    from app.provenance import provenance_report
    for fmt in ('JPEG','PNG','WEBP'):
        b=io.BytesIO();Image.new('RGB',(80,120)).save(b,fmt)
        assert inspect_image(b.getvalue())['width']==80
    p=tmp_path/'exif.jpg';im=Image.new('RGB',(80,120));ex=Image.Exif();ex[274]=6;ex[305]='ChatGPT';ex[34853]={1:'N',2:(10,0,0)}
    im.save(p,exif=ex)
    assert inspect_image(p.read_bytes())['width']==120
    preview=Image.open(io.BytesIO(preview_bytes(p)))
    assert preview.size==(120,80) and not preview.getexif()
    report=provenance_report(p)
    assert report['platform'] is None
    assert all(e.get('title')!='GPSInfo' for e in report['evidence'])


def test_budget_cache_and_restart(client,monkeypatch):
    monkeypatch.setattr(pipeline,'DAILY_LIMIT',1)
    pipeline.reserve_call('face')
    with pytest.raises(provider.ProviderError):pipeline.reserve_call('photography')
    a=upload(client,adult=True);aid=a['id']
    storage.run('INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)',('job',aid,'running','["face"]',storage.now(),storage.now()))
    storage.init_db()
    assert storage.one('SELECT status FROM jobs WHERE id=?',('job',))['status']=='interrupted'
    assert storage.one('SELECT status FROM model_calls')['status']=='interrupted'


def test_crop_changes_invalidate_face_result(client):
    a=upload(client,adult=True);aid=a['id']
    response=client.post(f'/api/assets/{aid}/target',json={'x':.2,'y':.1,'width':.6,'height':.5})
    assert response.status_code==200
    r=response.json()['report'];assert r['target']['transform']
    assert r['face']['total'] is None
    assert client.post(f'/api/assets/{aid}/target',json={'x':.9,'y':.1,'width':.6,'height':.5}).status_code==422
    assert client.delete(f'/api/assets/{aid}').status_code==200
    assert not storage.crop_path(aid).exists()


def test_provider_cancel_terminates_process(tmp_path,monkeypatch):
    import subprocess,sys,time
    p=tmp_path/'input.jpg';Image.new('RGB',(40,40)).save(p)
    real=subprocess.Popen;processes=[]
    def spawn(command,**kwargs):
        child=real([sys.executable,'-c','import time;time.sleep(60)'],**kwargs);processes.append(child);return child
    monkeypatch.setattr(provider.subprocess,'Popen',spawn)
    started=time.monotonic()
    with pytest.raises(provider.ProviderError,match='取消'):
        provider.evaluate('face',p,lambda:True)
    assert time.monotonic()-started<10
    assert all(child.poll() is not None for child in processes)


def test_model_gate_and_explicit_suggestions(client,monkeypatch):
    a=upload(client,adult=True);aid=a['id']
    storage.run('UPDATE assets SET external_consent=1 WHERE id=?',(aid,))
    def evaluate(module,path,cancelled,**kwargs):
        raw=model_output();raw['scene']='full_body'
        raw['dimensions']=[dict(key=k,status='assessable',score=8,evidence='测试可见依据',limitations=[]) for k,*_ in RULES['body']]
        return raw,{'input_tokens':1,'output_tokens':1}
    monkeypatch.setattr(provider,'evaluate',evaluate)
    jid=storage.new_id()
    storage.run('INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)',(jid,aid,'queued',json.dumps({'modules':['body'],'body_scoring':False}),storage.now(),storage.now()))
    pipeline.run_job(jid)
    result=json.loads(storage.get_asset(aid)['report_json'])['body']
    assert result['status']=='suggestions_only' and result['total'] is None
    assert all(d['score'] is None for d in result['dimensions'])


def test_upload_size_pixel_and_format_limits(client,monkeypatch):
    from app import main,images
    monkeypatch.setattr(main,'MAX_UPLOAD_BYTES',128)
    assert client.post('/api/assets',data={'rights_confirmed':'true'},files={'file':('x.png',b'x'*129)}).status_code==413
    monkeypatch.setattr(main,'MAX_UPLOAD_BYTES',20*1024*1024)
    monkeypatch.setattr(images,'MAX_PIXELS',100)
    buf=io.BytesIO();Image.new('RGB',(11,10)).save(buf,'PNG')
    assert client.post('/api/assets',data={'rights_confirmed':'true'},files={'file':('x.png',buf.getvalue())}).status_code==400
    buf=io.BytesIO();Image.new('RGB',(10,10)).save(buf,'PNG')
    assert client.post('/api/assets',data={'rights_confirmed':'true'},files={'file':('wrong.exe',buf.getvalue(),'application/octet-stream')}).status_code==200
    buf=io.BytesIO();Image.new('RGB',(10,10)).save(buf,'GIF')
    assert client.post('/api/assets',data={'rights_confirmed':'true'},files={'file':('fake.png',buf.getvalue(),'image/png')}).status_code==400


def test_exports_escape_user_content_and_original_unchanged(client):
    a=upload(client);aid=a['id'];original=storage.asset_path(aid).read_bytes()
    r=a['report'];r['provenance']['evidence'].append({'type':'exif','title':'Model','value':'<script>alert(1)</script>','level':'unverified'})
    storage.run('UPDATE assets SET report_json=? WHERE id=?',(json.dumps(r),aid))
    exported=client.get(f'/api/assets/{aid}/export.html').text
    assert '<script>alert(1)</script>' not in exported and '&lt;script&gt;' in exported
    assert client.get(f'/api/assets/{aid}/export.json').json()['face']['total'] is None
    assert storage.asset_path(aid).read_bytes()==original


def test_failed_and_cached_pipeline_does_not_invent_results(client,monkeypatch):
    a=upload(client,adult=True);aid=a['id'];storage.run('UPDATE assets SET external_consent=1 WHERE id=?',(aid,))
    calls=[]
    def evaluate(module,path,cancelled):
        calls.append(module)
        raw=model_output();raw['dimensions']=[dict(key=k,status='assessable',score=7,evidence='可见依据',limitations=[]) for k,*_ in RULES[module]]
        return raw,{'input_tokens':1,'output_tokens':1}
    monkeypatch.setattr(provider,'evaluate',evaluate)
    def run():
        jid=storage.new_id();storage.run('INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)',(jid,aid,'queued','["photography"]',storage.now(),storage.now()));pipeline.run_job(jid)
    run();run()
    assert calls==['photography']
    assert len(storage.many('SELECT * FROM assessments WHERE asset_id=?',(aid,)))==2
    assert json.loads(storage.get_asset(aid)['report_json'])['photography']['cache_reused'] is True


def test_same_face_selection_is_idempotent(client):
    a=upload(client,adult=True);aid=a['id'];box={'x':.2,'y':.1,'width':.6,'height':.5}
    client.post(f'/api/assets/{aid}/target',json=box)
    report=json.loads(storage.get_asset(aid)['report_json']);report['face']=finalize('face',model_output())
    storage.run('UPDATE assets SET report_json=? WHERE id=?',(json.dumps(report),aid))
    assert client.post(f'/api/assets/{aid}/target',json=box).json()['report']['face']['total']==70


def test_multiface_detection_with_authorized_representatives(tmp_path):
    from app.images import oriented_rgb,detect_faces,crop_face
    manifest=json.loads((Path(__file__).parent/'fixtures/representatives.json').read_text())
    root=Path(manifest['source_root'])
    if not root.exists():pytest.skip('本机授权原图不随源码分发')
    canvas=Image.new('RGB',(800,440),'white')
    for column,idx in enumerate((0,3)):
        image=oriented_rgb(root/manifest['samples'][idx]['relative_path'])
        faces=detect_faces(image);assert len(faces)==1
        crop,_=crop_face(image,faces[0]);crop.thumbnail((360,400));canvas.paste(crop,(column*400+20,20))
    assert len(detect_faces(canvas))==2


def test_unlimited_budget_keeps_ledger(client, monkeypatch):
    monkeypatch.setattr(pipeline, 'DAILY_LIMIT', 0)
    for _ in range(26):
        pipeline.reserve_call('face')
    usage = pipeline.usage_summary()
    assert usage['limit'] is None
    assert usage['calls'] == 26


def test_nested_camera_exif(tmp_path):
    from app.provenance import metadata
    image = Image.new('RGB', (100, 100))
    exif = Image.Exif()
    exif[271] = 'Example Camera'
    exif[272] = 'Model X'
    exif[34665] = {36867: '2026:10:14 12:00:00', 42036: 'Lens X', 34855: 200}
    path = tmp_path / 'camera.jpg'
    image.save(path, exif=exif)
    result = metadata(path)['exif']
    assert result['DateTimeOriginal'] == '2026:10:14 12:00:00'
    assert result['LensModel'] == 'Lens X'
    assert result['ISOSpeedRatings'] == '200'


def test_default_retention_and_existing_asset_migration(client):
    a=upload(client);aid=a['id']
    assert a['retained'] is True
    storage.run("UPDATE assets SET expires_at='2000-01-01' WHERE id=?",(aid,))
    storage.cleanup_expired()
    assert client.get(f'/api/assets/{aid}/preview').status_code==200
    storage.run("DELETE FROM app_migrations WHERE name='retain_photos_default'")
    storage.run('UPDATE assets SET retained=0 WHERE id=?',(aid,))
    storage.init_db()
    assert storage.get_asset(aid)['retained']==1
    storage.init_db()
    assert storage.get_asset(aid)['retained']==1


def test_source_display_separates_claim_confirmation_and_verification(client):
    from app.provenance import source_display
    p={'category':'unknown','signature_status':'valid','asset_binding_status':'valid','signer_trust':'untrusted','evidence':[{'type':'c2pa_action','value':'软件声明：ChatGPT；数字来源类型：http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'}]}
    result=source_display(p)
    assert result['label']=='含 AI 生成标记' and result['platform_basis']=='文件标记'
    assert source_display(p,'ai')['label']=='AI 图片'
    assert source_display(p,'ai')['basis']=='用户确认'
    assert p['category']=='unknown'
    assert source_display({'evidence':[{'type':'user_statement','value':'不是AI图片'}]})['label']=='无法确认'
    trusted={**p,'category':'ai_generated','signer_trust':'trusted','platform':'ChatGPT'}
    assert source_display(trusted,'camera')['basis']=='来源已验证'
    assert source_display(trusted,'camera')['label']=='AI 生成'
    a=upload(client);aid=a['id']
    assert client.post(f'/api/assets/{aid}/source',json={'source':'ai'}).json()['source_display']['basis']=='用户确认'
    assert client.post(f'/api/assets/{aid}/source',json={'source':'invented'}).status_code==422
    refreshed=client.post(f'/api/assets/{aid}/source/refresh').json()
    assert refreshed['user_source']=='ai' and refreshed['source_display']['label']=='AI 图片'
    assert client.post(f'/api/assets/{aid}/source',json={'source':'unknown'}).json()['source_display']['label']=='无法确认'


def test_votes_are_idempotent_persistent_and_protected(client):
    a=upload(client);aid=a['id']
    assert a['vote_count']==0 and a['voted'] is False
    for _ in range(2):
        r=client.put(f'/api/assets/{aid}/vote',json={'voted':True})
        assert r.status_code==200 and r.json()['vote_count']==1
    storage.init_db()
    assert storage.get_asset(aid)['voted']==1
    r=client.put(f'/api/assets/{aid}/vote',json={'voted':False})
    assert r.json()['vote_count']==0
    assert client.put(f'/api/assets/{aid}/vote',json={'voted':True},headers={'x-portrait-token':'wrong'}).status_code==403
    storage.delete_asset(aid)
    assert client.put(f'/api/assets/{aid}/vote',json={'voted':True}).status_code==404


def test_photography_tags_are_constrained_and_legacy_safe():
    from app.scoring import finalize, model_schema, RULES
    raw=model_output()
    raw['dimensions']=[dict(key=k,status='assessable',score=7,evidence='可见依据',limitations=[]) for k,*_ in RULES['photography']]
    assert finalize('photography',raw)['framing'] is None
    tagged=finalize('photography',{**raw,'framing':'半身','style':'自然清新'})
    assert tagged['framing']=='半身' and tagged['total']==70
    with pytest.raises(ValueError):finalize('photography',{**raw,'style':'随意编造'})
    assert {'framing','style'} <= set(model_schema('photography')['required'])


def test_body_visible_weights_decimal_and_minimum():
    from app.scoring import body_total_score, model_schema
    weights={r[0]:r[2] for r in RULES['body']}
    scores={'proportion':None,'chest':8.1,'waist':7.2,'hips':None,'legs':None,'harmony':8.0}
    assert body_total_score(scores,weights)==77.5
    assert body_total_score({**scores,'harmony':None},weights)==77.0
    assert body_total_score({k:(8.1 if k=='chest' else None) for k in weights},weights)==81.0
    assert body_total_score({k:None for k in weights},weights) is None
    assert body_total_score({k:0 for k in weights},weights)==0.0
    for invalid in (8.12, float('nan'), 11, True):
        with pytest.raises(ValueError): body_total_score({**scores,'chest':invalid},weights)
    assert 'BodyDimension' in model_schema('body')['$defs']
    raw=model_output();raw['scene']='head_only'
    raw['dimensions']=[dict(key=k,status='assessable',score=8.1,evidence='测试',limitations=[]) for k in weights]
    result=finalize('body',raw)
    assert result['total'] is None and result['assessed_count']==0


def test_body_old_result_hidden_but_history_retained(client):
    a=upload(client,adult=True);aid=a['id'];report=a['report']
    report['body'].update(status='completed',total=80,rubric_version='old-body')
    storage.run('UPDATE assets SET report_json=? WHERE id=?',(json.dumps(report),aid))
    public=storage.public_asset(storage.get_asset(aid))
    assert public['report']['body']['status']=='outdated'
    assert public['report']['body']['total'] is None
    assert json.loads(storage.get_asset(aid)['report_json'])['body']['total']==80


def test_old_face_result_hidden_but_history_retained(client):
    a = upload(client, adult=True)
    aid = a['id']
    report = a['report']
    report['face'] = finalize('face', model_output())
    report['face']['rubric_version'] = 'portrait-experimental-0.1'
    storage.run('UPDATE assets SET report_json=? WHERE id=?', (json.dumps(report), aid))
    public = storage.public_asset(storage.get_asset(aid))
    assert public['report']['face']['status'] == 'outdated'
    assert public['report']['face']['total'] is None
    assert json.loads(storage.get_asset(aid)['report_json'])['face']['total'] == 70


def test_body_ranking_handles_missing_dimensions(client):
    from app.comparison import ranking
    pid=client.post('/api/projects',json={'name':'身材对照'}).json()['id']
    for i in range(2):
        a=upload(client,adult=True);aid=a['id'];raw=model_output();raw['scene']='half_body'
        raw['dimensions']=[dict(key=k,status='assessable',score=7.5+i/10,evidence='可见轮廓',limitations=[]) for k,*_ in RULES['body']]
        next(d for d in raw['dimensions'] if d['key']=='hips').update(score=None,status='insufficient')
        body=finalize('body',raw);body.update(model='test',reasoning_effort='medium')
        report=a['report'];report['body']=body
        storage.run('UPDATE assets SET project_id=?,report_json=? WHERE id=?',(pid,json.dumps(report),aid))
        client.post(f'/api/projects/{pid}/groups',json={'name':str(i),'asset_ids':[aid],'representative_id':aid})
    r=ranking(pid,'body',None)
    assert len(r['entries'])==2 and len(r['pairs'])==1
    missing=[d for d in r['pairs'][0]['dimension_differences'] if d['key']=='hips']
    assert missing[0]['difference'] is None


def test_upload_consent_and_three_module_analysis(client, monkeypatch):
    from types import SimpleNamespace
    a=upload(client,adult=True);aid=a['id']
    client.post(f'/api/assets/{aid}/target',json={'x':.1,'y':.1,'width':.8,'height':.8})
    report=json.loads(storage.get_asset(aid)['report_json'])
    report['face_gate']={'assessable':True,'limitations':[]}
    storage.run('UPDATE assets SET report_json=? WHERE id=?',(json.dumps(report),aid))
    client.post(f'/api/assets/{aid}/consent',json={'adult_confirmed':True,'external_consent':True,'external_scope':'full_image'})
    calls=[]
    def evaluate(module,path,cancelled,**kwargs):
        calls.append(module);raw=model_output()
        raw['dimensions']=[dict(key=k,status='assessable',score=7,evidence='测试可见范围',limitations=[]) for k,*_ in RULES[module]]
        if module=='body':raw['scene']='full_body'
        return raw,{'input_tokens':1,'output_tokens':1}
    monkeypatch.setattr(provider,'evaluate',evaluate)
    monkeypatch.setattr(provider,'available',lambda:True)
    monkeypatch.setattr(pipeline,'EXECUTOR',SimpleNamespace(submit=lambda fn,*args:fn(*args)))
    r=client.post(f'/api/assets/{aid}/analyze',json={'modules':['face','photography','body']})
    assert r.status_code==200
    assert set(calls)=={'face','photography','body'}
    report=json.loads(storage.get_asset(aid)['report_json'])
    assert all(report[m]['total']==70 for m in calls)
    assert storage.one('SELECT status FROM jobs WHERE id=?',(r.json()['job_id'],))['status']=='completed'


def test_scoring_runs_two_modules_at_once(client, monkeypatch):
    from threading import Event, Lock, Thread

    a = upload(client, adult=True)
    aid = a['id']
    client.post(f'/api/assets/{aid}/target', json={'x': .1, 'y': .1, 'width': .8, 'height': .8})
    report = json.loads(storage.get_asset(aid)['report_json'])
    report['face_gate'] = {'assessable': True, 'limitations': []}
    storage.run('UPDATE assets SET report_json=?,external_consent=1 WHERE id=?', (json.dumps(report), aid))
    started, release, lock = Event(), Event(), Lock()
    active = peak = 0

    def evaluate(module, path, cancelled, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
        assert release.wait(5)
        with lock:
            active -= 1
        raw = model_output()
        raw['dimensions'] = [dict(key=k, status='assessable', score=7, evidence='测试可见依据', limitations=[])
                             for k, *_ in RULES[module]]
        if module == 'body':
            raw['scene'] = 'full_body'
        return raw, {'input_tokens': 1, 'output_tokens': 1}

    monkeypatch.setattr(provider, 'evaluate', evaluate)
    jid = storage.new_id()
    storage.run('INSERT INTO jobs(id,asset_id,status,modules_json,created_at,updated_at) VALUES(?,?,?,?,?,?)',
                (jid, aid, 'queued', json.dumps(['face', 'body', 'photography']), storage.now(), storage.now()))
    worker = Thread(target=pipeline.run_job, args=(jid,))
    worker.start()
    try:
        assert started.wait(5)
        assert peak == 2
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert storage.one('SELECT status FROM jobs WHERE id=?', (jid,))['status'] == 'completed'
    assert len(json.loads(storage.get_asset(aid)['report_json'])['usage']) == 3
