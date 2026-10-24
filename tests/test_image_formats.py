import io,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from app.images import inspect_image,oriented_rgb,preview_bytes,InvalidImage
from app.image_formats import FORMAT_MIME

@pytest.fixture
def client(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from app import config,storage
    from app.main import app
    monkeypatch.setattr(config,'DATA_DIR',tmp_path);monkeypatch.setattr(storage,'DATA_DIR',tmp_path)
    monkeypatch.setenv('BEAUTYPICK_PHOTO_MODEL','cv')
    with TestClient(app) as c:
        c.headers['x-portrait-token']=c.get('/api/session').json()['token']
        yield c

@pytest.mark.parametrize('fmt',['JPEG','PNG','WEBP','HEIF','AVIF','BMP','TIFF','GIF','JPEG2000'])
def test_decode_original_and_preview_roundtrip(tmp_path,fmt):
 rng=np.random.default_rng(7);im=Image.fromarray(rng.integers(20,230,(96,64,3),dtype=np.uint8));buf=io.BytesIO();im.save(buf,format=fmt);raw=buf.getvalue();path=tmp_path/'original';path.write_bytes(raw);meta=inspect_image(raw)
 assert meta['mime']==FORMAT_MIME[fmt] and (meta['width'],meta['height'])==(64,96)
 assert meta['sha256']==hashlib.sha256(raw).hexdigest();assert oriented_rgb(path).mode=='RGB'
 jpeg=Image.open(io.BytesIO(preview_bytes(path)));assert jpeg.format=='JPEG' and jpeg.size==(64,96)
 assert path.read_bytes()==raw

@pytest.mark.parametrize('fmt',['GIF','TIFF','WEBP','HEIF'])
def test_multiframe_rejected_without_silent_frame_loss(fmt):
 frames=[Image.new('RGB',(40,60),'red'),Image.new('RGB',(40,60),'blue')];buf=io.BytesIO();frames[0].save(buf,format=fmt,save_all=True,append_images=frames[1:])
 with pytest.raises(InvalidImage,match='动画或多页'):inspect_image(buf.getvalue())

def test_heif_in_fresh_worker_interpreter(tmp_path):
 p=tmp_path/'test.heic';Image.new('RGB',(64,96),'green').save(p,format='HEIF')
 code='from app.images import oriented_rgb;import sys;im=oriented_rgb(__import__("pathlib").Path(sys.argv[1]));assert im.mode=="RGB" and im.size==(64,96)'
 result=subprocess.run([sys.executable,'-c',code,str(p)],capture_output=True,text=True,timeout=30);assert result.returncode==0,result.stderr

def test_new_format_size_limit_and_corruption(monkeypatch):
 from app import images
 buf=io.BytesIO();Image.new('RGB',(40,60)).save(buf,format='HEIF');monkeypatch.setattr(images,'MAX_PIXELS',100)
 with pytest.raises(InvalidImage,match='像素数'):inspect_image(buf.getvalue())
 with pytest.raises(InvalidImage):inspect_image(b'not an image.heic')

@pytest.mark.parametrize('fmt,ext',[('HEIF','heic'),('AVIF','avif'),('BMP','bmp'),('TIFF','tiff'),('GIF','gif')])
def test_http_upload_new_formats(client,fmt,ext):
 buf=io.BytesIO();Image.new('RGB',(64,96),'green').save(buf,format=fmt)
 response=client.post('/api/assets',data={'rights_confirmed':'true','adult_confirmed':'true'},files={'file':('test.'+ext,buf.getvalue(),'application/octet-stream')});assert response.status_code==200,response.text
 a=response.json();assert a['mime']==FORMAT_MIME[fmt];r=client.get(a['preview_url']);assert r.status_code==200 and r.headers['content-type']=='image/jpeg'
