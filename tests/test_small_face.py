import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from app.images import face_assessability,MIN_FACE_PIXELS,oriented_rgb
from app.standard_face import prepare

def test_77_pixels_is_warning_not_rejection():
 im=Image.fromarray(np.random.default_rng(9).integers(0,255,(146,146,3),dtype=np.uint8))
 gate=face_assessability({'pixel_width':77,'pixel_height':77},im)
 assert gate['assessable'] and gate['warnings'] and gate['source_face_pixels']==77
 assert not face_assessability({'pixel_width':MIN_FACE_PIXELS-1,'pixel_height':77},im)['assessable']
def test_known_original_small_face_has_valid_standard_crop():
 root=Path(__file__).resolve().parents[1];path=root/'data/assets/89802855-038f-49ca-b35f-e8418c13ee38'
 if not path.exists():pytest.skip('private regression photo not distributed')
 import sqlite3
 with sqlite3.connect(f'file:{root}/data/portrait.db?mode=ro',uri=True) as con:target=json.loads(con.execute('SELECT target_json FROM assets WHERE id=?',(path.name,)).fetchone()[0])
 im=oriented_rgb(path);face,diag=prepare(im,target)
 assert face.size==(224,224) and diag['face_extent'][0]>40 and diag['face_extent'][1]>40
 box=diag['source_box'];assert box[2]-box[0]==pytest.approx(box[3]-box[1])
