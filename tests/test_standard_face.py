import numpy as np
import pytest
from PIL import Image
from app.standard_face import standard_crop


def points():
    rng=np.random.default_rng(42)
    return rng.uniform([180,160],[300,320],(478,2))


def test_face_occupancy_is_fixed_and_both_axes_keep_same_scale():
    image=Image.new('RGB',(600,600));p=points()
    crop,info=standard_crop(image,p)
    left,top,right,bottom=info['source_box']
    assert crop.size==(224,224)
    assert right-left==pytest.approx(bottom-top)
    assert max(np.ptp(p[:468],axis=0))/(right-left)==pytest.approx(1/1.24)
    assert info['rotation_degrees']==0


def test_different_original_framing_cannot_change_same_face_input():
    rng=np.random.default_rng(9)
    image=Image.fromarray(rng.integers(0,256,(600,600,3),dtype=np.uint8))
    p=points();a,_=standard_crop(image,p)
    cropped=image.crop((80,80,450,450))
    b,_=standard_crop(cropped,p-[80,80])
    assert a.tobytes()==b.tobytes()


def test_invalid_or_incomplete_input_has_no_loose_crop_fallback():
    image=Image.new('RGB',(600,600))
    with pytest.raises(ValueError):standard_crop(image,np.zeros((5,2)))
    with pytest.raises(ValueError):standard_crop(image,points()-[180,160])


def test_face_center_and_occupancy_do_not_depend_on_original_size():
    p=points()
    _,a=standard_crop(Image.new('RGB',(600,600)),p)
    _,b=standard_crop(Image.new('RGB',(1200,1200)),p*2)
    assert np.array(b['source_box'])==pytest.approx(np.array(a['source_box'])*2)
