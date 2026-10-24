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
    with pytest.raises(ValueError):standard_crop(image,points()-[190,170])


def test_face_center_and_occupancy_do_not_depend_on_original_size():
    p=points()
    _,a=standard_crop(Image.new('RGB',(600,600)),p)
    _,b=standard_crop(Image.new('RGB',(1200,1200)),p*2)
    assert np.array(b['source_box'])==pytest.approx(np.array(a['source_box'])*2)


def test_border_padding_keeps_the_same_square_and_face_scale():
    import json
    rng=np.random.default_rng(14)
    image=Image.fromarray(rng.integers(0,256,(260,200,3),dtype=np.uint8))
    p=rng.uniform([5,12],[195,246],(478,2))
    result,info=standard_crop(image,p)
    assert info['padding'] and info['padding_method']=='edge'
    assert json.loads(json.dumps(info))['padding'] is True
    left,top,right,bottom=info['source_box']
    assert right-left==pytest.approx(bottom-top)
    assert max(np.ptp(p[:468],axis=0))/(right-left)==pytest.approx(1/1.24)
    # Equivalent input with a physically extended edge border must be identical;
    # neither axes nor the face centre may be stretched/recentred.
    canvas=Image.fromarray(np.pad(np.asarray(image),((100,100),(100,100),(0,0)),mode='edge'))
    expected,normal=standard_crop(canvas,p+[100,100])
    assert not normal['padding']
    assert result.tobytes()==expected.tobytes()


def test_padding_never_fills_missing_facial_landmarks():
    p=points()-[200,180]
    with pytest.raises(ValueError,match='关键点超出'):
        standard_crop(Image.new('RGB',(200,200)),p)
