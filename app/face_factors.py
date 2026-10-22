"""Frozen one-factor face-input tests; no scoring/ranking-dependent preprocessing."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from .config import ROOT
from .engine import dependency_versions, file_fingerprint
from .face_preprocessing import INPUT_SIZE, SCALE, infer, landmarks, scale_score

VERSION = "face-single-factor-20261022-1"
SPECS = {
    "baseline": {"factor":1.,"resize":"pil_bilinear","rotation":False},
    "crop065": {"factor":.65,"resize":"pil_bilinear","rotation":False},
    "crop085": {"factor":.85,"resize":"pil_bilinear","rotation":False},
    "crop115": {"factor":1.15,"resize":"pil_bilinear","rotation":False},
    "bicubic": {"factor":1.,"resize":"pil_bicubic","rotation":False},
    "lanczos": {"factor":1.,"resize":"pil_lanczos","rotation":False},
    "cv_linear": {"factor":1.,"resize":"cv_linear","rotation":False},
    "cv_area": {"factor":1.,"resize":"cv_area","rotation":False},
    "rotation0": {"factor":.85,"resize":"warp_linear","rotation":False},
    "rotation_eye": {"factor":.85,"resize":"warp_linear","rotation":True},
}
LABELS = {"baseline":"原取景 · 双线性", "crop065":"收紧至65%", "crop085":"收紧至85%", "crop115":"放宽至115%",
          "bicubic":"PIL 双三次", "lanczos":"PIL Lanczos", "cv_linear":"OpenCV 双线性", "cv_area":"OpenCV 面积采样",
          "rotation0":"固定85%画幅 · 不旋转", "rotation_eye":"相同画幅 · 眼部水平对齐"}
GROUPS = {
    "crop": {"name":"裁脸松紧", "control":"baseline", "variants":["baseline","crop065","crop085","crop115"]},
    "resize": {"name":"缩放方式", "control":"baseline", "variants":["baseline","bicubic","lanczos","cv_linear","cv_area"]},
    "rotation": {"name":"旋转", "control":"rotation0", "variants":["rotation0","rotation_eye"]},
}


def identity():
    return {"version":VERSION,"model":"SCUT-FBP5500-ResNet50","specs":SPECS,"groups":GROUPS,
            "scale":SCALE,"input_size":INPUT_SIZE,"reference_used_for_scoring":False,
            "weights":{p:file_fingerprint(ROOT/"models"/p) for p in ("scut_fbp5500_beauty.pt","face_landmarker.task")},
            "implementation":{p:file_fingerprint(ROOT/"app"/p) for p in
                              ("face_factors.py","face_factor_trial.py","face_preprocessing.py","beauty_model.py","images.py")},
            "dependencies":dependency_versions()}


def source_box(base: dict, factor: float) -> tuple[int,int,int,int]:
    if factor==1.:
        return tuple(base[k] for k in ("left","top","right","bottom"))
    width,height=base["right"]-base["left"],base["bottom"]-base["top"]
    cx,cy=(base["left"]+base["right"])/2,(base["top"]+base["bottom"])/2
    w,h=round(width*factor),round(height*factor)
    left,top=round(cx-w/2),round(cy-h/2)
    return left,top,left+w,top+h


def eye_angle(points: np.ndarray) -> float:
    eyes=sorted([(points[33]+points[133])/2,(points[362]+points[263])/2],key=lambda p:p[0])
    d=eyes[1]-eyes[0]
    if np.linalg.norm(d)<8:raise ValueError("眼部像素不足")
    return math.degrees(math.atan2(d[1],d[0]))


def resize(image: Image.Image, method: str) -> Image.Image:
    methods={"pil_bilinear":Image.Resampling.BILINEAR,"pil_bicubic":Image.Resampling.BICUBIC,"pil_lanczos":Image.Resampling.LANCZOS}
    if method in methods:return image.resize((INPUT_SIZE,INPUT_SIZE),methods[method])
    if method not in ("cv_linear","cv_area"):raise ValueError("未知缩放方式")
    array=cv2.resize(np.asarray(image.convert("RGB")),(INPUT_SIZE,INPUT_SIZE),
                     interpolation=cv2.INTER_LINEAR if method=="cv_linear" else cv2.INTER_AREA)
    return Image.fromarray(array)


def rotation_matrix(box, degrees: float) -> np.ndarray:
    left,top,right,bottom=box
    if right-left!=bottom-top:raise ValueError("旋转组需要相同边长，不能混入拉伸")
    center=np.array([(left+right-1)/2,(top+bottom-1)/2])
    angle=math.radians(degrees);c,s=math.cos(angle),math.sin(angle)
    rotation=np.array([[c,s],[-s,c]])
    scale=(INPUT_SIZE-1)/(right-left-1)
    return np.column_stack((scale*rotation,(INPUT_SIZE-1)/2-scale*rotation@center))


def prepare_inputs(image: Image.Image, base: dict, points: np.ndarray | None, geometry_error: str | None=None) -> dict:
    image=image.convert("RGB");rgb=np.asarray(image);outputs={}
    for name,spec in SPECS.items():
        box=source_box(base,spec["factor"])
        left,top,right,bottom=box
        record={"spec":dict(spec),"source_box":list(box),"base_box":dict(base),"padding_fraction":0.}
        try:
            if min(left,top)<0 or right>image.width or bottom>image.height:
                raise ValueError("画幅超出原图，未补齐边缘，排除此项比较")
            if spec["resize"]=="warp_linear":
                if points is None:raise ValueError(geometry_error or "关键点不可用，旋转组缺少可靠依据")
                angle=eye_angle(points) if spec["rotation"] else 0.
                matrix=rotation_matrix(box,angle)
                transformed=points[:468]@matrix[:,:2].T+matrix[:,2]
                if np.any(transformed<0) or np.any(transformed>223):
                    raise ValueError("旋转后面部关键点被截断，排除此项比较")
                # Check the source footprint exactly, including rotated corners.
                inverse=cv2.invertAffineTransform(matrix)
                corners=np.array([[0.,0.],[223.,0.],[0.,223.],[223.,223.]])@inverse[:,:2].T+inverse[:,2]
                if np.any(corners<0) or np.any(corners[:,0]>image.width-1) or np.any(corners[:,1]>image.height-1):
                    raise ValueError("旋转采样超出原图，未补齐边缘，排除此项比较")
                actual=Image.fromarray(cv2.warpAffine(rgb,matrix,(INPUT_SIZE,INPUT_SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
                record.update(rotation_degrees=angle,matrix=matrix.tolist())
            else:
                if points is not None:
                    inside=(points[:468,0]>=left)&(points[:468,0]<right)&(points[:468,1]>=top)&(points[:468,1]<bottom)
                    record["landmark_coverage"]=float(inside.mean())
                    if not inside.all():raise ValueError("收紧画幅截断面部关键点，排除此项比较")
                actual=resize(image.crop(box),spec["resize"])
                record["rotation_degrees"]=0.
            record.update(status="ready",image=actual)
        except ValueError as exc:
            record.update(status="not_comparable",error=str(exc))
        outputs[name]=record
    return outputs


def geometry(image: Image.Image,target: dict):
    from .images import crop_face
    _,base=crop_face(image,target)
    try:
        points,_,_=landmarks(image,target)
        return base,points,None
    except Exception as exc:
        # Non-rotation tests do not depend on keypoint success.
        return base,None,str(exc)


def distance(raw_a,raw_b):
    return scale_score(raw_a)["unclipped_total"]-scale_score(raw_b)["unclipped_total"]
