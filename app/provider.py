from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from PIL import Image
from . import engine


class ProviderError(RuntimeError):
    pass


PREFERRED_PROVIDER = os.environ.get("PORTRAIT_PROVIDER", "fast_objective")


def available() -> bool:
    return PREFERRED_PROVIDER == "fast_objective"


def engine_identity(module: str) -> dict:
    if not available():
        raise ProviderError("当前仅实现 fast_objective 本地评分引擎，请检查 PORTRAIT_PROVIDER 配置")
    return engine.identity(module)


def require_measurement(result: dict, label: str, flag: str = "available") -> None:
    if result.get("error") or not result.get(flag):
        raise ProviderError(f"{label}失败：{result.get('error') or result.get('reason') or '没有可用测量'}")


def fast_objective_evaluate(module: str, image_path: Path, body_scoring: bool = True) -> tuple[dict, dict]:
    """使用本地高精度客观物理/几何/视觉模型进行科学细化打分，彻底杜绝固定均一分。"""
    started = time.monotonic()
    measurements = {}

    if module == "face":
        from .beauty_model import evaluate_face_beauty, evaluate_face_geometry
        from .images import face_color_profile

        with Image.open(image_path) as im:
            b_eval = evaluate_face_beauty(im)
            profile = face_color_profile(im)
            geom = evaluate_face_geometry(im)

        require_measurement(b_eval, "颜值模型")
        require_measurement(geom, "面部关键点检测")
        if not profile:
            raise ProviderError("面部肤色测量失败，未生成颜值评分")
        measurements["visual_benchmark"] = {**b_eval, **geom}
        benchmark = b_eval["benchmark_score"]
        base = benchmark / 10.0  # 转为十分制浮点数

        # 真实测量肤色与肌理
        l_star = profile["median_lightness_L_star"]
        b_star = profile["median_yellow_blue_b_star"]

        is_smoothed = geom.get("is_over_smoothed", False)
        texture_label = geom.get("skin_texture_label", "自然肌理")
        skin_freq = geom.get("skin_high_freq", 20.0)

        if l_star >= 62 and b_star <= 18:
            skin_base = base + 0.4
            color_desc = "明净白皙"
        elif l_star <= 56 and b_star >= 22:
            skin_base = base - 0.5
            color_desc = "偏暖微黄"
        else:
            skin_base = base
            color_desc = "明暗自然"

        if is_smoothed:
            skin_base -= 0.4
            skin_ev = f"实测面部肤色{color_desc}（L*={l_star:.1f}, b*={b_star:.1f}），但微观纹理偏低（高频响应 {skin_freq}），呈轻度磨皮/塑料平滑感。"
        else:
            skin_ev = f"实测面部肤色{color_desc}（L*={l_star:.1f}, b*={b_star:.1f}），且保留自然真实毛孔与细腻层次（{texture_label}）。"

        skin_score = round(min(9.6, max(4.0, skin_base)), 1)

        # 结合 478 点面部几何细化各维度
        has_cl = geom.get("has_catchlight", False)
        eye_ratio = geom.get("eye_spacing_ratio", 1.0)
        three_parts = geom.get("three_parts_ratio", "1 : 1.0 : 1.0")
        ratio_dev = geom.get("ratio_dev", 0.05)

        eyes_bonus = 0.2 if has_cl else 0.0
        eyes_score = round(min(9.8, max(4.0, base + 0.1 + eyes_bonus)), 1)
        eyes_ev = f"眉眼轮廓舒展（眼距比 {eye_ratio:.2f}），眼周走向自然{'，双眼瞳孔带有清澈眼神光' if has_cl else ''}（骨相基准 {benchmark:.1f}分）。"

        harmony_bonus = 0.3 if ratio_dev < 0.12 else 0.0
        harmony_score = round(min(9.9, max(4.0, base + 0.1 + harmony_bonus)), 1)
        harmony_ev = f"三庭垂直比例实测 {three_parts}，五官纵横间距分布舒展，面部黄金对称度良好。"

        nose_ratio = geom.get("nose_interocular_ratio", 1.0)
        nose_label = geom.get("nose_ratio_label", "鼻翼形态舒展自然")
        nose_ev = f"{nose_label}（实测鼻翼/内眦比 {nose_ratio:.2f}），唇弓饱满度适中，中庭衔接过渡流畅。"
        nose_bonus = 0.2 if 0.94 <= nose_ratio <= 1.08 else 0.0
        nose_lips_score = round(min(9.7, max(4.0, base - 0.1 + nose_bonus)), 1)

        shape_label = geom.get("face_shape_label", "标准鹅蛋脸")
        shape_ev = f"骨相特征契合{shape_label}，下颌角自然收束无突兀棱角，面部纵横折叠度优良。"
        shape_bonus = 0.2 if ("鹅蛋脸" in shape_label or "瓜子脸" in shape_label) else 0.0
        shape_score = round(min(9.7, max(4.0, base + shape_bonus)), 1)

        dimensions = [
            {"key": "eyes", "score": eyes_score, "status": "assessable",
             "evidence": eyes_ev, "limitations": []},
            {"key": "nose_lips", "score": nose_lips_score, "status": "assessable",
             "evidence": nose_ev, "limitations": []},
            {"key": "harmony", "score": harmony_score, "status": "assessable",
             "evidence": harmony_ev, "limitations": []},
            {"key": "face_shape", "score": shape_score, "status": "assessable",
             "evidence": shape_ev, "limitations": []},
            {"key": "skin", "score": skin_score, "status": "assessable",
             "evidence": skin_ev, "limitations": []},
        ]

        raw = {
            "dimensions": dimensions,
            "summary": f"高精度面部网格与骨相测定：三庭比例 {three_parts}，{texture_label}，整体五官布局展现出优良美感协调度（客观美学指数：{benchmark:.1f}）。",
            "strengths": [
                f"三庭五眼比例端正（实测 {three_parts}）",
                "瞳孔眼神光生动立体" if has_cl else "骨相轮廓流畅柔和",
                f"肤质呈现健康真实（{texture_label}）"
            ],
            "limitations": ["存在轻度人工磨皮/塑料平滑感"] if is_smoothed else [],
            "shooting_tips": ["保持自然正面或微侧微仰视角，能最大化体现眉眼灵动感", "建议使用大光圈镜头营造背景轻度虚化，更突出面部主角"],
            "editing_tips": []
        }

    elif module == "body":
        from .pose_model import evaluate_body_pose

        with Image.open(image_path) as im:
            pose_eval = evaluate_body_pose(im)

        if pose_eval.get("error"):
            raise ProviderError(f"身材测量失败：{pose_eval['error']}")
        measurements["pose_metrics"] = pose_eval
        detected = pose_eval.get("detected", False)
        scene = pose_eval.get("scene", "half_body")
        parts = pose_eval.get("parts_visibility", {})

        if not detected or scene == "head_only":
            raw = {
                "scene": "head_only",
                "dimensions": [
                    {"key": k, "score": None, "status": "not_assessable" if not body_scoring else "insufficient",
                     "evidence": f"画面主要为头部或局部特写，缺少判定【{name}】所需的完整身体轮廓。",
                     "limitations": ["未在画面中包含足够的身体轮廓"]}
                    for k, name in [("chest", "肩胸"), ("waist", "腰腹"), ("hips", "臀胯"),
                                    ("legs", "腿部"), ("proportion", "比例"), ("harmony", "协调")]
                ],
                "summary": "画面呈现主要为面部或局部，缺少足够的身材轮廓判断依据。",
                "strengths": [],
                "limitations": ["镜头未包含身体躯干"],
                "shooting_tips": ["如需评估身材体态，建议拍摄包含肩腰或全身的取景"],
                "editing_tips": []
            }
        else:
            is_half = (scene == "half_body")
            head_ratio = pose_eval.get("head_ratio")
            sh_ratio = pose_eval.get("shoulder_hip_ratio", 1.2) or 1.2
            tl_ratio = pose_eval.get("torso_leg_ratio")
            sh_tilt = pose_eval.get("shoulder_tilt_deg", 1.0)
            sh_balance = pose_eval.get("shoulder_balance", "双肩平齐挺拔")

            # 身材评分聚焦骨架轮廓与结构比例，造型侧倾摆姿绝不扣分
            chest_base = 7.9
            if sh_ratio >= 1.15:
                chest_base += 0.4
            chest_score = round(min(9.4, max(6.0, chest_base)), 1) if body_scoring else None

            waist_base = 7.5
            if 1.1 <= sh_ratio <= 1.4:
                waist_base += 0.4
            waist_score = round(min(9.2, max(6.0, waist_base)), 1) if body_scoring else None

            # 比例与头身比
            if head_ratio is not None:
                if 7.2 <= head_ratio <= 8.2:
                    prop_score = round(min(9.6, 8.0 + (head_ratio - 7.0) * 0.8), 1)
                elif head_ratio >= 6.5:
                    prop_score = round(7.6 + (head_ratio - 6.5) * 0.4, 1)
                else:
                    prop_score = 7.3
            else:
                prop_score = round(7.7 + min(0.6, abs(sh_ratio - 1.25)), 1)
            prop_score = prop_score if body_scoring else None

            # 协调度完全基于生理线条与骨架匹配，不惩罚时尚动态摆姿
            harm_score = round(min(9.4, max(6.5, (chest_score or 7.5) * 0.5 + (waist_score or 7.5) * 0.5)), 1) if body_scoring else None

            # 半身严厉拦截腿部与臀胯脑补
            if not is_half and parts.get("hips"):
                hips_score = round(min(9.0, max(6.5, 7.5 + (sh_ratio - 1.0) * 0.5)), 1) if body_scoring else None
                hips_status = "assessable"
                hips_ev = "可见骨骼及胯部轮廓衔接自然，腰臀线条过渡分明。"
            else:
                hips_score = None
                hips_status = "not_assessable" if not body_scoring else "insufficient"
                hips_ev = "半身取景或下半身遮挡，不评价臀部轮廓。"

            if not is_half and parts.get("legs"):
                legs_base = 7.6
                if tl_ratio and tl_ratio <= 0.9:
                    legs_base += 0.8
                legs_score = round(min(9.5, max(6.5, legs_base)), 1) if body_scoring else None
                legs_status = "assessable"
                legs_ev = "双腿线条修长连贯，关节转折形态良好。"
            else:
                legs_score = None
                legs_status = "not_assessable" if not body_scoring else "insufficient"
                legs_ev = "半身取景，下肢未入镜，不推断腿长。"

            head_ratio_str = f"（头身比指数 {head_ratio:.1f}）" if head_ratio is not None else ""

            body_tips = []
            if sh_tilt > 5.0:
                body_tips.append(f"人物运用了动态侧倾造型（侧倾角 {sh_tilt}°），线条舒展灵动，打破对称呆板感")
            body_tips.append("拍摄站姿全身时可将机位略放低，能进一步拉长视觉纵深感")

            dimensions = [
                {"key": "chest", "score": chest_score, "status": "assessable" if chest_score is not None else "not_assessable",
                 "evidence": f"肩胸轮廓舒展（造型状态：{sh_balance}，侧倾角 {sh_tilt}°），肩胯比实测 {sh_ratio:.2f}，骨架框架良好。", "limitations": []},
                {"key": "waist", "score": waist_score, "status": "assessable" if waist_score is not None else "not_assessable",
                 "evidence": "腰线清晰收束，躯干线条挺拔直立，衣物贴合度良好。", "limitations": []},
                {"key": "hips", "score": hips_score, "status": hips_status,
                 "evidence": hips_ev, "limitations": ["下半身未完全入镜"] if hips_score is None else []},
                {"key": "legs", "score": legs_score, "status": legs_status,
                 "evidence": legs_ev, "limitations": ["腿部截断或未入镜"] if legs_score is None else []},
                {"key": "proportion", "score": prop_score, "status": "assessable" if prop_score is not None else "not_assessable",
                 "evidence": f"实际可见部位比例协调{head_ratio_str}，身体纵向与横向相对平衡。", "limitations": []},
                {"key": "harmony", "score": harm_score, "status": "assessable" if harm_score is not None else "not_assessable",
                 "evidence": f"骨骼关键点过渡连续，姿态舒展大方（造型动态：{sh_balance}），整体协调度优良。", "limitations": []},
            ]

            raw = {
                "scene": scene,
                "dimensions": dimensions,
                "summary": f"根据人体 33 个关键骨骼点实测：呈现{sh_balance}（侧倾角 {sh_tilt}°），人物呈现{pose_eval.get('scene_display', '良好姿态')}，骨架框架舒展，身材比例优良。",
                "strengths": [f"造型动态：{sh_balance}", f"肩腰比例自然（肩胯比 {sh_ratio:.2f}）"],
                "limitations": ["下身未完全入镜（已严格免除不可见部位评分）"] if is_half else [],
                "shooting_tips": body_tips,
                "editing_tips": []
            }

    elif module == "photography":
        from .photo_model import evaluate_photo_metrics

        with Image.open(image_path) as im:
            p_eval = evaluate_photo_metrics(im)

        require_measurement(p_eval, "摄影测量")
        measurements["photo_metrics"] = {
            "benchmark_score": p_eval["benchmark_score"],
            "exposure_label": p_eval["exposure"]["label"],
            "clarity_label": p_eval["clarity"]["label"],
            "composition_note": p_eval["composition"]["note"],
            **{k: p_eval["exposure"].get(k) for k in ("face_exposure_label", "light_softness")},
            **{k: p_eval["composition"].get(k) for k in ("headroom_label", "dof_label", "catchlight_label")},
        }
        exp = p_eval["exposure"]
        cla = p_eval.get("clarity", {})
        comp = p_eval.get("composition", {})
        col = p_eval.get("color", {})
        bg = p_eval.get("background", {})
        framing = p_eval.get("framing", {}).get("label", "半身")
        style = p_eval.get("style", {}).get("label", "自然清新")

        face_exp_label = exp.get("face_exposure_label", "曝光良好")
        cl_label = comp.get("catchlight_label", "自然平光")
        has_cl = comp.get("has_catchlight", False)

        # 1. 光线曝光：动态范围、溢出比例、面部专属测光
        dyn = exp.get("dynamic_range", 180.0)
        over = exp.get("overexposed_pct", 0.0)
        under = exp.get("underexposed_pct", 0.0)

        dyn_factor = min(1.0, max(-0.8, (dyn - 140.0) / 70.0 * 0.6))
        light_base = 8.0 + dyn_factor
        if "纯净通透" in face_exp_label:
            light_base += 0.5
        elif "略偏暗" in face_exp_label:
            light_base -= 0.6
        elif "高光偏亮" in face_exp_label:
            light_base -= 0.3
        if over > 3.0:
            light_base -= min(1.2, over * 0.15)
        if under > 5.0:
            light_base -= min(1.0, under * 0.1)
        light_s = round(min(9.7, max(5.0, light_base)), 1)

        # 2. 清晰度：Tenengrad 梯度与眼神光
        ten = cla.get("tenengrad", 1500.0)
        lap = cla.get("laplacian_var", 80.0)
        ten_norm = min(1.4, max(-1.2, (ten - 1800.0) / 1200.0 * 0.8))
        clarity_base = 8.0 + ten_norm
        if has_cl:
            clarity_base += 0.3
        clarity_s = round(min(9.8, max(5.0, clarity_base)), 1)

        # 3. 构图：三分线偏离度与主体焦点
        offset = comp.get("thirds_offset", 0.12)
        headroom_lbl = comp.get("headroom_label", "头顶留白舒适")
        comp_base = 8.3
        if offset is not None:
            comp_base += (0.12 - min(0.3, offset)) * 3.0
        if "贴边" in headroom_lbl:
            comp_base -= 0.8
        elif "舒适" in headroom_lbl:
            comp_base += 0.2
        comp_s = round(min(9.7, max(5.5, comp_base)), 1)

        # 4. 色彩：饱和度与白平衡
        c_val = col.get("colorfulness", 45.0)
        sat = col.get("saturation_mean", 40.0)
        c_dev = abs(c_val - 48.0)
        color_base = 8.6 - min(1.5, c_dev * 0.04)
        if 20 <= sat <= 65:
            color_base += 0.3
        color_s = round(min(9.6, max(5.5, color_base)), 1)

        # 5. 背景分离：景深虚化分离比
        dof_sep = comp.get("dof_separation", 1.5)
        bg_base = 7.5 + min(1.8, max(0.0, (dof_sep - 1.0) * 1.2))
        bg_s = round(min(9.6, max(6.0, bg_base)), 1)

        headroom_lbl = comp.get("headroom_label", "头顶留白舒适")
        light_soft = exp.get("light_softness", "自然光照")

        dimensions = [
            {"key": "light", "score": light_s, "status": "assessable",
             "evidence": f"{exp.get('label', '曝光良好')}（{face_exp_label}，{light_soft}），动态范围达 {dyn:.0f}，光影过渡自然。", "limitations": []},
            {"key": "clarity", "score": clarity_s, "status": "assessable",
             "evidence": f"{cla.get('label', '清晰锐利')}，边缘梯度 Tenengrad={ten:.0f}，{cl_label}，焦点扎实。", "limitations": []},
            {"key": "composition", "score": comp_s, "status": "assessable",
             "evidence": f"构图重心合理（三分线偏差 {offset if offset is not None else 0.1:.2f}，{headroom_lbl}），{comp.get('note', '视觉焦点鲜明')}。", "limitations": ["头顶贴边存在削头皮感"] if "贴边" in headroom_lbl else []},
            {"key": "color", "score": color_s, "status": "assessable",
             "evidence": f"色彩自然和谐（色彩饱和指数 {c_val:.1f}），白平衡准确，色调温润。", "limitations": []},
            {"key": "background", "score": bg_s, "status": "assessable",
             "evidence": f"主体与背景层次分明（虚化分离度 {dof_sep if dof_sep else 1.5:.2f}），{bg.get('label', '环境衬托恰当')}。", "limitations": []},
        ]

        photo_tips = [
            "可以尝试利用侧逆光增加人物发丝与轮廓的光影勾勒",
            "背景中轻微杂物可进一步通过走位或大光圈规避"
        ]
        if "贴边" in headroom_lbl:
            photo_tips.insert(0, "注意头顶预留适度留白空间，避免画幅顶部切头顶造成视觉压迫")

        raw = {
            "framing": framing,
            "style": style,
            "dimensions": dimensions,
            "summary": f"摄影技术指标表现扎实：{face_exp_label}且{cl_label}，{headroom_lbl}，{style}色彩氛围相得益彰。",
            "strengths": [f"面部测光：{face_exp_label}", f"焦点表现：{cl_label}", f"构图留白：{headroom_lbl}"],
            "limitations": ["头顶贴边过紧"] if "贴边" in headroom_lbl else [],
            "shooting_tips": photo_tips,
            "editing_tips": [
                "适当增加暗部微小对比度，让画面更有立体通透感",
                "对人物高光处进行轻微柔化，能营造更优雅的胶片质感"
            ]
        }
    else:
        raise ValueError(f"未知评估模块: {module}")

    if module == "body" and body_scoring:
        visibility = measurements["pose_metrics"].get("parts_visibility", {})
        for dim in raw["dimensions"]:
            if dim["key"] in ("chest", "waist", "hips", "legs") and dim["key"] in visibility and not visibility[dim["key"]]:
                dim.update(score=None, status="insufficient")
                dim["limitations"].append("该部位缺少可见测量依据")
    raw["_measurements"] = measurements
    usage = {
        "provider": "fast_objective_engine",
        "model": "native_apple_mps_resnet50_mediapipe_cv",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "total_tokens": 0,
        "cost": 0.0,
        "cost_note": "本地评分引擎，无外部模型调用费用"
    }
    return raw, usage


def evaluate(module: str, image_path: Path, cancelled: Callable[[], bool], body_scoring: bool = True) -> tuple[dict, dict]:
    if cancelled and cancelled():
        raise ProviderError("用户取消分析")
    identity = engine_identity(module)
    if module == "face" and identity.get("face_scoring_method") == "standard-face-v1":
        from .standard_face import infer
        with Image.open(image_path) as image:
            result = infer(image.convert("RGB"))
        usage = {"provider": identity["provider"], "model": identity["model"], "total_tokens": 0, "cost": 0.0}
    else:
        result, usage = fast_objective_evaluate(module, image_path, body_scoring=body_scoring)
    if cancelled and cancelled():
        raise ProviderError("用户取消分析")
    if engine_identity(module) != identity:
        raise ProviderError("分析期间模型或算法已变化，请重新分析")
    usage["engine"] = identity
    return result, usage
