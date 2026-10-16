from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def evaluate_photo_metrics(image: Image.Image, target_box: dict | None = None) -> dict:
    """Analyze photographic quality, exposure, sharpness, color, and composition.
    
    Returns structured metrics and an objective benchmark score.
    """
    try:
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]

        # Downscale for performance if very large, but keep high enough for accurate sharpness
        scale = 1.0
        if max(h, w) > 1600:
            scale = 1600.0 / max(h, w)
            proc_rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            proc_rgb = rgb

        gray = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2GRAY)
        ph, pw = gray.shape[:2]

        # 1. Exposure and Dynamic Range
        lum_mean = float(np.mean(gray))
        lum_std = float(np.std(gray))
        p5, p95 = np.percentile(gray, [5, 95])
        dyn_range = float(p95 - p5)
        over_pct = float(np.mean(gray > 245) * 100)
        under_pct = float(np.mean(gray < 10) * 100)

        exposure_label = "曝光适中"
        if over_pct > 5.0 and under_pct > 5.0:
            exposure_label = "大光比/反差强烈"
        elif over_pct > 6.0:
            exposure_label = "偏亮/高光有溢出风险"
        elif under_pct > 8.0:
            exposure_label = "偏暗/暗部较深"
        elif lum_std < 35.0:
            exposure_label = "柔和/低反差"

        # 2. Clarity and Sharpness
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(sobelx**2 + sobely**2))

        if lap_var >= 120 or tenengrad >= 2000:
            clarity_label = "对焦锐利/细节丰富"
            clarity_score = 8.5
        elif lap_var >= 45 or tenengrad >= 900:
            clarity_label = "对焦正常/清晰度良好"
            clarity_score = 7.5
        elif lap_var >= 20:
            clarity_label = "轻微偏软/微模糊"
            clarity_score = 6.0
        else:
            clarity_label = "明显失焦或运动模糊"
            clarity_score = 4.0

        # 3. Color Harmony and Saturation
        r, g, b = proc_rgb[:, :, 0].astype(float), proc_rgb[:, :, 1].astype(float), proc_rgb[:, :, 2].astype(float)
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

        hsv = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]) / 255.0 * 100)

        # White balance tone tendency
        lab = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2LAB)
        b_mean = float(np.mean(lab[:, :, 2]) - 128)
        color_tone = "中性微暖" if b_mean > 5 else "清爽偏冷" if b_mean < -3 else "中性平衡"

        # 4. Composition and Subject Placement
        composition_note = "标准居中或均衡"
        thirds_offset = 0.0
        dof_separation = 1.0
        face_lum_mean = lum_mean
        face_exposure_label = "全幅曝光平衡"

        if not target_box:
            try:
                from .images import detect_faces
                faces = detect_faces(image)
                if faces:
                    target_box = faces[0]
            except Exception:
                pass

        if target_box:
            # target_box has x, y, width, height in normalized [0, 1]
            bx = target_box.get("x", 0.5)
            by = target_box.get("y", 0.5)
            bw_box = target_box.get("width", 0.2)
            bh_box = target_box.get("height", 0.2)
            cx, cy = bx + bw_box / 2, by + bh_box / 2

            # Distance to rule of thirds lines (x: 0.333, 0.667; y: 0.333, 0.667)
            dist_x = min(abs(cx - 0.333), abs(cx - 0.667), abs(cx - 0.5))
            dist_y = min(abs(cy - 0.333), abs(cy - 0.667))
            thirds_offset = round(float(math.hypot(dist_x, dist_y)), 3)

            if thirds_offset < 0.12:
                composition_note = "契合三分构图/视线留白舒适"
            elif abs(cx - 0.5) < 0.08:
                composition_note = "中心对称/主体突出构图"
            else:
                composition_note = "开放式偏心构图"

            # Subject vs Background blur ratio (Depth of Field separation)
            x1, y1 = max(0, int(bx * pw)), max(0, int(by * ph))
            x2, y2 = min(pw, int((bx + bw_box) * pw)), min(ph, int((by + bh_box) * ph))

            if y2 > y1 and x2 > x1:
                lap_all = cv2.Laplacian(gray, cv2.CV_64F)
                sub_var = float(lap_all[y1:y2, x1:x2].var())
                mask = np.ones((ph, pw), dtype=bool)
                mask[y1:y2, x1:x2] = False
                bg_var = float(lap_all[mask].var()) if mask.sum() > 200 else 1.0

                if bg_var > 0:
                    dof_separation = round(sub_var / max(1.0, bg_var), 2)

                # 面部专属加权测光与光质柔硬分析
                face_gray = gray[y1:y2, x1:x2]
                face_lum_mean = float(face_gray.mean())
                face_over = float(np.mean(face_gray > 240) * 100)
                face_under = float(np.mean(face_gray < 25) * 100)
                if 110 <= face_lum_mean <= 165 and face_over < 3.0:
                    face_exposure_label = "面部曝光纯净通透"
                elif face_lum_mean < 110:
                    face_exposure_label = "面部受光略偏暗"
                else:
                    face_exposure_label = "面部高光偏亮"

                f_gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
                f_gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
                face_grad = cv2.magnitude(f_gx, f_gy)
                grad_std = float(np.std(face_grad))
                if grad_std >= 38.0 or (face_over > 3.0 and face_under > 3.0):
                    light_softness = "直射硬光/阴影对比清晰"
                elif grad_std <= 20.0:
                    light_softness = "柔和漫射光/过渡温润"
                else:
                    light_softness = "自然柔顺光照"

        # 头顶留白与卡头皮构图检测 (Headroom)
        headroom_pct = round(float(target_box.get("y", 0.1)), 3) if target_box else 0.1
        if headroom_pct < 0.025:
            headroom_label = "头顶贴边/削头皮风险"
            headroom_note = "头顶距画幅顶边过紧，视觉缺乏呼吸感"
        elif 0.04 <= headroom_pct <= 0.22:
            headroom_label = "头顶留白舒适"
            headroom_note = "头顶与画框顶部比例适宜"
        else:
            headroom_label = "大幅留白构图"
            headroom_note = "上方空间充裕"

        dof_label = "浅景深良好/背景自然虚化" if dof_separation >= 1.6 else "大景深/环境细节清晰"

        # 眼神光检测 (Catchlight)
        has_catchlight = False
        try:
            from .beauty_model import evaluate_face_geometry
            geom = evaluate_face_geometry(image)
            has_catchlight = geom.get("has_catchlight", False)
        except Exception:
            pass

        catchlight_label = "眼神光清澈生动" if has_catchlight else "眼神光较微弱/自然平光"

        # Overall technical benchmark score (40 to 95)
        exp_score = 8.5 if 120 <= dyn_range <= 230 and over_pct < 3 and under_pct < 4 else 7.0 if over_pct < 7 else 5.5
        if "纯净通透" in face_exposure_label:
            exp_score = min(9.5, exp_score + 0.5)
        color_score = 8.5 if 25 <= colorfulness <= 75 and 15 <= sat_mean <= 70 else 7.0
        comp_score = 8.5 if "三分" in composition_note or "虚化" in dof_label else 7.5
        if has_catchlight:
            comp_score = min(9.5, comp_score + 0.3)
        if "贴边" in headroom_label:
            comp_score = max(6.0, comp_score - 0.5)

        benchmark = round(clarity_score * 0.3 + exp_score * 0.3 + color_score * 0.2 + comp_score * 0.2, 1) * 10.0
        benchmark = max(40.0, min(96.0, benchmark))

        return {
            "available": True,
            "benchmark_score": benchmark,
            "exposure": {
                "luminance_mean": round(lum_mean, 1),
                "dynamic_range": round(dyn_range, 1),
                "overexposed_pct": round(over_pct, 2),
                "underexposed_pct": round(under_pct, 2),
                "label": exposure_label,
                "face_lum_mean": round(face_lum_mean, 1),
                "face_exposure_label": face_exposure_label,
                "light_softness": light_softness if 'light_softness' in locals() else "自然光照",
            },
            "clarity": {
                "laplacian_var": round(lap_var, 1),
                "tenengrad": round(tenengrad, 1),
                "label": clarity_label,
            },
            "color": {
                "colorfulness": round(colorfulness, 1),
                "saturation_mean": round(sat_mean, 1),
                "tone": color_tone,
            },
            "composition": {
                "note": composition_note,
                "thirds_offset": thirds_offset,
                "dof_separation": dof_separation,
                "dof_label": dof_label,
                "headroom_label": headroom_label,
                "headroom_note": headroom_note,
                "headroom_pct": headroom_pct,
                "has_catchlight": has_catchlight,
                "catchlight_label": catchlight_label,
            },
        }
    except Exception as exc:
        logger.warning("Photographic metric analysis failed: %s", exc)
        return {
            "available": False,
            "error": f"摄影测量异常: {type(exc).__name__}",
        }


def photo_context_for_prompt(photo_res: dict) -> str:
    """Generate structured photographic technical measurements for Codex prompt."""
    if not photo_res or not photo_res.get("available"):
        return ""

    exp = photo_res["exposure"]
    cla = photo_res["clarity"]
    col = photo_res["color"]
    comp = photo_res["composition"]
    bench = photo_res["benchmark_score"]

    return (
        f"\n【图像摄影技术指标客观测量】：\n"
        f"1. 曝光与光比：动态范围 {exp['dynamic_range']}（高光溢出率 {exp['overexposed_pct']}%, 暗部死黑率 {exp['underexposed_pct']}%），评价倾向：{exp['label']}。\n"
        f"2. 清晰度与对焦：拉普拉斯方差 {cla['laplacian_var']}，Tenengrad 边缘能量 {cla['tenengrad']}，技术判定：{cla['label']}。\n"
        f"3. 色彩与白平衡：色彩丰富度指数 {col['colorfulness']}，平均饱和度 {col['saturation_mean']}%，整体色调：{col['tone']}。\n"
        f"4. 构图与景深：主体位置判断：{comp['note']}；景深分离度 {comp['dof_separation']}（{comp['dof_label']}）。\n"
        f"5. 客观技术质量基准：约 {bench:.1f} 分。请以这些物理测量数据作为 light, clarity, color, composition, background 5 个维度的打分事实依据，严禁捏造不存在的高清或忽视明显过曝。"
    )
