---
name: beauty-pick
description: 当用户需要使用“颜选 · BeautyPick”应用评估人像照片、分析颜值与身材比例、检查摄影技术质量及对比选片时使用此技能。支持来源凭据检查与多维排行。
metadata:
  version: "0.2.0"
---

# 颜选 · BeautyPick Skill

## 1. 核心定位与架构关系

- **应用项目路径**：`{{BEAUTYPICK_DIR}}`（由 `./setup.sh` 安装时自动填入本机真实绝对路径，或可通过环境变量 `$BEAUTYPICK_HOME` 覆盖）。
- **系统层级**：本机 React 19 + FastAPI + SQLite 应用。Skill 是 Agent 的调用入口与排查指引。
- **算法唯一事实来源**：正式颜值裁脸与总分实现在 `app/standard_face.py`、`app/face_preprocessing.py`；调度与其他评分位于 `app/provider.py`、`app/pipeline.py`、`app/scoring.py`。**Skill 中严禁维护独立或分叉的评分算法**。
- **评估引擎**：正式本地评分与独立研发试验分开。
  - **正式评分**：纯本地ResNet50、MediaPipe与图像测量，不依赖Codex账号或云端模型。颜值采用标准面部版；身材与摄影保留现有本地测量规则。
  - **研发试验**：Codex参考与其他候选代码仅用于离线对照，不是默认评分引擎；不得擅自发送图片、启用试验或将其结果覆盖正式记录。
- **数据保留**：所有图片及评分记录默认均为永久保存（`retained = 1`），无 24 小时自动过期机制。除非获得用户显式删除授权，严禁清理素材及历史台账。

---

## 2. 模块评分行为与原则

1. **三模块并行分析**：
   - 网页端上传及“重新分析”默认对**颜值、身材、摄影**全量分析；CLI 默认亦为全量三模块分析。
   - 研发人员如需单独测试某一模块，可在 CLI 中明确指定 `--modules face` 等。
2. **三模块评分规范与量化原则**：
   - **颜值 (Face, 满分 100)**：
     - 正式方式：从原图已选目标检测关键点，以固定12%边距生成正方形面部输入，保留宽高比例、不旋转，双线性缩放至224×224后由原ResNet50输出整体分。
     - 不输出独立部位评分，不通过五项加权回算；不得编造额头、鼻子、皮肤或骨相的独立美感结论。
     - 沿用旧固定百分制换算，保留一位小数、显示范围35–99。高分截顶不等于接近完美，现有模型审美准确性未通过广泛验证。
     - 关键点缺失、像素不足或标准框越界时明确失败，不回退到另一种裁脸或补默认分。
   - **身材 (Body, 满分 100)**：
     - 细项：肩胸25、腰腹20、臀胯15、腿部15、比例15、协调10（规则版本 `body-appearance-3.0`）。
     - **姿态与造型解耦原则**：模特因拍摄造型摆出的姿势（如高低肩、侧倾、扭胯等）已从身材形体中完全解耦，**严禁作为身材缺陷扣分**；骨骼姿态分析仅用于客观姿态数据记录（倾角、头身比）与生成实操拍摄建议。
     - 动态重权：不可见部位为 `null`，系统自动按可见项动态重权归一，有可见项即可出总分。
   - **摄影 (Photography, 满分 100)**：
     - 细项：曝光与光线25、清晰度与技术质量20、构图与视角20、色彩与后期20、背景与视觉叙事15。
     - **连续量化原则**：全面采用 **0.1 连续精细浮点分体系**（总分与细项均精确到 1 位小数），基于面部测光纯净度、Tenengrad 边缘梯度、黄金三分线偏差、主体与背景景深虚化分离比（`dof_separation`）拉开技术差距。
   - **严禁臆造**：总分为 `null` 严禁补零、补分或替换为主观目测分。
3. **任务失败与结果真实性**：
   - 若任务队列超时（180秒）、失败（`failed`）或被取消（`cancelled`），如实向用户汇报错误状态。**严禁使用历史旧分或缓存假冒本次调用输出**。
4. **来源证据 (Provenance)**：
   - 客观展示 C2PA 凭据与 EXIF。无 EXIF 不代表 AI，文件平台标记与用户说明严格区分展示。

---

## 3. CLI 常用操作指引

在项目主目录下通过虚拟环境运行 `python -m app.cli`（与 Web UI 共用同一底层 API）：

### 3.1 服务检查与启动
```bash
# 检查服务健康状态
.venv/bin/python -m app.cli status

# 若服务未运行（端口 8765 未监听），在项目根目录启动（仅绑定 127.0.0.1）
./start.sh
```

### 3.2 素材列表与上传
```bash
# 列出所有已保存资产及其评分卡状态
.venv/bin/python -m app.cli list

# 上传图片（需要显式确认版权与成年状态）
.venv/bin/python -m app.cli upload /path/to/image.jpg --rights-confirmed --adult-confirmed
```

### 3.3 触发评分与分析
```bash
# 全量三模块分析（默认包含颜值、身材、摄影）
.venv/bin/python -m app.cli analyze <asset_id> --allow-cloud --adult-confirmed --wait

# 研发单测：仅测试某一模块
.venv/bin/python -m app.cli analyze <asset_id> --modules face --allow-cloud --adult-confirmed --wait

# 强制重评（忽略缓存）
.venv/bin/python -m app.cli analyze <asset_id> --allow-cloud --force --wait
```

### 3.4 导出、排行与删除
```bash
# 导出评分报告（支持 json 或 html）
.venv/bin/python -m app.cli export <asset_id> /path/to/report.json --format json
.venv/bin/python -m app.cli export <asset_id> /path/to/report.html --format html

# 查看项目内确定性排序（非模型成对复评，基于已有分项）
.venv/bin/python -m app.cli ranking <project_id> --module face

# 删除素材（须有用户显式授权，会彻底删除图片与报告）
.venv/bin/python -m app.cli delete <asset_id> --confirm
```

---

## 4. 详细技术参考

关于系统数据模型、FastAPI 端点契约、任务队列机制及备份方案，请查阅配套文档与模版：
- [architecture-and-api.md](references/architecture-and-api.md)
- [templates/report_export_template.json](templates/report_export_template.json)

---

## 5. 注意事项与合规说明

1. **隐私安全**：所有评估均在本地离线或用户配置的受信端点执行，绝不向未授权第三方同步人像图片。
2. **能力边界**：美学评分是模型偏好与测量规则的输出，仅用于辅助选片，不声称客观颜值、真实骨相或健康诊断。

