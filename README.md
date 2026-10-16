# 颜选 · BeautyPick

> **科学客观、多维连续的人像美学评估与智能选片系统**  
> 结合计算机视觉、深度学习骨相基准与光学物理度量，为 **颜值**、**身材** 与 **摄影** 提供 0.1 连续精细评分与真实依据。支持 Web 界面与各类主流 AI Agent（Google Antigravity、OpenAI Codex、Claude Code）开箱即用无缝联动。

---

## ✨ 核心特性

- **三大模块独立与解耦评估**：
  - **颜值 (Face)**：ResNet50 骨相基准分、三庭五眼几何比例、瞳孔眼神光斑、微观肌理真实毛孔分析（过度磨皮扣分）。
  - **身材 (Body)**：采用 `body-appearance-3.0` 规范。**模特拍摄造型姿势（如高低肩、侧倾）与身材形体完全解耦**，绝不作为身材缺陷扣分；未入镜部位自适应动态重权归一。
  - **摄影 (Photography)**：**0.1 连续精细技术评分**。基于 Tenengrad 边缘能量梯度、面部专属点测光、黄金三分线偏离度与主体对背景景深虚化分离比（`dof_separation`），彻底拉开各照片技术梯度。
- **纯本地硬件加速度量引擎**：
  - 基于 Apple Silicon MPS / MediaPipe / OpenCV，毫秒级快速推算，支持纯本地离线运行（0 Token 消耗）。
  - 双轨设计：支持无缝对接已登录的 Codex CLI 或兼容视觉模型。
- **数字凭证与防伪核验 (C2PA / EXIF)**：
  - 自动读取 Content Credentials 数字签名与链条追溯，区分 AI 生成、相机直出与二次编辑。
- **开箱即用与 Agent Skill 深度集成**：
  - 附带标准化 Agent Skill，任何支持技能调用的 AI Agent 可直接调用底层 API 进行人像诊断与选片。

---

## 🚀 快速上手 (开箱即用)

### 1. 克隆并一键安装

在终端运行一键安装脚本（自动检查 Python 3.10+、配置虚拟环境、校验离线模型并自动向宿主机注册 Agent 技能）：

```bash
git clone https://github.com/your-username/beautypick.git
cd beautypick
./setup.sh
```

### 2. 启动服务

```bash
./start.sh
```

打开浏览器访问 **[http://127.0.0.1:8765](http://127.0.0.1:8765)** 即可开始选片与人像评估！

---

## 🤖 在 AI Agent 中使用 (Skill 插件)

`./setup.sh` 会自动检测并向你本机已安装的 AI Agent 环境注册 `beauty-pick` 技能：
- **Google Antigravity**: `~/.gemini/config/skills/beauty-pick`
- **OpenAI Codex**: `~/.agents/skills/beauty-pick` / `~/.codex/skills/beauty-pick`
- **Claude Code**: `~/.claude/skills/beauty-pick`

在 Agent 会话中，你可以直接输入：
> *“帮我用颜选评估一下刚刚拍摄的这几张人像照，哪张摄影构图更好？”*  
> *“用 beauty-pick 分析这张照片的颜值骨相基准和身材比例。”*

---

## 💻 命令行 (CLI) 常用指令

通过虚拟环境可直接调用与 Web 相同底层的 CLI 工具：

```bash
# 检查后端服务状态与模型就绪情况
.venv/bin/python -m app.cli status

# 查看已入库照片及评分台账
.venv/bin/python -m app.cli list

# 上传新照片并确认成年授权
.venv/bin/python -m app.cli upload /path/to/photo.jpg --rights-confirmed --adult-confirmed

# 触发三模块全量分析
.venv/bin/python -m app.cli analyze <asset_id> --allow-cloud --wait

# 导出独立 HTML 报告
.venv/bin/python -m app.cli export <asset_id> ./report.html --format html
```

---

## 📁 目录架构说明

```text
beautypick/
├── app/                  # FastAPI 后端服务、度量算法与 CLI
├── web/                  # React 19 + TypeScript + Vite 前端源码
│   └── dist/             # 预编译静态页面（用户免配 Node.js/npm）
├── skills/beauty-pick/   # 跨 Agent 标准 Skill 定义与技术契约
├── models/               # 本地离线视觉模型与 CAI 信任锚
├── scripts/              # 模型下载校验与工具脚本
├── setup.sh              # 一键环境安装与 Skill 注册脚本
├── start.sh              # 一键服务启动脚本
└── requirements.txt      # Python 核心依赖清单
```

---

## ⚖️ 免责与隐私声明

- 评分体系依据指定的美学、人体工学与摄影光学技术规范设计，旨在为拍摄选片与摄影提升提供量化参考，并非医学、真实相貌或社会性审美判定。
- 图像数据默认仅存储于本机 SQLite 数据库及本地资产目录中，未经用户显式确认不会向任何外部模型接口发送数据。

---

## 📄 License

MIT License
