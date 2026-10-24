# 颜选 · BeautyPick

[English](./README.md) | 中文

## 项目简介

颜选（BeautyPick）是本地人像评估与选片应用，提供Agent Skill调用入口。颜值采用本地模型整体评分，身材由本机千问结合候选参数评分，摄影采用 TOPIQ-IAA。分数反映模型偏好与启发式规则，不是客观颜值或经过验证的解剖学判断；正式评分无需Codex账号。

版本：**0.3.0**，详见[版本记录](./CHANGELOG.md)。

## 效果预览

![颜选 BeautyPick 人像评估与选片界面](./assets/preview.png)

## 核心功能

| 能力 | 它能帮你做什么 |
|---|---|
| 颜值整体评分 | 固定裁脸后由本地 FPEM 输出整体分，不生成无依据的部位分。 |
| 身材可见部分评分 | 本机千问结合 MediaPipe、FASHN 和候选参数评价肩胸、腰腹、臀胯、腿部；固定权重，缺项贡献为0。 |
| 摄影美感评分 | 本地 TOPIQ-IAA 输出整体分，不编造独立摄影分项。 |
| 来源证据检查 | 展示 C2PA、EXIF 和文件标记；缺少证据时不推断来源。 |
| 智能选片对照排行 | 在多张候选人像中基于已有客观技术分进行确定性多维横向排序，辅助快速挑片。 |

## 安装

### 配置要求

安装前需具备以下后台和本地运行条件。

| 项目 | 要求 |
|---|---|
| 完整后台 | Apple Silicon Mac、Python 3.13、支持指定MLX模型的Ollama |
| Windows客户端 | 通过OpenSSH隧道连接用户自有Mac后台；不在Windows原生运行MLX身材模型 |
| 模型 | FPEM、TOPIQ-IAA、FASHN、MediaPipe与指定千问版本，首次下载约4.3 GB，另需依赖空间 |
| Agent | 通过本机CLI/API调用，无需Codex账号；技能安装目录须显式指定 |

### 快速安装

向Agent发送安装请求，并明确只注册所需技能目录：

```text
请从 https://github.com/chemny/beauty-pick 安装BeautyPick，先检查运行条件，并只把技能注册到我指定的Agent目录。
```

已有技能目录不会覆盖，请先确认更新方式。详见[完整安装、Windows连接与故障处理](docs/runtime-install.md)。安装器不会自动同步其他Agent，也不会静默替换模型。

## 快速开始

向你的 Agent 发送日常对话指令开始评估人像照片：

```text
帮我用颜选评估这张人像照片，分析一下颜值、可见身材与摄影效果。
```

若使用本地可视化 Web 界面，直接运行 `./start.sh` 并访问 `http://127.0.0.1:8765` 即可。

## 使用示例

### 多图对比与挑片选优

```text
对比我相册里的这几张人像照片，使用 beauty-pick 按摄影构图和清晰度给出最佳推荐。
```

### 检查照片来源凭据与防伪

```text
检查这张照片的 C2PA 数字凭证和 EXIF 元数据，确认它是相机拍摄还是 AI 生成的。
```

### 获取拍摄与姿态改善建议

```text
使用 beauty-pick 分析这张照片的采光和头顶留白，给出改善面部受光与构图呼吸感的实操建议。
```

## 工作原理

颜选正式入口使用本地评分，Agent通过CLI/API调用应用：

1. **本地评分引擎**：颜值使用 FPEM，摄影使用 TOPIQ-IAA；身材使用本机千问与候选参数。分数表达模型及项目规则偏好，不是科学审美标准。
2. **身材参数与评分**：MediaPipe 与 FASHN 提供关键点、分割和候选轮廓，千问据可见信息评分。固定肩胸25%、腰腹30%、臀胯25%、腿部20%，缺失观察保持空值、总分贡献按0计算。
3. **服务与任务队列层**：基于 FastAPI 调度异步评估队列，并在 SQLite（`data/portrait.db`）中保留完整测评台账与快照。
4. **Agent 交互契约层**：标准 `SKILL.md` 规范结合 Pydantic 校验，确保输入输出精准受控。

## 目录结构

```text
beautypick/
├── SKILL.md                  # 根目录 Agent Skill 入口与操作规约
├── references/               # 系统架构参考与 REST API 端点契约
├── app/                      # FastAPI 后端服务、评分算法与 CLI
│   ├── scoring.py            # 维度权重、评分 Schema 与评分准则
│   ├── provider.py           # 评分引擎编排
│   ├── beauty_model.py       # 旧 ResNet50 回退适配器
│   ├── photo_model.py        # 旧 CV 摄影回退适配器
│   └── pose_model.py         # 姿态骨骼关键点提取与造型姿势解耦
├── web/                      # React 19 + TypeScript + Vite 前端工程
│   └── dist/                 # 预编译生产静态产物（免装 Node.js/npm）
├── models/                   # 本地离线模型文件与 CAI 信任锚
├── scripts/                  # 模型下载与校验工具脚本
├── assets/                   # 效果预览截图与公开示例素材
├── setup.sh                  # 显式安装与单目标 Skill 注册入口
├── start.sh                  # 服务启动脚本
└── requirements.txt          # Python 核心依赖定义
```

## 许可

本项目采用 [MIT 许可证](LICENSE)。

第三方实现与模型遵循各自条款，其中TOPIQ随附非商业许可。详见[第三方组件与声明](THIRD_PARTY.md)，不能以根MIT覆盖这些条件。

## 关于我

由 [@chemny](https://github.com/chemny) 维护。专为摄影师、创作者与开发者设计，旨在通过 AI Agent 提供本地人像评估与辅助选片工具。欢迎通过 GitHub Issues 交流反馈与贡献代码。
