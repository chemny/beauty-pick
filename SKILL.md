---
name: beauty-pick
metadata:
  version: "0.3.0"
description: 当用户需要使用本机“颜选 · BeautyPick”应用检查人像来源证据、评估颜值身材摄影、选片及排查评分流程。通过应用CLI/API操作，保护原图、历史结果与未修改模块。
---

# 颜选 · BeautyPick

## 定位与事实来源

Skill是Agent调用应用的入口，不自行打分、不维护另一套评分算法。任何Agent均可通过本机CLI/API使用；不能因为开发助手是Codex，就把生产评分改成Codex。

先从`BEAUTYPICK_HOME`、当前工作区或已确认的项目路径定位应用。安装记录可使用`{{BEAUTYPICK_DIR}}`占位符；未替换时它不是有效路径。路径变化先核实，不凭旧路径复制项目。

模型、提示、权重、缺项处理及版本以`/api/status`、实际报告引擎身份和项目代码为准，不能从旧文档推断。主要入口为`app/engine.py`、`provider.py`、`pipeline.py`、`scoring.py`及各模型适配器。不要把通用视觉特征称为经验证的骨相、真实围度、皮肤健康或统一科学审美标准。

## 安装与依赖

完整后台需要Apple Silicon Mac、Python 3.13及指定Ollama MLX模型，具体见项目的 `docs/runtime-install.md`。先检查 `scripts/doctor.py`，缺少依赖时说明用途与安装操作，获得安装授权后才执行 `python3.13 setup.py --pull-body`，完成后复检。缺少或变更的权重不能静默替换。

注册Skill必须显式指定一个新目录：`python3.13 setup.py --client-only --skill-dir <target>`，不覆盖已有目录，不同步其他Agent。Windows客户端使用OpenSSH连接用户自有Mac后台后的HTTP API；照片在该Mac处理，传图前需说明并获准，不在Windows执行下方POSIX虚拟环境命令。

## 执行边界

- 讨论、诊断和复盘先只读；用户明确实施后再修改。
- 区分隔离试验、既有结果迁移、正式上传/重评接入。获准更新预览不等于获准写正式库；迁移既有分数不等于已切换运行时。
- 用户冻结某模块时，完整保护其报告及算法。单模块排查使用明确的modules，不触发其他模块重算。
- 不自动删除素材、覆盖历史、同步其他Agent的技能副本或推送远程。
- 数据保留、授权和成年状态按实际应用及已获用户授权执行，不在界面擅自加回用户移除的交互。

## 测量与评分的通用要求

操作前阅读 [测量、评分和验证规则](references/measurement-and-validation.md)。核心约束：

1. 分开保存候选提取、测量/估计、可见性判断、审美评分和文字解释。
2. 尽可能保留候选参数及不确定性；不要仅因质量阈值不满足就提前丢弃整个部位。无信息时也不能伪造数据。
3. 裁图必须包含对应参数点，坐标、图像方向及模型输入必须能追溯。
4. 明确区分参数辅助、纯视觉和缺项默认贡献；出分率不等于测量覆盖率或准确率。
5. 不复制本轮具体阈值、权重、样例排名到技能作为永久标准；读取当前应用规则。
6. 模型失败必须是真失败；历史结果可受保护继续展示，但不能冒充本次成功输出。
7. 定位/识别模型不自动具备审美能力；参数定义和姿态修正分开验证，参考排序一致性不当作准确率。未获准替换时保留正式引擎。

## CLI使用

在已核实项目根目录运行，先检查服务与当前命令合同：

```bash
.venv/bin/python -m app.cli --help
.venv/bin/python -m app.cli status
.venv/bin/python -m app.cli list
```

服务未启动时按项目启动脚本或运行说明启动，仅绑定本机。上传及分析示例：

```bash
.venv/bin/python -m app.cli upload /path/to/image.jpg --rights-confirmed --adult-confirmed
.venv/bin/python -m app.cli analyze <asset_id> --allow-cloud --adult-confirmed --wait
.venv/bin/python -m app.cli analyze <asset_id> --modules face --allow-cloud --adult-confirmed --force --wait
.venv/bin/python -m app.cli export <asset_id> /path/to/report.json --format json
.venv/bin/python -m app.cli ranking <project_id> --module face
```

当前CLI保留`--allow-cloud`历史参数，不能据此声称本次分析发送到云端；实际处理位置必须核实运行引擎。若未来确需外发，仍需对应用户授权。删除仅限明确授权的对象，不把命令可用视为删除授权。

## 完成标准与汇报

- 核对实际上传和强制重评，不只验证直接模型函数或保存试验结果。
- 核对任务、数据库原始报告、API返回和页面展示是否一致；保存历史与缓存版本证据。
- 批量变更前备份，写入前检查并发变化，事务更新，验证原图与冻结模块不变；用户新上传/新任务不可误删或覆盖。
- 分别报告参数覆盖、实际出分、默认贡献、失败和质量待核对项；说明哪些已正式接入。
- 当前图片、故障、数值和临时版本留在项目记录，不写进通用技能或长期记忆。

架构与接口见 [应用接口参考](references/architecture-and-api.md)。
