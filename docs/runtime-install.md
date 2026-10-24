# 安装与本地运行

## 运行条件与模型

完整三模块后台使用 **Apple Silicon Mac、Python 3.13 和支持 MLX 的 Ollama**。FPEM、TOPIQ-IAA、MediaPipe 与 FASHN 在应用环境运行；千问通过本机11434端口运行。首次下载需要联网；评分时不需要Codex或云端账号。Windows可通过系统OpenSSH连接用户自有Mac后台，当前不提供在Windows原生运行MLX身材模型的承诺。

| 用途 | 文件或服务 | 约体积 |
| --- | --- | --- |
| 颜值 | 完整 FPEM 检查点 | 583 MB |
| 摄影 | 完整 TOPIQ-IAA 检查点 | 294 MB |
| 分割 | FASHN 检查点与配置 | 256 MB |
| 检测、关键点 | MediaPipe固定版本模型 | 45 MB |
| 身材评分 | `qwen3.5:2b-mlx`，摘要 `9dd89c688f18f01cb3636d01f9512d8daae2a531b3e726f59d68cf7b91c8358e` | 3.12 GB |

另需Python依赖和安装临时空间，请预留至少12 GB磁盘空间；实际内存需求随图片和并发变化。完整文件列表、大小、下载地址和SHA-256见 `models/manifest.json`，不会用体积更小但不同的模型替代。

千问精确版本：[Ollama官方模型页](https://ollama.com/library/qwen3.5:2b-mlx)。标签更新导致摘要不同会停止安装，不会静默更新评分引擎。

## Mac 完整安装

1. 安装 [Python 3.13](https://www.python.org/downloads/) 和 [Ollama](https://ollama.com/download)，启动Ollama。需要可运行指定MLX模型的版本；运行时按实际模型摘要和视觉能力复检，不仅判断命令存在。
2. 在仓库目录运行：

```bash
python3.13 setup.py --pull-body
```

安装器仅在项目内创建 `.venv`、`models/body-runtime/deps` 和下载模型文件，不写任何Agent目录。主环境与分割依赖分别锁定；所有模型下载后检查完整大小和SHA-256。已有但不匹配的文件不覆盖，请先确认其来源、移开留存后再重试。网络中断可重新运行，已通过校验的模型不重复下载。

3. 若要注册Skill，指定一个尚不存在的目录，不能覆盖已有技能：

```bash
python3.13 setup.py --client-only --skill-dir "$HOME/.agents/skills/beauty-pick"
```

其他Agent使用其明确指定的目录；不会自动发现、复制到多个客户端。已有技能需要先由用户决定如何保存/更新，安装器不会替用户删除。

4. 运行只读检查并启动：

```bash
.venv/bin/python scripts/doctor.py
./start.sh
```

打开 `http://127.0.0.1:8765`。未备齐模型、依赖或千问服务时，启动检查会指出问题，不显示“安装完成”。后台仅监听本机。

检查与CLI：

```bash
python3.13 setup.py --check
.venv/bin/python -m app.cli status
.venv/bin/python -m app.cli list
```

上传与重评使用网页，或按SKILL.md中的CLI调用。上传后核对三个分值；技术失败不冒充缺项0。身材缺失观察按现行固定权重贡献0，与模型运行失败不同。

## Windows 客户端连接自有 Mac 后台

Windows不安装MLX或替换千问模型。在Mac完成上述安装并启动服务；如用户同意远程连接，在Mac系统设置中开启“远程登录”，只授权所需账号。该系统设置由用户决定，安装器不修改。

Windows PowerShell使用系统OpenSSH建立隧道，将下例 `user@mac-host` 换成用户自己的Mac账号和地址：

```powershell
ssh -N -L 8765:127.0.0.1:8765 user@mac-host
```

保持窗口运行，Windows浏览器打开 `http://127.0.0.1:8765`。照片在用户Mac上处理；首次传图前向用户说明这一处理位置。不要为连接而将后台改为监听公网，也不要关闭SSH主机校验。

仅注册Skill可使用Windows已安装的Python：

```powershell
py -3 setup.py --client-only --skill-dir "$env:USERPROFILE\.agents\skills\beauty-pick"
```

Windows Skill通过隧道后的HTTP API使用服务；不执行 `.venv/bin/python` 或 `start.sh`。API会话与写入合同见 `references/architecture-and-api.md`。SSH不存在时安装Windows“OpenSSH客户端”可选功能；没有自有Mac后台时，当前完整身材引擎不能在该Windows机器独立运行。

## 分发和许可边界

应用自有代码采用根LICENSE；第三方实现、模型各自遵循随附许可，不能将整个依赖链概括成MIT。来源与声明见 [第三方组件](../THIRD_PARTY.md)。本次新增的FPEM、TOPIQ、FASHN和Heavy权重不进入Git，由安装器从上游固定地址获取；旧版已跟踪的回退素材不在此次自动清理。`data/`中的原图、数据库、历史和试验不随发布包分发。
