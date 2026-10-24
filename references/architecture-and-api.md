# 应用接口与排查入口

## 职责

React/TypeScript界面与`app.cli`共用FastAPI服务，SQLite保存资产、报告、历史和任务，本地文件保存原图/裁图。Skill提供调用规范，不承载评分算法。

具体引擎以运行时为准；优先检查`/api/status`、报告engine及模块适配器，不照抄历史模型清单。常见排查入口：

| 范围 | 项目代码 |
| --- | --- |
| API、授权、会话 | app/main.py |
| CLI参数与任务等待 | app/cli.py |
| 任务、缓存、结果保护 | app/pipeline.py、app/engine.py |
| 持久化与显示版本校验 | app/storage.py |
| 检测、质量与标准裁脸 | app/images.py、app/face_preprocessing.py、app/standard_face.py |
| 模型适配与提示 | app/provider.py、app/*_model.py、app/body_prompts.py |
| 候选参数与测量 | app/body_measure_worker.py、app/body_parameters.py及其几何模块 |
| 排行与来源证据 | app/comparison.py、app/provenance.py |

来源签名、文件标记和用户说明分开；没有EXIF不能直接断定AI，签名有效不证明人物真实。

## API

先用`GET /api/session`建立会话Cookie并获取token。写请求需要`x-portrait-token`；不要把token写入任务记录。具体字段以本机路由为准。

| 端点 | 方法 | 用途 |
| --- | --- | --- |
| /api/status | GET | 引擎与应用配置 |
| /api/assets | GET / POST | 列表／multipart上传 |
| /api/assets/{id}/consent | POST | 当前授权与成年状态 |
| /api/assets/{id}/analyze | POST | modules、force等分析参数 |
| /api/jobs | GET | 任务状态 |
| /api/jobs/{id}/cancel | POST | 取消任务 |
| /api/assets/{id}/history | GET | 历史结果 |
| /api/assets/{id}/export.json | GET | 完整JSON报告 |
| /api/assets/{id}/export.html | GET | HTML报告 |
| /api/assets/{id}/vote | PUT | 投票 |
| /api/assets/{id} | DELETE | 仅限获准删除的素材 |

队列并发、超时、保留策略和模块默认值读取实际配置；不要把旧版本固定数值当作当前合同。任务completed还需核对请求模块是否成功，partial/failed不能靠现存旧报告掩盖。

## 验证与项目记录

通用检查见[测量、评分和验证](measurement-and-validation.md)。具体版本、实验、样例问题及验证结果读取应用项目的`docs/validation.md`和对应试验记录，不在技能内重复维护。后台日志、模型诊断和缓存详情留在技术记录，不擅自增加普通用户界面的内部话术。
