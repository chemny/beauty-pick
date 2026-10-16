# 用户界面文案清理记录

2026-10-14。统计口径：独立修改项；重复渲染不重复计数；整段设置页按一个修改项统计。

前端共 52 项，覆盖照片列表、详情及折叠区、分组、排行及折叠区、设置、上传弹窗。

1. 删除模型生成的冗长总评，缺分仅显示状态（`web/src/App.tsx`）
2. 删除档案元数据解释（`web/src/App.tsx`）
3. 移除本机私有工作空间术语（`web/src/App.tsx`）
4. 技术连接状态改为服务状态（`web/src/App.tsx`）
5. 删除列表口号（`web/src/App.tsx`）
6. 删除列表功能复述（`web/src/App.tsx`）
7. 精简筛选说明（`web/src/App.tsx`）
8. 删除空状态重复教程（`web/src/App.tsx`）
9. 删除详情重复标题（`web/src/App.tsx`）
10. 删除裁脸实现说明（`web/src/App.tsx`）
11. 替换标准化术语（`web/src/App.tsx`）
12. 替换图片替代文字术语（`web/src/App.tsx`）
13. 简化折叠区名称（`web/src/App.tsx`）
14. 简化折叠区名称（`web/src/App.tsx`）
15. 精简异常提示（`web/src/App.tsx`）
16. 精简授权文案（`web/src/App.tsx`）
17. 替换外发术语（`web/src/App.tsx`）
18. 精简范围选项（`web/src/App.tsx`）
19. 精简范围选项（`web/src/App.tsx`）
20. 删除用户点名的云端解释（`web/src/App.tsx`）
21. 删除调用计数术语（`web/src/App.tsx`）
22. 删除停机补清理实现细节（`web/src/App.tsx`）
23. 删除用户点名的主观参考提示（`web/src/App.tsx`）
24. 删除分组口号（`web/src/App.tsx`）
25. 删除分组内部边界说明（`web/src/App.tsx`）
26. 删除项目概念解释（`web/src/App.tsx`）
27. 删除推荐算法解释（`web/src/App.tsx`）
28. 精简来源填写说明（`web/src/App.tsx`）
29. 精简档案备注标题（`web/src/App.tsx`）
30. 删除上传重复说明（`web/src/App.tsx`）
31. 精简上传成年确认（`web/src/App.tsx`）
32. 精简上传授权（`web/src/App.tsx`）
33. 删除页脚口号（`web/src/App.tsx`）
34. 删除确认去除缓存术语（`web/src/App.tsx`）
35. 项目删除确认精简（`web/src/App.tsx`）
36. 设置页删除引擎/模型/推理配置/日志说明/实验说明/规则观察指令/评分锚点/版本号/重复限制，保留权重与保存期限（`web/src/App.tsx`）
37. 统一导航名称（`web/src/App.tsx`）
38. 移除详情原始JSON下载入口（`web/src/App.tsx`）
39. 导出按钮改为用户用语（`web/src/App.tsx`）
40. 删除排行内部算法标签（`web/src/Ranking.tsx`）
41. 删除排行重复解释（`web/src/Ranking.tsx`）
42. 删除折叠标题实现说明（`web/src/Ranking.tsx`）
43. 精简权重操作说明（`web/src/Ranking.tsx`）
44. 删除算法及统计术语说明（`web/src/Ranking.tsx`）
45. 删除默认位置内部比较信息（`web/src/Ranking.tsx`）
46. 精简无结果提示（`web/src/Ranking.tsx`）
47. 精简对照选项（`web/src/Ranking.tsx`）
48. 删除对照重复长评语（`web/src/Ranking.tsx`）
49. 删除对照限制段落（`web/src/Ranking.tsx`）
50. 删除折叠两两对比长评语（`web/src/Ranking.tsx`）
51. 统一排行导出标签（`web/src/Ranking.tsx`）
52. 移除排行原始JSON下载入口（`web/src/Ranking.tsx`）

保留：有效评分与分项、摄影优缺点、档案参数、操作名称、未知/缺分状态、必要授权、删除确认、保存期限、权重合计校验。完整评估数据仍留在后台，不修改评分结果与授权机制。

53. 图片HTML报告：移除总评长文、分项长依据、判断限制、原始来源JSON、主观提示、规则版本和导出管理说明；保留档案、分数及摄影优缺点。
54. 排行HTML报告：移除算法说明、统计解释、完整内部JSON；保留排名与未入榜原因。

合计55项（按上述修改组计数）。JSON接口与后台原始记录保留，页面不提供原始JSON入口。

55. 分组内版本选择去掉“用户声明/未声明版本关系”术语，统一颜值名称；用户填写的来源备注原文保留。

验证：前端构建、lint通过；浏览器检查列表、详情展开区、设置、上传、排行与权重折叠区；两类HTML导出均返回200且无原始JSON块。未新增模型调用，8张资产保留。
