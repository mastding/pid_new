# pid_v2 会话交接摘要

生成时间：2026-05-13

## 1. 当前项目状态

项目目录：`D:\code\pid_v2`

当前项目是一个 PID 智能整定系统，已经从早期“上传数据 -> 单次辨识 -> 整定”的页面，逐步演进为偏工业风驾驶舱的前后端系统。重点方向包括：

- 基于历史数据导入的回路资产管理。
- 回路监控、单回路画像、趋势与频谱、数据质量等监控能力。
- 整定中心：整定先验、窗口候选、整定任务。
- 本体中心 / MCP 服务配置。
- 窗口候选流程引入本体/MCP 查询、策略生成、算法族运行、LLM 评审、准入结论。
- CodexSaver 已安装，用于低风险任务委托。

## 2. 前后端启动方式

后端：

```powershell
cd D:\code\pid_v2\backend
python -m uvicorn api.app:app --host 0.0.0.0 --port 4444 --reload
```

前端：

```powershell
cd D:\code\pid_v2\frontend
npm run dev
```

访问地址：

- 前端：`http://127.0.0.1:5173/monitoring`
- 后端 Swagger：`http://127.0.0.1:4444/docs`

最近一次确认：

- 前端监听：`127.0.0.1:5173`
- 后端监听：`0.0.0.0:4444`
- 两个 HTTP 检查均返回 `200`

## 3. CodexSaver 状态

CodexSaver 仓库：`D:\code\CodexSaver`

已完成：

- 安装为 editable Python 包：`codexsaver 0.2.0`
- 全局 MCP 配置写入：`C:\Users\dinglei\.codex\config.toml`
- MCP 启动器：`C:\Users\dinglei\.codexsaver\codexsaver_mcp.py`
- provider 配置：`C:\Users\dinglei\.codexsaver\config.json`
- provider：`deepseek`
- model：`deepseek-v4-flash`
- base_url：`https://api.deepseek.com`
- doctor 检查通过
- CLI 真实委托测试成功，`route=deepseek`，`status=success`

额外修复：

- `D:\code\CodexSaver\codexsaver\provider.py` 已兼容 OpenAI SDK 风格根地址。配置可以保留 `https://api.deepseek.com`，实际请求时自动补 `/chat/completions`。
- `D:\code\CodexSaver\codexsaver\config.py` 默认 DeepSeek 模型已调整为 `deepseek-v4-flash`。
- `D:\code\CodexSaver\tests\test_provider.py` 增加了根地址兼容测试。

使用策略已写入：

- 全局：`C:\Users\dinglei\.codex\AGENTS.md`
- 项目：`D:\code\pid_v2\AGENTS.md`

默认策略：

- 低风险文档、测试、代码搜索、代码解释、lint 修复、小型机械重构优先交给 CodexSaver。
- 架构、安全、数据库迁移、生产操作、最终审核仍由 Codex 处理。
- 任务不清楚时先 `dry_run`。

## 4. 重要前端页面和菜单方向

当前主入口：`/monitoring`

整体风格已经从“白色 AI 风”切换为偏工业控制平台风格：

- 深色背景。
- 紧密表格布局。
- 顶部工业状态栏。
- 左侧可折叠导航。
- 表格列宽支持拖动。
- 曲线图统一深色样式。

主要菜单规划：

- 工作台
- 回路监控
- 回路评估
- 根因诊断
- 整定中心
- 本体中心
- 系统设置

回路监控建议保留的子菜单：

- 全局回路看板
- 趋势与频谱
- 数据质量
- 单回路画像

整定中心子菜单调整方向：

- 整定先验
- 窗口候选
- 整定任务

本体中心：

- 知识图谱导入
- 图谱预览
- 可拖动动态图谱
- 后续可接图数据库或 MCP 查询

系统设置：

- MCP 服务配置
- LLM 模型配置

## 5. 历史数据与资产

已导入过的历史数据来自类似：

`D:\PID整定\5203回路数据\新建文件夹\*.xlsx`

典型回路包括：

- `5203_FIC_10103`
- `5203_FIC_20601`
- `5203_FIC_21601`
- `5203_FIC_22601`
- `5203_LIC_20502`
- `5203_TIC_10707`
- `5203_PIC_11401A`

资产层级方向：

- 石化工厂
- 运行部
- 装置
- 单元
- 回路

用户希望全局回路看板可以先基于装置资产目录选择节点，点击确认后只显示该节点下已导入的回路；没有导入回路的节点不显示或不展示数据。

## 6. LoopFeatures / 单回路画像

已实现并不断调整的核心方向：

单回路画像应直接复用 `LoopFeatures` 原始特征，不要混入旧的 `process_prior` 或窗口算法先验。

重要指标包括：

- `data_profile`
- `pv_stats`
- `mv_stats`
- `sp_stats`
- `constraint_raw`
- 数据点数、采样周期、时间范围
- PV/MV 范围、均值、标准差、跨度、跳变
- MV 活跃比例、平坦比例、反向频次、总行程
- 过程作用方向与置信度
- SNR
- 约束/饱和比例
- Harris 指数
- Cpk 过程能力指数
- 振荡指数与振荡周期

已经讨论过的原则：

- `identifiability_features` 不作为基础 LoopFeatures 原始指标，而应作为窗口识别/辨识 skill 的结果。
- 单回路画像支持选择时间范围，例如最近 8 小时、1 天、3 天。
- 选择时间范围后，所有画像指标必须基于该时间段重新计算。
- 窗口候选中的“数据画像”也应调用同一个 LoopFeatures 接口，确保和“单回路画像”一致。

## 7. 窗口候选流程

目标流程从上到下显示：

1. 数据画像
2. 本体检索
3. 策略生成
4. 算法族运行
5. LLM 评审
6. 准入结论

页面要求：

- 选择回路和时间范围后，不应立即展示候选窗口预览。
- 点击“开始本体驱动窗口评审”后，才串行执行全流程。
- 每个步骤显示状态：待执行、执行中、已完成、失败。
- 页面中按流程顺序展示内容。
- 本体返回原文、LLM 评审思维链/结果通过弹窗展示。
- 策略字段需要中文表格展示。
- 明确哪些策略字段被哪个算法族 provider 消费，哪些只是展示/审计。

窗口算法族包括：

- MV 阶跃
- MV 斜坡
- SP 阶跃
- 稳态扰动
- 滚动扫描

算法族运行结果需要展示：

- 每个算法族是否执行。
- 执行策略。
- 实际消费字段。
- 原因。
- 候选窗口列表。
- 每个候选窗口的质量分、判断、判断依据。

已发现并需注意的问题：

- LLM 评审结论里的“LLM 选中窗口”和候选窗口逐项评审里的优先窗口曾出现不一致，需要统一显示逻辑。
- “与算法分歧”含义：LLM 最终选择与确定性算法最高分窗口不一致，需要明确原因。
- 当前窗口候选有时还会显示旧的过程先验字段，需要清理 `process_prior` 相关读取。

## 8. 本体 / MCP 接入

目标：

在窗口候选或整定先验中，Agent 基于回路画像调用 MCP 本体工具查询：

- 干扰变量列表
- 最小阶跃幅值
- 最大噪声容忍度
- 预期时滞
- 预期时间常数
- 增益方向 / 正反作用
- 应优先或避免的窗口特征

已实现方向：

- 后端已有 MCP 配置相关代码：
  - `backend/core/mcp_config.py`
  - `backend/api/mcp_config_routes.py`
- 前端“系统设置 -> MCP 服务配置”已有页面，但样式曾有黑底黑字问题，已多轮优化。
- 用户配置了独立 MCP 服务提供本体知识问答和检索。
- 窗口候选中已尝试从 MCP 获取本体内容。
- 后端需要返回 `ontology_mcp_content_raw`，前端弹窗展示完整原文。

本体查询问题示例：

```text
查询 PID 回路 {loop_id} 的本体知识：变量角色、干扰变量、最小阶跃幅值、噪声容忍度、预期时滞/时间常数、增益方向、应优先或避开的窗口特征。
```

注意：

- 图谱不一定必须立即入库。第一阶段可以用 JSON / MCP 查询结果作为上下文。
- 后续如果需要多跳查询、关系过滤、跨装置推理、复杂问答，再考虑图数据库。

## 9. 整定任务与整定准入

整定任务页面方向：

- 页面顶部显示“发起整定任务”。
- 提供回路下拉框。
- 选择回路后显示当前回路、类型、候选窗口、指定窗口、当前准入状态。
- 下方显示整定准入校验。
- 再显示整定流程总览。

整定准入校验已经接过后端评估接口，显示：

- 数据质量
- 运行工况
- 约束/饱和
- 振荡状态
- 可辨识性

准入结论：

- 可整定
- 谨慎整定
- 不建议整定

用户要求：

- 不要显示多个重复提示框。
- 准入提醒文字要清晰，浅色提示框内文字改成黑色。
- 运行工况如果只是负荷变化/过渡工况，不应直接当成硬阻断，应作为软提醒。

## 10. 模型辨识与整定相关重点

已有讨论结论：

- `backend/core/pipeline/runner.py` 已升级为多轮辨识闭环：
  - 多轮辨识
  - LLM 评审
  - 精修建议
  - 再辨识
- `identification_refinement_advisor.py` 用于让 LLM 决定下一轮换窗口、缩模型池、提示 L 初值等。
- `identification_advisor.py` 现在只输出 `accept / downgrade`，不再 `reject` 直接中止。

模型拟合曲线：

- 前端希望能看到每个辨识模型拟合后的仿真曲线。
- 应放在“整定中心 -> 窗口候选”或“整定任务详情 / 全流程详情”中。
- 曲线包括 PV 实测、PV 仿真、MV。
- 需要清晰显示 X/Y 轴名称、tooltip 的 PV/MV 数值。
- 其他曲线图也应统一此显示方式。

已讨论过的模型问题：

- FOPDT/FO/SOPDT/IPDT 等模型。
- R2 与 NRMSE 计算来自实际 PV 与仿真 PV 的拟合误差。
- 如果 T 卡在下界或 zeta=0，应结合回路类型和物理先验判断可信度。
- 流量回路典型 T 较小；温度、液位、压力回路的 T 先验应不同。
- 优化器的约束应在拟合前体现，而不是拟合后才惩罚。

## 11. 趋势与频谱

已实现/规划：

- 支持选择回路。
- 支持选择时间范围。
- 趋势图显示 PV/MV，后续可叠加 SP、模式、报警、候选窗口标记。
- 用户要求 X 轴和 Y 轴可缩放，查看波动更直观。
- 建议增加“双 Y 轴 / 上下分图”开关：
  - PV 与 MV 量纲差异大时启用。
  - 上下两图各自坐标。

需要注意：

- 之前趋势与频谱一度可以选回路和时间范围，不要被后续重构覆盖掉。
- 当前可能只显示抽样点数，例如 `6465` 点，不一定是全量点。

## 12. 阀门 / 执行机构

用户希望确认是否有后端算法计算：

- 死区
- 回差
- 粘滞
- 卡涩

如果已有算法，需要在前端“阀门/执行机构”页面选择回路后展示，并提供公式说明。

如果没有，应明确标识为未接入后端，或先去掉未接菜单/功能块。

## 13. 重要代码位置

后端入口：

- `backend/main.py`
- `backend/api/app.py`

历史数据和回路画像：

- `backend/api/history_routes.py`
- `backend/core/history/store.py`
- `backend/core/shared/loop_features.py`

窗口候选与策略：

- `backend/core/pipeline/window_policy_advisor.py`
- `backend/core/pipeline/window_policy_scoring.py`
- `backend/core/pipeline/ontology_policy_builder.py`
- `backend/core/pipeline/ontology_mcp_context.py`
- `backend/core/providers/window/algorithm_families.py`
- `backend/core/providers/window/event_window_builder.py`
- `backend/core/providers/window/quality_score_selector.py`

辨识流程：

- `backend/core/pipeline/runner.py`
- `backend/core/pipeline/identification_advisor.py`
- `backend/core/pipeline/identification_refinement_advisor.py`
- `backend/core/algorithms/system_id.py`

评估：

- `backend/core/providers/evaluation/closed_loop_sim.py`
- `backend/core/algorithms/pid_evaluation.py`
- `backend/core/skills/assessment/assess_loop_assessment_skill.py`

MCP：

- `backend/core/mcp_config.py`
- `backend/api/mcp_config_routes.py`
- `backend/core/mcp_client.py`

前端：

- `frontend/src/pages/monitoring/LoopMonitoringPage.tsx`
- `frontend/src/pages/monitoring/LoopMonitoringPage.css`
- `frontend/src/pages/analysis/AnalysisPage.tsx`
- `frontend/src/services/api.ts`
- `frontend/src/types/tuning.ts`

说明文档：

- `docs/algorithm_formula_review.html`
- `docs/presentations/`

## 14. 当前 git 状态提醒

最近查看时，`pid_v2` 工作区有大量未提交修改和未跟踪文件，包括：

- `backend/api/app.py`
- `backend/api/history_routes.py`
- `backend/core/agent/consultant.py`
- `backend/core/history/store.py`
- `backend/core/pipeline/*`
- `backend/core/providers/evaluation/closed_loop_sim.py`
- `backend/core/skills/assessment/assess_loop_assessment_skill.py`
- `frontend/src/pages/analysis/AnalysisPage.tsx`
- `frontend/src/pages/monitoring/LoopMonitoringPage.tsx`
- `frontend/src/pages/monitoring/LoopMonitoringPage.css`
- `frontend/src/services/api.ts`
- `frontend/src/types/tuning.ts`
- `AGENTS.md`
- `backend/api/assistant_routes.py`
- `backend/api/prompt_config_routes.py`
- `backend/core/prompt_config.py`
- 多个截图和 PPT 图片
- `docs/algorithm_formula_review.html`
- `docs/presentations/`

不要随意 `git reset --hard` 或回退用户/历史修改。

## 15. 当前卡顿原因

当前长会话文件：

`C:\Users\dinglei\.codex\sessions\2026\04\21\rollout-2026-04-21T10-00-25-019dadc4-7242-7bf1-b81b-2fda4630a130.jsonl`

大小约：

- `783 MB`
- `28432` 行

结论：

- 每次输入新问题时，Codex 客户端卡顿主要是当前会话过大。
- CodexSaver MCP 启动约 `155 ms`，不是主要原因。
- 建议新开会话，并让新会话读取本摘要。

## 16. 新会话建议开场白

新会话中可以直接发：

```text
请先读取 D:\code\pid_v2\docs\session_handoff_2026-05-13.md，熟悉当前 pid_v2 项目上下文，然后继续工作。
```

如果要继续最近的问题，可以接着说：

```text
继续排查窗口候选页面：确保窗口候选里的数据画像完全复用 LoopFeatures 接口，并移除旧 process_prior 相关显示或读取。
```

或者：

```text
继续优化趋势与频谱页面：增加 X/Y 缩放、时间范围选择、双 Y 轴/上下分图开关，并确保点数说明清晰。
```

## 17. 建议下一步优先级

P1：

- 修正窗口候选页面中回路画像与单回路画像指标不一致的问题。
- 确认是否仍有旧 `process_prior` 模块被调用，能删则删，不能删则隔离并停止前端读取。
- 窗口候选全流程状态串行化：只有当前步骤执行中，后续保持待执行。
- 策略字段与 provider 消费字段落地到后端返回和前端中文表格。

P2：

- 趋势与频谱图增加缩放、双 Y 轴/上下分图。
- 单回路画像补 Harris、Cpk、振荡周期、SNR 公式说明。
- 阀门/执行机构页面确认后端算法并展示公式。

P3：

- 整定先验页面完善：监控/评估/诊断核心指标 + 本体查询原文 + LLM 先验解释。
- 本体中心图谱与 MCP 查询联动。
- 清理未接后端的菜单和功能块。
