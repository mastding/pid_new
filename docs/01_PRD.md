# PRD：PID 智能诊断与整定系统 V3.0

> 文档类型：产品需求文档（PRD）
> 状态：**Draft V1.0（待评审）**
> 版本：V3.0
> 主要读者：项目经理、产品、研发、测试、工艺工程师代表
> 关联文档：02 总体架构 / 03 领域模型 / 04 API / 05 算法与流水线 / 06 LLM 与 Skill / 07 前端 / 08 部署运维 / 09 本体契约 / 10 诊断决策树
> 配套代码基线：`pid_v2` 仓库 `v3` 长期分支（首次发布前不合并到 main）

---

## 0. 文档信息

### 0.1 修订记录

| 版本 | 日期 | 作者 | 变更 |
|---|---|---|---|
| V1.0 | 2026-05-08 | (待填) | 初稿 |

### 0.2 评审记录

| 评审人 | 角色 | 状态 | 意见 |
|---|---|---|---|
| (待填) | 产品 | 待评审 | — |
| (待填) | 控制工程师 | 待评审 | — |
| (待填) | 工艺工程师 | 待评审 | — |
| (待填) | 研发负责人 | 待评审 | — |

### 0.3 术语表

| 术语 | 含义 |
|---|---|
| PV | Process Variable，被控变量 |
| MV | Manipulated Variable，操作变量 |
| SP / SV | Set Point / Set Value，给定值 |
| DV | Disturbance Variable，扰动变量（用于 MISO 辨识） |
| LPM | Loop Performance Monitoring，回路性能监控 |
| SISO / MISO | 单输入单输出 / 多输入单输出 |
| Cpk | 过程能力指数 |
| Harris 指数 | 基于最小方差基准的闭环控制性能指数 η ∈ [0,1] |
| 工况（Condition） | 同一回路在不同生产负荷/操作模式下的子状态，阈值与先验可不同 |
| 本体（Ontology） | 工艺知识库，提供 USL/LSL、量程、耦合关系、DV 列表等元数据 |
| MCP | Model Context Protocol，本系统访问外部本体的协议 |
| Skill | LLM 可调用的原子能力 |
| Provider | 同一类能力的具体算法实现，由 Skill 装配 |
| Advisor | 嵌在流水线决策点的单次 LLM 顾问 |
| Consultant | 用户聊天框对应的工具调用 Agent |
| 诊断流水线 | 决定回路是否需要整定的状态机 |
| 整定流水线 | 现有的 5 阶段确定性流水线（数据→窗口→辨识→整定→评估） |

### 0.4 默认决策（本稿冻结待评审）

> 这些决策影响后续所有设计文档，请优先评审。每条都给出"修改后的影响范围"。

| 编号 | 决策项 | 默认值 | 修改影响 |
|---|---|---|---|
| **D-1** | V3.0 必做的诊断阶段 | 工况识别 / 开闭环 / SNR / Cpk / Harris(闭环)；阀门死区粘滞、耦合识别、Harris(开环 SP 阶跃)、MISO 延后 V3.1 | 文档 5、10 章节数 |
| **D-2** | 工程师确认机制 | 异步：诊断流水线产生 `EngineerConfirmation` 事件后挂起，工程师处理后通过 `/api/confirmations/{id}/respond` 触发重入 | 文档 2、3、4、10 |
| **D-3** | 诊断与整定的关系 | 两条独立流水线：诊断输出"是否进入整定 + 推荐入口窗口"，整定独立运行，参数通过 SessionContext 传递 | 文档 4、5、10 |
| **D-4** | 本体回写 | V3.0 不做。K/T/L 沉淀写到 `var/ontology_writeback/<loop_id>/<timestamp>.json`；V3.1 接通 MCP 写回 | 文档 9，回写相关 FR 状态 |
| **D-5** | 诊断阶段是否接 LLM 顾问 | V3.0 不接。诊断全部走确定性算法 + 阈值；保留现有 4 个整定 LLM 决策点；V3.1 给诊断加 LLM | 文档 6 |
| **D-6** | 算法资产复用边界 | `core/algorithms/*` 与 `core/policies/*` 全保留（仅注释中文化）；其余编排/契约/前端按本文档体系全部重写 | 文档 2、3、4、5、6、7 实施分工 |
| **D-7** | 仓库与分支策略 | 在 `pid_v2` 仓库开 `v3` 长期分支演进；首次发布前不合并 main；发布后旧 main 进入 `legacy/v2` 分支冷冻 | 文档 8 部署、CI |

如对任一项不同意，请在评审里把"默认值"列直接改写。

---

## 1. 产品定位

### 1.1 背景与机会

当前 `pid_v2` 是一个"算法工具"：给定一段 CSV，跑出 PID 参数。在工业现场实际使用时暴露了两个根本问题：

1. **不是所有回路都需要整定**。先于整定的诊断（"这条回路是不是阀门坏了 / 工况切换了 / 控制器在开环 / 有耦合共振"）几乎全靠人工，导致整定动作经常用错地方；
2. **整定的输入条件依赖工艺知识**。USL/LSL、量程、最大可接受 SP 阶跃幅度、DV 扰动变量、耦合回路位号 —— 这些信息都散在工艺工程师脑里和岗位操作法文档里，没有结构化注入。

V3.0 的目标是把系统从"算法工具"升级为"**LPM 诊断 + 整定 + 经验沉淀**"的闭环：

```
回路数据 → 工况识别 → 非参数排查 → 性能评估 → ┬→ 健康（不整定）
                                              ├→ 告警（不整定，待人工处置）
                                              └→ 进入整定 → 输出参数 + 经验沉淀
```

### 1.2 目标用户

| 角色 | 关注点 | 在系统中的主要动作 |
|---|---|---|
| 工艺工程师 | 工艺安全 / 阶跃幅度 / DV 选择 | 处理工程师确认事件、维护本体、复核告警 |
| 控制工程师 | PID 参数质量 / 整定策略 | 启动整定、评审顾问意见、沉淀经验 |
| 运维 / 班组长 | 回路总体健康 / 告警闭环 | 看诊断仪表盘、关闭低优先级告警、上报现场异常 |
| 算法工程师（项目内部） | 算法可替换 / 可观测 | 替换 Provider、调阈值、做离线评测 |

### 1.3 核心价值主张

- **不当整定的回路绝不动**（诊断先于整定）；
- **每个数值都可追溯**（工程师能看到每个 KPI 的数据证据 + 本体证据）；
- **算法可替换、决策可重放**（Skill / Provider 注册机制 + 会话日志）；
- **LLM 是可关闭的增量**（核心流程纯算法，关 LLM 不影响整定可用性）。

### 1.4 与现状的差异

| 维度 | pid_v2 现状 | V3.0 目标 |
|---|---|---|
| 入口 | 上传 CSV → 直接整定 | 诊断 → 决定是否整定 |
| 多回路 | 单回路 | 仍是单回路（V3.0），多回路 V3.x |
| 工况 | loop_type 单标签 | 多工况 + 本体绑定 |
| 告警 | 无 | 一等公民实体 |
| 工程师协同 | 仅前端 override | 异步确认事件机制 |
| 本体接入 | 仅可选只读 | 强依赖只读，写回 V3.1 |
| 辨识 | SISO | SISO（V3.0），MISO V3.1 |
| 性能基准 | 仅闭环仿真打分 | Cpk + Harris + 闭环仿真三视角 |

---

## 2. 范围与边界

### 2.1 V3.0 必做（In-Scope）

**诊断侧（按 D-1）**

- 工况识别（基于本体 + 数据画像）
- 开/闭环判断
- PV 噪声 SNR 评估与滤波档位建议
- Cpk 计算
- Harris 指数（闭环估计版）

**整定侧（基本沿用 pid_v2，迁到 v3 框架）**

- 数据加载、清洗、画像
- 窗口检测（5 个算法族 + 本体策略）
- 窗口选择（含 LLM 顾问）
- 系统辨识 + 评审 + 精修（含 LLM）
- PID 整定（IMC / Lambda / ZN / CHR）
- 闭环仿真评估 + 现实性检查

**工程师协同**

- 异步工程师确认事件（产生、查询、回应、重入）
- 告警事件管理（产生、列表、确认、关闭）

**知识与经验**

- 本体只读接入（MCP，多 server 配置）
- 本地经验沉淀（var/experience/）
- 经验在整定中被检索 + 在前端可浏览

**系统侧**

- 会话与可重放（var/sessions/jsonl 日志）
- LLM 模型配置中心（热切换）
- SSE 事件流（统一协议）
- 配置 / 可观测性 / 部署文档

### 2.2 V3.1+ 延后（Out-of-Scope）

| 编号 | 项 | 原因 |
|---|---|---|
| **D7-V3.1** | 阀门死区/粘滞检测 | 算法成熟度需要更多现场样本 |
| **D8-V3.1** | 耦合回路识别 | 强依赖本体覆盖完整度 |
| **D9-V3.1** | Harris 开环 SP 阶跃测试 | 需要先做阶跃测试向导 + 安全门 |
| **D10-V3.1** | MISO 辨识 | DV 列表需要本体覆盖 |
| **K3-V3.1** | 本体回写（K/T/L） | MCP 写回通道需要权限设计 |
| **多回路并行** | 单回路改多回路 | 框架支持但 V3.0 不暴露 UI |
| **告警长期统计仪表盘** | KPI 趋势面板 | 数据沉淀需要先跑 1-2 个月 |
| **诊断阶段 LLM 顾问** | 见 D-5 | V3.0 先验证确定性版本 |

### 2.3 永不做（Never）

- 直接对真实控制器下发 PID 参数（系统永远只输出建议，由工程师手动写组态）；
- 跳过工程师确认直接做阶跃测试（任何对生产装置的写动作都必须有人工 checkpoint）；
- 在缺失关键本体数据时使用"猜测值"继续整定（必须告警 + 阻断）。

---

## 3. 用户场景

> 共 5 个端到端场景，每个对应一类典型流程。

### 3.1 场景 A：日常巡检发现性能下降

**角色**：班组长 → 控制工程师

```
1. 班组长每日打开诊断仪表盘，看到 TIC-10707 卡片标红："Cpk=0.71 < 1.0"
2. 点入回路详情，看到诊断决策树：工况识别✓ / 开闭环✓ / SNR✓ / Cpk✗ / Harris=0.42
3. 决策树推荐"进入整定"
4. 控制工程师点击"启动整定"，进入整定流水线（沿用现有 5 阶段）
5. 整定结束，参数 + 评估报告回到该回路卡片，状态由"待整定"变为"待人工写组态"
```

### 3.2 场景 B：工程师确认事件回流

**角色**：控制工程师 → 工艺工程师

```
1. 整定流水线在窗口选择阶段产生告警："候选窗口本体未提供量程，无法计算 mv_span 阈值"
2. 系统生成 EngineerConfirmation 事件："请确认 TIC-10707 PV 量程"
3. 工艺工程师在工程师确认面板看到，填入量程 [50, 200] °C，提交
4. 系统自动重入流水线（resume），用新量程重新计算
5. 整定完成，事件状态变为 resolved
```

### 3.3 场景 C：诊断给出"无需整定"

**角色**：班组长

```
1. 班组长看到 LIC-2003 卡片绿："健康"
2. 点入详情：诊断决策树全绿（Cpk=1.32，Harris=0.78）
3. 系统不允许从该回路启动整定（按钮置灰，提示"该回路当前性能良好，不建议整定"）
4. 工程师如确实想强制整定，需在前端勾选"忽略诊断结论"并填写理由，事件入审计日志
```

### 3.4 场景 D：开环回路告警

**角色**：班组长 → 现场仪表工

```
1. FIC-105 卡片红："开环 - 控制器输出近 30 分钟未变化"
2. 系统产生 AlarmEvent("open_loop")，附带 MV 时序证据
3. 班组长一键派发到现场仪表工，告警状态 new → acknowledged
4. 仪表工现场处理后回填处置说明，告警关闭 resolved
5. 系统下次诊断该回路时如果开环消失，告警自动归档
```

### 3.5 场景 E：LLM 顾问被关闭仍可整定

**角色**：DCS 现场（无外网，无法调 LLM）

```
1. 工程师在配置页关闭"LLM 顾问总开关"
2. 启动整定，所有 4 个 LLM 决策点全部走确定性回退
3. 流水线正常输出参数（精度可能略低，质量分数会略降）
4. 报告里清晰标注"LLM 顾问已禁用，本次整定全确定性"
```

---

## 4. 功能需求（FR）

> 每条 FR 含：编号 / 名称 / 用户价值 / 描述 / 验收标准（GWT）。
> GWT = Given（前置）/ When（动作）/ Then（结果）。

### 4.1 诊断侧

#### FR-D1：工况识别

**用户价值**：让系统理解回路当前处于哪种生产状态，避免用错阈值。

**描述**：
- 输入：回路数据 + 本体的工况列表
- 算法：基于数据画像（PV 均值、MV 均值、负荷指标）做投票，匹配本体定义的工况；
- 失败回退：本体未定义工况 → 兜底单工况"default"。

**GWT**：
```
Given 本体已定义工况 [低负荷, 高负荷]，每个工况有 PV/MV 区间
When  上传一段 CSV，PV 均值落在低负荷区间
Then  诊断结果中 condition.id = "低负荷"，confidence ≥ 0.7

Given 本体未定义工况
When  上传 CSV
Then  condition.id = "default"，risk_flags 包含 "no_condition_in_ontology"
```

#### FR-D2：开/闭环判断

**用户价值**：避免对开环回路浪费整定动作。

**描述**：
- 算法：滚动窗口计算 ΔMV 占比；持续 60s 以上 ΔMV ≈ 0（< 0.1% 量程）→ 判开环；
- 产出：诊断节点结果 + 若开环则告警事件 `open_loop`。

**GWT**：
```
Given MV 在 5 分钟数据中的最大变化幅度 < 0.05% 量程
When  跑诊断流水线
Then  产生 AlarmEvent("open_loop")，pipeline_status = "blocked_open_loop"，不进入整定
```

#### FR-D3：PV 噪声 SNR 评估

**用户价值**：识别采样噪声并建议滤波档位。

**描述**：
- 算法：滚动窗口 var(PV) 与高频 PSD 比；
- 产出：SNR 数值 + 滤波档位建议（low/medium/high，对应一阶滤波 τ）；
- 注入下游：建议的 τ 注入窗口检测的 `signal_processing` 阶段。

**GWT**：
```
Given PV 高频功率占比 > 30%
When  跑诊断
Then  snr_metric.level = "low"，filter_recommendation.tau ≥ 3 × dt
      诊断报告标注"建议加滤波后再做整定"
```

#### FR-D4：Cpk 计算

**用户价值**：基于本体 USL/LSL 衡量过程能力，决定是否需要整定。

**描述**：
- 输入：本体提供的 USL、LSL、PV 数据；
- 算法：Cpk = min((USL-μ)/3σ, (μ-LSL)/3σ)；
- 阈值：Cpk < 1.0 → 推荐整定；本体缺 USL/LSL → 产生 EngineerConfirmation 事件。

**GWT**：
```
Given 本体提供 USL=200, LSL=50；PV 数据均值=180，σ=15
When  跑 Cpk
Then  Cpk ≈ 0.44，diagnostic_decision.recommend_tuning = true

Given 本体未提供 USL/LSL
When  跑 Cpk
Then  产生 EngineerConfirmation("missing_pv_limits")，诊断流水线挂起
```

#### FR-D5：Harris 指数（闭环估计版）

**用户价值**：用最小方差基准衡量当前控制器表现。

**描述**：
- 算法：闭环数据估计 deadtime → 拟合 AR 模型预测 PV → η = 1 − var(残差) / var(PV)；
- 阈值：η < 0.6 → 推荐整定；置信度 < 0.5 → 标"建议做 SP 阶跃测试"（V3.1 实现实际向导）。

**GWT**：
```
Given 闭环数据，deadtime 估计置信度 ≥ 0.5
When  跑 Harris
Then  返回 η ∈ [0,1] 与 confidence；η < 0.6 → diagnostic_decision.recommend_tuning = true
```

#### FR-D6：诊断结论汇总

**用户价值**：把多个诊断节点的结论合成一个明确的"下一步动作"。

**描述**：
- 输入：所有诊断节点的输出；
- 输出：`DiagnosticVerdict { status, recommend_tuning, blockers, alarms, confirmations, evidence }`；
- 状态机：healthy / needs_tuning / blocked / awaiting_confirmation。

**GWT**：
```
Given 所有节点正常，Cpk=1.3，Harris=0.78
Then  verdict.status = "healthy"，recommend_tuning = false

Given 开环节点产生告警
Then  verdict.status = "blocked"，blockers 含 "open_loop"

Given Cpk 缺 USL/LSL
Then  verdict.status = "awaiting_confirmation"，confirmations 含 "missing_pv_limits"
```

### 4.2 整定侧

#### FR-T1：数据加载与画像

**用户价值**：以一致的方式接受 CSV 输入并产出结构化画像。

**描述**：保留 pid_v2 现有 `load_dataset` + `summarize_data` Skill；改造点：
1. 增加"诊断结论传入"通道（诊断流水线已计算的画像可复用，避免二次计算）；
2. 失败时返回结构化错误，不抛异常。

**GWT**：
```
Given 一份 1000 行的合法 CSV
When  调用 load_dataset
Then  返回 cleaned_df, dt, data_profile，data_quality.warnings 为空
```

#### FR-T2：窗口检测与策略

**用户价值**：从历史数据中挑出适合做辨识的时段。

**描述**：保留 5 个算法族（sp_step / mv_step / mv_ramp / steady_disturbance / rolling_scan）+ 本体策略 + LLM 顾问；阈值由 policies 提供。

**GWT**：
```
Given 一份含 3 个 SP 阶跃的 CSV
When  跑 detect_windows
Then  candidate_windows 至少 3 个，其中 ≥1 个 window_usable_for_id = true

Given 本体策略 disabled_algorithm_families = ["rolling_scan"]
Then  rolling_scan 族不产出窗口
```

#### FR-T3：窗口选择

保留现有 user_override / LLM / fallback_deterministic / deterministic 4 种 mode。

#### FR-T4：辨识 + 评审 + 精修

保留多轮循环（1 + 2 = 3 轮上限）+ Phase 3 跨轮兜底。

#### FR-T5：PID 整定

保留 IMC / Lambda / ZN / CHR + 启发式推荐。

#### FR-T6：评估与回放

保留闭环仿真 + 现实性检查 + 时序回放数据。

> FR-T1 ~ T6 详细行为（公式、阈值、Provider 装配）参见文档 5。本 PRD 仅锁"功能存在 + 验收边界"。

### 4.3 工程师协同

#### FR-E1：异步工程师确认

**用户价值**：让需要工艺判断的环节安全暂停，等专家回应。

**描述**：
- 数据契约：`EngineerConfirmation { id, type, prompt, evidence, choices, status, response, expires_at }`；
- 状态机：pending → responded → applied / expired；
- 重入：流水线在 confirmation 处发出 SSE `engineer_confirmation_required` 后挂起会话；用户回应后通过 `/api/sessions/{id}/resume` 重启。

**GWT**：
```
Given 流水线在窗口选择阶段缺少 PV 量程
When  到达该节点
Then  产生 EngineerConfirmation(type="missing_pv_limits")，会话状态 = "awaiting_confirmation"
      SSE 流推 engineer_confirmation_required 事件后中止本次连接

Given 工程师在 24h 内回应
When  调用 /api/confirmations/{id}/respond + /api/sessions/{id}/resume
Then  流水线从挂起点继续，confirmation.status = "applied"

Given 工程师 24h 未回应
Then  confirmation.status = "expired"，会话状态 = "expired"，需手动重启
```

#### FR-E2：告警事件管理

**用户价值**：把诊断里"非整定"的结论沉淀成可处置的工单。

**描述**：
- 数据契约：`AlarmEvent { id, loop_id, type, severity, evidence, recommended_action, status, ack_user, resolution_note, created_at, resolved_at }`；
- 类型枚举（V3.0）：`open_loop / low_cpk / low_harris / missing_ontology_data`；
- 状态机：new → acknowledged → resolved / dismissed；
- 持久化：`var/alarms/<loop_id>/<id>.json`。

**GWT**：
```
Given FIC-105 触发开环
Then  AlarmEvent(type="open_loop", severity="high") 持久化，前端告警中心显示
```

#### FR-E3：阶跃测试向导（V3.1）

V3.0 不做。仅在文档 10 留 placeholder 节点。

### 4.4 知识与经验

#### FR-K1：本体只读接入

**用户价值**：把工艺知识结构化喂给系统。

**描述**：
- MCP 多 server 配置（保留现有 `mcp_config` + `mcp_client`）；
- 调用规范：见文档 9；
- 失败降级：本体不可达 → 诊断仍可降级运行（用兜底阈值 + risk_flags 标注）。

**GWT**：
```
Given MCP server "ontology" 可达，回路 TIC-10707 有完整定义
When  诊断启动
Then  ontology_meta.source = "mcp"，relevant_facts ≥ 5 项

Given MCP server 不可达
Then  诊断继续，ontology_meta.source = "fallback"，risk_flags 含 "ontology_unreachable"
```

#### FR-K2：本地经验沉淀与检索

**用户价值**：让历史整定结果在新整定时被检索和参考。

**描述**：
- 写：每次成功整定后写入 `var/experience/<loop_type>/<loop_id>/<timestamp>.json`，含 K/T/L、PID、评估、工况、操作员备注；
- 读：Consultant `search_experience` 工具按回路名/工况关键词检索，返回相似度排序的 N 条；
- V3.0 不做向量检索；用关键词 + 倒排索引足够。

**GWT**：
```
Given var/experience 中存在 TIC-10707 的 3 条历史整定
When  Consultant 调 search_experience(query="TIC-10707")
Then  返回 ≥3 条按时间倒序的记录
```

### 4.5 系统侧

#### FR-S1：会话与可重放

- 每次诊断/整定都生成 session_id；
- 全部 SSE 事件 + 工具调用 + LLM 思考链落到 `var/sessions/<id>.jsonl`；
- 前端 `/sessions` 页面按时间渲染回放；
- **必须支持离线重放**：仅凭 jsonl + 原 csv + model_config 快照，可在脚本里复刻整次决策（含 LLM 决策结果，但 LLM 调用可被 mock 替换）。

**GWT**：
```
Given 完成一次整定
When  打开 /sessions/{id}
Then  时间轴按事件顺序渲染所有 stage / tool_call / llm_thinking / advisor 事件
```

#### FR-S2：LLM 模型配置中心

保留现有热切换；接入诊断阶段（虽然 V3.0 诊断不接 LLM，但配置中心需为 V3.1 准备好）。

#### FR-S3：SSE 事件流统一协议

- 诊断流水线与整定流水线共用 SSE 协议；
- 事件类型：`stage / llm_thinking / tool_call / engineer_confirmation_required / alarm_raised / result / error`；
- 每个事件都有 `pipeline`（"diagnostic" | "tuning"）、`stage`、`status`、`payload` 字段；
- 详细 schema 见文档 4。

#### FR-S4：配置中心

- 模型配置、MCP 配置、阈值配置、LLM 总开关、诊断阈值；
- 全部支持热加载，落盘到 `var/config/`。

#### FR-S5：日志与可观测性

- 结构化 JSON 日志，按级别滚动；
- 必备指标：诊断/整定调用数、各阶段耗时、LLM 调用成功率、Advisor 回退率、告警/确认事件数；
- 详见文档 8。

---

## 5. 非功能需求（NFR）

### 5.1 性能

| 项 | 目标 | 备注 |
|---|---|---|
| 单回路诊断流水线（无 LLM） | ≤ 5 s | 数据 ≤ 10k 行 |
| 单回路整定流水线（无 LLM） | ≤ 3 s | 同上 |
| 单回路整定流水线（含 4 个 LLM 决策点） | ≤ 30 s | reasoner 模型，单点 ≤ 8s |
| Consultant 单轮响应 P95 | ≤ 12 s | 含 1-2 次 tool_call |
| SSE 首事件 | ≤ 500 ms | 用户感知 |

### 5.2 可靠性

- 单点 LLM 调用失败 → 回退确定性，整体流水线不中断；
- 本体不可达 → 诊断降级运行 + risk_flag；
- 异常都是结构化错误（错误码 + 中文 message），前端可分类处理；
- 进程崩溃后会话日志可继续读，session 不丢失。

### 5.3 本体 SLO

- 目标可用性：99%（业务时段）；
- 单次调用超时：5 s；
- 调用失败率告警阈值：> 5% / 10 分钟。

### 5.4 安全

- 本体内容是"不可信输入"：所有 MCP 文本必须经结构化字段化，不直接拼到 system prompt；
- 工程师确认事件不能在前端绕过（后端二次校验）；
- 配置项变更入审计日志；
- 默认 V3.0 单租户、单装置，不做权限/角色（V3.x 接 SSO）；
- 永远不允许跳过工程师确认对真实控制器写入参数。

### 5.5 可观测

- 每次诊断/整定一条 trace_id 贯穿前端 → 后端 → LLM 调用；
- 关键指标导出到 Prometheus（如部署）；
- 错误日志含可重现命令。

### 5.6 可扩展

- 新增诊断节点 = 新增 1 个 Skill + 1 个 Provider + 在决策树注册一个边；
- 新增整定 Provider = 实现 base interface + 装饰器注册；
- 新增 LLM 模型 = 配置中心改一行；
- 新增告警类型 = 在 alarm types 枚举加一项 + 前端国际化加文案。

---

## 6. 验收标准汇总

每条 FR 已附 GWT。系统级总验收（全部满足才能 V3.0 发布）：

1. **诊断主路径**：3 份代表性 CSV（流量 / 温度 / 液位）+ 各自完整本体 → 诊断结论与人工标注一致率 ≥ 90%；
2. **整定主路径**：与 pid_v2 现有黄金数据集相比，K/T/L 误差 ≤ 5%，整定参数评估分差 ≤ 3 分；
3. **告警主路径**：人为构造 5 类异常 → 全部触发对应告警类型，无误报；
4. **工程师确认主路径**：3 个典型确认场景 → 异步重入成功，无重复挂起；
5. **LLM 关闭场景**：4 个 LLM 决策点全关 → 整定流水线产出参数评估分 ≥ 60；
6. **会话重放**：随机抽 5 个 session → 离线脚本可完整重放，结果一致。

---

## 7. 度量指标

### 7.1 北极星指标

> "每月被系统正确诊断 + 被工程师采纳整定的回路数"

### 7.2 输入指标

- 接入回路数；
- 本体覆盖率（有完整定义的回路数 / 总回路数）；
- 诊断启动数 / 月。

### 7.3 过程指标

- 诊断流水线 P95 耗时；
- LLM Advisor 回退率（按决策点）；
- 工程师确认平均处理时长；
- 告警从 new 到 resolved 的中位时间。

### 7.4 输出指标

- 推荐整定接受率（工程师采纳推荐 PID 的比例）；
- 整定后 Cpk 提升中位数；
- 整定后 Harris 提升中位数；
- 误诊率（事后由工程师标注）。

---

## 8. 里程碑

| 里程碑 | 目标 | 时长（estimate） | 完成标志 |
|---|---|---|---|
| **M0** | 文档体系冻结 | 2 周 | PRD + 文档 9/10 + 文档 3/5/6 骨架评审通过 |
| **M1** | v3 框架 + 整定功能对等 | 3-4 周 | v3 分支跑完整定流水线，与 pid_v2 黄金数据集结果一致 |
| **M2** | 诊断 MVP | 4-6 周 | 5 个 V3.0 诊断节点 + 告警 + 工程师确认全链路打通 |
| **M3** | 前端整合 + 经验沉淀 + 文档同步 | 2-3 周 | 前端新页面 + var/experience 写入与检索 + 文档 4/7/8 完整 |
| **M4** | UAT + 上线准备 | 2 周 | 验收用例全过 + 部署文档 + 回滚预案 |

V3.0 总时长估算：**约 13-17 周**（约 3-4 个月）。

V3.1 不在本 PRD 范围，单独再开 PRD。

---

## 9. 风险与未决问题

### 9.1 高风险

| ID | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R-1 | 本体覆盖度不够 | 高 | Cpk/工况识别等节点频繁触发确认 | 配套提供"本体补全任务清单"工具 |
| R-2 | 工程师确认响应慢 | 中 | 流水线长时间挂起 | 配 24h 过期 + 邮件提醒（V3.1） |
| R-3 | LLM 服务抖动 | 中 | 整定质量波动 | Advisor 全部走结构化输出 + 确定性回退（已设计） |
| R-4 | Harris 闭环估计误判 | 中 | 错误地推荐整定 | V3.0 标 confidence；低 confidence 时只警示不阻断 |
| R-5 | v3 分支迁移引入回归 | 中 | M1 阶段失败 | 保留黄金数据集回归测试，每次合并必跑 |

### 9.2 未决问题

| ID | 问题 | 处理时点 |
|---|---|---|
| Q-1 | V3.0 是否需要管理员后台（用户管理、审计查询）？ | 评审决定 |
| Q-2 | 经验沉淀是否需要支持反向推送（"发现历史经验，建议是否覆盖当前推荐"）？ | M2 |
| Q-3 | 诊断流水线是否对外暴露给第三方系统（API key + 限流）？ | M3 |
| Q-4 | 多语言（仅中文 vs 中英双语）？ | M3 |
| Q-5 | 前端是否需要移动端适配？ | M3 |

---

## 附录 A：FR 速查矩阵

| 编号 | 名称 | V3.0 | V3.1 | 关联文档 |
|---|---|:-:|:-:|---|
| FR-D1 | 工况识别 | ✅ | | 5, 9, 10 |
| FR-D2 | 开/闭环判断 | ✅ | | 5, 10 |
| FR-D3 | PV SNR 评估 | ✅ | | 5, 10 |
| FR-D4 | Cpk | ✅ | | 5, 9, 10 |
| FR-D5 | Harris (闭环) | ✅ | | 5, 10 |
| FR-D6 | 诊断结论汇总 | ✅ | | 3, 10 |
| FR-D7 | 阀门死区/粘滞 | | ✅ | 5, 10 |
| FR-D8 | 耦合识别 | | ✅ | 5, 9, 10 |
| FR-D9 | Harris 开环阶跃 | | ✅ | 5, 10 |
| FR-D10 | MISO 辨识 | | ✅ | 5, 9 |
| FR-T1 | 数据加载画像 | ✅ | | 5 |
| FR-T2 | 窗口检测策略 | ✅ | | 5, 9 |
| FR-T3 | 窗口选择 | ✅ | | 5, 6 |
| FR-T4 | 辨识 + 评审 + 精修 | ✅ | | 5, 6 |
| FR-T5 | PID 整定 | ✅ | | 5 |
| FR-T6 | 评估回放 | ✅ | | 5 |
| FR-E1 | 异步工程师确认 | ✅ | | 3, 4, 10 |
| FR-E2 | 告警事件管理 | ✅ | | 3, 4, 10 |
| FR-E3 | 阶跃测试向导 | | ✅ | 7, 10 |
| FR-K1 | 本体只读接入 | ✅ | | 9 |
| FR-K2 | 本地经验沉淀检索 | ✅ | | 6 |
| FR-K3 | 本体回写 | | ✅ | 9 |
| FR-S1 | 会话与可重放 | ✅ | | 8 |
| FR-S2 | LLM 配置中心 | ✅ | | 8 |
| FR-S3 | SSE 事件协议统一 | ✅ | | 4 |
| FR-S4 | 配置中心 | ✅ | | 8 |
| FR-S5 | 日志与可观测 | ✅ | | 8 |

---

## 附录 B：与 pid_v2 现有代码的复用边界

| 模块 | V3.0 处置 | 备注 |
|---|---|---|
| `core/algorithms/*` | 全保留 | 注释中文化 |
| `core/policies/*` | 全保留 + 扩字段 | 加诊断阈值 |
| `core/providers/*` | 协议保留，扩实现 | 新加 Cpk/Harris/SNR/工况识别/开闭环 5 个 Provider |
| `core/skills/{base,registry}` | 保留 | 协议正确 |
| `core/skills/*` 子集 | 保留 + 扩 5+ 新 Skill | 见文档 6 |
| `core/pipeline/runner.py` | **重写** | 线性 → 决策树 + 重入 |
| `core/pipeline/events.py` | **重写** | 事件协议升级 |
| `core/agent/*` | 改造接 SkillRegistry | 工具透传到 Skill |
| `models/*` | **重写** | 新数据契约 |
| `api/*` | **重写** | 新接口骨架 |
| `frontend/*` | 大部分页面重写 | 新加 4-5 个页面 |
| MCP 客户端 | 保留 + 扩写回（V3.1） | |
| `_demo` skill / 已废 advisor | **删除** | |
| 黄金数据集与现有测试用例 | 保留 | 作为 M1 回归基线 |

---

## 附录 C：评审 checklist

> 请评审人按此 checklist 给意见。

- [ ] 默认决策 D-1 ~ D-7 是否同意？不同意请直接改"默认值"列
- [ ] V3.0 范围是否过大或过小？
- [ ] 用户场景 5 个是否覆盖典型工作流？是否有遗漏？
- [ ] 每条 FR 的 GWT 是否可验收（不模糊、可写测试）？
- [ ] NFR 的性能目标是否合理（不过激、不松）？
- [ ] 北极星指标是否衡量真正的业务价值？
- [ ] 里程碑总时长是否符合资源约束？
- [ ] 风险清单是否有遗漏的高风险项？
- [ ] 复用边界（附录 B）是否合理（保留 vs 重写）？
