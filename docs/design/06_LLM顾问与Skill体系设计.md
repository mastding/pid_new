# 文档 6：LLM 顾问与 Skill 体系设计

> 文档类型：详细设计（DD）
> 主要读者：算法工程师、后端工程师
> 关联文档：02 总体架构设计、04 API 接口设计、05 算法与流水线详细设计
> 配套代码：`backend/core/agent/`、`backend/core/skills/`、`backend/core/pipeline/{llm_advisor.py, identification_advisor.py, identification_refinement_advisor.py, window_policy_advisor.py}`、`backend/core/mcp_*`、`backend/core/model_config.py`

---

## 0. 文档说明

### 0.1 设计目标

1. 把 LLM 接入限定在两个清晰的角色：
   - **顾问（Advisor）**：嵌在确定性流水线的若干决策点，单次同步 + 结构化输出，失败必须可回退；
   - **顾问 Agent（Consultant）**：用户在前端会话里主动调起，工具调用循环（ReAct），用于解释/迭代/经验检索。
2. **不引入任何 Agent 框架**（AutoGen / LangChain / LangGraph）。整个 Consultant 是 ~80 行的 OpenAI function-calling 循环；
3. **Skill 是 LLM 视角的能力**：原子、确定性、自校验；统一通过 Pydantic 模型生成 JSON Schema；
4. 服务端可变状态（`LoopContext`）**永远不暴露**给 LLM，LLM 只通过 Skill 入参 + Skill 返回的精简 dict 与系统交互；
5. 通过模型配置中心（`model_config`）实现 LLM 模型可热插拔；通过 MCP 客户端实现外部知识接入。

### 0.2 边界与不做的事

- 顾问**不直接计算**：所有数值都来自 Skill / Provider 返回；
- 顾问**不写数据库**：经验存储由 Skill 接管；
- 顾问**不并行多 Agent**：Day 4 之前的 AutoGen 多 Agent 方案已下线；
- 顾问**不做工具白名单的运行时降级**：白名单由调用方在构造 `tool_definitions` 时确定；
- 顾问**不做长时记忆**：每次会话以前端传入的 `messages` + `session` 为准；持久化由会话日志实现。

### 0.3 名词约定

| 名词 | 含义 |
|---|---|
| **Advisor** | 嵌在 pipeline 里的单次 LLM 决策（窗口策略 / 选窗 / 评审 / 精修） |
| **Consultant** | 前端聊天框对应的工具调用循环 Agent |
| **Skill** | 注册到 `SkillRegistry` 的原子能力，可被 LLM 通过 function-call 调用 |
| **LoopContext** | 服务端会话内可变上下文，Skill 间共享，**对 LLM 不可见** |
| **MCP** | Model Context Protocol，本系统用作外部本体/知识接入通道 |
| **思考链（reasoning_content）** | reasoner 类模型返回的链式思考文本 |

---

## 1. 总体结构

### 1.1 模块视图

```
                          ┌───────────────────────────┐
                          │   FastAPI 路由             │
                          │ /api/tuning/...            │
                          │ /api/consult/stream        │
                          └──────────┬────────────────┘
                                     │
                ┌────────────────────┴──────────────────────┐
                ▼                                            ▼
   ┌─────────────────────────────┐         ┌─────────────────────────────┐
   │  pipeline.runner             │         │  agent.consultant           │
   │  ──────────────              │         │  ──────────────              │
   │  顺序编排 + SSE              │         │  单 Agent 工具循环           │
   │                              │         │                              │
   │  Advisor 调用：              │         │  System Prompt（顾问角色）  │
   │   ├── window_policy_advisor  │         │  TOOL_DEFINITIONS（5 个）   │
   │   ├── llm_advisor (window)   │         │  on_event（流式回调）        │
   │   ├── identification_advisor │         │                              │
   │   └── identification_refinement_advisor │         │
   └──────────┬───────────────────┘         └──────────┬──────────────────┘
              │                                         │
              │              共用基础设施                │
              ▼                                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │  model_config.store     ── 当前 LLM 配置（key/url/model）  │
   │  mcp_config.store       ── MCP server 配置                 │
   │  mcp_client             ── list_tools / call_tool          │
   │  session_log.record_*   ── 会话/工具调用持久化              │
   └──────────────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  SkillRegistry  ── 9+ 个 Skill                            │
   │   ├── data_understanding：load_dataset / summarize_data   │
   │   ├── window：detect_windows / select_window              │
   │   ├── dead_time：estimate_dead_time                       │
   │   ├── identification：identify_model                      │
   │   ├── tuning：generate_tuning_candidates                  │
   │   ├── evaluation：evaluate_tuning                         │
   │   ├── assessment：assess_loop_assessment                  │
   │   └── monitoring：assess_loop_monitoring                  │
   └──────────────────────────────────────────────────────────┘
```

### 1.2 两类 LLM 调用对比

| 维度 | Advisor（流水线决策点） | Consultant（前端会话） |
|---|---|---|
| 位置 | `pipeline/*_advisor.py` | `agent/consultant.py` |
| 触发 | 流水线运行时按阶段触发 | 用户在前端发送消息 |
| 模型偏好 | reasoner 类（deepseek-reasoner） | chat 类（默认与 reasoner 通用） |
| 调用形式 | 单次同步，结构化 JSON 输出 | 流式 + 工具调用循环（≤ 10 轮） |
| 失败处理 | 必须有确定性回退 | 直接把错误吐回会话，不阻断系统 |
| 工具集 | 不用工具，仅产生 JSON 决策 | TOOL_DEFINITIONS 5 个 |
| 上下文 | data_profile + 候选 + 本体 | messages + session（model/PID 当前值） |
| 输出 | 写回 `selection_meta` / `review_result` / `refinement` | 写回 SSE 事件流（text_chunk / tool_call / final） |

---

## 2. Skill 体系

### 2.1 Skill 协议

源码：`core/skills/base.py`

```python
class BaseSkill(ABC):
    name: ClassVar[str]                # snake_case，全局唯一，LLM 可见
    description: ClassVar[str]         # 中文，LLM 可见
    input_model: ClassVar[type[BaseModel]]  # Pydantic，自动生成 JSON Schema

    @abstractmethod
    def run(self, inputs: BaseModel, ctx: LoopContext) -> SkillResult: ...

    def invoke(self, raw_args: dict, ctx: LoopContext) -> SkillResult:
        # 1. inputs = self.input_model(**raw_args)  → 校验失败返回 SkillResult(success=False)
        # 2. result = self.run(inputs, ctx)         → 任意异常被吞，转 SkillResult(success=False)
        # 3. ctx.skill_log.append({...})            → 审计日志
```

**约束**：

1. **Skill 不抛异常**：任何失败都返回 `SkillResult(success=False, reasoning="...")`；
2. **入参 schema 由 Pydantic 模型生成**：`BaseSkill.to_openai_tool()` 自动产出 OpenAI function 定义；
3. **`description` 必须是中文**（项目约定）；
4. **不接收 `LoopContext` 来自 LLM**：context 是服务端隐藏状态，LLM 不知其字段；
5. **副作用要写到 context**：例如 `identify_model` 把 `best_model` 写回 `ctx.model`，`detect_windows` 写 `ctx.candidate_windows`；
6. **审计在基类**：每次 invoke 自动追加 `ctx.skill_log`，子类不需要管。

### 2.2 SkillResult 协议

```python
class SkillResult(BaseModel):
    success: bool
    data: dict[str, Any] = {}        # 形状对应每个 Skill 自己的 output 约定
    warnings: list[str] = []
    reasoning: str = ""

    def to_llm_dict(self) -> dict[str, Any]:
        """LLM 可见视图：省略空字段，节省 token。"""
```

### 2.3 LoopContext 字段约定

| 字段 | 类型 | 谁写 | 谁读 |
|---|---|---|---|
| `csv_path / loop_prefix / loop_type` | 入会话时给定 | runner、Skill | 全部 |
| `raw_df / cleaned_df / dt` | `load_dataset` | 全部下游 |
| `data_profile` | `summarize_data` / `load_dataset` | window / identification / tuning / advisors |
| `candidate_windows` | `detect_windows` | identify_model / select_window / evaluation |
| `selected_window_index` | `select_window` / 用户 override | identify_model |
| `model` | `identify_model` | tuning / evaluation |
| `confidence` | `identify_model` | tuning / evaluation |
| `pid_params` | `generate_tuning_candidates` | evaluation |
| `evaluation` | `evaluate_tuning` | 报告生成 |
| `skill_log` | base 类 | 报告 / 调试 |

**禁止跨 Skill 隐式依赖**：例如 `evaluate_tuning` 必须读到 `ctx.pid_params`，不能从外部传；调用顺序不满足时 Skill 必须立即返回 `success=False`。

### 2.4 Skill 注册表

源码：`core/skills/registry.py`

- 模块级单例 `registry`；
- `@register` 装饰器在子模块 import 时副作用注册；
- `core/skills/__init__.py` 显式 import 所有子包以触发注册；
- 提供：
  - `registry.invoke(name, args, ctx)`：服务端调用
  - `registry.to_openai_tools(names=None)`：给 Consultant 生成工具白名单
  - `registry.names()`：列出所有 Skill

### 2.5 Skill 全集与默认 Provider

| 类别 | Skill 名 | 默认 Provider | 主要入参 | 写入 context |
|---|---|---|---|---|
| data_understanding | `load_dataset` | `clean_csv_loader` | provider, loop_prefix, start/end_time | raw_df, cleaned_df, dt |
| data_understanding | `summarize_data` | `deterministic_profile` | provider | data_profile |
| window | `detect_windows` | `policy_composite` | provider, loop_type, max_windows, include_unusable, policy | candidate_windows |
| window | `select_window` | `quality_score_selector` | provider | selected_window_index |
| dead_time | `estimate_dead_time` | `cross_correlation` | provider, window_index | data_profile.dead_time |
| identification | `identify_model` | `transfer_function_fit` | provider, window_indices, use_usable_windows_only, model_pool, hint_L | model, confidence |
| tuning | `generate_tuning_candidates` | `classic_family` | provider | pid_params |
| evaluation | `evaluate_tuning` | `closed_loop_sim` | provider, tuning_unreliable, tuning_unreliable_reason | evaluation |
| assessment | `assess_loop_assessment` | 内置 | loop_id, loop_type | data_profile.assessment |
| monitoring | `assess_loop_monitoring` | 内置 | loop_id, loop_type | data_profile.monitoring |

### 2.6 Skill 详细规约模板（每个 Skill 一个章节）

下文以 4 个高频 Skill 为例展示规约形态；其余 Skill 沿用相同模板补全。

#### 2.6.1 `identify_model`

**用途**：在 1 个或多个候选窗口上做系统辨识，返回最佳模型 + attempts。

**输入**（`IdentifyModelInputs`）：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `provider` | str | `transfer_function_fit` | identification 类别下的 Provider |
| `window_indices` | list[int] \| None | None | 指定参与辨识的窗口索引（全局索引） |
| `use_usable_windows_only` | bool | True | 是否仅使用 `window_usable_for_id=True` 窗口 |
| `model_pool` | list[str] \| None | None | 强制模型白名单（如 `["FOPDT","SOPDT"]`） |
| `hint_L` | float \| None | None | 死区初值提示（精修使用） |

**输出**（`SkillResult.data`）：

```yaml
provider: str
best_model:
  model_type: "FO|FOPDT|SOPDT|IPDT|SOPDT_UNDER|IFOPDT"
  K, T, T1, T2, L, zeta
  r2_score, normalized_rmse, raw_rmse
  confidence
attempts: list[dict]            # 完整尝试明细
window_source: str              # best_model 来自哪个窗口
selection_reason: str
fit_preview: dict               # 时序快照
candidates: list                # AIC 比较 top
meta:
  window_count, attempt_count
```

**前置条件**：`ctx.cleaned_df is not None and ctx.dt is not None and ctx.candidate_windows`，否则 `success=False`。

**副作用**：写 `ctx.model`、`ctx.confidence`、`ctx.data_profile.identification`。

**典型失败**：

| 原因 | reasoning |
|---|---|
| 未加载数据 | "未检测到已加载的数据集，请先调用 load_dataset。" |
| 没有候选窗口 | "未检测到候选窗口，请先调用 detect_windows。" |
| 未知 provider | "未知系统辨识 provider: {name}" |
| 拟合全部失败 | "全部 attempts 失败：{首条错误}" |

#### 2.6.2 `generate_tuning_candidates`

**用途**：基于已辨识模型生成 PID 候选与推荐。

**输入**：`provider: str = "classic_family"`。

**输出**：

```yaml
recommended:
  strategy: "IMC|LAMBDA|ZN|CHR"
  Kp, Ki, Kd, Ti, Td
candidates: list[ {strategy, Kp, Ki, Kd, ...} ]
heuristic_strategy: str
heuristic_reason: str
tuning_unreliable: bool
tuning_unreliable_reason: str
```

**前置条件**：`ctx.model is not None`。

#### 2.6.3 `evaluate_tuning`

**用途**：闭环仿真 + 性能评分 + 现实性检查。

**输入**：`provider`、`tuning_unreliable`、`tuning_unreliable_reason`。

**输出**：见文档 5 § 2.7。

**前置条件**：`ctx.model is not None and ctx.pid_params is not None`。

#### 2.6.4 `search_experience`（规划中，前端 Consultant 使用）

**用途**：向量检索历史整定经验。

**输入**：`query: str`。

**输出**：`{ items: [ {loop_name, summary, similarity, params} ] }`。

**实现**：`services/experience_store`（V2.1 引入）。

---

## 3. Consultant Agent

### 3.1 设计要点

源码：`core/agent/consultant.py`（≈ 170 行）。

- **单 Agent**：没有 planner / executor / critic 等多 Agent；
- **流式**：用 `client.chat.completions.create(..., stream=True)`，对话内容逐块产出；
- **ReAct 由 OpenAI 协议原生支持**：模型决定要不要调用 tool；assistant 消息可同时含 `content` 与 `tool_calls`；
- **每个 tool 包 60s 超时**：异步 `asyncio.wait_for`；同步函数自动 `asyncio.to_thread`；
- **轮数硬上限**：`max_iterations=10`，到顶仍未生成 final 文本 → 输出"已达到最大迭代次数"占位回复；
- **事件协议**：

  ```yaml
  - type: text_chunk
    content: <增量文本>
  - type: tool_call
    name: <tool 名>
    args: <dict>
    result: <dict 或 {"error": "..."}>
  - type: final
    content: <汇总回复>
    iterations: <int>
  ```

### 3.2 工具集（TOOL_DEFINITIONS）

源码：`core/agent/tools.py`，5 个工具：

| 工具名 | 入参 | 行为 |
|---|---|---|
| `run_identification` | `window_index?`、`model_type?` | 在 session 指定 csv 上跑 `fit_best_model`，返回 model 参数与拟合 |
| `run_tuning` | `strategy?`、`lambda_factor?`、`kp_scale?`、`ki_scale?`、`kd_scale?` | 用当前 session 模型调 `select_best_strategy`，并按 scale 微调 |
| `run_evaluation` | `Kp`、`Ki`、`Kd` | `evaluate_pid_params` 闭环仿真 |
| `get_data_overview` | — | 读 session.csv_path，返回画像、窗口数、当前 model/PID |
| `search_experience` | `query` | 检索经验（V2.1 接入） |

> 与流水线 Skill 的差异：Consultant 的工具是**会话感知**的闭包（`_build_tool_handlers(session)`），它们直接复用 `algorithms/*` 函数，不经 SkillRegistry。
> **Day 5 计划**：把这些工具改造为透传到 SkillRegistry，让 Advisor 与 Consultant 共用同一个 Skill 集；当前两套并存以保持向后兼容。

### 3.3 系统提示词（System Prompt）

源码：`core/agent/prompts.py`。提示词遵循以下结构：

```
1. 角色（"资深过程控制工程师"）
2. 职责清单（审查辨识 / 解释决策 / 迭代优化 / 经验推荐）
3. 物理量级先验（按 loop_type 给典型 T、L 范围）
4. 行为准则（用中文 / 先结论 / 不编参数 / 风险提醒）
```

**修改约定**：

- 任何调整都要同时更新本文档；
- 加入新工具时必须在提示词中明确说明何时使用；
- 提示词需保持纯中文（项目约定）。

### 3.4 会话上下文（SessionContext）

源码：`api/consultant_routes.py`。前端每次请求都附带：

```python
class SessionContext(BaseModel):
    csv_path: str
    loop_type: str = "flow"
    dt: float = 1.0
    # 当前模型
    model_type, model_K, model_T, model_T1, model_T2, model_L
    model_r2, model_nrmse, model_confidence
    n_windows: int
    # 当前 PID
    Kp, Ki, Kd, Ti, Td
    tuning_strategy: str
```

**重点**：服务端不持久化 SessionContext，每条请求都是无状态的；状态由前端持有并附带。

### 3.5 调用入口与 SSE 格式

```
POST /api/consult/stream
body: {
  "messages": [{"role":"user|assistant","content":"..."}],
  "session": { ...SessionContext... },
  "max_iterations": 8
}

SSE event payload:
event: chunk
data: {"type":"text_chunk","content":"..."}

event: tool_call
data: {"type":"tool_call","name":"run_tuning","args":{...},"result":{...}}

event: final
data: {"type":"final","content":"...","iterations":3}
```

### 3.6 安全与可控性

| 维度 | 措施 |
|---|---|
| 工具超时 | 60s/调用 |
| 轮数上限 | `max_iterations`，默认 8，上限 15 |
| 编参数防御 | 系统提示词强制"所有数值来自工具返回" |
| 失败回退 | tool 失败返回 `{"error":...}`，让 LLM 在下一轮看到并自我修正 |
| 不稳定参数警告 | 系统提示词要求 LLM 主动警告 |
| 持久化 | 通过 `session_log.record_stream` 落地全部事件，便于复盘 |

---

## 4. Advisor 详细规约

> 4 个 Advisor 共用同样的实现模式：
>
> 1. 用 `model_cfg_store.get()` 取当前模型配置；
> 2. 同步 `OpenAI.chat.completions.create()`，不走 stream（仅最终结果）；
> 3. 解析 JSON（带正则容错去除 ```json``` 围栏）；
> 4. 任意异常返回 `None` 或 `available=False` 结构，让上层走确定性回退；
> 5. 思考链以 `llm_thinking` 事件单独发出（如有 reasoning_content）。

### 4.1 `ask_window_policy_via_llm`

- **位置**：`pipeline/window_policy_advisor.py`
- **触发点**：`ontology_policy` 阶段，base policy 已生成后；
- **输入**：`base_policy`、`data_profile`、`mcp_context`、`frontend_context`
- **输出 schema**：
  ```yaml
  algorithm_plan: [{family, state in {available,disabled}, reason}]
  disabled_algorithm_families: [str]
  merge_gap_s: float
  expected_gain_sign: -1 | 0 | 1 | "unknown"
  expected_time_constant_range_s: [low, high]
  expected_dead_time_range_s: [low, high]
  confidence: float
  llm_policy_reasoning_content: str   # 仅 reasoner 模型有
  llm_policy_raw_text: str
  ```
- **回退**：`base_policy`（保持系统可运行）。

### 4.2 `choose_window_via_llm`

- **位置**：`pipeline/llm_advisor.py`
- **触发点**：`window_selection` 阶段，pool 中至少 2 个窗口；
- **System Prompt**：判据要点 6 条（mv_span 优先、score 不盲信、漂移谨慎、mv_step 优先、本体一致性、即使全差也要选）；
- **输出 schema**：
  ```json
  {
    "chosen_index": <0~N-1>,
    "reasoning": "<≤200 字>",
    "ontology_evidence": [{"fact": "...", "source": "..."}],
    "window_judgements": [{"index": int, "verdict": "preferred|acceptable|risk", "reason": "..."}]
  }
  ```
- **回退**：`fallback_deterministic`（按 quality_score 选）。

### 4.3 `review_identification_via_llm`

- **位置**：`pipeline/identification_advisor.py`
- **触发点**：每轮辨识结束；
- **System Prompt**：8 条判据（K 符号、T 量级、R²/NRMSE、第二名差距、死区饱和、corr、SOPDT_UNDER 振荡核验、IFOPDT 积分对象核验）；
- **verdict 二元化**：只有 `accept` / `downgrade`，**没有 reject**（不让 LLM 直接终止流水线）；
- **输出 schema**：
  ```yaml
  verdict: "accept" | "downgrade"
  reason: str
  concerns: [str]
  available: bool      # LLM 是否可用
  reasoning_content: str
  raw_text: str
  error_type: str | null
  error_message: str | null
  ```
- **回退**：`available=False, verdict="accept", fallback=true`。

### 4.4 `ask_refinement_via_llm`

- **位置**：`pipeline/identification_refinement_advisor.py`
- **触发点**：`verdict==downgrade` 且仍有轮次预算；
- **输入**：上轮 best、attempts、review、windows_summary、algorithm_comparison、history_summary；
- **输出 schema**：
  ```yaml
  retry: bool
  rationale: str
  force_window_index: int | null
  force_model_types: [str]
  hint_L: float | null
  reasoning_content: str
  source: "llm"
  ```
- **回退**：`recommend_refinement_from_algorithm_comparison`（确定性策略）。

### 4.5 Advisor 失败矩阵

| Advisor | LLM 不可达 | LLM 返回非法 JSON | 索引/参数越界 | 触发动作 |
|---|---|---|---|---|
| window_policy | 用 base policy | 同 | — | 静默降级，policy_source="default" |
| window_selection | fallback_deterministic | 同 | 同 | mode=fallback_deterministic |
| identification_review | accept + fallback=true | 同 | 不可能 | 不触发精修 |
| identification_refinement | recommend_*（确定性） | 同 | 视为放弃重试 | Phase 3 兜底 |

---

## 5. 模型配置（model_config）

### 5.1 配置项

源码：`core/model_config.py`，单例 `store`。

```python
ModelConfig:
    model_name: str          # 例 "deepseek-reasoner" / "qwen-plus"
    model_api_url: str       # OpenAI 兼容端点
    model_api_key: str
    timeout_s: float = 30.0
    max_tokens: int | None = None
    extra_headers: dict | None = None
```

### 5.2 行为

- 进程启动从 `.env` / 配置文件加载默认值；
- 通过 `/api/model_config` 在线热更新，写盘到 `var/model_config.json`；
- Advisor 与 Consultant 都统一用 `model_cfg_store.get()`；
- 切换 reasoner ↔ chat 模型时无需改代码。

### 5.3 多模型策略（V2.x 规划）

| 决策点 | 推荐模型族 | 原因 |
|---|---|---|
| ontology_policy / window_selection / review / refinement | reasoner | 需要"思考链 + 一次性结构化判断" |
| Consultant | chat（reasoner 也能跑） | 工具循环对延迟敏感，chat 更便宜 |

---

## 6. MCP 集成

### 6.1 客户端

源码：`core/mcp_client.py`、`core/mcp_config.py`。

- 暴露 `list_tools(server_name)` 与 `call_tool(server_name, tool_name, args)`；
- 用于本体上下文：`fetch_loop_ontology_context_via_mcp(loop_name, loop_type)`，把 MCP 服务返回的本体描述塞进 `mcp_context`，再喂给 `ask_window_policy_via_llm` 和 `choose_window_via_llm`；
- 任何 MCP 调用失败都不阻塞流水线，仅在 `ontology_meta.ontology_mcp_error` 记录。

### 6.2 配置入口

`/api/mcp_config` 路由 + `var/mcp_config.json`：

```yaml
servers:
  - name: ontology
    url: "stdio:///path/to/ontology_mcp"  # 或 ws://...
    enabled: true
    auth: { ... }
```

### 6.3 安全

- MCP 调用结果是**不可信输入**：所有内容必须经 LLM 翻译/总结后再写入决策，不能直接写到 policy 字段；
- 内容长度被截断到 `mcp_content[:1200]` 进入 ontology_meta；
- 现阶段不支持 MCP 工具被 Consultant 直接调用（避免在用户会话中产生不可控的副作用）。

---

## 7. 会话日志与可观测性

### 7.1 落地结构

- `core/session_log.py`：以 `var/sessions/<session_id>.jsonl` 形式按事件 append；
- 每条事件包含：
  - `ts`、`type`、`stage`、`payload`、`source`（pipeline / consultant / advisor）；
- `record_stream(session_id, events)` 在 SSE 推送时同步落地。

### 7.2 字段规范

| 字段 | 备注 |
|---|---|
| `session_id` | 由前端生成 uuid |
| `tool_call.args / result` | Consultant 工具调用全留痕 |
| `advisor_payload` | 每个 Advisor 的 `raw_text + parsed_json` 都存档 |
| `llm_thinking` | reasoner 思考链单独存档，可用于 A/B 评测 |

### 7.3 用途

- 复盘：前端 `/sessions` 页面读取，按时间渲染；
- 评测：离线脚本基于 jsonl 跑指标（acceptance rate、refinement convergence、verdict distribution）；
- 安全：发现可疑指令注入时回查上下文。

---

## 8. 安全 / 一致性 / 可重放

### 8.1 LLM 输入侧

| 风险 | 措施 |
|---|---|
| 提示词注入（来自数据/本体） | 所有外部文本被结构化字段化，不直接拼到 system prompt；mcp_content 限长 1200 字符 |
| 数值幻觉 | Advisor 必须输出严格 JSON；Consultant 在 prompt 中要求"所有数值来自工具返回" |
| 跨回路串扰 | 每个会话独立 LoopContext，禁止全局可变状态 |
| 工具误用 | 5 个工具的 enum 限制；超时；schema 校验 |

### 8.2 LLM 输出侧

| 风险 | 措施 |
|---|---|
| 非法 JSON | 正则去围栏 + try/except → 回退 |
| 越界索引 | 上层校验 chosen_index in [0, len(pool)) → 不通过即回退 |
| verdict 越权 | review 只允许 accept/downgrade |
| 长 token 滚雪球 | Consultant 设 max_iterations，Advisor 单次调用 |

### 8.3 可重放性

- 会话 jsonl + 原 csv + model_config 快照 = 可在离线脚本里完整重放（含 LLM 决策轨迹）；
- Advisor 入参完全由确定性数据 + base policy 构成，可单独单元测试 mock LLM。

---

## 9. 测试策略

### 9.1 Skill 单元测试

| Skill | 用例 |
|---|---|
| `load_dataset` | 正常 / 空文件 / 缺失列 / NaN 行 |
| `summarize_data` | 字段完整性 / text_summary 长度 |
| `detect_windows` | 全开族 / disabled 族真不跑 / merge_gap_s 影响 |
| `select_window` | 单候选 / 多候选 / 全不可用 |
| `identify_model` | 黄金 FOPDT / 强制 model_pool / hint_L 收敛 / 前置缺失 → success=False |
| `generate_tuning_candidates` | 各 loop_type 的启发式策略一致性 |
| `evaluate_tuning` | passed 阈值 / 仿真发散 |

### 9.2 Advisor 测试（mock LLM）

- 用 `pytest.mock` 把 OpenAI client 替换为 fixture，覆盖：
  - 正常 JSON
  - 含 ```json``` 围栏
  - 含中文额外解释
  - 完全不可解析
  - 索引越界
  - 网络异常
  - 解析后字段缺失

### 9.3 Consultant 测试

- mock `client.chat.completions.create` 的 stream，构造一系列 chunk；
- 覆盖：
  - 单轮直接 final
  - 1 次 tool_call 后 final
  - 多轮 tool_call（≥ 3）
  - 工具超时
  - 工具抛错
  - max_iterations 触发兜底回复

### 9.4 端到端契约测试

- `tests/test_consult_stream.py`：
  - SSE 帧顺序合法（text_chunk* → tool_call* → ... → final）；
  - JSON 字段齐全；
  - 错误工具不阻断 final。

---

## 10. 演进与未决问题

| 项 | 现状 | 计划 |
|---|---|---|
| Consultant 工具 ↔ Skill 统一 | 两套并存（algorithms/ 直调 vs SkillRegistry） | V2.1 把 Consultant 工具改为 SkillRegistry pass-through |
| 工具白名单按角色 | 全部暴露 | V2.1 支持按 loop_type / 用户角色裁剪 |
| 经验存储 | 占位 search_experience | V2.1 引入 `services/experience_store` + 向量检索 |
| 模型族切换的 A/B | 手动切换 | V2.2 加离线评测脚本（基于 session jsonl） |
| Advisor 思考链不可见时降级 | reasoner 强相关 | V2.2 抽出 `prompt_pack`，按模型族切提示词 |
| MCP 工具暴露给 Consultant | 不暴露 | V2.x 评估安全模型后再考虑 |
| 多回路并行会话 | 单 SessionContext | V2.2 支持 `loops: list[SessionContext]` |

---

## 附录 A：Advisor / Consultant 一览速查

| 名称 | 文件 | 触发 | 模型 | 失败回退 |
|---|---|---|---|---|
| ask_window_policy_via_llm | `pipeline/window_policy_advisor.py` | ontology_policy | reasoner | base_policy |
| choose_window_via_llm | `pipeline/llm_advisor.py` | window_selection | reasoner | quality_score_selector |
| review_identification_via_llm | `pipeline/identification_advisor.py` | 每轮辨识结束 | reasoner | accept |
| ask_refinement_via_llm | `pipeline/identification_refinement_advisor.py` | downgrade 后 | reasoner | recommend_refinement_from_algorithm_comparison |
| run_consultant | `agent/consultant.py` | 用户聊天 | chat（兼容 reasoner） | 工具异常吐回会话 |

## 附录 B：Skill 输出在 LLM 视图下的精简形态

`SkillResult.to_llm_dict()` 会移除空字段，避免占用 token。例如 `identify_model` 成功时返回给 LLM 的形态：

```json
{
  "success": true,
  "data": {
    "provider": "transfer_function_fit",
    "best_model": {"model_type":"FOPDT","K":0.95,"T":8.1,"L":1.6,"r2_score":0.92,"confidence":0.78},
    "window_source": "sp_step_idx_120_280",
    "selection_reason": "AIC 最低且 R²>0.9",
    "candidates": [...],
    "meta": {"window_count":3,"attempt_count":12}
  },
  "reasoning": "已用 transfer_function_fit 完成系统辨识，尝试 12 次。"
}
```

## 附录 C：扩展 Skill 的开发指引

1. 在 `core/skills/<category>/` 新建 `my_skill.py`；
2. 定义 `Inputs(BaseModel)`、`MySkill(BaseSkill)`：
   - `name = "my_skill"`（snake_case）
   - `description = "中文描述..."`
   - `input_model = Inputs`
   - `run(inputs, ctx)` → `SkillResult(...)`
3. 用 `@register` 装饰类；
4. 在 `core/skills/__init__.py` 中 `from core.skills.<category> import my_skill`（触发注册）；
5. 在测试目录加单元测试（成功 / 前置缺失 / 异常吞噬）；
6. 如需在 Consultant 中暴露：在 `agent/tools.TOOL_DEFINITIONS` 添加对应 schema，并在 `_build_tool_handlers` 接出 closure。

## 附录 D：关键提示词来源（仅指针，避免重复）

| 提示词 | 文件 | 行号附近 |
|---|---|---|
| ontology_policy | `pipeline/window_policy_advisor.py` | `SYSTEM_PROMPT` |
| window_selection | `pipeline/llm_advisor.py` | `SYSTEM_PROMPT` |
| identification_review | `pipeline/identification_advisor.py` | `SYSTEM_PROMPT` |
| identification_refinement | `pipeline/identification_refinement_advisor.py` | `SYSTEM_PROMPT` |
| consultant | `agent/prompts.py` | `SYSTEM_PROMPT` |

> 任何提示词改动都必须同步更新本文档对应章节，并在 PR 描述里给出影响评估（哪些决策点可能行为变化、是否影响测试用例）。
