# 文档 4：API 接口设计

> 文档类型：详细设计（DD）
> 状态：**Draft V1.0（待评审）**
> 主要读者：前端工程师、后端工程师、第三方集成方
> 关联文档：01 PRD / 02 架构 / 03 领域模型 / 05 算法与流水线 / 10 诊断决策树
> 配套代码：V3.0 在 `pid_v2` 仓库 `v3` 分支重写 `backend/api/`

---

## 0. 文档说明

### 0.1 设计目标

1. 锁定 V3.0 全部 HTTP / SSE 接口契约；
2. 让前端在没有后端的情况下，按本文档 mock 即可开发；
3. 让后端在没有前端的情况下，按本文档逐接口写测试；
4. 与文档 3（数据契约）、文档 10（决策树事件）保持 1:1 一致。

### 0.2 设计原则

- **REST 资源化**：每个一等公民实体（session / alarm / confirmation / experience / config / model_config / mcp_config）有独立路由；
- **流式分离**：长流程（诊断 / 整定 / Consultant）走 SSE，状态机迁移走 HTTP POST；
- **错误结构化**：所有错误都有 `error_code` 中文 `error_message` `error_details`；
- **版本前缀**：所有路由 `/api/v3/`（V3.0 起），V2 旧接口在 legacy/v2 分支保留；
- **不暴露内部实现**：response 字段对应 `models/` 中的 Pydantic，序列化稳定。

---

## 1. 通用约定

### 1.1 路由版本

```
/api/v3/<resource>/...
```

V3.0 不维护多版本并存；后续如果有破坏性变更再升 v4。

### 1.2 内容协商

| 类型 | Content-Type |
|---|---|
| HTTP 请求 | `application/json; charset=utf-8` |
| HTTP 响应（成功） | `application/json; charset=utf-8` |
| SSE 流 | `text/event-stream; charset=utf-8` |
| 文件上传（CSV） | `multipart/form-data` |

### 1.3 鉴权（V3.0）

- 默认无鉴权（内网部署）；
- 可选 Basic Auth（配置 `feature_flags.auth.enabled = true`）；
- V3.x 接 SSO 后引入 Bearer Token。

### 1.4 通用 Header

| Header | 用途 | 必需 |
|---|---|---|
| `X-Trace-Id` | 端到端追踪 | 否（缺则后端生成） |
| `X-User` | 操作员标识（V3.0 简化） | 写操作必需 |
| `Last-Event-ID` | SSE 重连 | SSE 重连必需 |

### 1.5 错误响应

所有 4xx / 5xx 返回相同结构：

```json
{
  "error_code": "DATA_TOO_SHORT",
  "error_message": "数据时长不足，至少需要 60 秒",
  "error_details": {
    "actual_seconds": 42,
    "required_seconds": 60
  },
  "trace_id": "trace-xxxx"
}
```

错误码命名：`<DOMAIN>_<DETAIL>`，全大写下划线分隔。完整列表见附录 B。

### 1.6 时间格式

- 请求 / 响应：ISO8601 UTC，例 `2026-05-08T13:25:32Z`；
- 前端按用户时区展示。

### 1.7 分页约定

```
GET /api/v3/<resource>?page=1&page_size=20&sort=-created_at
```

响应：

```json
{
  "items": [...],
  "page": 1,
  "page_size": 20,
  "total": 137,
  "has_more": true
}
```

`page_size` 默认 20，最大 100。`sort` 用 `-` 前缀降序，无前缀升序。

### 1.8 限流

- 默认 100 req/min/IP（可配）；
- 超限返回 429 + `error_code=RATE_LIMITED`；
- SSE 流不计入计数，但同 IP 同 session 只允许 1 个并发流。

---

## 2. 接口分组速查

| 分组 | 前缀 | 主要接口 |
|---|---|---|
| 诊断 | `/api/v3/diagnostic/` | start / stream / verdict |
| 整定 | `/api/v3/tuning/` | start / stream / result |
| 会话 | `/api/v3/sessions/` | list / get / resume / cancel / events |
| 告警 | `/api/v3/alarms/` | list / get / acknowledge / resolve / dismiss |
| 工程师确认 | `/api/v3/confirmations/` | list / get / respond / cancel |
| 经验 | `/api/v3/experience/` | list / get / search |
| 数据 | `/api/v3/data/` | upload / preview / list_loops |
| 顾问聊天 | `/api/v3/consult/` | stream |
| 历史数据接入 | `/api/v3/history/` | query |
| 配置 | `/api/v3/config/` | model_config / mcp_config / thresholds / feature_flags |
| 健康检查 | `/healthz` | （无 `/api/v3` 前缀） |

---

## 3. SSE 事件协议（统一）

### 3.1 帧格式

每条 SSE 帧：

```
event: <type>
id: <event_id>
data: <json payload>

```

- `id` 是单调递增字符串（`"<session_id>:<n>"`）；
- 每条 SSE 帧后空行结束。

### 3.2 事件 envelope

```yaml
type: stage | llm_thinking | tool_call | alarm_raised |
      engineer_confirmation_required | result | error | heartbeat
pipeline: "diagnostic" | "tuning" | "consultant" | null
stage: str | null
status: "running" | "done" | null
event_id: str
ts: iso8601
trace_id: str
session_id: str
payload: object
```

### 3.3 通用事件

#### 3.3.1 heartbeat

每 15 s 发一次，避免代理断流：

```json
{ "type": "heartbeat", "session_id": "...", "ts": "..." }
```

#### 3.3.2 error

```json
{
  "type": "error",
  "pipeline": "diagnostic",
  "stage": "compute_cpk",
  "error_code": "ONTOLOGY_FIELD_MISSING",
  "error_message": "...",
  "trace_id": "..."
}
```

发 error 事件后 SSE 流主动关闭。

#### 3.3.3 result

每个流水线最后一条事件，载入完整结果：

```json
{
  "type": "result",
  "pipeline": "diagnostic",
  "session_id": "...",
  "payload": { /* 见各流水线规约 */ }
}
```

### 3.4 重连

```
GET /api/v3/sessions/{id}/events
Header: Last-Event-ID: <最后收到的 event_id>
```

服务端：

1. 校验 session_id 与 Last-Event-ID 来自同一会话；
2. 从 `events.jsonl` 重放 `event_id > Last-Event-ID` 的事件（最多 100 条）；
3. 后续推实时事件（如果 session 仍 running）；
4. 如 session 已 completed/failed/cancelled，重放完直接关闭流。

---

## 4. 诊断接口

### 4.1 启动诊断

```
POST /api/v3/diagnostic/start
```

**请求体**：

```json
{
  "csv_path": "/uploads/2026-05-08/tic-10707.csv",
  "loop_id": "5203_TIC_10707",
  "loop_type": "temperature",
  "selected_loop_prefix": null,
  "user": "engineer_li"
}
```

**响应**：

```json
{
  "session_id": "ses-9b22",
  "trace_id": "trace-xxxx",
  "stream_url": "/api/v3/sessions/ses-9b22/events"
}
```

前端拿到 `stream_url` 后立即建立 EventSource 连接。

**错误**：

| 状态 | error_code | 触发 |
|---|---|---|
| 400 | INVALID_INPUT | 必填缺失 |
| 404 | CSV_NOT_FOUND | csv_path 不存在 |
| 409 | SESSION_ALREADY_RUNNING | 同一 loop_id 已有 running 会话（V3.0 单实例约束） |

### 4.2 SSE 事件序列

```
event: stage          # identify_condition running
event: stage          # identify_condition done
event: stage          # detect_open_loop running
event: stage          # detect_open_loop done
... (5 个节点)
event: stage          # summarize_diagnostic_verdict done
event: result         # 最终 verdict
```

挂起场景：

```
event: stage          # compute_cpk running
event: engineer_confirmation_required
(SSE 流关闭)
```

工程师 resume 后，通过 `/api/v3/sessions/{id}/events` 接续，从 compute_cpk done 开始。

### 4.3 stage 事件 payload（按节点）

详见文档 10 § 2.x.5。这里给一例完整：

```json
{
  "type": "stage",
  "pipeline": "diagnostic",
  "stage": "compute_cpk",
  "status": "done",
  "event_id": "ses-9b22:23",
  "ts": "2026-05-08T13:25:30Z",
  "trace_id": "trace-xxxx",
  "session_id": "ses-9b22",
  "payload": {
    "cpk": 0.71,
    "cp": 0.95,
    "mu": 145.0,
    "sigma": 14.0,
    "within_limits_pct": 96.4,
    "threshold_used": 1.0,
    "recommend_tuning": true,
    "evidence": [
      { "fact": "USL", "value": 200, "source": "ontology", "path": "loop.pv.usl" },
      { "fact": "LSL", "value": 50,  "source": "ontology", "path": "loop.pv.lsl" }
    ]
  }
}
```

### 4.4 alarm_raised payload

```json
{
  "type": "alarm_raised",
  "pipeline": "diagnostic",
  "stage": "detect_open_loop",
  "session_id": "ses-9b22",
  "payload": {
    "alarm": {
      "id": "alarm-3f2a",
      "loop_id": "FIC-105",
      "type": "open_loop",
      "severity": "high",
      "blocking": true,
      "evidence": { ... },
      "recommended_action": "通知现场仪表工检查控制器是否在自动模式"
    }
  }
}
```

### 4.5 engineer_confirmation_required payload

```json
{
  "type": "engineer_confirmation_required",
  "pipeline": "diagnostic",
  "stage": "compute_cpk",
  "session_id": "ses-9b22",
  "payload": {
    "confirmation": {
      "id": "conf-1234",
      "type": "missing_pv_limits",
      "prompt": "...",
      "fields": [...],
      "evidence": {...},
      "expires_at": "2026-05-09T13:25:32Z"
    }
  }
}
```

### 4.6 result payload（DiagnosticVerdict）

```json
{
  "type": "result",
  "pipeline": "diagnostic",
  "session_id": "ses-9b22",
  "payload": {
    "verdict": {
      "verdict_id": "ver-77a1",
      "session_id": "ses-9b22",
      "loop_id": "5203_TIC_10707",
      "status": "needs_tuning",
      "recommend_tuning": true,
      "primary_reason": "...",
      "blockers": [],
      "confirmations": [],
      "risk_flags": [],
      "evidence": { ... },
      "recommended_tuning_hints": { "apply_filter": false }
    },
    "duration_ms": 4350
  }
}
```

### 4.7 查询历史诊断

```
GET /api/v3/diagnostic/verdicts/{verdict_id}
GET /api/v3/diagnostic/verdicts?loop_id=&status=&page=&page_size=
```

---

## 5. 整定接口

### 5.1 启动整定

```
POST /api/v3/tuning/start
```

**请求体**：

```json
{
  "csv_path": "/uploads/...",
  "loop_id": "FIC-105",
  "loop_type": "flow",
  "selected_loop_prefix": null,
  "selected_window_index": null,
  "use_llm_advisor": true,
  "stop_after": null,
  "algorithm_filter": null,
  "diagnostic_verdict_id": "ver-77a1",
  "force_reason": null,
  "user": "engineer_li"
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 默认 | 说明 |
|---|---|---|---|---|
| `diagnostic_verdict_id` | str | 否 | null | 关联诊断；缺省时不联动 |
| `stop_after` | enum | 否 | null | `"window_selection"` / `"identification"` |
| `force_reason` | str | 否 | null | 当 verdict.status != "needs_tuning" 时强制启动需填理由 |

**前置校验**：

- 若 `diagnostic_verdict_id` 提供且对应 verdict.status ∈ {blocked, awaiting_confirmation} → 返回 403 `BLOCKED_BY_DIAGNOSTIC`；
- 若 verdict.status == healthy 且 `force_reason` 为空 → 返回 403 `HEALTHY_LOOP_REQUIRES_FORCE`；
- 若 `force_reason` 非空 → 落审计 `var/audit/force_actions.jsonl`。

**响应**：

```json
{
  "session_id": "ses-tune-1",
  "trace_id": "...",
  "stream_url": "/api/v3/sessions/ses-tune-1/events"
}
```

### 5.2 SSE 事件序列

```
event: stage    # data_analysis running/done
event: stage    # ontology_policy running/done
event: stage    # window_selection running/done   (含 llm_thinking 子事件)
event: stage    # identification running/done    (round 0)
event: stage    # model_review running/done       (verdict)
event: stage    # identification_refinement done  (如 downgrade)
event: stage    # identification running/done    (round 1)
... (最多 3 round)
event: stage    # tuning running/done
event: stage    # evaluation running/done
event: result
```

### 5.3 result payload（TuningResult 完整视图）

```json
{
  "type": "result",
  "pipeline": "tuning",
  "session_id": "ses-tune-1",
  "payload": {
    "data_analysis": { "data_points": 9876, "sampling_time": 1.0, "candidate_windows": [...] },
    "window_selection": { "mode": "llm", "chosen_index": 2, "...": "..." },
    "model": {
      "model_type": "FOPDT",
      "K": 0.95, "T": 8.1, "L": 1.6,
      "r2_score": 0.92, "confidence": 0.78,
      "attempts": [...],
      "algorithm_comparison": [...]
    },
    "pid_params": {
      "Kp": 1.5, "Ki": 0.18, "Kd": 0.0,
      "Ti": 8.1, "Td": 0.0,
      "strategy": "IMC",
      "candidates": [...]
    },
    "evaluation": {
      "passed": true,
      "performance_score": 78.5,
      "final_rating": 4.0,
      "overshoot_percent": 8.2,
      "settle_time_s": 24.3,
      "time_series": { "t": [...], "sp": [...], "pv": [...], "mv": [...] },
      "reality_check": { "issues": [] }
    },
    "model_review": { "verdict": "accept", "reason": "...", "concerns": [] },
    "loop_type": "flow",
    "loop_name": "FIC-105"
  }
}
```

### 5.4 早停（stop_after）

| stop_after | 跑到哪 | result payload 形态 |
|---|---|---|
| `null` | 完整流程 | 完整 |
| `"window_selection"` | 选窗结束 | model/pid_params/evaluation = null |
| `"identification"` | 辨识 + 评审 + 精修结束 | pid_params/evaluation = null |

---

## 6. 会话接口

### 6.1 列表 / 详情

```
GET /api/v3/sessions?pipeline=&loop_id=&status=&page=
GET /api/v3/sessions/{id}
```

返回 SessionMeta + SessionState（不返回完整 events.jsonl）。

### 6.2 事件流（SSE）

```
GET /api/v3/sessions/{id}/events
Header: Last-Event-ID: <可选>
```

行为见 § 3.4。

### 6.3 重入（resume）

```
POST /api/v3/sessions/{id}/resume
```

**请求体**：

```json
{
  "user": "engineer_li"
}
```

**前置校验**：

- session.status 必须 = `awaiting_confirmation`；
- 对应 confirmation.status 必须 ∈ {responded}；
- input_hash 校验通过。

**响应**：

```json
{
  "session_id": "ses-9b22",
  "stream_url": "/api/v3/sessions/ses-9b22/events"
}
```

前端建立新 SSE 连接接续。

**错误**：

| 状态 | error_code |
|---|---|
| 409 | SESSION_NOT_AWAITING |
| 409 | CONFIRMATION_NOT_RESPONDED |
| 409 | INPUT_HASH_MISMATCH |
| 410 | CONFIRMATION_EXPIRED |

### 6.4 取消

```
POST /api/v3/sessions/{id}/cancel
body: { "user": "...", "reason": "..." }
```

session.status → `cancelled`，未完成的 confirmation 一并 cancel。

### 6.5 强制重启（不常用）

```
POST /api/v3/sessions/{id}/restart
body: { "user": "...", "reason": "..." }
```

仅用于 `failed` 或 `expired` 会话；从头再跑。

---

## 7. 告警接口

### 7.1 列表

```
GET /api/v3/alarms?loop_id=&status=&type=&severity=&from=&to=&page=&page_size=
```

支持过滤组合；默认按 `created_at` 倒序。

### 7.2 详情

```
GET /api/v3/alarms/{id}
```

### 7.3 状态迁移

```
POST /api/v3/alarms/{id}/acknowledge
body: { "ack_user": "..." }

POST /api/v3/alarms/{id}/resolve
body: { "resolution_note": "...", "user": "..." }

POST /api/v3/alarms/{id}/dismiss
body: { "reason": "false_positive|accepted_risk|other", "resolution_note": "...", "user": "..." }
```

非法迁移返回 409 + `error_code=INVALID_TRANSITION`。

### 7.4 SSE 推送

告警中心需要订阅"全局新告警"：

```
GET /api/v3/alarms/stream
```

服务端推：

```json
{ "type": "alarm_raised", "payload": { "alarm": { ... } } }
{ "type": "alarm_state_changed", "payload": { "alarm_id": "...", "from": "new", "to": "acknowledged" } }
```

---

## 8. 工程师确认接口

### 8.1 列表 / 详情

```
GET /api/v3/confirmations?status=pending&loop_id=&pipeline=&page=
GET /api/v3/confirmations/{id}
```

待办列表用 `status=pending`；前端工程师面板默认查这个。

### 8.2 回应

```
POST /api/v3/confirmations/{id}/respond
```

**请求体**：

```json
{
  "response": {
    "USL": 200,
    "LSL": 50,
    "writeback_to_ontology": true
  },
  "response_user": "engineer_li"
}
```

**校验**：

- confirmation.status = pending；
- response 字段满足 `fields` 定义（必填、类型、min/max、enum_values）；
- 写回字段触发 OntologyWriteback 落 `var/ontology_writeback/`。

**响应**：

```json
{
  "confirmation_id": "conf-1234",
  "status": "responded",
  "session_resume_url": "/api/v3/sessions/ses-9b22/resume"
}
```

前端拿到后立即调 resume 重启流水线。

### 8.3 取消

```
POST /api/v3/confirmations/{id}/cancel
body: { "reason": "...", "user": "..." }
```

同时 cancel 相关 session。

### 8.4 SSE 推送（待办通知）

```
GET /api/v3/confirmations/stream
```

```json
{ "type": "confirmation_created", "payload": { "confirmation": { ... } } }
{ "type": "confirmation_responded", "payload": { "confirmation_id": "..." } }
```

---

## 9. 数据接口

### 9.1 上传 CSV

```
POST /api/v3/data/upload
Content-Type: multipart/form-data
fields: file (csv), loop_hint (str, optional)
```

**响应**：

```json
{
  "csv_path": "/uploads/2026-05-08/abc-123.csv",
  "preview": {
    "columns": ["time", "PV", "MV", "SV"],
    "rows": 9876,
    "sample_time_s": 1.0,
    "first_rows": [...]    // 5 行
  },
  "loops": [
    { "loop_prefix": "TIC-10707", "pv": "TIC-10707.PV", "mv": "TIC-10707.OUT", "sv": "TIC-10707.SP" },
    ...
  ]
}
```

### 9.2 列出回路

```
GET /api/v3/data/loops?csv_path=...
```

### 9.3 数据预览

```
GET /api/v3/data/preview?csv_path=...&loop_prefix=...&start=0&end=500
```

返回时序片段，给前端原始数据图用。

---

## 10. 顾问聊天接口

### 10.1 流式对话

```
POST /api/v3/consult/stream
```

**请求体**：

```json
{
  "messages": [
    { "role": "user", "content": "TIC-10707 的整定结果是不是太保守了？" }
  ],
  "session": {
    "csv_path": "...",
    "loop_id": "TIC-10707",
    "loop_type": "temperature",
    "model_type": "FOPDT",
    "model_K": 0.95, "model_T": 8.1, "model_L": 1.6,
    "Kp": 1.5, "Ki": 0.18, "Kd": 0.0
  },
  "max_iterations": 8
}
```

**响应**：SSE 流

```
event: text_chunk     # 流式文本
event: tool_call      # LLM 调工具
event: tool_call      # ...
event: text_chunk     # 综合回复
event: final          # 完成
```

### 10.2 工具调用 payload

```json
{
  "type": "tool_call",
  "payload": {
    "name": "run_tuning",
    "args": { "strategy": "LAMBDA", "lambda_factor": 2.0 },
    "result": { "Kp": 1.1, "Ki": 0.13, "Kd": 0.0, "strategy": "LAMBDA" }
  }
}
```

### 10.3 final

```json
{ "type": "final", "payload": { "content": "...", "iterations": 3 } }
```

---

## 11. 经验接口

### 11.1 列表

```
GET /api/v3/experience?loop_id=&loop_type=&condition_id=&page=
```

### 11.2 详情

```
GET /api/v3/experience/{id}
```

### 11.3 检索

```
POST /api/v3/experience/search
body: { "query": "TIC-10707 高负荷", "top_k": 10 }
```

V3.0 实现：关键词 + 倒排索引匹配；V3.x 引入向量。

### 11.4 反向标注（V3.x）

预留：`PUT /api/v3/experience/{id}` 修改 tags / operator_note / 标记过期。V3.0 不支持。

---

## 12. 历史数据接入接口

### 12.1 查询

```
POST /api/v3/history/query
body: {
  "loop_id": "TIC-10707",
  "from": "2026-05-01T00:00:00Z",
  "to":   "2026-05-08T00:00:00Z",
  "tags": ["PV","MV","SV"]
}
```

**响应**：CSV 文件路径（保存在 uploads/）+ 预览。

V3.0 仅适配 Hollysys；其他系统由 `services/history_*.py` Adapter 扩展。

---

## 13. 配置接口

### 13.1 LLM 模型配置

```
GET  /api/v3/config/model
PUT  /api/v3/config/model
body: { "model_name": "deepseek-reasoner", "model_api_url": "...", "model_api_key": "..." }
```

PUT 后立即热生效。

### 13.2 MCP 配置

```
GET  /api/v3/config/mcp
PUT  /api/v3/config/mcp
body: { "servers": [ { "name": "ontology", "transport": "stdio", ... } ] }
POST /api/v3/config/mcp/{server}/test     # 测试连通性
```

### 13.3 阈值配置

```
GET  /api/v3/config/thresholds
PUT  /api/v3/config/thresholds
body: { "diagnostic.cpk.threshold": 1.0, ... }
```

支持点路径键，深度合并。

### 13.4 功能开关

```
GET  /api/v3/config/feature_flags
PUT  /api/v3/config/feature_flags
body: { "llm.enabled": true, "diagnostic.harris.enabled": true }
```

V3.0 关键开关：

| key | 默认 | 用途 |
|---|---|---|
| `llm.enabled` | true | 总开关；关闭后所有 advisor 走确定性 |
| `llm.advisor.window_selection.enabled` | true | 单个决策点开关 |
| `llm.advisor.identification_review.enabled` | true | |
| `llm.advisor.identification_refinement.enabled` | true | |
| `llm.advisor.window_policy.enabled` | true | |
| `llm.consultant.enabled` | true | 关闭后聊天接口返回 503 |
| `diagnostic.<node>.enabled` | true | 单节点开关 |
| `auth.enabled` | false | Basic Auth |

### 13.5 审计

所有 PUT 写入 `var/audit/config_changes.jsonl`：

```json
{ "ts": "...", "user": "...", "key": "diagnostic.cpk.threshold", "from": 1.0, "to": 0.9 }
```

---

## 14. 健康检查

```
GET /healthz
```

```json
{
  "ok": true,
  "version": "v3.0.0-rc1",
  "uptime_s": 3600,
  "dependencies": {
    "llm": "ok",
    "mcp": { "ontology": "ok", "ontology_legacy": "disabled" },
    "var_writable": true
  }
}
```

`ok=true` 即使依赖 degraded，因为系统设计为可降级。`ok=false` 仅当本地 var/ 不可写或核心模块崩溃。

---

## 15. OpenAPI / 文档生成

V3.0 用 FastAPI 自动生成 `/api/v3/openapi.json` + `/api/v3/docs` (Swagger UI)。

约束：

- 每个路由必须有 `summary` + `description`（中文）；
- 每个 Pydantic 模型字段必须有 `description`；
- 错误响应在每个路由用 `responses={409: {...}}` 标注。

---

## 16. 跨接口编排示例（端到端）

### 16.1 完整诊断 → 整定 流程

```
1. 上传数据
   POST /api/v3/data/upload  → csv_path, loops

2. 启动诊断
   POST /api/v3/diagnostic/start
        body: { csv_path, loop_id }
        → session_id, stream_url

3. 订阅 SSE
   GET /api/v3/sessions/{id}/events
   接收：stage * 6 → result

4. （如挂起）补充 confirmation
   POST /api/v3/confirmations/{id}/respond
        body: { response, response_user }
   POST /api/v3/sessions/{id}/resume
   订阅新 SSE 流

5. 拿到 verdict.status = "needs_tuning"
   POST /api/v3/tuning/start
        body: { csv_path, loop_id, diagnostic_verdict_id }
        → session_id, stream_url

6. 订阅 SSE 接收整定全过程

7. 整定结果落库 + 经验沉淀
   GET /api/v3/experience?loop_id=...   后续可见
```

### 16.2 用户在 Consultant 中要求微调

```
1. 整定完成后，前端把 SessionContext + 当前 PID 一起塞进 consult/stream

2. 用户："Ki 太大，再保守一点"

3. Consultant LLM 调 tool run_tuning(ki_scale=0.7)
   后端跑 select_best_strategy → 新 PID

4. 用户接受 → 前端调 /api/v3/tuning/sessions/{id}/override 落审计（V3.x）
   V3.0 用户自己抄数到 DCS
```

---

## 17. 跨域与代理

### 17.1 CORS

V3.0 默认仅同源；如需跨域，配置：

```yaml
api.cors.allowed_origins:
  - https://pid.example.com
  - http://localhost:5173
```

仅允许 GET / POST / PUT / DELETE，不允许通配 `*`。

### 17.2 反向代理（生产）

```
nginx → uvicorn:8000
       │
       ├── /api/v3/...  → backend
       ├── /healthz     → backend
       └── /            → frontend dist/
```

代理需要保留 `text/event-stream` 不缓冲：

```
proxy_buffering off;
proxy_read_timeout 3600;
```

---

## 18. 错误码全集（附录 A 提取）

> 附录 B 给完整列表。这里列高频。

| code | HTTP | 含义 |
|---|---|---|
| `INVALID_INPUT` | 400 | 入参校验失败 |
| `CSV_NOT_FOUND` | 404 | 文件不存在 |
| `SESSION_NOT_FOUND` | 404 | session_id 不存在 |
| `SESSION_NOT_AWAITING` | 409 | resume 时 session 状态不对 |
| `CONFIRMATION_NOT_RESPONDED` | 409 | resume 时 confirmation 没 respond |
| `CONFIRMATION_EXPIRED` | 410 | 过期 |
| `INPUT_HASH_MISMATCH` | 409 | csv 已变更，无法 resume |
| `BLOCKED_BY_DIAGNOSTIC` | 403 | 诊断 blocked 时启动整定 |
| `HEALTHY_LOOP_REQUIRES_FORCE` | 403 | 健康回路启动整定缺 force_reason |
| `INVALID_TRANSITION` | 409 | 状态机非法迁移 |
| `RATE_LIMITED` | 429 | 限流 |
| `LLM_UNAVAILABLE` | 503 | LLM 总开关关闭时聊天接口 |
| `DATA_TOO_SHORT` | 400 | 数据时长不足 |
| `ONTOLOGY_FIELD_MISSING` | 400 | 非可补充的字段缺失 |

---

## 19. 测试策略

### 19.1 契约测试

每个接口配一个 fixture：

- 正常请求 → 200 + schema 校验；
- 非法请求 → 4xx + `error_code` 命中预期；
- SSE 流：mock pipeline runner，校验事件顺序与 payload schema。

### 19.2 端到端测试

按 § 16.1 跑完整流程，断言：

- session.status 终态正确；
- alarms / confirmations / experience 正确落库；
- audit jsonl 含预期记录。

### 19.3 兼容性测试（V3.x）

如有 v3 → v4 升级，需保证 v3 客户端能在 v4 下跑通核心流程或得到明确错误。

---

## 20. 演进与未决问题

### 20.1 演进

| 触发 | 调整 |
|---|---|
| 多回路并行 | 增加 `/api/v3/loops/batch/start` |
| 多租户 | 路由前缀 `/api/v3/tenants/{id}/...` |
| GraphQL 需求 | 评估单独 `/graphql`，不替代 REST |

### 20.2 未决

| ID | 问题 | 处理 |
|---|---|---|
| Q-API1 | resume 是否能由后端自动触发（worker 检测 confirmation.responded）？ | M2 评审 |
| Q-API2 | force_reason 是否需要二次确认（管理员审批）？ | M3 |
| Q-API3 | 经验 search 是否要支持复杂 filter（DSL）？ | V3.x |
| Q-API4 | SSE 是否要支持 push 回前端进度百分比？ | M3 |
| Q-API5 | OpenAPI schema 是否要 ship 给本体侧（机器可读契约）？ | M3 |

---

## 附录 A：路由全集速查

```
POST   /api/v3/data/upload
GET    /api/v3/data/loops
GET    /api/v3/data/preview

POST   /api/v3/diagnostic/start
GET    /api/v3/diagnostic/verdicts
GET    /api/v3/diagnostic/verdicts/{id}

POST   /api/v3/tuning/start

GET    /api/v3/sessions
GET    /api/v3/sessions/{id}
GET    /api/v3/sessions/{id}/events
POST   /api/v3/sessions/{id}/resume
POST   /api/v3/sessions/{id}/cancel
POST   /api/v3/sessions/{id}/restart

GET    /api/v3/alarms
GET    /api/v3/alarms/{id}
POST   /api/v3/alarms/{id}/acknowledge
POST   /api/v3/alarms/{id}/resolve
POST   /api/v3/alarms/{id}/dismiss
GET    /api/v3/alarms/stream

GET    /api/v3/confirmations
GET    /api/v3/confirmations/{id}
POST   /api/v3/confirmations/{id}/respond
POST   /api/v3/confirmations/{id}/cancel
GET    /api/v3/confirmations/stream

GET    /api/v3/experience
GET    /api/v3/experience/{id}
POST   /api/v3/experience/search

POST   /api/v3/consult/stream

POST   /api/v3/history/query

GET    /api/v3/config/model       PUT
GET    /api/v3/config/mcp         PUT
POST   /api/v3/config/mcp/{server}/test
GET    /api/v3/config/thresholds  PUT
GET    /api/v3/config/feature_flags PUT

GET    /healthz
GET    /api/v3/openapi.json
GET    /api/v3/docs
```

## 附录 B：错误码全集（V3.0）

```yaml
# 通用
INVALID_INPUT                    # 400
RATE_LIMITED                     # 429
INTERNAL_ERROR                   # 500
SERVICE_UNAVAILABLE              # 503
NOT_FOUND                        # 404
INVALID_TRANSITION               # 409

# 数据
CSV_NOT_FOUND                    # 404
DATA_TOO_SHORT                   # 400
DATA_PARSE_ERROR                 # 400
LOOP_NOT_FOUND_IN_CSV            # 404

# Session
SESSION_NOT_FOUND                # 404
SESSION_ALREADY_RUNNING          # 409
SESSION_NOT_AWAITING             # 409
SESSION_ALREADY_COMPLETED        # 409
INPUT_HASH_MISMATCH              # 409

# Confirmation
CONFIRMATION_NOT_FOUND           # 404
CONFIRMATION_NOT_RESPONDED       # 409
CONFIRMATION_EXPIRED             # 410
CONFIRMATION_FIELD_INVALID       # 400

# Alarm
ALARM_NOT_FOUND                  # 404
ALARM_INVALID_TRANSITION         # 409

# Diagnostic / Tuning
BLOCKED_BY_DIAGNOSTIC            # 403
HEALTHY_LOOP_REQUIRES_FORCE      # 403
LOW_CONFIDENCE                   # 400 (LLM 关闭时辨识置信度过低)
ID_ERROR                         # 400 (辨识全失败)
NO_CANDIDATE_WINDOW              # 400

# Ontology
ONTOLOGY_FIELD_MISSING           # 400
ONTOLOGY_INVALID_DATA            # 400
ONTOLOGY_UNREACHABLE             # 503

# LLM
LLM_UNAVAILABLE                  # 503
LLM_TIMEOUT                      # 504
LLM_PARSE_ERROR                  # 502

# Auth (V3.x)
UNAUTHORIZED                     # 401
FORBIDDEN                        # 403
```

## 附录 C：前端调用示例（伪代码）

```ts
// 1. 启动诊断
const { session_id, stream_url } = await api.post("/diagnostic/start", { csv_path, loop_id });

// 2. 订阅 SSE
const es = new EventSource(stream_url);
es.addEventListener("stage", e => store.appendStage(JSON.parse(e.data)));
es.addEventListener("alarm_raised", e => store.appendAlarm(JSON.parse(e.data)));
es.addEventListener("engineer_confirmation_required", e => {
  const { confirmation } = JSON.parse(e.data).payload;
  store.openConfirmationModal(confirmation);
  es.close();   // SSE 流已被服务端关闭
});
es.addEventListener("result", e => store.setVerdict(JSON.parse(e.data).payload.verdict));

// 3. 工程师补充 + 重入
await api.post(`/confirmations/${confId}/respond`, { response, response_user });
const { stream_url: newUrl } = await api.post(`/sessions/${session_id}/resume`, { user });
const es2 = new EventSource(newUrl);
// ... 继续订阅
```

## 附录 D：跨文档一致性 checklist

- [ ] 03 Pydantic 模型字段 ↔ 本文 § 4-13 schema
- [ ] 10 决策树事件 ↔ 本文 § 4 SSE 事件
- [ ] 09 OntologyWriteback 触发 ↔ 本文 § 8.2 confirm respond
- [ ] 06 Consultant 协议 ↔ 本文 § 10
- [ ] 02 SSE envelope ↔ 本文 § 3
