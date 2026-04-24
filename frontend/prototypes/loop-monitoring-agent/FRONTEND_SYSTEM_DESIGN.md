# PID 智能整定驾驶舱前端系统设计

## 1. 产品定位

目标不是“上传一份 CSV 后跑整定”，而是面向现场工程师、工艺工程师、仪控工程师的监控型 Agent 工作台。

系统应该持续回答：

- 当前哪些回路不健康？
- 当前是什么工况？
- 问题是否真的来自 PID 参数？
- 是否需要整定？
- 是否适合现在整定？
- 如果不能整定，先处理什么？
- 如果可以整定，推荐哪些参数、风险是什么、是否允许下发？

因此前端主流程是：

```text
选择装置/回路
  ↓
查看健康状态与异常排名
  ↓
进入单回路画像
  ↓
Agent 评估工况、性能、数据质量、诊断原因
  ↓
决策是否进入辨识/整定
  ↓
生成建议动作或整定方案
  ↓
人工确认、下发、回写案例
```

## 2. 顶部主菜单

建议顶部一级菜单如下：

| 主菜单 | 目标用户问题 | 典型页面 |
|---|---|---|
| 工作台 | 今天哪些回路需要处理？ | 总览驾驶舱、待处理回路、本班任务、风险预警 |
| 回路监控 | 当前回路是否稳定？ | 全局回路看板、单回路画像、趋势与频谱、报警与事件 |
| 回路评估 | 回路运行质量如何？ | 性能评分、数据质量、工况识别、整定准备度 |
| 根因诊断 | 为什么控制不好？ | 诊断总览、阀门诊断、振荡诊断、模型可靠性 |
| 整定中心 | 如何生成和确认 PID 参数？ | 整定任务、窗口与辨识、参数候选、下发确认 |
| 经验中心 | 历史经验如何沉淀复用？ | 整定案例库、规则库、知识图谱、模型版本 |
| 系统设置 | 数据源、规则、权限如何配置？ | 数据源配置、回路主数据、规则配置、角色权限 |

## 3. 工作台

### 3.1 总览驾驶舱

用途：全厂或装置级入口。优先让用户看到异常回路、建议整定回路、不能整定但需要处理的回路。

核心组件：

- KPI 卡片：总回路数、健康运行、需要关注、建议整定、暂不建议整定、待确认下发。
- 回路健康矩阵：按装置/回路类型分组，颜色表示健康度。
- TOP 异常列表：按风险分、收益分、整定准备度排序。
- Agent 运行状态：当前正在分析的回路、最近一次分析时间、失败任务。
- 本班建议动作：继续观察、建议整定、建议做阶跃测试、建议检修阀门。

字段：

| 字段 | 含义 |
|---|---|
| loop_id | 回路唯一标识 |
| loop_name | 回路名称 |
| unit | 装置/单元 |
| loop_type | flow/pressure/temperature/level |
| health_score | 健康度 0-100 |
| loop_state | stable/oscillating/sluggish/noisy/saturated/disturbed |
| operating_condition | steady/load_change/startup/shutdown/grade_change |
| need_tuning | 是否建议整定 |
| tuning_readiness | ready/need_more_data/not_recommended |
| risk_level | low/medium/high/critical |
| expected_benefit | 预计收益或改善潜力 |
| last_analyzed_at | 最近分析时间 |

数据源：

- 回路主数据/点表。
- 实时或历史时序库。
- PID 参数库。
- 报警事件库。
- Agent 任务记录。

后端能力：

- `list_loop_health_summary`
- `compute_loop_health_indices`
- `diagnose_loop_state`
- `decide_tuning_need`

### 3.2 待处理回路

用途：形成工程师待办队列。

筛选项：

- 装置。
- 回路类型。
- 风险等级。
- 是否建议整定。
- 是否可自动辨识。
- 是否需要检修/操作确认。

表格字段：

| 字段 | 含义 |
|---|---|
| priority | 优先级 |
| loop_id | 回路 |
| diagnosis_summary | 诊断摘要 |
| evidence_tags | 证据标签，如 振荡/饱和/噪声/外扰 |
| readiness | 整定准备度 |
| next_action | 下一步建议 |
| owner | 责任人 |
| status | open/in_progress/done/deferred |

### 3.3 本班任务

用途：管理正在运行或需要人工确认的 Agent 任务。

任务类型：

- 监控分析任务。
- 诊断任务。
- 整定建议任务。
- 参数下发审批任务。
- 回写案例任务。

字段：

| 字段 | 含义 |
|---|---|
| task_id | 任务 ID |
| loop_id | 回路 |
| task_type | monitor/diagnose/tune/approval |
| status | running/succeeded/failed/waiting_approval |
| started_at | 开始时间 |
| duration_s | 耗时 |
| agent_version | Agent 版本 |
| result_summary | 结果摘要 |

## 4. 回路监控

### 4.1 全局回路看板

用途：按装置、类型、状态看所有回路。

视图：

- 卡片视图：每个回路一张卡。
- 矩阵视图：装置 × 回路类型。
- 趋势排行：健康度下降最快、振荡最强、MV 最活跃。

卡片字段：

| 字段 | 含义 |
|---|---|
| loop_id | 回路编号 |
| tag_pv/tag_sv/tag_mv | PV/SV/MV 位号 |
| mode_auto_pct | 自动模式比例 |
| oscillation_index | 振荡指数 |
| tracking_error | 跟踪误差 |
| mv_travel | MV 行程 |
| saturation_pct | 输出饱和比例 |
| noise_ratio | 噪声比 |

### 4.2 单回路画像

用途：单个回路的全量上下文。

页面区块：

- 基本信息：位号、设备、控制对象、单位、量程、回路类型。
- 当前 PID：Kp/Ki/Kd、PB/Ti/Td、正反作用、控制周期、输出上下限。
- 当前状态：模式、SP/PV/MV、阀位、报警、工况。
- 历史趋势：PV/SV/MV、阀位、负荷、上下游变量。
- Agent 结论：状态、诊断、是否建议整定。
- 相关案例：同类回路、历史整定记录。

需要连接的数据源：

- DCS 点表。
- 实时值接口。
- 历史时序库。
- PID 参数接口。
- 设备主数据。
- 工况标签库。

### 4.3 趋势与频谱

用途：解释振荡和动态特征。

组件：

- 多变量趋势图。
- 频谱图。
- 自相关/互相关图。
- MV-PV 滞后估计。
- 主周期卡片。
- 正反作用判断证据。

字段：

| 字段 | 含义 |
|---|---|
| dominant_period_s | 主振荡周期 |
| dominant_frequency | 主频 |
| pv_mv_corr | PV/MV 相关性 |
| best_lag_s | 最佳滞后 |
| direction_sign | 正/负响应方向 |
| spectral_energy | 主频能量占比 |

## 5. 回路评估

### 5.1 性能评分

用途：评价当前控制效果，不等同于辨识。

指标：

- IAE/ISE/ITAE。
- PV variance。
- SP tracking error。
- MV travel。
- MV reversal count。
- Harris index。
- oscillation index。
- saturation ratio。
- automatic mode ratio。

输出：

```json
{
  "performance_score": 0-100,
  "stability_score": 0-100,
  "tracking_score": 0-100,
  "actuator_score": 0-100,
  "economic_score": 0-100,
  "findings": []
}
```

### 5.2 数据质量

用途：判断历史数据能不能支撑评估和辨识。

字段：

- n_points。
- dt_median。
- dt_jitter。
- missing_ratio。
- duplicate_ratio。
- outlier_ratio。
- pv_flatline_ratio。
- mv_flatline_ratio。
- pv_span。
- mv_span。
- noise_ratio。

结论：

- `good`：可以评估和辨识。
- `warning`：可以评估，但辨识需谨慎。
- `bad`：不建议自动整定。

### 5.3 工况识别

用途：避免在开停车、负荷变化、工况切换时误判 PID。

工况类型：

- steady。
- load_ramp_up。
- load_ramp_down。
- grade_change。
- startup。
- shutdown。
- disturbance。
- unknown。

数据源：

- 负荷变量。
- 关键操作变量。
- 工况标签。
- 班报/操作事件。

### 5.4 整定准备度

用途：判断是否可以进入辨识和参数建议。

门禁：

- 自动模式比例是否足够。
- 当前工况是否稳定。
- MV 是否有足够激励。
- PV 是否有响应。
- MV 是否饱和。
- 阀门/仪表问题是否阻断。
- 安全约束是否允许建议参数。

输出：

```json
{
  "readiness": "ready|need_more_data|not_recommended",
  "score": 0-100,
  "gates": [
    {"name": "data_quality", "status": "pass|warn|block", "reason": "..."}
  ],
  "blocking_reasons": [],
  "allowed_next_steps": []
}
```

## 6. 根因诊断

### 6.1 诊断总览

诊断类别：

- PID 参数不佳。
- 阀门卡涩/死区/饱和。
- 仪表噪声/漂移/冻结。
- 外部扰动。
- 工况切换。
- 回路耦合。
- 控制结构不合理。
- 数据不足。

字段：

| 字段 | 含义 |
|---|---|
| cause_code | 原因编码 |
| cause_name | 原因名称 |
| confidence | 置信度 |
| severity | 严重程度 |
| evidence | 证据 |
| recommended_action | 建议动作 |

### 6.2 阀门诊断

指标：

- mv_saturation_pct。
- mv_reversal_count。
- mv_travel。
- stick_slip_score。
- deadband_score。
- valve_response_delay。

需要数据：

- MV。
- 阀位反馈。
- PV。
- 输出上下限。
- 阀门类型/量程。

### 6.3 振荡诊断

判断：

- 持续振荡还是偶发扰动。
- MV 驱动 PV，还是 PV/扰动驱动 MV。
- 是否多回路同频。
- 是否控制器过激。

需要数据：

- 本回路 PV/SV/MV。
- 上下游关联变量。
- 同装置其他回路 PV/MV。

### 6.4 模型可靠性

用途：解释辨识模型能不能用于整定。

字段：

- model_type。
- K/T/L/T1/T2/zeta。
- r2_score。
- normalized_rmse。
- confidence。
- fit_score。
- selected_window。
- lower_bound_hit。
- residual_autocorrelation。
- parameter_stability。
- competing_models。

## 7. 整定中心

### 7.1 整定任务

任务来源：

- Agent 建议。
- 用户手动发起。
- 定时巡检触发。
- 批量回路评估触发。

字段：

- task_id。
- loop_id。
- source。
- status。
- current_stage。
- started_by。
- use_llm_advisor。
- approval_status。

### 7.2 窗口与辨识

页面内容：

- 候选窗口列表。
- 每个窗口的 MV/PV 变化、score、usable、工况标签。
- 每个模型的拟合结果。
- 多轮辨识 attempts。
- LLM/refinement 建议。
- 模型可靠性诊断。

### 7.3 参数候选

字段：

- strategy。
- Kp/Ki/Kd。
- PB/Ti/Td。
- constraints_status。
- expected_performance。
- robustness_score。
- recommendation_reason。

### 7.4 下发确认

注意：下发属于高风险动作，前端必须做人工确认。

内容：

- 当前参数 vs 新参数。
- 参数变化比例。
- 风险提示。
- 适用工况。
- 回滚参数。
- 审批人。
- 下发状态。
- DCS 写入结果。

## 8. 经验中心

### 8.1 整定案例库

字段：

- case_id。
- loop_id。
- before_pid。
- after_pid。
- before_performance。
- after_performance。
- model。
- operating_condition。
- diagnosis。
- approval_record。
- effectiveness_after_24h/7d。

### 8.2 规则库

规则类型：

- 回路类型先验。
- T/K/L 合理范围。
- PB/Ti/Td 约束。
- 评分封顶规则。
- 禁止自动建议场景。
- 安全下发约束。

### 8.3 知识图谱

实体：

- 装置。
- 设备。
- 回路。
- 位号。
- 变量。
- 阀门。
- 控制对象。
- 工况。
- 物料。

关系：

- 控制。
- 影响。
- 上下游。
- 同设备。
- 同扰动源。
- 同安全约束。

## 9. 系统设置

### 9.1 数据源配置

数据源：

- 时序数据库。
- DCS 当前参数接口。
- 报警事件库。
- 操作日志。
- 工况标签库。
- 设备主数据。
- 案例库。
- LLM 配置。

字段：

- source_name。
- source_type。
- endpoint。
- auth_type。
- sampling_policy。
- retention。
- status。

### 9.2 回路主数据

字段：

- loop_id。
- loop_name。
- loop_type。
- pv_tag。
- sv_tag。
- mv_tag。
- valve_tag。
- unit。
- equipment。
- process_object。
- engineering_unit。
- range_low/range_high。
- action_direction。
- safety_constraints。

## 10. 推荐前端路由

```text
/monitor/workbench
/monitor/todos
/monitor/tasks
/monitor/alerts

/loops
/loops/:loopId/profile
/loops/:loopId/trends
/loops/:loopId/events

/assessment/performance
/assessment/data-quality
/assessment/operating-condition
/assessment/readiness

/diagnostics
/diagnostics/valve
/diagnostics/oscillation
/diagnostics/model-reliability

/tuning/tasks
/tuning/tasks/:taskId/identification
/tuning/tasks/:taskId/candidates
/tuning/tasks/:taskId/approval

/knowledge/cases
/knowledge/rules
/knowledge/graph
/knowledge/model-versions

/settings/data-sources
/settings/loop-master-data
/settings/policies
/settings/users
```

## 11. 后端能力反推

前端页面需要的能力可以整理成这些服务：

```text
loop_context_resolve
fetch_loop_history
fetch_pid_config
fetch_alarm_events
fetch_operation_events
fetch_operating_condition_tags
compute_loop_health_indices
assess_loop_data_quality
assess_control_performance
assess_operating_condition
assess_identifiability
diagnose_loop_state
diagnose_valve_actuator
diagnose_oscillation
diagnose_model_reliability
decide_tuning_need
find_identification_windows
identify_or_reuse_model
generate_tuning_candidates
evaluate_tuning_candidates
recommend_next_action
create_tuning_task
create_pid_change_ticket
record_monitoring_case
```

## 12. 第一阶段建议实现范围

第一阶段不要一口气做全菜单，建议先做可闭环的最小工程化版本：

1. 工作台：总览驾驶舱。
2. 回路监控：单回路画像。
3. 回路评估：数据质量 + 整定准备度。
4. 根因诊断：诊断总览 + 模型可靠性。
5. 整定中心：整定任务 + 参数候选。
6. 经验中心：整定案例库只做列表和详情。

这样既能支撑监控 Agent，又能复用当前已有整定链路。
