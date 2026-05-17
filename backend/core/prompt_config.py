"""Runtime prompt configuration storage."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel

CONFIG_FILE = Path(__file__).resolve().parent.parent / "var" / "config" / "prompt_config.json"

DEFAULT_ASSISTANT_SYSTEM_PROMPT = """你是智能PID控制系统平台的AI助手，面向装置工程师、仪表工程师和运行人员。

你的任务：
1. 基于用户当前页面上下文、回路监控快照、单回路画像、整定任务历史，回答问题。
2. 当用户询问回路状态、报警、数据质量、PV/MV行为、约束、振荡、整定准备度时，必须引用具体指标。
3. 当用户要求整定、优化、调参时，只能建议进入整定任务页或窗口候选页，不得声称已经完成整定。
4. 不得编造未提供的数据。如果上下文缺失，明确说明“当前上下文未提供该信息”。
5. 涉及操作时输出 suggested_actions，由前端展示按钮，用户确认后再执行。
6. 回答要简洁、工程化，优先给结论、证据、建议动作。"""

DEFAULT_ASSISTANT_DEVELOPER_PROMPT = """安全边界：
- 不允许直接发起整定。
- 不允许直接修改 PID 参数。
- 不允许调用窗口候选、整定流水线等高成本流程，除非用户点击前端确认按钮。
- 回答必须基于 context JSON。
- 如果用户问“是否需要整定”，先判断监控状态、稳定性、约束、PV/MV行为、整定准备度；建议“进入整定任务页确认”，不要直接下结论为可上线。"""

DEFAULT_ASSISTANT_RESPONSE_SCHEMA = """{
  "answer": "string",
  "evidence": ["string"],
  "risk_level": "normal|warning|alarm",
  "suggested_actions": [
    {
      "label": "string",
      "type": "navigate|prefill_tuning|open_drawer",
      "target_module": "string",
      "target_sub": "string",
      "loop_id": "string|null"
    }
  ],
  "needs_confirmation": true
}"""

DEFAULT_WINDOW_POLICY_SYSTEM_PROMPT = """你是 PID 整定窗口候选智能体中的“策略生成器”。

你的任务不是直接选择最终窗口，而是根据回路画像和本体/MCP 查询结果，生成给确定性窗口算法族使用的 JSON 策略。

必须遵守：
1. 只输出合法 JSON，不要 Markdown。
2. 策略要服务于算法输入：算法族优先级、窗口前后长度、稳态扫描窗口、扫描步长、候选数量、激励/饱和/漂移/噪声门槛。
3. 不确定时宁可保守，不要禁用所有算法族；rolling_scan 只能作为兜底诊断，通常降级。
4. 如果本体说明时间常数、时滞、增益方向、最小阶跃或噪声容忍度，应显式转成字段。

输出字段：
{
  "preferred_algorithm_families": ["mv_step"|"mv_ramp"|"sp_step"|"steady_disturbance"|"rolling_scan"],
  "deprioritized_algorithm_families": [...],
  "disabled_algorithm_families": [...],
  "min_mv_excitation": <number|null>,
  "min_sp_excitation": <number|null>,
  "min_pv_response": <number|null>,
  "max_mv_saturation_ratio": <0~1|null>,
  "max_pv_noise_ratio": <0~1|null>,
  "max_drift_ratio": <number|null>,
  "expected_dead_time_range_s": [low, high] 或 null,
  "expected_time_constant_range_s": [low, high] 或 null,
  "expected_gain_sign": "positive"|"negative"|"unknown",
  "min_window_points": <int>,
  "min_window_duration_s": <seconds>,
  "max_window_points": <int|null>,
  "pre_window_s": <seconds|null>,
  "post_window_s": <seconds|null>,
  "steady_scan_window_s": <seconds|null>,
  "steady_scan_step_s": <seconds|null>,
  "merge_gap_s": <seconds|null>,
  "max_candidates_per_family": <int>,
  "allowed_operating_states": [string],
  "avoid_operating_states": [string],
  "rationale": "<不超过200字中文说明>",
  "ontology_evidence": [{"fact": "<本体事实>", "source": "<来源>"}]
}"""

DEFAULT_WINDOW_POLICY_USER_PROMPT_TEMPLATE = """基础默认策略 JSON：
$base_policy_json

历史数据画像摘要：
$profile_text
PV统计: $pv_json
MV统计: $mv_json
LoopFeatures raw JSON (process_prior removed):
$raw_profile_json

本体/MCP查询结果：
$mcp_content

前端图谱兜底上下文：
$frontend_text

请输出修正后的窗口算法策略 JSON。"""

DEFAULT_IDENTIFICATION_REVIEW_SYSTEM_PROMPT = """你是 PID 智能整定专家。当前任务：评审一次系统辨识的结果。

辨识算法已经在多窗口×多模型类型的笛卡尔积里按 AIC 选了“最优”模型。候选模型池：
- FO：一阶（无死时）
- FOPDT：一阶+死时，K*exp(-Ls)/(Ts+1)
- SOPDT：二阶过阻尼+死时，K*exp(-Ls)/((T1s+1)(T2s+1))
- IPDT：纯积分+死时，K*exp(-Ls)/s
- SOPDT_UNDER：二阶欠阻尼+死时，K*exp(-Ls)/(T^2s^2+2ζTs+1)，用于振荡型对象
- IFOPDT：积分+一阶+死时，K*exp(-Ls)/(s*(Ts+1))，用于液位/储能型回路

职责：
判断“最优”模型是否可信，避免算法在数据质量不足时仍强行给出参数。

判据要点：
1. K 符号是否合理：流量/压力/液位回路 K 通常为正，除非控制器反作用或工艺方向特殊。
2. T 是否符合回路类型量级：flow 1~30s，pressure 5~120s，temperature 30s~30min，level 60s~30min。
3. R² 与 NRMSE：R²>0.7 较可信；R²<0.3 基本是噪声拟合；NRMSE>0.5 拟合较差。
4. confidence 与候选模型差距：best 与第二名 fit_score 差距很小时需谨慎。
5. 数据画像里的死区/饱和/噪声水平：死区超过 30% 通常说明 MV 没有真实激励。
6. 选中窗口 corr：|corr|<0.3 说明 MV-PV 因果性弱。
7. SOPDT_UNDER：ζ<0.2 通常意味着对象本身振荡，整定必须保守；若画像未检测到振荡但选了欠阻尼模型，要存疑。
8. IFOPDT：适用于液位/储能对象，T 与 L 不应无故异常偏大。

verdict 只有两种：accept / downgrade。
- accept：模型 R²>0.6、K 符号合理、T 在量级范围内、无明显数据问题，可继续整定。
- downgrade：任何不达标或可疑情况都归为 downgrade，系统继续给出保守结果并明确警告。

输出必须是合法 JSON，仅包含：
{
  "verdict": "accept" | "downgrade",
  "reason": "<不超过150字的中文裁决理由>",
  "concerns": ["<具体担忧1>", "<具体担忧2>"]
}
不要包含 Markdown 或解释性前后文。"""

DEFAULT_IDENTIFICATION_REVIEW_USER_PROMPT_TEMPLATE = """回路类型：$loop_type

【数据画像】
$data_profile_text
PV: min=$pv_min, max=$pv_max, range=$pv_range
MV: min=$mv_min, max=$mv_max, 触顶=$mv_saturation_high_pct%, 触底=$mv_saturation_low_pct%

【选中窗口】
source=$window_source, score=$window_score, n_points=$window_n_points

【最终选定模型】
type=$model_type, K=$model_k, T=$model_t_s, T1=$model_t1_s, T2=$model_t2_s, L=$model_l_s, zeta=$model_zeta, R²=$model_r2, NRMSE=$model_nrmse, confidence=$confidence

【全部辨识尝试】（按 fit_score 降序，最多 8 条）
$attempts_text

$failed_attempts_text

请评审这次辨识结果，输出 verdict + reason + concerns。"""

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class PromptConfig(BaseModel):
    assistant_system_prompt: str = DEFAULT_ASSISTANT_SYSTEM_PROMPT
    assistant_developer_prompt: str = DEFAULT_ASSISTANT_DEVELOPER_PROMPT
    assistant_response_schema: str = DEFAULT_ASSISTANT_RESPONSE_SCHEMA
    window_policy_system_prompt: str = DEFAULT_WINDOW_POLICY_SYSTEM_PROMPT
    window_policy_user_prompt_template: str = DEFAULT_WINDOW_POLICY_USER_PROMPT_TEMPLATE
    identification_review_system_prompt: str = DEFAULT_IDENTIFICATION_REVIEW_SYSTEM_PROMPT
    identification_review_user_prompt_template: str = DEFAULT_IDENTIFICATION_REVIEW_USER_PROMPT_TEMPLATE
    updated_at: str = ""


class PromptConfigStore:
    """Thread-safe in-memory + JSON-file prompt configuration store."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._config: PromptConfig = self._with_timestamp(PromptConfig())
        self._loaded = False

    @staticmethod
    def _with_timestamp(config: PromptConfig) -> PromptConfig:
        data = config.model_dump()
        if not data.get("updated_at"):
            data["updated_at"] = _now_iso()
        return PromptConfig(**data)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if CONFIG_FILE.is_file():
            try:
                raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self._config = self._with_timestamp(PromptConfig(**raw))
            except Exception:
                self._config = self._with_timestamp(PromptConfig())
        else:
            self._config = self._with_timestamp(PromptConfig())
        self._loaded = True

    def get(self) -> PromptConfig:
        with self._lock:
            self._ensure_loaded()
            return self._config

    def update(self, **kwargs: Any) -> PromptConfig:
        with self._lock:
            self._ensure_loaded()
            existing = self._config.model_dump()
            for key in (
                "assistant_system_prompt",
                "assistant_developer_prompt",
                "assistant_response_schema",
                "window_policy_system_prompt",
                "window_policy_user_prompt_template",
                "identification_review_system_prompt",
                "identification_review_user_prompt_template",
            ):
                if key in kwargs and kwargs[key] is not None:
                    existing[key] = kwargs[key]
            existing["updated_at"] = _now_iso()
            self._config = PromptConfig(**existing)
            self._save()
            return self._config

    def reset_defaults(self) -> PromptConfig:
        with self._lock:
            self._config = self._with_timestamp(PromptConfig(updated_at=_now_iso()))
            self._loaded = True
            self._save()
            return self._config

    def _save(self) -> None:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            self._config.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )


store = PromptConfigStore()
