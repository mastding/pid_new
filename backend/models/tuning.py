"""Tuning request/response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PIDParams(BaseModel):
    """PID controller parameters."""

    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0
    Ti: float = 0.0
    Td: float = 0.0
    strategy: str = ""
    description: str = ""


class TuningRequest(BaseModel):
    """Request to run the full tuning pipeline."""

    # Data source: either csv_path or history API params
    csv_path: str = ""
    loop_uri: str = ""
    start_time: str = ""
    end_time: str = ""

    # Loop metadata
    loop_name: str = ""
    loop_type: str = "flow"
    plant_type: str = ""
    scenario: str = ""
    control_object: str = ""

    # Options
    selected_loop_prefix: str | None = None
    selected_window_index: int | None = None
    use_consultant: bool = True
    use_llm_advisor: bool = True  # Day 4：是否在窗口选择阶段调用 LLM 顾问
    # 提前结束流水线的切点：
    #   "window_selection" — 数据分析菜单（跑到选窗为止）
    #   "identification"  — 系统辨识菜单（跑到辨识 + 精修循环结束）
    #   None              — 跑完全流程（整定菜单）
    stop_after: str | None = None
    # 候选窗口算法白名单：例如 ["sv_step", "mv_step"]。None / [] 表示不过滤。
    algorithm_filter: list[str] | None = None
    # 可选本体上下文：由前端或未来图数据库抽取，供 LLM 顾问判断窗口合理性。
    ontology_context: str | None = None


class TuningResult(BaseModel):
    """Complete result from tuning pipeline."""

    data_analysis: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    pid_params: PIDParams = Field(default_factory=PIDParams)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    tuning_advice: dict[str, Any] = Field(default_factory=dict)
    consultant_summary: str = ""


class ConsultantMessage(BaseModel):
    """A message in the consultant chat."""

    role: str  # "user" | "assistant"
    content: str
