"""技能（Skill）基类与共享类型。

设计要点：
- 技能是 LLM 可调用的、原子的、确定性的执行单元。
- 输入参数由 LLM 提供，经 Pydantic 校验，自带 JSON Schema。
- Context 是服务端可变状态，在一次整定会话的多个技能间共享，**永不暴露给 LLM**。
  技能可以读取并更新 context。
- 输出是 SkillResult；其中 `data` 字段对应技能的 `output_model` 形状，
  但保持为普通 dict 以便序列化为 JSON 给 LLM。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel


# ── 服务端上下文（永不发送给 LLM） ─────────────────────────────────────────

@dataclass
class LoopContext:
    """单回路在一次整定会话中共享的可变状态。

    当前面向单回路设计；多回路场景 = list[LoopContext]。
    各技能按需读取已有字段、写入新字段。
    """
    csv_path: str
    loop_prefix: str = ""
    loop_type: str = "flow"

    # 由数据加载类技能填充
    raw_df: pd.DataFrame | None = None
    cleaned_df: pd.DataFrame | None = None
    dt: float | None = None

    # 由数据理解类技能填充
    data_profile: dict[str, Any] = field(default_factory=dict)

    # 由窗口检测填充
    candidate_windows: list[dict[str, Any]] = field(default_factory=list)
    selected_window_index: int | None = None

    # 由系统辨识填充
    model: dict[str, Any] | None = None
    confidence: float | None = None

    # 由 PID 整定填充
    pid_params: dict[str, Any] | None = None

    # 由性能评估填充
    evaluation: dict[str, Any] | None = None

    # 技能调用审计日志（用于生成最终报告）
    skill_log: list[dict[str, Any]] = field(default_factory=list)


# ── 技能输出 ───────────────────────────────────────────────────────────────

class SkillResult(BaseModel):
    """所有技能的标准返回值。

    success=False 不会抛异常，是正常的流程控制。LLM 通过 `reasoning` 字段
    了解发生了什么。
    """
    success: bool
    data: dict[str, Any] = {}
    warnings: list[str] = []
    reasoning: str = ""

    def to_llm_dict(self) -> dict[str, Any]:
        """供 LLM 查看的紧凑字典（省略空字段，节省 token）。"""
        out: dict[str, Any] = {"success": self.success}
        if self.data:
            out["data"] = self.data
        if self.warnings:
            out["warnings"] = self.warnings
        if self.reasoning:
            out["reasoning"] = self.reasoning
        return out


# ── 技能基类 ───────────────────────────────────────────────────────────────

class BaseSkill(ABC):
    """所有技能的抽象基类。

    子类必须定义：
        name           — 唯一标识，snake_case（LLM 可见）
        description    — 自然语言描述（LLM 可见，必须中文）
        input_model    — Pydantic 模型，描述 LLM 提供的入参
        run()          — 实际执行逻辑

    基类自动从 input_model 生成 OpenAI tool schema。
    """
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, inputs: BaseModel, ctx: LoopContext) -> SkillResult:
        """执行技能。**禁止抛异常**，失败时返回 SkillResult(success=False)。"""
        ...

    def invoke(self, raw_args: dict[str, Any], ctx: LoopContext) -> SkillResult:
        """校验入参并执行。这是注册表对外暴露的入口。"""
        try:
            inputs = self.input_model(**raw_args)
        except Exception as exc:
            return SkillResult(
                success=False,
                reasoning=f"参数校验失败: {exc}",
            )
        try:
            result = self.run(inputs, ctx)
        except Exception as exc:
            result = SkillResult(
                success=False,
                reasoning=f"执行异常 ({type(exc).__name__}): {exc}",
            )

        # 写入审计日志
        ctx.skill_log.append({
            "skill": self.name,
            "args": raw_args,
            "success": result.success,
            "reasoning": result.reasoning,
        })
        return result

    @classmethod
    def to_openai_tool(cls) -> dict[str, Any]:
        """从 input_model 自动生成 OpenAI function-calling 工具 schema。"""
        schema = cls.input_model.model_json_schema()
        # 去除 OpenAI 不需要的 $defs / title 噪声
        params = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": params,
            },
        }


# 无入参技能使用此空模型
class NoInputs(BaseModel):
    """当一个技能不需要 LLM 提供任何参数时，用此模型作为 input_model。"""
    pass
