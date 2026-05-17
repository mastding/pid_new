"""Skill: build ontology-informed window/tuning policy."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.pipeline.ontology_policy_builder import build_window_selection_policy
from core.pipeline.window_policy_advisor import ask_window_policy_via_llm
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class BuildOntologyPolicyInputs(BaseModel):
    loop_name: str = Field("", description="回路名称/位号")
    loop_type: str | None = Field(None, description="回路类型覆盖值")
    frontend_context: str | None = Field(None, description="前端传入的本体上下文")
    mcp_context: dict[str, Any] | None = Field(None, description="外部已获取的 MCP 本体上下文")
    use_llm_advisor: bool = Field(True, description="是否允许 LLM 修正基础策略")


@register
class BuildOntologyPolicySkill(BaseSkill):
    name = "build_ontology_policy"
    description = "融合本体上下文、历史数据画像和回路类型，生成窗口选择与后续整定评估可复用的策略。"
    input_model = BuildOntologyPolicyInputs
    risk_level = "medium"
    preconditions = ["cleaned_df", "dt"]
    effects = [
        {"key": "data_profile.window_policy", "description": "本体/数据融合后的窗口选择策略"},
        {"key": "data_profile.ontology_context", "description": "本体来源与内容摘要"},
    ]
    stage = "ontology_policy"
    deterministic_gate = True

    def run(self, inputs: BuildOntologyPolicyInputs, ctx: LoopContext) -> SkillResult:
        loop_type = inputs.loop_type or ctx.loop_type
        base_policy = build_window_selection_policy(
            loop_name=inputs.loop_name or ctx.loop_prefix or "",
            loop_type=loop_type,
            data_profile=ctx.data_profile,
            mcp_context=inputs.mcp_context,
            frontend_context=inputs.frontend_context,
        )
        policy = base_policy
        source = "default"
        warnings: list[str] = []
        if inputs.use_llm_advisor:
            llm_policy = ask_window_policy_via_llm(
                base_policy=base_policy,
                data_profile=ctx.data_profile,
                mcp_context=inputs.mcp_context,
                frontend_context=inputs.frontend_context,
            )
            if llm_policy:
                policy = llm_policy
                source = "llm"
            else:
                warnings.append("LLM 策略顾问不可用，已使用确定性本体策略。")

        mcp_content = str((inputs.mcp_context or {}).get("content") or "")
        ontology_meta = {
            "ontology_context_source": "mcp" if mcp_content else "frontend" if inputs.frontend_context else "none",
            "ontology_context_used": bool(mcp_content or inputs.frontend_context),
            "ontology_mcp_server": (inputs.mcp_context or {}).get("server_name"),
            "ontology_mcp_tool": (inputs.mcp_context or {}).get("tool"),
            "ontology_mcp_query": (inputs.mcp_context or {}).get("query"),
            "ontology_mcp_content_preview": mcp_content[:1200],
            "ontology_mcp_content_raw": mcp_content,
            "ontology_mcp_content_chars": len(mcp_content),
            "ontology_mcp_error": str((inputs.mcp_context or {}).get("error") or "")[:500],
        }
        ctx.data_profile["window_policy"] = policy
        ctx.data_profile["ontology_context"] = {
            "source": source,
            "meta": ontology_meta,
            "frontend_context": inputs.frontend_context,
        }
        return SkillResult(
            success=True,
            data={
                "provider": "ontology_policy",
                "policy": policy,
                "source": source,
                "confidence": policy.get("confidence", 0.0),
                "ontology_source": (policy.get("ontology_facts") or {}).get("source", "none"),
                "ontology_meta": ontology_meta,
            },
            warnings=warnings,
            reasoning=f"已生成 {loop_type} 回路本体策略，来源 {source}。",
        )
