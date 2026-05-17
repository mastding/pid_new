"""Semi-autonomous skill orchestration primitives.

This module intentionally does not replace the tuning runner yet. It provides a
guarded execution layer so an LLM planner can insert retries/explanations around
the default template without bypassing deterministic PID safety gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.skills.base import LoopContext, SkillResult
from core.skills.registry import registry
from core.workflow_guard import GuardDecision, workflow_guard


DEFAULT_TUNING_TEMPLATE = [
    "load_dataset",
    "summarize_data",
    "build_ontology_policy",
    "detect_windows",
    "select_window",
    "identify_model",
    "review_identification",
    "generate_tuning_candidates",
    "build_simulation_scenarios",
    "evaluate_tuning",
]


DEFAULT_ASSESSMENT_TEMPLATE = [
    "assess_loop_monitoring",
    "assess_loop_assessment",
    "resolve_loop_ontology_context",
    "compute_cpk",
    "compute_harris_closed_loop",
    "diagnose_realtime_assessment",
    "decide_realtime_tuning_action",
]


@dataclass
class PlannedSkillCall:
    skill_name: str
    args: dict[str, Any] = field(default_factory=dict)
    initiated_by: str = "system"
    purpose: str = ""


@dataclass
class SkillExecutionRecord:
    call: PlannedSkillCall
    guard: GuardDecision
    result: SkillResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": {
                "skill_name": self.call.skill_name,
                "args": self.call.args,
                "initiated_by": self.call.initiated_by,
                "purpose": self.call.purpose,
            },
            "guard": self.guard.to_dict(),
            "result": self.result.to_llm_dict() if self.result is not None else None,
        }


class SkillOrchestrator:
    """Guarded executor for default-template plus LLM-planned insertions."""

    def default_args_for(
        self,
        skill_name: str,
        ctx: LoopContext,
        *,
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        override = dict((overrides or {}).get(skill_name, {}))
        defaults: dict[str, Any] = {
            "load_dataset": {},
            "summarize_data": {},
            "build_ontology_policy": {
                "loop_name": ctx.loop_prefix or "",
                "loop_type": ctx.loop_type,
                "use_llm_advisor": True,
            },
            "detect_windows": {
                "loop_type": ctx.loop_type,
                "policy": ctx.data_profile.get("window_policy"),
            },
            "select_window": {},
            "identify_model": {"use_usable_windows_only": True},
            "review_identification": {
                "use_llm_advisor": True,
                "allow_retry_plan": True,
            },
            "generate_tuning_candidates": {},
            "build_simulation_scenarios": {
                "scenario_mode": "loop_aware",
                "include_reverse": True,
                "include_robustness": True,
            },
            "evaluate_tuning": {
                "tuning_unreliable": bool(ctx.data_profile.get("tuning_unreliable", False)),
                "tuning_unreliable_reason": str(ctx.data_profile.get("tuning_unreliable_reason", "")),
            },
        }.get(skill_name, {})
        defaults.update(override)
        return defaults

    def build_default_plan(
        self,
        ctx: LoopContext,
        *,
        overrides: dict[str, dict[str, Any]] | None = None,
        initiated_by: str = "system",
        template: list[str] | None = None,
    ) -> list[PlannedSkillCall]:
        return [
            PlannedSkillCall(
                skill_name=name,
                args=dict((overrides or {}).get(name, {})),
                initiated_by=initiated_by,
                purpose=f"default_template:{name}",
            )
            for name in (template or DEFAULT_TUNING_TEMPLATE)
        ]

    def build_assessment_plan(
        self,
        ctx: LoopContext,
        *,
        include_formal_metrics: bool = True,
        initiated_by: str = "system",
    ) -> list[PlannedSkillCall]:
        template = list(DEFAULT_ASSESSMENT_TEMPLATE)
        if not include_formal_metrics:
            template = [
                name for name in template
                if name not in {"compute_cpk", "compute_harris_closed_loop"}
            ]
        return self.build_default_plan(ctx, initiated_by=initiated_by, template=template)

    def execute_one(self, call: PlannedSkillCall, ctx: LoopContext) -> SkillExecutionRecord:
        decision = workflow_guard.check(call.skill_name, ctx, initiated_by=call.initiated_by)
        if not decision.allowed:
            return SkillExecutionRecord(call=call, guard=decision, result=None)
        result = registry.invoke(call.skill_name, call.args, ctx)
        return SkillExecutionRecord(call=call, guard=decision, result=result)

    def execute_plan(self, calls: list[PlannedSkillCall], ctx: LoopContext) -> list[SkillExecutionRecord]:
        records: list[SkillExecutionRecord] = []
        for call in calls:
            defaults = self.default_args_for(call.skill_name, ctx)
            defaults.update({key: value for key, value in call.args.items() if value is not None})
            call.args = defaults
            record = self.execute_one(call, ctx)
            records.append(record)
            if not record.guard.allowed or not (record.result and record.result.success):
                break
        return records


skill_orchestrator = SkillOrchestrator()
