"""Skill: build loop-aware simulation scenarios."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.providers.evaluation.scenario_builder import build_simulation_scenarios
from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class BuildSimulationScenariosInputs(BaseModel):
    scenario_mode: str = Field("loop_aware", description="场景生成模式")
    include_reverse: bool = Field(True)
    include_robustness: bool = Field(True)
    conservative_level: str = Field("normal")


@register
class BuildSimulationScenariosSkill(BaseSkill):
    name = "build_simulation_scenarios"
    description = "根据回路类型、本体策略、历史数据画像和辨识模型生成闭环仿真测试场景。"
    input_model = BuildSimulationScenariosInputs
    risk_level = "medium"
    preconditions = ["model", "dt"]
    effects = [{"key": "data_profile.simulation_scenario", "description": "评估阶段复用的仿真场景包"}]
    stage = "evaluation"
    deterministic_gate = True

    def run(self, inputs: BuildSimulationScenariosInputs, ctx: LoopContext) -> SkillResult:
        if inputs.scenario_mode != "loop_aware":
            return SkillResult(success=False, reasoning=f"未知仿真场景模式: {inputs.scenario_mode}")
        scenario_pack = build_simulation_scenarios(
            loop_type=ctx.loop_type,
            model_params=dict(ctx.model or {}),
            dt=float(ctx.dt or 1.0),
            context={"ctx": ctx},
        )
        if not inputs.include_reverse:
            scenario_pack["scenarios"] = [s for s in scenario_pack.get("scenarios", []) if s.get("role") != "reverse"]
        if not inputs.include_robustness:
            scenario_pack["scenarios"] = [s for s in scenario_pack.get("scenarios", []) if s.get("role") != "robustness"]
        scenario_pack["conservative_level"] = inputs.conservative_level
        ctx.data_profile["simulation_scenario"] = scenario_pack
        return SkillResult(
            success=True,
            data={"provider": "loop_aware_scenario_builder", "simulation_scenario": scenario_pack},
            warnings=[scenario_pack["warning"]] if scenario_pack.get("warning") else [],
            reasoning=f"已生成 {scenario_pack.get('loop_label', ctx.loop_type)} 回路仿真场景。",
        )
