from __future__ import annotations

import asyncio

from core.pipeline.runner import run_tuning_pipeline


def test_runner_emits_default_workflow_plan_before_loading_data():
    async def _first_event():
        gen = run_tuning_pipeline(
            csv_path="missing.csv",
            loop_type="flow",
            use_llm_advisor=False,
        )
        return await anext(gen)

    event = asyncio.run(_first_event())

    assert event["type"] == "workflow_plan"
    assert event["planner_mode"] == "default_template"
    assert event["llm_insertions_enabled"] is False
    skill_names = [item["skill_name"] for item in event["skills"]]
    assert skill_names[0] == "load_dataset"
    assert "build_ontology_policy" in skill_names
    assert skill_names[-1] == "evaluate_tuning"
