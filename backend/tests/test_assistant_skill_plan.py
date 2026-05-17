from __future__ import annotations

from api.assistant_routes import _assistant_skill_plan, _assistant_workflow_plan_event


def test_assistant_skill_plan_exposes_registered_skill_metadata():
    plan = _assistant_skill_plan(
        "请分析 harris、cpk 指标并判断是否需要 PID 整定",
        {
            "status": "ok",
            "loop": {"loop_id": "5203_TIC_10107"},
            "monitoring": {"status": "warning"},
            "realtime_assessment": {
                "metrics": [{"name": "harris_index", "value": 0.42}],
                "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.8}],
                "decision": {"action": "recommend_tuning"},
            },
        },
    )

    names = [item["name"] for item in plan]
    assert names[0] == "load_loop_context"
    assert "diagnose_realtime_assessment" in names
    assert "decide_realtime_tuning_action" in names

    diagnose = next(item for item in plan if item["name"] == "diagnose_realtime_assessment")
    assert diagnose["risk_level"]
    assert diagnose["stage"]
    assert "preconditions" in diagnose
    assert "effects" in diagnose

    event = _assistant_workflow_plan_event(plan)
    assert event["type"] == "workflow_plan"
    assert event["planner_mode"] == "heuristic_intent_router"
    assert [item["skill_name"] for item in event["skills"]] == names
