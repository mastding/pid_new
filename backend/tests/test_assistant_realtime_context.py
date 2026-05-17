from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import assistant_routes
from api.assistant_routes import _assistant_skill_plan


def test_build_loop_context_includes_latest_realtime_assessment(monkeypatch):
    def fake_get_loop(loop_id: str):
        return {
            "loop_id": loop_id,
            "loop_type": "temperature",
            "source_filename": "sample.csv",
            "start_time": "2026-04-23 00:00:00",
            "end_time": "2026-04-23 08:00:00",
        }

    def fake_latest(**kwargs):
        return {
            "items": [
                {
                    "snapshot_id": "asmt_test",
                    "created_at": "2026-05-17T00:00:00Z",
                    "time_window": {"range": "8h"},
                    "risk_level": "medium",
                    "score": 0.72,
                    "decision": {"decision": "tuning_recommended", "need_tuning": True},
                    "metrics": [{"name": "harris", "value": 0.42, "level": "poor", "confidence": 0.8, "success": True}],
                    "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.7, "severity": "medium", "action": "review"}],
                    "ontology": {"missing_fields": ["pv_spec_limits.lsl"]},
                    "skill_trace": [{"skill_name": "compute_harris_closed_loop", "risk_level": "medium", "status": "completed"}],
                }
            ]
        }

    monkeypatch.setattr(assistant_routes, "get_loop", fake_get_loop)
    monkeypatch.setattr(assistant_routes, "get_loop_monitoring", lambda *args, **kwargs: {"monitoring": {"status": "normal"}})
    monkeypatch.setattr(assistant_routes, "get_loop_features", lambda *args, **kwargs: {"identity": {"loop_id": "5203_TIC_10707"}})
    monkeypatch.setattr(assistant_routes, "assess_loop", lambda *args, **kwargs: {"summary": {"decision": "ready"}})
    monkeypatch.setattr(assistant_routes.realtime_assessment_service, "latest", fake_latest)

    context = assistant_routes._build_loop_context({"loop_id": "5203_TIC_10707"})

    assert context["status"] == "ok"
    assert context["realtime_assessment"]["decision"]["need_tuning"] is True
    assert context["realtime_assessment"]["metrics"][0]["name"] == "harris"
    assert context["realtime_assessment"]["skill_trace"][0]["skill_name"] == "compute_harris_closed_loop"


def test_assistant_skill_plan_selects_diagnosis_and_tuning_events():
    plan = _assistant_skill_plan(
        "请结合 harris、cpk 指标诊断是否需要 PID 整定建议",
        {
            "status": "ok",
            "loop": {"loop_id": "5203_TIC_10707"},
            "monitoring": {"overall_score": 0.72},
            "realtime_assessment": {
                "metrics": [{"name": "harris", "value": 0.42}],
                "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.72}],
                "decision": {"need_tuning": True, "required_confirmations": ["engineer_review"]},
            },
        },
    )

    names = [item["name"] for item in plan]
    assert names[0] == "load_loop_context"
    assert "diagnose_realtime_assessment" in names
    assert "decide_realtime_tuning_action" in names
    assert all(item["type"] == "tool_event" for item in plan)


def test_assistant_skill_plan_selects_auto_tuning_review_event():
    plan = _assistant_skill_plan(
        "这个整定结果的仿真曲线和评分能不能上线，需要复核什么",
        {
            "status": "ok",
            "loop": {"loop_id": "5203_TIC_10707"},
            "realtime_assessment": {
                "decision": {"need_tuning": True},
            },
        },
    )

    names = [item["name"] for item in plan]
    assert "review_auto_tuning_result" in names
    review_event = next(item for item in plan if item["name"] == "review_auto_tuning_result")
    assert review_event["risk_level"] == "high"
