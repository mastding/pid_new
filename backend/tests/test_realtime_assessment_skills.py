from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.skills  # noqa: F401
from core.skills.registry import registry
from core.skills.realtime.decide_realtime_tuning_action_skill import decide_realtime_tuning_action
from core.skills.realtime.diagnose_realtime_assessment_skill import diagnose_realtime_assessment
from core.skills.evaluation.review_auto_tuning_result_skill import review_auto_tuning_result


def test_realtime_assessment_skills_are_registered_with_metadata():
    names = set(registry.names())

    assert "diagnose_realtime_assessment" in names
    assert "decide_realtime_tuning_action" in names
    assert "review_auto_tuning_result" in names
    assert registry.metadata("diagnose_realtime_assessment")["stage"] == "diagnosis"
    assert registry.metadata("decide_realtime_tuning_action")["risk_level"] == "high"
    assert registry.metadata("review_auto_tuning_result")["stage"] == "result_review"


def test_diagnose_realtime_assessment_recommends_pid_after_good_gate_and_poor_harris():
    diagnosis = diagnose_realtime_assessment(
        assessment={"tuning_readiness": {"decision": "ready"}, "summary": {"decision": "ready"}},
        monitoring={"monitoring": {"constraints": {"score": 0.96, "mv_saturation_ratio": 0.02}}},
        harris_metric={"name": "harris", "value": 0.42, "level": "poor"},
        cpk_metric={"name": "cpk", "value": 1.2, "level": "weak"},
        ontology={"case_id": "case_5203"},
    )

    assert diagnosis[0]["root_cause"] == "pid_parameters"
    assert diagnosis[0]["confidence"] >= 0.55


def test_decide_realtime_tuning_action_blocks_high_risk_data_quality():
    result = decide_realtime_tuning_action(
        monitoring={"monitoring": {"overall_score": 0.4}},
        assessment={
            "summary": {"decision": "caution"},
            "tuning_readiness": {"blocking_reasons": [{"type": "data_quality"}]},
        },
        diagnosis=[{"root_cause": "data_quality", "confidence": 0.8, "severity": "high"}],
        harris_metric={"name": "harris", "value": 0.9, "level": "good"},
        cpk_metric=None,
    )

    assert result["risk_level"] == "high"
    assert result["decision"]["decision"] == "blocked"
    assert result["decision"]["need_tuning"] is False


def test_decide_realtime_tuning_action_recommends_tuning_for_pid_root_cause():
    result = decide_realtime_tuning_action(
        monitoring={"monitoring": {"overall_score": 0.7}},
        assessment={"summary": {"decision": "ready"}, "tuning_readiness": {"blocking_reasons": []}},
        diagnosis=[{"root_cause": "pid_parameters", "confidence": 0.72, "severity": "medium"}],
        harris_metric={"name": "harris", "value": 0.42, "level": "poor"},
        cpk_metric={"name": "cpk", "value": 1.4, "level": "good"},
    )

    assert result["decision"]["decision"] == "tuning_recommended"
    assert result["decision"]["need_tuning"] is True
    assert result["decision"]["required_confirmations"] == ["engineer_review"]


def test_review_auto_tuning_result_allows_clean_pass_for_confirmation():
    review = review_auto_tuning_result(
        evaluation={
            "passed": True,
            "final_rating": 8.4,
            "performance_score": 82.0,
            "robustness_score": 0.82,
            "mv_saturation_pct": 0.0,
            "overshoot_percent": 8.0,
        },
        pid_params={"kp": 1.2, "ki": 0.1, "kd": 0.0},
        ontology_context={"missing_fields": []},
    )

    assert review["decision"] == "ready_for_engineer_confirmation"
    assert review["can_adopt"] is True
    assert review["warnings"] == []


def test_review_auto_tuning_result_requires_revision_on_weak_evidence():
    review = review_auto_tuning_result(
        evaluation={
            "passed": False,
            "final_rating": 5.8,
            "performance_score": 60.0,
            "robustness_score": 0.5,
            "mv_saturation_pct": 9.0,
            "overshoot_percent": 24.0,
        },
        pid_params={"kp": 1.2},
        ontology_context={"missing_fields": ["pv_spec_limits.usl"]},
    )

    assert review["decision"] == "revise_required"
    assert review["can_adopt"] is False
    assert "ontology_context_review" in review["required_confirmations"]
    assert len(review["warnings"]) >= 4
