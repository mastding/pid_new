from __future__ import annotations

from core.policies.constraints import (
    PB_MAX,
    PB_MIN,
    tuning_ti_min,
    tuning_unreliable_summary,
    validate_pid_candidate,
)
from core.policies.loop_priors import (
    MIN_REASONABLE_T,
    MODEL_ORDER,
    min_reasonable_t,
    model_order_for_loop,
    normalize_loop_type,
    reality_t_range_for_loop,
)
from core.policies.scoring_rules import (
    adaptive_reality_check_t,
    apply_score_caps,
    final_rating,
    stability_limits,
)


def test_loop_priors_normalize_and_model_order():
    assert normalize_loop_type(" Temperature ") == "temperature"
    assert model_order_for_loop("flow") == MODEL_ORDER["flow"]
    assert model_order_for_loop("unknown") == ["FOPDT", "FO", "SOPDT", "IPDT"]


def test_loop_priors_min_reasonable_t_respects_dt_and_floor():
    assert min_reasonable_t("flow", 0.2) == 1.0
    assert min_reasonable_t("temperature", 5.0) == 30.0
    assert min_reasonable_t("level", 30.0) == 90.0
    assert reality_t_range_for_loop("level") == (300.0, 1800.0)


def test_constraints_validate_pid_candidate_for_ti_and_pb():
    candidate = {"Kp": 0.01, "Ti": 30.0}
    reasons = validate_pid_candidate(candidate, "temperature")
    assert any("TI=" in reason for reason in reasons)
    assert any("PB=" in reason for reason in reasons)


def test_constraints_accept_reasonable_candidate():
    candidate = {"Kp": 1.0, "Ti": 120.0}
    reasons = validate_pid_candidate(candidate, "temperature")
    assert reasons == []
    assert tuning_ti_min("pressure") == 10.0
    summary = tuning_unreliable_summary("temperature")
    assert "TI=60s" in summary
    assert str(PB_MIN) in summary and str(PB_MAX) in summary


def test_scoring_rules_stability_limits_and_adaptive_reality():
    level_limits = stability_limits("level", 300.0)
    flow_limits = stability_limits("flow", 10.0)
    assert level_limits["settling_time_limit"] > flow_limits["settling_time_limit"]

    assert adaptive_reality_check_t("level", 100.0, 0.9) == 300.0
    assert adaptive_reality_check_t("level", 3000.0, 0.9) == 1800.0
    mid = adaptive_reality_check_t("temperature", 300.0, 0.2)
    assert 300.0 <= mid <= 1200.0


def test_scoring_rules_final_rating_and_caps():
    assert final_rating(8.0, 0.8) > final_rating(3.0, 0.8)

    perf, final, readiness, passed, reasons = apply_score_caps(
        perf_score=7.5,
        final_rating_score=7.8,
        readiness_score=7.6,
        confidence=0.4,
        reality_diverged=True,
        reality_score=4.2,
        loop_type="temperature",
        tuning_unreliable=True,
        tuning_unreliable_reason="guardrail hit",
        passed=True,
    )
    assert perf == 3.0
    assert final == 3.0
    assert readiness == 3.0
    assert passed is False
    assert len(reasons) == 3
