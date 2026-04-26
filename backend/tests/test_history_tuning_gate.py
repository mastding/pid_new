from __future__ import annotations

from api.history_routes import _tuning_blocked_by_assessment


def test_tuning_gate_blocks_blocked_decision():
    blocked, payload = _tuning_blocked_by_assessment({
        "summary": {"decision": "blocked", "decision_text": "not ready"},
        "tuning_readiness": {"gate_checks": []},
    })

    assert blocked is True
    assert payload["decision"] == "blocked"


def test_tuning_gate_blocks_hard_failed_check():
    blocked, payload = _tuning_blocked_by_assessment({
        "summary": {"decision": "caution"},
        "tuning_readiness": {
            "gate_checks": [
                {"name": "data_quality", "passed": False, "severity": "high", "message": "bad data"},
                {"name": "oscillation", "passed": False, "severity": "medium", "message": "watch"},
            ],
        },
    })

    assert blocked is True
    assert [item["name"] for item in payload["failed_checks"]] == ["data_quality"]


def test_tuning_gate_allows_caution_without_hard_failure():
    blocked, payload = _tuning_blocked_by_assessment({
        "summary": {"decision": "caution"},
        "tuning_readiness": {
            "blocking_reasons": [{"type": "condition", "severity": "medium", "message": "confirm load"}],
            "gate_checks": [
                {"name": "condition", "passed": True, "severity": "ok", "message": "confirm"},
            ],
        },
    })

    assert blocked is False
    assert payload["decision"] == "caution"
