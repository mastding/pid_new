from __future__ import annotations

from api import history_routes
from api.history_routes import _auto_tuning_task_start_gate, _tuning_blocked_by_assessment


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


def test_auto_tuning_task_start_gate_allows_manual_tuning_without_task():
    blocked, payload = _auto_tuning_task_start_gate(None, "5203_TIC_10707")

    assert blocked is False
    assert payload == {}


def test_auto_tuning_task_start_gate_blocks_missing_task(monkeypatch):
    monkeypatch.setattr(history_routes.realtime_assessment_store, "get_tuning_task", lambda task_id: None)

    blocked, payload = _auto_tuning_task_start_gate("missing_task", "5203_TIC_10707")

    assert blocked is True
    assert payload["blocking_reasons"][0]["type"] == "task_not_found"


def test_auto_tuning_task_start_gate_requires_confirmed_pending_task(monkeypatch):
    monkeypatch.setattr(
        history_routes.realtime_assessment_store,
        "get_tuning_task",
        lambda task_id: {
            "task_id": task_id,
            "loop_id": "5203_TIC_10707",
            "status": "pending_review",
            "result": {},
        },
    )

    blocked, payload = _auto_tuning_task_start_gate("task_1", "5203_TIC_10707")

    assert blocked is True
    assert payload["blocking_reasons"][0]["type"] == "task_not_confirmed"


def test_auto_tuning_task_start_gate_blocks_loop_mismatch(monkeypatch):
    monkeypatch.setattr(
        history_routes.realtime_assessment_store,
        "get_tuning_task",
        lambda task_id: {
            "task_id": task_id,
            "loop_id": "5203_TIC_10707",
            "status": "pending",
            "result": {"prepare": {"guard": {"allowed": True}}},
        },
    )

    blocked, payload = _auto_tuning_task_start_gate("task_1", "5203_TIC_10107")

    assert blocked is True
    assert payload["blocking_reasons"][0]["type"] == "loop_mismatch"


def test_auto_tuning_task_start_gate_blocks_failed_prepare_guard(monkeypatch):
    monkeypatch.setattr(
        history_routes.realtime_assessment_store,
        "get_tuning_task",
        lambda task_id: {
            "task_id": task_id,
            "loop_id": "5203_TIC_10707",
            "status": "pending",
            "result": {"prepare": {"guard": {"allowed": False, "reason": "blocked by data quality"}}},
        },
    )

    blocked, payload = _auto_tuning_task_start_gate("task_1", "5203_TIC_10707")

    assert blocked is True
    assert payload["blocking_reasons"][0]["type"] == "prepare_guard_blocked"


def test_auto_tuning_task_start_gate_allows_confirmed_task(monkeypatch):
    monkeypatch.setattr(
        history_routes.realtime_assessment_store,
        "get_tuning_task",
        lambda task_id: {
            "task_id": task_id,
            "loop_id": "5203_TIC_10707",
            "status": "pending",
            "result": {"prepare": {"guard": {"allowed": True}}},
        },
    )

    blocked, payload = _auto_tuning_task_start_gate("task_1", "5203_TIC_10707")

    assert blocked is False
    assert payload["decision"] == "pass"
