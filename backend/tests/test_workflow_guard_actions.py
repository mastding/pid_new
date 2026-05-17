from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.workflow_guard import WorkflowGuard


def test_workflow_guard_action_blocks_unmet_preconditions():
    guard = WorkflowGuard()

    decision = guard.check_action(
        "create_auto_tuning_task",
        risk_level="high",
        preconditions={
            "snapshot_exists": True,
            "decision_recommends_tuning": False,
        },
        initiated_by="system",
    )

    assert decision.allowed is False
    assert decision.risk_level == "high"
    assert decision.unmet_preconditions == ["decision_recommends_tuning"]


def test_workflow_guard_action_blocks_llm_high_risk():
    guard = WorkflowGuard(max_llm_risk_level="medium")

    decision = guard.check_action(
        "prepare_auto_tuning_task",
        risk_level="high",
        preconditions={"engineer_confirmed": True},
        initiated_by="llm",
    )

    assert decision.allowed is False
    assert "LLM" in decision.reason


def test_workflow_guard_action_allows_system_high_risk_when_ready():
    guard = WorkflowGuard()

    decision = guard.check_action(
        "prepare_auto_tuning_task",
        risk_level="high",
        preconditions={"engineer_confirmed": True},
        initiated_by="system",
    )

    assert decision.allowed is True
