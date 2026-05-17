from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.realtime.assessment_service import (
    PrepareAutoTuningTaskRequest,
    RealtimeAssessmentRequest,
    RealtimeAssessmentService,
)


def test_run_auto_creates_pending_review_task(monkeypatch, tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")

    monkeypatch.setattr(service, "_select_loops", lambda request: [{"loop_id": "5203_TIC_10707"}])

    async def fake_run_one(loop_id: str, request: RealtimeAssessmentRequest):
        payload = {
            "snapshot_id": "asmt_auto_task",
            "loop_id": loop_id,
            "asset_id": "5203",
            "loop_type": "temperature",
            "created_at": "2026-05-17T00:00:00Z",
            "time_window": {"range": request.time_range},
            "risk_level": "medium",
            "score": 0.72,
            "ontology": {"context_id": "onto_auto_task", "missing_fields": []},
            "metrics": [],
            "diagnosis": [],
            "decision": {"decision": "tuning_recommended", "need_tuning": True, "summary": "unit test"},
            "skill_trace": [],
        }
        return service.store.save_snapshot(payload)

    monkeypatch.setattr(service, "run_one", fake_run_one)

    import asyncio

    result = asyncio.run(service.run(RealtimeAssessmentRequest(
        loop_ids=["5203_TIC_10707"],
        auto_create_tasks=True,
    )))

    assert result["saved"] == 1
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["status"] == "pending_review"
    assert result["tasks"][0]["trigger_mode"] == "realtime_assessment"


def test_prepare_tuning_task_requires_confirmation(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    snapshot = {
        "snapshot_id": "asmt_prepare",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h", "start_time": "2026-04-23 02:00:00", "end_time": "2026-04-23 10:00:00"},
        "risk_level": "medium",
        "ontology": {"case_id": "case_1", "facts": {"loop_type": "temperature"}, "missing_fields": []},
        "metrics": [{"name": "harris", "value": 0.42}],
        "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.72}],
        "decision": {"decision": "tuning_recommended", "need_tuning": True, "blocked": False},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)
    task = service.create_tuning_task("asmt_prepare", trigger_mode="unit_test")

    prepared = service.prepare_tuning_task(task["task_id"], PrepareAutoTuningTaskRequest(confirm=False))

    assert prepared["guard"]["allowed"] is False
    assert prepared["guard"]["requires_confirmation"] is True
    assert "engineer_confirmed" in prepared["guard"]["unmet_preconditions"]
    assert prepared["task"]["status"] == "pending_review"
    assert prepared["tuning_request"]["loop_id"] == "5203_TIC_10707"
    assert prepared["tuning_request"]["start_time"] == "2026-04-23 02:00:00"


def test_prepare_tuning_task_confirm_marks_pending(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    snapshot = {
        "snapshot_id": "asmt_confirm",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h", "start_time": "2026-04-23 02:00:00", "end_time": "2026-04-23 10:00:00"},
        "risk_level": "medium",
        "ontology": {"case_id": "case_1", "facts": {}, "missing_fields": []},
        "metrics": [],
        "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.72}],
        "decision": {"decision": "tuning_recommended", "need_tuning": True, "blocked": False},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)
    task = service.create_tuning_task("asmt_confirm", trigger_mode="unit_test")

    prepared = service.prepare_tuning_task(task["task_id"], PrepareAutoTuningTaskRequest(confirm=True))

    assert prepared["guard"]["allowed"] is True
    assert prepared["task"]["guard"]["allowed"] is True
    assert prepared["task"]["status"] == "pending"
    assert prepared["tuning_request"]["ontology_context"]["snapshot_id"] == "asmt_confirm"


def test_prepare_tuning_task_keeps_blocked_snapshot_blocked(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    snapshot = {
        "snapshot_id": "asmt_blocked",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h"},
        "risk_level": "high",
        "ontology": {"case_id": "case_1", "facts": {}, "missing_fields": []},
        "metrics": [],
        "diagnosis": [{"root_cause": "data_quality", "confidence": 0.8}],
        "decision": {"decision": "blocked", "need_tuning": False, "blocked": True},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)
    task = service.create_tuning_task("asmt_blocked", trigger_mode="unit_test")

    prepared = service.prepare_tuning_task(task["task_id"], PrepareAutoTuningTaskRequest(confirm=True))

    assert prepared["guard"]["allowed"] is False
    assert prepared["guard"]["blocked"] is True
    assert prepared["task"]["status"] == "blocked"


def test_create_tuning_task_blocks_when_not_recommended(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    snapshot = {
        "snapshot_id": "asmt_observe",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h"},
        "risk_level": "normal",
        "ontology": {"case_id": "case_1", "facts": {}, "missing_fields": []},
        "metrics": [],
        "diagnosis": [{"root_cause": "no_dominant_fault", "confidence": 0.45}],
        "decision": {"decision": "observe", "need_tuning": False, "blocked": False},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)

    task = service.create_tuning_task("asmt_observe", trigger_mode="unit_test")

    assert task["status"] == "blocked"
    assert task["guard"]["allowed"] is False
    assert "decision_recommends_tuning" in task["guard"]["unmet_preconditions"]


def test_create_tuning_task_reuses_unfinished_loop_task(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    snapshot = {
        "snapshot_id": "asmt_dup_1",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h"},
        "risk_level": "medium",
        "ontology": {"case_id": "case_1", "facts": {}, "missing_fields": []},
        "metrics": [],
        "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.72}],
        "decision": {"decision": "tuning_recommended", "need_tuning": True, "blocked": False},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)
    first = service.create_tuning_task("asmt_dup_1", trigger_mode="unit_test")

    second = service.create_tuning_task("asmt_dup_1", trigger_mode="unit_test")

    assert second["task_id"] == first["task_id"]
    assert second["reused_existing"] is True
    assert "no_unfinished_auto_tuning_task" in second["guard"]["unmet_preconditions"]
    assert service.list_tuning_tasks(loop_id="5203_TIC_10707")["total"] == 1


def test_create_tuning_task_skips_during_cooldown(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    service.update_monitor_config({"auto_tuning_cooldown_hours": 24})
    snapshot = {
        "snapshot_id": "asmt_cooldown_1",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h"},
        "risk_level": "medium",
        "ontology": {"case_id": "case_1", "facts": {}, "missing_fields": []},
        "metrics": [],
        "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.72}],
        "decision": {"decision": "tuning_recommended", "need_tuning": True, "blocked": False},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)
    first = service.create_tuning_task("asmt_cooldown_1", trigger_mode="unit_test")
    service.store.update_tuning_task(
        first["task_id"],
        {
            "status": "completed",
            "updated_at": "2999-05-17T00:00:00Z",
            "result": {"pipeline": {"completed_at": "2999-05-17T00:00:00Z"}},
        },
    )

    skipped = service.create_tuning_task("asmt_cooldown_1", trigger_mode="unit_test")

    assert skipped["status"] == "skipped"
    assert skipped["cooldown"]["active"] is True
    assert "cooldown_elapsed" in skipped["guard"]["unmet_preconditions"]
    assert service.list_tuning_tasks(loop_id="5203_TIC_10707")["total"] == 1


def test_get_tuning_task_result_returns_review_and_summary(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    snapshot = {
        "snapshot_id": "asmt_result_1",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h"},
        "risk_level": "medium",
        "ontology": {"case_id": "case_1", "facts": {}, "missing_fields": []},
        "metrics": [],
        "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.72}],
        "decision": {"decision": "tuning_recommended", "need_tuning": True, "blocked": False},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)
    task = service.create_tuning_task("asmt_result_1", trigger_mode="unit_test")
    service.store.update_tuning_task(
        task["task_id"],
        {
            "status": "completed",
            "updated_at": "2026-05-17T00:10:00Z",
            "result": {
                "pipeline": {
                    "review": {"decision": "ready_for_engineer_confirmation"},
                    "tuning_summary": {"pid_params": {"kp": 1.0}},
                }
            },
        },
    )

    result = service.get_tuning_task_result(task["task_id"])

    assert result["task"]["status"] == "completed"
    assert result["review"]["decision"] == "ready_for_engineer_confirmation"
    assert result["tuning_summary"]["pid_params"]["kp"] == 1.0


def test_get_model_review_snapshot_merges_assessment_and_latest_result(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")
    snapshot = {
        "snapshot_id": "asmt_model_review_1",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h"},
        "risk_level": "medium",
        "assessment": {
            "data_quality": {"score": 0.9},
            "identification_suitability": {
                "score": 0.7,
                "excitation_score": 0.65,
                "response_observability_score": 0.75,
                "window_count": 4,
                "usable_window_count": 2,
            },
            "tuning_readiness": {"score": 0.8},
        },
        "ontology": {"case_id": "case_1", "facts": {}, "missing_fields": []},
        "metrics": [{"name": "harris", "value": 0.42, "success": True}],
        "diagnosis": [{"root_cause": "pid_parameters", "confidence": 0.72}],
        "decision": {"decision": "tuning_recommended", "need_tuning": True, "blocked": False},
        "skill_trace": [],
    }
    service.store.save_snapshot(snapshot)
    task = service.create_tuning_task("asmt_model_review_1", trigger_mode="unit_test")
    service.store.update_tuning_task(
        task["task_id"],
        {
            "status": "completed",
            "updated_at": "2026-05-17T00:10:00Z",
            "result": {
                "pipeline": {
                    "completed_at": "2026-05-17T00:10:00Z",
                    "review": {"decision": "conditional_review"},
                    "tuning_summary": {
                        "pid_params": {"Kp": 1.5},
                        "evaluation": {"final_rating": 7.5, "robustness_score": 7.0},
                    },
                }
            },
        },
    )

    result = service.get_model_review_snapshot("5203_TIC_10707")

    assert result["loop_id"] == "5203_TIC_10707"
    assert result["snapshot"]["snapshot_id"] == "asmt_model_review_1"
    assert result["latest_completed_task"]["task_id"] == task["task_id"]
    assert result["review"]["decision"] == "conditional_review"
    assert result["pid_params"]["Kp"] == 1.5
    assert result["reliability_level"] in {"caution", "reliable"}
    assert [item["stage"] for item in result["evidence_chain"]] == [
        "realtime_assessment",
        "formal_metrics",
        "identification_readiness",
        "auto_tuning_result",
    ]


def test_monitor_tick_skips_when_disabled(tmp_path):
    service = RealtimeAssessmentService()
    service.store = service.store.__class__(tmp_path / "assessment.sqlite3")

    import asyncio

    result = asyncio.run(service.run_monitor_tick())

    assert result["status"] == "skipped"
    assert result["config"]["enabled"] is False
