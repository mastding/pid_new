from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.realtime.sqlite_store import RealtimeAssessmentStore


def _snapshot(snapshot_id: str = "asmt_test_1") -> dict:
    return {
        "snapshot_id": snapshot_id,
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h", "start_time": "2026-04-23 02:00:00", "end_time": "2026-04-23 10:00:00"},
        "risk_level": "medium",
        "score": 0.72,
        "ontology": {
            "context_id": "onto_test_1",
            "case_id": None,
            "source": "test",
            "missing_fields": ["pv_spec_limits.lsl"],
        },
        "metrics": [
            {"name": "harris", "value": 0.42, "level": "poor", "confidence": 0.8, "success": True},
            {"name": "cpk", "value": None, "level": "unavailable", "confidence": 0.25, "success": False},
        ],
        "diagnosis": [
            {
                "diagnosis_id": "diag_test_1",
                "root_cause": "pid_parameters",
                "confidence": 0.7,
                "severity": "medium",
                "evidence": [{"harris": 0.42}],
                "action": "review tuning",
            }
        ],
        "decision": {"decision": "tuning_recommended", "need_tuning": True, "blocked": False},
        "skill_trace": [
            {
                "trace_id": "trace_test_1",
                "skill_name": "compute_harris_closed_loop",
                "risk_level": "medium",
                "status": "completed",
                "inputs_summary": {"loop_id": "5203_TIC_10707"},
                "outputs_summary": {"eta": 0.42},
                "guard": {"allowed": True},
                "duration_ms": 12,
                "created_at": "2026-05-17T00:00:00Z",
            }
        ],
    }


def test_realtime_assessment_store_roundtrip(tmp_path):
    store = RealtimeAssessmentStore(tmp_path / "assessment.sqlite3")
    saved = store.save_snapshot(_snapshot())

    loaded = store.get_snapshot(saved["snapshot_id"])
    assert loaded is not None
    assert loaded["loop_id"] == "5203_TIC_10707"
    assert loaded["decision"]["need_tuning"] is True

    latest = store.list_latest(loop_id="5203_TIC_10707")
    assert len(latest) == 1
    assert latest[0]["snapshot_id"] == saved["snapshot_id"]


def test_realtime_assessment_store_task_roundtrip(tmp_path):
    store = RealtimeAssessmentStore(tmp_path / "assessment.sqlite3")
    store.save_snapshot(_snapshot())
    task = store.create_tuning_task({
        "task_id": "att_test_1",
        "snapshot_id": "asmt_test_1",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "status": "pending_review",
        "trigger_mode": "manual",
        "trigger_reason": "unit test",
        "created_at": "2026-05-17T00:00:00Z",
        "updated_at": "2026-05-17T00:00:00Z",
    })

    tasks = store.list_tuning_tasks(loop_id="5203_TIC_10707")
    assert tasks == [task]

    loaded = store.get_tuning_task("att_test_1")
    assert loaded == task

    updated = store.update_tuning_task(
        "att_test_1",
        {
            "status": "pending",
            "updated_at": "2026-05-17T00:01:00Z",
            "result": {"prepare": {"guard": {"allowed": True}}},
        },
    )
    assert updated is not None
    assert updated["status"] == "pending"
    assert updated["result"]["prepare"]["guard"]["allowed"] is True


def test_realtime_monitor_config_roundtrip(tmp_path):
    store = RealtimeAssessmentStore(tmp_path / "assessment.sqlite3")

    default_config = store.get_monitor_config()
    assert default_config["time_range"] == "8h"
    assert default_config["enabled"] is False

    saved = store.update_monitor_config({
        "enabled": True,
        "asset_id": "5203",
        "loop_ids": ["5203_TIC_10707"],
        "interval_seconds": 600,
        "updated_at": "2026-05-17T00:00:00Z",
    })

    loaded = store.get_monitor_config()
    assert loaded == saved
    assert loaded["enabled"] is True
    assert loaded["loop_ids"] == ["5203_TIC_10707"]
