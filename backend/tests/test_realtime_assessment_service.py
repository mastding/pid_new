from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.realtime.assessment_service import RealtimeAssessmentRequest, RealtimeAssessmentService


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
