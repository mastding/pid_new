from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.app import app


def test_realtime_assessment_routes_run_latest_and_detail(monkeypatch):
    from api import realtime_assessment_routes as routes

    snapshot = {
        "snapshot_id": "asmt_route_1",
        "loop_id": "5203_TIC_10707",
        "asset_id": "5203",
        "loop_type": "temperature",
        "created_at": "2026-05-17T00:00:00Z",
        "time_window": {"range": "8h"},
        "risk_level": "medium",
        "decision": {"decision": "observe", "need_tuning": False},
        "metrics": [],
        "diagnosis": [],
        "skill_trace": [],
    }

    async def fake_run(request):
        assert request.time_range == "8h"
        return {"total": 1, "saved": 1, "items": [snapshot], "tasks": [], "errors": []}

    monkeypatch.setattr(routes.realtime_assessment_service, "run", fake_run)
    monkeypatch.setattr(routes.realtime_assessment_service, "latest", lambda **kwargs: {"total": 1, "items": [snapshot], "summary": {}})
    monkeypatch.setattr(routes.realtime_assessment_service, "get", lambda snapshot_id: snapshot if snapshot_id == "asmt_route_1" else None)

    with TestClient(app) as client:
        run_resp = client.post("/api/realtime-assessments/run", json={"loop_ids": ["5203_TIC_10707"]})
        assert run_resp.status_code == 200
        assert run_resp.json()["items"][0]["time_window"]["range"] == "8h"

        latest_resp = client.get("/api/realtime-assessments/latest?loop_id=5203_TIC_10707")
        assert latest_resp.status_code == 200
        assert latest_resp.json()["total"] == 1

        detail_resp = client.get("/api/realtime-assessments/asmt_route_1")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["snapshot_id"] == "asmt_route_1"


def test_realtime_monitor_routes(monkeypatch):
    from api import realtime_assessment_routes as routes

    config = {
        "config_id": "default",
        "enabled": True,
        "asset_id": "5203",
        "loop_ids": [],
        "time_range": "8h",
        "interval_seconds": 900,
        "include_formal_metrics": True,
        "auto_create_tasks": True,
    }

    monkeypatch.setattr(routes.realtime_assessment_service, "get_monitor_config", lambda: config)
    monkeypatch.setattr(routes.realtime_assessment_service, "update_monitor_config", lambda updates: {**config, **updates})

    async def fake_tick(force=False):
        assert force is True
        return {"status": "completed", "config": config, "result": {"total": 0, "saved": 0, "items": [], "errors": []}}

    monkeypatch.setattr(routes.realtime_assessment_service, "run_monitor_tick", fake_tick)
    with TestClient(app) as client:
        assert client.get("/api/realtime-monitor/config").json()["time_range"] == "8h"
        update_resp = client.put("/api/realtime-monitor/config", json={"enabled": False, "interval_seconds": 300})
        assert update_resp.status_code == 200
        assert update_resp.json()["enabled"] is False
        tick_resp = client.post("/api/realtime-monitor/tick", json={"force": True})
        assert tick_resp.status_code == 200
        assert tick_resp.json()["status"] == "completed"
        scheduler_resp = client.get("/api/realtime-monitor/scheduler")
        assert scheduler_resp.status_code == 200
        assert isinstance(scheduler_resp.json()["running"], bool)
