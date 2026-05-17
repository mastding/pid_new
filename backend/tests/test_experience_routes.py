from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.app import app


def test_experience_routes_list_and_attach(monkeypatch):
    from api import experience_routes as routes

    monkeypatch.setattr(
        routes.experience_store,
        "list_skills",
        lambda: {"total": 1, "items": [{"skill_name": "identify_model", "snapshot_count": 1}]},
    )
    monkeypatch.setattr(
        routes.experience_store,
        "list_snapshots",
        lambda **kwargs: {
            "total": 1,
            "items": [{"skill_name": kwargs.get("skill_name"), "snapshot_id": "snap_1"}],
        },
    )
    monkeypatch.setattr(
        routes.experience_store,
        "attach_outcome",
        lambda **kwargs: {"ok": True, "snapshot": {"snapshot_id": kwargs["snapshot_id"], "observable_outcomes": kwargs["outcome"]}},
    )
    monkeypatch.setattr(
        routes.experience_store,
        "similar_loops",
        lambda **kwargs: {
            "loop_id": kwargs["loop_id"],
            "total": 1,
            "items": [{"loop_id": "5203_TIC_10107", "similarity_score": 0.82}],
        },
    )

    with TestClient(app) as client:
        skills = client.get("/api/experience/skills")
        assert skills.status_code == 200
        assert skills.json()["items"][0]["skill_name"] == "identify_model"

        snapshots = client.get("/api/experience/snapshots?skill_name=identify_model")
        assert snapshots.status_code == 200
        assert snapshots.json()["items"][0]["snapshot_id"] == "snap_1"

        outcome = client.post(
            "/api/experience/outcomes",
            json={
                "skill_name": "identify_model",
                "snapshot_id": "snap_1",
                "outcome": {"human_label": "good"},
            },
        )
        assert outcome.status_code == 200
        assert outcome.json()["snapshot"]["observable_outcomes"]["human_label"] == "good"

        similar = client.get("/api/experience/similar-loops/5203_TIC_10707")
        assert similar.status_code == 200
        assert similar.json()["items"][0]["loop_id"] == "5203_TIC_10107"
