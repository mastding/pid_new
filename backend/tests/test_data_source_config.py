from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data_sources import DataSourceConfigStore


def test_data_source_config_masks_secret(tmp_path):
    store = DataSourceConfigStore(tmp_path / "data_sources.json")

    saved = store.save({
        "items": [{
            "id": "hist_1",
            "source_name": "Historian",
            "source_type": "historian",
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8080,
            "database": "PID_HISTORY",
            "username": "reader",
            "password": "secret-token",
            "polling_interval_s": 30,
        }]
    })

    assert saved["items"][0]["secret_present"] is True
    assert saved["items"][0]["password"] == "******"
    loaded = store.load()
    assert loaded["items"][0]["password"] == "******"
    assert "secret" not in loaded["items"][0]

    retained = store.save({"items": [{**loaded["items"][0], "password": "******", "host": "10.0.0.2"}]})
    assert retained["items"][0]["secret_present"] is True
    assert retained["items"][0]["host"] == "10.0.0.2"
