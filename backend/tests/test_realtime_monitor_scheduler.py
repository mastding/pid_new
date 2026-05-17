from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.realtime.monitor_scheduler import monitor_tick_due


def test_monitor_tick_due_requires_enabled():
    assert monitor_tick_due({"enabled": False}, datetime(2026, 5, 17, 12, 0, 0)) is False


def test_monitor_tick_due_when_no_last_run():
    assert monitor_tick_due({"enabled": True}, datetime(2026, 5, 17, 12, 0, 0)) is True


def test_monitor_tick_due_respects_interval():
    now = datetime(2026, 5, 17, 12, 0, 0)
    recent = (now - timedelta(seconds=120)).isoformat() + "Z"
    stale = (now - timedelta(seconds=901)).isoformat() + "Z"

    assert monitor_tick_due({"enabled": True, "interval_seconds": 900, "last_run_at": recent}, now) is False
    assert monitor_tick_due({"enabled": True, "interval_seconds": 900, "last_run_at": stale}, now) is True
