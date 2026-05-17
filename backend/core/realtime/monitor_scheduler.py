"""Background scheduler for realtime loop assessment."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any

from core.realtime.assessment_service import RealtimeAssessmentService, realtime_assessment_service

logger = logging.getLogger(__name__)


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def monitor_tick_due(config: dict[str, Any], now: datetime | None = None) -> bool:
    """Return whether the scheduler should run one monitor tick now."""
    if not config.get("enabled"):
        return False
    now = now or datetime.utcnow()
    last_run_at = _parse_iso(config.get("last_run_at"))
    if last_run_at is None:
        return True
    try:
        interval = max(60, int(config.get("interval_seconds") or 900))
    except (TypeError, ValueError):
        interval = 900
    return (now - last_run_at).total_seconds() >= interval


class RealtimeMonitorScheduler:
    """Small in-process scheduler.

    This is intentionally modest: it checks persisted config periodically and
    delegates the actual work to RealtimeAssessmentService. Deployment can later
    replace it with an external scheduler without changing assessment logic.
    """

    def __init__(self, service: RealtimeAssessmentService | None = None, poll_seconds: int = 30) -> None:
        self.service = service or realtime_assessment_service
        self.poll_seconds = max(5, int(poll_seconds))
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="realtime-monitor-scheduler")

    async def stop(self) -> None:
        if not self._task:
            return
        self._stop.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def run_once_if_due(self) -> dict[str, Any]:
        config = self.service.get_monitor_config()
        if not monitor_tick_due(config):
            return {"status": "skipped", "reason": "not due", "config": config}
        return await self.service.run_monitor_tick(force=True)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once_if_due()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Realtime monitor scheduler tick failed")
                self.service.update_monitor_config({
                    "last_scheduler_error": str(exc)[:500],
                })
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass


realtime_monitor_scheduler = RealtimeMonitorScheduler()
