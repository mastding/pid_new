"""SQLite persistence for realtime assessment snapshots.

The current project keeps imported history data on disk. Realtime assessment
results need a queryable, durable audit trail, so this module stores only the
derived snapshots and traces in SQLite while leaving raw series ownership
unchanged.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


_DB_PATH = Path(__file__).resolve().parents[2] / "var" / "realtime_assessment.sqlite3"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class RealtimeAssessmentStore:
    """Small SQLite repository for assessment snapshots and task shells."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS loop_assessment_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    loop_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    loop_type TEXT NOT NULL,
                    time_range TEXT NOT NULL,
                    time_start TEXT,
                    time_end TEXT,
                    risk_level TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    need_tuning INTEGER NOT NULL DEFAULT 0,
                    score REAL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_assessment_loop_created
                    ON loop_assessment_snapshots(loop_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_assessment_asset_created
                    ON loop_assessment_snapshots(asset_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_assessment_decision
                    ON loop_assessment_snapshots(decision, risk_level);

                CREATE TABLE IF NOT EXISTS performance_metric_results (
                    metric_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL,
                    level TEXT,
                    confidence REAL,
                    success INTEGER NOT NULL DEFAULT 0,
                    raw_json TEXT NOT NULL,
                    FOREIGN KEY(snapshot_id) REFERENCES loop_assessment_snapshots(snapshot_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_snapshot
                    ON performance_metric_results(snapshot_id, metric_name);

                CREATE TABLE IF NOT EXISTS ontology_context_snapshots (
                    context_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    loop_id TEXT NOT NULL,
                    case_id TEXT,
                    source TEXT,
                    facts_json TEXT NOT NULL,
                    missing_fields_json TEXT NOT NULL,
                    FOREIGN KEY(snapshot_id) REFERENCES loop_assessment_snapshots(snapshot_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS loop_diagnosis_results (
                    diagnosis_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    root_cause TEXT NOT NULL,
                    confidence REAL,
                    severity TEXT,
                    evidence_json TEXT NOT NULL,
                    action TEXT,
                    FOREIGN KEY(snapshot_id) REFERENCES loop_assessment_snapshots(snapshot_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS skill_execution_traces (
                    trace_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    guard_json TEXT NOT NULL,
                    duration_ms INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(snapshot_id) REFERENCES loop_assessment_snapshots(snapshot_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS auto_tuning_tasks (
                    task_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    loop_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_mode TEXT NOT NULL,
                    trigger_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    FOREIGN KEY(snapshot_id) REFERENCES loop_assessment_snapshots(snapshot_id)
                );

                CREATE TABLE IF NOT EXISTS realtime_monitor_config (
                    config_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    asset_id TEXT,
                    loop_ids_json TEXT NOT NULL,
                    time_range TEXT NOT NULL DEFAULT '8h',
                    interval_seconds INTEGER NOT NULL DEFAULT 900,
                    include_formal_metrics INTEGER NOT NULL DEFAULT 1,
                    auto_create_tasks INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_review_snapshots (
                    review_id TEXT PRIMARY KEY,
                    loop_id TEXT NOT NULL,
                    source_snapshot_id TEXT,
                    source_task_id TEXT,
                    reliability_level TEXT NOT NULL,
                    reliability_score REAL,
                    recommended_action TEXT,
                    generated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_model_review_loop_generated
                    ON model_review_snapshots(loop_id, generated_at DESC);
                """
            )

    def save_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = str(payload["snapshot_id"])
        metrics = list(payload.get("metrics") or [])
        diagnosis = list(payload.get("diagnosis") or [])
        ontology = payload.get("ontology") or {}
        traces = list(payload.get("skill_trace") or [])
        created_at = str(payload.get("created_at") or "")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO loop_assessment_snapshots (
                    snapshot_id, loop_id, asset_id, loop_type, time_range, time_start,
                    time_end, risk_level, decision, need_tuning, score, created_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    str(payload.get("loop_id") or ""),
                    str(payload.get("asset_id") or ""),
                    str(payload.get("loop_type") or "unknown"),
                    str((payload.get("time_window") or {}).get("range") or ""),
                    (payload.get("time_window") or {}).get("start_time"),
                    (payload.get("time_window") or {}).get("end_time"),
                    str(payload.get("risk_level") or "potential"),
                    str((payload.get("decision") or {}).get("decision") or "unknown"),
                    1 if (payload.get("decision") or {}).get("need_tuning") else 0,
                    payload.get("score"),
                    created_at,
                    _json_dumps(payload),
                ),
            )
            conn.execute("DELETE FROM performance_metric_results WHERE snapshot_id = ?", (snapshot_id,))
            conn.execute("DELETE FROM ontology_context_snapshots WHERE snapshot_id = ?", (snapshot_id,))
            conn.execute("DELETE FROM loop_diagnosis_results WHERE snapshot_id = ?", (snapshot_id,))
            conn.execute("DELETE FROM skill_execution_traces WHERE snapshot_id = ?", (snapshot_id,))
            for metric in metrics:
                conn.execute(
                    """
                    INSERT INTO performance_metric_results (
                        metric_id, snapshot_id, metric_name, value, level, confidence,
                        success, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{snapshot_id}:{metric.get('name')}",
                        snapshot_id,
                        str(metric.get("name") or ""),
                        metric.get("value"),
                        metric.get("level"),
                        metric.get("confidence"),
                        1 if metric.get("success") else 0,
                        _json_dumps(metric),
                    ),
                )
            conn.execute(
                """
                INSERT INTO ontology_context_snapshots (
                    context_id, snapshot_id, loop_id, case_id, source, facts_json,
                    missing_fields_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(ontology.get("context_id") or f"{snapshot_id}:ontology"),
                    snapshot_id,
                    str(payload.get("loop_id") or ""),
                    ontology.get("case_id"),
                    ontology.get("source"),
                    _json_dumps(ontology),
                    _json_dumps(ontology.get("missing_fields") or []),
                ),
            )
            for index, item in enumerate(diagnosis):
                conn.execute(
                    """
                    INSERT INTO loop_diagnosis_results (
                        diagnosis_id, snapshot_id, root_cause, confidence, severity,
                        evidence_json, action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(item.get("diagnosis_id") or f"{snapshot_id}:diag:{index}"),
                        snapshot_id,
                        str(item.get("root_cause") or "unknown"),
                        item.get("confidence"),
                        item.get("severity"),
                        _json_dumps(item.get("evidence") or []),
                        item.get("action"),
                    ),
                )
            for index, trace in enumerate(traces):
                conn.execute(
                    """
                    INSERT INTO skill_execution_traces (
                        trace_id, snapshot_id, skill_name, risk_level, status,
                        inputs_json, outputs_json, guard_json, duration_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(trace.get("trace_id") or f"{snapshot_id}:trace:{index}"),
                        snapshot_id,
                        str(trace.get("skill_name") or ""),
                        str(trace.get("risk_level") or "low"),
                        str(trace.get("status") or "completed"),
                        _json_dumps(trace.get("inputs_summary") or {}),
                        _json_dumps(trace.get("outputs_summary") or {}),
                        _json_dumps(trace.get("guard") or {}),
                        trace.get("duration_ms"),
                        str(trace.get("created_at") or created_at),
                    ),
                )
        return payload

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM loop_assessment_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if not row:
            return None
        return _json_loads(row["payload_json"], None)

    def list_latest(
        self,
        *,
        asset_id: str | None = None,
        loop_id: str | None = None,
        risk_level: str | None = None,
        decision: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if asset_id and asset_id != "all":
            clauses.append("asset_id = ?")
            params.append(asset_id)
        if loop_id:
            clauses.append("loop_id = ?")
            params.append(loop_id)
        if risk_level:
            clauses.append("risk_level = ?")
            params.append(risk_level)
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        query = f"""
            SELECT payload_json
            FROM loop_assessment_snapshots
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_json_loads(row["payload_json"], {}) for row in rows]

    def create_tuning_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO auto_tuning_tasks (
                    task_id, snapshot_id, loop_id, asset_id, status, trigger_mode,
                    trigger_reason, created_at, updated_at, payload_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["task_id"],
                    payload["snapshot_id"],
                    payload["loop_id"],
                    payload["asset_id"],
                    payload["status"],
                    payload["trigger_mode"],
                    payload.get("trigger_reason"),
                    payload["created_at"],
                    payload["updated_at"],
                    _json_dumps(payload),
                    _json_dumps(payload.get("result")) if payload.get("result") is not None else None,
                ),
            )
        return payload

    def get_tuning_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM auto_tuning_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if not row:
            return None
        return _json_loads(row["payload_json"], None)

    def update_tuning_task(self, task_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        current = self.get_tuning_task(task_id)
        if not current:
            return None
        updated = {**current, **changes}
        result = updated.get("result")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE auto_tuning_tasks
                SET status = ?,
                    trigger_reason = ?,
                    updated_at = ?,
                    payload_json = ?,
                    result_json = ?
                WHERE task_id = ?
                """,
                (
                    updated.get("status"),
                    updated.get("trigger_reason"),
                    updated.get("updated_at"),
                    _json_dumps(updated),
                    _json_dumps(result) if result is not None else None,
                    task_id,
                ),
            )
        return updated

    def list_tuning_tasks(
        self,
        *,
        status: str | None = None,
        loop_id: str | None = None,
        asset_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if loop_id:
            clauses.append("loop_id = ?")
            params.append(loop_id)
        if asset_id and asset_id != "all":
            clauses.append("asset_id = ?")
            params.append(asset_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM auto_tuning_tasks {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_json_loads(row["payload_json"], {}) for row in rows]

    def find_unfinished_tuning_task(self, loop_id: str) -> dict[str, Any] | None:
        unfinished = {"pending_review", "pending", "running"}
        tasks = self.list_tuning_tasks(loop_id=loop_id, limit=100)
        for task in tasks:
            if str(task.get("status") or "") in unfinished:
                return task
        return None

    def latest_finished_tuning_task(self, loop_id: str) -> dict[str, Any] | None:
        finished = {"completed"}
        tasks = self.list_tuning_tasks(loop_id=loop_id, limit=100)
        for task in tasks:
            if str(task.get("status") or "") in finished:
                return task
        return None

    def save_model_review_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_review_snapshots (
                    review_id, loop_id, source_snapshot_id, source_task_id,
                    reliability_level, reliability_score, recommended_action,
                    generated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(payload["review_id"]),
                    str(payload.get("loop_id") or ""),
                    (payload.get("snapshot") or {}).get("snapshot_id") if isinstance(payload.get("snapshot"), dict) else None,
                    (payload.get("latest_completed_task") or {}).get("task_id") if isinstance(payload.get("latest_completed_task"), dict) else None,
                    str(payload.get("reliability_level") or "insufficient_evidence"),
                    payload.get("reliability_score"),
                    payload.get("recommended_action"),
                    str(payload.get("generated_at") or ""),
                    _json_dumps(payload),
                ),
            )
        return payload

    def latest_model_review_snapshot(self, loop_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM model_review_snapshots
                WHERE loop_id = ?
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                (loop_id,),
            ).fetchone()
        if not row:
            return None
        return _json_loads(row["payload_json"], None)

    def list_model_review_snapshots(self, loop_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if loop_id:
            clauses.append("loop_id = ?")
            params.append(loop_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload_json
                FROM model_review_snapshots
                {where}
                ORDER BY generated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_json_loads(row["payload_json"], {}) for row in rows]

    def get_monitor_config(self, config_id: str = "default") -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM realtime_monitor_config WHERE config_id = ?",
                (config_id,),
            ).fetchone()
        if row:
            return _json_loads(row["payload_json"], {})
        return {
            "config_id": config_id,
            "enabled": False,
            "asset_id": None,
            "loop_ids": [],
            "time_range": "8h",
            "interval_seconds": 900,
            "include_formal_metrics": True,
            "auto_create_tasks": True,
            "auto_tuning_cooldown_hours": 24,
            "updated_at": "",
        }

    def update_monitor_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        config_id = str(payload.get("config_id") or "default")
        current = self.get_monitor_config(config_id)
        updated = {**current, **payload, "config_id": config_id}
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO realtime_monitor_config (
                    config_id, enabled, asset_id, loop_ids_json, time_range,
                    interval_seconds, include_formal_metrics, auto_create_tasks,
                    updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config_id,
                    1 if updated.get("enabled") else 0,
                    updated.get("asset_id"),
                    _json_dumps(updated.get("loop_ids") or []),
                    str(updated.get("time_range") or "8h"),
                    int(updated.get("interval_seconds") or 900),
                    1 if updated.get("include_formal_metrics", True) else 0,
                    1 if updated.get("auto_create_tasks", True) else 0,
                    str(updated.get("updated_at") or ""),
                    _json_dumps(updated),
                ),
            )
        return updated


realtime_assessment_store = RealtimeAssessmentStore()
