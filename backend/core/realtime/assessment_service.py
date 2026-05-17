"""Realtime assessment service for ontology-backed PID loop monitoring."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.history.store import (
    assess_loop,
    compute_loop_cpk,
    compute_loop_harris,
    get_loop,
    get_loop_monitoring,
    list_loops,
)
from core.ontology_rules import resolve_loop_ontology_facts
from core.realtime.sqlite_store import realtime_assessment_store
from core.skills.realtime.decide_realtime_tuning_action_skill import decide_realtime_tuning_action
from core.skills.realtime.diagnose_realtime_assessment_skill import diagnose_realtime_assessment
from core.workflow_guard import workflow_guard


DEFAULT_AUTO_TUNING_COOLDOWN_HOURS = 24.0


@dataclass
class RealtimeAssessmentRequest:
    loop_ids: list[str] | None = None
    asset_id: str | None = None
    time_range: str = "8h"
    start_time: str | None = None
    end_time: str | None = None
    force_refresh: bool = False
    include_formal_metrics: bool = True
    auto_create_tasks: bool = False


@dataclass
class PrepareAutoTuningTaskRequest:
    confirm: bool = False
    use_llm_advisor: bool = True
    selected_window_index: int | None = None


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _asset_id_for_loop(loop: dict[str, Any]) -> str:
    loop_id = str(loop.get("loop_id") or "")
    if "_" in loop_id:
        return loop_id.split("_", 1)[0]
    if "-" in loop_id:
        return loop_id.split("-", 1)[0]
    return str(loop.get("asset_id") or loop.get("source_filename") or "default")


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _range_delta(time_range: str) -> timedelta:
    text = (time_range or "8h").strip().lower()
    try:
        if text.endswith("h"):
            return timedelta(hours=float(text[:-1] or 8))
        if text.endswith("d"):
            return timedelta(days=float(text[:-1] or 1))
        if text.endswith("m"):
            return timedelta(minutes=float(text[:-1] or 60))
    except ValueError:
        pass
    return timedelta(hours=8)


def _loop_window(loop: dict[str, Any], request: RealtimeAssessmentRequest) -> tuple[str | None, str | None]:
    if request.start_time or request.end_time:
        return request.start_time, request.end_time
    end_dt = _parse_dt(loop.get("end_time")) or datetime.utcnow()
    start_dt = end_dt - _range_delta(request.time_range)
    return start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S")


def _hours_since(value: Any) -> float | None:
    dt = _parse_dt(value)
    if not dt:
        return None
    return max(0.0, (datetime.utcnow() - dt).total_seconds() / 3600.0)


def _clamp01(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (number == number):
        return None
    return max(0.0, min(1.0, number))


def _score_from_10(value: Any) -> float | None:
    try:
        return _clamp01(float(value) / 10.0)
    except (TypeError, ValueError):
        return None


def _metric_by_name(snapshot: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not snapshot:
        return None
    for metric in snapshot.get("metrics") or []:
        if str(metric.get("name") or "") == name:
            return metric
    return None


def _trace(
    *,
    snapshot_id: str,
    skill_name: str,
    risk_level: str,
    status: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "trace_id": f"{snapshot_id}:{skill_name}:{uuid.uuid4().hex[:8]}",
        "skill_name": skill_name,
        "risk_level": risk_level,
        "status": status,
        "inputs_summary": inputs,
        "outputs_summary": outputs,
        "guard": {"allowed": True, "reason": "system scheduled assessment"},
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "created_at": _now_iso(),
    }


class RealtimeAssessmentService:
    def __init__(self) -> None:
        self.store = realtime_assessment_store

    async def run(self, request: RealtimeAssessmentRequest) -> dict[str, Any]:
        loops = self._select_loops(request)
        snapshots = []
        tasks = []
        errors = []
        for loop in loops:
            try:
                snapshot = await self.run_one(str(loop["loop_id"]), request)
                snapshots.append(snapshot)
                if request.auto_create_tasks and (snapshot.get("decision") or {}).get("need_tuning"):
                    tasks.append(self.create_tuning_task(
                        str(snapshot["snapshot_id"]),
                        confirm=False,
                        trigger_mode="realtime_assessment",
                        reason=(snapshot.get("decision") or {}).get("summary"),
                    ))
            except Exception as exc:
                errors.append({"loop_id": loop.get("loop_id"), "error": str(exc)[:500]})
        return {
            "total": len(loops),
            "saved": len(snapshots),
            "items": snapshots,
            "tasks": tasks,
            "errors": errors,
        }

    async def run_one(self, loop_id: str, request: RealtimeAssessmentRequest) -> dict[str, Any]:
        loop = get_loop(loop_id)
        if not loop:
            raise ValueError("loop_id not found")
        asset_id = _asset_id_for_loop(loop)
        loop_type = str(loop.get("loop_type") or "unknown")
        start_time, end_time = _loop_window(loop, request)
        created_at = _now_iso()
        snapshot_id = f"asmt_{loop_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        traces: list[dict[str, Any]] = []

        started = time.perf_counter()
        monitoring = get_loop_monitoring(loop_id, start_time=start_time, end_time=end_time)
        traces.append(_trace(
            snapshot_id=snapshot_id,
            skill_name="assess_loop_monitoring",
            risk_level="low",
            status="completed" if not monitoring.get("error") else "failed",
            inputs={"loop_id": loop_id, "start_time": start_time, "end_time": end_time},
            outputs={"status": (monitoring.get("monitoring") or {}).get("status"), "error": monitoring.get("error")},
            started=started,
        ))
        if monitoring.get("error"):
            raise ValueError(str(monitoring.get("error")))

        started = time.perf_counter()
        assessment = assess_loop(loop_id, start_time=start_time, end_time=end_time)
        traces.append(_trace(
            snapshot_id=snapshot_id,
            skill_name="assess_loop_assessment",
            risk_level="medium",
            status="completed" if not assessment.get("error") else "failed",
            inputs={"loop_id": loop_id, "start_time": start_time, "end_time": end_time},
            outputs={"decision": (assessment.get("summary") or {}).get("decision"), "error": assessment.get("error")},
            started=started,
        ))

        started = time.perf_counter()
        ontology_facts = await resolve_loop_ontology_facts(
            loop_id=loop_id,
            loop_type=loop_type,
            force_refresh=request.force_refresh,
        )
        spec_limits = ontology_facts.get("pv_spec_limits") or {}
        missing_fields = []
        if not ontology_facts.get("case_id"):
            missing_fields.append("operating_case.case_id")
        if spec_limits.get("lsl") is None:
            missing_fields.append("pv_spec_limits.lsl")
        if spec_limits.get("usl") is None:
            missing_fields.append("pv_spec_limits.usl")
        ontology = {
            "context_id": f"onto_{snapshot_id}",
            "case_id": ontology_facts.get("case_id"),
            "source": (ontology_facts.get("raw_context") or {}).get("source") or "registered_mcp_tool",
            "facts": ontology_facts,
            "pv_spec_limits": spec_limits,
            "relation_hints": ontology_facts.get("relation_hints") or [],
            "missing_fields": missing_fields,
        }
        traces.append(_trace(
            snapshot_id=snapshot_id,
            skill_name="resolve_loop_ontology_context",
            risk_level="medium",
            status="completed",
            inputs={"loop_id": loop_id, "loop_type": loop_type, "force_refresh": request.force_refresh},
            outputs={"case_id": ontology.get("case_id"), "missing_fields": missing_fields},
            started=started,
        ))

        metrics: list[dict[str, Any]] = []
        harris_metric = None
        cpk_metric = None
        if request.include_formal_metrics:
            started = time.perf_counter()
            cpk_result = compute_loop_cpk(
                loop_id,
                start_time=start_time,
                end_time=end_time,
                spec_limits=spec_limits if spec_limits.get("lsl") is not None and spec_limits.get("usl") is not None else None,
            )
            cpk = cpk_result.get("cpk") or {}
            cpk_metric = {
                "name": "cpk",
                "value": cpk.get("value"),
                "level": cpk.get("level"),
                "confidence": 0.85 if cpk_result.get("success") else 0.25,
                "success": bool(cpk_result.get("success")),
                "raw": cpk_result,
            }
            metrics.append(cpk_metric)
            traces.append(_trace(
                snapshot_id=snapshot_id,
                skill_name="compute_cpk",
                risk_level="medium",
                status="completed" if not cpk_result.get("error") else "failed",
                inputs={"loop_id": loop_id, "spec_limits_source": (cpk_result.get("limits") or {}).get("source")},
                outputs={"value": cpk.get("value"), "level": cpk.get("level"), "success": cpk_result.get("success")},
                started=started,
            ))

            started = time.perf_counter()
            harris_result = compute_loop_harris(loop_id, start_time=start_time, end_time=end_time)
            harris = harris_result.get("harris") or {}
            harris_metric = {
                "name": "harris",
                "value": harris.get("eta"),
                "level": harris.get("level"),
                "confidence": harris.get("confidence"),
                "success": bool(harris_result.get("success")),
                "raw": harris_result,
            }
            metrics.append(harris_metric)
            traces.append(_trace(
                snapshot_id=snapshot_id,
                skill_name="compute_harris_closed_loop",
                risk_level="medium",
                status="completed" if not harris_result.get("error") else "failed",
                inputs={"loop_id": loop_id, "error_basis": "auto"},
                outputs={"eta": harris.get("eta"), "level": harris.get("level"), "success": harris_result.get("success")},
                started=started,
            ))

        started = time.perf_counter()
        diagnosis = diagnose_realtime_assessment(
            assessment=assessment,
            monitoring=monitoring,
            harris_metric=harris_metric,
            cpk_metric=cpk_metric,
            ontology=ontology,
        )
        traces.append(_trace(
            snapshot_id=snapshot_id,
            skill_name="diagnose_realtime_assessment",
            risk_level="medium",
            status="completed",
            inputs={"snapshot_id": snapshot_id},
            outputs={"primary": diagnosis[0] if diagnosis else None, "count": len(diagnosis)},
            started=started,
        ))

        started = time.perf_counter()
        decision_result = decide_realtime_tuning_action(
            monitoring=monitoring,
            assessment=assessment,
            diagnosis=diagnosis,
            harris_metric=harris_metric,
            cpk_metric=cpk_metric,
        )
        risk_level = str(decision_result.get("risk_level") or "potential")
        score = decision_result.get("score")
        decision = decision_result["decision"]
        traces.append(_trace(
            snapshot_id=snapshot_id,
            skill_name="decide_realtime_tuning_action",
            risk_level="high",
            status="completed",
            inputs={"snapshot_id": snapshot_id, "diagnosis_count": len(diagnosis)},
            outputs={"decision": decision.get("decision"), "risk_level": risk_level, "need_tuning": decision.get("need_tuning")},
            started=started,
        ))

        payload = {
            "snapshot_id": snapshot_id,
            "loop_id": loop_id,
            "asset_id": asset_id,
            "loop_type": loop_type,
            "created_at": created_at,
            "time_window": {"range": request.time_range, "start_time": start_time, "end_time": end_time},
            "risk_level": risk_level,
            "score": score,
            "ontology": ontology,
            "metrics": metrics,
            "monitoring": monitoring,
            "assessment": assessment,
            "diagnosis": diagnosis,
            "decision": decision,
            "skill_trace": traces,
        }
        return self.store.save_snapshot(payload)

    def latest(
        self,
        *,
        asset_id: str | None = None,
        loop_id: str | None = None,
        risk_level: str | None = None,
        decision: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        items = self.store.list_latest(
            asset_id=asset_id,
            loop_id=loop_id,
            risk_level=risk_level,
            decision=decision,
            limit=limit,
        )
        return {"total": len(items), "items": items, "summary": self._summary(items)}

    def get(self, snapshot_id: str) -> dict[str, Any] | None:
        return self.store.get_snapshot(snapshot_id)

    def create_tuning_task(
        self,
        snapshot_id: str,
        *,
        confirm: bool = False,
        trigger_mode: str = "manual",
        reason: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.store.get_snapshot(snapshot_id)
        if not snapshot:
            raise ValueError("snapshot_id not found")
        decision = snapshot.get("decision") or {}
        loop_id = str(snapshot.get("loop_id") or "")
        unfinished_task = self.store.find_unfinished_tuning_task(loop_id) if loop_id else None
        config = self.get_monitor_config()
        cooldown_hours = float(config.get("auto_tuning_cooldown_hours") or DEFAULT_AUTO_TUNING_COOLDOWN_HOURS)
        latest_finished_task = self.store.latest_finished_tuning_task(loop_id) if loop_id else None
        finished_at = ((latest_finished_task or {}).get("result") or {}).get("pipeline", {}).get("completed_at")
        since_finished_hours = _hours_since(finished_at or (latest_finished_task or {}).get("updated_at"))
        cooldown_active = (
            cooldown_hours > 0
            and since_finished_hours is not None
            and since_finished_hours < cooldown_hours
        )
        guard = workflow_guard.check_action(
            "create_auto_tuning_task",
            risk_level="high",
            preconditions={
                "snapshot_exists": True,
                "decision_recommends_tuning": bool(decision.get("need_tuning")),
                "source_assessment_not_blocked": not bool(decision.get("blocked")),
                "no_unfinished_auto_tuning_task": unfinished_task is None,
                "cooldown_elapsed": not cooldown_active,
            },
            initiated_by="system",
        ).to_dict()
        if unfinished_task:
            existing = {
                **unfinished_task,
                "reused_existing": True,
                "guard": guard,
                "duplicate_reason": "unfinished task exists for loop",
            }
            return existing
        if cooldown_active:
            return {
                "task_id": f"skipped_{snapshot.get('loop_id')}_{int(time.time())}_{uuid.uuid4().hex[:6]}",
                "snapshot_id": snapshot_id,
                "loop_id": snapshot.get("loop_id"),
                "asset_id": snapshot.get("asset_id"),
                "status": "skipped",
                "trigger_mode": trigger_mode,
                "trigger_reason": reason or decision.get("summary"),
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "guard": guard,
                "cooldown": {
                    "active": True,
                    "cooldown_hours": cooldown_hours,
                    "since_finished_hours": since_finished_hours,
                    "latest_task_id": (latest_finished_task or {}).get("task_id"),
                },
                "assessment_decision": decision,
            }
        status = "pending_review"
        if not guard["allowed"]:
            status = "blocked"
        elif confirm and decision.get("need_tuning"):
            status = "pending"
        payload = {
            "task_id": f"att_{snapshot.get('loop_id')}_{int(time.time())}_{uuid.uuid4().hex[:6]}",
            "snapshot_id": snapshot_id,
            "loop_id": snapshot.get("loop_id"),
            "asset_id": snapshot.get("asset_id"),
            "status": status,
            "trigger_mode": trigger_mode,
            "trigger_reason": reason or decision.get("summary"),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "guard": guard,
            "assessment_decision": decision,
            "source_snapshot": {
                "risk_level": snapshot.get("risk_level"),
                "time_window": snapshot.get("time_window"),
                "primary_diagnosis": (snapshot.get("diagnosis") or [None])[0],
            },
        }
        return self.store.create_tuning_task(payload)

    def prepare_tuning_task(
        self,
        task_id: str,
        request: PrepareAutoTuningTaskRequest | None = None,
    ) -> dict[str, Any]:
        request = request or PrepareAutoTuningTaskRequest()
        task = self.store.get_tuning_task(task_id)
        if not task:
            raise ValueError("task_id not found")
        snapshot = self.store.get_snapshot(str(task.get("snapshot_id") or ""))
        if not snapshot:
            raise ValueError("source snapshot not found")

        decision = snapshot.get("decision") or {}
        blocked = bool(decision.get("blocked")) or task.get("status") == "blocked"
        need_tuning = bool(decision.get("need_tuning"))
        requires_confirmation = need_tuning and not request.confirm
        action_guard = workflow_guard.check_action(
            "prepare_auto_tuning_task",
            risk_level="high",
            preconditions={
                "task_exists": True,
                "source_snapshot_exists": True,
                "decision_recommends_tuning": need_tuning,
                "source_assessment_not_blocked": not blocked,
                "engineer_confirmed": bool(request.confirm),
            },
            initiated_by="system",
        ).to_dict()
        guard = {
            **action_guard,
            "allowed": action_guard["allowed"],
            "blocked": blocked,
            "requires_confirmation": requires_confirmation,
            "reason": (
                "source assessment is blocked"
                if blocked else
                "engineer confirmation is required before tuning"
                if requires_confirmation else
                "source assessment does not recommend tuning"
                if not need_tuning else
                "ready to start tuning pipeline"
            ),
        }

        window = snapshot.get("time_window") or {}
        ontology = snapshot.get("ontology") or {}
        tuning_request = {
            "loop_id": snapshot.get("loop_id"),
            "loop_type": snapshot.get("loop_type"),
            "loop_name": snapshot.get("loop_id"),
            "start_time": window.get("start_time"),
            "end_time": window.get("end_time"),
            "selected_window_index": request.selected_window_index,
            "use_llm_advisor": request.use_llm_advisor,
            "ontology_context": {
                "snapshot_id": snapshot.get("snapshot_id"),
                "case_id": ontology.get("case_id"),
                "facts": ontology.get("facts"),
                "missing_fields": ontology.get("missing_fields") or [],
                "decision": decision,
                "primary_diagnosis": (snapshot.get("diagnosis") or [None])[0],
                "metrics": snapshot.get("metrics") or [],
            },
        }
        result = {
            "prepared_at": _now_iso(),
            "guard": guard,
            "tuning_request": tuning_request,
        }

        next_status = task.get("status")
        if blocked:
            next_status = "blocked"
        elif guard["allowed"]:
            next_status = "pending"
        elif need_tuning:
            next_status = "pending_review"

        updated = self.store.update_tuning_task(
            task_id,
            {
                "status": next_status,
                "updated_at": _now_iso(),
                "result": {**(task.get("result") or {}), "prepare": result},
            },
        )
        return {"task": updated or task, "guard": guard, "tuning_request": tuning_request}

    def list_tuning_tasks(
        self,
        *,
        status: str | None = None,
        loop_id: str | None = None,
        asset_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        items = self.store.list_tuning_tasks(status=status, loop_id=loop_id, asset_id=asset_id, limit=limit)
        return {"total": len(items), "items": items}

    def get_tuning_task_result(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_tuning_task(task_id)
        if not task:
            raise ValueError("task_id not found")
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        pipeline = result.get("pipeline") if isinstance(result, dict) else {}
        return {
            "task": task,
            "review": pipeline.get("review") if isinstance(pipeline, dict) else None,
            "tuning_summary": pipeline.get("tuning_summary") if isinstance(pipeline, dict) else None,
            "pipeline": pipeline if isinstance(pipeline, dict) else {},
        }

    def get_model_review_snapshot(self, loop_id: str) -> dict[str, Any]:
        snapshots = self.store.list_latest(loop_id=loop_id, limit=1)
        snapshot = snapshots[0] if snapshots else None
        latest_completed_task = self.store.latest_finished_tuning_task(loop_id)
        task_result = latest_completed_task.get("result") if isinstance(latest_completed_task, dict) else {}
        pipeline = task_result.get("pipeline") if isinstance(task_result, dict) else {}
        review = pipeline.get("review") if isinstance(pipeline, dict) else None
        tuning_summary = pipeline.get("tuning_summary") if isinstance(pipeline, dict) else None
        evaluation = tuning_summary.get("evaluation") if isinstance(tuning_summary, dict) else None
        pid_params = tuning_summary.get("pid_params") if isinstance(tuning_summary, dict) else None

        assessment = snapshot.get("assessment") if isinstance(snapshot, dict) else {}
        data_quality = assessment.get("data_quality") if isinstance(assessment, dict) else {}
        identifiability = (
            assessment.get("identification_suitability")
            or assessment.get("identifiability")
            or {}
        ) if isinstance(assessment, dict) else {}
        readiness = (
            assessment.get("tuning_readiness")
            or assessment.get("readiness")
            or {}
        ) if isinstance(assessment, dict) else {}
        final_rating = evaluation.get("final_rating") if isinstance(evaluation, dict) else None
        robustness_score = evaluation.get("robustness_score") if isinstance(evaluation, dict) else None
        review_decision = review.get("decision") if isinstance(review, dict) else None

        score_inputs = [
            _clamp01(data_quality.get("score") if isinstance(data_quality, dict) else None),
            _clamp01(identifiability.get("score") if isinstance(identifiability, dict) else None),
            _clamp01(identifiability.get("excitation_score") if isinstance(identifiability, dict) else None),
            _clamp01(identifiability.get("response_observability_score") if isinstance(identifiability, dict) else None),
            _clamp01(readiness.get("score") if isinstance(readiness, dict) else None),
            _score_from_10(final_rating),
            _score_from_10(robustness_score),
        ]
        usable_scores = [value for value in score_inputs if value is not None]
        reliability_score = round(sum(usable_scores) / len(usable_scores), 4) if usable_scores else None
        if review_decision == "revise_required":
            reliability_level = "unreliable"
        elif reliability_score is None:
            reliability_level = "insufficient_evidence"
        elif reliability_score >= 0.82:
            reliability_level = "reliable"
        elif reliability_score >= 0.65:
            reliability_level = "caution"
        else:
            reliability_level = "unreliable"

        harris = _metric_by_name(snapshot, "harris")
        cpk = _metric_by_name(snapshot, "cpk")
        evidence_chain = [
            {
                "stage": "realtime_assessment",
                "available": bool(snapshot),
                "snapshot_id": snapshot.get("snapshot_id") if isinstance(snapshot, dict) else None,
                "created_at": snapshot.get("created_at") if isinstance(snapshot, dict) else None,
                "risk_level": snapshot.get("risk_level") if isinstance(snapshot, dict) else None,
                "decision": (snapshot.get("decision") or {}).get("decision") if isinstance(snapshot, dict) else None,
            },
            {
                "stage": "formal_metrics",
                "available": bool(harris or cpk),
                "harris": harris,
                "cpk": cpk,
            },
            {
                "stage": "identification_readiness",
                "available": bool(identifiability),
                "score": identifiability.get("score") if isinstance(identifiability, dict) else None,
                "excitation_score": identifiability.get("excitation_score") if isinstance(identifiability, dict) else None,
                "response_observability_score": identifiability.get("response_observability_score") if isinstance(identifiability, dict) else None,
                "window_count": identifiability.get("window_count") if isinstance(identifiability, dict) else None,
                "usable_window_count": identifiability.get("usable_window_count") if isinstance(identifiability, dict) else None,
            },
            {
                "stage": "auto_tuning_result",
                "available": bool(latest_completed_task),
                "task_id": latest_completed_task.get("task_id") if isinstance(latest_completed_task, dict) else None,
                "completed_at": pipeline.get("completed_at") if isinstance(pipeline, dict) else None,
                "final_rating": final_rating,
                "review_decision": review_decision,
            },
        ]
        recommended_action = "collect_more_evidence"
        if reliability_level == "reliable":
            recommended_action = "allow_engineer_review"
        elif reliability_level == "caution":
            recommended_action = "use_conservative_review"
        elif review_decision == "revise_required":
            recommended_action = "rerun_tuning_or_fix_model"

        generated_at = _now_iso()
        review_snapshot = {
            "review_id": f"mrs_{loop_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}",
            "loop_id": loop_id,
            "generated_at": generated_at,
            "reliability_score": reliability_score,
            "reliability_level": reliability_level,
            "recommended_action": recommended_action,
            "snapshot": snapshot,
            "latest_completed_task": latest_completed_task,
            "review": review if isinstance(review, dict) else None,
            "tuning_summary": tuning_summary if isinstance(tuning_summary, dict) else None,
            "pid_params": pid_params if isinstance(pid_params, dict) else None,
            "evaluation": evaluation if isinstance(evaluation, dict) else None,
            "evidence_chain": evidence_chain,
        }
        return self.store.save_model_review_snapshot(review_snapshot)

    def list_model_review_snapshots(self, *, loop_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        items = self.store.list_model_review_snapshots(loop_id=loop_id, limit=limit)
        return {"total": len(items), "items": items}

    def get_monitor_config(self) -> dict[str, Any]:
        return self.store.get_monitor_config()

    def update_monitor_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        payload = {
            **updates,
            "updated_at": _now_iso(),
        }
        return self.store.update_monitor_config(payload)

    async def run_monitor_tick(self, *, force: bool = False) -> dict[str, Any]:
        config = self.get_monitor_config()
        if not force and not config.get("enabled"):
            return {
                "status": "skipped",
                "reason": "realtime monitor is disabled",
                "config": config,
            }
        request = RealtimeAssessmentRequest(
            loop_ids=list(config.get("loop_ids") or []) or None,
            asset_id=config.get("asset_id"),
            time_range=str(config.get("time_range") or "8h"),
            include_formal_metrics=bool(config.get("include_formal_metrics", True)),
            auto_create_tasks=bool(config.get("auto_create_tasks", True)),
        )
        result = await self.run(request)
        updated = self.update_monitor_config({"last_run_at": _now_iso(), "last_result": {
            "total": result.get("total"),
            "saved": result.get("saved"),
            "task_count": len(result.get("tasks") or []),
            "error_count": len(result.get("errors") or []),
        }})
        return {"status": "completed", "config": updated, "result": result}

    def _select_loops(self, request: RealtimeAssessmentRequest) -> list[dict[str, Any]]:
        if request.loop_ids:
            loops = []
            for loop_id in request.loop_ids:
                loop = get_loop(loop_id)
                if loop:
                    loops.append(loop)
            return loops
        asset_id = request.asset_id
        loops = list_loops()
        if asset_id and asset_id != "all":
            loops = [loop for loop in loops if _asset_id_for_loop(loop) == asset_id]
        return loops

    def _summary(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        levels = {"high": 0, "medium": 0, "low": 0, "potential": 0, "normal": 0}
        decisions: dict[str, int] = {}
        assets: dict[str, int] = {}
        for item in items:
            levels[str(item.get("risk_level") or "potential")] = levels.get(str(item.get("risk_level") or "potential"), 0) + 1
            dec = str((item.get("decision") or {}).get("decision") or "unknown")
            decisions[dec] = decisions.get(dec, 0) + 1
            asset = str(item.get("asset_id") or "default")
            assets[asset] = assets.get(asset, 0) + 1
        return {"risk_levels": levels, "decisions": decisions, "assets": assets}


realtime_assessment_service = RealtimeAssessmentService()
