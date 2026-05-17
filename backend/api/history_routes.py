"""Historical loop data API endpoints."""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.model_config import store as model_cfg_store
from core.history.store import (
    assess_loop,
    compute_loop_cpk,
    compute_loop_harris,
    get_loop_features,
    get_loop_monitoring,
    get_loop,
    get_loop_profile_bundle,
    import_history_file,
    list_loop_windows,
    list_loops,
    load_loop_series,
)
from core.pipeline.ontology_mcp_context import fetch_loop_ontology_context_via_mcp
from core.ontology_rules import evaluate_loop_ontology_rules, resolve_loop_ontology_facts
from core.pipeline.runner import run_tuning_pipeline
from core.session_log import record_stream
from models import TuningRequest

router = APIRouter(tags=["history"])


_RISK_RANGE_SECONDS = {
    "8h": 8 * 3600,
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
}


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("T", " ").split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _format_dt(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else None


def _risk_asset_name(loop: dict[str, Any]) -> str:
    loop_id = str(loop.get("loop_id") or "")
    if loop_id.startswith("5203_"):
        return "分馏/回流系统"
    prefix = loop_id.split("_", 1)[0] if "_" in loop_id else ""
    if prefix:
        return f"{prefix}装置"
    return "未归属"


def _loop_time_window(loop: dict[str, Any], time_range: str, start_time: str | None, end_time: str | None) -> tuple[str | None, str | None]:
    if start_time or end_time:
        return start_time, end_time
    seconds = _RISK_RANGE_SECONDS.get(time_range)
    if not seconds:
        return None, None
    end_dt = _parse_dt(loop.get("end_time")) or datetime.now()
    return _format_dt(end_dt - timedelta(seconds=seconds)), _format_dt(end_dt)


def _risk_level_from_severity(value: str | None) -> str:
    text = str(value or "").lower()
    if text in {"critical", "alarm", "high"}:
        return "high"
    if text in {"warning", "warn", "medium"}:
        return "medium"
    if text in {"low", "info"}:
        return "low"
    return "potential"


def _risk_level_from_score(score: float | None) -> str:
    if score is None:
        return "potential"
    if score < 0.45:
        return "high"
    if score < 0.65:
        return "medium"
    if score < 0.82:
        return "low"
    return "potential"


def _risk_type_label(type_key: str) -> str:
    labels = {
        "data_quality": "数据质量风险",
        "data_health": "数据质量风险",
        "stability": "稳定性风险",
        "oscillation": "振荡风险",
        "pv_mv_behavior": "阀门/执行机构风险",
        "actuator": "阀门/执行机构风险",
        "constraint": "约束/饱和风险",
        "constraints": "约束/饱和风险",
        "tracking": "设定值跟踪风险",
        "response_observability": "模型/响应风险",
        "operating_condition": "工况波动风险",
        "noise": "测量噪声风险",
        "monitoring": "综合监控风险",
    }
    return labels.get(type_key, type_key or "综合监控风险")


def _risk_candidates(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for event in snapshot.get("events") or []:
        if not isinstance(event, dict):
            continue
        candidates.append({
            "type": str(event.get("type") or "monitoring"),
            "level": _risk_level_from_severity(str(event.get("severity") or "")),
            "message": event.get("message") or event.get("name") or "监控事件触发风险",
            "trigger": event.get("name") or event.get("type") or "监控事件",
            "score": None,
        })
    for alert in snapshot.get("alerts") or []:
        if not isinstance(alert, dict):
            continue
        candidates.append({
            "type": str(alert.get("type") or "monitoring"),
            "level": _risk_level_from_severity(str(alert.get("severity") or "")),
            "message": alert.get("message") or "监控告警触发风险",
            "trigger": alert.get("type") or "监控告警",
            "score": None,
        })

    indicator_specs = [
        ("data_health", "数据质量指标偏低"),
        ("stability", "稳定性指标偏低"),
        ("pv_mv_behavior", "PV/MV 行为指标偏低"),
        ("constraints", "约束健康指标偏低"),
        ("tracking", "设定值跟踪指标偏低"),
        ("response_observability", "响应可观测性偏低"),
    ]
    for key, label in indicator_specs:
        block = snapshot.get(key) or {}
        if not isinstance(block, dict):
            continue
        score = block.get("score")
        try:
            score_value = None if score is None else float(score)
        except Exception:
            score_value = None
        if score_value is None or score_value >= 0.95:
            continue
        candidates.append({
            "type": key,
            "level": _risk_level_from_score(score_value),
            "message": f"{label}，当前得分 {round(score_value * 100)} 分，建议结合趋势、频谱和控制性能核对原因。",
            "trigger": label,
            "score": score_value,
        })

    operating = snapshot.get("operating_condition") or {}
    if isinstance(operating, dict):
        suitability = str(operating.get("tuning_suitability") or "")
        if suitability in {"cautious", "not_recommended"}:
            candidates.append({
                "type": "operating_condition",
                "level": "medium" if suitability == "cautious" else "high",
                "message": f"当前工况 {operating.get('condition_label') or '-'}，整定适宜性为 {suitability}。",
                "trigger": "运行工况",
                "score": operating.get("confidence"),
            })

    overall_score = snapshot.get("overall_score")
    try:
        overall_value = None if overall_score is None else float(overall_score)
    except Exception:
        overall_value = None
    if overall_value is not None and overall_value < 0.92 and not candidates:
        candidates.append({
            "type": "monitoring",
            "level": _risk_level_from_score(overall_value),
            "message": f"综合监控评分 {round(overall_value * 100)} 分，建议纳入风险巡检并查看分项指标。",
            "trigger": "综合监控评分",
            "score": overall_value,
        })

    return candidates


def _risk_level_rank(level: str) -> int:
    return {"high": 4, "medium": 3, "low": 2, "potential": 1, "handled": 0}.get(level, 0)


def _risk_score(snapshot: dict[str, Any], level: str, alert_count: int, indicator_score: float | None) -> int:
    score = indicator_score if indicator_score is not None else snapshot.get("overall_score")
    try:
        score_value = float(score)
    except Exception:
        score_value = 0.72
    base = round((1.0 - max(0.0, min(1.0, score_value))) * 100)
    bonus = {"high": 42, "medium": 28, "low": 16, "potential": 8}.get(level, 6)
    return max(1, min(100, base + bonus + alert_count * 5))


def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


@router.post("/history/import")
async def import_history(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Import one or more historical loop files into the offline repository."""
    dataset_id = uuid.uuid4().hex[:12]
    imported: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for file in files:
        tmp_path = _save_upload(file)
        try:
            imported.extend(import_history_file(tmp_path, file.filename or "uploaded", dataset_id))
        except Exception as exc:
            errors.append({"filename": file.filename or "uploaded", "error": str(exc)})

    return {
        "dataset_id": dataset_id,
        "imported_count": len(imported),
        "loops": imported,
        "errors": errors,
    }


@router.get("/history/loops")
def history_loops() -> dict[str, Any]:
    loops = list_loops()
    return {"total": len(loops), "items": loops}


@router.get("/history/loops/{loop_id}")
def history_loop(loop_id: str) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")
    return loop


@router.get("/history/loops/{loop_id}/series")
def history_loop_series(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    max_points: int = 4000,
) -> dict[str, Any]:
    return load_loop_series(
        loop_id,
        start_time=start_time,
        end_time=end_time,
        max_points=max_points,
    )


@router.get("/history/loops/{loop_id}/features")
def history_loop_features(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = get_loop_features(loop_id, start_time=start_time, end_time=end_time)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/loops/{loop_id}/monitoring")
def history_loop_monitoring(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = get_loop_monitoring(loop_id, start_time=start_time, end_time=end_time)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/risk-alerts")
def history_risk_alerts(
    asset: str | None = None,
    time_range: str = "8h",
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    loops = list_loops()
    asset_filter = None if not asset or asset == "all" else asset
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    loaded_count = 0

    for loop in loops:
        asset_name = _risk_asset_name(loop)
        if asset_filter and asset_name != asset_filter:
            continue
        loop_start, loop_end = _loop_time_window(loop, time_range, start_time, end_time)
        snapshot_payload = get_loop_monitoring(
            str(loop.get("loop_id")),
            start_time=loop_start,
            end_time=loop_end,
        )
        if snapshot_payload.get("error"):
            errors.append({"loop_id": loop.get("loop_id"), "error": snapshot_payload.get("error")})
            continue
        loaded_count += 1
        snapshot = snapshot_payload.get("monitoring") or {}
        candidates = _risk_candidates(snapshot)
        if not candidates:
            continue
        alert_count = len(snapshot.get("alerts") or [])
        candidate = sorted(
            candidates,
            key=lambda item: (
                _risk_level_rank(str(item.get("level") or "potential")),
                _risk_score(snapshot, str(item.get("level") or "potential"), alert_count, item.get("score")),
            ),
            reverse=True,
        )[0]
        level = str(candidate.get("level") or "potential")
        risk_score = _risk_score(snapshot, level, alert_count, candidate.get("score"))
        rows.append({
            "key": str(loop.get("loop_id")),
            "level": level,
            "type": str(candidate.get("type") or "monitoring"),
            "type_label": _risk_type_label(str(candidate.get("type") or "monitoring")),
            "loop_id": loop.get("loop_id"),
            "loop_type": loop.get("loop_type", "unknown"),
            "asset": asset_name,
            "description": candidate.get("message") or "监控指标触发风险提示。",
            "trigger": candidate.get("trigger") or "监控快照",
            "duration": f"{loop_start or '-'} ~ {loop_end or '-'}",
            "risk_score": risk_score,
            "found_at": loop_end or loop.get("end_time"),
            "alert_count": alert_count,
            "overall_score": snapshot.get("overall_score"),
            "status": snapshot.get("status"),
        })

    rows.sort(key=lambda item: int(item.get("risk_score") or 0), reverse=True)
    levels = ("high", "medium", "low", "potential", "handled")
    counts = {level: sum(1 for row in rows if row.get("level") == level) for level in levels}
    type_distribution: dict[str, int] = {}
    asset_distribution: dict[str, int] = {}
    trend_map: dict[str, dict[str, int]] = {}
    for row in rows:
        type_label = str(row.get("type_label") or "-")
        asset_name = str(row.get("asset") or "-")
        type_distribution[type_label] = type_distribution.get(type_label, 0) + 1
        asset_distribution[asset_name] = asset_distribution.get(asset_name, 0) + 1
        found_at = _parse_dt(row.get("found_at"))
        bucket = found_at.strftime("%m-%d") if found_at else "未标记"
        trend_map.setdefault(bucket, {level: 0 for level in levels})
        trend_map[bucket][str(row.get("level") or "potential")] += 1

    asset_options = [{"label": name, "value": name} for name in sorted({_risk_asset_name(loop) for loop in loops})]
    return {
        "time_range": time_range,
        "start_time": start_time,
        "end_time": end_time,
        "asset": asset or "all",
        "total_loops": len(loops) if not asset_filter else sum(1 for loop in loops if _risk_asset_name(loop) == asset_filter),
        "loaded_count": loaded_count,
        "total_risk": len(rows),
        "counts": counts,
        "items": rows,
        "type_distribution": [{"label": label, "value": value} for label, value in sorted(type_distribution.items(), key=lambda item: item[1], reverse=True)],
        "asset_distribution": [{"label": label, "value": value} for label, value in sorted(asset_distribution.items(), key=lambda item: item[1], reverse=True)],
        "trend": [{"date": key, **value} for key, value in sorted(trend_map.items())],
        "asset_options": asset_options,
        "errors": errors[:20],
    }


@router.get("/history/loops/{loop_id}/assessment")
def history_loop_assessment(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = assess_loop(loop_id, start_time=start_time, end_time=end_time)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/loops/{loop_id}/harris")
def history_loop_harris(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    error_basis: str = "auto",
    force_deadtime_samples: int | None = None,
    ar_order_override: int | None = None,
) -> dict[str, Any]:
    result = compute_loop_harris(
        loop_id,
        start_time=start_time,
        end_time=end_time,
        error_basis=error_basis,
        force_deadtime_samples=force_deadtime_samples,
        ar_order_override=ar_order_override,
    )
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/loops/{loop_id}/cpk")
async def history_loop_cpk(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    refresh_ontology: bool = False,
) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")
    spec_limits: dict[str, Any] | None = None
    try:
        facts = await resolve_loop_ontology_facts(
            loop_id=loop_id,
            loop_type=str(loop.get("loop_type") or "unknown"),
            force_refresh=refresh_ontology,
        )
        parsed_limits = facts.get("pv_spec_limits") or {}
        if parsed_limits.get("lsl") is not None and parsed_limits.get("usl") is not None:
            spec_limits = parsed_limits
    except Exception:
        spec_limits = None
    result = compute_loop_cpk(
        loop_id,
        start_time=start_time,
        end_time=end_time,
        spec_limits=spec_limits,
    )
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


@router.get("/history/loops/{loop_id}/ontology-rules")
async def history_loop_ontology_rules(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    refresh_ontology: bool = False,
) -> dict[str, Any]:
    result = await evaluate_loop_ontology_rules(
        loop_id,
        start_time=start_time,
        end_time=end_time,
        force_refresh=refresh_ontology,
    )
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


def _compact_tuning_prior_context(
    *,
    features: dict[str, Any],
    monitoring: dict[str, Any],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    snapshot = monitoring.get("monitoring") or {}
    return {
        "loop": {
            "id": features.get("identity", {}).get("loop_id"),
            "type": features.get("identity", {}).get("loop_type"),
            "sample_time_s": features.get("data_profile", {}).get("sample_time_median_s"),
            "row_count": features.get("data_profile", {}).get("row_count"),
            "time_start": features.get("data_profile", {}).get("time_start"),
            "time_end": features.get("data_profile", {}).get("time_end"),
        },
        "monitoring": {
            "status": snapshot.get("status"),
            "overall_score": snapshot.get("overall_score"),
            "alerts": snapshot.get("alerts") or [],
            "data_health": snapshot.get("data_health"),
            "stability": snapshot.get("stability"),
            "operating_condition": snapshot.get("operating_condition"),
            "constraints": snapshot.get("constraints"),
            "response_observability": snapshot.get("response_observability"),
        },
        "assessment": {
            "summary": assessment.get("summary"),
            "performance": assessment.get("performance"),
            "tuning_readiness": assessment.get("tuning_readiness"),
            "identification_suitability": assessment.get("identification_suitability"),
            "diagnostics": assessment.get("diagnostics"),
        },
        "raw_features": {
            "pv_stats": features.get("pv_stats"),
            "mv_stats": features.get("mv_stats"),
            "sp_stats": features.get("sp_stats"),
            "data_quality": features.get("data_quality"),
            "operating_condition_profile": features.get("operating_condition_profile"),
            "pv_mv_relation_raw": features.get("pv_mv_relation_raw"),
            "frequency_raw": features.get("frequency_raw"),
            "oscillation_raw": features.get("oscillation_raw"),
            "performance_raw": features.get("performance_raw"),
            "actuator_profile": features.get("actuator_profile"),
            "excitation_profile": features.get("excitation_profile"),
            "constraint_raw": features.get("constraint_raw"),
            "scale_profile": features.get("scale_profile"),
        },
    }


def _build_tuning_prior_prompt(
    *,
    loop_id: str,
    loop_type: str,
    core_context: dict[str, Any],
    ontology_context: dict[str, Any] | None,
) -> str:
    ontology_text = ""
    ontology_rules_text = ""
    if ontology_context:
        ontology_text = str(ontology_context.get("content") or ontology_context.get("error") or "")
        rule_evaluation = ontology_context.get("rule_evaluation")
        if rule_evaluation:
            ontology_rules_text = json.dumps(rule_evaluation, ensure_ascii=False, default=str, indent=2)
    return (
        "你是一名资深 PID 整定专家，请基于两个上下文生成“整定先验”的可解释性评审。\n"
        "注意：整定先验只作为工程建议和风险提示，不作为硬约束拦截整定流程。\n\n"
        f"回路：{loop_id}，类型：{loop_type}\n\n"
        "上下文 1：监控、评估、诊断和原始画像指标（JSON）\n"
        f"{json.dumps(core_context, ensure_ascii=False, default=str, indent=2)}\n\n"
        "上下文 2：本体/MCP 返回的回路知识\n"
        f"{ontology_text or '未获取到本体上下文，请仅基于历史数据指标说明。'}\n\n"
        "上下文 3：本体规则评估结果（JSON，规则只作工程评审提示，不作强约束自动拦截）\n"
        f"{ontology_rules_text or '暂无本体规则评估结果。'}\n\n"
        "请输出：\n"
        "1. 当前回路是否适合进入整定，以及主要依据。\n"
        "2. 建议优先使用哪些历史片段或窗口特征，应该避开什么片段。\n"
        "3. 对辨识模型的先验建议，包括增益方向、可能时间尺度、时滞、噪声/饱和/振荡风险。\n"
        "4. 对后续 PID 整定策略的建议，包括保守程度、需要人工确认的事项。\n"
        "5. 明确哪些结论来自历史数据，哪些来自本体知识，哪些只是低置信度推断。\n"
        "要求：中文、条理化、不要编造本体中没有的事实；如果本体和数据冲突，请明确指出冲突。"
    )


class TuningPriorReviewRequest(BaseModel):
    core_context: dict[str, Any] = Field(default_factory=dict)
    ontology: dict[str, Any] | None = None


async def _build_tuning_prior_core_payload(
    loop_id: str,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    bundle = get_loop_profile_bundle(loop_id, start_time=start_time, end_time=end_time)
    if bundle.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    if bundle.get("error"):
        return bundle

    features = bundle["features"]
    monitoring = bundle["monitoring"]
    assessment = bundle["assessment"]
    loop_type = str(bundle.get("loop_type") or features.get("identity", {}).get("loop_type") or "unknown")
    core_context = _compact_tuning_prior_context(
        features=features,
        monitoring=monitoring,
        assessment=assessment,
    )
    return {
        "loop_id": loop_id,
        "loop_type": loop_type,
        "start_time": start_time,
        "end_time": end_time,
        "features": features,
        "monitoring": monitoring,
        "assessment": assessment,
        "core_context": core_context,
    }


@router.get("/history/loops/{loop_id}/tuning-prior/core")
async def history_loop_tuning_prior_core(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    return await _build_tuning_prior_core_payload(loop_id, start_time=start_time, end_time=end_time)


@router.get("/history/loops/{loop_id}/tuning-prior/ontology")
async def history_loop_tuning_prior_ontology(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")
    loop_type = str(loop.get("loop_type") or "unknown")
    try:
        ontology_context = await fetch_loop_ontology_context_via_mcp(
            loop_name=loop_id,
            loop_type=loop_type,
            max_chars=12000,
        )
    except Exception as exc:
        ontology_context = {
            "source": "registered_mcp_tool",
            "error": str(exc),
            "content": "",
        }
    ontology_payload = ontology_context or {
        "source": "registered_mcp_tool",
        "content": "",
        "error": "no enabled MCP chat tool returned ontology context",
    }
    try:
        rule_eval = await evaluate_loop_ontology_rules(
            loop_id,
            start_time=start_time,
            end_time=end_time,
        )
        ontology_payload = {
            **ontology_payload,
            "rule_evaluation": {
                "rule_pack_version": rule_eval.get("rule_pack_version"),
                "summary": rule_eval.get("summary"),
                "ontology_facts": rule_eval.get("ontology_facts"),
                "rules": rule_eval.get("rules"),
            },
        }
    except Exception as exc:
        ontology_payload = {
            **ontology_payload,
            "rule_evaluation_error": str(exc),
        }
    return {
        "loop_id": loop_id,
        "loop_type": loop_type,
        "start_time": start_time,
        "end_time": end_time,
        "ontology": ontology_payload,
    }


@router.post("/history/loops/{loop_id}/tuning-prior/review")
async def history_loop_tuning_prior_review(
    loop_id: str,
    body: TuningPriorReviewRequest,
) -> dict[str, Any]:
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")
    loop_type = str(loop.get("loop_type") or body.core_context.get("loop", {}).get("type") or "unknown")
    prompt = _build_tuning_prior_prompt(
        loop_id=loop_id,
        loop_type=loop_type,
        core_context=body.core_context,
        ontology_context=body.ontology,
    )
    model_cfg = model_cfg_store.get()
    if not model_cfg.model_api_key or not model_cfg.model_api_url:
        return {
            "loop_id": loop_id,
            "loop_type": loop_type,
            "prompt": prompt,
            "review": "",
            "error": "模型配置未完成，请先在系统设置 / 模型配置中填写 API 地址、Key 和模型名称。",
        }
    try:
        client = AsyncOpenAI(
            api_key=model_cfg.model_api_key,
            base_url=model_cfg.model_api_url,
            timeout=90.0,
        )
        resp = await client.chat.completions.create(
            model=model_cfg.model_name,
            messages=[
                {"role": "system", "content": "你是资深 PID 整定专家。输出中文、可审计、面向工程师的整定先验评审。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1600,
        )
        msg = resp.choices[0].message
        review = str(getattr(msg, "content", "") or "").strip()
        if not review:
            return {
                "loop_id": loop_id,
                "loop_type": loop_type,
                "prompt": prompt,
                "review": "",
                "error": "模型调用完成，但未返回可展示的先验评审说明。",
                "advisory_only": True,
            }
        return {
            "loop_id": loop_id,
            "loop_type": loop_type,
            "prompt": prompt,
            "review": review,
            "advisory_only": True,
        }
    except Exception as exc:
        return {
            "loop_id": loop_id,
            "loop_type": loop_type,
            "prompt": prompt,
            "review": "",
            "error": str(exc)[:500],
            "advisory_only": True,
        }


@router.get("/history/loops/{loop_id}/tuning-prior")
async def history_loop_tuning_prior(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    core_payload = await _build_tuning_prior_core_payload(loop_id, start_time=start_time, end_time=end_time)
    loop_type = str(core_payload.get("loop_type") or "unknown")

    ontology_context: dict[str, Any] | None
    try:
        ontology_context = await fetch_loop_ontology_context_via_mcp(
            loop_name=loop_id,
            loop_type=loop_type,
            max_chars=12000,
        )
    except Exception as exc:
        ontology_context = {
            "source": "registered_mcp_tool",
            "error": str(exc),
            "content": "",
        }

    core_context = core_payload["core_context"]
    prompt = _build_tuning_prior_prompt(
        loop_id=loop_id,
        loop_type=loop_type,
        core_context=core_context,
        ontology_context=ontology_context,
    )

    return {
        "loop_id": loop_id,
        "loop_type": loop_type,
        "start_time": start_time,
        "end_time": end_time,
        "features": core_payload.get("features"),
        "monitoring": core_payload.get("monitoring"),
        "assessment": core_payload.get("assessment"),
        "core_context": core_context,
        "ontology": ontology_context or {
            "source": "registered_mcp_tool",
            "content": "",
            "error": "no enabled MCP chat tool returned ontology context",
        },
        "prompt": prompt,
    }


@router.get("/history/loops/{loop_id}/windows")
def history_loop_windows(
    loop_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = list_loop_windows(loop_id, start_time=start_time, end_time=end_time)
    if result.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    return result


def _tuning_blocked_by_assessment(assessment: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    readiness = assessment.get("tuning_readiness") or {}
    summary = assessment.get("summary") or {}
    decision = readiness.get("decision") or summary.get("decision")
    gate_checks = readiness.get("gate_checks") or []
    blocking_reasons = readiness.get("blocking_reasons") or []
    hard_failed_checks = [
        item for item in gate_checks
        if not item.get("passed") and str(item.get("severity") or "").lower() in {"critical", "high", "error", "blocked"}
    ]
    blocked = decision == "blocked" or bool(hard_failed_checks)
    return blocked, {
        "decision": decision,
        "decision_text": summary.get("decision_text"),
        "recommended_next_action": summary.get("recommended_next_action"),
        "recommended_next_action_text": summary.get("recommended_next_action_text"),
        "blocking_reasons": blocking_reasons,
        "failed_checks": hard_failed_checks,
    }


async def _blocked_history_tune_sse(loop_id: str, gate: dict[str, Any]):
    payload = {
        "type": "error",
        "stage": "tuning_gate",
        "message": "当前回路未通过整定准入校验，已阻止发起整定任务",
        "loop_id": loop_id,
        "data": gate,
    }
    yield f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
    yield "data: {\"type\": \"done\"}\n\n"


async def _history_tune_sse(request: TuningRequest, csv_path: str, loop_id: str):
    meta_init = {
        "csv_name": f"history:{loop_id}",
        "loop_type": request.loop_type,
        "loop_name": request.loop_name or loop_id,
        "use_llm_advisor": request.use_llm_advisor,
        "selected_window_index": request.selected_window_index,
        "history_loop_id": loop_id,
        "stop_after": request.stop_after,
        "algorithm_filter": request.algorithm_filter,
        "ontology_context_present": bool(request.ontology_context),
        "start_time": request.start_time,
        "end_time": request.end_time,
    }
    inner = run_tuning_pipeline(
        csv_path=csv_path,
        loop_type=request.loop_type,
        loop_name=request.loop_name or loop_id,
        selected_loop_prefix=request.selected_loop_prefix,
        selected_window_index=request.selected_window_index,
        plant_type=request.plant_type,
        scenario=request.scenario,
        control_object=request.control_object,
        use_llm_advisor=request.use_llm_advisor,
        stop_after=request.stop_after,  # type: ignore[arg-type]
        algorithm_filter=request.algorithm_filter,
        ontology_context=request.ontology_context,
        start_time=request.start_time or None,
        end_time=request.end_time or None,
    )
    async for event in record_stream(kind="tune", meta_init=meta_init, gen=inner):
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    yield "data: {\"type\": \"done\"}\n\n"


@router.post("/history/loops/{loop_id}/tune/stream")
async def tune_history_loop_stream(
    loop_id: str,
    loop_type: str = Form(""),
    loop_name: str = Form(""),
    plant_type: str = Form(""),
    scenario: str = Form(""),
    control_object: str = Form(""),
    selected_window_index: int | None = Form(None),
    use_llm_advisor: bool = Form(True),
    stop_after: str | None = Form(None),
    # 逗号分隔的算法白名单，例如 "sv_step,mv_step"
    algorithm_filter: str | None = Form(None),
    ontology_context: str | None = Form(None),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
):
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")

    assessment = assess_loop(loop_id, start_time=start_time, end_time=end_time)
    blocked, gate = _tuning_blocked_by_assessment(assessment)
    if blocked:
        return StreamingResponse(
            _blocked_history_tune_sse(loop_id, gate),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    algorithm_filter_list: list[str] | None = None
    if algorithm_filter:
        algorithm_filter_list = [s.strip() for s in algorithm_filter.split(",") if s.strip()]
    request = TuningRequest(
        csv_path=str(loop["csv_path"]),
        loop_type=loop_type or str(loop.get("loop_type") or "flow"),
        loop_name=loop_name or loop_id,
        plant_type=plant_type,
        scenario=scenario,
        control_object=control_object,
        selected_loop_prefix=None,
        selected_window_index=selected_window_index,
        use_llm_advisor=use_llm_advisor,
        stop_after=stop_after,
        algorithm_filter=algorithm_filter_list,
        ontology_context=ontology_context,
        start_time=start_time or "",
        end_time=end_time or "",
    )
    return StreamingResponse(
        _history_tune_sse(request, str(loop["csv_path"]), loop_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
