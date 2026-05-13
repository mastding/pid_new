"""Historical loop data API endpoints."""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.model_config import store as model_cfg_store
from core.history.store import (
    assess_loop,
    get_loop_features,
    get_loop_monitoring,
    get_loop,
    import_history_file,
    list_loop_windows,
    list_loops,
    load_loop_series,
)
from core.pipeline.ontology_mcp_context import fetch_loop_ontology_context_via_mcp
from core.pipeline.runner import run_tuning_pipeline
from core.session_log import record_stream
from models import TuningRequest

router = APIRouter(tags=["history"])


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
    if ontology_context:
        ontology_text = str(ontology_context.get("content") or ontology_context.get("error") or "")
    return (
        "你是一名资深 PID 整定专家，请基于两个上下文生成“整定先验”的可解释性评审。\n"
        "注意：整定先验只作为工程建议和风险提示，不作为硬约束拦截整定流程。\n\n"
        f"回路：{loop_id}，类型：{loop_type}\n\n"
        "上下文 1：监控、评估、诊断和原始画像指标（JSON）\n"
        f"{json.dumps(core_context, ensure_ascii=False, default=str, indent=2)}\n\n"
        "上下文 2：本体/MCP 返回的回路知识\n"
        f"{ontology_text or '未获取到本体上下文，请仅基于历史数据指标说明。'}\n\n"
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
    loop = get_loop(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="loop_id not found")

    features = get_loop_features(loop_id, start_time=start_time, end_time=end_time)
    if features.get("error") == "loop_id not found":
        raise HTTPException(status_code=404, detail="loop_id not found")
    monitoring = get_loop_monitoring(loop_id, start_time=start_time, end_time=end_time)
    assessment = assess_loop(loop_id, start_time=start_time, end_time=end_time)
    loop_type = str(loop.get("loop_type") or features.get("identity", {}).get("loop_type") or "unknown")
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
    return {
        "loop_id": loop_id,
        "loop_type": loop_type,
        "start_time": start_time,
        "end_time": end_time,
        "ontology": ontology_context or {
            "source": "registered_mcp_tool",
            "content": "",
            "error": "no enabled MCP chat tool returned ontology context",
        },
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
