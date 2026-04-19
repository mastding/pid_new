"""PID consultant agent API endpoint.

POST /api/consult/stream  — SSE stream, accepts JSON body with messages + session.
The tool handlers are closures over the session context (csv_path, current model, etc.)
so the LLM agent can iteratively adjust identification / tuning / evaluation.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.agent.consultant import run_consultant
from core.algorithms.data_analysis import load_and_prepare_dataset
from core.algorithms.pid_evaluation import evaluate_pid_params
from core.algorithms.pid_tuning import select_best_strategy
from core.algorithms.system_id import fit_best_model
from core.session_log import record_stream

router = APIRouter(tags=["consultant"])


# ── Session context ───────────────────────────────────────────────────────────

class SessionContext(BaseModel):
    csv_path: str
    loop_type: str = "flow"
    dt: float = 1.0

    # Current identified model
    model_type: str = "FOPDT"
    model_K: float = 1.0
    model_T: float = 10.0
    model_T1: float = 10.0
    model_T2: float = 0.0
    model_L: float = 1.0
    model_r2: float = 0.0
    model_nrmse: float = 0.0
    model_confidence: float = 0.5
    n_windows: int = 0

    # Current PID params
    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0
    Ti: float = 0.0
    Td: float = 0.0
    tuning_strategy: str = ""


class ConsultRequest(BaseModel):
    messages: list[dict[str, Any]]
    session: SessionContext
    max_iterations: int = Field(default=8, ge=1, le=15)


# ── Tool handlers (closures over session) ─────────────────────────────────────

def _build_tool_handlers(session: SessionContext) -> dict[str, Any]:
    """Build tool handler closures bound to the current session state."""

    def get_data_overview() -> dict[str, Any]:
        return {
            "csv_path": session.csv_path,
            "loop_type": session.loop_type,
            "sampling_time_s": session.dt,
            "n_windows": session.n_windows,
            "current_model": {
                "type": session.model_type,
                "K": session.model_K,
                "T": session.model_T,
                "T1": session.model_T1,
                "T2": session.model_T2,
                "L": session.model_L,
                "R2": session.model_r2,
                "NRMSE": session.model_nrmse,
                "confidence": session.model_confidence,
            },
            "current_pid": {
                "Kp": session.Kp,
                "Ki": session.Ki,
                "Kd": session.Kd,
                "Ti": session.Ti,
                "Td": session.Td,
                "strategy": session.tuning_strategy,
            },
        }

    def run_identification(
        window_index: int | None = None,
        model_type: str | None = None,
    ) -> dict[str, Any]:
        try:
            dataset = load_and_prepare_dataset(
                csv_path=session.csv_path,
                selected_loop_prefix=None,
                selected_window_index=window_index,
            )
        except Exception as exc:
            return {"error": f"数据加载失败: {exc}"}

        force_types = [model_type] if model_type and model_type != "AUTO" else None
        try:
            result = fit_best_model(
                cleaned_df=dataset["cleaned_df"],
                candidate_windows=dataset["candidate_windows"],
                actual_dt=dataset["dt"],
                loop_type=session.loop_type,
                force_model_types=force_types,
            )
        except Exception as exc:
            return {"error": f"辨识失败: {exc}"}

        m = result["model"]
        c = result["confidence"]
        # Update session in-place
        session.model_type = m.model_type.value
        session.model_K = m.K
        session.model_T = m.T
        session.model_T1 = m.T1
        session.model_T2 = m.T2
        session.model_L = m.L
        session.model_r2 = m.r2_score
        session.model_nrmse = m.normalized_rmse
        session.model_confidence = c.confidence
        session.n_windows = len(dataset["candidate_windows"])

        return {
            "model_type": m.model_type.value,
            "K": round(m.K, 4),
            "T": round(m.T, 2),
            "T1": round(m.T1, 2),
            "T2": round(m.T2, 2),
            "L": round(m.L, 2),
            "r2_score": round(m.r2_score, 4),
            "normalized_rmse": round(m.normalized_rmse, 4),
            "confidence": round(c.confidence, 3),
            "confidence_quality": c.quality,
            "window_source": result["window_source"],
            "selection_reason": result["selection_reason"],
        }

    def run_tuning(
        strategy: str = "AUTO",
        kp_scale: float = 1.0,
        ki_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> dict[str, Any]:
        if strategy == "AUTO":
            strategy = None  # type: ignore[assignment]

        mp = {
            "model_type": session.model_type,
            "K": session.model_K,
            "T": session.model_T,
            "T1": session.model_T1,
            "T2": session.model_T2,
            "L": session.model_L,
        }
        try:
            result = select_best_strategy(
                K=session.model_K,
                T=session.model_T,
                L=session.model_L,
                dt=session.dt,
                loop_type=session.loop_type,
                model_type=session.model_type,
                model_params=mp,
                confidence=session.model_confidence,
                nrmse=session.model_nrmse,
                r2=session.model_r2,
            )
        except Exception as exc:
            return {"error": f"整定失败: {exc}"}

        # If a specific strategy was requested, find it in candidates
        best = result["best"] or {}
        if strategy and strategy != "AUTO":
            for c in result["all_candidates"]:
                if c.get("strategy") == strategy.upper():
                    best = c
                    break

        # Apply scale factors
        Kp = round(float(best.get("Kp", 0)) * max(kp_scale, 0.1), 6)
        Ki = round(float(best.get("Ki", 0)) * max(ki_scale, 0.1), 6)
        Kd = round(float(best.get("Kd", 0)) * max(kd_scale, 0.1), 6)
        Ti = round(float(best.get("Ti", 0)), 2)
        Td = round(float(best.get("Td", 0)), 2)

        # Update session in-place
        session.Kp = Kp
        session.Ki = Ki
        session.Kd = Kd
        session.Ti = Ti
        session.Td = Td
        session.tuning_strategy = best.get("strategy", "")

        return {
            "strategy": best.get("strategy", ""),
            "description": best.get("description", ""),
            "Kp": Kp, "Ki": Ki, "Kd": Kd,
            "Ti": Ti, "Td": Td,
            "kp_scale_applied": kp_scale,
            "ki_scale_applied": ki_scale,
            "kd_scale_applied": kd_scale,
            "heuristic_reason": result["heuristic_reason"],
            "all_candidates": [
                {"strategy": c.get("strategy"), "Kp": round(c.get("Kp", 0), 4)}
                for c in result["all_candidates"]
            ],
        }

    def run_evaluation(
        Kp: float,
        Ki: float,
        Kd: float,
    ) -> dict[str, Any]:
        mp = {
            "model_type": session.model_type,
            "K": session.model_K,
            "T1": session.model_T1,
            "T2": session.model_T2,
            "L": session.model_L,
        }
        try:
            result = evaluate_pid_params(
                Kp=Kp, Ki=Ki, Kd=Kd,
                model_type=session.model_type,
                model_params=mp,
                K=session.model_K,
                T=session.model_T,
                L=session.model_L,
                dt=session.dt,
                loop_type=session.loop_type,
                confidence=session.model_confidence,
            )
        except Exception as exc:
            return {"error": f"评估失败: {exc}"}

        # Update session PID if caller wants to commit
        session.Kp = Kp
        session.Ki = Ki
        session.Kd = Kd

        return {
            "passed": result["passed"],
            "performance_score": result["performance_score"],
            "final_rating": result["final_rating"],
            "readiness_score": result["readiness_score"],
            "robustness_score": result["robustness_score"],
            "is_stable": result["is_stable"],
            "overshoot_percent": result["overshoot_percent"],
            "settling_time_s": result["settling_time_s"],
            "steady_state_error": result["steady_state_error"],
            "oscillation_count": result["oscillation_count"],
            "mv_saturation_pct": result["mv_saturation_pct"],
            "recommendation": result["recommendation"],
        }

    def search_experience(query: str = "") -> dict[str, Any]:
        # Placeholder — experience store not yet implemented
        return {
            "matches": [],
            "note": "经验库功能即将上线，暂无匹配记录。",
            "query": query,
        }

    return {
        "get_data_overview": get_data_overview,
        "run_identification": run_identification,
        "run_tuning": run_tuning,
        "run_evaluation": run_evaluation,
        "search_experience": search_experience,
    }


# ── SSE endpoint ──────────────────────────────────────────────────────────────

async def _consult_sse(request: ConsultRequest):
    handlers = _build_tool_handlers(request.session)
    inner = run_consultant(
        messages=request.messages,
        tool_handlers=handlers,
        max_iterations=request.max_iterations,
    )
    last_user = next(
        (m.get("content") for m in reversed(request.messages) if m.get("role") == "user"),
        "",
    )
    meta_init = {
        "csv_name": request.session.csv_path.split("/")[-1].split("\\")[-1],
        "loop_type": request.session.loop_type,
        "user_prompt": str(last_user)[:200],
        "n_messages": len(request.messages),
    }
    async for event in record_stream(kind="consult", meta_init=meta_init, gen=inner):
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
    yield 'data: {"type": "done"}\n\n'


@router.post("/consult/stream")
async def consult_stream(request: ConsultRequest):
    """Run the PID consultant agent and stream events via SSE.

    The client sends conversation messages + a session context (csv_path,
    current model params, current PID params). The agent can call any of
    the 5 tools (run_identification, run_tuning, run_evaluation,
    get_data_overview, search_experience) to iteratively refine the result.
    """
    return StreamingResponse(
        _consult_sse(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
