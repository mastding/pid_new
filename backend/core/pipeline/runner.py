"""整定流水线（Day 4：在窗口选择处接入 LLM 顾问）。

四个阶段：data_analysis → window_selection → identification → tuning → evaluation
LLM 仅在 window_selection 决策点参与；失败自动回退到确定性 fit_score 选窗。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Literal

# 提前结束的切点。None 表示跑完全流程。
StopAfter = Literal["window_selection", "identification"] | None

from core.algorithms.data_analysis import load_and_prepare_dataset
from core.algorithms.system_id import fit_best_model
from core.algorithms.pid_tuning import select_best_strategy
from core.algorithms.pid_evaluation import evaluate_pid_params
from core.pipeline.events import error_event, result_event, stage_event
from core.pipeline.llm_advisor import choose_window_via_llm
from core.pipeline.ontology_policy_builder import build_window_selection_policy
from core.pipeline.ontology_mcp_context import fetch_loop_ontology_context_via_mcp
from core.pipeline.refinement_policy import recommend_refinement_from_algorithm_comparison
from core.pipeline.skill_orchestrator import PlannedSkillCall, skill_orchestrator
from core.pipeline.window_algorithm_family import window_algorithm_family
from core.pipeline.window_policy_scoring import apply_window_policy_to_candidates
from core.shared.loop_features import extract_loop_features
from core.skills import LoopContext, SkillResult
from models.process_model import ModelConfidence, ModelType, ProcessModel


_WINDOW_ALGORITHM_FAMILIES = ["sp_step", "mv_step", "mv_ramp", "steady_disturbance", "rolling_scan"]


def _raw_loop_features_for_window_agent(
    *,
    df: Any,
    loop_name: str,
    loop_type: str,
    sample_time_s: float,
    csv_path: str,
) -> dict[str, Any]:
    """Build the window-agent profile from raw LoopFeatures only.

    process_prior is intentionally excluded here: ontology/LLM strategy should
    interpret raw features instead of inheriting older time-constant priors.
    """
    features = extract_loop_features(
        df,
        loop_id=loop_name or "uploaded_loop",
        loop_type=loop_type or "unknown",
        source_file=csv_path,
        sample_time_s=sample_time_s,
        loop_name=loop_name or None,
    )
    features.pop("process_prior", None)
    if "data_quality" not in features and isinstance(features.get("data_quality_raw"), dict):
        features["data_quality"] = features["data_quality_raw"]
    dp = features.get("data_profile") or {}
    pv = features.get("pv_stats") or {}
    mv = features.get("mv_stats") or {}
    constraint = features.get("constraint_raw") or {}
    relation = features.get("pv_mv_relation_raw") or {}
    features["text_summary"] = (
        f"rows={dp.get('row_count')}, valid_rows={dp.get('valid_row_count')}, "
        f"sample={dp.get('sample_time_median_s')}s, "
        f"PV={pv.get('min')}~{pv.get('max')} span={pv.get('span')}, "
        f"MV={mv.get('min')}~{mv.get('max')} span={mv.get('span')}, "
        f"MV_saturation={constraint.get('mv_saturation_ratio')}, "
        f"direction_raw={relation.get('process_direction')}, "
        f"direction_confidence={relation.get('process_direction_confidence')}"
    )
    return features


def _overlay_window_algorithm_family_summaries(
    *,
    candidate_windows: list[dict[str, Any]],
    policy: dict[str, Any],
    previous: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    previous_by_family = {
        str(item.get("family")): item
        for item in (previous or [])
        if isinstance(item, dict) and item.get("family")
    }
    plan_by_family = {
        str(item.get("family")): item
        for item in (policy.get("algorithm_plan") or [])
        if isinstance(item, dict) and item.get("family")
    }
    disabled = {str(item) for item in policy.get("disabled_algorithm_families") or []}
    summaries: list[dict[str, Any]] = []
    for family in _WINDOW_ALGORITHM_FAMILIES:
        windows = [w for w in candidate_windows if window_algorithm_family(w) == family]
        previous_item = previous_by_family.get(family, {})
        plan = plan_by_family.get(family, {})
        policy_state = str(plan.get("state") or previous_item.get("policy_state") or "available")
        run_state = str(previous_item.get("run_state") or ("disabled" if family in disabled else "ran"))
        if family in disabled:
            run_state = "disabled"
            policy_state = "disabled"
        summaries.append({
            "family": family,
            "provider": previous_item.get("provider", family),
            "run_state": run_state,
            "policy_state": policy_state,
            "policy_reason": plan.get("reason") or previous_item.get("policy_reason", ""),
            "event_count": int(previous_item.get("event_count", 0) or 0),
            "window_count": len(windows),
            "usable_count": sum(1 for w in windows if w.get("window_usable_for_id")),
            "best_score": round(max([float(w.get("window_quality_score", 0.0)) for w in windows] or [0.0]), 4),
        })
    return summaries


def _build_skill_context(
    *,
    csv_path: str,
    loop_type: str,
    dataset: dict[str, Any],
    candidate_windows: list[dict[str, Any]],
    selected_window_index: int | None = None,
    data_profile: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    pid_params: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> LoopContext:
    ctx = LoopContext(csv_path=csv_path, loop_type=loop_type)
    ctx.cleaned_df = dataset["cleaned_df"]
    ctx.dt = dataset["dt"]
    ctx.candidate_windows = list(candidate_windows)
    ctx.selected_window_index = selected_window_index
    if data_profile:
        ctx.data_profile.update(data_profile)
    if model:
        ctx.model = dict(model)
    if pid_params:
        ctx.pid_params = dict(pid_params)
    if confidence is not None:
        ctx.confidence = float(confidence)
    return ctx


def _invoke_guarded_skill(
    name: str,
    args: dict[str, Any],
    ctx: LoopContext,
    *,
    initiated_by: str = "system",
) -> SkillResult:
    record = skill_orchestrator.execute_one(
        PlannedSkillCall(skill_name=name, args=args, initiated_by=initiated_by),
        ctx,
    )
    if not record.guard.allowed:
        return SkillResult(
            success=False,
            reasoning=record.guard.reason,
            data={"workflow_guard": record.guard.to_dict()},
        )
    if record.result is None:
        return SkillResult(
            success=False,
            reasoning="skill orchestrator did not return a result",
            data={"workflow_guard": record.guard.to_dict()},
        )
    return record.result


def _process_model_from_skill(best_model: dict[str, Any]) -> ProcessModel:
    raw_mt = best_model.get("model_type", "FOPDT")
    if isinstance(raw_mt, ModelType):
        mt = raw_mt.value
    else:
        mt = str(raw_mt).split(".")[-1].upper()
    return ProcessModel(
        model_type=ModelType(mt),
        K=float(best_model.get("K", 0.0) or 0.0),
        T=float(best_model.get("T", 0.0) or 0.0),
        T1=float(best_model.get("T1", 0.0) or 0.0),
        T2=float(best_model.get("T2", 0.0) or 0.0),
        L=float(best_model.get("L", 0.0) or 0.0),
        zeta=float(best_model.get("zeta", 0.0) or 0.0),
        r2_score=float(best_model.get("r2_score", 0.0) or 0.0),
        normalized_rmse=float(best_model.get("normalized_rmse", 0.0) or 0.0),
        raw_rmse=float(best_model.get("raw_rmse", 0.0) or 0.0),
        success=True,
    )


def _confidence_from_skill(best_model: dict[str, Any]) -> ModelConfidence:
    conf = float(best_model.get("confidence", 0.0) or 0.0)
    if conf >= 0.85:
        quality = "excellent"
    elif conf >= 0.7:
        quality = "good"
    elif conf >= 0.5:
        quality = "fair"
    else:
        quality = "poor"
    return ModelConfidence(
        confidence=conf,
        quality=quality,
        recommendation="",
        r2_score=float(best_model.get("r2_score", 0.0) or 0.0),
        rmse_score=max(0.0, 1.0 - float(best_model.get("normalized_rmse", 1.0) or 1.0)),
    )


def _window_meta_by_source(windows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(w.get("window_source", "")): {
            "window_algorithm": w.get("window_algorithm", ""),
            "window_algorithm_label": w.get("window_algorithm_label", ""),
            "window_quality_score": float(w.get("window_quality_score", 0.0) or 0.0),
            "window_score_breakdown": w.get("window_score_breakdown", {}),
        }
        for w in windows
    }


def _algorithm_comparison(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_algorithm: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if not attempt.get("success"):
            continue
        algorithm = str(attempt.get("window_algorithm") or attempt.get("window_algorithm_label") or "unknown")
        current = best_by_algorithm.get(algorithm)
        if current is None or float(attempt.get("fit_score", -1e12)) > float(current.get("fit_score", -1e12)):
            best_by_algorithm[algorithm] = attempt

    comparison = []
    for algorithm, attempt in best_by_algorithm.items():
        comparison.append({
            "algorithm": algorithm,
            "algorithm_label": attempt.get("window_algorithm_label") or algorithm,
            "window_source": attempt.get("window_source", ""),
            "model_type": attempt.get("model_type", ""),
            "fit_score": float(attempt.get("fit_score", 0.0)),
            "r2_score": float(attempt.get("r2_score", 0.0)),
            "normalized_rmse": float(attempt.get("normalized_rmse", 0.0)),
            "confidence": float(attempt.get("confidence", 0.0)),
            "window_quality_score": float(attempt.get("window_quality_score", 0.0)),
        })
    return sorted(comparison, key=lambda item: item["fit_score"], reverse=True)


async def run_tuning_pipeline(
    *,
    csv_path: str,
    loop_type: str = "flow",
    selected_loop_prefix: str | None = None,
    selected_window_index: int | None = None,
    loop_name: str = "",
    plant_type: str = "",
    scenario: str = "",
    control_object: str = "",
    use_llm_advisor: bool = True,
    stop_after: StopAfter = None,
    algorithm_filter: list[str] | None = None,
    ontology_context: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the full deterministic tuning pipeline with SSE events.

    Yields stage events for frontend progress, then a final result event.
    No LLM calls - purely deterministic computation.

    stop_after: 提前结束流水线，发一个精简版 result 事件后就 return：
      - "window_selection"：跑到选窗结束（数据分析菜单）
      - "identification"：跑到辨识 + 精修循环结束（系统辨识菜单）
      - None：跑完全流程（默认）

    algorithm_filter: 候选窗口的算法族白名单（如 ["sv_step", "mv_step"]）；命中
      `window_algorithm` 或 `window_algorithm_label` 任一即保留。None 或空列表表示不过滤。

    ontology_context: 可选本体上下文；窗口 LLM 顾问会结合变量角色、动态先验、工况场景等
      判断候选窗口是否符合工艺知识。
    """

    # ── Stage 1: Data Analysis ──────────────────────────────────────────
    plan_ctx = LoopContext(
        csv_path=csv_path,
        loop_prefix=selected_loop_prefix or "",
        loop_type=loop_type,
    )
    default_plan = skill_orchestrator.build_default_plan(plan_ctx)
    yield {
        "type": "workflow_plan",
        "planner_mode": "default_template",
        "llm_insertions_enabled": bool(use_llm_advisor),
        "skills": [
            {
                "skill_name": call.skill_name,
                "initiated_by": call.initiated_by,
                "purpose": call.purpose,
            }
            for call in default_plan
        ],
    }

    yield stage_event("data_analysis", "running")
    dataset: dict[str, Any] | None = None
    load_ctx = LoopContext(
        csv_path=csv_path,
        loop_prefix=selected_loop_prefix or "",
        loop_type=loop_type,
    )
    load_result = _invoke_guarded_skill(
        "load_dataset",
        {
            "loop_prefix": selected_loop_prefix,
            "start_time": start_time,
            "end_time": end_time,
        },
        load_ctx,
    )
    if load_result.success and load_ctx.cleaned_df is not None and load_ctx.dt is not None:
        dataset = {
            "cleaned_df": load_ctx.cleaned_df,
            "dt": load_ctx.dt,
            "step_events": [],
            "candidate_windows": [],
            "data_points": int(load_result.data.get("data_points", len(load_ctx.cleaned_df))),
            "quality_metrics": None,
            "window_detection_meta": {},
        }

    if dataset is None:
        try:
            dataset = load_and_prepare_dataset(
                csv_path=csv_path,
                selected_loop_prefix=selected_loop_prefix,
                selected_window_index=selected_window_index,
                loop_type=loop_type,
                start_time=start_time,
                end_time=end_time,
            )
        except ValueError as exc:
            yield error_event(str(exc), stage="data_analysis", error_code="DATA_ERROR")
            return

    # 数据画像先于窗口算法生成：本体/LLM 策略需要知道历史数据的基本状态。
    data_profile: dict[str, Any] = {}
    try:
        data_profile = _raw_loop_features_for_window_agent(
            df=dataset["cleaned_df"],
            loop_name=loop_name,
            loop_type=loop_type,
            sample_time_s=float(dataset["dt"]),
            csv_path=csv_path,
        )
    except Exception:
        ctx_profile = LoopContext(csv_path=csv_path, loop_type=loop_type)
        ctx_profile.cleaned_df = dataset["cleaned_df"]
        ctx_profile.dt = dataset["dt"]
        profile_result = _invoke_guarded_skill("summarize_data", {}, ctx_profile)
        if profile_result.success:
            data_profile = profile_result.data

    # data_analysis 阶段只产出"数据画像"——窗口检测属于 window_selection 阶段。
    # 这里**不再**带 candidate_windows / usable_windows / step_events 这些字段，
    # 否则前端会误以为"数据分析阶段已经选过窗"。
    yield stage_event("data_analysis", "done", {
        "data_points": dataset["data_points"],
        "sampling_time": dataset["dt"],
        "data_profile": data_profile,
    })

    # ── 早期门禁：MV 全程饱和 → 数据无激励，整定不可行 ──────────────────────────
    # 在花掉 4 分钟跑 MCP + LLM 之前，先用画像里现成的指标判一刀。
    # 实测一段 MV 95.6% 时间贴边的数据，跑完整流水线也只能产出 K 符号错的不可信模型；
    # 既无意义又浪费 LLM token。
    saturation_ratio = 0.0
    constraint_raw = data_profile.get("constraint_raw") if isinstance(data_profile, dict) else None
    if isinstance(constraint_raw, dict):
        saturation_ratio = float(constraint_raw.get("mv_saturation_ratio") or 0.0)
    actuator = data_profile.get("actuator_profile") if isinstance(data_profile, dict) else None
    if isinstance(actuator, dict):
        saturation_ratio = max(saturation_ratio, float(actuator.get("mv_saturation_ratio") or 0.0))
    if saturation_ratio >= 0.7:
        block_reason = (
            f"MV 在所选数据区间内 {saturation_ratio*100:.1f}% 时间贴边或饱和，"
            "执行机构没有真实激励，无法可靠辨识；建议更换时间区间或先排查阀门 / 联锁。"
        )
        empty_meta = {
            "mode": "blocked",
            "chosen_index": -1,
            "deterministic_index": -1,
            "deterministic_score": 0.0,
            "reasoning": block_reason,
            "formal_identification_allowed": False,
            "diagnostic_identification_allowed": False,
            "stop_reason": block_reason,
            "window_policy": None,
            "candidate_window_count": 0,
            "usable_window_count_pre_policy": 0,
            "step_event_count": 0,
            "algorithm_filter": list(algorithm_filter) if algorithm_filter else None,
            "ontology_context_source": "skipped",
            "ontology_context_used": False,
            "mv_saturation_ratio": saturation_ratio,
            "window_candidate_decision": {
                "selected_window_indices": [],
                "rejected_window_indices": [],
                "fallback_window_indices": [],
                "formal_identification_allowed": False,
                "diagnostic_identification_allowed": False,
                "stop_reason": block_reason,
                "primary_reason": block_reason,
                "ontology_evidence": [],
                "data_evidence": [{
                    "fact": "mv_saturation_ratio",
                    "value": round(saturation_ratio, 4),
                    "source": "data_profile",
                }],
                "window_judgements": [],
                "recommended_identification_plan": {"mode": "blocked"},
                "risk_flags": ["mv_saturated"],
            },
        }
        yield stage_event("window_selection", "done", empty_meta)
        yield result_event({
            "stop_after": "window_selection",
            "pipeline_status": "blocked_mv_saturated",
            "formal_identification_blocked": True,
            "block_reason": block_reason,
            "diagnostic_identification_allowed": False,
            "data_analysis": {
                "data_points": dataset["data_points"],
                "sampling_time": dataset["dt"],
                "step_events": [],
                "candidate_windows": [],
                "quality_metrics": dataset.get("quality_metrics"),
            },
            "window_selection": empty_meta,
            "model": None,
            "pid_params": None,
            "evaluation": None,
            "model_review": None,
            "loop_type": loop_type,
            "loop_name": loop_name,
        })
        return

    # 提前发 ontology_policy:running，让前端在 MCP 拉取期间也能显示"本体检索中"。
    # 之前是把这个事件放在 MCP 调用之后，MCP 慢/超时（最长 60s）时前端会有一段沉默，
    # 看上去像"卡在数据画像"或"页面变空白"。
    yield stage_event("ontology_policy", "running", {"phase": "fetching_mcp_context"})

    ontology_context_for_llm = ontology_context
    ontology_meta: dict[str, Any] = {
        "ontology_context_source": "frontend" if ontology_context else "none",
        "ontology_context_used": bool(ontology_context),
    }
    mcp_context: dict[str, Any] | None = None
    if use_llm_advisor and loop_name.strip():
        try:
            # MCP chat 工具内部走的是本体 LLM 推理，实测 30-60s。
            # 整体超时设 100s（略大于单 server 90s），覆盖少量 server 顺序回退场景。
            mcp_context = await asyncio.wait_for(
                fetch_loop_ontology_context_via_mcp(
                    loop_name=loop_name,
                    loop_type=loop_type,
                ),
                timeout=100.0,
            )
        except asyncio.TimeoutError:
            mcp_context = {
                "source": "registered_mcp_tool",
                "error": "MCP 本体检索整体超时 (100s)，已跳过",
                "content": "",
            }
        if mcp_context and mcp_context.get("content"):
            payload: dict[str, Any] = {
                "source": "mcp_ontology_context",
                "mcp_context": mcp_context,
            }
            if ontology_context:
                payload["frontend_context_fallback"] = ontology_context
            ontology_context_for_llm = json.dumps(payload, ensure_ascii=False)
            mcp_content = str(mcp_context.get("content", ""))
            ontology_meta.update({
                "ontology_context_source": "mcp",
                "ontology_context_used": True,
                "ontology_mcp_server": mcp_context.get("server_name"),
                "ontology_mcp_tool": mcp_context.get("tool"),
                "ontology_mcp_query": mcp_context.get("query"),
                "ontology_mcp_content_preview": mcp_content[:1200],
                "ontology_mcp_content_raw": mcp_content,
                "ontology_mcp_content_chars": len(mcp_content),
            })
        elif mcp_context and mcp_context.get("error"):
            ontology_meta.update({
                "ontology_mcp_error": str(mcp_context.get("error", ""))[:500],
            })

    # MCP 阶段结束，开始构建/请求策略；前端可切换到"策略生成中"。
    yield stage_event("ontology_policy", "running", {"phase": "building_policy"})
    ontology_ctx = _build_skill_context(
        csv_path=csv_path,
        loop_type=loop_type,
        dataset=dataset,
        candidate_windows=[],
        data_profile=data_profile,
    )
    ontology_skill = await asyncio.to_thread(
        _invoke_guarded_skill,
        "build_ontology_policy",
        {
            "loop_name": loop_name,
            "loop_type": loop_type,
            "frontend_context": ontology_context,
            "mcp_context": mcp_context,
            "use_llm_advisor": use_llm_advisor,
        },
        ontology_ctx,
    )
    if ontology_skill.success:
        window_policy = ontology_skill.data.get("policy", {})
        policy_source = str(ontology_skill.data.get("source") or "default")
        ontology_meta.update(ontology_skill.data.get("ontology_meta", {}))
        data_profile["window_policy"] = window_policy
        data_profile["ontology_context"] = ontology_ctx.data_profile.get("ontology_context", {})
        reasoning_chain = str(window_policy.get("llm_policy_reasoning_content") or "")
        if reasoning_chain:
            yield {
                "type": "llm_thinking",
                "stage": "ontology_policy",
                "model": "deepseek-reasoner",
                "reasoning_content": reasoning_chain,
                "raw_text": window_policy.get("llm_policy_raw_text", ""),
            }
    else:
        window_policy = build_window_selection_policy(
            loop_name=loop_name,
            loop_type=loop_type,
            data_profile=data_profile,
            mcp_context=mcp_context,
            frontend_context=ontology_context,
        )
        policy_source = "fallback"
    yield stage_event("ontology_policy", "done", {
        "policy": window_policy,
        "confidence": window_policy.get("confidence", 0.0),
        "source": policy_source,
        "ontology_source": (window_policy.get("ontology_facts") or {}).get("source", "none"),
    })

    # 策略生成后再运行窗口算法族，让前/后窗、稳态扫描窗口、合并间隔等来自策略。
    detect_ctx = LoopContext(
        csv_path=csv_path,
        loop_prefix=selected_loop_prefix or "",
        loop_type=loop_type,
    )
    detect_ctx.cleaned_df = dataset["cleaned_df"]
    detect_ctx.dt = dataset["dt"]
    detect_result = _invoke_guarded_skill(
        "detect_windows",
        {"loop_type": loop_type, "policy": window_policy},
        detect_ctx,
    )
    if detect_result.success:
        candidate_windows = list(detect_ctx.candidate_windows)
        if selected_window_index is not None and 0 <= selected_window_index < len(candidate_windows):
            candidate_windows.insert(0, candidate_windows.pop(selected_window_index))
        dataset["candidate_windows"] = candidate_windows
        dataset["step_events"] = [None] * int(detect_result.data.get("step_event_count", 0))
        dataset["window_detection_meta"] = (
            detect_result.data.get("meta", {}) if isinstance(detect_result.data, dict) else {}
        )
    else:
        candidate_windows = list(dataset.get("candidate_windows") or [])
        dataset["candidate_windows"] = candidate_windows
        dataset["window_detection_meta"] = {
            "error": detect_result.reasoning,
            "policy_applied": bool(window_policy),
        }

    candidate_windows = dataset["candidate_windows"]

    # 应用算法白名单（命中 window_algorithm 或 window_algorithm_label 任一即保留）。
    # 在 usable_windows 计算之前过滤，让后续选窗 / 辨识全程只看白名单内的窗口。
    if algorithm_filter:
        allowed = {str(a).strip() for a in algorithm_filter if str(a).strip()}
        if allowed:
            candidate_windows = [
                w for w in candidate_windows
                if str(w.get("window_algorithm", "")) in allowed
                or str(w.get("window_algorithm_label", "")) in allowed
            ]
            dataset["candidate_windows"] = candidate_windows

    usable_windows = [w for w in candidate_windows if w.get("window_usable_for_id")]

    # 阶段语义：data_analysis 仅负责数据画像（在前面已经发过一次 done）；
    # 候选窗口数、可用窗口数、算法白名单、窗口检测元数据都属于 window_selection 阶段，
    # 之后会在 selection_meta 里统一带出，避免给前端"数据分析也在选窗"的错觉。

    if not candidate_windows:
        block_reason = "未发现任何候选窗口，不建议继续正式系统辨识和 PID 整定。"
        if algorithm_filter:
            block_reason += " 当前算法白名单过滤后为空，请放宽算法筛选。"
        selection_meta = {
            "mode": "blocked",
            "chosen_index": -1,
            "deterministic_index": -1,
            "deterministic_score": 0.0,
            "reasoning": block_reason,
            "formal_identification_allowed": False,
            "diagnostic_identification_allowed": False,
            "stop_reason": block_reason,
            "window_policy": window_policy,
            **ontology_meta,
            "window_policy_results": [],
            "candidate_window_count": 0,
            "usable_window_count_pre_policy": 0,
            "step_event_count": 0,
            "algorithm_filter": list(algorithm_filter) if algorithm_filter else None,
            "window_detection_meta": dataset.get("window_detection_meta", {}),
            "window_candidate_decision": {
                "selected_window_indices": [],
                "rejected_window_indices": [],
                "fallback_window_indices": [],
                "formal_identification_allowed": False,
                "diagnostic_identification_allowed": False,
                "stop_reason": block_reason,
                "primary_reason": block_reason,
                "ontology_evidence": [],
                "data_evidence": [{
                    "fact": "candidate_window_count",
                    "value": 0,
                    "source": "detect_windows",
                }],
                "window_judgements": [],
                "recommended_identification_plan": {"mode": "blocked"},
                "risk_flags": ["no_candidate_window"],
            },
        }
        yield stage_event("window_selection", "done", selection_meta)
        yield result_event({
            "stop_after": "window_selection",
            "pipeline_status": "blocked_no_candidate_window",
            "formal_identification_blocked": True,
            "block_reason": block_reason,
            "diagnostic_identification_allowed": False,
            "data_analysis": {
                "data_points": dataset["data_points"],
                "sampling_time": dataset["dt"],
                "step_events": dataset["step_events"],
                "candidate_windows": candidate_windows,
                "quality_metrics": dataset.get("quality_metrics"),
            },
            "window_selection": selection_meta,
            "model": None,
            "pid_params": None,
            "evaluation": None,
            "model_review": None,
            "loop_type": loop_type,
            "loop_name": loop_name,
        })
        return

    # ── Stage 1.5: Window Selection (algorithm providers -> LLM -> gate) ─
    yield stage_event("window_selection", "running", {"phase": "algorithm"})

    selection_meta: dict[str, Any] = {
        "window_algorithm_family_summaries": (
            dataset.get("window_detection_meta", {}).get("family_summaries", [])
            if isinstance(dataset.get("window_detection_meta"), dict)
            else []
        ),
        # data_analysis 阶段不再发"候选/可用窗口数"；这些都属于 window_selection。
        # 把窗口检测的元数据原样搬到这里，前端读 window_selection.* 即可。
        "candidate_window_count": len(candidate_windows),
        "usable_window_count_pre_policy": len(usable_windows),
        "step_event_count": len(dataset["step_events"]),
        "algorithm_filter": list(algorithm_filter) if algorithm_filter else None,
        "window_detection_meta": dataset.get("window_detection_meta", {}),
    }

    selection_meta.update(ontology_meta)
    selection_meta["window_policy"] = window_policy

    candidate_windows, policy_results = apply_window_policy_to_candidates(candidate_windows, window_policy)
    dataset["candidate_windows"] = candidate_windows
    usable_windows = [w for w in candidate_windows if w.get("window_usable_for_id")]
    pool = usable_windows if usable_windows else candidate_windows
    pool_indices = [candidate_windows.index(w) for w in pool]
    selection_meta["window_policy_results"] = policy_results
    selection_meta["window_algorithm_family_summaries"] = _overlay_window_algorithm_family_summaries(
        candidate_windows=candidate_windows,
        policy=window_policy,
        previous=selection_meta.get("window_algorithm_family_summaries", []),
    )

    deterministic_ctx = _build_skill_context(
        csv_path=csv_path,
        loop_type=loop_type,
        dataset=dataset,
        candidate_windows=pool,
        selected_window_index=selected_window_index,
    )
    deterministic_result = _invoke_guarded_skill(
        "select_window",
        {"provider": "quality_score_selector"},
        deterministic_ctx,
    )
    if deterministic_result.success:
        deterministic_pool_idx = int(deterministic_result.data.get("chosen_index", 0))
        deterministic_score = float(deterministic_result.data.get("score", 0.0))
        deterministic_summary = deterministic_result.data.get("chosen_window_summary")
    else:
        deterministic_pool_idx = max(
            range(len(pool)),
            key=lambda i: float(pool[i].get("window_quality_score", 0.0)),
        )
        deterministic_score = float(
            pool[deterministic_pool_idx].get("window_quality_score", 0.0)
        )
        deterministic_summary = None
    deterministic_global_idx = pool_indices[deterministic_pool_idx]
    deterministic_window = candidate_windows[deterministic_global_idx]
    selection_meta.update({
        "deterministic_index": deterministic_global_idx,
        "deterministic_score": deterministic_score,
        "deterministic_window_summary": deterministic_summary or {
            "source": deterministic_window.get("window_source"),
            "score": float(deterministic_window.get("window_quality_score", 0.0)),
            "n_points": int(deterministic_window.get("window_end_idx", 0))
            - int(deterministic_window.get("window_start_idx", 0)),
        },
        "policy_adjusted_usable_windows": len(usable_windows),
        "policy_adjusted_candidate_windows": len(candidate_windows),
    })

    yield stage_event("window_selection", "running", {"phase": "llm"})

    chosen_global_idx: int
    if selected_window_index is not None and 0 <= selected_window_index < len(candidate_windows):
        # 工程师手动指定，最高优先级
        chosen_global_idx = selected_window_index
        selection_meta.update({
            "mode": "user_override",
            "chosen_index": chosen_global_idx,
            "reasoning": "工程师手动指定窗口",
        })
    elif use_llm_advisor and len(pool) > 1:
        # 同步 LLM 调用，丢到线程里避免堵 event loop
        advisor = await asyncio.to_thread(
            choose_window_via_llm,
            data_profile=data_profile,
            candidate_windows=pool,
            loop_type=loop_type,
            ontology_context=ontology_context_for_llm,
        )

        if advisor is not None:
            chosen_pool_idx = advisor["chosen_index"]
            chosen_global_idx = pool_indices[chosen_pool_idx]
            # 把 R1 的可展示分析摘要作为独立事件发出，前端可以折叠展示
            reasoning_chain = advisor.get("reasoning_content", "") or ""
            if reasoning_chain:
                yield {
                    "type": "llm_thinking",
                    "stage": "window_selection",
                    "model": "deepseek-reasoner",
                    "reasoning_content": reasoning_chain,
                    "raw_text": advisor.get("raw_text", ""),
                }
            window_judgements: list[dict[str, Any]] = []
            for item in advisor.get("window_judgements", []) or []:
                try:
                    pool_idx = int(item.get("index"))
                    global_idx = pool_indices[pool_idx]
                except (TypeError, ValueError, IndexError):
                    continue
                mapped = dict(item)
                mapped["pool_index"] = pool_idx
                mapped["index"] = global_idx
                mapped["window_source"] = pool[pool_idx].get("window_source")
                mapped["window_quality_score"] = pool[pool_idx].get("window_quality_score")
                window_judgements.append(mapped)
            selection_meta.update({
                "mode": "llm",
                "chosen_index": chosen_global_idx,
                "reasoning": advisor["reasoning"],
                "llm_reasoning_chain_len": len(reasoning_chain),
                "agreed_with_deterministic": chosen_global_idx == deterministic_global_idx,
                "ontology_evidence": advisor.get("ontology_evidence", []),
                "window_judgements": window_judgements,
            })
        else:
            chosen_global_idx = deterministic_global_idx
            selection_meta.update({
                "mode": "fallback_deterministic",
                "chosen_index": chosen_global_idx,
                "reasoning": "LLM 顾问不可用或返回非法，回退到 quality_score 最高窗口",
            })
    else:
        # LLM 关闭或池里只有一个窗口
        chosen_global_idx = deterministic_global_idx
        selection_meta.update({
            "mode": "deterministic",
            "chosen_index": chosen_global_idx,
            "reasoning": (
                "仅一个候选窗口" if len(pool) <= 1 else "LLM 顾问已关闭，按 quality_score 选窗"
            ),
        })

    # window_selection 给的"首选窗口"只用于展示。实际辨识把全部 usable 窗口都喂给
    # fit_best_model，让它通过 AIC 在「窗口×模型」笛卡尔积里挑最优。
    # 之前锁死单窗口会让信号量级最大但拟合最差的窗口（如夹杂操作员手动干预）误胜。
    chosen_window = candidate_windows[chosen_global_idx]
    windows_for_fit = list(pool)
    # 把选中的窗口提到首位（仅影响打分相同时的 tie-break / 展示顺序）
    if chosen_window in windows_for_fit:
        windows_for_fit.remove(chosen_window)
        windows_for_fit.insert(0, chosen_window)
    if chosen_global_idx == deterministic_global_idx and deterministic_summary is not None:
        selection_meta["chosen_window_summary"] = deterministic_summary
    else:
        selection_meta["chosen_window_summary"] = {
            "source": chosen_window.get("window_source"),
            "score": float(chosen_window.get("window_quality_score", 0.0)),
            "n_points": int(chosen_window.get("window_end_idx", 0))
            - int(chosen_window.get("window_start_idx", 0)),
        }

    yield stage_event("window_selection", "running", {"phase": "gate"})

    selected_indices = [candidate_windows.index(w) for w in usable_windows if w in candidate_windows]
    rejected_indices = [i for i, w in enumerate(candidate_windows) if not w.get("window_usable_for_id")]
    formal_identification_allowed = bool(selected_indices)
    diagnostic_identification_allowed = bool(candidate_windows)
    stop_reason = None
    primary_reason = "存在可用于正式辨识的候选窗口，可以继续进入系统辨识。"
    risk_flags: list[str] = []
    if not formal_identification_allowed:
        stop_reason = (
            "窗口候选阶段未发现满足最低准入条件的可用窗口，"
            "不建议继续正式系统辨识和 PID 整定。"
        )
        primary_reason = (
            "当前数据只允许诊断性辨识：可用来解释激励不足、饱和、扰动或窗口质量问题，"
            "但不能生成整定参数。"
        )
        risk_flags.append("no_formal_identification_window")

    window_candidate_decision = {
        "selected_window_indices": selected_indices,
        "rejected_window_indices": rejected_indices,
        "fallback_window_indices": pool_indices if not selected_indices else [],
        "formal_identification_allowed": formal_identification_allowed,
        "diagnostic_identification_allowed": diagnostic_identification_allowed,
        "stop_reason": stop_reason,
        "primary_reason": primary_reason,
        "ontology_evidence": selection_meta.get("ontology_evidence", []),
        "data_evidence": [
            {
                "fact": "usable_window_count",
                "value": len(usable_windows),
                "source": "detect_windows",
            },
            {
                "fact": "candidate_window_count",
                "value": len(candidate_windows),
                "source": "detect_windows",
            },
        ],
        "window_judgements": selection_meta.get("window_judgements", []),
        "recommended_identification_plan": {
            "mode": "formal" if formal_identification_allowed else "diagnostic_only",
            "model_types": ["FO", "FOPDT", "SOPDT"],
            "constraints": {
                "expected_gain_sign": window_policy.get("expected_gain_sign", "unknown"),
                "expected_time_constant_range_s": window_policy.get("expected_time_constant_range_s"),
                "expected_dead_time_range_s": window_policy.get("expected_dead_time_range_s"),
            },
        },
        "risk_flags": risk_flags,
    }
    selection_meta.update({
        "formal_identification_allowed": formal_identification_allowed,
        "diagnostic_identification_allowed": diagnostic_identification_allowed,
        "stop_reason": stop_reason,
        "window_candidate_decision": window_candidate_decision,
    })

    yield stage_event("window_selection", "done", selection_meta)

    # 早停点 #1：数据分析菜单只关心到选窗为止
    if stop_after == "window_selection":
        yield result_event({
            "stop_after": "window_selection",
            "data_analysis": {
                "data_points": dataset["data_points"],
                "sampling_time": dataset["dt"],
                "step_events": dataset["step_events"],
                "candidate_windows": candidate_windows,
                "quality_metrics": dataset.get("quality_metrics"),
            },
            "window_selection": selection_meta,
            "model": None,
            "pid_params": None,
            "evaluation": None,
            "model_review": None,
            "loop_type": loop_type,
            "loop_name": loop_name,
        })
        return

    # ── Stage 2 / 2.5: Identification + Review with refinement loop ─────
    # Phase 2: 多轮"辨识 → 评审 → 精修指令 → 辨识"循环，最多 MAX_REFINEMENT_ROUNDS 轮重试
    # （即 1 次初始辨识 + 至多 2 次精修 = 最多 3 次辨识调用）。
    if not formal_identification_allowed:
        yield stage_event("identification", "done", {
            "mode": "blocked",
            "formal_identification_allowed": False,
            "diagnostic_identification_allowed": diagnostic_identification_allowed,
            "reason": stop_reason,
        })
        yield result_event({
            "stop_after": "window_selection",
            "pipeline_status": "blocked_no_formal_window",
            "formal_identification_blocked": True,
            "block_reason": stop_reason,
            "diagnostic_identification_allowed": diagnostic_identification_allowed,
            "data_analysis": {
                "data_points": dataset["data_points"],
                "sampling_time": dataset["dt"],
                "step_events": dataset["step_events"],
                "candidate_windows": candidate_windows,
                "quality_metrics": dataset.get("quality_metrics"),
            },
            "window_selection": selection_meta,
            "model": None,
            "pid_params": None,
            "evaluation": None,
            "model_review": None,
            "loop_type": loop_type,
            "loop_name": loop_name,
        })
        return

    MAX_REFINEMENT_ROUNDS = 2

    # 给精修顾问看的窗口摘要（不含底层 df）
    windows_summary: list[dict[str, Any]] = [
        {
            "index": i,
            "source": w.get("window_source", ""),
            "n_points": int(w.get("window_end_idx", 0)) - int(w.get("window_start_idx", 0)),
            "score": float(w.get("window_quality_score", 0.0)),
            "corr": float(w.get("window_corr", 0.0)),
            "algorithm": w.get("window_algorithm", ""),
            "algorithm_label": w.get("window_algorithm_label", ""),
            "score_breakdown": w.get("window_score_breakdown", {}),
        }
        for i, w in enumerate(windows_for_fit)
    ]

    # 跨轮累积，Phase 3 用来挑"尽力而为"的最高分模型
    all_round_records: list[dict[str, Any]] = []
    refinement_history: list[dict[str, Any]] = []

    # 当前轮的辨识参数（首轮用默认）
    current_force_models: list[str] | None = None
    current_force_window_idx: int | None = None
    current_force_L_hint: float | None = None

    final_model = None
    final_confidence = None
    final_id_result: dict[str, Any] | None = None
    final_review_result: dict[str, Any] | None = None
    review_unreliable = False
    review_unreliable_reason = ""

    for round_idx in range(MAX_REFINEMENT_ROUNDS + 1):  # round 0 = 初始辨识
        # ── 本轮辨识 ──
        running_data = {"round": round_idx, "max_rounds": MAX_REFINEMENT_ROUNDS}
        if round_idx > 0:
            running_data["refinement"] = {
                "force_window_index": current_force_window_idx,
                "force_model_types": current_force_models or [],
                "hint_L": current_force_L_hint,
            }
        yield stage_event("identification", "running", running_data)

        # 按精修指令准备本轮的窗口集合
        round_windows = windows_for_fit
        if current_force_window_idx is not None and 0 <= current_force_window_idx < len(windows_for_fit):
            round_windows = [windows_for_fit[current_force_window_idx]]
        round_window_meta = _window_meta_by_source(round_windows)

        skill_ctx = _build_skill_context(
            csv_path=csv_path,
            loop_type=loop_type,
            dataset=dataset,
            candidate_windows=round_windows,
            selected_window_index=current_force_window_idx,
            data_profile=data_profile,
        )
        skill_id = _invoke_guarded_skill(
            "identify_model",
            {
                "provider": "transfer_function_fit",
                "use_usable_windows_only": False,
                "model_pool": current_force_models,
                "hint_L": current_force_L_hint,
            },
            skill_ctx,
        )
        if skill_id.success and skill_id.data.get("best_model"):
            id_result = {
                "model": _process_model_from_skill(skill_id.data["best_model"]),
                "confidence": _confidence_from_skill(skill_id.data["best_model"]),
                "window_source": skill_id.data.get("window_source", ""),
                "selection_reason": skill_id.data.get("selection_reason", ""),
                "fit_preview": skill_id.data.get("fit_preview", {}),
                "candidates": skill_id.data.get("candidates", []),
                "attempts": skill_id.data.get("attempts", []),
            }
        else:
            try:
                id_result = fit_best_model(
                    cleaned_df=dataset["cleaned_df"],
                    candidate_windows=round_windows,
                    actual_dt=dataset["dt"],
                    loop_type=loop_type,
                    quality_metrics=dataset.get("quality_metrics"),
                    force_model_types=current_force_models,
                    force_L_hint=current_force_L_hint,
                )
            except ValueError as exc:
                yield error_event(str(exc), stage="identification", error_code="ID_ERROR")
                return

        model = id_result["model"]
        confidence = id_result["confidence"]

        # 整理本轮 attempts 给前端
        raw_attempts = id_result.get("attempts", []) or []
        attempts_payload: list[dict[str, Any]] = []
        for a in raw_attempts:
            source = str(a.get("window_source", ""))
            window_meta = round_window_meta.get(source, {})
            if a.get("success"):
                attempts_payload.append({
                    "model_type": a.get("model_type"),
                    "window_source": source,
                    "window_algorithm": a.get("window_algorithm") or window_meta.get("window_algorithm", ""),
                    "window_algorithm_label": a.get("window_algorithm_label") or window_meta.get("window_algorithm_label", ""),
                    "window_quality_score": float(a.get("window_quality_score", window_meta.get("window_quality_score", 0.0)) or 0.0),
                    "window_score_breakdown": a.get("window_score_breakdown") or window_meta.get("window_score_breakdown", {}),
                    "K": float(a.get("K", 0.0)),
                    "T": float(a.get("T", 0.0)),
                    "T1": float(a.get("T1", 0.0)),
                    "T2": float(a.get("T2", 0.0)),
                    "L": float(a.get("L", 0.0)),
                    "zeta": float(a["zeta"]) if a.get("zeta") is not None else None,
                    "r2_score": float(a.get("r2_score", 0.0)),
                    "normalized_rmse": float(a.get("normalized_rmse", 0.0)),
                    "fit_score": float(a.get("fit_score", 0.0)),
                    "confidence": float(a.get("confidence", 0.0)),
                    "degenerate_T": bool(a.get("degenerate_T", False)),
                    "fit_preview": a.get("fit_preview"),
                    "success": True,
                    "round": round_idx,
                })
            else:
                attempts_payload.append({
                    "model_type": a.get("model_type"),
                    "window_source": source,
                    "window_algorithm": a.get("window_algorithm") or window_meta.get("window_algorithm", ""),
                    "window_algorithm_label": a.get("window_algorithm_label") or window_meta.get("window_algorithm_label", ""),
                    "success": False,
                    "error": str(a.get("error", ""))[:200],
                    "round": round_idx,
                })
        attempts_payload.sort(
            key=lambda x: float(x.get("fit_score", -1e12)) if x.get("success") else -1e12,
            reverse=True,
        )

        yield stage_event("identification", "done", {
            "round": round_idx,
            "max_rounds": MAX_REFINEMENT_ROUNDS,
            "model_type": model.model_type.value,
            "K": model.K,
            "T": model.T or (model.T1 + model.T2),
            "L": model.L,
            "r2_score": model.r2_score,
            "confidence": confidence.confidence,
            "window_source": id_result["window_source"],
            "best_window_source": id_result["window_source"],
            "attempts": attempts_payload,
            "algorithm_comparison": _algorithm_comparison(attempts_payload),
        })

        # 置信度极低 → 走 best-effort：标 review_unreliable=True 让评估阶段封顶 + 给警告，
        # 但不再 abort 流水线（之前 LLM 关闭时直接抛 LOW_CONFIDENCE，会让"是否能跑完"
        # 取决于 LLM 是否启用，违反"LLM 是可关闭增量"的设计原则）。
        if confidence.confidence < 0.35 and round_idx == 0 and not use_llm_advisor:
            review_unreliable = True
            review_unreliable_reason = (
                f"模型置信度 {confidence.confidence:.2f} 过低（LLM 顾问已关闭，无评审兜底）；"
                "已按尽力而为输出参数，请勿直接下发"
            )
            final_model = model
            final_confidence = confidence
            final_id_result = id_result
            break

        # ── 本轮评审 ──
        review_result: dict[str, Any] | None = None
        retry_plan_from_review: dict[str, Any] | None = None
        if use_llm_advisor:
            yield stage_event("model_review", "running", {"round": round_idx})
            best_for_review = {
                "model_type": model.model_type.value,
                "K": model.K, "T": model.T, "T1": model.T1, "T2": model.T2, "L": model.L,
                "r2_score": model.r2_score,
                "normalized_rmse": model.normalized_rmse,
                "window_source": id_result["window_source"],
            }
            review_ctx = _build_skill_context(
                csv_path=csv_path,
                loop_type=loop_type,
                dataset=dataset,
                candidate_windows=candidate_windows,
                selected_window_index=chosen_global_idx,
                data_profile=data_profile,
                model={**best_for_review, "confidence": confidence.confidence},
                confidence=confidence.confidence,
            )
            review_skill = await asyncio.to_thread(
                _invoke_guarded_skill,
                "review_identification",
                {
                    "chosen_window_summary": selection_meta.get("chosen_window_summary", {}),
                    "attempts": id_result.get("attempts", []),
                    "windows_summary": windows_summary,
                    "algorithm_comparison": _algorithm_comparison(attempts_payload),
                    "history_summary": [
                        {
                            "round": h["round"],
                            "window_index": h["force_window_index"],
                            "model_types": h["force_models"],
                            "hint_L": h["force_L_hint"],
                            "best_type": h["model"].model_type.value,
                            "best_r2": float(h["model"].r2_score),
                            "verdict": (h["review"] or {}).get("verdict", "accept"),
                        }
                        for h in all_round_records
                    ],
                    "round_idx": round_idx,
                    "max_refinement_rounds": MAX_REFINEMENT_ROUNDS,
                    "allow_retry_plan": True,
                    "use_llm_advisor": use_llm_advisor,
                },
                review_ctx,
            )
            review_result = dict(review_skill.data) if review_skill.success else {
                "available": False,
                "verdict": "accept",
                "reason": review_skill.reasoning or "模型评审 skill 不可用",
                "concerns": [],
                "fallback": True,
            }
            retry_plan_from_review = review_result.get("retry_plan") if isinstance(review_result.get("retry_plan"), dict) else None
            if not review_result.get("available", True):
                # LLM 评审失败 → 默认采纳本轮，跳出循环
                failure_type = str(review_result.get("error_type", "")).strip() or "unknown"
                failure_message = str(review_result.get("error_message", "")).strip() or "LLM 评审失败"
                yield stage_event("model_review", "done", {
                    "round": round_idx,
                    "verdict": "accept",
                    "reason": "LLM 评审不可用，默认采纳算法选择的模型",
                    "concerns": [],
                    "fallback": True,
                    "error_type": failure_type,
                    "error_message": failure_message,
                    "raw_text": review_result.get("raw_text", ""),
                })
                review_result = {
                    "available": False,
                    "verdict": "accept",
                    "reason": "LLM 评审不可用",
                    "concerns": [],
                    "fallback": True,
                    "error_type": failure_type,
                    "error_message": failure_message,
                    "raw_text": review_result.get("raw_text", ""),
                }
            else:
                verdict = review_result["verdict"]
                reasoning_chain = review_result.get("reasoning_content", "") or ""
                if reasoning_chain:
                    yield {
                        "type": "llm_thinking",
                        "stage": "model_review",
                        "round": round_idx,
                        "model": "deepseek-reasoner",
                        "reasoning_content": reasoning_chain,
                        "raw_text": review_result.get("raw_text", ""),
                    }
                yield stage_event("model_review", "done", {
                    "round": round_idx,
                    "verdict": verdict,
                    "reason": review_result["reason"],
                    "concerns": review_result["concerns"],
                    "fallback": False,
                    "error_type": None,
                    "error_message": None,
                })

        # 把本轮快照存进跨轮记录（Phase 3 用）
        all_round_records.append({
            "round": round_idx,
            "id_result": id_result,
            "model": model,
            "confidence": confidence,
            "attempts_payload": attempts_payload,
            "review": review_result,
            "force_models": current_force_models,
            "force_window_index": current_force_window_idx,
            "force_L_hint": current_force_L_hint,
        })

        verdict_now = (review_result or {}).get("verdict", "accept")

        # accept 或 LLM 关闭 → 收敛
        if not use_llm_advisor or verdict_now == "accept":
            final_model = model
            final_confidence = confidence
            final_id_result = id_result
            final_review_result = review_result
            break

        # downgrade 但已用完轮次预算 → 跳到 Phase 3 兜底
        if round_idx >= MAX_REFINEMENT_ROUNDS:
            review_unreliable = True
            review_unreliable_reason = (
                f"经过 {MAX_REFINEMENT_ROUNDS + 1} 轮辨识仍被降级：{review_result['reason']}"
            )
            break

        algorithm_comparison = _algorithm_comparison(attempts_payload)

        # ── 询问精修顾问下一轮怎么改 ──
        yield stage_event("identification_refinement", "running", {"round": round_idx + 1})
        refinement = retry_plan_from_review

        if refinement is None or not refinement.get("retry"):
            deterministic_refinement = recommend_refinement_from_algorithm_comparison(
                loop_type=loop_type,
                windows_summary=windows_summary,
                algorithm_comparison=algorithm_comparison,
                last_best=best_for_review,
                last_review=review_result,
            )
            if deterministic_refinement is not None:
                refinement = deterministic_refinement
            else:
                # LLM 不可用或主动放弃重试 → 跳出循环走 Phase 3
                yield stage_event("identification_refinement", "done", {
                    "round": round_idx + 1,
                    "retry": False,
                    "source": "llm" if refinement is not None else "none",
                    "rationale": (refinement or {}).get("rationale", "LLM 精修顾问不可用或决定放弃重试，且确定性策略无可用备选"),
                })
                review_unreliable = True
                review_unreliable_reason = (
                    f"第 {round_idx} 轮被降级，精修顾问不再建议重试：{review_result['reason']}"
                )
                break

        if refinement.get("source") == "deterministic_algorithm_policy":
            yield stage_event("identification_refinement", "done", {
                "round": round_idx + 1,
                "retry": True,
                "source": refinement.get("source"),
                "rationale": refinement.get("rationale", ""),
                "force_window_index": refinement.get("force_window_index"),
                "force_model_types": refinement.get("force_model_types") or [],
                "hint_L": refinement.get("hint_L"),
                "recommended_algorithm": refinement.get("recommended_algorithm", ""),
                "recommended_algorithm_label": refinement.get("recommended_algorithm_label", ""),
                "recommended_window_source": refinement.get("recommended_window_source", ""),
                "evidence": refinement.get("evidence", {}),
            })
            refinement_history.append({
                "round": round_idx + 1,
                "rationale": refinement.get("rationale", ""),
                "force_window_index": refinement.get("force_window_index"),
                "force_model_types": refinement.get("force_model_types") or [],
                "hint_L": refinement.get("hint_L"),
            })
            current_force_window_idx = refinement.get("force_window_index")
            current_force_models = refinement.get("force_model_types") or None
            current_force_L_hint = refinement.get("hint_L")
            continue

        # 收到重试指令 → 配置下一轮
        rc = refinement.get("reasoning_content", "") or ""
        if rc:
            yield {
                "type": "llm_thinking",
                "stage": "identification_refinement",
                "round": round_idx + 1,
                "model": "deepseek-reasoner",
                "reasoning_content": rc,
                "raw_text": refinement.get("raw_text", ""),
            }
        yield stage_event("identification_refinement", "done", {
            "round": round_idx + 1,
            "retry": True,
            "rationale": refinement.get("rationale", ""),
            "force_window_index": refinement.get("force_window_index"),
            "force_model_types": refinement.get("force_model_types") or [],
            "hint_L": refinement.get("hint_L"),
        })
        refinement_history.append({
            "round": round_idx + 1,
            "rationale": refinement.get("rationale", ""),
            "force_window_index": refinement.get("force_window_index"),
            "force_model_types": refinement.get("force_model_types") or [],
            "hint_L": refinement.get("hint_L"),
        })
        current_force_window_idx = refinement.get("force_window_index")
        current_force_models = refinement.get("force_model_types") or None
        current_force_L_hint = refinement.get("hint_L")

    # ── 循环结束：确定最终用于整定的模型 ──
    # Phase 2 默认用最后一轮（accept 或循环耗尽）；Phase 3 在下面接管"跨轮选最高分"
    if final_model is None:
        # 走到这里意味着 downgrade 用完轮次或精修放弃
        def _round_score(record: dict[str, Any]) -> tuple[float, float]:
            attempts = record.get("attempts_payload") or []
            best_fit = max(
                (
                    float(a.get("fit_score", -1e12))
                    for a in attempts
                    if a.get("success")
                ),
                default=float(record["model"].r2_score),
            )
            return best_fit, float(record["confidence"].confidence)

        best = max(all_round_records, key=_round_score)
        final_model = best["model"]
        final_confidence = best["confidence"]
        final_id_result = best["id_result"]
        final_review_result = best["review"]

    # 重新绑定原变量名，让下游 tuning/evaluation 代码不用改
    model = final_model
    confidence = final_confidence
    id_result = final_id_result
    final_attempts_payload = next(
        (
            record["attempts_payload"]
            for record in all_round_records
            if record.get("id_result") is final_id_result
        ),
        all_round_records[-1]["attempts_payload"] if all_round_records else [],
    )

    # 早停点 #2：系统辨识菜单只关心到模型辨识 + 评审 + 精修结束
    if stop_after == "identification":
        yield result_event({
            "stop_after": "identification",
            "data_analysis": {
                "data_points": dataset["data_points"],
                "sampling_time": dataset["dt"],
                "step_events": dataset["step_events"],
                "candidate_windows": candidate_windows,
                "quality_metrics": dataset.get("quality_metrics"),
            },
            "window_selection": selection_meta,
            "model": {
                "model_type": model.model_type.value,
                "K": model.K,
                "T": model.T,
                "T1": model.T1,
                "T2": model.T2,
                "L": model.L,
                "r2_score": model.r2_score,
                "normalized_rmse": model.normalized_rmse,
                "confidence": confidence.confidence,
                "confidence_quality": confidence.quality,
                "window_source": id_result["window_source"],
                "selection_reason": id_result["selection_reason"],
                "fit_preview": id_result.get("fit_preview", {}),
                "candidates": id_result.get("candidates", []),
                "attempts": final_attempts_payload,
                "algorithm_comparison": _algorithm_comparison(final_attempts_payload),
            },
            "pid_params": None,
            "evaluation": None,
            "model_review": (
                {
                    "verdict": final_review_result["verdict"],
                    "reason": final_review_result["reason"],
                    "concerns": final_review_result["concerns"],
                }
                if final_review_result is not None else None
            ),
            "refinement_history": refinement_history,
            "loop_type": loop_type,
            "loop_name": loop_name,
        })
        return

    # ── Stage 3: PID Tuning ─────────────────────────────────────────────
    yield stage_event("tuning", "running")
    tuning_params = model.to_tuning_params(dataset["dt"], dataset["data_points"])
    tuning_model_params = {
        "model_type": model.model_type.value,
        "K": model.K, "T": model.T,
        "T1": model.T1, "T2": model.T2, "L": model.L,
    }
    skill_ctx = _build_skill_context(
        csv_path=csv_path,
        loop_type=loop_type,
        dataset=dataset,
        candidate_windows=candidate_windows,
        selected_window_index=chosen_global_idx,
        data_profile=data_profile,
        model=tuning_model_params,
        confidence=confidence.confidence,
    )
    skill_ctx.model.update({
        "r2_score": model.r2_score,
        "normalized_rmse": model.normalized_rmse,
    })
    skill_tuning = _invoke_guarded_skill(
        "generate_tuning_candidates",
        {"provider": "classic_family"},
        skill_ctx,
    )
    if skill_tuning.success and skill_tuning.data.get("recommended"):
        tuning_result = {
            "best": skill_tuning.data.get("recommended", {}),
            "heuristic_strategy": skill_tuning.data.get("heuristic_strategy", ""),
            "heuristic_reason": skill_tuning.data.get("heuristic_reason", ""),
            "all_candidates": skill_tuning.data.get("candidates", []),
            "tuning_unreliable": skill_tuning.data.get("tuning_unreliable", False),
            "tuning_unreliable_reason": skill_tuning.data.get("tuning_unreliable_reason", ""),
        }
    else:
        tuning_result = select_best_strategy(
            K=tuning_params["K"],
            T=tuning_params["T"],
            L=tuning_params["L"],
            dt=dataset["dt"],
            loop_type=loop_type,
            model_type=model.model_type.value,
            model_params=tuning_model_params,
            confidence=confidence.confidence,
            nrmse=model.normalized_rmse,
            r2=model.r2_score,
        )

    pid = tuning_result["best"] or {}
    yield stage_event("tuning", "done", {
        "strategy": pid.get("strategy", ""),
        "Kp": pid.get("Kp", 0.0),
        "Ki": pid.get("Ki", 0.0),
        "Kd": pid.get("Kd", 0.0),
    })

    # ── Stage 4: Evaluation ─────────────────────────────────────────────
    yield stage_event("evaluation", "running")
    eval_model_params = {
        "model_type": model.model_type.value,
        "K": model.K,
        "T1": model.T1 if model.T1 > 0 else model.T,
        "T2": model.T2,
        "L": model.L,
    }
    combined_unreliable = bool(tuning_result.get("tuning_unreliable")) or review_unreliable
    combined_unreliable_reason = str(tuning_result.get("tuning_unreliable_reason", ""))
    if review_unreliable:
        combined_unreliable_reason = (
            f"{combined_unreliable_reason}；{review_unreliable_reason}"
            if combined_unreliable_reason else review_unreliable_reason
        )
    eval_ctx = _build_skill_context(
        csv_path=csv_path,
        loop_type=loop_type,
        dataset=dataset,
        candidate_windows=candidate_windows,
        selected_window_index=chosen_global_idx,
        data_profile=data_profile,
        model=eval_model_params,
        pid_params=pid,
        confidence=confidence.confidence,
    )
    if isinstance(selection_meta, dict) and selection_meta.get("window_policy"):
        eval_ctx.data_profile["window_policy"] = selection_meta.get("window_policy")
    scenario_skill = _invoke_guarded_skill(
        "build_simulation_scenarios",
        {"scenario_mode": "loop_aware", "include_reverse": True, "include_robustness": True},
        eval_ctx,
    )
    if scenario_skill.success:
        eval_ctx.data_profile["simulation_scenario"] = scenario_skill.data.get("simulation_scenario")
    skill_eval = _invoke_guarded_skill(
        "evaluate_tuning",
        {
            "provider": "closed_loop_sim",
            "tuning_unreliable": combined_unreliable,
            "tuning_unreliable_reason": combined_unreliable_reason,
        },
        eval_ctx,
    )
    if skill_eval.success and skill_eval.data:
        eval_result = skill_eval.data
    else:
        eval_result = evaluate_pid_params(
            Kp=pid["Kp"],
            Ki=pid["Ki"],
            Kd=pid["Kd"],
            model_type=model.model_type.value,
            model_params=eval_model_params,
            K=model.K,
            T=model.T,
            L=model.L,
            dt=dataset["dt"],
            loop_type=loop_type,
            confidence=confidence.confidence,
            tuning_unreliable=combined_unreliable,
            tuning_unreliable_reason=combined_unreliable_reason,
        )

    yield stage_event("evaluation", "done", {
        "passed": eval_result.get("passed", False),
        "performance_score": eval_result.get("performance_score", 0.0),
        "final_rating": eval_result.get("final_rating", 0.0),
        "overshoot_percent": eval_result.get("overshoot_percent", 0.0),
    })

    # ── Final Result ────────────────────────────────────────────────────
    yield result_event({
        "data_analysis": {
            "data_points": dataset["data_points"],
            "sampling_time": dataset["dt"],
            "step_events": dataset["step_events"],
            "candidate_windows": candidate_windows,
            "quality_metrics": dataset.get("quality_metrics"),
        },
        "window_selection": selection_meta,
        "model": {
            "model_type": model.model_type.value,
            "K": model.K,
            "T": model.T,
            "T1": model.T1,
            "T2": model.T2,
            "L": model.L,
            "r2_score": model.r2_score,
            "normalized_rmse": model.normalized_rmse,
            "confidence": confidence.confidence,
            "confidence_quality": confidence.quality,
            "window_source": id_result["window_source"],
            "selection_reason": id_result["selection_reason"],
            "fit_preview": id_result.get("fit_preview", {}),
            "candidates": id_result.get("candidates", []),
            "attempts": final_attempts_payload,
            "algorithm_comparison": _algorithm_comparison(final_attempts_payload),
        },
        "pid_params": {
            "Kp": pid["Kp"],
            "Ki": pid["Ki"],
            "Kd": pid["Kd"],
            "Ti": pid.get("Ti", 0.0),
            "Td": pid.get("Td", 0.0),
            "strategy": pid.get("strategy", ""),
            "candidates": tuning_result.get("all_candidates", []),
        },
        "evaluation": eval_result,
        "model_review": (
            {
                "verdict": final_review_result["verdict"],
                "reason": final_review_result["reason"],
                "concerns": final_review_result["concerns"],
            }
            if final_review_result is not None else None
        ),
        "loop_type": loop_type,
        "loop_name": loop_name,
    })
