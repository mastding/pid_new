"""整定流水线（Day 4：在窗口选择处接入 LLM 顾问）。

四个阶段：data_analysis → window_selection → identification → tuning → evaluation
LLM 仅在 window_selection 决策点参与；失败自动回退到确定性 fit_score 选窗。
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from core.algorithms.data_analysis import load_and_prepare_dataset
from core.algorithms.system_id import fit_best_model
from core.algorithms.pid_tuning import select_best_strategy
from core.algorithms.pid_evaluation import evaluate_pid_params
from core.pipeline.events import error_event, result_event, stage_event
from core.pipeline.identification_advisor import review_identification_via_llm
from core.pipeline.identification_refinement_advisor import ask_refinement_via_llm
from core.pipeline.llm_advisor import choose_window_via_llm
from core.pipeline.refinement_policy import recommend_refinement_from_algorithm_comparison
from core.skills import LoopContext, registry
from models.process_model import ModelConfidence, ModelType, ProcessModel


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
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the full deterministic tuning pipeline with SSE events.

    Yields stage events for frontend progress, then a final result event.
    No LLM calls - purely deterministic computation.
    """

    # ── Stage 1: Data Analysis ──────────────────────────────────────────
    yield stage_event("data_analysis", "running")
    dataset: dict[str, Any] | None = None
    load_ctx = LoopContext(
        csv_path=csv_path,
        loop_prefix=selected_loop_prefix or "",
        loop_type=loop_type,
    )
    load_result = registry.invoke(
        "load_dataset",
        {"loop_prefix": selected_loop_prefix},
        load_ctx,
    )
    if load_result.success and load_ctx.cleaned_df is not None and load_ctx.dt is not None:
        detect_result = registry.invoke("detect_windows", {}, load_ctx)
        if detect_result.success:
            candidate_windows = list(load_ctx.candidate_windows)
            if selected_window_index is not None and 0 <= selected_window_index < len(candidate_windows):
                candidate_windows.insert(0, candidate_windows.pop(selected_window_index))
            dataset = {
                "cleaned_df": load_ctx.cleaned_df,
                "dt": load_ctx.dt,
                "step_events": [None] * int(detect_result.data.get("step_event_count", 0)),
                "candidate_windows": candidate_windows,
                "data_points": int(load_result.data.get("data_points", len(load_ctx.cleaned_df))),
                "quality_metrics": None,
            }

    if dataset is None:
        try:
            dataset = load_and_prepare_dataset(
                csv_path=csv_path,
                selected_loop_prefix=selected_loop_prefix,
                selected_window_index=selected_window_index,
                loop_type=loop_type,
            )
        except ValueError as exc:
            yield error_event(str(exc), stage="data_analysis", error_code="DATA_ERROR")
            return

    candidate_windows = dataset["candidate_windows"]
    usable_windows = [w for w in candidate_windows if w.get("window_usable_for_id")]

    yield stage_event("data_analysis", "done", {
        "data_points": dataset["data_points"],
        "sampling_time": dataset["dt"],
        "step_events": len(dataset["step_events"]),
        "candidate_windows": len(candidate_windows),
        "usable_windows": len(usable_windows),
    })

    if not usable_windows and not candidate_windows:
        yield error_event(
            "未发现可用于系统辨识的数据窗口",
            stage="data_analysis",
            error_code="NO_USABLE_WINDOWS",
        )
        return

    # 数据画像：window_selection 与 identification_review 都要用，提前算一次
    data_profile: dict[str, Any] = {}
    if use_llm_advisor:
        ctx_profile = LoopContext(csv_path=csv_path, loop_type=loop_type)
        ctx_profile.cleaned_df = dataset["cleaned_df"]
        ctx_profile.dt = dataset["dt"]
        profile_result = registry.invoke("summarize_data", {}, ctx_profile)
        if profile_result.success:
            data_profile = profile_result.data

    # ── Stage 1.5: Window Selection (LLM advisor or deterministic) ──────
    yield stage_event("window_selection", "running")

    # 候选池：优先 usable，没有就退而求其次用全部
    pool = usable_windows if usable_windows else candidate_windows
    pool_indices = [candidate_windows.index(w) for w in pool]

    # 确定性 baseline：池里 quality_score 最高的那个
    deterministic_ctx = _build_skill_context(
        csv_path=csv_path,
        loop_type=loop_type,
        dataset=dataset,
        candidate_windows=pool,
        selected_window_index=selected_window_index,
    )
    deterministic_result = registry.invoke(
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

    selection_meta: dict[str, Any] = {
        "deterministic_index": deterministic_global_idx,
        "deterministic_score": deterministic_score,
    }

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
        )

        if advisor is not None:
            chosen_pool_idx = advisor["chosen_index"]
            chosen_global_idx = pool_indices[chosen_pool_idx]
            # 把 R1 的思维链作为独立事件发出，前端可以折叠展示
            reasoning_chain = advisor.get("reasoning_content", "") or ""
            if reasoning_chain:
                yield {
                    "type": "llm_thinking",
                    "stage": "window_selection",
                    "model": "deepseek-reasoner",
                    "reasoning_content": reasoning_chain,
                    "raw_text": advisor.get("raw_text", ""),
                }
            selection_meta.update({
                "mode": "llm",
                "chosen_index": chosen_global_idx,
                "reasoning": advisor["reasoning"],
                "llm_reasoning_chain_len": len(reasoning_chain),
                "agreed_with_deterministic": chosen_global_idx == deterministic_global_idx,
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

    yield stage_event("window_selection", "done", selection_meta)

    # ── Stage 2 / 2.5: Identification + Review with refinement loop ─────
    # Phase 2: 多轮"辨识 → 评审 → 精修指令 → 辨识"循环，最多 MAX_REFINEMENT_ROUNDS 轮重试
    # （即 1 次初始辨识 + 至多 2 次精修 = 最多 3 次辨识调用）。
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
        skill_id = registry.invoke(
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
                    "zeta": float(a.get("zeta", 0.0)) if a.get("zeta") is not None else 0.0,
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

        # 置信度极低 → 这一轮没救，但不再 abort 流程；让 review 给降级判断 + 进 Phase 3 兜底
        # （Phase 3 改之前只在 round 0 阻断）
        if confidence.confidence < 0.35 and round_idx == 0 and not use_llm_advisor:
            yield error_event(
                f"模型置信度 {confidence.confidence:.2f} 过低，无法可靠整定",
                stage="identification",
                error_code="LOW_CONFIDENCE",
            )
            return

        # ── 本轮评审 ──
        review_result: dict[str, Any] | None = None
        if use_llm_advisor:
            yield stage_event("model_review", "running", {"round": round_idx})
            best_for_review = {
                "model_type": model.model_type.value,
                "K": model.K, "T": model.T, "T1": model.T1, "T2": model.T2, "L": model.L,
                "r2_score": model.r2_score,
                "normalized_rmse": model.normalized_rmse,
                "window_source": id_result["window_source"],
            }
            review_result = await asyncio.to_thread(
                review_identification_via_llm,
                loop_type=loop_type,
                data_profile=data_profile,
                chosen_window_summary=selection_meta.get("chosen_window_summary", {}),
                best_model=best_for_review,
                attempts=id_result.get("attempts", []),
                confidence=confidence.confidence,
            )
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
        refinement = await asyncio.to_thread(
            ask_refinement_via_llm,
            loop_type=loop_type,
            round_idx=round_idx + 1,
            max_rounds=MAX_REFINEMENT_ROUNDS,
            data_profile=data_profile,
            windows_summary=windows_summary,
            algorithm_comparison=algorithm_comparison,
            last_best=best_for_review,
            last_attempts=id_result.get("attempts", []),
            last_review=review_result,
            history_summary=[
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
        )

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
    skill_tuning = registry.invoke(
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
    skill_eval = registry.invoke(
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
