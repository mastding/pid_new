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
from core.pipeline.llm_advisor import choose_window_via_llm
from core.skills import LoopContext, registry


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
    try:
        dataset = load_and_prepare_dataset(
            csv_path=csv_path,
            selected_loop_prefix=selected_loop_prefix,
            selected_window_index=selected_window_index,
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

    # ── Stage 1.5: Window Selection (LLM advisor or deterministic) ──────
    yield stage_event("window_selection", "running")

    # 候选池：优先 usable，没有就退而求其次用全部
    pool = usable_windows if usable_windows else candidate_windows
    pool_indices = [candidate_windows.index(w) for w in pool]

    # 确定性 baseline：池里 quality_score 最高的那个
    deterministic_pool_idx = max(
        range(len(pool)),
        key=lambda i: float(pool[i].get("window_quality_score", 0.0)),
    )
    deterministic_global_idx = pool_indices[deterministic_pool_idx]

    selection_meta: dict[str, Any] = {
        "deterministic_index": deterministic_global_idx,
        "deterministic_score": float(
            pool[deterministic_pool_idx].get("window_quality_score", 0.0)
        ),
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
        # 先算数据画像供 LLM 参考
        ctx = LoopContext(csv_path=csv_path)
        ctx.cleaned_df = dataset["cleaned_df"]
        ctx.dt = dataset["dt"]
        profile_result = registry.invoke("summarize_data", {}, ctx)
        data_profile = profile_result.data if profile_result.success else {}

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

    # 把选中的窗口提到列表首位（fit_best_model 仍会迭代，但首选项靠前以体现优先）
    chosen_window = candidate_windows[chosen_global_idx]
    # 单窗口模式：只让 fit_best_model 处理选中的那一个
    windows_for_fit = [chosen_window]
    selection_meta["chosen_window_summary"] = {
        "source": chosen_window.get("window_source"),
        "score": float(chosen_window.get("window_quality_score", 0.0)),
        "n_points": int(chosen_window.get("window_end_idx", 0))
        - int(chosen_window.get("window_start_idx", 0)),
    }

    yield stage_event("window_selection", "done", selection_meta)

    # ── Stage 2: System Identification ──────────────────────────────────
    yield stage_event("identification", "running")
    try:
        id_result = fit_best_model(
            cleaned_df=dataset["cleaned_df"],
            candidate_windows=windows_for_fit,
            actual_dt=dataset["dt"],
            loop_type=loop_type,
            quality_metrics=dataset.get("quality_metrics"),
        )
    except ValueError as exc:
        yield error_event(str(exc), stage="identification", error_code="ID_ERROR")
        return

    model = id_result["model"]
    confidence = id_result["confidence"]

    yield stage_event("identification", "done", {
        "model_type": model.model_type.value,
        "K": model.K,
        "T": model.T or (model.T1 + model.T2),
        "L": model.L,
        "r2_score": model.r2_score,
        "confidence": confidence.confidence,
        "window_source": id_result["window_source"],
    })

    if confidence.confidence < 0.35:
        yield error_event(
            f"模型置信度 {confidence.confidence:.2f} 过低，无法可靠整定",
            stage="identification",
            error_code="LOW_CONFIDENCE",
        )
        return

    # ── Stage 3: PID Tuning ─────────────────────────────────────────────
    yield stage_event("tuning", "running")
    tuning_params = model.to_tuning_params(dataset["dt"], dataset["data_points"])
    tuning_model_params = {
        "model_type": model.model_type.value,
        "K": model.K, "T": model.T,
        "T1": model.T1, "T2": model.T2, "L": model.L,
    }
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
            "attempts": id_result.get("attempts", []),
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
        "loop_type": loop_type,
        "loop_name": loop_name,
    })
