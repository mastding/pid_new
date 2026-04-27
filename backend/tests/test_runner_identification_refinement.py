from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pipeline import runner as runner_mod
from core.pipeline.runner import run_tuning_pipeline
from models.process_model import ModelConfidence, ModelType, ProcessModel


class _SkillResult:
    def __init__(self, success: bool, data: dict):
        self.success = success
        self.data = data


def _make_registry_invoke_stub(original_invoke):
    def _invoke(name, args, ctx):
        if name == "summarize_data":
            return _SkillResult(True, {"text_summary": "profile"})
        if name in {"identify_model", "generate_tuning_candidates", "evaluate_tuning"}:
            return _SkillResult(False, {})
        return original_invoke(name, args, ctx)

    return _invoke


def _collect_events(gen):
    return asyncio.run(_collect_async(gen))


async def _collect_async(gen):
    return [ev async for ev in gen]


def _dataset():
    return {
        "data_points": 200,
        "dt": 1.0,
        "step_events": [1, 2],
        "cleaned_df": object(),
        "quality_metrics": {"ok": True},
        "candidate_windows": [
            {
                "window_source": "w0",
                "window_start_idx": 0,
                "window_end_idx": 80,
                "window_quality_score": 0.95,
                "window_corr": 0.8,
                "window_usable_for_id": True,
                "window_algorithm": "mv_step",
                "window_algorithm_label": "mv_step",
            },
            {
                "window_source": "w1",
                "window_start_idx": 80,
                "window_end_idx": 180,
                "window_quality_score": 0.85,
                "window_corr": 0.7,
                "window_usable_for_id": True,
                "window_algorithm": "steady_disturbance",
                "window_algorithm_label": "steady_disturbance_scan",
            },
        ],
    }


def _id_result(model_type: ModelType, source: str, fit_score: float, conf: float):
    model = ProcessModel(
        model_type=model_type,
        K=1.2,
        T=10.0,
        T1=0.0,
        T2=0.0,
        L=2.0,
        r2_score=0.72,
        normalized_rmse=0.15,
        success=True,
    )
    confidence = ModelConfidence(
        confidence=conf,
        quality="fair",
        recommendation="ok",
        r2_score=model.r2_score,
        rmse_score=0.8,
    )
    attempts = [
        {
            "model_type": model_type.value,
            "window_source": source,
            "K": model.K,
            "T": model.T,
            "L": model.L,
            "r2_score": model.r2_score,
            "normalized_rmse": model.normalized_rmse,
            "fit_score": fit_score,
            "confidence": conf,
            "success": True,
        }
    ]
    return {
        "model": model,
        "confidence": confidence,
        "window_source": source,
        "selection_reason": "best fit",
        "fit_preview": {},
        "candidates": [],
        "attempts": attempts,
    }


def test_runner_retries_identification_with_refinement(monkeypatch):
    fit_calls: list[dict] = []
    refinement_calls: list[dict] = []

    monkeypatch.setattr(runner_mod, "load_and_prepare_dataset", lambda **kwargs: _dataset())
    monkeypatch.setattr(runner_mod.registry, "invoke", _make_registry_invoke_stub(runner_mod.registry.invoke))
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **kwargs: None)

    id_results = iter(
        [
            _id_result(ModelType.FOPDT, "w0", fit_score=0.55, conf=0.52),
            _id_result(ModelType.SOPDT, "w1", fit_score=0.88, conf=0.81),
        ]
    )

    def fake_fit_best_model(**kwargs):
        fit_calls.append(
            {
                "candidate_sources": [w["window_source"] for w in kwargs["candidate_windows"]],
                "force_model_types": kwargs.get("force_model_types"),
                "force_L_hint": kwargs.get("force_L_hint"),
            }
        )
        return next(id_results)

    review_results = iter(
        [
            {"verdict": "downgrade", "reason": "一轮不够可靠", "concerns": ["换窗"]},
            {"verdict": "accept", "reason": "第二轮可接受", "concerns": []},
        ]
    )

    monkeypatch.setattr(runner_mod, "fit_best_model", fake_fit_best_model)
    monkeypatch.setattr(runner_mod, "review_identification_via_llm", lambda **kwargs: next(review_results))
    monkeypatch.setattr(
        runner_mod,
        "ask_refinement_via_llm",
        lambda **kwargs: (refinement_calls.append(kwargs) or {
            "retry": True,
            "rationale": "切换到第二个窗口并限制模型池",
            "force_window_index": 1,
            "force_model_types": ["SOPDT"],
            "hint_L": 4.0,
            "reasoning_content": "refine",
            "raw_text": "{}",
        }),
    )
    monkeypatch.setattr(
        runner_mod,
        "select_best_strategy",
        lambda **kwargs: {"best": {"Kp": 1.0, "Ki": 0.1, "Kd": 0.01, "strategy": "IMC"}, "all_candidates": []},
    )
    monkeypatch.setattr(
        runner_mod,
        "evaluate_pid_params",
        lambda **kwargs: {"passed": True, "performance_score": 90.0, "final_rating": 8.8, "overshoot_percent": 5.0},
    )

    events = _collect_events(run_tuning_pipeline(csv_path="dummy.csv", loop_type="flow", use_llm_advisor=True))

    assert len(fit_calls) == 2
    assert refinement_calls
    assert refinement_calls[0]["algorithm_comparison"]
    assert fit_calls[0]["candidate_sources"] == ["w0", "w1"]
    assert fit_calls[0]["force_model_types"] is None
    assert fit_calls[1]["candidate_sources"] == ["w1"]
    assert fit_calls[1]["force_model_types"] == ["SOPDT"]
    assert fit_calls[1]["force_L_hint"] == 4.0

    refinement_done = [
        ev for ev in events
        if ev.get("type") == "stage" and ev.get("stage") == "identification_refinement" and ev.get("status") == "done"
    ]
    assert refinement_done
    assert refinement_done[0]["data"]["retry"] is True
    assert refinement_done[0]["data"]["force_window_index"] == 1
    assert refinement_done[0]["data"]["force_model_types"] == ["SOPDT"]

    result_events = [ev for ev in events if ev.get("type") == "result"]
    assert result_events
    result = result_events[-1]["data"]
    assert result["model"]["model_type"] == "SOPDT"
    assert result["model_review"]["verdict"] == "accept"
    assert result["model"]["attempts"][0]["window_algorithm"] in {"mv_step", "steady_disturbance"}

    id_done = [
        ev for ev in events
        if ev.get("type") == "stage" and ev.get("stage") == "identification" and ev.get("status") == "done"
    ]
    assert id_done
    assert id_done[-1]["data"]["algorithm_comparison"]


def test_runner_uses_deterministic_refinement_when_llm_refinement_unavailable(monkeypatch):
    fit_calls: list[dict] = []

    monkeypatch.setattr(runner_mod, "load_and_prepare_dataset", lambda **kwargs: _dataset())
    monkeypatch.setattr(runner_mod.registry, "invoke", _make_registry_invoke_stub(runner_mod.registry.invoke))
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **kwargs: None)

    first = _id_result(ModelType.FOPDT, "w0", fit_score=20.0, conf=0.45)
    first["attempts"] = [
        {
            "model_type": "FOPDT",
            "window_source": "w0",
            "K": 1.2,
            "T": 10.0,
            "L": 2.0,
            "r2_score": 0.50,
            "normalized_rmse": 0.20,
            "fit_score": 20.0,
            "confidence": 0.45,
            "success": True,
        },
        {
            "model_type": "FO",
            "window_source": "w1",
            "K": 1.0,
            "T": 8.0,
            "L": 0.0,
            "r2_score": 0.62,
            "normalized_rmse": 0.15,
            "fit_score": 18.0,
            "confidence": 0.60,
            "success": True,
        },
    ]
    second = _id_result(ModelType.FO, "w1", fit_score=18.0, conf=0.60)
    id_results = iter([first, second])

    def fake_fit_best_model(**kwargs):
        fit_calls.append(
            {
                "candidate_sources": [w["window_source"] for w in kwargs["candidate_windows"]],
                "force_model_types": kwargs.get("force_model_types"),
            }
        )
        return next(id_results)

    review_results = iter(
        [
            {"verdict": "downgrade", "reason": "首选模型不稳", "concerns": ["R2偏低"]},
            {"verdict": "accept", "reason": "备选窗口可接受", "concerns": []},
        ]
    )
    monkeypatch.setattr(runner_mod, "fit_best_model", fake_fit_best_model)
    monkeypatch.setattr(runner_mod, "review_identification_via_llm", lambda **kwargs: next(review_results))
    monkeypatch.setattr(runner_mod, "ask_refinement_via_llm", lambda **kwargs: None)
    monkeypatch.setattr(
        runner_mod,
        "select_best_strategy",
        lambda **kwargs: {"best": {"Kp": 1.0, "Ki": 0.1, "Kd": 0.01, "strategy": "IMC"}, "all_candidates": []},
    )
    monkeypatch.setattr(
        runner_mod,
        "evaluate_pid_params",
        lambda **kwargs: {"passed": True, "performance_score": 80.0, "final_rating": 7.5, "overshoot_percent": 5.0},
    )

    events = _collect_events(run_tuning_pipeline(csv_path="dummy.csv", loop_type="flow", use_llm_advisor=True))

    assert len(fit_calls) == 2
    assert fit_calls[1]["candidate_sources"] == ["w1"]
    assert fit_calls[1]["force_model_types"][0] == "FO"
    refinements = [
        ev for ev in events
        if ev.get("type") == "stage" and ev.get("stage") == "identification_refinement" and ev.get("status") == "done"
    ]
    assert refinements
    assert refinements[0]["data"]["source"] == "deterministic_algorithm_policy"
    assert refinements[0]["data"]["recommended_algorithm"] == "steady_disturbance"


def test_runner_marks_unreliable_when_refinement_stops(monkeypatch):
    eval_calls: list[dict] = []

    monkeypatch.setattr(runner_mod, "load_and_prepare_dataset", lambda **kwargs: _dataset())
    monkeypatch.setattr(runner_mod.registry, "invoke", _make_registry_invoke_stub(runner_mod.registry.invoke))
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **kwargs: None)
    monkeypatch.setattr(
        runner_mod,
        "fit_best_model",
        lambda **kwargs: _id_result(ModelType.FOPDT, "w0", fit_score=0.55, conf=0.52),
    )
    monkeypatch.setattr(
        runner_mod,
        "review_identification_via_llm",
        lambda **kwargs: {"verdict": "downgrade", "reason": "模型不够可靠", "concerns": ["R2偏低"]},
    )
    monkeypatch.setattr(runner_mod, "ask_refinement_via_llm", lambda **kwargs: {"retry": False, "rationale": "没有更好方案"})
    monkeypatch.setattr(
        runner_mod,
        "select_best_strategy",
        lambda **kwargs: {"best": {"Kp": 2.0, "Ki": 0.2, "Kd": 0.0, "strategy": "LAMBDA"}, "all_candidates": []},
    )

    def fake_evaluate(**kwargs):
        eval_calls.append(kwargs)
        return {"passed": False, "performance_score": 60.0, "final_rating": 5.5, "overshoot_percent": 12.0}

    monkeypatch.setattr(runner_mod, "evaluate_pid_params", fake_evaluate)

    events = _collect_events(run_tuning_pipeline(csv_path="dummy.csv", loop_type="flow", use_llm_advisor=True))

    assert eval_calls
    assert eval_calls[0]["tuning_unreliable"] is True
    assert "模型不够可靠" in eval_calls[0]["tuning_unreliable_reason"]

    result_events = [ev for ev in events if ev.get("type") == "result"]
    assert result_events
    result = result_events[-1]["data"]
    assert result["model_review"]["verdict"] == "downgrade"


def test_runner_uses_best_round_when_all_reviews_downgrade(monkeypatch):
    eval_calls: list[dict] = []

    monkeypatch.setattr(runner_mod, "load_and_prepare_dataset", lambda **kwargs: _dataset())
    monkeypatch.setattr(runner_mod.registry, "invoke", _make_registry_invoke_stub(runner_mod.registry.invoke))
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **kwargs: None)

    id_results = iter(
        [
            _id_result(ModelType.FOPDT, "w0", fit_score=0.91, conf=0.70),
            _id_result(ModelType.SOPDT, "w1", fit_score=0.62, conf=0.65),
            _id_result(ModelType.FO, "w1", fit_score=0.40, conf=0.50),
        ]
    )
    monkeypatch.setattr(runner_mod, "fit_best_model", lambda **kwargs: next(id_results))
    monkeypatch.setattr(
        runner_mod,
        "review_identification_via_llm",
        lambda **kwargs: {"verdict": "downgrade", "reason": "始终不够稳", "concerns": ["保守处理"]},
    )

    refinements = iter(
        [
            {"retry": True, "rationale": "换窗", "force_window_index": 1, "force_model_types": ["SOPDT"], "hint_L": 2.0},
            {"retry": True, "rationale": "再缩模型", "force_window_index": 1, "force_model_types": ["FO"], "hint_L": 1.0},
        ]
    )
    monkeypatch.setattr(runner_mod, "ask_refinement_via_llm", lambda **kwargs: next(refinements))
    monkeypatch.setattr(
        runner_mod,
        "select_best_strategy",
        lambda **kwargs: {"best": {"Kp": 1.5, "Ki": 0.15, "Kd": 0.0, "strategy": "IMC"}, "all_candidates": []},
    )

    def fake_evaluate(**kwargs):
        eval_calls.append(kwargs)
        return {"passed": False, "performance_score": 58.0, "final_rating": 5.2, "overshoot_percent": 11.0}

    monkeypatch.setattr(runner_mod, "evaluate_pid_params", fake_evaluate)

    events = _collect_events(run_tuning_pipeline(csv_path="dummy.csv", loop_type="flow", use_llm_advisor=True))

    assert eval_calls
    assert eval_calls[0]["model_type"] == "FOPDT"
    assert "始终不够稳" in eval_calls[0]["tuning_unreliable_reason"]

    result_events = [ev for ev in events if ev.get("type") == "result"]
    assert result_events
    result = result_events[-1]["data"]
    assert result["model"]["model_type"] == "FOPDT"
    assert result["model"]["window_source"] == "w0"


def test_runner_combines_tuning_and_review_unreliable_reasons(monkeypatch):
    eval_calls: list[dict] = []

    monkeypatch.setattr(runner_mod, "load_and_prepare_dataset", lambda **kwargs: _dataset())
    monkeypatch.setattr(runner_mod.registry, "invoke", _make_registry_invoke_stub(runner_mod.registry.invoke))
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **kwargs: None)
    monkeypatch.setattr(
        runner_mod,
        "fit_best_model",
        lambda **kwargs: _id_result(ModelType.FOPDT, "w0", fit_score=0.55, conf=0.52),
    )
    monkeypatch.setattr(
        runner_mod,
        "review_identification_via_llm",
        lambda **kwargs: {"verdict": "downgrade", "reason": "辨识存疑", "concerns": ["R2偏低"]},
    )
    monkeypatch.setattr(runner_mod, "ask_refinement_via_llm", lambda **kwargs: {"retry": False, "rationale": "放弃重试"})
    monkeypatch.setattr(
        runner_mod,
        "select_best_strategy",
        lambda **kwargs: {
            "best": {"Kp": 2.0, "Ki": 0.2, "Kd": 0.0, "strategy": "LAMBDA"},
            "all_candidates": [],
            "tuning_unreliable": True,
            "tuning_unreliable_reason": "整定参数偏激进",
        },
    )

    def fake_evaluate(**kwargs):
        eval_calls.append(kwargs)
        return {"passed": False, "performance_score": 52.0, "final_rating": 4.8, "overshoot_percent": 15.0}

    monkeypatch.setattr(runner_mod, "evaluate_pid_params", fake_evaluate)

    _collect_events(run_tuning_pipeline(csv_path="dummy.csv", loop_type="flow", use_llm_advisor=True))

    assert eval_calls
    assert eval_calls[0]["tuning_unreliable"] is True
    assert "整定参数偏激进" in eval_calls[0]["tuning_unreliable_reason"]
    assert "辨识存疑" in eval_calls[0]["tuning_unreliable_reason"]


def test_runner_records_model_review_failure_details(monkeypatch):
    monkeypatch.setattr(runner_mod, "load_and_prepare_dataset", lambda **kwargs: _dataset())
    monkeypatch.setattr(runner_mod.registry, "invoke", _make_registry_invoke_stub(runner_mod.registry.invoke))
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **kwargs: None)
    monkeypatch.setattr(
        runner_mod,
        "fit_best_model",
        lambda **kwargs: _id_result(ModelType.FOPDT, "w0", fit_score=0.55, conf=0.52),
    )
    monkeypatch.setattr(
        runner_mod,
        "review_identification_via_llm",
        lambda **kwargs: {
            "available": False,
            "error_type": "invalid_json",
            "error_message": "模型评审返回不是合法 JSON",
            "raw_text": "oops",
        },
    )
    monkeypatch.setattr(
        runner_mod,
        "select_best_strategy",
        lambda **kwargs: {"best": {"Kp": 1.0, "Ki": 0.1, "Kd": 0.0, "strategy": "IMC"}, "all_candidates": []},
    )
    monkeypatch.setattr(
        runner_mod,
        "evaluate_pid_params",
        lambda **kwargs: {"passed": True, "performance_score": 80.0, "final_rating": 7.0, "overshoot_percent": 5.0},
    )

    events = _collect_events(run_tuning_pipeline(csv_path="dummy.csv", loop_type="flow", use_llm_advisor=True))

    review_done = [
        ev for ev in events
        if ev.get("type") == "stage" and ev.get("stage") == "model_review" and ev.get("status") == "done"
    ]
    assert review_done
    review_data = review_done[-1]["data"]
    assert review_data["fallback"] is True
    assert review_data["error_type"] == "invalid_json"
    assert review_data["error_message"] == "模型评审返回不是合法 JSON"
    assert review_data["raw_text"] == "oops"
