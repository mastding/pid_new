from __future__ import annotations

import csv
from pathlib import Path

import pytest

from core.shared import provider_registry
from core.skills import LoopContext, registry


def _make_csv(path: Path, n: int = 600) -> None:
    base_ts = 1_700_000_000_000
    rows = []
    sv = 50.0
    pv = 50.0
    mv = 30.0
    for i in range(n):
        ts = base_ts + i * 1000
        if i == n // 2:
            sv = 60.0
        mv = 30.0 + (sv - 50.0) * 0.8
        pv += (mv * 0.05 - pv * 0.05) * 0.1 + (sv - pv) * 0.02
        rows.append((ts, sv, pv, mv))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sv", "pv", "mv"])
        w.writerows(rows)


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> str:
    p = tmp_path / "synthetic.csv"
    _make_csv(p)
    return str(p)


def test_provider_registry_has_first_batch_providers():
    assert "clean_csv_loader" in provider_registry.names("dataset_loading")
    assert "deterministic_profile" in provider_registry.names("data_profile")
    assert "history_rule_based" in provider_registry.names("window_detection")
    assert "quality_score_selector" in provider_registry.names("window_selection")
    assert "cross_correlation" in provider_registry.names("dead_time")
    assert "transfer_function_fit" in provider_registry.names("identification")
    assert {"imc", "lambda", "zn", "chr"} <= set(provider_registry.names("tuning_strategy"))
    assert "classic_family" in provider_registry.names("tuning")
    assert "closed_loop_response" in provider_registry.names("evaluation_simulation")
    assert "step_response_scoring" in provider_registry.names("evaluation_scoring")
    assert "adaptive_typical_t" in provider_registry.names("evaluation_reality_check")
    assert "closed_loop_sim" in provider_registry.names("evaluation")


def test_registry_has_first_batch_skills():
    names = set(registry.names())
    assert "detect_windows" in names
    assert "select_window" in names
    assert "estimate_dead_time" in names
    assert "identify_model" in names
    assert "generate_tuning_candidates" in names
    assert "evaluate_tuning" in names


def test_first_batch_skill_chain_smoke(synthetic_csv: str):
    ctx = LoopContext(csv_path=synthetic_csv, loop_type="flow")

    load_res = registry.invoke("load_dataset", {}, ctx)
    assert load_res.success, load_res.reasoning
    assert load_res.data["provider"] == "clean_csv_loader"

    profile_res = registry.invoke("summarize_data", {}, ctx)
    assert profile_res.success, profile_res.reasoning
    assert "text_summary" in profile_res.data

    win_res = registry.invoke("detect_windows", {}, ctx)
    assert win_res.success, win_res.reasoning
    assert win_res.data["provider"] == "history_rule_based"
    assert win_res.data["candidate_count"] >= 1

    select_res = registry.invoke("select_window", {}, ctx)
    assert select_res.success, select_res.reasoning
    assert select_res.data["provider"] == "quality_score_selector"
    assert isinstance(select_res.data["chosen_index"], int)
    assert ctx.selected_window_index == select_res.data["chosen_index"]

    l_res = registry.invoke("estimate_dead_time", {}, ctx)
    assert l_res.success, l_res.reasoning
    assert l_res.data["provider"] == "cross_correlation"

    id_res = registry.invoke("identify_model", {}, ctx)
    assert id_res.success, id_res.reasoning
    assert id_res.data["provider"] == "transfer_function_fit"
    assert id_res.data["best_model"]

    tune_res = registry.invoke("generate_tuning_candidates", {}, ctx)
    assert tune_res.success, tune_res.reasoning
    assert tune_res.data["provider"] == "classic_family"
    assert tune_res.data["recommended"]

    eval_res = registry.invoke("evaluate_tuning", {}, ctx)
    assert eval_res.success, eval_res.reasoning
    assert eval_res.data["provider"] == "closed_loop_sim"
    assert "performance_score" in eval_res.data


def test_method_level_tuning_providers_feed_classic_family():
    provider = provider_registry.get("tuning", "classic_family")
    assert provider is not None

    result = provider.tune(
        K=1.2,
        T=12.0,
        L=2.0,
        dt=1.0,
        loop_type="flow",
        model_type="FOPDT",
        model_params={"K": 1.2, "T": 12.0, "L": 2.0},
        confidence=0.9,
        nrmse=0.03,
        r2=0.98,
        context={},
    )

    assert result["provider"] == "classic_family"
    assert result["best"] is not None
    strategies = {c["strategy"] for c in result["all_candidates"]}
    assert {"IMC", "LAMBDA", "ZN", "CHR"} <= strategies


def test_method_level_evaluation_providers_feed_closed_loop_sim():
    provider = provider_registry.get("evaluation", "closed_loop_sim")
    assert provider is not None

    result = provider.evaluate(
        Kp=1.0,
        Ki=0.05,
        Kd=0.0,
        model_type="FOPDT",
        model_params={"model_type": "FOPDT", "K": 1.0, "T": 12.0, "L": 2.0},
        K=1.0,
        T=12.0,
        L=2.0,
        dt=1.0,
        loop_type="flow",
        confidence=0.85,
        tuning_unreliable=False,
        tuning_unreliable_reason="",
        context={},
    )

    assert result["provider"] == "closed_loop_sim"
    assert "performance_score" in result
    assert "reality_check_score" in result
    assert "simulation" in result
