"""扩展模型（SOPDT_UNDER / IFOPDT）的辨识能力测试。

策略：用各模型的离散仿真器从已知真值参数生成 PV，然后让 fit_best_model
仅在该模型类型上拟合，验证回归参数与真值在合理误差内一致。

不依赖真实数据文件；不依赖 LLM。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.algorithms.system_id import (
    _sim_ifopdt,
    _sim_sopdt_under,
    fit_best_model,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_step_mv(n: int, step_idx: int, base: float = 0.0, amp: float = 1.0) -> np.ndarray:
    mv = np.full(n, base, dtype=float)
    mv[step_idx:] = base + amp
    return mv


def _df_from(mv: np.ndarray, pv: np.ndarray, dt: float) -> pd.DataFrame:
    n = len(mv)
    base = pd.Timestamp("2025-01-01")
    ts = pd.to_datetime([base + pd.Timedelta(seconds=i * dt) for i in range(n)])
    return pd.DataFrame({"timestamp": ts, "PV": pv, "MV": mv, "SV": pv})


def _whole_window(df: pd.DataFrame) -> dict:
    return {
        "window_source": "test_full",
        "window_start_idx": 0,
        "window_end_idx": len(df),
        "window_usable_for_id": True,
        "window_drift_ratio": 0.0,
        "window_corr": 0.9,
    }


# ── SOPDT_UNDER ────────────────────────────────────────────────────────────

def test_fit_sopdt_under_recovers_oscillatory():
    """欠阻尼 SOPDT：K=2, T=10, ζ=0.3, L=2 → 拟合应在 ±30% 内回收。"""
    dt = 0.5
    n = 600
    mv = _make_step_mv(n, step_idx=50, base=0.0, amp=1.0)
    pv = _sim_sopdt_under(mv, K=2.0, T=10.0, zeta=0.3, L=2.0, dt=dt)
    # 加微量噪声让数值更现实，但不破坏拟合
    rng = np.random.default_rng(0)
    pv = pv + 0.005 * rng.standard_normal(n)

    df = _df_from(mv, pv, dt)
    res = fit_best_model(
        cleaned_df=df,
        candidate_windows=[_whole_window(df)],
        actual_dt=dt,
        loop_type="pressure",
        force_model_types=["SOPDT_UNDER"],
    )

    model = res["model"]
    assert model.success, f"拟合失败: {res.get('selection_reason')}"
    assert model.model_type.value == "SOPDT_UNDER"
    # K 真值 2.0；允许 ±30% 误差
    assert 1.4 < model.K < 2.6, f"K 偏离过大: {model.K}"
    # T 真值 10；允许 ±40%（欠阻尼振荡周期对参数较敏感）
    assert 6.0 < model.T < 14.0, f"T 偏离过大: {model.T}"
    # ζ 真值 0.3；允许较宽
    assert 0.1 < model.zeta < 0.6, f"zeta 偏离过大: {model.zeta}"
    # 拟合优度应当不错
    assert model.r2_score > 0.85, f"R² 过低: {model.r2_score}"


def test_fit_sopdt_under_to_tuning_params_maps_to_double_T():
    """SOPDT_UNDER → tuning 参数应把 T 映射为 2T 作为保守等效。"""
    from models.process_model import ModelType, ProcessModel

    m = ProcessModel(model_type=ModelType.SOPDT_UNDER, K=1.5, T=8.0, zeta=0.3, L=1.0)
    p = m.to_tuning_params(dt=0.5, n_points=600)
    # T = max(2*8, 0.5) = 16
    assert p["T"] == pytest.approx(16.0)
    assert p["K"] == 1.5
    assert p["L"] == 1.0


# ── IFOPDT ─────────────────────────────────────────────────────────────────

def test_fit_ifopdt_recovers_integrator_with_lag():
    """积分+一阶+死时：K=0.05, T=20, L=5 → 拟合应回收主参数。"""
    dt = 1.0
    n = 800
    mv = _make_step_mv(n, step_idx=100, base=0.0, amp=1.0)
    pv = _sim_ifopdt(mv, K=0.05, T=20.0, L=5.0, dt=dt)
    rng = np.random.default_rng(1)
    pv = pv + 0.01 * rng.standard_normal(n)

    df = _df_from(mv, pv, dt)
    res = fit_best_model(
        cleaned_df=df,
        candidate_windows=[_whole_window(df)],
        actual_dt=dt,
        loop_type="level",
        force_model_types=["IFOPDT"],
    )

    model = res["model"]
    assert model.success, f"拟合失败: {res.get('selection_reason')}"
    assert model.model_type.value == "IFOPDT"
    # K 真值 0.05；积分增益对噪声较敏感，给 ±50% 余量
    assert 0.025 < model.K < 0.10, f"K 偏离过大: {model.K}"
    # 拟合优度应当不错（积分对象 R² 通常很高）
    assert model.r2_score > 0.85, f"R² 过低: {model.r2_score}"


def test_fit_ifopdt_to_tuning_params_combines_L_and_T():
    """IFOPDT → tuning 时 L_eff = L + T（一阶滞后近似为额外死时）。"""
    from models.process_model import ModelType, ProcessModel

    m = ProcessModel(model_type=ModelType.IFOPDT, K=0.05, T=20.0, L=5.0)
    p = m.to_tuning_params(dt=1.0, n_points=800)
    assert p["K"] == pytest.approx(0.05)
    assert p["T"] == pytest.approx(20.0)
    # L_eff = L + T = 25
    assert p["L"] == pytest.approx(25.0)


# ── 整定路径 dispatch ──────────────────────────────────────────────────────

def test_tuning_dispatch_for_new_models():
    """两个新模型类型应能被 apply_tuning_rules 正确分发到底层规则。"""
    from core.algorithms.pid_tuning import apply_tuning_rules

    # SOPDT_UNDER → 应映射到 tune_sopdt（T1=T2=T）
    out1 = apply_tuning_rules(
        K=2.0, T=10.0, L=2.0,
        strategy="LAMBDA", model_type="SOPDT_UNDER",
        model_params={"K": 2.0, "T": 10.0, "L": 2.0, "zeta": 0.3},
    )
    assert "Kp" in out1 and "Ti" in out1
    assert out1["Kp"] > 0.0

    # IFOPDT → 应映射到 tune_ipdt，等效死时 = L + T = 25
    out2 = apply_tuning_rules(
        K=0.05, T=20.0, L=5.0,
        strategy="LAMBDA", model_type="IFOPDT",
        model_params={"K": 0.05, "T": 20.0, "L": 5.0},
    )
    assert "Kp" in out2 and "Ti" in out2
    assert out2["Kp"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
