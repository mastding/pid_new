"""summarize_data 技能及其内部分析器的单元测试。

使用多组合成数据，分别触发饱和/死区/噪声/振荡等判据。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from core.skills import LoopContext, registry


def _write_csv(path: Path, timestamps_ms, sv, pv, mv) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sv", "pv", "mv"])
        for row in zip(timestamps_ms, sv, pv, mv):
            w.writerow(row)


def _load_into_ctx(csv_path: str) -> LoopContext:
    ctx = LoopContext(csv_path=csv_path)
    r = registry.invoke("load_dataset", {}, ctx)
    assert r.success, r.reasoning
    return ctx


# ──────────────────────────────────────────────────────────────────────
# 场景 1：正常一阶响应，低噪声，无饱和
# ──────────────────────────────────────────────────────────────────────

def test_summary_normal_response(tmp_path: Path):
    n = 600
    ts = [1_700_000_000_000 + i * 1000 for i in range(n)]
    sv = [50.0] * 300 + [60.0] * 300
    mv_arr = np.array([30.0] * 300 + [38.0] * 300, dtype=float)
    # 纯一阶响应，无噪声
    pv_arr = np.zeros(n)
    pv_arr[0] = 50.0
    tau = 30.0
    for i in range(1, n):
        pv_arr[i] = pv_arr[i - 1] + (sv[i] - pv_arr[i - 1]) * (1.0 - np.exp(-1.0 / tau))

    p = tmp_path / "normal.csv"
    _write_csv(p, ts, sv, pv_arr, mv_arr)

    ctx = _load_into_ctx(str(p))
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success, res.reasoning

    profile = res.data
    # 应无饱和、无死区、低噪声
    assert profile["mv_stats"]["saturation_high_pct"] < 60  # MV 只有两段稳态，允许
    assert profile["noise"]["noise_level"] in {"low", "medium"}
    assert profile["deadzone"]["evidence_ratio"] < 0.5
    # 含中文文字摘要
    assert any("\u4e00" <= c <= "\u9fff" for c in profile["text_summary"])
    # 上下文已填充
    assert ctx.data_profile is not None and ctx.data_profile["text_summary"]


# ──────────────────────────────────────────────────────────────────────
# 场景 2：MV 长时间触顶
# ──────────────────────────────────────────────────────────────────────

def test_summary_mv_saturation(tmp_path: Path):
    n = 500
    ts = [1_700_000_000_000 + i * 1000 for i in range(n)]
    sv = [80.0] * n
    # MV 被钉在 100（触顶）300 点
    mv = [50.0] * 100 + [100.0] * 300 + [90.0] * 100
    # PV 缓慢上升不完
    pv = np.linspace(50, 75, n)

    p = tmp_path / "saturated.csv"
    _write_csv(p, ts, sv, pv, mv)

    ctx = _load_into_ctx(str(p))
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success

    assert res.data["mv_stats"]["saturation_high_pct"] > 30
    # 应当产出警告
    assert any("触顶" in w for w in res.warnings)


# ──────────────────────────────────────────────────────────────────────
# 场景 3：死区明显 —— MV 频繁抖动但 PV 不响应
# ──────────────────────────────────────────────────────────────────────

def test_summary_deadzone(tmp_path: Path):
    n = 400
    ts = [1_700_000_000_000 + i * 1000 for i in range(n)]
    sv = [50.0] * n
    rng = np.random.default_rng(0)
    # MV 在 40~60 大幅抖动
    mv = 50.0 + 10.0 * rng.standard_normal(n)
    # PV 基本不动（只有小噪声）
    pv = 50.0 + 0.01 * rng.standard_normal(n)

    p = tmp_path / "deadzone.csv"
    _write_csv(p, ts, sv, pv, mv)

    ctx = _load_into_ctx(str(p))
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success

    dz = res.data["deadzone"]
    assert dz["evidence_ratio"] > 0.5, f"期待死区证据占比 >0.5，实得 {dz['evidence_ratio']}"


# ──────────────────────────────────────────────────────────────────────
# 场景 4：振荡 —— 注入正弦
# ──────────────────────────────────────────────────────────────────────

def test_summary_oscillation(tmp_path: Path):
    n = 800
    dt = 1.0
    ts = [1_700_000_000_000 + i * int(dt * 1000) for i in range(n)]
    sv = [50.0] * n
    mv = [40.0] * n
    # 周期 20s 的正弦叠加在 50 上
    t = np.arange(n, dtype=float) * dt
    pv = 50.0 + 5.0 * np.sin(2.0 * np.pi * t / 20.0)

    p = tmp_path / "oscillation.csv"
    _write_csv(p, ts, sv, pv, mv)

    ctx = _load_into_ctx(str(p))
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success

    osc = res.data["oscillation"]
    assert osc["detected"] is True
    # 允许 ±20% 误差
    assert 16 <= osc["period_sec"] <= 24, f"期望周期约 20s，实得 {osc['period_sec']}"


# ──────────────────────────────────────────────────────────────────────
# 场景 5：高噪声
# ──────────────────────────────────────────────────────────────────────

def test_summary_high_noise(tmp_path: Path):
    n = 500
    ts = [1_700_000_000_000 + i * 1000 for i in range(n)]
    sv = [50.0] * n
    mv = [40.0] * n
    rng = np.random.default_rng(1)
    # 信号趋势跨度 5，注入 σ=5 的噪声（去噪后仍应显著高于趋势 → high）
    pv = 45.0 + np.linspace(0, 5, n) + 5.0 * rng.standard_normal(n)

    p = tmp_path / "noisy.csv"
    _write_csv(p, ts, sv, pv, mv)

    ctx = _load_into_ctx(str(p))
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success

    assert res.data["noise"]["noise_level"] == "high"
    assert any("噪声" in w for w in res.warnings)


# ──────────────────────────────────────────────────────────────────────
# 场景 6：先决条件 —— 未 load_dataset 直接调 summarize_data
# ──────────────────────────────────────────────────────────────────────

def test_summary_requires_load(tmp_path: Path):
    ctx = LoopContext(csv_path="/fake.csv")
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success is False
    assert "load_dataset" in res.reasoning


# ──────────────────────────────────────────────────────────────────────
# 场景 7：注册 + schema + 中文描述
# ──────────────────────────────────────────────────────────────────────

def test_summarize_data_registered_and_schema():
    assert "summarize_data" in registry.names()
    tools = registry.to_openai_tools(["summarize_data"])
    desc = tools[0]["function"]["description"]
    assert any("\u4e00" <= c <= "\u9fff" for c in desc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
