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
    """构造真实死区场景：MV 出现多次清晰阶跃，但 PV 始终不响应。

    新算法（lag-aware）要求"真 MV 阶跃"才计入分母，所以测试也要给真阶跃，
    而不是纯噪声 —— 否则就是"无指令可判"的中性场景。
    """
    n = 400
    ts = [1_700_000_000_000 + i * 1000 for i in range(n)]
    sv = [50.0] * n
    rng = np.random.default_rng(0)
    # MV 每 40 点跳一次，幅度 ±5（远超 1% 量程的 0.1，远超抖动 0.01 的 8 倍）
    mv = np.zeros(n)
    base = 50.0
    for i in range(n):
        if (i // 40) % 2 == 0:
            mv[i] = base + 5.0
        else:
            mv[i] = base - 5.0
    mv = mv + 0.01 * rng.standard_normal(n)  # 极小抖动
    # PV 完全不动（恒定 50），加微量随机扰动避免 load_dataset 量化产生意外
    pv = np.full(n, 50.0)

    p = tmp_path / "deadzone.csv"
    _write_csv(p, ts, sv, pv, mv)

    ctx = _load_into_ctx(str(p))
    # 默认 loop_type=flow → lag=10s，刚好在 40 点的 MV 阶跃区段内能扫到
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success

    dz = res.data["deadzone"]
    assert dz["events_total"] > 0, f"期待识别到 MV 阶跃事件，实得 events_total={dz['events_total']}"
    assert dz["evidence_ratio"] > 0.5, (
        f"期待死区证据占比 >0.5，实得 {dz['evidence_ratio']} "
        f"(events_total={dz['events_total']}, evidence_count={dz['evidence_count']})"
    )


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
    """非低噪声场景：高斯噪声叠加在小幅趋势上。

    新分级阈值（low <0.5% / medium <1.5% / high ≥1.5%）相比旧版收紧的反向：放宽。
    旧版 σ=5 → rel=0.69% → high；新版同口径数据 → medium。
    "high" 现在保留给量化栅格异常 / 严重失真等极端场景，所以这里只验证"非低"。
    """
    n = 500
    ts = [1_700_000_000_000 + i * 1000 for i in range(n)]
    sv = [50.0] * n
    mv = [40.0] * n
    rng = np.random.default_rng(1)
    pv = 45.0 + np.linspace(0, 5, n) + 5.0 * rng.standard_normal(n)

    p = tmp_path / "noisy.csv"
    _write_csv(p, ts, sv, pv, mv)

    ctx = _load_into_ctx(str(p))
    res = registry.invoke("summarize_data", {}, ctx)
    assert res.success

    # 新口径：σ=5 噪声经 kernel-9 中值滤波后落在 medium 区间
    assert res.data["noise"]["noise_level"] in {"medium", "high"}, (
        f"期待非低噪声，实得 {res.data['noise']}"
    )


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
