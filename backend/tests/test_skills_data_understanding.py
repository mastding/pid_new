"""data_understanding 子包技能的单元测试。

使用临时合成 CSV，不依赖外部数据文件。
"""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import pytest

from core.skills import LoopContext, registry


def _make_csv(path: Path, n: int = 600, with_step: bool = True) -> None:
    """生成一份带 SV 阶跃的合成 CSV，确保有可识别窗口。"""
    base_ts = 1_700_000_000_000  # 毫秒
    rows = []
    sv = 50.0
    pv = 50.0
    mv = 30.0
    for i in range(n):
        ts = base_ts + i * 1000  # 1s 采样
        if with_step and i == n // 2:
            sv = 60.0  # SV 阶跃
        # 简单一阶响应近似
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


def test_load_dataset_skill_registered():
    assert "load_dataset" in registry.names()
    assert "detect_candidate_windows" in registry.names()


def test_load_dataset_basic(synthetic_csv: str):
    """加载合成 CSV 应返回基本元信息，并把 cleaned_df 写入 ctx。"""
    ctx = LoopContext(csv_path=synthetic_csv)
    result = registry.invoke("load_dataset", {}, ctx)

    assert result.success, f"加载失败: {result.reasoning}"
    assert result.data["data_points"] > 500
    assert math.isclose(result.data["sampling_time"], 1.0, abs_tol=0.01)
    assert "PV" in result.data["columns"]
    assert "MV" in result.data["columns"]
    # 上下文已填充
    assert ctx.cleaned_df is not None
    assert ctx.dt is not None
    assert len(ctx.cleaned_df) > 500


def test_load_dataset_missing_file():
    """文件不存在时应返回 success=False 而不抛异常。"""
    ctx = LoopContext(csv_path="/path/does/not/exist.csv")
    result = registry.invoke("load_dataset", {}, ctx)
    assert result.success is False
    assert ctx.cleaned_df is None


def test_detect_windows_requires_load_first():
    """未加载数据时调用窗口检测应返回友好错误。"""
    ctx = LoopContext(csv_path="/fake.csv")
    result = registry.invoke("detect_candidate_windows", {}, ctx)
    assert result.success is False
    assert "load_dataset" in result.reasoning


def test_detect_windows_after_load(synthetic_csv: str):
    """合成数据含 SV 阶跃，应至少检测到一个候选窗口。"""
    ctx = LoopContext(csv_path=synthetic_csv)
    load_res = registry.invoke("load_dataset", {}, ctx)
    assert load_res.success

    win_res = registry.invoke("detect_candidate_windows", {}, ctx)
    assert win_res.success, f"窗口检测失败: {win_res.reasoning}"
    assert win_res.data["candidate_count"] >= 1
    # 上下文已填充
    assert len(ctx.candidate_windows) == win_res.data["candidate_count"]
    # 摘要字段齐全
    if win_res.data["windows"]:
        w = win_res.data["windows"][0]
        for key in ("index", "start", "end", "n_points", "score", "usable", "source"):
            assert key in w


def test_audit_log_records_both_skills(synthetic_csv: str):
    """两个技能调用都应写入 ctx.skill_log。"""
    ctx = LoopContext(csv_path=synthetic_csv)
    registry.invoke("load_dataset", {}, ctx)
    registry.invoke("detect_candidate_windows", {}, ctx)
    assert len(ctx.skill_log) == 2
    assert ctx.skill_log[0]["skill"] == "load_dataset"
    assert ctx.skill_log[1]["skill"] == "detect_candidate_windows"


def test_openai_tool_schema_for_new_skills():
    """两个技能的 OpenAI schema 都应可生成。"""
    tools = registry.to_openai_tools(["load_dataset", "detect_candidate_windows"])
    assert len(tools) == 2
    names = {t["function"]["name"] for t in tools}
    assert names == {"load_dataset", "detect_candidate_windows"}
    # description 应为中文
    for t in tools:
        desc = t["function"]["description"]
        assert any("\u4e00" <= ch <= "\u9fff" for ch in desc), "description 必须含中文"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
