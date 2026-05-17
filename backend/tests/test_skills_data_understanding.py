from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from core.skills import LoopContext, registry


def _make_csv(path: Path, n: int = 600, with_step: bool = True) -> None:
    base_ts = 1_700_000_000_000
    rows = []
    sv = 50.0
    pv = 50.0
    mv = 30.0
    for i in range(n):
        ts = base_ts + i * 1000
        if with_step and i == n // 2:
            sv = 60.0
        mv = 30.0 + (sv - 50.0) * 0.8
        pv += (mv * 0.05 - pv * 0.05) * 0.1 + (sv - pv) * 0.02
        rows.append((ts, sv, pv, mv))

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "sv", "pv", "mv"])
        writer.writerows(rows)


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> str:
    path = tmp_path / "synthetic.csv"
    _make_csv(path)
    return str(path)


def test_load_dataset_skill_registered():
    assert "load_dataset" in registry.names()
    assert "detect_windows" in registry.names()
    assert "detect_candidate_windows" not in registry.names()


def test_load_dataset_basic(synthetic_csv: str):
    ctx = LoopContext(csv_path=synthetic_csv)
    result = registry.invoke("load_dataset", {}, ctx)

    assert result.success, f"load_dataset failed: {result.reasoning}"
    assert result.data["data_points"] > 500
    assert math.isclose(result.data["sampling_time"], 1.0, abs_tol=0.01)
    assert "PV" in result.data["columns"]
    assert "MV" in result.data["columns"]
    assert ctx.cleaned_df is not None
    assert ctx.dt is not None
    assert len(ctx.cleaned_df) > 500


def test_load_dataset_missing_file():
    ctx = LoopContext(csv_path="/path/does/not/exist.csv")
    result = registry.invoke("load_dataset", {}, ctx)
    assert result.success is False
    assert ctx.cleaned_df is None


def test_detect_windows_requires_load_first():
    ctx = LoopContext(csv_path="/fake.csv")
    result = registry.invoke("detect_windows", {}, ctx)
    assert result.success is False
    assert "load_dataset" in result.reasoning


def test_detect_windows_after_load(synthetic_csv: str):
    ctx = LoopContext(csv_path=synthetic_csv)
    load_res = registry.invoke("load_dataset", {}, ctx)
    assert load_res.success

    win_res = registry.invoke("detect_windows", {}, ctx)
    assert win_res.success, f"detect_windows failed: {win_res.reasoning}"
    assert win_res.data["candidate_count"] >= 1
    assert len(ctx.candidate_windows) == win_res.data["candidate_count"]
    if win_res.data["windows"]:
        window = win_res.data["windows"][0]
        for key in ("index", "start", "end", "n_points", "score", "usable", "source"):
            assert key in window


def test_audit_log_records_both_skills(synthetic_csv: str):
    ctx = LoopContext(csv_path=synthetic_csv)
    registry.invoke("load_dataset", {}, ctx)
    registry.invoke("detect_windows", {}, ctx)
    assert len(ctx.skill_log) == 2
    assert ctx.skill_log[0]["skill"] == "load_dataset"
    assert ctx.skill_log[1]["skill"] == "detect_windows"


def test_openai_tool_schema_for_current_skills():
    tools = registry.to_openai_tools(["load_dataset", "detect_windows"])
    assert len(tools) == 2
    names = {tool["function"]["name"] for tool in tools}
    assert names == {"load_dataset", "detect_windows"}
    for tool in tools:
        desc = tool["function"]["description"]
        assert any("\u4e00" <= ch <= "\u9fff" for ch in desc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
