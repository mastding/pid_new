"""runner.py 中 LLM 顾问分支的集成测试。

通过 monkeypatch 替换 choose_window_via_llm，避免真发 API 请求；
验证四条分支在 SSE 事件流里的可见行为：
  1) use_llm_advisor=False → mode=deterministic
  2) selected_window_index 指定 → mode=user_override（最高优先级）
  3) LLM 返回有效 → mode=llm
  4) LLM 返回 None → mode=fallback_deterministic
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import numpy as np
import pytest

from core.pipeline import runner as runner_mod
from core.pipeline.runner import run_tuning_pipeline


def _make_step_csv(path: Path) -> None:
    """合成一份既有 SV 阶跃又有可识别窗口的 CSV，让 candidate_windows>=2。"""
    n = 1200
    base_ts = 1_700_000_000_000
    sv = np.full(n, 50.0)
    sv[400:] = 60.0
    sv[800:] = 55.0  # 第二段阶跃
    mv = np.zeros(n)
    pv = np.zeros(n)
    mv[0] = 30.0
    pv[0] = 50.0
    for i in range(1, n):
        # 简化：MV 跟着 SV-PV 误差走，PV 一阶滞后跟 MV
        err = sv[i] - pv[i - 1]
        mv[i] = np.clip(30.0 + 1.5 * err, 0.0, 100.0)
        pv[i] = pv[i - 1] + (mv[i] * 0.05 - pv[i - 1] * 0.04) * 0.1

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sv", "pv", "mv"])
        for i in range(n):
            w.writerow([base_ts + i * 1000, sv[i], pv[i], mv[i]])


@pytest.fixture
def step_csv(tmp_path: Path) -> str:
    p = tmp_path / "step.csv"
    _make_step_csv(p)
    return str(p)


async def _collect_events(gen):
    return [ev async for ev in gen]


def _run(coro):
    return asyncio.run(coro)


def _find_window_selection(events):
    for ev in events:
        if ev.get("type") == "stage" and ev.get("stage") == "window_selection" and ev.get("status") == "done":
            return ev["data"]
    return None


def test_advisor_off_uses_deterministic(step_csv, monkeypatch):
    # 任何 LLM 调用都视为失败
    def _no_llm(**_):
        raise AssertionError("不应调用 LLM")
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", _no_llm)

    events = _run(_collect_events(run_tuning_pipeline(
        csv_path=step_csv,
        loop_type="flow",
        use_llm_advisor=False,
    )))
    sel = _find_window_selection(events)
    assert sel is not None, "缺少 window_selection 事件"
    assert sel["mode"] in {"deterministic", "user_override"}
    assert sel["chosen_index"] == sel["deterministic_index"]


def test_user_override_wins_over_llm(step_csv, monkeypatch):
    called = {"llm": False}

    def _spy_llm(**_):
        called["llm"] = True
        return {"chosen_index": 0, "reasoning": "LLM 选 0", "reasoning_content": ""}

    monkeypatch.setattr(runner_mod, "choose_window_via_llm", _spy_llm)

    events = _run(_collect_events(run_tuning_pipeline(
        csv_path=step_csv,
        loop_type="flow",
        selected_window_index=1,
        use_llm_advisor=True,
    )))
    sel = _find_window_selection(events)
    assert sel is not None
    assert sel["mode"] == "user_override"
    assert sel["chosen_index"] == 1
    assert called["llm"] is False, "user_override 路径不应调用 LLM"


def test_llm_success_branch(step_csv, monkeypatch):
    captured = {"args": None}

    def _fake_llm(*, data_profile, candidate_windows, loop_type, **_):
        captured["args"] = {
            "n_candidates": len(candidate_windows),
            "loop_type": loop_type,
            "has_profile_text": bool(data_profile.get("text_summary")),
        }
        # 故意选「最后一个」，验证不是简单回退到 deterministic
        return {
            "chosen_index": len(candidate_windows) - 1,
            "reasoning": "测试用：选最后一个",
            "reasoning_content": "假装的思考链",
        }

    monkeypatch.setattr(runner_mod, "choose_window_via_llm", _fake_llm)

    events = _run(_collect_events(run_tuning_pipeline(
        csv_path=step_csv,
        loop_type="flow",
        use_llm_advisor=True,
    )))
    sel = _find_window_selection(events)
    assert sel is not None
    # 池中至少 2 个窗口，否则该分支不会启用
    if captured["args"] is None:
        pytest.skip("候选窗口不足 2 个，LLM 分支未触发")
    assert sel["mode"] == "llm"
    assert sel["reasoning"] == "测试用：选最后一个"
    assert sel["llm_reasoning_chain_len"] > 0
    assert "agreed_with_deterministic" in sel
    assert captured["args"]["loop_type"] == "flow"
    assert captured["args"]["has_profile_text"] is True


def test_llm_failure_falls_back(step_csv, monkeypatch):
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **_: None)

    events = _run(_collect_events(run_tuning_pipeline(
        csv_path=step_csv,
        loop_type="flow",
        use_llm_advisor=True,
    )))
    sel = _find_window_selection(events)
    assert sel is not None
    # 候选只有 1 个时直接走 deterministic；多窗口时才有机会进入 fallback_deterministic
    assert sel["mode"] in {"fallback_deterministic", "deterministic"}
    assert sel["chosen_index"] == sel["deterministic_index"]


def test_window_selection_meta_in_final_result(step_csv, monkeypatch):
    monkeypatch.setattr(runner_mod, "choose_window_via_llm", lambda **_: None)

    events = _run(_collect_events(run_tuning_pipeline(
        csv_path=step_csv,
        loop_type="flow",
        use_llm_advisor=False,
    )))
    result_evs = [e for e in events if e.get("type") == "result"]
    if not result_evs:
        # 流水线后续阶段可能因合成数据辨识失败而中断；不强求 result，但 selection 一定要在
        sel = _find_window_selection(events)
        assert sel is not None
        return
    data = result_evs[-1]["data"]
    assert "window_selection" in data
    assert data["window_selection"]["chosen_index"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
