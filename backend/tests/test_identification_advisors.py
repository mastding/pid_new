from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pipeline import identification_advisor as review_mod
from core.pipeline import identification_refinement_advisor as refine_mod


class _FakeCompletions:
    def __init__(self, message):
        self._message = message

    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self._message)],
            request=kwargs,
        )


class _FakeOpenAI:
    def __init__(self, *, message):
        self.chat = SimpleNamespace(completions=_FakeCompletions(message))


def test_review_advisor_maps_reject_to_downgrade(monkeypatch):
    monkeypatch.setattr(review_mod.settings, "model_api_key", "k")
    monkeypatch.setattr(review_mod.settings, "model_api_url", "http://example.test")
    monkeypatch.setattr(review_mod.settings, "model_name", "fake-model")

    message = SimpleNamespace(
        content='{"verdict":"reject","reason":"模型可疑","concerns":["K符号异常"]}',
        reasoning_content="chain",
    )
    monkeypatch.setattr(
        review_mod,
        "OpenAI",
        lambda **kwargs: _FakeOpenAI(message=message),
    )

    result = review_mod.review_identification_via_llm(
        loop_type="flow",
        data_profile={"text_summary": "ok", "pv_stats": {}, "mv_stats": {}},
        chosen_window_summary={"source": "w0", "score": 0.9, "n_points": 100},
        best_model={"model_type": "FOPDT", "K": 1.0, "T": 5.0, "L": 1.0},
        attempts=[],
        confidence=0.6,
    )

    assert result is not None
    assert result["verdict"] == "downgrade"
    assert result["reason"] == "模型可疑"
    assert result["concerns"] == ["K符号异常"]
    assert result["reasoning_content"] == "chain"


def test_review_advisor_returns_failure_details_for_invalid_verdict(monkeypatch):
    monkeypatch.setattr(review_mod.settings, "model_api_key", "k")
    monkeypatch.setattr(review_mod.settings, "model_api_url", "http://example.test")
    monkeypatch.setattr(review_mod.settings, "model_name", "fake-model")

    message = SimpleNamespace(
        content='{"verdict":"maybe","reason":"bad","concerns":[]}',
        reasoning_content="",
    )
    monkeypatch.setattr(
        review_mod,
        "OpenAI",
        lambda **kwargs: _FakeOpenAI(message=message),
    )

    result = review_mod.review_identification_via_llm(
        loop_type="flow",
        data_profile={"text_summary": "ok", "pv_stats": {}, "mv_stats": {}},
        chosen_window_summary={"source": "w0", "score": 0.9, "n_points": 100},
        best_model={"model_type": "FOPDT", "K": 1.0, "T": 5.0, "L": 1.0},
        attempts=[],
        confidence=0.6,
    )

    assert result is not None
    assert result["available"] is False
    assert result["error_type"] == "invalid_verdict"
    assert "illegal verdict" not in result["error_message"]
    assert "maybe" in result["error_message"]


def test_refinement_advisor_sanitizes_window_models_and_hint(monkeypatch):
    monkeypatch.setattr(refine_mod.settings, "model_api_key", "k")
    monkeypatch.setattr(refine_mod.settings, "model_api_url", "http://example.test")
    monkeypatch.setattr(refine_mod.settings, "model_name", "fake-model")

    message = SimpleNamespace(
        content=(
            '{"retry":true,"rationale":"换模型重试","force_window_index":"2",'
            '"force_model_types":["ifopdt","bad","SOPDT_UNDER"],"hint_L":"3.5"}'
        ),
        reasoning_content="refine-chain",
    )
    monkeypatch.setattr(
        refine_mod,
        "OpenAI",
        lambda **kwargs: _FakeOpenAI(message=message),
    )

    result = refine_mod.ask_refinement_via_llm(
        loop_type="level",
        round_idx=1,
        max_rounds=2,
        data_profile={"text_summary": "ok"},
        windows_summary=[],
        last_best={"model_type": "IPDT", "window_source": "w0"},
        last_attempts=[],
        last_review={"verdict": "downgrade", "reason": "差", "concerns": []},
        history_summary=[],
    )

    assert result is not None
    assert result["retry"] is True
    assert result["rationale"] == "换模型重试"
    assert result["force_window_index"] == 2
    assert result["force_model_types"] == ["IFOPDT", "SOPDT_UNDER"]
    assert result["hint_L"] == 3.5
    assert result["reasoning_content"] == "refine-chain"
