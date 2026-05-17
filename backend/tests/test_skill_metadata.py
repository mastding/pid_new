from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.skills  # noqa: F401
from core.skills.registry import registry


def test_all_registered_skills_have_risk_stage_and_effects():
    missing: list[str] = []
    for meta in registry.all_metadata():
        if not meta.get("risk_level"):
            missing.append(f"{meta['name']}:risk_level")
        if not meta.get("stage") or meta.get("stage") == "general":
            missing.append(f"{meta['name']}:stage")
        if not meta.get("effects"):
            missing.append(f"{meta['name']}:effects")
        if not isinstance(meta.get("preconditions"), list):
            missing.append(f"{meta['name']}:preconditions")

    assert missing == []


def test_high_risk_skills_are_not_llm_direct_by_default():
    high_risk = [
        meta["name"]
        for meta in registry.all_metadata()
        if meta.get("risk_level") == "high"
    ]

    assert {"identify_model", "generate_tuning_candidates", "evaluate_tuning"}.issubset(set(high_risk))
