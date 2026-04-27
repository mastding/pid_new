from __future__ import annotations

import asyncio

from api.config_routes import get_policy_config


def test_get_policy_config_exposes_refinement_rules():
    payload = asyncio.run(get_policy_config())

    assert "loop_priors" in payload
    assert "refinement" in payload
    assert payload["refinement"]["fallback_rule"]["min_confidence"] == 0.25
    assert payload["refinement"]["model_fallbacks"]["temperature"] == ["FOPDT", "SOPDT", "SOPDT_UNDER"]
    assert payload["loop_priors"]["reality_t_ranges"]["level"] == {"min": 300.0, "max": 1800.0}
