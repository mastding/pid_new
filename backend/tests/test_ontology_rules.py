from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ontology_rules


def test_resolve_loop_ontology_facts_parses_structured_json(monkeypatch):
    ontology_rules.clear_ontology_fact_cache()

    async def fake_fetch_loop_ontology_context_via_mcp(**kwargs):
        return {
            "source": "test",
            "content": '{"case_id": "steady_case", "pv_lsl": 10, "pv_usl": 20}',
        }

    monkeypatch.setattr(
        ontology_rules,
        "fetch_loop_ontology_context_via_mcp",
        fake_fetch_loop_ontology_context_via_mcp,
    )

    facts = asyncio.run(
        ontology_rules.resolve_loop_ontology_facts(
            loop_id="L1",
            loop_type="flow",
            force_refresh=True,
        )
    )

    assert facts["schema_version"] == ontology_rules.ONTOLOGY_FACT_SCHEMA_VERSION
    assert facts["case_id"] == "steady_case"
    assert facts["pv_spec_limits"]["lsl"] == 10
    assert facts["pv_spec_limits"]["usl"] == 20
    assert facts["pv_spec_limits"]["source"] == "ontology_structured_json"
    assert facts["cache"]["hit"] is False


def test_resolve_loop_ontology_facts_uses_cache_and_force_refresh(monkeypatch):
    ontology_rules.clear_ontology_fact_cache()
    calls = {"count": 0}

    async def fake_fetch_loop_ontology_context_via_mcp(**kwargs):
        calls["count"] += 1
        return {
            "source": "test",
            "content": (
                '{"case_id": "case_%d", "pv_lsl": 1, "pv_usl": 2}'
                % calls["count"]
            ),
        }

    monkeypatch.setattr(
        ontology_rules,
        "fetch_loop_ontology_context_via_mcp",
        fake_fetch_loop_ontology_context_via_mcp,
    )

    first = asyncio.run(ontology_rules.resolve_loop_ontology_facts(loop_id="L2", loop_type="flow"))
    second = asyncio.run(ontology_rules.resolve_loop_ontology_facts(loop_id="L2", loop_type="flow"))
    refreshed = asyncio.run(
        ontology_rules.resolve_loop_ontology_facts(
            loop_id="L2",
            loop_type="flow",
            force_refresh=True,
        )
    )

    assert calls["count"] == 2
    assert first["case_id"] == "case_1"
    assert second["case_id"] == "case_1"
    assert second["cache"]["hit"] is True
    assert refreshed["case_id"] == "case_2"
    assert refreshed["cache"]["hit"] is False
