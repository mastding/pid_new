"""Expose how WindowSelectionPolicy fields are consumed by window providers."""
from __future__ import annotations

from typing import Any


WINDOW_POLICY_FIELD_LABELS: dict[str, str] = {
    "preferred_algorithm_families": "preferred algorithm families",
    "deprioritized_algorithm_families": "deprioritized algorithm families",
    "disabled_algorithm_families": "disabled algorithm families",
    "algorithm_plan": "algorithm family execution plan",
    "min_mv_excitation": "minimum MV excitation",
    "min_sp_excitation": "minimum SP excitation",
    "min_pv_response": "minimum PV response",
    "max_mv_saturation_ratio": "maximum MV saturation ratio",
    "max_pv_noise_ratio": "maximum PV noise ratio",
    "max_drift_ratio": "maximum PV drift ratio",
    "expected_dead_time_range_s": "expected dead-time range",
    "expected_time_constant_range_s": "expected time-constant range",
    "expected_gain_sign": "expected process gain sign",
    "min_window_points": "minimum window points",
    "min_window_duration_s": "minimum window duration",
    "max_window_points": "maximum window points",
    "pre_window_s": "pre-event window length",
    "post_window_s": "post-event window length",
    "steady_scan_window_s": "steady scan window length",
    "steady_scan_step_s": "steady scan step",
    "merge_gap_s": "event merge gap",
    "max_candidates_per_family": "maximum candidates per family",
    "allowed_operating_states": "allowed operating states",
    "avoid_operating_states": "avoided operating states",
    "scoring_weights": "policy scoring weights",
    "hard_guards": "hard guards",
    "soft_penalties": "soft penalties",
    "rationale": "policy rationale",
    "ontology_facts": "ontology facts",
}


FAMILY_CONSUMED_FIELDS: dict[str, list[str]] = {
    "sp_step": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "min_sp_excitation",
        "pre_window_s",
        "post_window_s",
        "max_window_points",
        "merge_gap_s",
        "min_window_points",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "mv_step": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "min_mv_excitation",
        "pre_window_s",
        "post_window_s",
        "max_window_points",
        "merge_gap_s",
        "min_window_points",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "mv_ramp": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "post_window_s",
        "pre_window_s",
        "max_window_points",
        "merge_gap_s",
        "max_candidates_per_family",
        "min_window_points",
        "min_mv_excitation",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "steady_disturbance": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "steady_scan_window_s",
        "steady_scan_step_s",
        "merge_gap_s",
        "max_candidates_per_family",
        "min_window_points",
        "min_mv_excitation",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
    "rolling_scan": [
        "algorithm_plan",
        "preferred_algorithm_families",
        "deprioritized_algorithm_families",
        "disabled_algorithm_families",
        "pre_window_s",
        "post_window_s",
        "max_window_points",
        "min_window_points",
        "min_pv_response",
        "max_mv_saturation_ratio",
        "max_drift_ratio",
    ],
}


DOWNSTREAM_HINT_FIELDS = {
    "expected_dead_time_range_s",
    "expected_time_constant_range_s",
    "expected_gain_sign",
}


DISPLAY_ONLY_FIELDS = {
    "max_pv_noise_ratio",
    "min_window_duration_s",
    "allowed_operating_states",
    "avoid_operating_states",
    "scoring_weights",
    "hard_guards",
    "soft_penalties",
    "rationale",
    "ontology_facts",
}


FIELD_USAGE_NOTES: dict[str, str] = {
    "algorithm_plan": "Controls whether each algorithm family is run, deprioritized, or disabled.",
    "preferred_algorithm_families": "Contributes to family plan and policy consistency scoring.",
    "deprioritized_algorithm_families": "Contributes to family plan and applies a soft score penalty.",
    "disabled_algorithm_families": "Prevents disabled families from running and hard-blocks matching candidates.",
    "min_mv_excitation": "Raises MV step / steady-disturbance excitation thresholds and hard-blocks weak MV windows.",
    "min_sp_excitation": "Raises SP step detection threshold.",
    "min_pv_response": "Hard-blocks windows whose PV response is below the policy threshold.",
    "max_mv_saturation_ratio": "Filters steady-disturbance windows and penalizes/highlights saturated candidates.",
    "max_drift_ratio": "Penalizes or blocks windows dominated by slow PV drift.",
    "min_window_points": "Controls steady scan minimum length and hard-blocks very short candidate windows.",
    "max_window_points": "Caps step/ramp/fallback post-event extraction length.",
    "pre_window_s": "Controls baseline length before event-centered windows.",
    "post_window_s": "Controls event-centered window horizon and MV ramp detection horizons.",
    "steady_scan_window_s": "Controls steady-disturbance rolling scan window length.",
    "steady_scan_step_s": "Controls steady-disturbance rolling scan stride.",
    "merge_gap_s": "Controls event merge distance across algorithm families.",
    "max_candidates_per_family": "Limits MV ramp and steady-disturbance candidate count.",
    "expected_dead_time_range_s": "Passed downstream as identification/model-review context; not used by window generators yet.",
    "expected_time_constant_range_s": "Passed downstream as identification/model-review context; not used by window generators yet.",
    "expected_gain_sign": "Passed downstream as model plausibility context; not used by window generators yet.",
    "max_pv_noise_ratio": "Recorded for audit/UI; current deterministic window providers do not consume it yet.",
    "min_window_duration_s": "Recorded for audit/UI; concrete window length is currently driven by pre/post/steady scan fields.",
    "allowed_operating_states": "Recorded for audit/UI; operating-state classifier is not wired into window filtering yet.",
    "avoid_operating_states": "Recorded for audit/UI; operating-state classifier is not wired into window filtering yet.",
}


def _field_consumers() -> dict[str, list[str]]:
    consumers: dict[str, list[str]] = {}
    for family, fields in FAMILY_CONSUMED_FIELDS.items():
        for field in fields:
            consumers.setdefault(field, []).append(family)
    return consumers


def build_policy_field_usage() -> list[dict[str, Any]]:
    """Return an auditable field-to-provider consumption map."""
    consumers = _field_consumers()
    rows: list[dict[str, Any]] = []
    fields = list(WINDOW_POLICY_FIELD_LABELS)
    for field in fields:
        used_by = consumers.get(field, [])
        if used_by:
            status = "consumed"
        elif field in DOWNSTREAM_HINT_FIELDS:
            status = "downstream_hint"
        else:
            status = "display_only"
        rows.append({
            "field": field,
            "label": WINDOW_POLICY_FIELD_LABELS.get(field, field),
            "status": status,
            "consumed_by": used_by,
            "note": FIELD_USAGE_NOTES.get(field, ""),
        })
    return rows


def enrich_policy_field_usage(policy: dict[str, Any]) -> dict[str, Any]:
    """Attach field usage metadata to a policy dict and its algorithm plan."""
    enriched = dict(policy)
    field_usage = build_policy_field_usage()
    enriched["field_usage"] = field_usage
    plan = []
    for item in enriched.get("algorithm_plan") or []:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "")
        plan_item = dict(item)
        consumed_fields = FAMILY_CONSUMED_FIELDS.get(family, [])
        plan_item["consumed_policy_fields"] = consumed_fields
        plan_item["consumed_policy_field_labels"] = [
            WINDOW_POLICY_FIELD_LABELS.get(field, field) for field in consumed_fields
        ]
        plan.append(plan_item)
    enriched["algorithm_plan"] = plan
    return enriched
