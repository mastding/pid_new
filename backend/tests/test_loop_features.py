from __future__ import annotations

import numpy as np
import pandas as pd

from core.shared.loop_features import extract_loop_features
from core.skills import LoopContext, registry
from core.skills.monitoring.assess_loop_monitoring_skill import assess_loop_monitoring_from_features


def _make_history_df(n: int = 240, *, include_sp: bool = True) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01 00:00:00", periods=n, freq="30s")
    mv = np.full(n, 45.0, dtype=float)
    mv[40:90] += np.linspace(0.0, 8.0, 50)
    mv[90:160] += 8.0
    mv[160:] += 4.0 * np.sin(np.arange(n - 160) / 12.0)
    pv = 100.0 + np.cumsum((mv - 45.0) * 0.015)
    pv += 0.2 * np.sin(np.arange(n) / 5.0)

    data = {
        "timestamp": ts,
        "PV": pv,
        "MV": mv,
    }
    if include_sp:
        sp = np.full(n, 100.0, dtype=float)
        sp[120:] = 102.0
        data["SV"] = sp
    return pd.DataFrame(data)


def test_extract_loop_features_returns_raw_observable_groups():
    features = extract_loop_features(
        _make_history_df(),
        loop_id="5203_FIC_10103",
        loop_type="flow",
        source_file="sample.csv",
        dataset_id="dataset_a",
        sample_time_s=30.0,
    )

    assert features["identity"]["loop_id"] == "5203_FIC_10103"
    assert features["identity"]["loop_type"] == "flow"
    assert features["data_profile"]["row_count"] == 240
    assert features["data_profile"]["sample_time_median_s"] == 30.0
    assert features["data_quality_raw"]["missing_ratio_total"] == 0.0
    assert features["pv_stats"]["available"] is True
    assert features["mv_stats"]["move_count"] > 0
    assert features["sp_stats"]["available"] is True
    assert features["sp_tracking_raw"]["sp_available"] is True
    assert features["event_raw"]["mv_adjacent_change_count"] > 0
    assert features["pv_mv_relation_raw"]["estimated_direction_raw"] in {"positive", "negative", "uncertain"}
    assert "pv_dominant_period_s" in features["frequency_raw"]
    assert "rolling_pv_std_median" in features["stationarity_raw"]
    assert features["operating_summary_raw"]["data_has_timestamp"] is True


def test_extract_loop_features_omits_skill_outputs():
    features = extract_loop_features(
        _make_history_df(),
        loop_id="L1",
        loop_type="flow",
        sample_time_s=30.0,
    )
    text = str(features)

    assert "candidate_windows" not in text
    assert "usable_window_count" not in text
    assert "best_window_score" not in text
    assert "oscillation_detected" not in text
    assert "nonlinearity_detected" not in text
    assert "recommend_tuning" not in text
    assert "diagnosis_causes" not in text


def test_extract_loop_features_handles_missing_sp_column():
    features = extract_loop_features(
        _make_history_df(include_sp=False),
        loop_id="L2",
        loop_type="flow",
        sample_time_s=30.0,
    )

    assert features["identity"]["sp_column"] is None
    assert features["sp_stats"]["available"] is False
    assert features["sp_tracking_raw"]["sp_available"] is False
    assert "SP/SV column not available" in features["sp_tracking_raw"]["reason"]


def test_assess_loop_monitoring_from_features_is_snapshot_not_diagnosis():
    features = extract_loop_features(
        _make_history_df(),
        loop_id="L3",
        loop_type="flow",
        sample_time_s=30.0,
    )

    result = assess_loop_monitoring_from_features(features)

    assert result["status"] in {"normal", "warning", "alarm"}
    assert 0.0 <= result["overall_score"] <= 1.0
    assert "data_health" in result
    assert "stability" in result
    assert "pv_mv_behavior" in result
    assert "recommend_tuning" not in str(result)
    assert "primary_causes" not in str(result)


def test_assess_loop_monitoring_skill_registered_and_runs():
    assert "assess_loop_monitoring" in set(registry.names())

    df = _make_history_df()
    ctx = LoopContext(csv_path="synthetic.csv", loop_type="flow")
    ctx.cleaned_df = df
    ctx.dt = 30.0

    result = registry.invoke("assess_loop_monitoring", {"loop_id": "L4"}, ctx)

    assert result.success, result.reasoning
    assert result.data["provider"] == "raw_feature_rules"
    assert result.data["features"]["identity"]["loop_id"] == "L4"
    assert result.data["monitoring"]["status"] in {"normal", "warning", "alarm"}
