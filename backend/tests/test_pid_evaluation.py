from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.algorithms import pid_evaluation as pe


def test_adaptive_reality_check_t_is_data_aware_for_level():
    t = pe._adaptive_reality_check_t("level", identified_t=194.69, confidence=0.81)
    assert t == 300.0

    t2 = pe._adaptive_reality_check_t("level", identified_t=900.0, confidence=0.8)
    assert 900.0 < t2 < 1800.0


def test_level_loop_uses_relaxed_settling_limit():
    limits = pe._stability_limits("level", 600.0)
    assert limits["settling_time_limit"] == 3600.0

    sim = pe._simulate(
        {"model_type": "FOPDT", "K": 1.0, "T1": 600.0, "T2": 0.0, "L": 0.0},
        {"Kp": 0.8, "Ki": 0.0035, "Kd": 0.0},
        sp_initial=50.0,
        sp_final=60.0,
        n_steps=2500,
        dt=1.0,
        loop_type="level",
    )
    assert sim["settling_time"] > 600.0
    assert sim["is_stable"] is True


def test_simulation_handles_fast_model_without_runtime_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        sim = pe._simulate(
            {"model_type": "FOPDT", "K": 5.0, "T1": 0.05, "T2": 0.0, "L": 0.0},
            {"Kp": 8.0, "Ki": 2.0, "Kd": 5.0},
            sp_initial=50.0,
            sp_final=60.0,
            n_steps=300,
            dt=1.0,
            loop_type="flow",
        )

    assert caught == []
    assert all(isinstance(sim[key], (int, float, bool)) for key in [
        "is_stable",
        "overshoot",
        "settling_time",
        "steady_state_error",
        "oscillation_count",
        "decay_ratio",
        "rise_time",
    ])
    assert all(abs(v) < 1e9 for v in sim["pv_history"] if isinstance(v, (int, float)))
    assert all(abs(v) <= 100.0 for v in sim["mv_history"] if isinstance(v, (int, float)))
