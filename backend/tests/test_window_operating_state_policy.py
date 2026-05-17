from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.algorithms.data_analysis import score_window


def test_score_window_applies_ontology_operating_state_avoidance():
    mv = np.array([0.0] * 60 + [100.0] * 60)
    pv = np.concatenate([np.zeros(60), np.linspace(0.0, 10.0, 60)])
    df = pd.DataFrame({"MV": mv, "PV": pv})

    result = score_window(
        df,
        policy={
            "max_mv_saturation_ratio": 0.1,
            "allowed_operating_states": ["stable", "mild_load_change"],
            "avoid_operating_states": ["hard_saturation"],
        },
    )

    assert result["operating_state"] == "hard_saturation"
    assert result["passed"] is False
    assert result["score_breakdown"]["operating_state_penalty"] == 0.55
    assert any("operating_state hard_saturation" in reason for reason in result["reasons"])
