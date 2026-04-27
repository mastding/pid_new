from __future__ import annotations

from core.pipeline.refinement_policy import recommend_refinement_from_algorithm_comparison


def test_recommend_refinement_picks_different_algorithm_candidate():
    result = recommend_refinement_from_algorithm_comparison(
        loop_type="flow",
        windows_summary=[
            {"index": 0, "source": "mv_step_1"},
            {"index": 1, "source": "steady_disturbance_1"},
        ],
        algorithm_comparison=[
            {
                "algorithm": "mv_step",
                "algorithm_label": "mv_step",
                "window_source": "mv_step_1",
                "model_type": "FOPDT",
                "fit_score": 20.0,
                "r2_score": 0.50,
                "confidence": 0.45,
                "window_quality_score": 0.90,
            },
            {
                "algorithm": "steady_disturbance",
                "algorithm_label": "steady_disturbance_scan",
                "window_source": "steady_disturbance_1",
                "model_type": "FO",
                "fit_score": 18.0,
                "r2_score": 0.62,
                "confidence": 0.60,
                "window_quality_score": 0.82,
            },
        ],
        last_best={"model_type": "FOPDT", "window_source": "mv_step_1"},
        last_review={"reason": "R2偏低"},
    )

    assert result is not None
    assert result["retry"] is True
    assert result["source"] == "deterministic_algorithm_policy"
    assert result["force_window_index"] == 1
    assert result["force_model_types"][0] == "FO"
    assert result["recommended_algorithm"] == "steady_disturbance"
    assert result["policy"]["min_confidence"] == 0.25


def test_recommend_refinement_returns_none_without_diverse_candidate():
    result = recommend_refinement_from_algorithm_comparison(
        loop_type="flow",
        windows_summary=[{"index": 0, "source": "mv_step_1"}],
        algorithm_comparison=[
            {
                "algorithm": "mv_step",
                "window_source": "mv_step_1",
                "model_type": "FOPDT",
                "fit_score": 20.0,
                "r2_score": 0.50,
                "confidence": 0.45,
                "window_quality_score": 0.90,
            },
        ],
        last_best={"model_type": "FOPDT", "window_source": "mv_step_1"},
        last_review={"reason": "R2偏低"},
    )

    assert result is None
