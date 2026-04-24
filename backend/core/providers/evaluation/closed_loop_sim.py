"""Aggregate evaluation provider composed from simulation/scoring/reality-check providers."""
from __future__ import annotations

from core.algorithms.pid_evaluation import _final_rating, _perturb
from core.policies.scoring_rules import apply_score_caps
from core.providers.evaluation.base import BaseEvaluationProvider
from core.shared import provider_registry, register_provider


@register_provider("evaluation")
class ClosedLoopSimulationProvider(BaseEvaluationProvider):
    name = "closed_loop_sim"

    def evaluate(
        self,
        *,
        Kp: float,
        Ki: float,
        Kd: float,
        model_type: str,
        model_params,
        K: float,
        T: float,
        L: float,
        dt: float,
        loop_type: str,
        confidence: float = 1.0,
        tuning_unreliable: bool = False,
        tuning_unreliable_reason: str = "",
        context=None,
    ) -> dict[str, object]:
        sim_provider = provider_registry.get("evaluation_simulation", "closed_loop_response")
        score_provider = provider_registry.get("evaluation_scoring", "step_response_scoring")
        reality_provider = provider_registry.get("evaluation_reality_check", "adaptive_typical_t")
        if sim_provider is None or score_provider is None or reality_provider is None:
            raise RuntimeError("evaluation aggregate provider requires simulation/scoring/reality providers")

        mp = dict(model_params or {})
        mt = (mp.get("model_type") or model_type or "FOPDT").strip().upper()
        mp.setdefault("model_type", mt)
        mp.setdefault("K", K)
        if mt == "SOPDT":
            mp.setdefault("T1", mp.get("T", T))
            mp.setdefault("T2", mp.get("T2", 0.0))
        elif mt == "IPDT":
            mp.setdefault("T1", max(T, 1e-3))
            mp.setdefault("T2", 0.0)
        else:
            mp.setdefault("T1", T)
            mp.setdefault("T2", 0.0)
        mp.setdefault("L", L)

        pid = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
        t1_for_sim = float(mp.get("T1", mp.get("T", 10.0)))
        sim_dt = max(0.05, min(float(dt), t1_for_sim / 10.0))
        n_steps = max(500, int(600.0 / sim_dt))

        fwd = sim_provider.simulate(
            model_params=mp,
            pid_params=pid,
            sp_initial=50.0,
            sp_final=60.0,
            n_steps=n_steps,
            dt=sim_dt,
            loop_type=loop_type,
            context=context,
        )
        perf = score_provider.score(simulation=fwd, context=context)
        perf_score = float(perf["score"])
        perf_details = dict(perf["details"])

        rev = sim_provider.simulate(
            model_params=mp,
            pid_params=pid,
            sp_initial=60.0,
            sp_final=50.0,
            n_steps=n_steps,
            dt=sim_dt,
            loop_type=loop_type,
            context=context,
        )
        rev_score = float(score_provider.score(simulation=rev, context=context)["score"])

        rob_scores: list[float] = []
        for variant in _perturb(mp):
            vsim = sim_provider.simulate(
                model_params=variant,
                pid_params=pid,
                sp_initial=50.0,
                sp_final=60.0,
                n_steps=n_steps,
                dt=sim_dt,
                loop_type=loop_type,
                context=context,
            )
            rob_scores.append(float(score_provider.score(simulation=vsim, context=context)["score"]))
        rob_score = round(min(rob_scores) * 0.6 + (sum(rob_scores) / len(rob_scores)) * 0.4, 2) if rob_scores else 0.0

        reality = reality_provider.check(
            model_params=mp,
            pid_params=pid,
            perf_score=perf_score,
            confidence=confidence,
            loop_type=loop_type,
            n_steps=n_steps,
            dt=sim_dt,
            context=context,
        )
        reality_score = float(reality["reality_score"])
        reality_diverged = bool(reality["diverged"])
        typical_t = float(reality["typical_t"])

        mv_hist = fwd.get("mv_history", [])
        sat_pct = (sum(1 for v in mv_hist if v <= 0.5 or v >= 99.5) / max(len(mv_hist), 1)) * 100.0
        mv_tv = sum(abs(mv_hist[i] - mv_hist[i - 1]) for i in range(1, len(mv_hist)))
        constraint_score = round(
            min(10.0, max(0.0, 10.0
                - min(5.0, sat_pct / 8.0)
                - min(3.0, mv_tv / 250.0)
                - (2.5 if not fwd["is_stable"] else 0.0))),
            2,
        )

        readiness = round(min(10.0, max(0.0,
            0.45 * perf_score
            + 0.20 * rev_score
            + 0.20 * rob_score
            + 0.15 * constraint_score
        )), 2)

        final = _final_rating(perf_score, confidence)
        passed = bool(readiness >= 7.0 and fwd["is_stable"] and rev["is_stable"] and sat_pct < 35.0)
        perf_score, final, readiness, passed, cap_reasons = apply_score_caps(
            perf_score=perf_score,
            final_rating_score=final,
            readiness_score=readiness,
            confidence=confidence,
            reality_diverged=reality_diverged,
            reality_score=reality_score,
            loop_type=loop_type,
            tuning_unreliable=tuning_unreliable,
            tuning_unreliable_reason=tuning_unreliable_reason,
            passed=passed,
        )

        if cap_reasons:
            recommendation = "暂不建议上线：" + "；".join(cap_reasons)
        elif passed and readiness >= 8.5:
            recommendation = "建议进入受控条件下的小扰动试投。"
        elif passed:
            recommendation = "建议保守试投，并保留人工确认。"
        else:
            recommendation = "暂不建议直接上线，建议继续回流优化。"

        return {
            "provider": self.name,
            "passed": passed,
            "performance_score": perf_score,
            "final_rating": final,
            "readiness_score": readiness,
            "robustness_score": rob_score,
            "constraint_score": constraint_score,
            "is_stable": fwd["is_stable"],
            "overshoot_percent": fwd["overshoot"],
            "settling_time_s": fwd["settling_time"],
            "steady_state_error": fwd["steady_state_error"],
            "oscillation_count": fwd["oscillation_count"],
            "decay_ratio": fwd["decay_ratio"],
            "rise_time_s": fwd["rise_time"],
            "mv_saturation_pct": round(sat_pct, 2),
            "performance_details": perf_details,
            "reality_check_score": round(reality_score, 2),
            "reality_check_typical_T": typical_t,
            "reality_check_diverged": reality_diverged,
            "score_caps_applied": cap_reasons,
            "recommendation": recommendation,
            "simulation": {
                "pv_history": fwd["pv_history"],
                "mv_history": fwd["mv_history"],
                "sp_history": fwd["sp_history"],
                "dt": sim_dt,
            },
        }
