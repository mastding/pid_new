"""Aggregate evaluation provider composed from simulation/scoring/reality-check providers."""
from __future__ import annotations

from core.algorithms.pid_evaluation import _final_rating, _perturb
from core.policies.scoring_rules import apply_score_caps
from core.providers.evaluation.base import BaseEvaluationProvider
from core.providers.evaluation.scenario_builder import build_simulation_scenarios
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
        ctx = (context or {}).get("ctx") if isinstance(context, dict) else None
        existing_scenario = None
        if ctx is not None and isinstance(getattr(ctx, "data_profile", None), dict):
            existing_scenario = ctx.data_profile.get("simulation_scenario")
        scenario_pack = existing_scenario if isinstance(existing_scenario, dict) else build_simulation_scenarios(
            loop_type=loop_type,
            model_params=mp,
            dt=dt,
            context=context,
        )
        primary = scenario_pack["primary"]
        reverse = scenario_pack["reverse"]
        sim_dt = float(primary["dt"])
        n_steps = int(primary["n_steps"])

        fwd = sim_provider.simulate(
            model_params=mp,
            pid_params=pid,
            sp_initial=float(primary["sp_initial"]),
            sp_final=float(primary["sp_final"]),
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
            sp_initial=float(reverse["sp_initial"]),
            sp_final=float(reverse["sp_final"]),
            n_steps=int(reverse["n_steps"]),
            dt=float(reverse["dt"]),
            loop_type=loop_type,
            context=context,
        )
        rev_score = float(score_provider.score(simulation=rev, context=context)["score"])

        rob_scores: list[float] = []
        worst_robustness_sim: dict[str, object] | None = None
        worst_robustness_score: float | None = None
        for variant in _perturb(mp):
            vsim = sim_provider.simulate(
                model_params=variant,
                pid_params=pid,
                sp_initial=float(primary["sp_initial"]),
                sp_final=float(primary["sp_final"]),
                n_steps=n_steps,
                dt=sim_dt,
                loop_type=loop_type,
                context=context,
            )
            variant_score = float(score_provider.score(simulation=vsim, context=context)["score"])
            rob_scores.append(variant_score)
            if worst_robustness_score is None or variant_score < worst_robustness_score:
                worst_robustness_score = variant_score
                worst_robustness_sim = vsim
        rob_score = round(min(rob_scores) * 0.6 + (sum(rob_scores) / len(rob_scores)) * 0.4, 2) if rob_scores else 0.0

        if isinstance(context, dict):
            context["evaluation_primary_scenario"] = primary
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
            # 把入参的不可靠标志也回传给前端，前端据此显示红条警告。
            # 之前这两个字段会被 cap_reasons 吞进 recommendation 文本里，但前端
            # 难以做条件渲染。
            "tuning_unreliable": bool(tuning_unreliable),
            "tuning_unreliable_reason": tuning_unreliable_reason or "",
            "simulation": {
                "pv_history": fwd["pv_history"],
                "mv_history": fwd["mv_history"],
                "sp_history": fwd["sp_history"],
                "dt": sim_dt,
                "scenario_id": primary["id"],
            },
            "simulation_traces": {
                "nominal_sp_step": {
                    "label": primary["label"],
                    "role": "primary",
                    "score": perf_score,
                    "overshoot_percent": fwd["overshoot"],
                    "settling_time_s": fwd["settling_time"],
                    "is_stable": fwd["is_stable"],
                    "pv_history": fwd["pv_history"],
                    "mv_history": fwd["mv_history"],
                    "sp_history": fwd["sp_history"],
                    "dt": sim_dt,
                    "scenario_id": primary["id"],
                },
                "reverse_sp_step": {
                    "label": reverse["label"],
                    "role": "reverse",
                    "score": rev_score,
                    "overshoot_percent": rev["overshoot"],
                    "settling_time_s": rev["settling_time"],
                    "is_stable": rev["is_stable"],
                    "pv_history": rev["pv_history"],
                    "mv_history": rev["mv_history"],
                    "sp_history": rev["sp_history"],
                    "dt": float(reverse["dt"]),
                    "scenario_id": reverse["id"],
                },
                "robustness_worst_case": {
                    "label": "鲁棒性最差场景",
                    "role": "robustness",
                    "score": worst_robustness_score,
                    "overshoot_percent": (worst_robustness_sim or {}).get("overshoot"),
                    "settling_time_s": (worst_robustness_sim or {}).get("settling_time"),
                    "is_stable": (worst_robustness_sim or {}).get("is_stable"),
                    "pv_history": (worst_robustness_sim or {}).get("pv_history", []),
                    "mv_history": (worst_robustness_sim or {}).get("mv_history", []),
                    "sp_history": (worst_robustness_sim or {}).get("sp_history", []),
                    "dt": sim_dt,
                    "scenario_id": "robustness_worst_case",
                },
            },
            "simulation_scenario": scenario_pack,
        }
