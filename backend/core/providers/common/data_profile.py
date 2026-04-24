"""Data profile provider based on deterministic analyzers."""
from __future__ import annotations

from typing import Any

from core.shared import register_provider
from core.skills.data_understanding import _analyzers as A


def _text_summary(profile: dict[str, Any]) -> str:
    parts: list[str] = []
    pv = profile["pv_stats"]
    mv = profile["mv_stats"]
    noise = profile["noise"]
    osc = profile["oscillation"]
    dz = profile["deadzone"]

    parts.append(f"PV range {pv['min']}~{pv['max']} (span {pv['range']})")
    parts.append(f"MV range {mv['min']}~{mv['max']}")

    if mv["saturation_high_pct"] > 5 or mv["saturation_low_pct"] > 5:
        parts.append(
            f"MV saturation high/low {mv['saturation_high_pct']}%/{mv['saturation_low_pct']}%"
        )
    parts.append(f"PV noise {noise['noise_level']} ({noise['pv_noise_std']})")

    events_total = dz.get("events_total", 0)
    lag_used = dz.get("lag_used_s", 0.0)
    mv_thr = dz.get("mv_step_threshold", 0.0)
    if events_total == 0:
        parts.append(
            f"Deadzone undetermined (lag {lag_used}s, MV step threshold {mv_thr})"
        )
    else:
        ratio = dz["evidence_ratio"]
        evidence = dz.get("evidence_count", 0)
        suspect = "suspected" if ratio > 0.3 else "not obvious"
        parts.append(
            f"Deadzone {suspect} ({evidence}/{events_total}, ratio {ratio:.0%}, lag {lag_used}s)"
        )
    if osc["detected"]:
        parts.append(f"Oscillation detected T~{osc['period_sec']}s")

    return "; ".join(parts)


@register_provider("data_profile")
class DataProfileProvider:
    name = "deterministic_profile"

    def summarize(
        self,
        *,
        df,
        dt: float,
        loop_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        noise = A.analyze_noise(df)
        profile = {
            "pv_stats": A.analyze_pv_range(df),
            "mv_stats": A.analyze_mv_saturation(df),
            "noise": noise,
            "deadzone": A.analyze_deadzone(
                df,
                pv_noise_std=noise["pv_noise_std"],
                dt=float(dt),
                loop_type=loop_type,
            ),
            "oscillation": A.analyze_oscillation(df, dt=float(dt)),
            "disturbance": A.analyze_disturbance(df),
        }
        profile["text_summary"] = _text_summary(profile)

        warnings: list[str] = []
        if profile["mv_stats"]["saturation_high_pct"] > 30:
            warnings.append("MV high saturation ratio may distort identification.")
        if profile["mv_stats"]["saturation_low_pct"] > 30:
            warnings.append("MV low saturation ratio may distort identification.")
        if profile["deadzone"]["evidence_ratio"] > 0.5:
            warnings.append("High deadzone evidence ratio may weaken small-signal tuning.")
        if profile["noise"]["noise_level"] == "high":
            warnings.append("PV noise is high; consider stronger denoising before identification.")

        return {
            "provider": self.name,
            "profile": profile,
            "warnings": warnings,
            "reasoning": profile["text_summary"],
        }
