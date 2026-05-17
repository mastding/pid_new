"""Lightweight experience store backed by learning snapshot JSON files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.history.store import get_loop_features, list_loops
from core.learning import attach_outcome, iter_snapshots


_VAR_ROOT = Path(__file__).resolve().parent.parent.parent / "var"
_FALLBACK_LEARNING_DIR = _VAR_ROOT / "learning" / "snapshots"
_EXTERNAL_SKILLS_DIR = _VAR_ROOT / "skills"


def _safe_ts(item: dict[str, Any]) -> float:
    try:
        return float(item.get("ts") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _snapshot_dirs() -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    if _FALLBACK_LEARNING_DIR.exists():
        for path in _FALLBACK_LEARNING_DIR.iterdir():
            if path.is_dir():
                dirs[path.name] = path
    if _EXTERNAL_SKILLS_DIR.exists():
        for skill_dir in _EXTERNAL_SKILLS_DIR.iterdir():
            snapshots_dir = skill_dir / "learning" / "snapshots"
            if skill_dir.is_dir() and snapshots_dir.exists():
                dirs.setdefault(skill_dir.name, snapshots_dir)
    return dirs


def _summarize_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    payload_keys = [
        key for key in item.keys()
        if key not in {
            "snapshot_id",
            "ts",
            "skill_name",
            "skill_version",
            "code_origin",
            "observable_outcomes",
        }
    ]
    return {
        "snapshot_id": item.get("snapshot_id"),
        "ts": item.get("ts"),
        "skill_name": item.get("skill_name"),
        "skill_version": item.get("skill_version"),
        "code_origin": item.get("code_origin"),
        "has_outcome": isinstance(item.get("observable_outcomes"), dict),
        "observable_outcomes": item.get("observable_outcomes"),
        "payload_keys": payload_keys,
        "payload_preview": {key: item.get(key) for key in payload_keys[:8]},
    }


def _feature_float(features: dict[str, Any], *path: str) -> float | None:
    current: Any = features
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _closeness(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return max(0.0, 1.0 - abs(a - b) / scale)


class ExperienceStore:
    """Read/write view over learning snapshots produced by tuning skills."""

    def list_skills(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for skill_name in sorted(_snapshot_dirs().keys()):
            snapshots = list(iter_snapshots(skill_name))
            with_outcome = [item for item in snapshots if isinstance(item.get("observable_outcomes"), dict)]
            latest = max(snapshots, key=_safe_ts, default=None)
            items.append({
                "skill_name": skill_name,
                "snapshot_count": len(snapshots),
                "outcome_count": len(with_outcome),
                "latest_ts": latest.get("ts") if latest else None,
                "latest_snapshot_id": latest.get("snapshot_id") if latest else None,
            })
        return {"total": len(items), "items": items}

    def list_snapshots(
        self,
        *,
        skill_name: str | None = None,
        only_with_outcome: bool = False,
        limit: int = 100,
    ) -> dict[str, Any]:
        skill_names = [skill_name] if skill_name else sorted(_snapshot_dirs().keys())
        snapshots: list[dict[str, Any]] = []
        for name in skill_names:
            for item in iter_snapshots(name, only_with_outcome=only_with_outcome):
                snapshots.append(_summarize_snapshot(item))
        snapshots.sort(key=_safe_ts, reverse=True)
        capped = snapshots[: max(1, min(int(limit), 500))]
        return {"total": len(snapshots), "items": capped}

    def attach_outcome(self, *, skill_name: str, snapshot_id: str, outcome: dict[str, Any]) -> dict[str, Any]:
        ok = attach_outcome(skill_name, snapshot_id, outcome)
        if not ok:
            raise ValueError("snapshot_id not found")
        matches = [
            _summarize_snapshot(item)
            for item in iter_snapshots(skill_name)
            if item.get("snapshot_id") == snapshot_id
        ]
        return {"ok": True, "snapshot": matches[0] if matches else None}

    def similar_loops(self, *, loop_id: str, limit: int = 10) -> dict[str, Any]:
        target_features = get_loop_features(loop_id)
        if target_features.get("error"):
            raise ValueError("loop_id not found")
        target_type = str((target_features.get("identity") or {}).get("loop_type") or "")
        target_vector = {
            "sample_time": _feature_float(target_features, "data_profile", "sample_time_median_s"),
            "pv_span": _feature_float(target_features, "pv_stats", "span"),
            "mv_span": _feature_float(target_features, "mv_stats", "span"),
            "mv_travel": _feature_float(target_features, "mv_stats", "travel_per_hour"),
            "excitation": _feature_float(target_features, "excitation_profile", "usable_excitation_ratio"),
            "saturation_free": _feature_float(target_features, "excitation_profile", "saturation_free_ratio"),
        }
        rows: list[dict[str, Any]] = []
        for loop in list_loops():
            candidate_id = str(loop.get("loop_id") or "")
            if not candidate_id or candidate_id == loop_id:
                continue
            try:
                features = get_loop_features(candidate_id)
            except Exception:
                continue
            if features.get("error"):
                continue
            candidate_type = str((features.get("identity") or {}).get("loop_type") or loop.get("loop_type") or "")
            type_score = 1.0 if candidate_type == target_type else 0.35
            vector_score = (
                0.18 * _closeness(target_vector["sample_time"], _feature_float(features, "data_profile", "sample_time_median_s"))
                + 0.18 * _closeness(target_vector["pv_span"], _feature_float(features, "pv_stats", "span"))
                + 0.18 * _closeness(target_vector["mv_span"], _feature_float(features, "mv_stats", "span"))
                + 0.14 * _closeness(target_vector["mv_travel"], _feature_float(features, "mv_stats", "travel_per_hour"))
                + 0.16 * _closeness(target_vector["excitation"], _feature_float(features, "excitation_profile", "usable_excitation_ratio"))
                + 0.16 * _closeness(target_vector["saturation_free"], _feature_float(features, "excitation_profile", "saturation_free_ratio"))
            )
            score = round(0.4 * type_score + 0.6 * vector_score, 4)
            rows.append({
                "loop_id": candidate_id,
                "loop_type": candidate_type,
                "source_filename": loop.get("source_filename"),
                "similarity_score": score,
                "evidence": {
                    "same_loop_type": candidate_type == target_type,
                    "sample_time_s": _feature_float(features, "data_profile", "sample_time_median_s"),
                    "pv_span": _feature_float(features, "pv_stats", "span"),
                    "mv_span": _feature_float(features, "mv_stats", "span"),
                    "usable_excitation_ratio": _feature_float(features, "excitation_profile", "usable_excitation_ratio"),
                    "saturation_free_ratio": _feature_float(features, "excitation_profile", "saturation_free_ratio"),
                },
            })
        rows.sort(key=lambda item: float(item.get("similarity_score") or 0.0), reverse=True)
        return {
            "loop_id": loop_id,
            "loop_type": target_type,
            "total": len(rows),
            "items": rows[: max(1, min(int(limit), 50))],
        }


experience_store = ExperienceStore()
