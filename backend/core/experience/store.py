"""Lightweight experience store backed by learning snapshot JSON files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

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


experience_store = ExperienceStore()
