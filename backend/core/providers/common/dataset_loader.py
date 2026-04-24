"""Dataset loading provider for cleaned process data."""
from __future__ import annotations

from typing import Any

from core.algorithms.data_analysis import _detect_loops, _load_clean_only, _read_csv
from core.shared import register_provider


@register_provider("dataset_loading")
class DatasetLoaderProvider:
    name = "clean_csv_loader"

    def load(
        self,
        *,
        csv_path: str,
        selected_loop_prefix: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cleaned, dt = _load_clean_only(
            csv_path=csv_path,
            selected_loop_prefix=selected_loop_prefix,
            start_time=start_time,
            end_time=end_time,
        )

        loop_summary: list[dict[str, str]] = []
        try:
            raw = _read_csv(csv_path)
            loops = _detect_loops(raw)
            loop_summary = [
                {
                    "prefix": str(loop.get("prefix", "")),
                    "pv_col": str(loop.get("pv_col", "")),
                    "mv_col": str(loop.get("mv_col", "")),
                }
                for loop in loops
            ]
        except Exception:
            loop_summary = []

        return {
            "provider": self.name,
            "cleaned_df": cleaned,
            "dt": dt,
            "data_points": len(cleaned),
            "columns": [c for c in ["timestamp", "SV", "PV", "MV"] if c in cleaned.columns],
            "loops_in_csv": loop_summary,
            "meta": {
                "selected_loop_prefix": selected_loop_prefix or "",
                "time_span_sec": float(dt) * len(cleaned),
            },
        }
