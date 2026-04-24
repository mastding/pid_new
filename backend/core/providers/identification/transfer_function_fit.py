"""Transfer-function fitting identification provider."""
from __future__ import annotations

import pandas as pd

from core.algorithms.system_id import fit_best_model
from core.providers.identification.base import BaseIdentificationProvider
from core.shared import register_provider


@register_provider("identification")
class TransferFunctionFitProvider(BaseIdentificationProvider):
    name = "transfer_function_fit"

    def identify(
        self,
        *,
        cleaned_df: pd.DataFrame,
        candidate_windows: list[dict],
        actual_dt: float,
        loop_type: str,
        quality_metrics=None,
        force_model_types=None,
        force_l_hint=None,
        context=None,
    ) -> dict[str, object]:
        result = fit_best_model(
            cleaned_df=cleaned_df,
            candidate_windows=candidate_windows,
            actual_dt=actual_dt,
            loop_type=loop_type,
            quality_metrics=quality_metrics,
            force_model_types=force_model_types,
            force_L_hint=force_l_hint,
        )
        result["provider"] = self.name
        return result
