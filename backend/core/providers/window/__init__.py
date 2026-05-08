"""Window detection providers."""

from core.providers.window.algorithm_families import (
    MvRampWindowFamilyProvider,
    MvStepWindowFamilyProvider,
    RollingScanWindowFamilyProvider,
    SpStepWindowFamilyProvider,
    SteadyDisturbanceWindowFamilyProvider,
)
from core.providers.window.history_rule_based import HistoryRuleBasedWindowProvider
from core.providers.window.policy_composite import PolicyCompositeWindowProvider
from core.providers.window.quality_score_selector import QualityScoreWindowSelectionProvider

__all__ = [
    "HistoryRuleBasedWindowProvider",
    "MvRampWindowFamilyProvider",
    "MvStepWindowFamilyProvider",
    "PolicyCompositeWindowProvider",
    "QualityScoreWindowSelectionProvider",
    "RollingScanWindowFamilyProvider",
    "SpStepWindowFamilyProvider",
    "SteadyDisturbanceWindowFamilyProvider",
]
