"""Window detection providers."""

from core.providers.window.history_rule_based import HistoryRuleBasedWindowProvider
from core.providers.window.quality_score_selector import QualityScoreWindowSelectionProvider

__all__ = ["HistoryRuleBasedWindowProvider", "QualityScoreWindowSelectionProvider"]
