"""Backward-compatible wrapper for the legacy detect_candidate_windows skill.

The new canonical entry point is ``detect_windows`` under ``core.skills.window``.
We keep this skill name registered so older prompts, tests, and smoke scripts
continue to work while all window-detection logic flows through the new skill.
"""
from __future__ import annotations

from core.skills.base import BaseSkill, LoopContext, NoInputs, SkillResult
from core.skills.registry import register
from core.skills.window.detect_windows_skill import DetectWindowsInputs, DetectWindowsSkill


@register
class DetectCandidateWindowsSkill(BaseSkill):
    name = "detect_candidate_windows"
    description = (
        "兼容旧入口：检测候选辨识窗口。"
        "内部会转调新的 detect_windows 技能，返回候选窗口数量、可用窗口数量和窗口摘要。"
    )
    input_model = NoInputs

    def run(self, inputs: NoInputs, ctx: LoopContext) -> SkillResult:
        result = DetectWindowsSkill().run(DetectWindowsInputs(), ctx)
        if not result.success:
            return result

        warnings = list(result.warnings)
        warnings.append("detect_candidate_windows is deprecated; prefer detect_windows.")
        return SkillResult(
            success=True,
            data=dict(result.data),
            warnings=warnings,
            reasoning=result.reasoning,
        )
