"""Skill package and auto-registration entry point."""

from core.skills.base import BaseSkill, LoopContext, NoInputs, SkillResult
from core.skills.registry import register, registry

# Import subpackages for side-effect registration.
from core.skills import _demo  # noqa: F401
from core.skills import data_understanding  # noqa: F401
from core.skills import dead_time  # noqa: F401
from core.skills import evaluation  # noqa: F401
from core.skills import identification  # noqa: F401
from core.skills import monitoring  # noqa: F401
from core.skills import tuning  # noqa: F401
from core.skills import window  # noqa: F401

__all__ = [
    "BaseSkill",
    "LoopContext",
    "NoInputs",
    "SkillResult",
    "register",
    "registry",
]
