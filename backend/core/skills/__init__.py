"""Skill package and auto-registration entry point."""

from core.skills.base import BaseSkill, LoopContext, NoInputs, SkillResult
from core.skills.registry import register, registry

# Import subpackages for side-effect registration.
from core.skills import assessment  # noqa: F401
from core.skills import data_understanding  # noqa: F401
from core.skills import dead_time  # noqa: F401
from core.skills import evaluation  # noqa: F401
from core.skills import identification  # noqa: F401
from core.skills import monitoring  # noqa: F401
from core.skills import ontology  # noqa: F401
from core.skills import tuning  # noqa: F401
from core.skills import window  # noqa: F401

# 发现并注册 `backend/var/skills/<name>/` 下的外部 skill（按目录即插件）。
# 任何加载失败都被内部 try/except 兜住，不影响主线 skill 的注册。
from core.skills.external_loader import discover_external_skills  # noqa: E402

discover_external_skills()

__all__ = [
    "BaseSkill",
    "LoopContext",
    "NoInputs",
    "SkillResult",
    "register",
    "registry",
]
