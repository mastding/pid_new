"""技能包 —— LLM 可调用的原子工作单元。

import 本包即触发全局技能注册。各子包（data_understanding / identification /
tuning / evaluation）在模块加载时通过 @register 装饰器自动注册自身的技能。
"""
from core.skills.base import BaseSkill, LoopContext, NoInputs, SkillResult
from core.skills.registry import register, registry

# 副作用导入：每个模块通过 @register 自注册
from core.skills import _demo  # noqa: F401
from core.skills import data_understanding  # noqa: F401

__all__ = [
    "BaseSkill",
    "LoopContext",
    "NoInputs",
    "SkillResult",
    "register",
    "registry",
]
