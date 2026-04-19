"""技能注册表 —— 全局技能目录。

各技能在 import 时通过 @register 装饰器自动注册。注册表提供：
  - 列出所有已注册技能
  - 为任意子集（或全部）生成 OpenAI tool schema
  - 将 LLM 的 tool_call 调度到对应技能执行
"""
from __future__ import annotations

from typing import Any

from core.skills.base import BaseSkill, LoopContext, SkillResult


class SkillRegistry:
    """内存型技能注册表。通过模块级单例使用。"""

    def __init__(self) -> None:
        self._skills: dict[str, type[BaseSkill]] = {}

    def register(self, skill_cls: type[BaseSkill]) -> type[BaseSkill]:
        """注册一个技能类（也可作为 @register 装饰器使用）。"""
        if not hasattr(skill_cls, "name") or not skill_cls.name:
            raise ValueError(f"{skill_cls.__name__} 必须定义类属性 `name`")
        if skill_cls.name in self._skills:
            raise ValueError(f"技能名冲突: {skill_cls.name}")
        self._skills[skill_cls.name] = skill_cls
        return skill_cls

    def get(self, name: str) -> type[BaseSkill] | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills.keys())

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """构造 OpenAI tools 数组。names 为 None 时返回所有技能。"""
        selected = names or self.names()
        tools = []
        for n in selected:
            cls = self._skills.get(n)
            if cls is None:
                raise KeyError(f"未知技能: {n}")
            tools.append(cls.to_openai_tool())
        return tools

    def invoke(self, name: str, args: dict[str, Any], ctx: LoopContext) -> SkillResult:
        """将 LLM 发来的单次 tool_call 调度给对应技能。"""
        cls = self._skills.get(name)
        if cls is None:
            return SkillResult(success=False, reasoning=f"未知技能: {name}")
        return cls().invoke(args, ctx)


# 模块级单例
registry = SkillRegistry()


def register(skill_cls: type[BaseSkill]) -> type[BaseSkill]:
    """在 import 时声明一个技能的装饰器。"""
    return registry.register(skill_cls)
