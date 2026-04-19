"""调试用示例技能 —— 用于端到端验证注册表链路。

不在生产环境中使用，真实技能就位后可删除。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.skills.base import BaseSkill, LoopContext, SkillResult
from core.skills.registry import register


class _EchoInputs(BaseModel):
    message: str = Field(..., description="要回显的内容")
    repeat: int = Field(1, ge=1, le=10, description="重复次数")


@register
class EchoSkill(BaseSkill):
    name = "_demo_echo"
    description = "调试用回显技能：把输入字符串重复 N 次返回。仅用于注册表链路测试，生产环境不调用。"
    input_model = _EchoInputs

    def run(self, inputs: _EchoInputs, ctx: LoopContext) -> SkillResult:
        echoed = (inputs.message + " ") * inputs.repeat
        return SkillResult(
            success=True,
            data={"echoed": echoed.strip(), "loop_prefix": ctx.loop_prefix},
            reasoning=f"已回显 {inputs.repeat} 次",
        )
