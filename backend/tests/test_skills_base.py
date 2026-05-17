from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from core.skills import BaseSkill, LoopContext, SkillResult, register, registry


class _EchoInputs(BaseModel):
    message: str = Field(...)
    repeat: int = Field(1, ge=1, le=10)


@register
class TestEchoSkill(BaseSkill):
    name = "_test_echo"
    description = "测试用回显技能"
    input_model = _EchoInputs

    def run(self, inputs: _EchoInputs, ctx: LoopContext) -> SkillResult:
        return SkillResult(
            success=True,
            data={"echoed": " ".join([inputs.message] * inputs.repeat), "loop_prefix": ctx.loop_prefix},
        )


def test_registry_has_real_skills():
    assert "_demo_echo" not in registry.names()
    assert "load_dataset" in registry.names()


def test_to_openai_tool_schema_shape():
    tools = registry.to_openai_tools(["_test_echo"])
    assert len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "_test_echo"
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert "message" in params["properties"]
    assert "repeat" in params["properties"]
    assert "message" in params["required"]


def test_invoke_success():
    ctx = LoopContext(csv_path="/fake.csv", loop_prefix="L1")
    result = registry.invoke("_test_echo", {"message": "hi", "repeat": 3}, ctx)
    assert result.success is True
    assert result.data["echoed"] == "hi hi hi"
    assert result.data["loop_prefix"] == "L1"
    assert len(ctx.skill_log) == 1
    assert ctx.skill_log[0]["skill"] == "_test_echo"
    assert ctx.skill_log[0]["success"] is True


def test_invoke_validation_failure():
    ctx = LoopContext(csv_path="/fake.csv")
    result = registry.invoke("_test_echo", {"message": "x", "repeat": 999}, ctx)
    assert result.success is False
    assert "参数校验失败" in result.reasoning


def test_invoke_unknown_skill():
    ctx = LoopContext(csv_path="/fake.csv")
    result = registry.invoke("nonexistent_skill", {}, ctx)
    assert result.success is False
    assert "未知技能" in result.reasoning


def test_to_llm_dict_compact():
    result = SkillResult(success=True)
    assert result.to_llm_dict() == {"success": True}

    result_with_payload = SkillResult(success=True, data={"x": 1}, warnings=["w"])
    payload = result_with_payload.to_llm_dict()
    assert "data" in payload and "warnings" in payload
    assert "reasoning" not in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
