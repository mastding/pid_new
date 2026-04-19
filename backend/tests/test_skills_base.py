"""技能基础设施（base + registry + demo）烟雾测试。"""
from __future__ import annotations

import pytest

from core.skills import LoopContext, registry


def test_registry_has_demo_skill():
    """注册表应包含示例技能。"""
    assert "_demo_echo" in registry.names()


def test_to_openai_tool_schema_shape():
    """生成的 OpenAI tool schema 结构应符合预期。"""
    tools = registry.to_openai_tools(["_demo_echo"])
    assert len(tools) == 1
    t = tools[0]
    assert t["type"] == "function"
    assert t["function"]["name"] == "_demo_echo"
    params = t["function"]["parameters"]
    assert params["type"] == "object"
    assert "message" in params["properties"]
    assert "repeat" in params["properties"]
    assert "message" in params["required"]


def test_invoke_success():
    """正常调用：返回 success=True，并写入审计日志。"""
    ctx = LoopContext(csv_path="/fake.csv", loop_prefix="L1")
    result = registry.invoke("_demo_echo", {"message": "hi", "repeat": 3}, ctx)
    assert result.success is True
    assert result.data["echoed"] == "hi hi hi"
    assert result.data["loop_prefix"] == "L1"
    # 审计日志已记录
    assert len(ctx.skill_log) == 1
    assert ctx.skill_log[0]["skill"] == "_demo_echo"
    assert ctx.skill_log[0]["success"] is True


def test_invoke_validation_failure():
    """非法入参不应抛异常，而是返回 success=False。"""
    ctx = LoopContext(csv_path="/fake.csv")
    # repeat=999 违反 Field(le=10) 约束
    result = registry.invoke("_demo_echo", {"message": "x", "repeat": 999}, ctx)
    assert result.success is False
    assert "参数校验失败" in result.reasoning


def test_invoke_unknown_skill():
    """调用未注册的技能名应返回 success=False。"""
    ctx = LoopContext(csv_path="/fake.csv")
    result = registry.invoke("nonexistent_skill", {}, ctx)
    assert result.success is False
    assert "未知技能" in result.reasoning


def test_to_llm_dict_compact():
    """空字段不应出现在传给 LLM 的字典中（节省 token）。"""
    from core.skills import SkillResult
    r = SkillResult(success=True)
    assert r.to_llm_dict() == {"success": True}

    r2 = SkillResult(success=True, data={"x": 1}, warnings=["w"])
    d = r2.to_llm_dict()
    assert "data" in d and "warnings" in d
    assert "reasoning" not in d  # 空字符串被省略


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
