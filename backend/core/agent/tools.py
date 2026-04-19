"""Tool definitions for the PID consultant agent.

Each tool wraps a deterministic computation from core.algorithms.
The LLM decides when and how to call these tools during iterative tuning.
"""
from __future__ import annotations

from typing import Any

# OpenAI function-calling tool schema definitions
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_identification",
            "description": "对指定数据窗口运行系统辨识，返回过程模型参数 (K, T, L) 和拟合质量",
            "parameters": {
                "type": "object",
                "properties": {
                    "window_index": {
                        "type": "integer",
                        "description": "候选窗口序号 (从 0 开始)，不传则使用默认最优窗口",
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["FO", "FOPDT", "SOPDT", "IPDT"],
                        "description": "指定模型类型，不传则自动选择最优",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tuning",
            "description": "基于过程模型计算 PID 参数，可指定整定策略",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["IMC", "LAMBDA", "ZN", "CHR", "AUTO"],
                        "description": "整定策略，AUTO 为自动选择",
                    },
                    "lambda_factor": {
                        "type": "number",
                        "description": "Lambda 策略的 λ 系数 (相对于 T)，越大越保守",
                    },
                    "kp_scale": {
                        "type": "number",
                        "description": "Kp 缩放因子，用于微调 (如 0.8 表示减小 20%)",
                    },
                    "ki_scale": {
                        "type": "number",
                        "description": "Ki 缩放因子",
                    },
                    "kd_scale": {
                        "type": "number",
                        "description": "Kd 缩放因子",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_evaluation",
            "description": "评估 PID 参数的闭环控制性能，返回超调、调节时间、稳定性等指标",
            "parameters": {
                "type": "object",
                "properties": {
                    "Kp": {"type": "number", "description": "比例增益"},
                    "Ki": {"type": "number", "description": "积分增益"},
                    "Kd": {"type": "number", "description": "微分增益"},
                },
                "required": ["Kp", "Ki", "Kd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_overview",
            "description": "获取当前数据的概览信息，包括候选窗口、采样时间等",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_experience",
            "description": "搜索历史整定经验，寻找类似回路的整定记录",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词 (如回路名称、装置类型、工况)",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
