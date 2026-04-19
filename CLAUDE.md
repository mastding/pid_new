# PID V2 - Intelligent Tuning System

## Architecture

Two-layer architecture: **deterministic pipeline** (pure Python computation) + **LLM consultant** (optional, for review/iteration/explanation).

- `backend/core/algorithms/` — Pure computation, no LLM, no framework. Testable independently.
- `backend/core/pipeline/runner.py` — Sequential orchestration with SSE events. ~100 lines.
- `backend/core/agent/consultant.py` — Single LLM agent with tool-calling loop. ~80 lines. No framework.
- `backend/api/` — FastAPI REST endpoints.
- `frontend/` — React 18 + Ant Design Pro + TypeScript SPA.

## Tech Stack

- Backend: Python 3.11+, FastAPI, NumPy, SciPy, Pandas, OpenAI SDK (for Qwen/DashScope)
- Frontend: React 18, TypeScript, Ant Design Pro, Vite, @ant-design/charts
- LLM: Qwen (via DashScope OpenAI-compatible API)
- No agent framework (AutoGen, LangChain, etc.)

## Key Conventions

- **语言规约（强制）**：所有代码文件中的注释、docstring、字符串字面量、日志、错误消息、LLM 提示词（system prompt / tool description / 用户可见提示）一律使用中文。变量名、函数名、类名、文件名仍用英文 snake_case。新增或修改文件时遵循此规则；遇到旧文件中的英文文本，做改动时顺手翻译为中文（见 Karpathy 第 3 条「surgical changes」原则，不要为了翻译而大规模改动无关代码）。
- Loop types: flow, pressure, temperature, level
- Model types: FO, FOPDT, SOPDT, IPDT
- All algorithms in `core/algorithms/` must be pure functions (no side effects, no global state)
- Pydantic models in `backend/models/` for all API contracts
- Frontend uses `@/` path alias for `src/`

## Migration Status

Core algorithms are being migrated from `D:/code/pid_new/` with optimizations:
- See README.md for the 14 identified issues and their fixes
- Each algorithm file has TODO comments listing specific optimizations to apply

## External Services (preserved from pid_new)

- History Data API: Hollysys PID Agent
- Knowledge Graph API: GraphRAG service
- Config via .env (see .env.example)

---

# Karpathy Guidelines

## 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

## 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

## 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

## 4. Goal-Driven Execution
Define success criteria. Loop until verified.
