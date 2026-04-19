"""DeepSeek-Reasoner 函数调用连通性烟雾测试（Day 3 风险验证点）。

目的：
  在写真正的 LLM 集成（Day 4）之前，先实测 deepseek-reasoner（R1）是否稳定支持
  OpenAI function calling。如不支持，验证「reasoner 规划 + 本地 dispatch」回退路径。

用法：
    cd backend && python -m scripts.smoke_reasoner

输出：
  - 终端打印架构推荐（A 纯 reasoner / B 双模型回退）
  - 全量 transcript 落盘到 backend/scripts/smoke_results/<timestamp>.json
  - 退出码：0 = 至少一种模式通过，1 = 全部失败
"""
from __future__ import annotations

import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import settings
from core.skills import LoopContext, registry

# Windows 控制台默认 GBK，强制 UTF-8 以避免 ✓/✗ 等符号编码失败
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# ── 配置 ─────────────────────────────────────────────────────────────────

CSV_PATH = str(Path(__file__).resolve().parent.parent.parent / "data.csv")
SKILL_NAMES = ["load_dataset", "summarize_data", "detect_candidate_windows"]
LLM_TIMEOUT_SEC = 90.0
MAX_ITERATIONS = 8
RESULTS_DIR = Path(__file__).resolve().parent / "smoke_results"

SYSTEM_PROMPT = (
    "你是 PID 控制回路整定专家。当前任务：分析用户上传的运行数据 CSV，给出整定思路建议。\n"
    "\n"
    "你可以调用以下工具（均为后端确定性算法）：\n"
    "  - load_dataset：第一步必调。加载并清洗 CSV（自动解析时间戳、归一化列名、PV 去噪）。\n"
    "  - summarize_data：load 之后调用。生成 7 维数据画像（PV 量程/MV 饱和/死区/噪声/振荡/干扰/回路类型）。\n"
    "  - detect_candidate_windows：可选。在数据画像之后调用，列出可用于辨识的候选数据窗口。\n"
    "\n"
    "约定：\n"
    "  - 每次至多调用一个工具，等结果返回后再决定下一步。\n"
    "  - 调用顺序原则上是 load_dataset → summarize_data → detect_candidate_windows。\n"
    "  - 拿到全部信息后，用中文给出整定思路建议（包含：数据质量评价、推断的回路类型、"
    "    辨识窗口建议、是否需要预处理改进等）。回答务必紧扣 summarize_data 返回的具体数值。\n"
)

USER_MESSAGE = (
    f"请分析这份 PID 运行数据并给出整定思路建议。CSV 路径已在系统中预设，"
    f"你只需依次调用工具即可，无需提供文件路径。"
)


# ── 工具函数 ─────────────────────────────────────────────────────────────

def _make_client() -> OpenAI:
    return OpenAI(api_key=settings.model_api_key, base_url=settings.model_api_url)


def _short(obj: Any, limit: int = 500) -> str:
    """把对象压成字符串并截断，防止终端刷屏。"""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > limit:
        s = s[:limit] + f"... ({len(s)} chars total)"
    return s


def _print_section(title: str) -> None:
    print(f"\n{'═' * 60}\n  {title}\n{'═' * 60}")


# ── 模式 A：纯 deepseek-reasoner + tools ───────────────────────────────

def probe_pure_reasoner(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """直接用 reasoner + function calling，看能否端到端跑通。"""
    _print_section("模式 A：纯 deepseek-reasoner + tools")

    client = _make_client()
    ctx = LoopContext(csv_path=CSV_PATH)
    tools = registry.to_openai_tools(SKILL_NAMES)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE},
    ]

    tool_call_count = 0
    json_decode_failures = 0
    final_text = ""

    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- 第 {it} 轮 ---")
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                tools=tools,
                timeout=LLM_TIMEOUT_SEC,
            )
            dt = time.time() - t0
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"  ✗ API 调用失败：{err}")
            transcript.append({"mode": "A", "iter": it, "error": err})
            return {
                "ok": False,
                "reason": f"API 错误：{err}",
                "iterations": it,
                "tool_calls": tool_call_count,
            }

        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            print(f"  [思考链 {len(reasoning)} 字]: {reasoning[:200]}...")
        if msg.content:
            print(f"  [回答]: {msg.content[:300]}")

        transcript.append({
            "mode": "A",
            "iter": it,
            "elapsed_sec": round(dt, 2),
            "reasoning_content": reasoning,
            "content": msg.content,
            "tool_calls": [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ],
        })

        if not msg.tool_calls:
            print(f"  [无工具调用 → 终止] 用时 {dt:.1f}s")
            final_text = msg.content or ""
            break

        # 把 assistant 消息追加到上下文
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # 执行每个 tool_call
        for tc in msg.tool_calls:
            tool_call_count += 1
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                json_decode_failures += 1
                args = {}
                print(f"  ⚠ JSON 解析失败 ({json_decode_failures}/2 容忍): {exc}")
                if json_decode_failures > 2:
                    return {
                        "ok": False,
                        "reason": "tool_call.arguments JSON 反复解析失败",
                        "iterations": it,
                        "tool_calls": tool_call_count,
                    }

            print(f"  → 调用 {name}({_short(args, 200)})")
            result = registry.invoke(name, args, ctx)
            result_dict = result.to_llm_dict()
            print(f"    返回 success={result.success}: {_short(result_dict, 300)}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result_dict, ensure_ascii=False, default=str),
            })

    # 验证：是否真的调用了关键工具
    called_skills = set()
    for entry in transcript:
        if entry.get("mode") == "A":
            for tc in entry.get("tool_calls", []):
                called_skills.add(tc["name"])

    if "load_dataset" not in called_skills:
        return {
            "ok": False,
            "reason": "LLM 未调用 load_dataset 关键工具",
            "iterations": it,
            "tool_calls": tool_call_count,
            "called_skills": sorted(called_skills),
        }

    if not final_text or len(final_text) < 50:
        return {
            "ok": False,
            "reason": "LLM 未产出有意义的最终回答",
            "iterations": it,
            "tool_calls": tool_call_count,
            "final_text_len": len(final_text),
        }

    return {
        "ok": True,
        "iterations": it,
        "tool_calls": tool_call_count,
        "called_skills": sorted(called_skills),
        "final_text_len": len(final_text),
        "final_text": final_text,
    }


# ── 模式 B：reasoner 规划 + 本地 dispatch ─────────────────────────────

PLANNING_PROMPT = (
    "你是 PID 整定专家，正在以「规划者」身份工作。每一步你只能输出严格的 JSON，"
    "格式为：{\"action\": \"<工具名或finish>\", \"args\": {...}, \"reason\": \"中文理由\"}。\n"
    "\n"
    "可用工具：\n"
    "  - load_dataset(loop_prefix?: string, start_time?: string, end_time?: string)\n"
    "  - summarize_data()\n"
    "  - detect_candidate_windows()\n"
    "  - finish(answer: string)：当信息充分时调用，answer 为给用户的中文整定建议。\n"
    "\n"
    "调用顺序原则：load_dataset → summarize_data → detect_candidate_windows → finish。\n"
    "\n"
    "**只输出 JSON，不要任何前后缀文字、不要 markdown 代码块、不要解释。**"
)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict[str, Any] | None:
    """从 reasoner 回答中抽取 JSON，容忍少量前后缀。"""
    if not text:
        return None
    # 优先尝试整体解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退而求其次：匹配第一个大括号块
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def probe_dual_model(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """reasoner 输出结构化 JSON 规划，本地 dispatch 调 skill。"""
    _print_section("模式 B：reasoner 规划 + 本地 dispatch")

    client = _make_client()
    ctx = LoopContext(csv_path=CSV_PATH)

    history: list[dict[str, Any]] = []  # 记录之前的（action, result）对，喂给下一轮 reasoner

    final_answer = ""
    parse_failures = 0

    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- 第 {it} 轮 ---")

        # 构造给 reasoner 的提示
        user_parts = [
            f"任务：分析 PID 运行数据并给出整定建议。",
            f"已执行步骤摘要：",
        ]
        if not history:
            user_parts.append("（尚未执行任何动作）")
        else:
            for i, h in enumerate(history, 1):
                user_parts.append(
                    f"  {i}. action={h['action']}, success={h['success']}, "
                    f"data_summary={_short(h['result_summary'], 300)}"
                )
        user_parts.append("\n请输出下一步动作的 JSON。")
        user_msg = "\n".join(user_parts)

        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": PLANNING_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                timeout=LLM_TIMEOUT_SEC,
            )
            dt = time.time() - t0
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"  ✗ API 调用失败：{err}")
            transcript.append({"mode": "B", "iter": it, "error": err})
            return {"ok": False, "reason": f"API 错误：{err}", "iterations": it}

        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        raw_text = msg.content or ""

        if reasoning:
            print(f"  [思考链 {len(reasoning)} 字]")
        print(f"  [输出]: {raw_text[:400]}")
        print(f"  [用时]: {dt:.1f}s")

        plan = _extract_json(raw_text)
        if not plan or "action" not in plan:
            parse_failures += 1
            print(f"  ⚠ JSON 解析失败 ({parse_failures}/2 容忍)")
            transcript.append({
                "mode": "B", "iter": it, "elapsed_sec": round(dt, 2),
                "raw_text": raw_text, "parse_failed": True,
            })
            if parse_failures > 2:
                return {
                    "ok": False,
                    "reason": "reasoner 输出 JSON 反复解析失败",
                    "iterations": it,
                }
            history.append({
                "action": "(unparseable)",
                "success": False,
                "result_summary": "上轮你的输出不是合法 JSON，请重试。",
            })
            continue

        action = plan["action"]
        args = plan.get("args", {}) or {}
        reason = plan.get("reason", "")

        transcript.append({
            "mode": "B",
            "iter": it,
            "elapsed_sec": round(dt, 2),
            "reasoning_content": reasoning,
            "plan": plan,
        })

        # 终止动作
        if action == "finish":
            final_answer = args.get("answer", "") or reason
            print(f"  ✓ finish: {final_answer[:300]}")
            break

        # 普通技能调用
        if action not in SKILL_NAMES:
            print(f"  ⚠ 未知 action: {action}")
            history.append({
                "action": action,
                "success": False,
                "result_summary": f"未知工具名 {action}，可用工具仅：{', '.join(SKILL_NAMES)}, finish。",
            })
            continue

        result = registry.invoke(action, args, ctx)
        rd = result.to_llm_dict()
        print(f"  → {action}({_short(args, 150)}) success={result.success}")
        print(f"    返回: {_short(rd, 300)}")
        history.append({
            "action": action,
            "success": result.success,
            "result_summary": rd,
        })

    # 收尾
    if not final_answer:
        return {
            "ok": False,
            "reason": "未在迭代上限内产出 finish 动作",
            "iterations": it,
            "history_len": len(history),
        }

    if len(final_answer) < 50:
        return {
            "ok": False,
            "reason": f"最终回答过短（{len(final_answer)} 字）",
            "iterations": it,
            "final_answer": final_answer,
        }

    called = sorted({h["action"] for h in history if h["success"]})
    return {
        "ok": True,
        "iterations": it,
        "called_skills": called,
        "history_len": len(history),
        "final_text_len": len(final_answer),
        "final_text": final_answer,
    }


# ── 主入口 ───────────────────────────────────────────────────────────────

def main() -> int:
    print(f"DeepSeek 烟雾测试")
    print(f"  CSV: {CSV_PATH}")
    print(f"  endpoint: {settings.model_api_url}")
    print(f"  configured model: {settings.model_name}")
    print(f"  暴露技能: {SKILL_NAMES}")

    if not Path(CSV_PATH).exists():
        print(f"\n✗ CSV 不存在: {CSV_PATH}")
        return 1
    if not settings.model_api_key or not settings.model_api_url:
        print(f"\n✗ .env 中 MODEL_API_KEY / MODEL_API_URL 未配置")
        return 1

    transcript: list[dict[str, Any]] = []

    # 模式 A
    try:
        result_a = probe_pure_reasoner(transcript)
    except Exception as exc:
        result_a = {"ok": False, "reason": f"未捕获异常: {exc}", "trace": traceback.format_exc()}
        print(f"\n  模式 A 抛出未捕获异常:\n{result_a['trace']}")

    # 模式 B（仅 A 失败时跑，避免浪费 token）
    result_b: dict[str, Any] | None = None
    if not result_a.get("ok"):
        try:
            result_b = probe_dual_model(transcript)
        except Exception as exc:
            result_b = {"ok": False, "reason": f"未捕获异常: {exc}", "trace": traceback.format_exc()}
            print(f"\n  模式 B 抛出未捕获异常:\n{result_b['trace']}")

    # 落盘
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "csv": CSV_PATH,
                "endpoint": settings.model_api_url,
                "skills": SKILL_NAMES,
            },
            "result_a": result_a,
            "result_b": result_b,
            "transcript": transcript,
        }, f, ensure_ascii=False, indent=2, default=str)

    # 总结
    _print_section("验证结果")
    a_status = "✓ PASS" if result_a.get("ok") else "✗ FAIL"
    a_iter = result_a.get("iterations")
    a_tools = result_a.get("tool_calls")
    a_reason = result_a.get("reason", f"轮次={a_iter}, 工具调用={a_tools}")
    print(f"[A] pure_reasoner: {a_status}  ({a_reason})")
    if result_b is not None:
        b_status = "✓ PASS" if result_b.get("ok") else "✗ FAIL"
        b_iter = result_b.get("iterations")
        b_reason = result_b.get("reason", f"轮次={b_iter}")
        print(f"[B] dual_model:    {b_status}  ({b_reason})")
    else:
        print("[B] dual_model:    (未跑，A 已通过)")

    if result_a.get("ok"):
        recommended = "A (纯 reasoner)"
    elif result_b and result_b.get("ok"):
        recommended = "B (双模型回退：reasoner 规划 + 本地 dispatch)"
    else:
        recommended = "无（两条路径都失败，需进一步排查）"
    print(f"\n>>> 推荐架构: {recommended}")
    print(f">>> 详细 transcript: {out_path}")

    return 0 if (result_a.get("ok") or (result_b and result_b.get("ok"))) else 1


if __name__ == "__main__":
    sys.exit(main())
