"""DeepSeek-Reasoner 鍑芥暟璋冪敤杩為€氭€х儫闆炬祴璇曪紙Day 3 椋庨櫓楠岃瘉鐐癸級銆?
鐩殑锛?  鍦ㄥ啓鐪熸鐨?LLM 闆嗘垚锛圖ay 4锛変箣鍓嶏紝鍏堝疄娴?deepseek-reasoner锛圧1锛夋槸鍚︾ǔ瀹氭敮鎸?  OpenAI function calling銆傚涓嶆敮鎸侊紝楠岃瘉銆宺easoner 瑙勫垝 + 鏈湴 dispatch銆嶅洖閫€璺緞銆?
鐢ㄦ硶锛?    cd backend && python -m scripts.smoke_reasoner

杈撳嚭锛?  - 缁堢鎵撳嵃鏋舵瀯鎺ㄨ崘锛圓 绾?reasoner / B 鍙屾ā鍨嬪洖閫€锛?  - 鍏ㄩ噺 transcript 钀界洏鍒?backend/scripts/smoke_results/<timestamp>.json
  - 閫€鍑虹爜锛? = 鑷冲皯涓€绉嶆ā寮忛€氳繃锛? = 鍏ㄩ儴澶辫触
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

# Windows 鎺у埗鍙伴粯璁?GBK锛屽己鍒?UTF-8 浠ラ伩鍏?鉁?鉁?绛夌鍙风紪鐮佸け璐?try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# 鈹€鈹€ 閰嶇疆 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

CSV_PATH = str(Path(__file__).resolve().parent.parent.parent / "data.csv")
SKILL_NAMES = ["load_dataset", "summarize_data", "detect_windows"]
LLM_TIMEOUT_SEC = 90.0
MAX_ITERATIONS = 8
RESULTS_DIR = Path(__file__).resolve().parent / "smoke_results"

SYSTEM_PROMPT = (
    "浣犳槸 PID 鎺у埗鍥炶矾鏁村畾涓撳銆傚綋鍓嶄换鍔★細鍒嗘瀽鐢ㄦ埛涓婁紶鐨勮繍琛屾暟鎹?CSV锛岀粰鍑烘暣瀹氭€濊矾寤鸿銆俓n"
    "\n"
    "浣犲彲浠ヨ皟鐢ㄤ互涓嬪伐鍏凤紙鍧囦负鍚庣纭畾鎬х畻娉曪級锛歕n"
    "  - load_dataset锛氱涓€姝ュ繀璋冦€傚姞杞藉苟娓呮礂 CSV锛堣嚜鍔ㄨВ鏋愭椂闂存埑銆佸綊涓€鍖栧垪鍚嶃€丳V 鍘诲櫔锛夈€俓n"
    "  - summarize_data锛歭oad 涔嬪悗璋冪敤銆傜敓鎴?7 缁存暟鎹敾鍍忥紙PV 閲忕▼/MV 楗卞拰/姝诲尯/鍣０/鎸崱/骞叉壈/鍥炶矾绫诲瀷锛夈€俓n"
    "  - detect_windows锛氬彲閫夈€傚湪鏁版嵁鐢诲儚涔嬪悗璋冪敤锛屽垪鍑哄彲鐢ㄤ簬杈ㄨ瘑鐨勫€欓€夋暟鎹獥鍙ｃ€俓n"
    "\n"
    "绾﹀畾锛歕n"
    "  - 姣忔鑷冲璋冪敤涓€涓伐鍏凤紝绛夌粨鏋滆繑鍥炲悗鍐嶅喅瀹氫笅涓€姝ャ€俓n"
    "  - 璋冪敤椤哄簭鍘熷垯涓婃槸 load_dataset 鈫?summarize_data 鈫?detect_windows銆俓n"
    "  - 鎷垮埌鍏ㄩ儴淇℃伅鍚庯紝鐢ㄤ腑鏂囩粰鍑烘暣瀹氭€濊矾寤鸿锛堝寘鍚細鏁版嵁璐ㄩ噺璇勪环銆佹帹鏂殑鍥炶矾绫诲瀷銆?
    "    杈ㄨ瘑绐楀彛寤鸿銆佹槸鍚﹂渶瑕侀澶勭悊鏀硅繘绛夛級銆傚洖绛斿姟蹇呯揣鎵?summarize_data 杩斿洖鐨勫叿浣撴暟鍊笺€俓n"
)

USER_MESSAGE = (
    f"璇峰垎鏋愯繖浠?PID 杩愯鏁版嵁骞剁粰鍑烘暣瀹氭€濊矾寤鸿銆侰SV 璺緞宸插湪绯荤粺涓璁撅紝"
    f"浣犲彧闇€渚濇璋冪敤宸ュ叿鍗冲彲锛屾棤闇€鎻愪緵鏂囦欢璺緞銆?
)


# 鈹€鈹€ 宸ュ叿鍑芥暟 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _make_client() -> OpenAI:
    return OpenAI(api_key=settings.model_api_key, base_url=settings.model_api_url)


def _short(obj: Any, limit: int = 500) -> str:
    """鎶婂璞″帇鎴愬瓧绗︿覆骞舵埅鏂紝闃叉缁堢鍒峰睆銆?""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > limit:
        s = s[:limit] + f"... ({len(s)} chars total)"
    return s


def _print_section(title: str) -> None:
    print(f"\n{'鈺? * 60}\n  {title}\n{'鈺? * 60}")


# 鈹€鈹€ 妯″紡 A锛氱函 deepseek-reasoner + tools 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def probe_pure_reasoner(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """鐩存帴鐢?reasoner + function calling锛岀湅鑳藉惁绔埌绔窇閫氥€?""
    _print_section("妯″紡 A锛氱函 deepseek-reasoner + tools")

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
        print(f"\n--- 绗?{it} 杞?---")
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
            print(f"  鉁?API 璋冪敤澶辫触锛歿err}")
            transcript.append({"mode": "A", "iter": it, "error": err})
            return {
                "ok": False,
                "reason": f"API 閿欒锛歿err}",
                "iterations": it,
                "tool_calls": tool_call_count,
            }

        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            print(f"  [鎬濊€冮摼 {len(reasoning)} 瀛梋: {reasoning[:200]}...")
        if msg.content:
            print(f"  [鍥炵瓟]: {msg.content[:300]}")

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
            print(f"  [鏃犲伐鍏疯皟鐢?鈫?缁堟] 鐢ㄦ椂 {dt:.1f}s")
            final_text = msg.content or ""
            break

        # 鎶?assistant 娑堟伅杩藉姞鍒颁笂涓嬫枃
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

        # 鎵ц姣忎釜 tool_call
        for tc in msg.tool_calls:
            tool_call_count += 1
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                json_decode_failures += 1
                args = {}
                print(f"  鈿?JSON 瑙ｆ瀽澶辫触 ({json_decode_failures}/2 瀹瑰繊): {exc}")
                if json_decode_failures > 2:
                    return {
                        "ok": False,
                        "reason": "tool_call.arguments JSON 鍙嶅瑙ｆ瀽澶辫触",
                        "iterations": it,
                        "tool_calls": tool_call_count,
                    }

            print(f"  鈫?璋冪敤 {name}({_short(args, 200)})")
            result = registry.invoke(name, args, ctx)
            result_dict = result.to_llm_dict()
            print(f"    杩斿洖 success={result.success}: {_short(result_dict, 300)}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result_dict, ensure_ascii=False, default=str),
            })

    # 楠岃瘉锛氭槸鍚︾湡鐨勮皟鐢ㄤ簡鍏抽敭宸ュ叿
    called_skills = set()
    for entry in transcript:
        if entry.get("mode") == "A":
            for tc in entry.get("tool_calls", []):
                called_skills.add(tc["name"])

    if "load_dataset" not in called_skills:
        return {
            "ok": False,
            "reason": "LLM 鏈皟鐢?load_dataset 鍏抽敭宸ュ叿",
            "iterations": it,
            "tool_calls": tool_call_count,
            "called_skills": sorted(called_skills),
        }

    if not final_text or len(final_text) < 50:
        return {
            "ok": False,
            "reason": "LLM 鏈骇鍑烘湁鎰忎箟鐨勬渶缁堝洖绛?,
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


# 鈹€鈹€ 妯″紡 B锛歳easoner 瑙勫垝 + 鏈湴 dispatch 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

PLANNING_PROMPT = (
    "浣犳槸 PID 鏁村畾涓撳锛屾鍦ㄤ互銆岃鍒掕€呫€嶈韩浠藉伐浣溿€傛瘡涓€姝ヤ綘鍙兘杈撳嚭涓ユ牸鐨?JSON锛?
    "鏍煎紡涓猴細{\"action\": \"<宸ュ叿鍚嶆垨finish>\", \"args\": {...}, \"reason\": \"涓枃鐞嗙敱\"}銆俓n"
    "\n"
    "鍙敤宸ュ叿锛歕n"
    "  - load_dataset(loop_prefix?: string, start_time?: string, end_time?: string)\n"
    "  - summarize_data()\n"
    "  - detect_windows()\n"
    "  - finish(answer: string)锛氬綋淇℃伅鍏呭垎鏃惰皟鐢紝answer 涓虹粰鐢ㄦ埛鐨勪腑鏂囨暣瀹氬缓璁€俓n"
    "\n"
    "璋冪敤椤哄簭鍘熷垯锛歭oad_dataset 鈫?summarize_data 鈫?detect_windows 鈫?finish銆俓n"
    "\n"
    "**鍙緭鍑?JSON锛屼笉瑕佷换浣曞墠鍚庣紑鏂囧瓧銆佷笉瑕?markdown 浠ｇ爜鍧椼€佷笉瑕佽В閲娿€?*"
)


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict[str, Any] | None:
    """浠?reasoner 鍥炵瓟涓娊鍙?JSON锛屽蹇嶅皯閲忓墠鍚庣紑銆?""
    if not text:
        return None
    # 浼樺厛灏濊瘯鏁翠綋瑙ｆ瀽
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 閫€鑰屾眰鍏舵锛氬尮閰嶇涓€涓ぇ鎷彿鍧?    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def probe_dual_model(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """reasoner 杈撳嚭缁撴瀯鍖?JSON 瑙勫垝锛屾湰鍦?dispatch 璋?skill銆?""
    _print_section("妯″紡 B锛歳easoner 瑙勫垝 + 鏈湴 dispatch")

    client = _make_client()
    ctx = LoopContext(csv_path=CSV_PATH)

    history: list[dict[str, Any]] = []  # 璁板綍涔嬪墠鐨勶紙action, result锛夊锛屽杺缁欎笅涓€杞?reasoner

    final_answer = ""
    parse_failures = 0

    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- 绗?{it} 杞?---")

        # 鏋勯€犵粰 reasoner 鐨勬彁绀?        user_parts = [
            f"浠诲姟锛氬垎鏋?PID 杩愯鏁版嵁骞剁粰鍑烘暣瀹氬缓璁€?,
            f"宸叉墽琛屾楠ゆ憳瑕侊細",
        ]
        if not history:
            user_parts.append("锛堝皻鏈墽琛屼换浣曞姩浣滐級")
        else:
            for i, h in enumerate(history, 1):
                user_parts.append(
                    f"  {i}. action={h['action']}, success={h['success']}, "
                    f"data_summary={_short(h['result_summary'], 300)}"
                )
        user_parts.append("\n璇疯緭鍑轰笅涓€姝ュ姩浣滅殑 JSON銆?)
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
            print(f"  鉁?API 璋冪敤澶辫触锛歿err}")
            transcript.append({"mode": "B", "iter": it, "error": err})
            return {"ok": False, "reason": f"API 閿欒锛歿err}", "iterations": it}

        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        raw_text = msg.content or ""

        if reasoning:
            print(f"  [鎬濊€冮摼 {len(reasoning)} 瀛梋")
        print(f"  [杈撳嚭]: {raw_text[:400]}")
        print(f"  [鐢ㄦ椂]: {dt:.1f}s")

        plan = _extract_json(raw_text)
        if not plan or "action" not in plan:
            parse_failures += 1
            print(f"  鈿?JSON 瑙ｆ瀽澶辫触 ({parse_failures}/2 瀹瑰繊)")
            transcript.append({
                "mode": "B", "iter": it, "elapsed_sec": round(dt, 2),
                "raw_text": raw_text, "parse_failed": True,
            })
            if parse_failures > 2:
                return {
                    "ok": False,
                    "reason": "reasoner 杈撳嚭 JSON 鍙嶅瑙ｆ瀽澶辫触",
                    "iterations": it,
                }
            history.append({
                "action": "(unparseable)",
                "success": False,
                "result_summary": "涓婅疆浣犵殑杈撳嚭涓嶆槸鍚堟硶 JSON锛岃閲嶈瘯銆?,
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

        # 缁堟鍔ㄤ綔
        if action == "finish":
            final_answer = args.get("answer", "") or reason
            print(f"  鉁?finish: {final_answer[:300]}")
            break

        # 鏅€氭妧鑳借皟鐢?        if action not in SKILL_NAMES:
            print(f"  鈿?鏈煡 action: {action}")
            history.append({
                "action": action,
                "success": False,
                "result_summary": f"鏈煡宸ュ叿鍚?{action}锛屽彲鐢ㄥ伐鍏蜂粎锛歿', '.join(SKILL_NAMES)}, finish銆?,
            })
            continue

        result = registry.invoke(action, args, ctx)
        rd = result.to_llm_dict()
        print(f"  鈫?{action}({_short(args, 150)}) success={result.success}")
        print(f"    杩斿洖: {_short(rd, 300)}")
        history.append({
            "action": action,
            "success": result.success,
            "result_summary": rd,
        })

    # 鏀跺熬
    if not final_answer:
        return {
            "ok": False,
            "reason": "鏈湪杩唬涓婇檺鍐呬骇鍑?finish 鍔ㄤ綔",
            "iterations": it,
            "history_len": len(history),
        }

    if len(final_answer) < 50:
        return {
            "ok": False,
            "reason": f"鏈€缁堝洖绛旇繃鐭紙{len(final_answer)} 瀛楋級",
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


# 鈹€鈹€ 涓诲叆鍙?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main() -> int:
    print(f"DeepSeek 鐑熼浘娴嬭瘯")
    print(f"  CSV: {CSV_PATH}")
    print(f"  endpoint: {settings.model_api_url}")
    print(f"  configured model: {settings.model_name}")
    print(f"  鏆撮湶鎶€鑳? {SKILL_NAMES}")

    if not Path(CSV_PATH).exists():
        print(f"\n鉁?CSV 涓嶅瓨鍦? {CSV_PATH}")
        return 1
    if not settings.model_api_key or not settings.model_api_url:
        print(f"\n鉁?.env 涓?MODEL_API_KEY / MODEL_API_URL 鏈厤缃?)
        return 1

    transcript: list[dict[str, Any]] = []

    # 妯″紡 A
    try:
        result_a = probe_pure_reasoner(transcript)
    except Exception as exc:
        result_a = {"ok": False, "reason": f"鏈崟鑾峰紓甯? {exc}", "trace": traceback.format_exc()}
        print(f"\n  妯″紡 A 鎶涘嚭鏈崟鑾峰紓甯?\n{result_a['trace']}")

    # 妯″紡 B锛堜粎 A 澶辫触鏃惰窇锛岄伩鍏嶆氮璐?token锛?    result_b: dict[str, Any] | None = None
    if not result_a.get("ok"):
        try:
            result_b = probe_dual_model(transcript)
        except Exception as exc:
            result_b = {"ok": False, "reason": f"鏈崟鑾峰紓甯? {exc}", "trace": traceback.format_exc()}
            print(f"\n  妯″紡 B 鎶涘嚭鏈崟鑾峰紓甯?\n{result_b['trace']}")

    # 钀界洏
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

    # 鎬荤粨
    _print_section("楠岃瘉缁撴灉")
    a_status = "鉁?PASS" if result_a.get("ok") else "鉁?FAIL"
    a_iter = result_a.get("iterations")
    a_tools = result_a.get("tool_calls")
    a_reason = result_a.get("reason", f"杞={a_iter}, 宸ュ叿璋冪敤={a_tools}")
    print(f"[A] pure_reasoner: {a_status}  ({a_reason})")
    if result_b is not None:
        b_status = "鉁?PASS" if result_b.get("ok") else "鉁?FAIL"
        b_iter = result_b.get("iterations")
        b_reason = result_b.get("reason", f"杞={b_iter}")
        print(f"[B] dual_model:    {b_status}  ({b_reason})")
    else:
        print("[B] dual_model:    (鏈窇锛孉 宸查€氳繃)")

    if result_a.get("ok"):
        recommended = "A (绾?reasoner)"
    elif result_b and result_b.get("ok"):
        recommended = "B (鍙屾ā鍨嬪洖閫€锛歳easoner 瑙勫垝 + 鏈湴 dispatch)"
    else:
        recommended = "鏃狅紙涓ゆ潯璺緞閮藉け璐ワ紝闇€杩涗竴姝ユ帓鏌ワ級"
    print(f"\n>>> 鎺ㄨ崘鏋舵瀯: {recommended}")
    print(f">>> 璇︾粏 transcript: {out_path}")

    return 0 if (result_a.get("ok") or (result_b and result_b.get("ok"))) else 1


if __name__ == "__main__":
    sys.exit(main())
