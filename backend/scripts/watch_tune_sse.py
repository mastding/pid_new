"""命令行直观看 /api/tune/stream 的 SSE 事件流。

用法（在 backend 目录下）：
    python -m scripts.watch_tune_sse              # 默认用项目根 data.csv
    python -m scripts.watch_tune_sse path\to.csv  # 指定 CSV
    python -m scripts.watch_tune_sse path\to.csv --no-llm  # 关 LLM 顾问对比

输出会突出 window_selection 阶段的 LLM 选择结果。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_CSV = "D:/code/pid_v2/data.csv"
API_URL = "http://localhost:4444/api/tune/stream"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=DEFAULT_CSV)
    ap.add_argument("--no-llm", action="store_true", help="关闭 LLM 顾问，跑确定性 baseline")
    ap.add_argument("--loop-type", default="flow")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ 找不到 CSV: {csv_path}")
        return 1

    print(f"→ POST {API_URL}")
    print(f"  csv = {csv_path}")
    print(f"  use_llm_advisor = {not args.no_llm}")
    print("=" * 70)

    with open(csv_path, "rb") as f:
        files = {"file": (csv_path.name, f, "text/csv")}
        data = {
            "loop_type": args.loop_type,
            "use_llm_advisor": "false" if args.no_llm else "true",
        }
        resp = requests.post(API_URL, files=files, data=data, stream=True, timeout=600)
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                print(f"  [非法 JSON] {payload[:120]}")
                continue

            etype = ev.get("type")
            if etype == "stage":
                stage = ev.get("stage", "")
                status = ev.get("status", "")
                d = ev.get("data", {}) or {}
                marker = ""
                if stage == "window_selection" and status == "done":
                    mode = d.get("mode", "?")
                    marker = f"  ★ 选窗模式={mode}, chosen={d.get('chosen_index')}, deterministic={d.get('deterministic_index')}"
                    if mode == "llm":
                        agree = d.get("agreed_with_deterministic")
                        marker += f", 与 baseline 一致={agree}, 思考链={d.get('llm_reasoning_chain_len',0)}字"
                print(f"[{stage:18s} {status:7s}] {json.dumps({k:v for k,v in d.items() if k!='reasoning'}, ensure_ascii=False)[:300]}")
                if marker:
                    print(marker)
                if "reasoning" in d:
                    print(f"   理由: {d['reasoning']}")
            elif etype == "result":
                ws = ev["data"].get("window_selection", {})
                model = ev["data"].get("model", {})
                pid = ev["data"].get("pid_params", {})
                ev2 = ev["data"].get("evaluation", {})
                print()
                print("=" * 70)
                print("【最终结果】")
                print(f"  选窗: mode={ws.get('mode')}, chosen={ws.get('chosen_index')}")
                print(f"  模型: {model.get('model_type')}, K={model.get('K'):.4f}, T={model.get('T'):.2f}, L={model.get('L')}, r²={model.get('r2_score'):.4f}, conf={model.get('confidence'):.3f}")
                print(f"  PID:  Kp={pid.get('Kp'):.3f}, Ki={pid.get('Ki'):.3f}, Kd={pid.get('Kd'):.3f} ({pid.get('strategy')})")
                print(f"  评估: passed={ev2.get('passed')}, score={ev2.get('performance_score')}, rating={ev2.get('final_rating')}, 超调={ev2.get('overshoot_percent'):.2f}%")
            elif etype == "error":
                print(f"❌ ERROR [{ev.get('stage')}] {ev.get('message')}")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
