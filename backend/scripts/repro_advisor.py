"""连续多次跑同一个 CSV，验证 LLM 选窗的稳定性 / 不确定性。"""
from __future__ import annotations

import json
import sys
import time

import requests

URL = "http://localhost:4444/api/tune/stream"
CSV = "D:/PID整定/2026-03-03.csv"
N = 3

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def one_trial(trial: int) -> dict:
    t0 = time.time()
    chosen = "-"
    mode = "-"
    r2 = conf = None
    passed = "-"
    reasoning = ""
    err = ""
    with open(CSV, "rb") as f:
        r = requests.post(
            URL,
            files={"file": ("x.csv", f, "text/csv")},
            data={"loop_type": "flow", "use_llm_advisor": "true"},
            stream=True,
            timeout=300,
        )
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if ev.get("stage") == "window_selection" and ev.get("status") == "done":
                d = ev.get("data", {})
                mode = d.get("mode")
                chosen = d.get("chosen_index")
                reasoning = (d.get("reasoning") or "")[:120]
            if ev.get("stage") == "identification" and ev.get("status") == "done":
                d = ev.get("data", {})
                r2 = d.get("r2_score")
                conf = d.get("confidence")
            if ev.get("stage") == "evaluation" and ev.get("status") == "done":
                passed = ev.get("data", {}).get("passed")
            if ev.get("type") == "error":
                err = ev.get("error_code", "ERROR")
                passed = err
                break
    return {
        "trial": trial,
        "mode": mode,
        "chosen": chosen,
        "r2": r2,
        "conf": conf,
        "passed": passed,
        "wall": time.time() - t0,
        "reasoning": reasoning,
    }


def main():
    print(f"连发 {N} 次 /api/tune/stream，CSV={CSV}, use_llm_advisor=true")
    print("-" * 80)
    rows = []
    for i in range(1, N + 1):
        row = one_trial(i)
        rows.append(row)
        r2s = f"{row['r2']:.3f}" if row["r2"] is not None else "-"
        cs = f"{row['conf']:.3f}" if row["conf"] is not None else "-"
        print(f"#{row['trial']} mode={row['mode']:<22s} chosen={row['chosen']!s:>3} "
              f"r2={r2s:>7} conf={cs:>6} passed={str(row['passed']):>6} {row['wall']:>5.1f}s")
        print(f"   理由: {row['reasoning']}")
    chosen_set = sorted({r["chosen"] for r in rows})
    print()
    print(f"5 次中 LLM 选过的窗口集合: {chosen_set}")
    print(f"成功次数: {sum(1 for r in rows if r['passed'] is True)} / {N}")


if __name__ == "__main__":
    main()
