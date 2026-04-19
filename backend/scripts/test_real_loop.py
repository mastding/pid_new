"""一次性测试脚本：用真实回路 CSV 跑完整 pipeline 并打印关键结果。

用法：
    cd backend && python -m scripts.test_real_loop
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline.runner import run_tuning_pipeline


CSV = r"D:\data\20260416_split_by_loop\20260416_5203_LIC_11502.csv"


async def main() -> None:
    # 先转储候选窗口
    from core.algorithms.data_analysis import load_and_prepare_dataset
    ds = load_and_prepare_dataset(csv_path=CSV)
    print(f"\n[候选窗口共 {len(ds['candidate_windows'])} 个]")
    for i, w in enumerate(ds["candidate_windows"]):
        print(f"  #{i} src={w.get('source')} usable={w.get('window_usable_for_id')} "
              f"score={w.get('window_quality_score'):.3f} "
              f"start={w.get('start_idx')} end={w.get('end_idx')} "
              f"len={w.get('end_idx')-w.get('start_idx')} "
              f"reasons={w.get('window_disqualify_reasons','')}")

    print()
    final_result: dict | None = None
    stages: list[tuple[str, str, dict]] = []

    async for ev in run_tuning_pipeline(
        csv_path=CSV,
        loop_type="level",
        use_llm_advisor=False,  # 先跑确定性流程，避免 LLM 不确定性
    ):
        t = ev.get("type")
        if t == "stage":
            stage = ev.get("stage")
            status = ev.get("status")
            data = ev.get("data") or {}
            stages.append((stage, status, data))
            print(f"[stage] {stage:20s} {status:8s} {json.dumps(data, ensure_ascii=False, default=str)[:200]}")
        elif t == "result":
            final_result = ev.get("data")
            print("[result] 收到最终结果")
        elif t == "error":
            print(f"[error] {ev}")
            return
        else:
            print(f"[event] {t}: {str(ev)[:160]}")

    if not final_result:
        print("\n!! 未收到 result 事件")
        return

    print("\n" + "=" * 70)
    print("最终结果摘要")
    print("=" * 70)

    ws = final_result.get("window_selection") or {}
    print(f"\n[窗口选择] mode={ws.get('mode')} chosen={ws.get('chosen_index')} reason={ws.get('reason','')[:80]}")

    model = final_result.get("model") or {}
    print(f"\n[模型辨识]")
    print(f"  type      : {model.get('model_type')}")
    print(f"  K (gain)  : {model.get('K')}")
    print(f"  T (tau)   : {model.get('T')}")
    print(f"  L (delay) : {model.get('L')}")
    print(f"  R²        : {model.get('r2_score')}")
    print(f"  confidence: {model.get('confidence')}")

    pid = final_result.get("pid_params") or {}
    print(f"\n[PID 整定 - 推荐策略 {pid.get('strategy')}]")
    Kp = pid.get("Kp")
    print(f"  Kp = {Kp}")
    print(f"  Ki = {pid.get('Ki')}")
    print(f"  Kd = {pid.get('Kd')}")
    print(f"  Ti = {pid.get('Ti')} s")
    print(f"  Td = {pid.get('Td')} s")
    if Kp and Kp > 0:
        print(f"  PB = {100/Kp:.2f} %  (DCS 表达)")

    cands = pid.get("candidates") or []
    if cands:
        print(f"\n[候选策略 {len(cands)} 个]")
        for c in cands:
            kp = c.get('Kp', 0)
            pb = f"{100/kp:.2f}%" if kp > 0 else "-"
            star = " ★" if c.get('is_recommended') else ""
            print(f"  {c.get('strategy'):10s} Kp={kp:.4f} Ti={c.get('Ti'):.1f}s Td={c.get('Td'):.1f}s PB={pb}{star}")

    ev_ = final_result.get("evaluation") or {}
    print(f"\n[性能评估]")
    print(f"  passed              : {ev_.get('passed')}")
    print(f"  performance_score   : {ev_.get('performance_score')}")
    print(f"  final_rating        : {ev_.get('final_rating')}")
    print(f"  is_stable           : {ev_.get('is_stable')}")
    print(f"  decay_ratio         : {ev_.get('decay_ratio')}")
    print(f"  overshoot_percent   : {ev_.get('overshoot_percent')}")
    print(f"  settling_time       : {ev_.get('settling_time')}")
    print(f"  IAE / ITAE          : {ev_.get('iae')} / {ev_.get('itae')}")

    print(f"\n[与现场对比]")
    print(f"  现场 (DCS): PB=60   TI=600s   TD=0   →  Kp ≈ {100/60:.4f}")
    if Kp:
        print(f"  本流程    : PB={100/Kp:.2f}% TI={pid.get('Ti'):.1f}s TD={pid.get('Td'):.1f}s  Kp={Kp}")


if __name__ == "__main__":
    asyncio.run(main())
