"""批量跑真实回路 CSV，对比当前流水线效果。

用法：
    cd backend && python -m scripts.test_real_loop
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline.runner import run_tuning_pipeline


DATA_DIR = Path(r"D:\data\20260416_split_by_loop")

# (文件名, loop_type, 现场 PB, 现场 TI, 现场 TD)
CASES = [
    ("20260416_5203_FIC_21005.csv", "flow",        666, 20,   0),
    ("20260416_5203_LIC_11502.csv", "level",        60, 600,  0),
    ("20260416_5203_PIC_11201.csv", "pressure",     33, 333,  0),
    ("20260416_5203_PIC_11501.csv", "pressure",     30, 350,  0),
    ("20260416_5203_PIC_21901.csv", "pressure",     50, 300,  1),
    ("20260416_5203_TIC_11303.csv", "temperature",  66, 666, 20),
    ("20260416_5203_TIC_20201.csv", "temperature",  50, 350, 10),
]


async def run_one(csv_path: str, loop_type: str) -> dict:
    """跑一个回路，返回精简结果或错误。"""
    final = None
    last_error = None
    win_info = None
    model_info = None
    async for ev in run_tuning_pipeline(
        csv_path=csv_path,
        loop_type=loop_type,
        use_llm_advisor=False,
    ):
        t = ev.get("type")
        if t == "stage" and ev.get("status") == "done":
            stage = ev.get("stage")
            data = ev.get("data") or {}
            if stage == "window_selection":
                win_info = data
            elif stage == "identification":
                model_info = data
        elif t == "result":
            final = ev.get("data")
        elif t == "error":
            last_error = (ev.get("stage"), ev.get("error_code"), ev.get("message"))
    return {"final": final, "error": last_error, "window": win_info, "model": model_info}


def fmt_row(name: str, loop_type: str, pb_field, ti_field, td_field, res: dict) -> str:
    if res["error"]:
        stage, code, msg = res["error"]
        return f"{name:18s} {loop_type:11s} ❌ [{stage}/{code}] {msg[:60]}"
    f = res["final"]
    if not f:
        return f"{name:18s} {loop_type:11s} ❌ 无结果"
    pid = f.get("pid_params") or {}
    ev = f.get("evaluation") or {}
    model = f.get("model") or {}
    Kp = pid.get("Kp", 0)
    pb_calc = f"{100/Kp:.1f}" if Kp > 0 else "-"
    return (
        f"{name:18s} {loop_type:11s} ✅ "
        f"模型={model.get('model_type','-'):6s} R²={float(model.get('r2_score',0)):+.2f} "
        f"conf={float(model.get('confidence',0)):.2f} | "
        f"PB={pb_calc:>6s}% TI={pid.get('Ti',0):.0f}s TD={pid.get('Td',0):.0f}s | "
        f"score={float(ev.get('performance_score',0)):.1f} pass={ev.get('passed')} | "
        f"现场 PB={pb_field}/TI={ti_field}/TD={td_field}"
    )


async def debug_lic() -> None:
    """详细查看 LIC_11502 的窗口和模型尝试。"""
    from core.algorithms.data_analysis import load_and_prepare_dataset
    from core.algorithms.system_id import fit_best_model
    csv = str(DATA_DIR / "20260416_5203_LIC_11502.csv")
    ds = load_and_prepare_dataset(csv_path=csv, loop_type="level")
    print(f"\n[LIC_11502 候选窗口（loop_type=level）]")
    for i, w in enumerate(ds["candidate_windows"][:5]):
        ws, we = w.get("window_start_idx"), w.get("window_end_idx")
        print(f"  #{i} usable={w.get('window_usable_for_id')} score={w.get('window_quality_score'):.3f} "
              f"window=[{ws},{we}] len={we-ws} mv_span={w.get('window_mv_span'):.2f} "
              f"pv_span={w.get('window_pv_span'):.3f} corr={w.get('window_corr'):.3f} "
              f"reasons={w.get('window_quality_reasons','')}")

    if not ds["candidate_windows"]:
        return
    # 不指定模型，看默认逻辑选啥
    print(f"\n[默认（不强制），loop_type=level]")
    res = fit_best_model(
        cleaned_df=ds["cleaned_df"],
        candidate_windows=ds["candidate_windows"],
        actual_dt=ds["dt"],
        loop_type="level",
    )
    m = res.get("model"); conf = res.get("confidence")
    print(f"  默认胜出 = {m.model_type.value} K={m.K:+.4f} T={m.T:.2f} L={m.L:.2f} "
          f"R²={m.r2_score:+.3f} conf={conf.confidence:.3f} window={res.get('window_source')}")
    print(f"  attempts (前 8 个):")
    for a in (res.get("attempts") or [])[:8]:
        print(f"    {a}")

    print(f"\n[强制尝试每个模型，使用全部候选窗口，loop_type=level]")
    for mt in ["IPDT", "FOPDT", "FO", "SOPDT"]:
        try:
            res = fit_best_model(
                cleaned_df=ds["cleaned_df"],
                candidate_windows=ds["candidate_windows"],
                actual_dt=ds["dt"],
                loop_type="level",
                force_model_types=[mt],
            )
            m = res.get("model")
            conf = res.get("confidence")
            ws = res.get("window_source", "?")
            print(f"  {mt:6s} K={m.K:+.4f} T={m.T:.2f} L={m.L:.2f} "
                  f"R²={m.r2_score:+.3f} conf={conf.confidence if conf else '?'} window={ws}")
        except Exception as exc:
            print(f"  {mt:6s} 失败: {exc}")
    print()


async def main() -> None:
    await debug_lic()
    print(f"{'='*150}")
    print(f"{'回路':18s} {'类型':11s} 状态 / 结果")
    print(f"{'='*150}")
    for fname, loop_type, pb, ti, td in CASES:
        csv = str(DATA_DIR / fname)
        name = fname.replace("20260416_5203_", "").replace(".csv", "")
        try:
            res = await run_one(csv, loop_type)
            print(fmt_row(name, loop_type, pb, ti, td, res))
        except Exception as exc:
            print(f"{name:18s} {loop_type:11s} 💥 异常: {exc}")
    print(f"{'='*150}")


if __name__ == "__main__":
    asyncio.run(main())
