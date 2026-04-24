from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.algorithms.data_analysis import build_candidate_windows, load_and_prepare_dataset


def test_build_candidate_windows_detects_sustained_mv_activity():
    dt = 5.0
    n = 720
    ts = pd.date_range("2025-01-01", periods=n, freq=f"{int(dt)}s")

    mv = np.full(n, 50.0, dtype=float)
    mv[120:144] += np.linspace(0.0, 3.0, 24)
    mv[144:220] += 3.0
    mv[360:390] -= np.linspace(0.0, 4.0, 30)
    mv[390:470] -= 4.0
    mv += 0.05 * np.sin(np.arange(n) / 7.0)

    pv = np.full(n, 40.0, dtype=float)
    pv[150:260] += np.linspace(0.0, 1.5, 110)
    pv[400:520] -= np.linspace(0.0, 1.8, 120)

    df = pd.DataFrame({"timestamp": ts, "SV": 40.0, "PV": pv, "MV": mv})
    windows, events = build_candidate_windows(df, dt=dt, loop_type="temperature")

    assert len(events) >= 2
    assert len(windows) >= 2
    assert any(w["type"] == "mv_ramp" for w in windows)


def test_load_and_prepare_dataset_accepts_test_as_timestamp(tmp_path: Path):
    csv_path = tmp_path / "loop.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Test", "Loop.PV", "Loop.MV", "Loop.SV"])
        for i in range(6):
            w.writerow([f"2025/01/01 00:00:{i * 5:02d}", 40 + i * 0.1, 50 + i * 0.2, 45])

    res = load_and_prepare_dataset(csv_path=str(csv_path), loop_type="temperature")
    assert res["dt"] == 5.0


def test_load_and_prepare_dataset_accepts_xlsx(tmp_path: Path):
    xlsx_path = tmp_path / "loop.xlsx"
    df = pd.DataFrame(
        {
            "Test": [f"2025/01/01 00:00:{i * 5:02d}" for i in range(6)],
            "Loop.PV": [40 + i * 0.1 for i in range(6)],
            "Loop.MV": [50 + i * 0.2 for i in range(6)],
            "Loop.SV": [45 for _ in range(6)],
        }
    )
    df.to_excel(xlsx_path, index=False, engine="xlsxwriter")

    res = load_and_prepare_dataset(csv_path=str(xlsx_path), loop_type="temperature")
    assert res["dt"] == 5.0
