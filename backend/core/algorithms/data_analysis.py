"""Data analysis: CSV loading, cleaning, window detection and selection.

Key improvements over pid_new:
- #1: Detect MV step events when SV is absent/constant (manual-mode trials).
- #2: Adaptive window padding based on dt × estimated time constant,
      not raw data-length fractions.
- #3: Default window selected by quality score, not max amplitude.
- #13: MV is never denoised; PV uses edge-preserving median filter only.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.algorithms.signal_processing import denoise_mv, denoise_pv

# ── Column detection ─────────────────────────────────────────────────────────

_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "ts", "date", "test", "时间", "时间戳", "采集时间"],
    "SV": ["sv", "sp", "setpoint", "set_point", "target", "给定", "设定值", "目标值"],
    "PV": ["pv", "cv", "measurement", "feedback", "过程值", "测量值", "反馈值", "实际值"],
    "MV": ["mv", "op", "output", "valve", "opening", "开度", "阀位", "控制输出", "输出值"],
}


def _canon(text: str) -> str:
    t = str(text).replace("\ufeff", "").strip().lower()
    t = re.sub(r"[\(\（\[].*?[\)\）\]]", "", t)
    t = t.replace("%", "")
    t = re.sub(r"[\s\-_./\\]+", "", t)
    t = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", t)
    return t


def _detect_loops(df: pd.DataFrame) -> list[dict[str, str]]:
    loops: dict[str, dict[str, str]] = {}
    for col in df.columns:
        m = re.match(r"^(?P<prefix>.+?)\.(?P<sig>pv|sv|mv)\s*$", str(col).strip(), re.IGNORECASE)
        if not m:
            continue
        prefix = m.group("prefix").strip()
        sig = m.group("sig").upper()
        key = _canon(prefix)
        if not key:
            continue
        entry = loops.setdefault(key, {"prefix": prefix})
        entry[f"{sig.lower()}_col"] = str(col)
    return [e for e in loops.values() if "pv_col" in e and "mv_col" in e]


def _normalize_columns(df: pd.DataFrame, selected_loop_prefix: str | None = None) -> pd.DataFrame:
    canon_map = {_canon(col): col for col in df.columns}
    rename: dict[str, str] = {}

    loops = _detect_loops(df)
    chosen: dict[str, str] | None = None
    if loops:
        if len(loops) > 1 and not selected_loop_prefix:
            prefixes = [l["prefix"] for l in loops]
            raise ValueError(f"检测到多个回路，请选择: {prefixes}")
        if selected_loop_prefix:
            key = _canon(selected_loop_prefix)
            chosen = next((l for l in loops if _canon(l["prefix"]) == key), None)
        if chosen is None:
            chosen = loops[0]

    if chosen:
        for sig in ("PV", "MV", "SV"):
            col_key = f"{sig.lower()}_col"
            if col_key in chosen and chosen[col_key] in df.columns:
                rename[chosen[col_key]] = sig
        for alias in ["timestamp"] + _ALIASES["timestamp"]:
            c = _canon(alias)
            if c and c in canon_map and canon_map[c] not in rename:
                rename[canon_map[c]] = "timestamp"
                break
    else:
        for std, aliases in _ALIASES.items():
            if std in {"PV", "MV", "SV", "timestamp"}:
                for alias in [std] + aliases:
                    c = _canon(alias)
                    if c and c in canon_map and canon_map[c] not in rename:
                        rename[canon_map[c]] = std
                        break

    result = df.rename(columns=rename).copy()
    if "PV" not in result.columns or "MV" not in result.columns:
        raise ValueError(
            f"找不到 PV/MV 列。原始列: {list(df.columns)}，映射: {rename}"
        )
    return result


# ── Timestamp parsing ────────────────────────────────────────────────────────

def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        return df
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if ts.dropna().abs().median() > 1e11 else "s"
        df["timestamp"] = (
            pd.to_datetime(ts, unit=unit, errors="coerce", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    else:
        df["timestamp"] = pd.to_datetime(ts, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _estimate_dt(df: pd.DataFrame, fallback: float = 1.0) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return fallback
    deltas = df["timestamp"].diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    return float(deltas.median()) if not deltas.empty else fallback


def _read_xlsx(xlsx_path: str) -> pd.DataFrame:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    def _col_index(ref: str) -> int:
        letters = "".join(ch for ch in ref if ch.isalpha()).upper()
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch) - 64)
        return max(idx - 1, 0)

    def _cell_value(cell: ET.Element, shared: list[str]) -> str:
        cell_type = cell.attrib.get("t", "")
        if cell_type == "inlineStr":
            node = cell.find("main:is/main:t", ns)
            return node.text if node is not None and node.text is not None else ""
        node = cell.find("main:v", ns)
        if node is None or node.text is None:
            return ""
        if cell_type == "s":
            try:
                return shared[int(node.text)]
            except Exception:
                return node.text
        return node.text

    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                shared_strings.append("".join((t.text or "") for t in si.findall(".//main:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = workbook.find("main:sheets", ns)
        if sheets is None or not list(sheets):
            return pd.DataFrame()
        first_sheet = list(sheets)[0]
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("rel:Relationship", ns):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            return pd.DataFrame()

        sheet_path = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
        sheet = ET.fromstring(zf.read(sheet_path))
        rows: list[list[str]] = []
        max_cols = 0
        for row in sheet.findall(".//main:sheetData/main:row", ns):
            values: list[str] = []
            for cell in row.findall("main:c", ns):
                idx = _col_index(cell.attrib.get("r", "A1"))
                while len(values) <= idx:
                    values.append("")
                values[idx] = _cell_value(cell, shared_strings)
            max_cols = max(max_cols, len(values))
            rows.append(values)

    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    return pd.DataFrame(rows)


# ── CSV loading ──────────────────────────────────────────────────────────────

def _read_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = _read_xlsx(str(path))
        raw = raw.dropna(how="all").reset_index(drop=True)
        if raw.empty:
            return pd.DataFrame()

        alias_sets = {
            std: [_canon(a) for a in [std] + aliases]
            for std, aliases in _ALIASES.items()
        }
        best_idx, best_score = 0, -1
        for idx in range(min(80, len(raw))):
            row = raw.iloc[idx].tolist()
            cells = [_canon(str(c)) for c in row if pd.notna(c) and str(c).strip()]
            if not cells:
                continue
            pv = any(any(a and (a == c or a in c) for a in alias_sets["PV"]) for c in cells)
            mv = any(any(a and (a == c or a in c) for a in alias_sets["MV"]) for c in cells)
            score = (5 if pv else 0) + (5 if mv else 0)
            if score > best_score:
                best_score, best_idx = score, idx

        data = raw.iloc[best_idx + 1:].copy() if best_score >= 10 else raw.copy()
        headers = raw.iloc[best_idx].tolist() if best_score >= 10 else raw.iloc[0].tolist()
        data.columns = [
            str(h).strip() if pd.notna(h) and str(h).strip() else f"Unnamed_{i}"
            for i, h in enumerate(headers)
        ]
        if "timestamp" not in data.columns:
            # Some exported XLSX files encode the Chinese header "timestamp" in a
            # non-standard way. Fall back to detecting a datetime-like column by
            # its values so dt/window logic does not silently use 1s.
            for col in list(data.columns)[:5]:
                sample = data[col].dropna().astype(str).head(50)
                if sample.empty:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce")
                if float(parsed.notna().mean()) >= 0.8:
                    data = data.rename(columns={col: "timestamp"})
                    break
        data = data.dropna(how="all").reset_index(drop=True)
        return data

    encodings = ["utf-8", "gbk", "gb18030", "iso-8859-1"]
    alias_sets = {
        std: [_canon(a) for a in [std] + aliases]
        for std, aliases in _ALIASES.items()
    }

    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc) as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            continue

        best_idx, best_score = 0, -1
        for idx in range(min(80, len(lines))):
            raw = lines[idx].strip()
            if not raw or raw.count(",") + raw.count(";") + raw.count("\t") < 1:
                continue
            cells = [_canon(c) for c in re.split(r"[,\t;]", raw) if c.strip()]
            pv = any(any(a and (a == c or a in c) for a in alias_sets["PV"]) for c in cells)
            mv = any(any(a and (a == c or a in c) for a in alias_sets["MV"]) for c in cells)
            score = (5 if pv else 0) + (5 if mv else 0)
            if score > best_score:
                best_score, best_idx = score, idx

        skip = best_idx if best_score >= 10 else 0
        header_line = lines[skip] if 0 <= skip < len(lines) else ""
        delim = max([",", "\t", ";"], key=lambda ch: header_line.count(ch))

        try:
            return pd.read_csv(csv_path, encoding=enc, skiprows=skip, sep=delim)
        except Exception:
            continue

    return pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")


# ── Window quality ───────────────────────────────────────────────────────────

def _robust_noise(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    d = np.diff(values.astype(float))
    return float(np.median(np.abs(d - np.median(d))))


def score_window(df: pd.DataFrame, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assess identification suitability of a data segment."""
    mv = df["MV"].to_numpy(dtype=float)
    pv = df["PV"].to_numpy(dtype=float)
    mv_span = float(np.max(mv) - np.min(mv)) if mv.size else 0.0
    pv_span = float(np.max(pv) - np.min(pv)) if pv.size else 0.0
    mv_noise = _robust_noise(mv)
    pv_noise = _robust_noise(pv)
    mv_eff = mv_span >= max(mv_noise * 12.0, 1e-6) and float(np.std(mv)) >= max(mv_noise * 4.0, 1e-6)
    pv_eff = pv_span >= max(pv_noise * 10.0, 1e-6) and float(np.std(pv)) >= max(pv_noise * 3.0, 1e-6)

    # Cross-correlation with lag search (fix #4: not zero-lag only)
    corr = 0.0
    if mv.size >= 10:
        mv_c = mv - float(np.mean(mv))
        pv_c = pv - float(np.mean(pv))
        mv_std = float(np.std(mv_c))
        pv_std = float(np.std(pv_c))
        if mv_std > 1e-12 and pv_std > 1e-12:
            max_lag = min(mv.size // 4, 50)
            for lag in range(0, max_lag + 1):
                a = mv_c[:-lag] if lag > 0 else mv_c
                b = pv_c[lag:] if lag > 0 else pv_c
                if a.size < 10:
                    break
                s = float(np.mean(a * b) / (mv_std * pv_std))
                if s > corr:
                    corr = s

    sat_ratio = 0.0
    if mv_span > 1e-9:
        low = float(np.min(mv)) + 0.01 * mv_span
        high = float(np.max(mv)) - 0.01 * mv_span
        sat_ratio = max(float(np.mean(mv <= low)), float(np.mean(mv >= high)))

    drift_ratio = 0.0
    if pv.size >= 5 and pv_span > 1e-9:
        x = np.arange(pv.size, dtype=float)
        slope = float(np.polyfit(x, pv, 1)[0])
        drift_ratio = abs(slope) * float(pv.size - 1) / pv_span

    mv_excitation_score = 1.0 if mv_eff else min(
        0.99,
        max(
            mv_span / max(mv_noise * 12.0, 1e-6),
            float(np.std(mv)) / max(mv_noise * 4.0, 1e-6),
        ) * 0.5,
    )
    pv_response_score = 1.0 if pv_eff else min(
        0.99,
        max(
            pv_span / max(pv_noise * 10.0, 1e-6),
            float(np.std(pv)) / max(pv_noise * 3.0, 1e-6),
        ) * 0.5,
    )
    correlation_score = max(0.0, min(corr / 0.4, 1.0))
    saturation_score = max(0.0, 1.0 - min(sat_ratio / 0.6, 1.0) * 0.7)
    drift_score = max(0.0, 1.0 - min(drift_ratio / 1.0, 1.0) * 0.25)

    reasons: list[str] = []
    if not mv_eff:
        reasons.append("MV激励不足")
    if not pv_eff:
        reasons.append("PV响应不足")
    if corr < 0.05:
        reasons.append("MV与PV相关性弱")
    if sat_ratio > 0.4:
        reasons.append("MV疑似饱和")

    score = 0.4 * mv_excitation_score + 0.4 * pv_response_score + 0.2 * correlation_score
    score *= saturation_score
    score *= drift_score

    usable = mv_eff and pv_eff and corr >= 0.05 and sat_ratio <= 0.6
    policy = policy if isinstance(policy, dict) else {}
    min_pv_response = _policy_float(policy, "min_pv_response")
    max_drift_ratio = _policy_float(policy, "max_drift_ratio")
    max_mv_saturation_ratio = _policy_float(policy, "max_mv_saturation_ratio")
    if min_pv_response is not None and pv_span < min_pv_response:
        reasons.append(f"PV响应幅值低于策略下限({min_pv_response:g})")
        usable = False
        score *= 0.7
    if max_drift_ratio is not None and drift_ratio > max_drift_ratio:
        reasons.append(f"PV漂移比例超过策略上限({max_drift_ratio:g})")
        usable = False
        score *= 0.7
    if max_mv_saturation_ratio is not None and sat_ratio > max_mv_saturation_ratio:
        reasons.append(f"MV饱和/贴边比例超过策略上限({max_mv_saturation_ratio:g})")
        usable = False
        score *= 0.7
    score_breakdown = {
        "mv_excitation": round(float(mv_excitation_score), 4),
        "pv_response": round(float(pv_response_score), 4),
        "lag_correlation": round(float(correlation_score), 4),
        "saturation_penalty": round(float(saturation_score), 4),
        "drift_penalty": round(float(drift_score), 4),
    }
    raw_metrics = {
        "mv_noise": round(float(mv_noise), 6),
        "pv_noise": round(float(pv_noise), 6),
        "mv_std": round(float(np.std(mv)), 6) if mv.size else 0.0,
        "pv_std": round(float(np.std(pv)), 6) if pv.size else 0.0,
        "saturation_ratio": round(float(sat_ratio), 6),
        "drift_ratio": round(float(drift_ratio), 6),
    }
    return {
        "passed": bool(usable),
        "score": float(max(0.0, min(score, 1.0))),
        "reasons": reasons,
        "score_breakdown": score_breakdown,
        "raw_metrics": raw_metrics,
        "mv_span": mv_span,
        "pv_span": pv_span,
        "corr": corr,
        "saturation_ratio": sat_ratio,
        "drift_ratio": drift_ratio,
    }


# ── Step event detection ─────────────────────────────────────────────────────

def _detect_sv_steps(df: pd.DataFrame, threshold: float = 0.5) -> list[dict[str, Any]]:
    sv = df["SV"].to_numpy(dtype=float)
    if sv.size < 4:
        return []
    sv_diff = np.diff(sv)
    noise = float(np.median(np.abs(sv_diff - np.median(sv_diff))))
    thr = max(threshold, noise * 6.0, 1e-6)
    indices = np.where(np.abs(sv_diff) >= thr)[0] + 1
    if indices.size == 0:
        return []

    merge_gap = max(3, min(20, len(df) // 200))
    window = max(3, min(30, len(df) // 100))
    groups: list[list[int]] = []
    cur = [int(indices[0])]
    for idx in indices[1:]:
        if int(idx) - cur[-1] <= merge_gap:
            cur.append(int(idx))
        else:
            groups.append(cur)
            cur = [int(idx)]
    groups.append(cur)

    events: list[dict[str, Any]] = []
    for group in groups:
        center = int(round(sum(group) / len(group)))
        sv_before = float(np.median(sv[max(0, center - window):center]))
        sv_after = float(np.median(sv[center:min(len(df), center + window)]))
        amp = abs(sv_after - sv_before)
        if amp < threshold:
            continue
        events.append({
            "start_idx": max(0, center - merge_gap),
            "end_idx": min(len(df), center + merge_gap + 1),
            "amplitude": amp,
            "sv_start": sv_before,
            "sv_end": sv_after,
            "type": "step_up" if sv_after > sv_before else "step_down",
        })
    return events


def _detect_mv_steps(df: pd.DataFrame, policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fix #1: Detect significant MV steps for manual-mode identification data."""
    mv = df["MV"].to_numpy(dtype=float)
    if mv.size < 10:
        return []
    mv_diff = np.abs(np.diff(mv))
    noise = float(np.median(np.abs(mv_diff - np.median(mv_diff))))
    thr = max(noise * 8.0, (float(np.max(mv)) - float(np.min(mv))) * 0.05, 1e-6)
    policy_min_mv = _policy_float(policy, "min_mv_excitation")
    if policy_min_mv is not None:
        thr = max(thr, policy_min_mv)
    indices = np.where(mv_diff >= thr)[0]
    if indices.size == 0:
        return []

    merge_gap = max(5, min(30, len(df) // 100))
    top_k = 8
    # Pick top-k by amplitude, well-separated
    order = np.argsort(mv_diff[indices])[::-1]
    selected: list[int] = []
    for oi in order:
        idx = int(indices[oi])
        if any(abs(idx - s) < merge_gap for s in selected):
            continue
        selected.append(idx)
        if len(selected) >= top_k:
            break

    events: list[dict[str, Any]] = []
    for idx in selected:
        events.append({
            "start_idx": idx,
            "end_idx": min(len(df), idx + 2),
            "amplitude": float(mv_diff[idx]),
            "type": "mv_step",
        })
    return events


def _detect_mv_activity_segments(
    df: pd.DataFrame,
    dt: float,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Detect sustained MV activity, not just one-sample jumps.

    This is meant for closed-loop history where MV often ramps over tens of
    seconds or minutes instead of showing a sharp step in one sample.
    """
    mv = df["MV"].to_numpy(dtype=float)
    n = mv.size
    if n < 20:
        return []

    mv_s = pd.Series(mv).rolling(window=3, center=True, min_periods=1).median().to_numpy()
    mv_range = float(np.max(mv_s) - np.min(mv_s))
    point_noise = _robust_noise(mv_s)
    if mv_range <= 1e-9:
        return []

    policy = policy if isinstance(policy, dict) else {}
    top_k = max(1, _policy_int(policy, "max_candidates_per_family", default=8) or 8)
    selected: list[dict[str, Any]] = []
    seen_centers: list[int] = []

    post_s = _policy_float(policy, "post_window_s")
    if post_s:
        horizons_s = (
            max(30.0, min(post_s / 12.0, 300.0)),
            max(60.0, min(post_s / 6.0, 600.0)),
            max(120.0, min(post_s / 3.0, 1200.0)),
        )
    else:
        horizons_s = (30.0, 60.0, 120.0)

    for horizon_s in horizons_s:
        w = int(max(6, round(horizon_s / max(dt, 1e-6))))
        if w >= max(n // 3, 12):
            continue

        net_change = np.abs(mv_s[w:] - mv_s[:-w])
        if net_change.size == 0:
            continue

        thr = max(point_noise * 10.0, mv_range * 0.015, 0.5)
        candidate_idx = np.where(net_change >= thr)[0]
        if candidate_idx.size == 0:
            continue

        min_gap = max(w // 2, 12)
        order = candidate_idx[np.argsort(net_change[candidate_idx])[::-1]]
        for idx in order:
            center = int(idx + w // 2)
            if any(abs(center - prev) < min_gap for prev in seen_centers):
                continue
            selected.append({
                "start_idx": int(idx),
                "end_idx": int(min(n, idx + w)),
                "amplitude": float(net_change[idx]),
                "type": "mv_ramp",
            })
            seen_centers.append(center)
            if len(selected) >= top_k:
                return selected

    return selected


def _detect_steady_disturbance_segments(
    df: pd.DataFrame,
    dt: float,
    loop_type: str | None = None,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Scan longer history for usable steady-disturbance windows.

    These are not sharp steps or ramps. They are longer periods where MV has
    enough natural movement and PV visibly responds, while MV is not pinned to
    the observed limits. This gives the identification stage another option
    when historical data contains normal operating variation instead of tests.
    """
    policy = policy if isinstance(policy, dict) else {}
    n = len(df)
    min_points = max(40, _policy_int(policy, "min_window_points", default=80) or 80)
    if n < min_points:
        return []

    target_s = _LOOP_POST_S.get((loop_type or "").strip().lower(), _LOOP_POST_DEFAULT)
    scan_window_s = _policy_float(policy, "steady_scan_window_s") or target_s
    win = int(max(min_points, min(round(scan_window_s / max(dt, 1e-6)), max(min_points, n // 3))))
    if win >= n:
        win = max(40, n // 2)
    scan_step_s = _policy_float(policy, "steady_scan_step_s")
    step = max(10, int(round(scan_step_s / max(dt, 1e-6)))) if scan_step_s else max(10, win // 4)

    mv_all = df["MV"].to_numpy(dtype=float)
    mv_range = float(np.max(mv_all) - np.min(mv_all)) if mv_all.size else 0.0
    mv_noise = _robust_noise(mv_all)
    min_mv_span = max(mv_noise * 12.0, mv_range * 0.015, 0.3)
    policy_min_mv = _policy_float(policy, "min_mv_excitation")
    if policy_min_mv is not None:
        min_mv_span = max(min_mv_span, policy_min_mv)
    max_sat = _policy_float(policy, "max_mv_saturation_ratio")
    if max_sat is None:
        max_sat = 0.35
    max_candidates = max(1, _policy_int(policy, "max_candidates_per_family", default=4) or 4)

    candidates: list[dict[str, Any]] = []
    for start in range(0, max(n - win + 1, 1), step):
        end = min(n, start + win)
        if end - start < 40:
            continue
        seg = df.iloc[start:end]
        quality = score_window(seg, policy=policy)
        if quality["mv_span"] < min_mv_span:
            continue
        if quality["saturation_ratio"] > max_sat:
            continue
        # Keep medium-quality windows too; they are useful as fallbacks and
        # make the candidate pool explainable in the UI.
        if quality["score"] < 0.35:
            continue
        candidates.append({
            "start_idx": int(start),
            "end_idx": int(end),
            "amplitude": float(quality["mv_span"]),
            "type": "steady_disturbance",
            "pre_scored_quality": quality,
        })

    candidates.sort(key=lambda item: float(item["pre_scored_quality"]["score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for item in candidates:
        center = (int(item["start_idx"]) + int(item["end_idx"])) // 2
        if any(abs(center - ((int(prev["start_idx"]) + int(prev["end_idx"])) // 2)) < win // 2 for prev in selected):
            continue
        selected.append(item)
        if len(selected) >= max_candidates:
            break
    return selected


def _merge_events(
    primary_events: list[dict[str, Any]],
    extra_events: list[dict[str, Any]],
    *,
    proximity: int,
) -> list[dict[str, Any]]:
    merged = list(primary_events)
    for ev in extra_events:
        center = int((int(ev["start_idx"]) + int(ev["end_idx"])) // 2)
        covered = any(
            abs(center - int((int(base["start_idx"]) + int(base["end_idx"])) // 2)) < proximity
            for base in merged
        )
        if not covered:
            merged.append(ev)
    return merged


# ── Adaptive window builder ──────────────────────────────────────────────────

# 阶跃后窗口目标长度（秒），按回路类型给定。
# 经验：辨识窗口至少覆盖 3~5 倍过程时间常数。
# - 流量回路 T 通常 1~10s，120s 远超
# - 压力 T 5~60s，300s 够
# - 温度 T 60~600s，需要 1800s（30 分钟）
# - 液位常是积分型或 T 数百秒，需要 2400s（40 分钟）
_LOOP_POST_S = {
    "flow":         120.0,
    "pressure":     300.0,
    "temperature": 1800.0,
    "level":       2400.0,
}
_LOOP_POST_DEFAULT = 300.0


def _policy_float(policy: dict[str, Any] | None, key: str, default: float | None = None) -> float | None:
    if not isinstance(policy, dict):
        return default
    value = policy.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _policy_int(policy: dict[str, Any] | None, key: str, default: int | None = None) -> int | None:
    value = _policy_float(policy, key, None)
    if value is None:
        return default
    return int(round(value))


def _adaptive_padding(
    amplitude: float,
    dt: float,
    n: int,
    loop_type: str | None = None,
    policy: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Fix #2: Compute pre/post padding from dt, dataset size, and loop dynamics.

    post 长度按 loop_type 决定，以覆盖该类型典型时间常数的 3~5 倍。
    pre = max(20, post/10) 以提供阶跃前基线。
    硬上限 = 数据集 25%，避免任何单窗口吃掉整段数据。
    """
    policy = policy if isinstance(policy, dict) else {}
    target_post_s = _policy_float(policy, "post_window_s")
    if target_post_s is None:
        target_post_s = _LOOP_POST_S.get((loop_type or "").strip().lower(), _LOOP_POST_DEFAULT)
    target_pre_s = _policy_float(policy, "pre_window_s")

    max_policy_pts = _policy_int(policy, "max_window_points")
    max_pts = max(200, n // 4)
    if max_policy_pts is not None:
        max_pts = max(40, min(max_pts, max_policy_pts))
    post = int(min(target_post_s / max(dt, 1e-6), max_pts))
    post = max(post, int(60.0 / max(dt, 1e-6)))  # 兜底至少 60s
    if target_pre_s is not None:
        pre = int(max(1, min(target_pre_s / max(dt, 1e-6), post)))
    else:
        pre = max(20, min(post // 10, int(60.0 / max(dt, 1e-6))))
    return pre, post


def build_candidate_windows(
    df: pd.DataFrame,
    dt: float,
    loop_type: str | None = None,
    policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build candidate identification windows from step events.

    Returns:
        (candidate_windows, step_events)

    Fix #1: Uses both SV-step and MV-step detection.
    Fix #2: Adaptive padding based on dt.
    Fix #3: Selects best window by quality score.
    """
    n = len(df)
    if n < 20:
        return [], []

    step_events: list[dict[str, Any]] = []
    if "SV" in df.columns and df["SV"].nunique(dropna=True) > 1:
        sv_thr = max(0.5, float(df["SV"].std(ddof=0) * 0.2))
        policy_min_sp = _policy_float(policy, "min_sp_excitation")
        if policy_min_sp is not None:
            sv_thr = max(sv_thr, policy_min_sp)
        step_events = _detect_sv_steps(df, threshold=sv_thr)

    mv_steps = _detect_mv_steps(df, policy=policy)
    mv_activity = _detect_mv_activity_segments(df, dt, policy=policy)
    steady_disturbance = _detect_steady_disturbance_segments(df, dt, loop_type, policy=policy)

    # Merge: keep explicit SV steps first, then add MV step/ramp candidates.
    merge_gap_s = _policy_float(policy, "merge_gap_s", 180.0) or 180.0
    merge_pts = max(40, int(merge_gap_s / max(dt, 1e-6)))
    all_events = _merge_events(step_events, mv_steps, proximity=40)
    all_events = _merge_events(all_events, mv_activity, proximity=merge_pts)
    all_events = _merge_events(all_events, steady_disturbance, proximity=merge_pts)

    if not all_events:
        # Fallback: single window around largest MV change
        mv = df["MV"].to_numpy(dtype=float)
        if mv.size >= 2:
            center = int(np.argmax(np.abs(np.diff(mv))))
            pre, post = _adaptive_padding(float(np.max(np.abs(np.diff(mv)))), dt, n, loop_type, policy=policy)
            all_events = [{
                "start_idx": center,
                "end_idx": min(n, center + 2),
                "amplitude": float(np.max(np.abs(np.diff(mv)))),
                "type": "mv_fallback",
            }]

    windows: list[dict[str, Any]] = []
    for ev in all_events:
        center = int(ev["start_idx"])
        amp = float(ev.get("amplitude", 1.0))
        if ev.get("type") == "steady_disturbance":
            w_start = max(0, int(ev.get("start_idx", 0)))
            w_end = min(n, int(ev.get("end_idx", n)))
        else:
            pre, post = _adaptive_padding(amp, dt, n, loop_type, policy=policy)
            w_start = max(0, center - pre)
            w_end = min(n, center + post)
        if w_end - w_start < 20:
            continue
        seg = df.iloc[w_start:w_end]
        quality = score_window(seg, policy=policy)
        algorithm = "sv_step" if ev.get("type") in {"step_up", "step_down"} else str(ev.get("type", "unknown"))
        if algorithm == "mv_fallback":
            algorithm_label = "largest_mv_change"
        elif algorithm == "steady_disturbance":
            algorithm_label = "steady_disturbance_scan"
        else:
            algorithm_label = algorithm
        windows.append({
            **ev,
            "window_algorithm": algorithm,
            "window_algorithm_label": algorithm_label,
            "window_selection_basis": "score_window: mv_excitation + pv_response + lag_correlation - saturation/drift",
            "window_start_idx": w_start,
            "window_end_idx": w_end,
            "window_usable_for_id": quality["passed"],
            "window_quality_score": quality["score"],
            "window_quality_reasons": quality["reasons"],
            "window_score_breakdown": quality["score_breakdown"],
            "window_quality_metrics": quality["raw_metrics"],
            "window_mv_span": quality["mv_span"],
            "window_pv_span": quality["pv_span"],
            "window_corr": quality["corr"],
        })

    # Fix #3: sort by quality score, not amplitude
    windows.sort(key=lambda w: (
        int(bool(w.get("window_usable_for_id"))),
        float(w.get("window_quality_score", 0.0)),
    ), reverse=True)

    # Label windows
    type_counter: dict[str, int] = {}
    for w in windows:
        base = str(w.get("window_algorithm") or "")
        if base in {"step_up", "step_down"}:
            base = "sv_step"
        if base not in {"sv_step", "mv_step", "mv_ramp", "steady_disturbance", "mv_fallback"}:
            base = "mv_change"
        type_counter[base] = type_counter.get(base, 0) + 1
        w["window_source"] = f"{base}_{type_counter[base]}"

    return windows, all_events


# ── 主入口 ────────────────────────────────────────────────────────────────

def _load_clean_only(
    *,
    csv_path: str,
    selected_loop_prefix: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[pd.DataFrame, float]:
    """完成数据加载 + 列归一化 + 时间解析 + 数值清洗 + PV 去噪。

    返回 (cleaned_df, dt)。**不做窗口检测**，便于 skill 层各自独立调用。
    供 `load_and_prepare_dataset` 与 `load_dataset` 技能共用。
    """
    raw = _read_csv(csv_path)
    df = _normalize_columns(raw, selected_loop_prefix=selected_loop_prefix)
    df = _parse_timestamps(df)

    # 时间段切片
    if start_time or end_time:
        start_dt = pd.to_datetime(start_time, errors="coerce") if start_time else None
        end_dt = pd.to_datetime(end_time, errors="coerce") if end_time else None
        if start_dt is not None and not pd.isna(start_dt):
            df = df[df["timestamp"] >= start_dt]
        if end_dt is not None and not pd.isna(end_dt):
            df = df[df["timestamp"] <= end_dt]
        df = df.reset_index(drop=True)
        if len(df) < 20:
            raise ValueError(f"所选时间段数据不足 20 行 (得到 {len(df)} 行)")

    # 数值化 + 插值
    for col in ["SV", "PV", "MV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PV", "MV"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("清洗后数据为空，请检查 PV/MV 列")
    for col in ["SV", "PV", "MV"]:
        if col in df.columns:
            df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    dt = _estimate_dt(df)

    # 修复 #13：只对 PV 去噪，MV 保持原始
    cleaned = df.copy()
    if len(cleaned) >= 25:
        cleaned["PV"] = denoise_pv(cleaned["PV"].to_numpy(), noise_level="auto")
        # MV 故意不滤波

    return cleaned, dt


def load_and_prepare_dataset(
    *,
    csv_path: str,
    selected_loop_prefix: str | None = None,
    selected_window_index: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    loop_type: str | None = None,
) -> dict[str, Any]:
    """加载 CSV、清洗、去噪（仅 PV）、检测候选窗口的整套流程。

    返回 dict，包含：
        cleaned_df, dt, step_events, candidate_windows,
        data_points, quality_metrics（或 None）
    """
    cleaned, dt = _load_clean_only(
        csv_path=csv_path,
        selected_loop_prefix=selected_loop_prefix,
        start_time=start_time,
        end_time=end_time,
    )
    candidate_windows, step_events = build_candidate_windows(cleaned, dt, loop_type=loop_type)

    # 应用 selected_window_index 重排（把指定窗口提到首位）
    if selected_window_index is not None and 0 <= selected_window_index < len(candidate_windows):
        candidate_windows.insert(0, candidate_windows.pop(selected_window_index))

    return {
        "cleaned_df": cleaned,
        "dt": dt,
        "step_events": step_events,
        "candidate_windows": candidate_windows,
        "data_points": len(cleaned),
        "quality_metrics": None,
    }
