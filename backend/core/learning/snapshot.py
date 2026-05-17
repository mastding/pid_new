"""统一的 snapshot 写入接口。

- 算法层 / 内置 skill / 外部 skill 共用同一套 API
- 每条 snapshot 含 ``snapshot_id / ts / skill_name / skill_version /
  code_origin / payload / observable_outcomes``
- 落点规则：
  1) 若 var/skills/<dir>/learning/snapshots/ 存在（dir 名匹配 skill_name 后缀），落到这里
  2) 否则落到 var/learning/snapshots/<skill_name>/

后续 ExperienceStore（V3.x）通过 ``attach_outcome`` 把部署后观察值补到对应
snapshot 文件里；recalibrate 脚本通过 ``iter_snapshots`` 读 + 训练。

写入失败被 try/except 兜住——绝不影响主流程。这是基础设施代码的硬性约束。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# var/ 根目录：相对当前文件 core/learning/snapshot.py → ../../var/
_VAR_ROOT = Path(__file__).resolve().parent.parent.parent / "var"
_EXTERNAL_SKILLS_DIR = _VAR_ROOT / "skills"
_FALLBACK_LEARNING_DIR = _VAR_ROOT / "learning" / "snapshots"

# external skill 目录名 ↔ skill_name 的映射缓存（避免每次都扫盘）
_SKILL_DIR_CACHE: dict[str, Path] | None = None


def resolve_snapshot_dir(skill_name: str) -> Path:
    """决定某 skill 的 snapshot 应该落到哪个目录。

    优先尝试外部 skill 自己的 ``var/skills/<dir>/learning/snapshots/``，
    匹配规则：目录名是 skill_name 的后缀（如 ``harris_closed_loop`` 匹配
    ``compute_harris_closed_loop``）。

    匹配不到时落到 ``var/learning/snapshots/<skill_name>/``。
    """
    global _SKILL_DIR_CACHE
    if _SKILL_DIR_CACHE is None:
        _SKILL_DIR_CACHE = _build_skill_dir_cache()

    cached = _SKILL_DIR_CACHE.get(skill_name)
    if cached is not None:
        return cached / "learning" / "snapshots"

    # 后缀匹配：skill_name 包含某个外部 skill 目录名
    for dir_name, dir_path in _SKILL_DIR_CACHE.items():
        if skill_name == dir_name or skill_name.endswith(dir_name):
            _SKILL_DIR_CACHE[skill_name] = dir_path
            return dir_path / "learning" / "snapshots"

    return _FALLBACK_LEARNING_DIR / skill_name


def _build_skill_dir_cache() -> dict[str, Path]:
    cache: dict[str, Path] = {}
    if not _EXTERNAL_SKILLS_DIR.exists():
        return cache
    for d in _EXTERNAL_SKILLS_DIR.iterdir():
        if d.is_dir() and (d / "manifest.yaml").exists():
            cache[d.name] = d
            # 同时读 manifest 取真名（如 compute_harris_closed_loop），加进 cache
            try:
                manifest_name = _read_manifest_skill_name(d / "manifest.yaml")
                if manifest_name:
                    cache[manifest_name] = d
            except Exception:
                pass
    return cache


def _read_manifest_skill_name(manifest_path: Path) -> str | None:
    """轻量读取 manifest.yaml 的 skill.name；失败返回 None。

    不引入 yaml 依赖（学习层不该依赖 yaml）——做简单字符串扫描。
    """
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except Exception:
        return None
    # 查找 "skill:" 块下的 "name:" 字段（粗解析；99% 情况够用）
    in_skill_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "skill:":
            in_skill_block = True
            continue
        if in_skill_block:
            if not line.startswith(" ") and not line.startswith("\t"):
                # 离开 skill 块
                break
            if stripped.startswith("name:"):
                value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                return value or None
    return None


def record_snapshot(
    skill_name: str,
    payload: dict[str, Any],
    *,
    skill_version: str = "",
    code_origin: str = "algorithm",
) -> str | None:
    """写一条 snapshot；返回 snapshot_id，失败时 None。

    参数说明：
        skill_name: 目标 skill 名（如 "identify_model"）；用于决定落点
        payload: skill 特定的字段（predicted_*, inputs, attempts 等）
        skill_version: 可选，便于回放时区分版本
        code_origin: "algorithm" | "inline_skill" | "external_skill" | "consultant_tool"
                     用来区分这条样本是哪条调用路径产生的——学习训练时可分类
    """
    try:
        snap_dir = resolve_snapshot_dir(skill_name)
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        record = {
            "snapshot_id": snap_id,
            "ts": time.time(),
            "skill_name": skill_name,
            "skill_version": skill_version or None,
            "code_origin": code_origin,
            # payload 字段直接展开到顶层，方便 jq 查询
            **payload,
            # 部署后由 ExperienceStore 补齐
            "observable_outcomes": None,
        }
        out_path = snap_dir / f"{snap_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        return snap_id
    except Exception as e:
        # 学习层挂掉不能影响业务流；写日志告警即可
        logger.debug("record_snapshot 失败 (skill=%s): %s", skill_name, e)
        return None


def attach_outcome(
    skill_name: str,
    snapshot_id: str,
    outcome: dict[str, Any],
) -> bool:
    """把部署后观察值（observable_outcomes）补到已存在的 snapshot 里。

    供 V3.x 的 ExperienceStore 在监控任务 cron 触发时调用。
    """
    try:
        snap_dir = resolve_snapshot_dir(skill_name)
        path = snap_dir / f"{snapshot_id}.json"
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        existing = data.get("observable_outcomes") or {}
        if isinstance(existing, dict):
            existing.update(outcome)
            data["observable_outcomes"] = existing
        else:
            data["observable_outcomes"] = outcome
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        logger.debug("attach_outcome 失败 (skill=%s, id=%s): %s", skill_name, snapshot_id, e)
        return False


def iter_snapshots(
    skill_name: str,
    *,
    only_with_outcome: bool = False,
) -> Iterable[dict[str, Any]]:
    """遍历某个 skill 的所有 snapshot；recalibrate 脚本读训练数据时用。

    Args:
        only_with_outcome: True 时只产出已有 observable_outcomes 的样本
    """
    snap_dir = resolve_snapshot_dir(skill_name)
    if not snap_dir.exists():
        return
    for f in sorted(snap_dir.glob("*.json")):
        try:
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue
        if only_with_outcome and not isinstance(data.get("observable_outcomes"), dict):
            continue
        yield data


def clear_cache() -> None:
    """手动清空 skill_dir 缓存。仅测试用。"""
    global _SKILL_DIR_CACHE
    _SKILL_DIR_CACHE = None
