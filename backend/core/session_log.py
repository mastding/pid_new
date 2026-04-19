"""会话持久化：把每次 /api/tune/stream 或 /api/consult/stream 的全部 SSE
事件落盘成 JSONL，再加一份 meta.json 摘要，便于事后回放与分析。

目录布局：
  backend/var/sessions/<YYYYMMDD>/<task_id>/
    meta.json     —— 任务元数据（CSV、回路、最终结果摘要、用时）
    events.jsonl  —— 每行一个 SSE 事件，原样保留

设计取舍：
  * 文件优先，无数据库依赖。一次性读全表也够快（量大时再加索引）。
  * 失败/中断也保留事件流，meta.error 记原因。
  * 每次 finalize 时做轻量清理：保留最近 MAX_KEEP 个会话。
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

logger = logging.getLogger(__name__)

# 默认会话根目录：backend/var/sessions
ROOT = Path(__file__).resolve().parent.parent / "var" / "sessions"
MAX_KEEP = 200  # 滚动保留最近 N 个会话


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _today_dir() -> str:
    return datetime.now().strftime("%Y%m%d")


class SessionRecorder:
    """单次会话的事件记录器。线程安全：内部加锁保证 finalize 与 record 不冲突。"""

    def __init__(self, *, kind: str, meta_init: dict[str, Any]) -> None:
        self.kind = kind  # "tune" | "consult"
        self.task_id = uuid.uuid4().hex[:12]
        self.day = _today_dir()
        self.dir = ROOT / self.day / self.task_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.start_ts = time.time()
        self.n_events = 0
        self.last_stage: str | None = None
        self.final_result: dict[str, Any] | None = None
        self.error: str | None = None
        self.meta: dict[str, Any] = {
            "task_id": self.task_id,
            "kind": kind,
            "created_at": _now_iso(),
            **meta_init,
        }
        self._lock = asyncio.Lock()
        # 立即落一个空的 meta，方便列表立刻能看到正在跑的任务
        self._write_meta_sync()

    def _write_meta_sync(self) -> None:
        try:
            self.meta_path.write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.exception("写 meta.json 失败")

    async def record(self, event: dict[str, Any]) -> None:
        """写一行事件到 jsonl；同时维护 meta 的轻量摘要字段。"""
        async with self._lock:
            self.n_events += 1
            line = json.dumps(
                {"_seq": self.n_events, "_t": time.time() - self.start_ts, **event},
                ensure_ascii=False,
                default=str,
            )
            try:
                with self.events_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                logger.exception("写 events.jsonl 失败")

            # 顺手挑出 meta 里关心的几个字段
            ev_type = event.get("type")
            stage = event.get("stage")
            if stage:
                self.last_stage = stage
            if ev_type == "result":
                self.final_result = event.get("data") or {}
            elif ev_type == "error":
                self.error = event.get("message") or event.get("error_code") or "unknown"

    async def finalize(self) -> None:
        """收尾：补全 meta，刷新到磁盘，并触发滚动清理。"""
        logger.debug("finalize task_id=%s n_events=%d", self.task_id, self.n_events)
        async with self._lock:
            self.meta["ended_at"] = _now_iso()
            self.meta["duration_s"] = round(time.time() - self.start_ts, 2)
            self.meta["n_events"] = self.n_events
            self.meta["last_stage"] = self.last_stage
            if self.error:
                self.meta["error"] = self.error
                self.meta["status"] = "error"
            else:
                self.meta["status"] = "ok"

            if self.final_result:
                ev = self.final_result.get("evaluation") or {}
                ws = self.final_result.get("window_selection") or {}
                model = self.final_result.get("model") or {}
                pid = self.final_result.get("pid_params") or {}
                self.meta["summary"] = {
                    "passed": ev.get("passed"),
                    "performance_score": ev.get("performance_score"),
                    "final_rating": ev.get("final_rating"),
                    "is_stable": ev.get("is_stable"),
                    "decay_ratio": ev.get("decay_ratio"),
                    "overshoot_percent": ev.get("overshoot_percent"),
                    "model_type": model.get("model_type"),
                    "r2_score": model.get("r2_score"),
                    "confidence": model.get("confidence"),
                    "strategy": pid.get("strategy"),
                    "selection_mode": ws.get("mode"),
                    "chosen_index": ws.get("chosen_index"),
                    "deterministic_index": ws.get("deterministic_index"),
                    "agreed_with_deterministic": ws.get("agreed_with_deterministic"),
                }

            self._write_meta_sync()

        # 清理放锁外，免得阻塞下一个并发会话
        try:
            await asyncio.to_thread(_rotate_old_sessions, MAX_KEEP)
        except Exception:
            logger.exception("清理旧会话失败")


def _rotate_old_sessions(max_keep: int) -> None:
    """按 created_at 时间戳保留最近 max_keep 个，老的整目录删掉。"""
    if not ROOT.exists():
        return
    sessions: list[tuple[float, Path]] = []
    for day_dir in ROOT.iterdir():
        if not day_dir.is_dir():
            continue
        for sess in day_dir.iterdir():
            if not sess.is_dir():
                continue
            try:
                mtime = (sess / "meta.json").stat().st_mtime
            except FileNotFoundError:
                mtime = sess.stat().st_mtime
            sessions.append((mtime, sess))
    if len(sessions) <= max_keep:
        return
    sessions.sort(key=lambda x: x[0], reverse=True)
    for _, old in sessions[max_keep:]:
        try:
            shutil.rmtree(old, ignore_errors=True)
        except Exception:
            logger.warning("删除旧会话失败: %s", old)
    # 再把空的 day 目录也清掉
    for day_dir in ROOT.iterdir():
        if day_dir.is_dir() and not any(day_dir.iterdir()):
            day_dir.rmdir()


# ── 通用包装器：把任意 SSE 异步生成器套上记录 ────────────────────────────────


async def record_stream(
    *,
    kind: str,
    meta_init: dict[str, Any],
    gen: AsyncGenerator[dict[str, Any], None],
    inject_task_id: bool = True,
) -> AsyncGenerator[dict[str, Any], None]:
    """包装一个 SSE 事件生成器：
      1. 创建 SessionRecorder
      2. 先发一个 session_start 事件给前端（含 task_id）
      3. 透传所有事件，并落盘
      4. finally 里 finalize
    """
    rec = SessionRecorder(kind=kind, meta_init=meta_init)
    if inject_task_id:
        start_ev = {"type": "session_start", "task_id": rec.task_id, "kind": kind}
        await rec.record(start_ev)
        yield start_ev

    try:
        async for ev in gen:
            await rec.record(ev)
            yield ev
    except Exception as exc:
        err_ev = {"type": "error", "message": f"流被中断: {exc}", "error_code": "STREAM_ABORTED"}
        await rec.record(err_ev)
        yield err_ev
        raise
    finally:
        await rec.finalize()


# ── 列表与详情读取（供 REST 端点使用）────────────────────────────────────────


def list_sessions(limit: int = 100, kind: str | None = None) -> list[dict[str, Any]]:
    """扫描所有 meta.json，按时间倒序返回。"""
    if not ROOT.exists():
        return []
    out: list[dict[str, Any]] = []
    for day_dir in ROOT.iterdir():
        if not day_dir.is_dir():
            continue
        for sess in day_dir.iterdir():
            meta_path = sess / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if kind and meta.get("kind") != kind:
                continue
            meta["_day"] = day_dir.name
            out.append(meta)
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out[:limit]


def get_session(task_id: str) -> dict[str, Any] | None:
    """按 task_id 找会话；返回 {meta, events}。找不到返回 None。"""
    if not ROOT.exists():
        return None
    for day_dir in ROOT.iterdir():
        sess = day_dir / task_id
        if sess.is_dir():
            try:
                meta = json.loads((sess / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                meta = {"task_id": task_id, "error": "meta.json 不可读"}
            events: list[dict[str, Any]] = []
            ep = sess / "events.jsonl"
            if ep.is_file():
                with ep.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return {"meta": meta, "events": events}
    return None


def delete_session(task_id: str) -> bool:
    if not ROOT.exists():
        return False
    for day_dir in ROOT.iterdir():
        sess = day_dir / task_id
        if sess.is_dir():
            shutil.rmtree(sess, ignore_errors=True)
            return True
    return False
