"""统一的学习反馈基础设施。

主要导出：
- ``record_snapshot``：写一条决策样本到对应 skill 的 learning/snapshots/
- ``attach_outcome``：部署后回填 observable_outcomes（ExperienceStore 用，V3.x）
- ``iter_snapshots``：离线 recalibrate 脚本读取样本时用

设计目标：算法层 / 内置 skill / 外部 skill 都能用同一套 API 写样本，
落到统一目录，避免每处自己造轮子。
"""

from core.learning.snapshot import (
    attach_outcome,
    iter_snapshots,
    record_snapshot,
    resolve_snapshot_dir,
)

__all__ = [
    "record_snapshot",
    "attach_outcome",
    "iter_snapshots",
    "resolve_snapshot_dir",
]
