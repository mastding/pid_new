"""统一 snapshot API 的单元测试。

测点：
1. record_snapshot 写到 fallback 目录（无对应 external skill 时）
2. record_snapshot 写到 external skill 自带目录（skill_name 后缀匹配）
3. attach_outcome 正确补回 observable_outcomes
4. iter_snapshots 过滤 only_with_outcome
5. 异常路径不抛出（写权限失败时 record_snapshot 返回 None）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.learning import (
    attach_outcome,
    iter_snapshots,
    record_snapshot,
    resolve_snapshot_dir,
)
from core.learning import snapshot as snapshot_mod


@pytest.fixture(autouse=True)
def reset_cache():
    """每个测试前清掉 skill_dir 缓存，避免互相污染。"""
    snapshot_mod.clear_cache()
    yield
    snapshot_mod.clear_cache()


@pytest.fixture
def isolated_var(tmp_path, monkeypatch):
    """把 _VAR_ROOT 重定向到临时目录，避免测试污染真实 var/。"""
    monkeypatch.setattr(snapshot_mod, "_VAR_ROOT", tmp_path)
    monkeypatch.setattr(
        snapshot_mod, "_EXTERNAL_SKILLS_DIR", tmp_path / "skills"
    )
    monkeypatch.setattr(
        snapshot_mod, "_FALLBACK_LEARNING_DIR", tmp_path / "learning" / "snapshots"
    )
    snapshot_mod.clear_cache()
    return tmp_path


def test_record_to_fallback_dir(isolated_var):
    """没有匹配的 external skill 时，写到 var/learning/snapshots/<skill>/"""
    snap_id = record_snapshot(
        "identify_model",
        {"inputs": {"loop_type": "flow"}, "best_predicted": {"K": 1.0}},
    )
    assert snap_id is not None
    expected_dir = isolated_var / "learning" / "snapshots" / "identify_model"
    assert expected_dir.exists()
    files = list(expected_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["snapshot_id"] == snap_id
    assert data["skill_name"] == "identify_model"
    assert data["code_origin"] == "algorithm"
    assert data["observable_outcomes"] is None
    assert data["inputs"]["loop_type"] == "flow"


def test_record_to_external_skill_dir_via_suffix_match(isolated_var):
    """skill_name 是 external skill 目录名的后缀时，落到该 skill 自己的目录。
    模拟：external dir 'harris_closed_loop' 接收 skill_name='compute_harris_closed_loop'
    """
    # 模拟一个 external skill 目录（仅含 manifest 占位）
    fake_skill = isolated_var / "skills" / "harris_closed_loop"
    fake_skill.mkdir(parents=True)
    manifest = fake_skill / "manifest.yaml"
    manifest.write_text(
        "skill:\n  name: compute_harris_closed_loop\n  version: 1.0.0\n",
        encoding="utf-8",
    )

    snap_id = record_snapshot(
        "compute_harris_closed_loop",
        {"eta_predicted": 0.5},
        skill_version="1.0.0-test",
    )
    assert snap_id is not None
    expected = fake_skill / "learning" / "snapshots"
    assert expected.exists()
    files = list(expected.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["skill_version"] == "1.0.0-test"
    assert data["eta_predicted"] == 0.5


def test_resolve_snapshot_dir_suffix(isolated_var):
    """resolve_snapshot_dir 应该按 manifest 真名匹配。"""
    fake_skill = isolated_var / "skills" / "harris_closed_loop"
    fake_skill.mkdir(parents=True)
    (fake_skill / "manifest.yaml").write_text(
        "skill:\n  name: compute_harris_closed_loop\n", encoding="utf-8"
    )

    # manifest 真名
    p1 = resolve_snapshot_dir("compute_harris_closed_loop")
    assert p1 == fake_skill / "learning" / "snapshots"

    # 目录名（短名）也命中
    p2 = resolve_snapshot_dir("harris_closed_loop")
    assert p2 == fake_skill / "learning" / "snapshots"

    # 完全不相关的 skill 走 fallback
    p3 = resolve_snapshot_dir("evaluate_tuning")
    assert "learning/snapshots/evaluate_tuning" in p3.as_posix()


def test_attach_outcome_appends(isolated_var):
    snap_id = record_snapshot("identify_model", {"best": {"K": 1.0}})
    assert snap_id is not None

    ok = attach_outcome(
        "identify_model",
        snap_id,
        {"actual_deploy_label": "success", "delta_eta_7d": 0.18},
    )
    assert ok

    files = list(
        (isolated_var / "learning" / "snapshots" / "identify_model").glob("*.json")
    )
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["observable_outcomes"]["actual_deploy_label"] == "success"
    assert data["observable_outcomes"]["delta_eta_7d"] == 0.18

    # 再次 attach 应该合并而不是覆盖
    ok2 = attach_outcome(
        "identify_model",
        snap_id,
        {"human_label": "good", "delta_eta_7d": 0.22},  # 覆盖一个字段
    )
    assert ok2
    data2 = json.loads(files[0].read_text(encoding="utf-8"))
    assert data2["observable_outcomes"]["actual_deploy_label"] == "success"  # 保留
    assert data2["observable_outcomes"]["human_label"] == "good"             # 新增
    assert data2["observable_outcomes"]["delta_eta_7d"] == 0.22              # 覆盖


def test_attach_outcome_missing_returns_false(isolated_var):
    assert attach_outcome("identify_model", "does_not_exist", {"x": 1}) is False


def test_iter_snapshots_filters_outcome(isolated_var):
    id_a = record_snapshot("evaluate_tuning", {"a": 1})
    id_b = record_snapshot("evaluate_tuning", {"a": 2})
    attach_outcome("evaluate_tuning", id_b, {"actual_deploy_label": "success"})

    all_snaps = list(iter_snapshots("evaluate_tuning"))
    assert len(all_snaps) == 2

    with_outcome = list(iter_snapshots("evaluate_tuning", only_with_outcome=True))
    assert len(with_outcome) == 1
    assert with_outcome[0]["snapshot_id"] == id_b


def test_record_failure_returns_none_without_raising(isolated_var, monkeypatch):
    """模拟磁盘写失败：record_snapshot 不能抛异常。"""
    def boom(*a, **kw):
        raise OSError("disk full")

    # 让 mkdir 失败
    monkeypatch.setattr(Path, "mkdir", boom)
    snap_id = record_snapshot("identify_model", {"x": 1})
    assert snap_id is None


def test_payload_can_have_complex_values(isolated_var):
    """payload 含 numpy-like / nested dict 时，用 default=str 兜底序列化。"""
    snap_id = record_snapshot(
        "identify_model",
        {
            "attempts_summary": [{"r2": 0.95, "model": "FOPDT"}],
            "weird_value": object(),  # 非 JSON-native，靠 default=str
        },
    )
    assert snap_id is not None
