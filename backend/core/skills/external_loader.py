"""External Skill 包加载器。

设计目标：把 ``backend/var/skills/<skill_name>/`` 这样的目录视作"目录即插件"，
启动时按 manifest 自动注册到现有 SkillRegistry。**完全不动**核心 skill 协议
（``BaseSkill`` / ``SkillRegistry`` / ``provider_registry``），只是在它们之上
加一个发现入口。

约定：

- 每个 skill 目录必须含 ``manifest.yaml``、``__init__.py`` 和入口模块。
- skill 的父目录（``backend/var/skills/``）会被自动加到 ``sys.path``，让
  skill 内部 import 走绝对路径（如 ``harris_closed_loop.implementation.skill``）。
- 任何 skill 加载失败只会落日志，不抛出，避免坏 skill 阻断后端启动。
- ``manifest.testing.run_on_load`` 为 true 时，加载完会跑该 skill 的 golden
  自检；不通过则**不注册**该 skill。

manifest 关键字段（详见 SKILL.md 示例）：

```yaml
skill:
  name: compute_harris_closed_loop
  version: 1.0.0
entry:
  class: harris_closed_loop.implementation.skill:HarrisClosedLoopSkill
providers:
  deadtime_estimator:
    - name: identified_model
      class: harris_closed_loop.providers.deadtime_identified:IdentifiedModelDeadtime
testing:
  run_on_load: false
```
"""
from __future__ import annotations

import importlib
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

from core.skills.base import BaseSkill
from core.skills.registry import registry

logger = logging.getLogger(__name__)

# 默认 skill 包根目录：backend/var/skills/
# 路径相对当前文件：core/skills/external_loader.py → ../../var/skills/
_DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parent.parent.parent / "var" / "skills"


class ExternalSkillLoadError(RuntimeError):
    """加载某个 external skill 失败时抛出；上层会 catch 并落日志。"""


def discover_external_skills(skills_root: Path | str | None = None) -> dict[str, Any]:
    """扫描 skills_root 下所有 manifest.yaml，按序注册。

    返回每个 skill 的加载状态摘要（成功 / 跳过 / 失败原因），调试和健康检查可用。
    """
    root = Path(skills_root) if skills_root else _DEFAULT_SKILLS_ROOT
    summary: dict[str, Any] = {"root": str(root), "skills": {}}
    if not root.exists():
        logger.info("外部 skill 根目录不存在，跳过：%s", root)
        return summary

    # 把 skill 根目录加到 sys.path，让 skill 内部能用绝对 import
    # （如 from harris_closed_loop.implementation.error_signal import …）
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue
        manifest_path = skill_dir / "manifest.yaml"
        if not manifest_path.exists():
            continue
        try:
            info = _load_one_skill(skill_dir, manifest_path)
            summary["skills"][skill_dir.name] = {"status": "loaded", **info}
            logger.info(
                "[external-skill] 已加载 %s v%s",
                info.get("skill_name"),
                info.get("version"),
            )
        except ExternalSkillLoadError as e:
            summary["skills"][skill_dir.name] = {"status": "failed", "error": str(e)}
            logger.error("[external-skill] 加载失败 %s：%s", skill_dir.name, e)
        except Exception as e:  # 兜底
            summary["skills"][skill_dir.name] = {
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=4),
            }
            logger.error(
                "[external-skill] 加载意外异常 %s：%s\n%s",
                skill_dir.name,
                e,
                traceback.format_exc(limit=4),
            )
    return summary


def _load_one_skill(skill_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    skill_meta = manifest.get("skill", {})
    name = str(skill_meta.get("name") or "")
    version = str(skill_meta.get("version") or "")
    lifecycle = manifest.get("lifecycle", {}) or {}
    if not name:
        raise ExternalSkillLoadError("manifest.skill.name 缺失")
    if lifecycle.get("enabled") is False:
        raise ExternalSkillLoadError("manifest.lifecycle.enabled = false，已禁用")

    entry = manifest.get("entry", {}) or {}
    entry_class = str(entry.get("class") or "")
    if not entry_class or ":" not in entry_class:
        raise ExternalSkillLoadError(
            f"manifest.entry.class 必须形如 'pkg.module:ClassName'，得到 {entry_class!r}"
        )

    skill_cls = _import_class(entry_class)
    if not isinstance(skill_cls, type) or not issubclass(skill_cls, BaseSkill):
        raise ExternalSkillLoadError(
            f"entry class {entry_class} 不是 BaseSkill 子类"
        )
    if getattr(skill_cls, "name", None) != name:
        raise ExternalSkillLoadError(
            f"skill 类的 name='{getattr(skill_cls, 'name', None)}' 与 manifest 不一致 ({name})"
        )

    # 把 skill 根目录路径暴露给类，便于实现读取自己的 config/ tests/ learning/
    skill_cls.skill_dir = skill_dir  # type: ignore[attr-defined]
    skill_cls.skill_version = version  # type: ignore[attr-defined]
    skill_cls.skill_manifest = manifest  # type: ignore[attr-defined]

    # 注册子 provider（如果有）。失败只记录，不阻断 skill 注册。
    provider_status = _register_providers(manifest.get("providers", {}) or {})

    # 加载前自检
    testing_cfg = manifest.get("testing", {}) or {}
    if testing_cfg.get("run_on_load"):
        ok, report = _run_golden(skill_dir, testing_cfg)
        if not ok:
            raise ExternalSkillLoadError(
                f"加载前 golden 自检未通过：{report}"
            )

    # 注册 skill。SkillRegistry 自带"重名报错"，这里在重名时尝试覆盖（升级）：
    # 删旧版本再注册新版本。
    if registry.get(name) is not None:
        # 私下覆盖（registry 没有公开 unregister API；直接动 _skills）
        registry._skills.pop(name, None)
    registry.register(skill_cls)

    return {
        "skill_name": name,
        "version": version,
        "rollout": lifecycle.get("rollout", "stable"),
        "providers": provider_status,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ExternalSkillLoadError(f"manifest yaml 解析失败：{e}") from e
    if not isinstance(data, dict):
        raise ExternalSkillLoadError("manifest 顶层必须是 mapping")
    return data


def _import_class(spec: str):
    module_path, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ExternalSkillLoadError(f"无法 import 模块 {module_path}：{e}") from e
    if not hasattr(mod, attr):
        raise ExternalSkillLoadError(f"模块 {module_path} 中找不到 {attr!r}")
    return getattr(mod, attr)


def _register_providers(providers_section: dict[str, Any]) -> dict[str, Any]:
    """按 manifest.providers 中声明的类，注册到全局 provider_registry。

    格式：
        providers:
          <category>:
            - name: <provider_name>
              class: <module>:<ClassName>
              priority: 100
              default: true
    """
    from core.shared import provider_registry  # 延迟 import，避免循环

    status: dict[str, list[dict[str, Any]]] = {}
    for category, items in providers_section.items():
        if not isinstance(items, list):
            continue
        category_status: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pname = str(item.get("name") or "")
            pclass = str(item.get("class") or "")
            if not pname or not pclass:
                category_status.append({"name": pname, "status": "skipped", "reason": "缺少 name 或 class"})
                continue
            try:
                cls = _import_class(pclass)
                instance = cls() if isinstance(cls, type) else cls
                # 兼容 provider 既可能是 class 也可能是已实例化的对象
                if not getattr(instance, "name", None):
                    instance.name = pname  # type: ignore[attr-defined]
                # 已存在同名 provider 时静默覆盖（升级语义）
                existing = provider_registry.get(category, pname)
                if existing is not None:
                    provider_registry._providers.get(category, {}).pop(pname, None)
                provider_registry.register(category, instance)
                category_status.append({"name": pname, "status": "registered"})
            except Exception as e:  # provider 失败不阻断 skill
                category_status.append({
                    "name": pname,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                })
        status[category] = category_status
    return status


def _run_golden(skill_dir: Path, testing_cfg: dict[str, Any]) -> tuple[bool, str]:
    """运行 skill 自带的 golden 测试（仅当 manifest 声明 run_on_load=true 时）。

    实现策略：把 ``tests/`` 当作普通 pytest 目录跑一次；失败原文返回。
    """
    tests_dir = skill_dir / "tests"
    if not tests_dir.exists():
        return True, "no tests/ dir, skipped"
    try:
        # 用子进程跑，避免污染主进程的 pytest 配置
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests_dir), "-q", "--no-header"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(skill_dir.parent),
        )
        if result.returncode == 0:
            return True, result.stdout.strip().splitlines()[-1] if result.stdout else "ok"
        out = (result.stdout or "") + (result.stderr or "")
        return False, out.strip()[-500:]
    except Exception as e:
        return False, f"golden 运行异常：{e}"
