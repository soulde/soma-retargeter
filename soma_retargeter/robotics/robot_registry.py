# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for discovering and resolving robot retargeting targets from manifest files."""
import os
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable, ClassVar

import soma_retargeter.io.utils as utils

_BUILTIN_ROOT = utils.get_assets_dir() / 'robotics'

@dataclass
class RobotTarget:
    name: str
    manifest_dir: str
    desc: dict
    retarget_configs: dict
    csv_config_factory: object | None = None


@dataclass
class TargetRobot:
    """Stable registration contract for installed robot subpackages.

    Packages may use ``registry_api_version`` to select v0.2 configuration
    files without breaking installations using the original callback API.
    """
    registry_api_version: ClassVar[int] = 2
    name: str
    retargeter_config: str
    get_config_base: Callable[[], Path]
    get_mjcf_path: Callable[[], Path]
    retargeter_configs: dict[str, str] | None = None
    csv_config_factory: Callable | None = None


class RobotRegistry:
    """
    Discovers robot targets from ``manifest.json`` files so that new robots
    can be added without any code changes.

    Each robot directory contains a ``manifest.json`` with at minimum::

        {
            "name": "my_robot",
            "desc": { "urdf_path": "desc/robot.urdf" },
            "retarget_configs": {
                "soma": "configs/soma_to_robot_retargeter_config.json"
            }
        }

    A manifest may also declare multiple targets via a ``"targets"`` array.

    The registry scans a built-in directory on first access. Additional
    directories can be registered at runtime via :meth:`add_search_path`;
    targets found there **override** built-in targets that share the same
    name, so users can iterate on a config without modifying the repository.
    """

    def __init__(self, builtin_root: str):
        """
        Args:
            builtin_root: Absolute path to the directory tree containing bundled robot manifests.
        """
        self._builtin_root = builtin_root
        self._targets: dict[str, RobotTarget] = {}
        self._initialized = False
        self._extra_paths: list[str] = []
        self._plugins_loaded = False
        self._registered_targets: dict[str, RobotTarget] = {}

    def _adapt_target(self, robot: TargetRobot) -> RobotTarget:
        root = Path(robot.get_config_base()).resolve()
        configs = {"soma": robot.retargeter_config}
        if robot.retargeter_configs is None:
            soma_path = Path(robot.retargeter_config)
            smplx_path = soma_path.with_name(soma_path.name.replace("soma_to_", "smplx_to_"))
            if smplx_path != soma_path and (root / smplx_path).is_file():
                configs["smplx"] = str(smplx_path)
        else:
            configs.update(robot.retargeter_configs)
        return RobotTarget(
            name=robot.name, manifest_dir=str(root),
            desc={"xml_path": str(Path(robot.get_mjcf_path()).resolve())},
            retarget_configs=configs, csv_config_factory=robot.csv_config_factory,
        )

    def register(self, robot: TargetRobot):
        """Register a callback-style robot, retaining it across GUI reloads."""
        self._ensure_loaded()
        target = self._adapt_target(robot)
        self._registered_targets[target.name] = target
        self._targets[target.name] = target

    def add_search_path(self, path: str):
        """
        Register an additional directory tree to scan for manifests.

        Robots discovered here override built-in robots that share the same
        name, allowing users to iterate on configs without modifying the
        repository.
        """
        abs_path = os.path.abspath(path)
        self._extra_paths.append(abs_path)
        if self._initialized:
            self._scan_directory(abs_path)

    def get(self, name: str) -> RobotTarget:
        """Look up a robot target by name. Raises ``ValueError`` if unknown."""
        self._ensure_loaded()
        target = self._targets.get(name)
        if target is None:
            available = ", ".join(sorted(self._targets.keys()))
            raise ValueError(f"Unknown target type: [{name}]. Available: {available}")
        return target

    def list_names(self) -> list[str]:
        """Return a sorted list of all registered target names."""
        self._ensure_loaded()
        return sorted(self._targets.keys())

    def reload(self):
        """Clear all cached targets and re-scan the built-in root plus any extra paths."""
        self._targets = {}
        self._initialized = False
        self._plugins_loaded = False
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Scan the built-in root and any extra paths on first access."""
        if not self._initialized:
            self._scan_directory(self._builtin_root)
            self._initialized = True
            self._load_legacy_plugins()
            self._targets.update(self._registered_targets)
            for path in self._extra_paths:
                self._scan_directory(path)

    def _load_legacy_plugins(self):
        """Adapt installed ``soma_retargeter.targets`` packages to the v0.2 registry."""
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        for plugin in entry_points(group="soma_retargeter.targets"):
            try:
                register = plugin.load()
                def register_legacy(legacy):
                    self._targets[legacy.name] = self._adapt_target(legacy)
                register(register_legacy, TargetRobot)
            except Exception as error:
                print(f"[WARNING] Failed to load target package [{plugin.name}]: {error}")

    def _scan_directory(self, root: str):
        """Recursively walk ``root`` and load every ``manifest.json`` found.

        Args:
            root: Directory to scan. Silently skips if the path does not exist.
        """
        if not os.path.isdir(root):
            return
        for dirpath, _, filenames in os.walk(root):
            if "manifest.json" in filenames:
                self._load_manifest(os.path.join(dirpath, "manifest.json"))

    def _load_manifest(self, path: str):
        """Parse a ``manifest.json`` file and register all valid robot targets it declares.

        Args:
            path: Absolute path to the manifest file.
        """
        data = utils.load_json(path)
        manifest_dir = os.path.dirname(os.path.abspath(path))
        entries = data.get("targets", [data])
        for entry in entries:
            if "name" not in entry:
                print(f"[WARNING] Malformed manifest {path}: entry is missing 'name', skipping")
                continue
            name = entry["name"]
            if "desc" not in entry:
                print(f"[WARNING] Robot '{name}' in {path} has no 'desc', skipping")
                continue
            self._targets[name] = RobotTarget(
                name=name,
                manifest_dir=manifest_dir,
                desc=entry["desc"],
                retarget_configs=entry.get("retarget_configs", {}),
            )

registry = RobotRegistry(_BUILTIN_ROOT)


def register_robots_path(path: str):
    """Register an additional directory tree to scan for robot manifest.json files."""
    registry.add_search_path(path)


def list_available_targets() -> list[str]:
    """Return a sorted list of all registered robot target names."""
    return registry.list_names()


def list_available_soma_targets() -> list[str]:
    """Return a sorted list of robots that have a soma retargeter config configured.

    Use this instead of ``list_available_targets()`` anywhere a soma retargeting
    pipeline is required (e.g. the BVH converter, IK optimizer). Robots registered
    via the Robot Configurator but not yet fully configured are excluded.
    """
    return [name for name in registry.list_names()
            if registry.get(name).retarget_configs.get("soma")]


def list_available_source_targets(source: str) -> list[str]:
    """List configured robots for a source dataset."""
    return [name for name in registry.list_names()
            if get_source_config_path(registry.get(name), source)]


def get_source_config_path(target: RobotTarget, source: str) -> str | None:
    return target.retarget_configs.get(source)


def get_robot_asset_root(target: str) -> str:
    """Return the manifest directory for a robot target."""
    return registry.get(target).manifest_dir
