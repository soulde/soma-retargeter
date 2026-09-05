# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from typing import Callable, Dict, List

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.assets.usd as usd_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}


@dataclass
class TargetRobot:
    """
    Description of a retargeting target robot.

    Robots register through ``register_target``; built-ins are registered at
    import time and external packages register through the
    ``soma_retargeter.targets`` entry point group.

    Attributes:
        name: Target name used in configs and the UI (e.g. "unitree_g1").
        retargeter_config: Retargeter config path relative to the robot's
            config base directory.
        get_config_base: Callable returning the directory that contains the
            ``<name>/...`` config subtree.
        get_mjcf_path: Callable returning the path to the robot MJCF model.
    """
    name: str
    retargeter_config: str
    get_config_base: Callable[[], Path]
    get_mjcf_path: Callable[[], Path]


_TARGET_REGISTRY: Dict[str, TargetRobot] = {}
_EXTERNAL_TARGETS_DISCOVERED = False


def register_target(robot: TargetRobot) -> None:
    """
    Register a retargeting target robot.

    Args:
        robot (TargetRobot): The robot description. A duplicate name overwrites
            the previous registration.
    """
    _TARGET_REGISTRY[robot.name] = robot


def _g1_mjcf_path() -> Path:
    import newton

    return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"


def _register_builtin_targets() -> None:
    configs_dir = io_utils.get_configs_dir
    register_target(TargetRobot(
        name='unitree_g1',
        retargeter_config='unitree_g1/soma_to_g1_retargeter_config.json',
        get_config_base=configs_dir,
        get_mjcf_path=_g1_mjcf_path))
    register_target(TargetRobot(
        name='dr02',
        retargeter_config='dr02/soma_to_dr02_retargeter_config.json',
        get_config_base=configs_dir,
        get_mjcf_path=lambda: io_utils.get_config_file('dr02', 'mjcf/dr02_robot.xml')))


def _discover_external_targets() -> None:
    """Load external robot packages via the soma_retargeter.targets entry points."""
    global _EXTERNAL_TARGETS_DISCOVERED
    if _EXTERNAL_TARGETS_DISCOVERED:
        return
    _EXTERNAL_TARGETS_DISCOVERED = True
    from importlib.metadata import entry_points

    for entry_point in entry_points(group='soma_retargeter.targets'):
        try:
            register = entry_point.load()
            register(register_target, TargetRobot)
        except Exception as e:
            print(f"[WARNING]: Failed to load target package [{entry_point.name}]: {e}")


def _get_registered_robot(target: str) -> TargetRobot:
    """
    Look up a registered robot by name, discovering external packages once.

    Args:
        target (str): The target robot name.

    Returns:
        TargetRobot: The registered robot description.

    Raises:
        ValueError: If the name is not registered.
    """
    _register_builtin_targets()
    _discover_external_targets()
    try:
        return _TARGET_REGISTRY[target]
    except KeyError:
        allowed = ", ".join(sorted(_TARGET_REGISTRY.keys()))
        raise ValueError(f"Unknown target type: [{target}]. Allowed values are: {allowed}") from None


def get_registered_targets() -> List[str]:
    """
    List all registered target robot names (built-in and external packages).

    Returns:
        list[str]: Sorted target names.
    """
    _register_builtin_targets()
    _discover_external_targets()
    return sorted(_TARGET_REGISTRY.keys())


def get_source_str_from_type(source: SourceType) -> str:
    """
    Get the string name associated with a given source type.

    Args:
        source (SourceType): The source type enum value.

    Returns:
        str: The string representation of the source type.
    """
    return _SOURCE_TYPE_TO_STR[source]


def get_source_type_from_str(source: str) -> SourceType:
    """
    Convert a string to its corresponding SourceType enum value.

    Args:
        source (str): The string representation of a source.

    Returns:
        SourceType: The corresponding source type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid source type.
    """
    try:
        return _STR_TO_SOURCE_TYPE[source]
    except KeyError:
        allowed = ", ".join(_STR_TO_SOURCE_TYPE.keys())
        raise ValueError(f"Unknown source type: [{source}]. Allowed values: {allowed}") from None


def get_source_model_mesh(source: SourceType, skeleton) -> dict:
    """
    Retrieve model mesh for a given source type.

    Args:
        source (SourceType): The source type for which properties should be retrieved.
        skeleton: The skeleton associated with the source model, used for loading the mesh.

    Returns:
        SkeletalMesh: The skeleton mesh for the given source type.

    Raises:
        ValueError: If the source type is not recognized.
    """
    if source == SourceType.SOMA:
        return usd_utils.load_skeletal_mesh_from_usd(
            str(io_utils.get_config_file('soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    raise ValueError(f"Unknown source type {source}.")


def get_retargeter_config(source: SourceType, target: str) -> dict:
    """
    Load the retargeter configuration between a specific source and target.

    Args:
        source (SourceType): The source type.
        target (str): The target robot name (e.g. "unitree_g1", "chocolate").

    Returns:
        dict: The loaded JSON configuration for the retargeter.

    Raises:
        ValueError: If the source or target type is not supported.
    """
    if source != SourceType.SOMA:
        raise ValueError(f"Unknown source type [{source}] for target [{target}].")

    robot = _get_registered_robot(target)
    return io_utils.load_json(robot.get_config_base() / robot.retargeter_config)


def resolve_config_path(relative: str):
    """
    Resolve a config-relative path, dispatching registered robots to their packages.

    Args:
        relative (str): Path relative to a configs root, using the '<robot>/...'
            convention (e.g. 'chocolate/mjcf/chocolate_robot.xml' or
            'soma/soma_zero_frame0.bvh').

    Returns:
        Path to the configuration file.
    """
    top = str(relative).split('/')[0]
    if top in _TARGET_REGISTRY:
        return _TARGET_REGISTRY[top].get_config_base() / relative
    return io_utils.get_config_file(relative)


def get_robot_mjcf_path(target: str) -> Path:
    """
    Retrieve the MJCF model path for a given target robot.

    Args:
        target (str): The target robot name.

    Returns:
        Path to the robot MJCF file.

    Raises:
        ValueError: If the target robot type is not registered.
    """
    return _get_registered_robot(target).get_mjcf_path()
