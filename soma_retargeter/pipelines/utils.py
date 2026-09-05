# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
import os

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.assets.usd as usd_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()


class TargetType(IntEnum):
    """Enumeration of supported target model types."""
    UNITREE_G1 = auto()
    DR02 = auto()
    CHOCOLATE = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}

_TARGET_TYPE_TO_STR = {
    TargetType.UNITREE_G1 : "unitree_g1",
    TargetType.DR02 : "dr02",
    TargetType.CHOCOLATE : "chocolate",
}
_STR_TO_TARGET_TYPE = {s : t for t, s in _TARGET_TYPE_TO_STR.items()}


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


def get_target_str_from_type(target: TargetType) -> str:
    """
    Get the string name associated with a given target type.

    Args:
        target (TargetType): The target type enum value.

    Returns:
        str: The string representation of the target type.
    """
    return _TARGET_TYPE_TO_STR[target]


def get_target_type_from_str(target: str) -> TargetType:
    """
    Convert a string to its corresponding TargetType enum value.

    Args:
        target (str): The string representation of a target.

    Returns:
        TargetType: The corresponding target type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid target type.
    """
    try:
        return _STR_TO_TARGET_TYPE[target]
    except KeyError:
        allowed = ", ".join(_STR_TO_TARGET_TYPE.keys())
        raise ValueError(f"Unknown target type: [{target}]. Allowed values: {allowed}") from None


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


def get_retargeter_config(source: SourceType, target: TargetType) -> dict:
    """
    Load the retargeter configuration between a specific source and target.

    Args:
        source (SourceType): The source type.
        target (TargetType): The target type.

    Returns:
        dict: The loaded JSON configuration for the retargeter.

    Raises:
        ValueError: If the source or target type is not supported.
    """
    if target == TargetType.UNITREE_G1:
        relative = 'unitree_g1/soma_to_g1_retargeter_config.json'
    elif target == TargetType.DR02:
        relative = 'dr02/soma_to_dr02_retargeter_config.json'
    elif target == TargetType.CHOCOLATE:
        relative = 'chocolate/soma_to_chocolate_retargeter_config.json'
    else:
        raise ValueError(f"Unknown target type [{target}].")

    if source != SourceType.SOMA:
        raise ValueError(f"Unknown source type [{source}] for target [{target}].")

    return io_utils.load_json(resolve_config_path(relative))


# Robots whose assets are not shipped in this repository. Their configs and
# models live in external packages (e.g. soma-chocolate) exposing the same
# '<robot>/...' layout via get_config_dir().
_EXTERNAL_ROBOT_PACKAGES = {
    'chocolate': 'soma_chocolate',
}


def _get_external_config_dir(robot: str):
    package = _EXTERNAL_ROBOT_PACKAGES[robot]
    try:
        module = __import__(package)
    except ImportError:
        raise ImportError(
            f"Robot [{robot}] requires the external package [{package}]. "
            f"Install it with: pip install -e ~/soma-{robot}") from None
    return module.get_config_dir()


def resolve_config_path(relative: str):
    """
    Resolve a config-relative path, dispatching external robots to their packages.

    Args:
        relative (str): Path relative to the configs directory, using the
            '<robot>/...' convention (e.g. 'chocolate/mjcf/chocolate_robot.xml').

    Returns:
        Path to the configuration file.
    """
    top = str(relative).split('/')[0].split(os.sep)[0]
    if top in _EXTERNAL_ROBOT_PACKAGES:
        return _get_external_config_dir(top) / relative
    return io_utils.get_config_file(relative)


def get_robot_mjcf_path(target: TargetType):
    """
    Retrieve the MJCF model path for a given target robot.

    The Unitree G1 model is downloaded from the Newton asset server, while the
    remaining robots ship their MJCF and meshes inside this package.

    Args:
        target (TargetType): The target robot type.

    Returns:
        Path to the robot MJCF file (str for downloaded Newton assets).

    Raises:
        ValueError: If the target robot type is not supported.
    """
    import newton

    if target == TargetType.UNITREE_G1:
        return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"
    if target == TargetType.DR02:
        return resolve_config_path('dr02/mjcf/dr02_robot.xml')
    if target == TargetType.CHOCOLATE:
        return resolve_config_path('chocolate/mjcf/chocolate_robot.xml')

    raise ValueError(f"Unknown target type [{target}].")
