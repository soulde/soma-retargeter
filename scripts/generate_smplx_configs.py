#!/usr/bin/env python3
"""Generate smplx_to_<robot> retargeter and scaler configs from the existing
soma_to_<robot> configs.

The scaler's per-joint rotation offsets align the human joint frame with the
robot link frame. They are derived from the calibrated SOMA values instead of
being hand-tuned: at the rest pose both sources must produce the same effector
frame (q_soma * q_offset_soma == q_smplx * q_offset_smplx), so

    q_offset_smplx = q_smplx^-1 * q_soma * q_offset_soma

Run inside the soma-retargeter environment::

    .venv/bin/python scripts/generate_smplx_configs.py \
        [--chocolate-config-dir /home/jvwei/soma-chocolate/soma_chocolate/configs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

from soma_retargeter.assets.bvh import load_bvh
from soma_retargeter.assets.smplx import create_smplx_skeleton
import soma_retargeter.utils.io_utils as io_utils

# SOMA joint name -> SMPL-X body joint name
SOMA_TO_SMPLX = {
    "Hips": "pelvis",
    "Chest": "spine3",
    "Neck1": "neck",
    "LeftLeg": "left_hip", "RightLeg": "right_hip",
    "LeftShin": "left_knee", "RightShin": "right_knee",
    "LeftFoot": "left_ankle", "RightFoot": "right_ankle",
    "LeftToe": "left_foot", "RightToe": "right_foot",
    "LeftToeBase": "left_foot", "RightToeBase": "right_foot",  # SOMA BVH uses *ToeBase
    "LeftArm": "left_shoulder", "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow", "RightForeArm": "right_elbow",
    "LeftHand": "left_wrist", "RightHand": "right_wrist",
}

SMPLX_PARENTS = {
    "pelvis": "",
    "spine3": "pelvis",
    "neck": "spine3",
    "left_hip": "pelvis", "right_hip": "pelvis",
    "left_knee": "left_hip", "right_knee": "right_hip",
    "left_ankle": "left_knee", "right_ankle": "right_knee",
    "left_foot": "left_ankle", "right_foot": "right_ankle",
    "left_shoulder": "spine3", "right_shoulder": "spine3",
    "left_elbow": "left_shoulder", "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow", "right_wrist": "right_elbow",
}


def rest_global_rotations(skeleton, world_rotation: R | None = None,
                          local_transforms=None) -> dict[str, R]:
    """Global rest-pose rotation per joint (identity root world transform).

    Args:
        world_rotation: Optional rotation applied on top of every global
            rotation, e.g. the SpaceConverter rotation the soma pipeline
            applies to bring SOMA BVH data (Y-up) into the pipeline frame.
        local_transforms: Optional per-joint local transforms (e.g. one frame
            of an animation); defaults to the skeleton reference pose.
    """
    if local_transforms is None:
        local_transforms = skeleton.reference_local_transforms
    global_tx = np.asarray(skeleton.compute_global_transforms(
        local_transforms, wp.transform_identity()))
    result = {}
    for i, name in enumerate(skeleton.joint_names):
        # each row is [px, py, pz, qx, qy, qz, qw]
        rot = R.from_quat(global_tx[i, 3:7])
        if world_rotation is not None:
            rot = world_rotation * rot
        result[name] = rot
    return result


def generate_scaler_config(soma_scaler: dict, soma_rest: dict, smplx_rest: dict) -> dict:
    joint_scales, joint_parents, joint_offsets = {}, {}, {}
    for soma_name, smplx_name in SOMA_TO_SMPLX.items():
        if soma_name not in soma_scaler["joint_scales"]:
            continue
        if smplx_name in joint_scales:
            continue
        joint_scales[smplx_name] = soma_scaler["joint_scales"][soma_name]
        joint_parents[smplx_name] = SMPLX_PARENTS[smplx_name]

        t_offset, q_offset = soma_scaler["joint_offsets"][soma_name]
        q_soma = soma_rest.get(soma_name)
        if q_soma is None:
            # Joint not present in the rest BVH (e.g. LeftToe vs LeftToeBase);
            # keep the SOMA offset unchanged.
            joint_offsets[smplx_name] = [t_offset, q_offset]
            continue
        q_smplx = smplx_rest[smplx_name]
        q_new = q_smplx.inv() * q_soma * R.from_quat(q_offset)
        joint_offsets[smplx_name] = [t_offset, q_new.as_quat().tolist()]  # xyzw

    # SMPL-X mocap subjects stand with the pelvis around 0.90 m while the SOMA
    # corpus the original assumption was tuned on sits near 0.95 m; scale the
    # assumption accordingly so the robot is not pulled into a crouch.
    height_assumption = soma_scaler["human_height_assumption"] * 0.90 / 0.95
    return {
        "robot_type": soma_scaler["robot_type"],
        "human_root_name": "pelvis",
        "human_height_assumption": round(height_assumption, 3),
        "joint_scales": joint_scales,
        "joint_parents": joint_parents,
        "joint_offsets": joint_offsets,
    }


def generate_retargeter_config(soma_config: dict, robot_dir: str) -> dict:
    config = dict(soma_config)
    config["initialization_pose"] = None
    config["feet_joint_names"] = ["left_ankle", "right_ankle"]

    ik_map = {}
    for soma_name, entry in soma_config["ik_map"].items():
        smplx_name = SOMA_TO_SMPLX[soma_name]
        if smplx_name not in ik_map:
            ik_map[smplx_name] = entry
    config["ik_map"] = ik_map
    return config


def process_robot(config_dir: Path, robot_dir: str, soma_rest: dict, smplx_rest: dict) -> None:
    # Config file names follow the robot's own naming (soma_to_g1_*, soma_to_dr02_*,
    # soma_to_chocolate_*); locate them by prefix instead of guessing.
    scaler_candidates = sorted(config_dir.glob(f"{robot_dir}/soma_to_*_scaler_config.json"))
    retargeter_candidates = sorted(config_dir.glob(f"{robot_dir}/soma_to_*_retargeter_config.json"))
    if len(scaler_candidates) != 1 or len(retargeter_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one soma_to_* config pair in {config_dir / robot_dir}, found "
            f"{scaler_candidates} and {retargeter_candidates}")

    soma_scaler = json.loads(scaler_candidates[0].read_text())
    soma_config = json.loads(retargeter_candidates[0].read_text())

    smplx_scaler = generate_scaler_config(soma_scaler, soma_rest, smplx_rest)
    out_dir = config_dir / robot_dir
    (out_dir / "smplx_to_scaler_config.json").write_text(json.dumps(smplx_scaler, indent=2) + "\n")

    robot_name = retargeter_candidates[0].name.replace("soma_to_", "").replace("_retargeter_config.json", "")
    smplx_config = generate_retargeter_config(soma_config, robot_dir)
    smplx_config["human_robot_scaler_config"] = f"{robot_dir}/smplx_to_scaler_config.json"
    (out_dir / f"smplx_to_{robot_name}_retargeter_config.json").write_text(json.dumps(smplx_config, indent=2) + "\n")
    print(f"Wrote smplx configs for [{robot_name}] in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chocolate-config-dir", type=Path, default=None,
                        help="soma-chocolate package configs directory")
    args = parser.parse_args()

    soma_skel, soma_anim = load_bvh(str(io_utils.get_config_file("soma", "soma_zero_frame0.bvh")))
    # The soma pipeline feeds BVH data through a SpaceConverter ("Mujoco"
    # facing direction) that pre-rotates everything by Rx(90) to reach the
    # pipeline frame; the SMPL-X loader does this internally instead.
    soma_world_rotation = R.from_euler("x", 90, degrees=True)
    smplx_skel = create_smplx_skeleton(up_axis="+Y")
    # Calibrate against the zero-frame ANIMATION pose, not the bind pose:
    # SOMA BVH motion channels carry a constant world alignment on top of the
    # bind offsets, so frame 0 of the zero-pose clip is the true "standing
    # T-pose in animation convention" reference the retargeter is tuned for.
    soma_rest = rest_global_rotations(
        soma_skel, soma_world_rotation, local_transforms=soma_anim.get_local_transforms(0))
    smplx_rest = rest_global_rotations(smplx_skel)

    process_robot(io_utils.get_configs_dir(), "unitree_g1", soma_rest, smplx_rest)
    process_robot(io_utils.get_configs_dir(), "dr02", soma_rest, smplx_rest)
    if args.chocolate_config_dir is not None:
        process_robot(args.chocolate_config_dir, "chocolate", soma_rest, smplx_rest)


if __name__ == "__main__":
    main()
