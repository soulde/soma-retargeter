#!/usr/bin/env python3
"""Convert soma-retargeter CSV motion to a BeyondMimic-style NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import newton
import warp as wp

from soma_retargeter.assets.csv import get_csv_config_for_target, load_csv
from soma_retargeter.assets.beyondmimic_npz import (
    finite_difference, quaternion_angular_velocity, resample_motion, save_npz,
)
from soma_retargeter.pipelines import utils as pipeline_utils


def convert_csv_to_npz(csv_path: str | Path, npz_path: str | Path, robot_type: str,
                       input_fps: float = 30.0, output_fps: float | None = None) -> None:
    """Load a retargeted CSV, evaluate Newton FK, and save motion tensors."""
    output_fps = input_fps if output_fps is None else output_fps
    csv_config = get_csv_config_for_target(robot_type)
    buffer = load_csv(str(csv_path), fps=input_fps, csv_config=csv_config)
    frames = np.asarray(buffer.data, dtype=np.float32)
    frames = resample_motion(frames, input_fps, output_fps)

    builder = newton.ModelBuilder()
    builder.add_mjcf(str(pipeline_utils.get_robot_mjcf_path(robot_type)))
    model = builder.finalize()
    if frames.shape[1] != model.joint_coord_count:
        raise ValueError(f"CSV has {frames.shape[1] - 7} joints, but model expects {model.joint_coord_count - 7}")

    body_pos, body_quat = [], []
    state = model.state()
    zero_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    for frame in frames:
        q = wp.array(frame, dtype=wp.float32)
        newton.eval_fk(model, q, wp.array(zero_qd, dtype=wp.float32), state)
        body_q = state.body_q.numpy().copy()
        body_pos.append(body_q[:, :3])
        body_quat.append(body_q[:, 3:7])

    body_pos = np.asarray(body_pos, dtype=np.float32)
    body_quat_xyzw = np.asarray(body_quat, dtype=np.float32)
    # BeyondMimic/Isaac Lab convention is wxyz; Newton stores xyzw.
    body_quat_wxyz = body_quat_xyzw[..., [3, 0, 1, 2]]
    joint_pos = frames[:, 7:]
    joint_vel = finite_difference(joint_pos, output_fps)
    body_lin_vel = finite_difference(body_pos, output_fps)
    body_ang_vel = np.stack(
        [quaternion_angular_velocity(body_quat_xyzw[:, i], output_fps)
         for i in range(body_quat_xyzw.shape[1])], axis=1)
    body_names = [str(label).rsplit("/", 1)[-1] for label in builder.body_label]
    # CSV headers describe the exported DoF order; Newton also contains fixed
    # joints (for example toe/wrist links), so builder.joint_label is not a
    # one-to-one list of the 29 coordinates in joint_pos.
    joint_names = [str(name).removesuffix("_dof") for name in csv_config.csv_header[7:]]
    save_npz(npz_path, fps=output_fps, joint_pos=joint_pos, joint_vel=joint_vel,
             body_pos_w=body_pos, body_quat_w=body_quat_wxyz,
             body_lin_vel_w=body_lin_vel, body_ang_vel_w=body_ang_vel,
             joint_names=joint_names, body_names=body_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--robot", required=True, choices=pipeline_utils.get_registered_targets())
    parser.add_argument("--input-fps", type=float,
                        help="Input FPS; inferred from a trailing _<fps>hz filename marker when omitted")
    parser.add_argument("--output-fps", type=float)
    args = parser.parse_args()
    convert_csv_to_npz(args.csv, args.npz, args.robot, args.input_fps, args.output_fps)


if __name__ == "__main__":
    main()
