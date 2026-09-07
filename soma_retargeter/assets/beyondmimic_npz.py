"""BeyondMimic-compatible motion export helpers.

The input frame layout is the internal SOMA layout: root translation (m),
root quaternion (Newton xyzw), followed by joint coordinates (rad).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """Differentiate samples along time, returning units per second."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, 1.0 / float(fps), axis=0, edge_order=1).astype(np.float32)


def resample_motion(frames: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    """Resample root translation and joints linearly and root rotation with SLERP."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] < 8:
        raise ValueError("frames must have shape (N, 8+), with root pose and joints")
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError("frame rates must be positive")
    if frames.shape[0] == 0 or np.isclose(input_fps, output_fps):
        return frames.copy()
    duration = (frames.shape[0] - 1) / float(input_fps)
    count = int(round(duration * output_fps)) + 1
    new_t = np.minimum(np.arange(count, dtype=np.float64) / float(output_fps), duration)
    old_t = np.arange(frames.shape[0], dtype=np.float64) / float(input_fps)
    out = np.empty((count, frames.shape[1]), dtype=np.float32)
    out[:, :3] = np.stack([np.interp(new_t, old_t, frames[:, i]) for i in range(3)], axis=1)
    # scipy uses the same xyzw order as Newton.
    rotations = Slerp(old_t, Rotation.from_quat(frames[:, 3:7]))(new_t)
    out[:, 3:7] = rotations.as_quat().astype(np.float32)
    out[:, 7:] = np.stack(
        [np.interp(new_t, old_t, frames[:, i]) for i in range(7, frames.shape[1])], axis=1
    )
    return out


def quaternion_angular_velocity(quaternions_xyzw: np.ndarray, fps: float) -> np.ndarray:
    """Compute world-frame angular velocity from xyzw quaternion samples."""
    q = np.asarray(quaternions_xyzw, dtype=np.float64)
    if len(q) < 2:
        return np.zeros((len(q), 3), dtype=np.float32)
    if len(q) == 2:
        rel = (Rotation.from_quat(q[0]).inv() * Rotation.from_quat(q[1])).as_rotvec() * float(fps)
        return np.repeat(rel[None], 2, axis=0).astype(np.float32)
    rotations = Rotation.from_quat(q)
    # Match Isaac Lab's SO(3) derivative: q_next * conjugate(q_prev),
    # centered over two samples, with endpoint values repeated.
    rel = (rotations[2:] * rotations[:-2].inv()).as_rotvec() * (float(fps) / 2.0)
    result = np.empty((len(q), 3), dtype=np.float64)
    result[0] = rel[0]
    result[-1] = rel[-1]
    if len(q) > 2:
        result[1:-1] = rel
    return result.astype(np.float32)


def save_npz(path: str | Path, *, fps: float, joint_pos: np.ndarray,
             joint_vel: np.ndarray, body_pos_w: np.ndarray, body_quat_w: np.ndarray,
             body_lin_vel_w: np.ndarray, body_ang_vel_w: np.ndarray,
             joint_names: list[str] | None = None,
             body_names: list[str] | None = None) -> None:
    """Write the BeyondMimic motion fields and optional name metadata."""
    payload = dict(
        fps=np.float32(fps), joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32), body_pos_w=np.asarray(body_pos_w, dtype=np.float32),
        body_quat_w=np.asarray(body_quat_w, dtype=np.float32), body_lin_vel_w=np.asarray(body_lin_vel_w, dtype=np.float32),
        body_ang_vel_w=np.asarray(body_ang_vel_w, dtype=np.float32),
    )
    if joint_names is not None:
        payload["joint_names"] = np.asarray(joint_names)
    if body_names is not None:
        payload["body_names"] = np.asarray(body_names)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
