#!/usr/bin/env python3
"""Report Chocolate locomotion NPZ kinematics without deleting any files."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def assess_motion(name, linear_body, angular_body):
    mean_v = np.mean(linear_body, axis=0)
    planar_speed = np.linalg.norm(linear_body[:, :2], axis=1)
    metrics = {
        "mean_vx": float(mean_v[0]),
        "mean_vy": float(mean_v[1]),
        "mean_speed": float(np.mean(planar_speed)),
        "p90_speed": float(np.percentile(planar_speed, 90)),
        "mean_abs_yaw_rate": float(np.mean(np.abs(angular_body[:, 2]))),
    }
    lower = name.lower()
    flags = []
    if metrics["mean_speed"] < 0.1 and "turn_" not in lower:
        flags.append("low_planar_speed")
    if "sideway_left" in lower and metrics["mean_vy"] <= 0.1:
        flags.append("direction_mismatch_left")
    if "sideway_right" in lower and metrics["mean_vy"] >= -0.1:
        flags.append("direction_mismatch_right")
    if "backward" in lower and metrics["mean_vx"] >= -0.1:
        flags.append("direction_mismatch_backward")
    if ("forward" in lower or "_ff_" in lower) and metrics["mean_vx"] <= 0.1:
        flags.append("direction_mismatch_forward")
    if lower.startswith("turn_") and metrics["mean_abs_yaw_rate"] < 0.1:
        flags.append("low_turn_rate")
    return metrics, flags


def _rotate_inverse_wxyz(quaternion, vector):
    q_w = quaternion[:, :1]
    q_vec = quaternion[:, 1:]
    cross = np.cross(q_vec, vector)
    return vector - 2.0 * q_w * cross + 2.0 * np.cross(q_vec, cross)


def inspect_npz(path):
    with np.load(path, allow_pickle=False) as data:
        required = (
            "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w", "body_names",
        )
        missing = [key for key in required if key not in data]
        if missing:
            return {}, ["missing:" + ",".join(missing)]
        arrays = [data[key] for key in required if key not in ("fps", "body_names")]
        if any(not np.isfinite(array).all() for array in arrays):
            return {}, ["nonfinite"]
        names = [str(name) for name in data["body_names"]]
        root = names.index("base") if "base" in names else 0
        quat = np.asarray(data["body_quat_w"][:, root], dtype=float)
        linear = _rotate_inverse_wxyz(quat, data["body_lin_vel_w"][:, root])
        angular = _rotate_inverse_wxyz(quat, data["body_ang_vel_w"][:, root])
    return assess_motion(path.name, linear, angular)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_root", type=Path)
    parser.add_argument("report_csv", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.npz_root.rglob("*.npz"))
    rows = []
    flag_counts = Counter()
    for path in paths:
        metrics, flags = inspect_npz(path)
        flag_counts.update(flags)
        rows.append({
            "file": str(path.relative_to(args.npz_root)),
            **{key: f"{value:.6f}" for key, value in metrics.items()},
            "flags": ";".join(flags),
        })

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "mean_vx", "mean_vy", "mean_speed", "p90_speed",
              "mean_abs_yaw_rate", "flags"]
    with args.report_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "total": len(rows),
        "flagged": sum(bool(row["flags"]) for row in rows),
        "flag_counts": dict(sorted(flag_counts.items())),
        "policy": "report only; no NPZ was deleted, moved, or excluded",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
