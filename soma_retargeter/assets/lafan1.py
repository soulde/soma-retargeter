"""Validated loader for BVH files using the standard LAFAN1 skeleton."""

from pathlib import Path

import numpy as np
import warp as wp

from soma_retargeter.animation.animation_buffer import create_animation_buffer_for_skeleton
from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.assets.bvh import load_bvh


STANDARD_LAFAN1_JOINT_NAMES = (
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
)

STANDARD_LAFAN1_PARENT_INDICES = (
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16,
    11, 18, 19, 20,
)


def validate_lafan1_skeleton(skeleton: Skeleton) -> None:
    """Reject BVHs that do not use the canonical 22-joint LAFAN1 hierarchy."""

    if len(set(skeleton.joint_names)) != len(skeleton.joint_names):
        raise ValueError("LAFAN1 skeleton contains duplicate joint names.")
    if tuple(skeleton.joint_names) != STANDARD_LAFAN1_JOINT_NAMES:
        raise ValueError(
            "LAFAN1 skeleton joint names/order do not match the standard hierarchy.")
    if tuple(np.asarray(skeleton.parent_indices).tolist()) != STANDARD_LAFAN1_PARENT_INDICES:
        raise ValueError("LAFAN1 skeleton topology does not match the standard hierarchy.")
    if not np.isfinite(skeleton.reference_local_transforms).all():
        raise ValueError("LAFAN1 skeleton contains non-finite local transforms.")


def load_lafan1_bvh(
    path: str | Path,
    input_skeleton: Skeleton | None = None,
):
    """Load one standard LAFAN1 BVH without resampling or phase splitting."""

    source_skeleton, animation = load_bvh(str(path))
    validate_lafan1_skeleton(source_skeleton)
    if not np.isfinite(animation.local_transforms).all():
        raise ValueError("LAFAN1 animation contains non-finite local transforms.")
    if not np.isfinite(animation.sample_rate) or animation.sample_rate <= 0:
        raise ValueError("LAFAN1 animation sample rate must be finite and positive.")

    source_skeleton.up_axis = wp.vec3(0.0, 0.0, 1.0)
    source_skeleton.forward_axis = wp.vec3(0.0, -1.0, 0.0)

    if input_skeleton is None:
        return source_skeleton, animation
    return input_skeleton, create_animation_buffer_for_skeleton(animation, input_skeleton)
