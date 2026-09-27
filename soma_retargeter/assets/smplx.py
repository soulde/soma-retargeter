# SPDX-FileCopyrightText: Copyright (c) 2026 soulde. All rights reserved.
# SPDX-LicenseFileCopyrightText: SPDX-License-Identifier: Apache-2.0

"""SMPL-X (AMASS-style) motion loader.

Loads AMASS/SMPL-X ``.npz`` files (e.g. the KIT dataset) into the same
``(Skeleton, AnimationBuffer)`` pair produced by ``load_bvh``, so they can be
fed into the Newton retargeting pipeline.

The rest skeleton (joint names, parents, zero-pose offsets) is read from
``configs/smplx/smplx_rest_skeleton.json``, which was exported once from the
SMPL-X neutral body model with ``scripts/export_smplx_skeleton.py``. No smplx
or torch dependency is needed at runtime.

SMPL-X data is Y-up with forward +Z; soma-retargeter skeletons are Z-up with
forward -Y. A -90 degree rotation about X maps one to the other, so it is
baked into the root joint of both the rest pose and every animation frame.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation as R

import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.animation.animation_buffer import AnimationBuffer, create_animation_buffer_for_skeleton
from soma_retargeter.animation.skeleton import Skeleton

NUM_BODY_JOINTS = 22  # pelvis + 21 joints driven by pose_body

# Rotates standard SMPL-X Y-up/+Z-forward data into the pipeline's Z-up frame
# (+90 degrees about X maps +Y to +Z and +Z to -Y). Z-up exports (e.g. the
# KIT stageii files) need no rotation.
Y_UP_TO_Z_UP = R.from_euler("x", 90, degrees=True)
AXIS_ROTATIONS = {
    "+Y": Y_UP_TO_Z_UP,
    "-Y": Y_UP_TO_Z_UP.inv(),
    "+Z": R.identity(),
    "-Z": R.from_euler("x", 180, degrees=True),
}


def load_smplx_rest_skeleton_json() -> dict:
    """Load the exported SMPL-X rest skeleton description."""
    path = io_utils.get_config_file("smplx", "smplx_rest_skeleton.json")
    with open(path) as f:
        return json.load(f)


def create_smplx_skeleton(rest_data: dict | None = None, up_axis: str = "+Y") -> Skeleton:
    """Build a ``Skeleton`` from the exported SMPL-X rest skeleton JSON.

    Args:
        rest_data: Pre-loaded rest skeleton description.
        up_axis: Up axis of the source data convention ("+Y" for standard
            SMPL-X/AMASS, "+Z" for Z-up exports such as the KIT stageii files).
    """
    if rest_data is None:
        rest_data = load_smplx_rest_skeleton_json()

    joint_names = list(rest_data["joint_names"])
    parent_indices = np.asarray(rest_data["parents"], dtype=np.int32)
    offsets = np.asarray(rest_data["offsets"], dtype=np.float32)

    local_transforms = [wp.transform() for _ in range(len(joint_names))]
    root_quat = AXIS_ROTATIONS[up_axis].as_quat()  # xyzw
    root_offset = AXIS_ROTATIONS[up_axis].apply(offsets[0])
    local_transforms[0] = wp.transform(wp.vec3(*root_offset), wp.quat(*root_quat))
    for i in range(1, len(joint_names)):
        local_transforms[i] = wp.transform(wp.vec3(*offsets[i]), wp.quat_identity())

    skeleton = Skeleton(len(joint_names), joint_names, parent_indices, local_transforms)
    # The stored offsets are already in the converted Z-up frame only at the
    # root level; the skeleton itself keeps the default pipeline axes since the
    # up-axis rotation is baked into the root transform.
    skeleton.up_axis = wp.vec3(0, 0, 1)
    skeleton.forward_axis = wp.vec3(0, -1, 0)
    return skeleton


def detect_up_axis(root_orient: np.ndarray, num_samples: int = 50) -> str:
    """Detect the world up axis of a SMPL-X export.

    Rotates the canonical head-pelvis direction by each frame's root
    orientation and picks the world axis the body is most aligned with.
    Standard SMPL-X/AMASS data is "+Y"; the KIT stageii exports are "+Z".

    Args:
        root_orient: (N, 3) root orientation rotvecs.
        num_samples: Number of frames to average over.

    Returns:
        str: One of "+Y", "-Y", "+Z", "-Z".
    """
    rest_json = load_smplx_rest_skeleton_json()
    names = rest_json["joint_names"]
    pelvis_pos = np.asarray(rest_json["offsets"][names.index("pelvis")])
    # Head position in the canonical rest pose (sum of offsets along the chain).
    head_pos = np.zeros(3)
    chain = {name: i for i, name in enumerate(names)}
    joint = "head"
    while joint != "pelvis":
        idx = chain[joint]
        head_pos = head_pos + np.asarray(rest_json["offsets"][idx])
        joint = names[rest_json["parents"][idx]]
    up_dir = head_pos - pelvis_pos

    indices = np.linspace(0, len(root_orient) - 1, num=min(num_samples, len(root_orient)))
    world_dirs = R.from_rotvec(root_orient[indices.astype(int)]).apply(up_dir)
    mean_dir = world_dirs.mean(axis=0)
    axis = int(np.argmax(np.abs(mean_dir)))
    sign = "+" if mean_dir[axis] > 0 else "-"
    return f"{sign}{'XYZ'[axis]}"


def detect_up_axis_for_files(npz_files) -> str:
    """Majority-vote up axis across a batch of SMPL-X files (after normalization)."""
    import collections

    votes = collections.Counter()
    for path in npz_files:
        data = np.load(str(path), allow_pickle=True)
        if "poses" in data.files:
            root_orient = np.asarray(data["poses"], dtype=np.float64)[:, :3]
        else:
            root_orient = np.asarray(data["root_orient"], dtype=np.float64)
        rots = R.from_rotvec(root_orient)
        votes[detect_up_axis((rots.mean().inv() * rots).as_rotvec())] += 1
    return votes.most_common(1)[0][0]


def load_smplx_npz(npz_file: str, input_skeleton: Skeleton | None = None):
    """
    Load an AMASS/SMPL-X animation file and create ``Skeleton`` and
    ``AnimationBuffer`` objects.

    Args:
        npz_file: Path to the SMPL-X ``.npz`` file. Supports both the split
            stageii convention (``root_orient``, ``pose_body``) and the raw
            AMASS convention (single ``poses`` array).
        input_skeleton: Optional SOMA skeleton to conform the animation to.
            The returned motion is always on the canonical SOMA skeleton
            (see ``convert_animation_to_soma_skeleton``).

    Returns:
        tuple (Skeleton, AnimationBuffer) on the SOMA skeleton, in the SOMA
        BVH frame convention (the caller applies the soma facing converter).
    """
    data = np.load(npz_file, allow_pickle=True)

    if "poses" in data.files:
        poses = np.asarray(data["poses"], dtype=np.float64)
        root_orient = poses[:, :3]
        pose_body = poses[:, 3:3 * NUM_BODY_JOINTS]
    else:
        root_orient = np.asarray(data["root_orient"], dtype=np.float64)
        pose_body = np.asarray(data["pose_body"], dtype=np.float64)

    trans = np.asarray(data["trans"], dtype=np.float64)
    num_frames = pose_body.shape[0]

    if "mocap_frame_rate" in data.files:
        fps = float(data["mocap_frame_rate"])
    elif "mocap_framerate" in data.files:
        fps = float(data["mocap_framerate"])
    else:
        fps = 30.0

    # Exports wrap the canonical SMPL-X parameters in a world frame by
    # pre-multiplying the root orientation with a constant rotation (e.g. the
    # KIT stageii files carry ~Rx(-90) so a standing person's root orientation
    # is far from identity). Remove the clip-wide constant (chordal mean of
    # the root orientations, i.e. the dominant standing pose) so an upright
    # person maps back to the canonical pose; the residual is real motion.
    root_rots = R.from_rotvec(root_orient)
    normalize = root_rots.mean().inv()
    root_orient = (normalize * root_rots).as_rotvec()

    # Detect the world up axis from the normalized head-pelvis direction
    # ("+Y" for canonical SMPL-X/AMASS).
    up_axis = detect_up_axis(root_orient)
    up_rotation = AXIS_ROTATIONS[up_axis]

    # The motion is first built on a fresh SMPL-X skeleton, then transferred
    # onto the canonical SOMA skeleton (input_skeleton conforming happens at
    # the end of the conversion, SOMA rig in / SOMA rig out).
    skeleton = create_smplx_skeleton(up_axis=up_axis)

    # reference_local_transforms is an (num_joints, 7) float32 array of
    # [px, py, pz, qx, qy, qz, qw]; non-root rest rotations are identity.
    offsets = skeleton.reference_local_transforms

    root_rot = up_rotation * R.from_rotvec(root_orient)  # per-frame world root rotation
    # Positions must go through the same total world rotation as orientations
    # (normalize, then up-axis conversion), otherwise the ground plane tilts.
    total_rotation = up_rotation * normalize
    # SMPL-X LBS rotates about the pelvis joint itself, so world joints are
    # transl + J0 + root_R @ (J_i - J0): the canonical pelvis offset J0 rides
    # along unrotated on top of the transl channel.
    canonical_j0 = np.asarray(load_smplx_rest_skeleton_json()["offsets"][0], dtype=np.float64)
    root_pos = total_rotation.apply(trans + canonical_j0)
    joint_rot = R.from_rotvec(pose_body.reshape(num_frames, NUM_BODY_JOINTS - 1, 3))

    root_quat = root_rot.as_quat()  # (N, 4) xyzw
    joint_quat = joint_rot.as_quat()  # (N, 21, 4) xyzw

    local_transforms = np.zeros((num_frames, NUM_BODY_JOINTS), dtype=wp.transform)
    for f in range(num_frames):
        local_transforms[f, 0] = wp.transform(
            wp.vec3(*root_pos[f]), wp.quat(*root_quat[f]))
        for j in range(1, NUM_BODY_JOINTS):
            local_transforms[f, j] = wp.transform(
                wp.vec3(*offsets[j, :3]), wp.quat(*joint_quat[f, j - 1]))

    animation_buffer = AnimationBuffer(skeleton, num_frames, fps, local_transforms)
    return convert_animation_to_soma_skeleton(skeleton, animation_buffer, input_skeleton)


# ================================ #
#   SOMA-skeleton conversion       #
# ================================ #

# SMPL-X body joint -> SOMA BVH joint whose global rotation it drives.
SMPLX_TO_SOMA_JOINT = {
    "pelvis": "Hips",
    "spine3": "Chest",
    "neck": "Neck1",
    "left_shoulder": "LeftArm", "left_elbow": "LeftForeArm", "left_wrist": "LeftHand",
    "right_shoulder": "RightArm", "right_elbow": "RightForeArm", "right_wrist": "RightHand",
    "left_hip": "LeftLeg", "left_knee": "LeftShin", "left_ankle": "LeftFoot", "left_foot": "LeftToeBase",
    "right_hip": "RightLeg", "right_knee": "RightShin", "right_ankle": "RightFoot", "right_foot": "RightToeBase",
}

_SOMA_SKELETON_CACHE = None
_SOMA_ZERO_LOCALS_CACHE = None


def _soma_reference_skeleton() -> Skeleton:
    """The canonical SOMA skeleton (loaded once from the zero-frame BVH)."""
    global _SOMA_SKELETON_CACHE, _SOMA_ZERO_LOCALS_CACHE
    if _SOMA_SKELETON_CACHE is None:
        from soma_retargeter.assets.bvh import load_bvh

        _SOMA_SKELETON_CACHE, zero_anim = load_bvh(
            str(io_utils.get_config_file("soma", "soma_zero_frame0.bvh")))
        # The SOMA bind pose (raw offsets) is not a standing pose; frame 0 of
        # the zero-pose clip carries the channel-level rest pose production
        # BVHs are expressed against. Undriven joints use these locals.
        _SOMA_ZERO_LOCALS_CACHE = np.asarray(zero_anim.get_local_transforms(0)).copy()
    return _SOMA_SKELETON_CACHE


def convert_animation_to_soma_skeleton(
    smplx_skeleton: Skeleton, animation: AnimationBuffer,
    input_skeleton: Skeleton | None = None,
):
    """Transfer an SMPL-X animation onto the canonical SOMA skeleton.

    The retargeting configs are calibrated against the SOMA rig, so — like the
    LAFAN1 loader — the source is converted with a fixed in-code transform and
    the soma_to_* configs are reused as-is (no per-source offset calibration).

    Each SOMA joint listed in ``SMPLX_TO_SOMA_JOINT`` is driven so its global
    rotation equals the SMPL-X joint's global rotation (moved into the SOMA
    joint frame by the fixed per-joint correction). The Root local is solved
    so Hips lands on the pelvis target, and undriven joints (spine segments,
    neck2/head, fingers, eyes) keep the zero-frame clip's local transforms —
    the channel-level rest pose production BVHs build on. The result is
    expressed in the SOMA BVH frame (Y-up), exactly like BVH data, so the app
    applies the same "Mujoco" facing converter to it as to real SOMA input.

    Args:
        smplx_skeleton: The SMPL-X skeleton the animation is currently on.
        animation: The animation in the pipeline (Z-up) frame.
        input_skeleton: Optional existing SOMA skeleton to conform the
            animation to (all SOMA clips share the same rig).

    Returns:
        tuple (Skeleton, AnimationBuffer) on the SOMA skeleton.
    """
    soma_skel = _soma_reference_skeleton()
    zero_locals = np.asarray(_SOMA_ZERO_LOCALS_CACHE)
    num_frames = animation.num_frames
    soma_idx = {n: i for i, n in enumerate(soma_skel.joint_names)}
    smplx_idx = {n: i for i, n in enumerate(smplx_skeleton.joint_names)}

    # pipeline frame (Z-up) -> SOMA BVH frame (Y-up); the app's converter
    # applies the inverse (+90 deg about X) afterwards, like for soma BVHs.
    rx_inv = R.from_euler("x", -90, degrees=True)

    # Fixed per-joint frame correction, calibrated once from the matched rest
    # pair (SMPL-X bind pose <-> SOMA zero-frame standing pose). The SOMA rig
    # binds horizontally, so its standing joint frames differ from the SMPL-X
    # canonical ones by a constant rotation per joint; without it the driven
    # globals would fold the skeleton.
    soma_zero_g = np.asarray(soma_skel.compute_global_transforms(
        [wp.transform(wp.vec3(*z[:3]), wp.quat(*z[3:7])) for z in zero_locals]))
    smplx_rest_g = np.asarray(smplx_skeleton.compute_global_transforms(
        smplx_skeleton.reference_local_transforms))
    frame_fix = {}
    for xname, sname in SMPLX_TO_SOMA_JOINT.items():
        i = smplx_idx[xname]
        r_soma = R.from_quat(soma_zero_g[soma_idx[sname], 3:7])
        r_smplx_bvh = rx_inv * R.from_quat(smplx_rest_g[i, 3:7])
        frame_fix[soma_idx[sname]] = r_soma * r_smplx_bvh.inv()

    parents = soma_skel.parent_indices
    hips_i = soma_idx["Hips"]
    hips_local = np.asarray(zero_locals[hips_i], dtype=np.float64)
    hips_q0 = R.from_quat(hips_local[3:7])
    hips_t0 = hips_local[:3]

    local_transforms = np.zeros((num_frames, soma_skel.num_joints), dtype=wp.transform)
    for f in range(num_frames):
        g = np.asarray(smplx_skeleton.compute_global_transforms(
            animation.get_local_transforms(f)))
        tgt = {}
        for xname, sname in SMPLX_TO_SOMA_JOINT.items():
            i = smplx_idx[xname]
            j = soma_idx[sname]
            tgt[j] = (
                rx_inv.apply(g[i, :3].astype(np.float64)),
                frame_fix[j] * (rx_inv * R.from_quat(g[i, 3:7])),
            )

        grot = [None] * soma_skel.num_joints
        row = local_transforms[f]
        for j in range(soma_skel.num_joints):
            p = parents[j]
            lz = np.asarray(zero_locals[j], dtype=np.float64)
            if p == -1:
                # Solve the Root local so Hips (keeping its zero-frame local)
                # lands on the pelvis target.
                hips_pos, hips_rot = tgt[hips_i]
                root_rot = hips_rot * hips_q0.inv()
                t = hips_pos - root_rot.apply(hips_t0)
                row[j] = wp.transform(wp.vec3(*t), wp.quat(*root_rot.as_quat()))
                grot[j] = root_rot
            elif j == hips_i:
                row[j] = wp.transform(wp.vec3(*lz[:3]), wp.quat(*lz[3:7]))
                grot[j] = grot[p] * hips_q0
            elif j in tgt:
                _, rot = tgt[j]
                local_rot = grot[p].inv() * rot
                row[j] = wp.transform(wp.vec3(*lz[:3]), wp.quat(*local_rot.as_quat()))
                grot[j] = rot
            else:
                row[j] = wp.transform(wp.vec3(*lz[:3]), wp.quat(*lz[3:7]))
                grot[j] = grot[p] * R.from_quat(lz[3:7])

    converted = AnimationBuffer(soma_skel, num_frames, animation.sample_rate, local_transforms)
    if input_skeleton is None:
        return soma_skel, converted
    return input_skeleton, create_animation_buffer_for_skeleton(converted, input_skeleton)
