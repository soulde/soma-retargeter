# SPDX-FileCopyrightText: Copyright (c) 2026 soulde. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation as R

from soma_retargeter.assets.smplx import (
    NUM_BODY_JOINTS, create_smplx_skeleton, load_smplx_npz)


KIT_SAMPLE = "/home/jvwei/GMR-private/data/KIT/forward/walking_medium07_stageii.npz"


def _make_smplx_npz(tmp_path, num_frames=5, knee_bend=0.0):
    data = {
        "root_orient": np.zeros((num_frames, 3), dtype=np.float64),
        "pose_body": np.zeros((num_frames, 3 * (NUM_BODY_JOINTS - 1)), dtype=np.float64),
        "trans": np.zeros((num_frames, 3), dtype=np.float64),
        "mocap_frame_rate": np.float64(100.0),
    }
    if knee_bend:
        # Bend the left knee (rotation about the parent-frame x axis).
        data["pose_body"][:, 3 * 3 + 0] = knee_bend
    path = tmp_path / "motion_stageii.npz"
    np.savez(path, **data)
    return path, data


def test_skeleton_rest_pose_matches_exported_offsets():
    skeleton = create_smplx_skeleton()
    assert skeleton.num_joints == NUM_BODY_JOINTS
    assert skeleton.joint_names[0] == "pelvis"
    assert skeleton.joint_index("left_foot") > 0

    global_tx = np.asarray(skeleton.compute_global_transforms(
        skeleton.reference_local_transforms, wp.transform_identity()))
    # Y-up rest pose rotated into Z-up: the head must be above the pelvis.
    pelvis_z = global_tx[skeleton.joint_index("pelvis"), 2]
    head_z = global_tx[skeleton.joint_index("head"), 2]
    assert head_z > pelvis_z + 0.4


def test_load_npz_zero_pose_reproduces_rest_skeleton(tmp_path):
    path, _ = _make_smplx_npz(tmp_path, knee_bend=0.0)
    skeleton, animation = load_smplx_npz(str(path))
    assert animation.num_frames == 5
    assert animation.sample_rate == 100.0

    global_tx = np.asarray(animation.compute_global_transforms(0))
    rest_tx = np.asarray(skeleton.compute_global_transforms(
        skeleton.reference_local_transforms, wp.transform_identity()))
    np.testing.assert_allclose(global_tx, rest_tx, atol=1e-5)


def test_load_npz_rotates_root_translation_to_z_up(tmp_path):
    path, data = _make_smplx_npz(tmp_path)
    data["trans"][:, 1] = 1.5  # up in SMPL-X Y-up coordinates
    np.savez(path, **data)

    from soma_retargeter.assets.smplx import load_smplx_rest_skeleton_json
    j0 = load_smplx_rest_skeleton_json()["offsets"][0]  # canonical pelvis offset

    _, animation = load_smplx_npz(str(path))
    root_tx = animation.get_local_transforms(0)[0]  # [px, py, pz, qx, qy, qz, qw]
    # Y-up (+Z-forward) rotates to Z-up (-Y-forward); the unrotated pelvis
    # offset J0 rides along with the translation.
    assert root_tx[2] == pytest.approx(1.5 + j0[1], abs=1e-4)  # now along +Z
    assert root_tx[1] == pytest.approx(-j0[2], abs=1e-4)


def test_load_npz_joint_rotation_order(tmp_path):
    path, _ = _make_smplx_npz(tmp_path, knee_bend=0.5)
    _, animation = load_smplx_npz(str(path))

    # pose_body joint 3 (left_knee, index 3 in the 21 body joints) carries a
    # 0.5 rad rotation about x; verify the local transform matches.
    local = animation.get_local_transforms(0)  # (num_joints, 7) [p, q xyzw]
    knee_idx = create_smplx_skeleton().joint_index("left_knee")
    expected = R.from_rotvec([0.5, 0.0, 0.0]).as_quat()  # xyzw
    actual = local[knee_idx, 3:7]
    # q and -q encode the same rotation.
    if np.dot(expected, actual) < 0:
        actual = -actual
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.skipif(not __import__("os").path.exists(KIT_SAMPLE),
                    reason="KIT sample not available on this machine")
def test_load_kit_sample():
    skeleton, animation = load_smplx_npz(KIT_SAMPLE)
    assert skeleton.num_joints == NUM_BODY_JOINTS
    assert animation.num_frames > 0
    assert animation.sample_rate == pytest.approx(120.0)  # KIT mocap frame rate
    # Pelvis height in Z-up should reach a plausible humanoid range over the
    # clip (the first frame may be a crouched/ground-contact pose).
    pelvis_idx = skeleton.joint_index("pelvis")
    pelvis_z = max(
        np.asarray(animation.compute_global_transforms(f))[pelvis_idx, 2]
        for f in range(0, animation.num_frames, 50))
    assert 0.5 < pelvis_z < 1.5
