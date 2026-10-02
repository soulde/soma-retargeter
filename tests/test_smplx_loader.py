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
        # Standing pelvis height (SMPL-X Y-up) so the figure is on the ground.
        "trans": np.tile([0.0, 0.95, 0.0], (num_frames, 1)),
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


def test_load_npz_retains_native_smplx_skeleton(tmp_path):
    path, _ = _make_smplx_npz(tmp_path, knee_bend=0.0)
    skeleton, animation = load_smplx_npz(str(path))
    assert animation.num_frames == 5
    assert animation.sample_rate == 100.0

    assert skeleton.num_joints == NUM_BODY_JOINTS
    assert "pelvis" in skeleton.joint_names
    assert "left_shoulder" in skeleton.joint_names
    assert "Hips" not in skeleton.joint_names
    global_tx = np.asarray(animation.compute_global_transforms(0))
    pelvis_z = global_tx[skeleton.joint_index("pelvis"), 2]
    head_z = global_tx[skeleton.joint_index("head"), 2]
    assert 0.5 < pelvis_z < 1.4
    assert head_z > pelvis_z + 0.4


def test_load_npz_rotates_root_translation_to_z_up(tmp_path):
    path, data = _make_smplx_npz(tmp_path)
    data["trans"][:, 1] = 1.5  # up in SMPL-X Y-up coordinates
    np.savez(path, **data)

    skeleton, animation = load_smplx_npz(str(path))
    # The loader rotates the source's +Y up direction to pipeline +Z while
    # retaining the source skeleton and its pelvis root.
    from soma_retargeter.assets.smplx import load_smplx_rest_skeleton_json
    j0_y = load_smplx_rest_skeleton_json()["offsets"][0][1]
    pelvis_z = np.asarray(animation.compute_global_transforms(0))[
        skeleton.joint_index("pelvis"), 2]
    assert pelvis_z == pytest.approx(1.5 + j0_y, abs=0.02)


def test_load_npz_preserves_smplx_local_joint_rotations(tmp_path):
    path, _ = _make_smplx_npz(tmp_path, knee_bend=0.5)
    skeleton, animation = load_smplx_npz(str(path))

    knee = skeleton.joint_index("left_knee")
    actual = R.from_quat(animation.local_transforms[0, knee][3:7])
    expected = R.from_rotvec([0.5, 0.0, 0.0])
    assert np.degrees((expected.inv() * actual).magnitude()) < 1e-3


@pytest.mark.skipif(not __import__("os").path.exists(KIT_SAMPLE),
                    reason="KIT sample not available on this machine")
def test_load_kit_sample():
    skeleton, animation = load_smplx_npz(KIT_SAMPLE)
    assert "pelvis" in skeleton.joint_names
    assert animation.num_frames > 0
    assert animation.sample_rate == pytest.approx(120.0)  # KIT mocap frame rate
    # The native SMPL-X skeleton is expressed in the pipeline's Z-up frame.
    pelvis_idx = skeleton.joint_index("pelvis")
    pelvis_z = max(
        np.asarray(animation.compute_global_transforms(f))[pelvis_idx, 2]
        for f in range(0, animation.num_frames, 50))
    assert 0.5 < pelvis_z < 1.5
