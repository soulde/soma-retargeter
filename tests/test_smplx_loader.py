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


def test_load_npz_converts_onto_soma_skeleton(tmp_path):
    path, _ = _make_smplx_npz(tmp_path, knee_bend=0.0)
    skeleton, animation = load_smplx_npz(str(path))
    assert animation.num_frames == 5
    assert animation.sample_rate == 100.0

    # The motion is transferred onto the canonical SOMA skeleton (Y-up BVH
    # frame): Hips/LeftArm naming, upright stance.
    assert "Hips" in skeleton.joint_names
    assert "LeftArm" in skeleton.joint_names
    global_tx = np.asarray(animation.compute_global_transforms(0))
    hips_y = global_tx[skeleton.joint_index("Hips"), 1]
    head_y = global_tx[skeleton.joint_index("Head"), 1]
    assert 0.5 < hips_y < 1.4
    assert head_y > hips_y + 0.4


def test_load_npz_rotates_root_translation_to_z_up(tmp_path):
    path, data = _make_smplx_npz(tmp_path)
    data["trans"][:, 1] = 1.5  # up in SMPL-X Y-up coordinates
    np.savez(path, **data)

    skeleton, animation = load_smplx_npz(str(path))
    # Up (+Y in SMPL-X) stays up (+Y) in the SOMA BVH frame after the fixed
    # conversion; the Hips rides on the pelvis target (transl + canonical J0).
    from soma_retargeter.assets.smplx import load_smplx_rest_skeleton_json
    j0_y = load_smplx_rest_skeleton_json()["offsets"][0][1]
    hips_y = np.asarray(animation.compute_global_transforms(0))[
        skeleton.joint_index("Hips"), 1]
    assert hips_y == pytest.approx(1.5 + j0_y, abs=0.02)


def test_load_npz_drives_soma_joints_with_smplx_rotations(tmp_path):
    path, _ = _make_smplx_npz(tmp_path, knee_bend=0.5)
    skeleton, animation = load_smplx_npz(str(path))

    # Contract: each driven SOMA joint's global rotation equals the SMPL-X
    # joint's global rotation moved into the BVH frame and corrected by the
    # fixed per-joint frame fix (calibrated from the matched rest pair).
    smplx_skel = create_smplx_skeleton()
    ref = np.asarray(smplx_skel.reference_local_transforms).copy()
    knee = smplx_skel.joint_index("left_knee")
    ref[knee, 3:7] = R.from_rotvec([0.5, 0.0, 0.0]).as_quat()
    locals_ = [wp.transform(wp.vec3(*t[:3]), wp.quat(*t[3:7])) for t in ref]
    g = np.asarray(smplx_skel.compute_global_transforms(locals_))
    rx_inv = R.from_euler("x", -90, degrees=True)

    from soma_retargeter.assets.smplx import _soma_reference_skeleton, _SOMA_ZERO_LOCALS_CACHE, SMPLX_TO_SOMA_JOINT
    soma_skel = _soma_reference_skeleton()
    soma_zero_g = np.asarray(soma_skel.compute_global_transforms(
        [wp.transform(wp.vec3(*z[:3]), wp.quat(*z[3:7])) for z in _SOMA_ZERO_LOCALS_CACHE]))
    smplx_rest_g = np.asarray(smplx_skel.compute_global_transforms(
        smplx_skel.reference_local_transforms))

    soma_g = np.asarray(animation.compute_global_transforms(0))
    for xname, sname in SMPLX_TO_SOMA_JOINT.items():
        i = smplx_skel.joint_index(xname)
        j = skeleton.joint_index(sname)
        fix = R.from_quat(soma_zero_g[j, 3:7]) * (rx_inv * R.from_quat(smplx_rest_g[i, 3:7])).inv()
        expected = fix * (rx_inv * R.from_quat(g[i, 3:7]))
        actual = R.from_quat(soma_g[j, 3:7])
        assert np.degrees((expected.inv() * actual).magnitude()) < 1e-3


@pytest.mark.skipif(not __import__("os").path.exists(KIT_SAMPLE),
                    reason="KIT sample not available on this machine")
def test_load_kit_sample():
    skeleton, animation = load_smplx_npz(KIT_SAMPLE)
    assert "Hips" in skeleton.joint_names
    assert animation.num_frames > 0
    assert animation.sample_rate == pytest.approx(120.0)  # KIT mocap frame rate
    # Pelvis height (Y in the SOMA BVH frame) should reach a plausible
    # humanoid range over the clip.
    hips_idx = skeleton.joint_index("Hips")
    hips_y = max(
        np.asarray(animation.compute_global_transforms(f))[hips_idx, 1]
        for f in range(0, animation.num_frames, 50))
    assert 0.5 < hips_y < 1.5
