import os

import numpy as np
import pytest
import warp as wp

from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import Skeleton


LAFAN_SAMPLE = "/home/jvwei/datasets/lafan1/walk1_subject1.bvh"


def _standard_skeleton():
    from soma_retargeter.assets.lafan1 import (
        STANDARD_LAFAN1_JOINT_NAMES,
        STANDARD_LAFAN1_PARENT_INDICES,
    )

    transforms = np.asarray([wp.transform_identity()] * 22, dtype=np.float32)
    return Skeleton(
        22,
        STANDARD_LAFAN1_JOINT_NAMES,
        STANDARD_LAFAN1_PARENT_INDICES,
        transforms,
    )


@pytest.mark.skipif(not os.path.exists(LAFAN_SAMPLE), reason="LAFAN1 sample unavailable")
def test_load_standard_lafan1_sample():
    from soma_retargeter.assets.lafan1 import (
        STANDARD_LAFAN1_JOINT_NAMES,
        load_lafan1_bvh,
    )

    skeleton, animation = load_lafan1_bvh(LAFAN_SAMPLE)

    assert animation.sample_rate == pytest.approx(30.0, abs=1e-3)
    assert skeleton.joint_names == list(STANDARD_LAFAN1_JOINT_NAMES)
    assert skeleton.num_joints == 22
    assert list(skeleton.up_axis) == [0.0, 0.0, 1.0]
    assert list(skeleton.forward_axis) == [0.0, -1.0, 0.0]
    assert np.isfinite(skeleton.reference_local_transforms).all()
    assert np.isfinite(animation.local_transforms).all()
    global_pose = np.asarray(animation.compute_global_transforms(0))
    hips_z = global_pose[skeleton.joint_index("Hips"), 2]
    head_z = global_pose[skeleton.joint_index("Head"), 2]
    assert 0.5 < hips_z < 1.5
    assert head_z > hips_z + 0.4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda s: s.joint_names.__setitem__(4, "MissingToe"), "joint names"),
        (lambda s: s.joint_names.__setitem__(4, "LeftFoot"), "duplicate"),
        (lambda s: s.parent_indices.__setitem__(4, 2), "topology"),
    ],
)
def test_validate_rejects_nonstandard_skeleton(mutation, message):
    from soma_retargeter.assets.lafan1 import validate_lafan1_skeleton

    skeleton = _standard_skeleton()
    mutation(skeleton)
    with pytest.raises(ValueError, match=message):
        validate_lafan1_skeleton(skeleton)


def test_loader_preserves_positive_bvh_sample_rate(monkeypatch, tmp_path):
    import soma_retargeter.assets.lafan1 as lafan1

    skeleton = _standard_skeleton()
    animation = AnimationBuffer(skeleton, num_frames=2, sample_rate=60.0)
    monkeypatch.setattr(lafan1, "load_bvh", lambda path, input_skeleton=None: (skeleton, animation))

    loaded_skeleton, loaded_animation = lafan1.load_lafan1_bvh(tmp_path / "motion.bvh")

    assert loaded_skeleton is skeleton
    assert loaded_animation is animation
    assert loaded_animation.sample_rate == 60.0


def test_loader_rejects_nonpositive_sample_rate(monkeypatch, tmp_path):
    import soma_retargeter.assets.lafan1 as lafan1

    skeleton = _standard_skeleton()
    animation = AnimationBuffer(skeleton, num_frames=1, sample_rate=0.0)
    monkeypatch.setattr(lafan1, "load_bvh", lambda path, input_skeleton=None: (skeleton, animation))

    with pytest.raises(ValueError, match="sample rate"):
        lafan1.load_lafan1_bvh(tmp_path / "motion.bvh")
