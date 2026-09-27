from soma_retargeter.assets.lafan1 import load_lafan1_bvh
from soma_retargeter.pipelines.utils import (
    SourceType,
    get_source_model_mesh,
    motion_source_descriptor,
)
from soma_retargeter.utils.space_conversion_utils import get_view_transform_for_source
import numpy as np


def test_lafan1_motion_source_descriptor():
    descriptor = motion_source_descriptor("lafan1")

    assert descriptor.extension == ".bvh"
    assert descriptor.load is load_lafan1_bvh
    assert descriptor.root_transform_is_identity is True


def test_lafan1_has_no_bundled_mesh():
    assert get_source_model_mesh(SourceType.LAFAN1, skeleton=None) is None


def test_lafan1_view_transform_is_identity_after_loader_normalizes_axes():
    transform = get_view_transform_for_source("lafan1", "Mujoco")
    assert np.allclose(transform.p, (0.0, 0.0, 0.0))
    assert np.allclose(transform.q, (0.0, 0.0, 0.0, 1.0))


def test_existing_source_descriptors_remain_compatible():
    assert motion_source_descriptor("soma").extension == ".bvh"
    assert motion_source_descriptor("soma").root_transform_is_identity is False
    assert motion_source_descriptor("smplx").extension == ".npz"
    # SMPL-X motions are converted onto the SOMA skeleton in the SOMA BVH
    # frame, so they follow the same converter path as soma BVHs.
    assert motion_source_descriptor("smplx").root_transform_is_identity is False
