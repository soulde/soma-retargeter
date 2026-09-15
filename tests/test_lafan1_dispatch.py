from soma_retargeter.assets.lafan1 import load_lafan1_bvh
from soma_retargeter.pipelines.utils import (
    SourceType,
    get_source_model_mesh,
    motion_source_descriptor,
)


def test_lafan1_motion_source_descriptor():
    descriptor = motion_source_descriptor("lafan1")

    assert descriptor.extension == ".bvh"
    assert descriptor.load is load_lafan1_bvh
    assert descriptor.root_transform_is_identity is True


def test_lafan1_has_no_bundled_mesh():
    assert get_source_model_mesh(SourceType.LAFAN1, skeleton=None) is None


def test_existing_source_descriptors_remain_compatible():
    assert motion_source_descriptor("soma").extension == ".bvh"
    assert motion_source_descriptor("soma").root_transform_is_identity is False
    assert motion_source_descriptor("smplx").extension == ".npz"
    assert motion_source_descriptor("smplx").root_transform_is_identity is True
