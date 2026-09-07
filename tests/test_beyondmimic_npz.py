import numpy as np

from soma_retargeter.assets.beyondmimic_npz import (
    finite_difference,
    resample_motion,
)
from soma_retargeter.utils.frame_sampling import split_frame_phases


def test_resample_motion_preserves_endpoints_and_interpolates_dofs():
    frames = np.array(
        [[0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1, 2]],
        dtype=np.float32,
    )
    sampled = resample_motion(frames, input_fps=1.0, output_fps=2.0)
    np.testing.assert_allclose(sampled[:, 0], [0, 0.5, 1.0])
    np.testing.assert_allclose(sampled[:, -1], [0, 1, 2])
    assert sampled.shape == (3, 8)


def test_finite_difference_uses_seconds():
    values = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    np.testing.assert_allclose(finite_difference(values, fps=2.0)[:, 0], 4.0)


def test_split_frame_phases_handles_non_divisible_frame_count():
    frames = np.arange(20, dtype=np.float32).reshape(10, 2)
    phases = split_frame_phases(frames, stride=4)
    assert [phase.shape[0] for phase in phases] == [3, 3, 2, 2]
    np.testing.assert_array_equal(phases[0][:, 0], [0, 8, 16])
    np.testing.assert_array_equal(phases[1][:, 0], [2, 10, 18])
