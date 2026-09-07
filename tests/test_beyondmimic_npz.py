import numpy as np

from soma_retargeter.assets.beyondmimic_npz import resample_motion, finite_difference


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
