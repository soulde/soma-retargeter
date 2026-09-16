import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from report_locomotion_npz_quality import assess_motion


def test_left_sideway_with_forward_only_velocity_is_flagged():
    metrics, flags = assess_motion(
        "walk_sideway_left_loop.bvh",
        np.tile([0.6, 0.0, 0.0], (20, 1)),
        np.zeros((20, 3)),
    )

    assert np.isclose(metrics["mean_vx"], 0.6)
    assert "direction_mismatch_left" in flags


def test_left_sideway_with_positive_lateral_velocity_is_retained_without_flag():
    _, flags = assess_motion(
        "walk_sideway_left_loop.bvh",
        np.tile([0.0, 0.4, 0.0], (20, 1)),
        np.zeros((20, 3)),
    )

    assert "direction_mismatch_left" not in flags
