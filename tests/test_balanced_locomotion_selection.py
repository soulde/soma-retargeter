import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from select_balanced_locomotion import classify_motion, select_balanced_pairs


def test_classify_motion_separates_lateral_directions_and_gaits():
    assert classify_motion("walk_sideway_left_loop_001__A031.bvh") == "walk_left"
    assert classify_motion("walk_sideway_right_loop_001__A030.bvh") == "walk_right"
    assert classify_motion("jog_sideway_045_loop_001__A037.bvh") == "jog_diagonal"
    assert classify_motion("jog_backward_loop_001__A028.bvh") == "jog_backward"
    assert classify_motion("turn_walk_270_R_001__A420.bvh") == "turn_walk"
    assert classify_motion("stairs_climbing_down_loop_R_102__A301.bvh") == "stairs"


def test_selection_keeps_original_and_mirror_together():
    names = [
        "locomotion/walk_sideway_left_loop_001__A031.bvh",
        "locomotion/walk_sideway_left_loop_001__A031_M.bvh",
        "locomotion/walk_sideway_left_loop_002__A024.bvh",
        "locomotion/walk_sideway_left_loop_002__A024_M.bvh",
    ]

    selected, counts = select_balanced_pairs(names, {"walk_left": 2})

    assert selected == [
        "locomotion/walk_sideway_left_loop_001__A031.bvh",
        "locomotion/walk_sideway_left_loop_001__A031_M.bvh",
    ]
    assert counts == {"walk_left": 2}


def test_selection_never_includes_an_unpaired_motion():
    names = [
        "locomotion/jog_forward_loop_001__A034.bvh",
        "locomotion/jog_forward_loop_002__A039.bvh",
        "locomotion/jog_forward_loop_002__A039_M.bvh",
    ]

    selected, counts = select_balanced_pairs(names, {"jog_forward": 4})

    assert selected == [
        "locomotion/jog_forward_loop_002__A039.bvh",
        "locomotion/jog_forward_loop_002__A039_M.bvh",
    ]
    assert counts == {"jog_forward": 2}
