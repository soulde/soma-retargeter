import os

import numpy as np
import pytest
import warp as wp

from app.csv_to_npz import convert_csv_to_npz
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.assets import csv as csv_utils
from soma_retargeter.assets.lafan1 import load_lafan1_bvh
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


DATA_ROOT = "/home/jvwei/datasets/lafan1"
MOTIONS = (
    "walk1_subject1.bvh",
    "run1_subject2.bvh",
    "dance1_subject1.bvh",
)


@pytest.mark.parametrize("filename", MOTIONS)
def test_lafan1_chocolate_short_clip_retargets_and_exports(filename, tmp_path):
    path = os.path.join(DATA_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"missing local LAFAN1 fixture: {path}")

    skeleton, source = load_lafan1_bvh(path)
    start = min(100, max(0, source.num_frames - 12))
    frame_count = min(12, source.num_frames - start)
    short = AnimationBuffer(
        skeleton,
        frame_count,
        source.sample_rate,
        source.local_transforms[start:start + frame_count].copy(),
    )

    pipeline = NewtonPipeline(skeleton, "lafan1", "chocolate")
    pipeline.add_input_motions([short], [wp.transform_identity()], True)
    assert np.max(np.std(pipeline.input_targets[0][:, :, :3], axis=0)) > 1e-4
    output = pipeline.execute()[0]
    qpos = np.stack(output.data).astype(np.float32)

    assert qpos.shape == (frame_count, 7 + 23)
    assert output.sample_rate == pytest.approx(source.sample_rate)
    assert np.isfinite(qpos).all()
    assert np.all((qpos[:, 2] > 0.3) & (qpos[:, 2] < 1.5))
    lower = pipeline.ik_model.joint_limit_lower.numpy()[6:]
    upper = pipeline.ik_model.joint_limit_upper.numpy()[6:]
    assert np.all(qpos[:, 7:] >= lower - 1e-5)
    assert np.all(qpos[:, 7:] <= upper + 1e-5)
    if wp.get_device().is_cuda:
        assert np.max(np.std(qpos[:, 7:], axis=0)) > 1e-4
    # The source legs must remain independently constrained rather than being
    # accidentally mirrored onto the same Chocolate coordinates.
    assert not np.allclose(qpos[:, 10:16], qpos[:, 16:22], atol=1e-5)

    csv_path = tmp_path / f"{filename[:-4]}.csv"
    npz_path = tmp_path / f"{filename[:-4]}.npz"
    csv_utils.save_csv(
        csv_path, output, csv_utils.get_csv_config_for_target("chocolate"))
    convert_csv_to_npz(
        csv_path, npz_path, "chocolate",
        input_fps=source.sample_rate, output_fps=50.0,
    )

    with np.load(npz_path) as exported:
        expected_frames = round((frame_count - 1) / source.sample_rate * 50.0) + 1
        assert float(exported["fps"]) == 50.0
        assert exported["joint_pos"].shape == (expected_frames, 23)
        for field in (
            "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        ):
            assert np.isfinite(exported[field]).all()
