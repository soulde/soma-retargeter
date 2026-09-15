import os
import subprocess

import numpy as np

from soma_retargeter.assets.csv import Chocolate23DOF_CSVConfig


SCRIPT = os.path.abspath("scripts/export_lafan1_chocolate_npz50.sh")


def _write_inputs(root, csv_root):
    root.mkdir()
    csv_root.mkdir()
    (root / "walk_fixture.bvh").write_text(
        "HIERARCHY\nMOTION\nFrames: 2\nFrame Time: 0.0333333333\n")
    header = Chocolate23DOF_CSVConfig.csv_header
    row0 = np.zeros(len(header), dtype=np.float32)
    row1 = row0.copy()
    row0[0], row1[0] = 0, 1
    row0[3] = row1[3] = 80.0
    np.savetxt(
        csv_root / "walk_fixture.csv", np.stack([row0, row1]), delimiter=",",
        header=",".join(header), comments="",
    )


def test_export_directly_resamples_and_resumes(tmp_path):
    source = tmp_path / "source"
    csv_root = tmp_path / "csv"
    npz_root = tmp_path / "npz"
    _write_inputs(source, csv_root)
    env = {
        **os.environ,
        "INPUT_ROOT": str(source),
        "CSV_ROOT": str(csv_root),
        "NPZ_ROOT": str(npz_root),
        "SKIP_RETARGET": "1",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }

    generated = subprocess.run([SCRIPT], env=env, capture_output=True, text=True)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    output = npz_root / "walk_fixture.npz"
    with np.load(output) as data:
        assert float(data["fps"]) == 50.0
        assert data["joint_pos"].shape == (3, 23)
    assert not list(npz_root.glob("*phase*.npz"))

    refused = subprocess.run([SCRIPT], env=env, capture_output=True, text=True)
    assert refused.returncode != 0
    assert "Refusing to overwrite" in refused.stderr

    resumed = subprocess.run(
        [SCRIPT], env={**env, "RESUME": "1"},
        check=True, capture_output=True, text=True,
    )
    assert "Skipping verified output walk_fixture" in resumed.stdout
