import csv
import os
from pathlib import Path
import subprocess

import numpy as np

from soma_retargeter.assets.csv import get_csv_config_for_target


def test_standard_regeneration_exports_original_120hz_csv_directly_to_50hz(tmp_path):
    csv_root = tmp_path / "csv"
    npz_root = tmp_path / "npz"
    csv_root.mkdir()
    config = get_csv_config_for_target("chocolate")
    rows = []
    for frame in range(4):
        row = np.zeros(len(config.csv_header), dtype=float)
        row[0] = frame
        row[3] = 87.0
        row[7] = frame
        rows.append(row)
    with (csv_root / "walk.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(config.csv_header)
        writer.writerows(rows)

    env = os.environ.copy()
    env.update({
        "SKIP_RETARGET": "1",
        "CSV_ROOT": str(csv_root),
        "NPZ_ROOT": str(npz_root),
    })
    result = subprocess.run(
        ["bash", "scripts/regenerate_chocolate_amp_smooth_20260909.sh"],
        cwd=Path(__file__).parents[1], env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    outputs = list(npz_root.glob("*.npz"))
    assert [path.name for path in outputs] == ["walk.npz"]
    with np.load(outputs[0], allow_pickle=False) as data:
        assert float(data["fps"]) == 50.0
        assert data["joint_pos"].shape == (2, 23)

    env["RESUME"] = "1"
    resumed = subprocess.run(
        ["bash", "scripts/regenerate_chocolate_amp_smooth_20260909.sh"],
        cwd=Path(__file__).parents[1], env=env,
        capture_output=True, text=True,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Skipping existing" in resumed.stdout
