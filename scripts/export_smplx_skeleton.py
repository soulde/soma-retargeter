#!/usr/bin/env python3
"""Export the SMPL-X rest skeleton (parents + zero-pose joint offsets) to JSON.

Run inside an environment that has ``smplx`` and ``torch`` installed, e.g.::

    ~/GMR-private/.venv/bin/python scripts/export_smplx_skeleton.py \
        --body-models /home/jvwei/GMR-private/GMR/assets/body_models \
        --output soma_retargeter/configs/smplx/smplx_rest_skeleton.json

Only the 22 body joints (root + the 21 joints driven by ``pose_body``) are
exported; jaw/eye/hand joints are dropped because their pose channels are
zeroed during retargeting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES

NUM_BODY_JOINTS = 22  # root (pelvis) + 21 joints driven by pose_body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-models", type=Path, required=True,
                        help="Directory containing SMPLX_NEUTRAL.pkl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = smplx.create(str(args.body_models), "smplx", gender="neutral",
                         use_pca=False, ext="pkl")
    output = model()
    joints = output.joints[0, :NUM_BODY_JOINTS].detach().cpu().numpy()  # (22, 3)
    parents = np.asarray(model.parents[:NUM_BODY_JOINTS], dtype=int)

    names = list(JOINT_NAMES[:NUM_BODY_JOINTS])
    # Offsets are the canonical joint positions relative to the parent. The
    # pelvis keeps its own canonical position J0: SMPL-X computes world joints
    # as transl + root_R @ J_i, so the FK root must carry J0 as its offset.
    offsets = [joints[0].tolist()]
    for i in range(1, NUM_BODY_JOINTS):
        offsets.append((joints[i] - joints[parents[i]]).tolist())

    data = {
        "description": "SMPL-X neutral rest skeleton, 22 body joints, "
                       "original SMPL-X coordinates (Y-up, forward +Z)",
        "up_axis": [0.0, 1.0, 0.0],
        "forward_axis": [0.0, 0.0, 1.0],
        "joint_names": names,
        "parents": parents.tolist(),
        "offsets": offsets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {args.output} ({NUM_BODY_JOINTS} joints)")


if __name__ == "__main__":
    main()
