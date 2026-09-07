#!/usr/bin/env python3
"""Split 120 Hz CSV motion into four 30 Hz phases and export each at 50 Hz."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from app.csv_to_npz import convert_csv_to_npz
from soma_retargeter.pipelines import utils as pipeline_utils
from soma_retargeter.utils.frame_sampling import split_frame_phases


def split_csv(csv_path: Path, output_dir: Path, stride: int = 4,
              phase_fps: float = 30.0) -> list[Path]:
    """Write one decimated CSV for each phase, preserving the source header."""
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    frames = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    outputs = []
    for phase, phase_frames in enumerate(split_frame_phases(frames, stride)):
        if phase_frames.shape[0] == 0:
            continue
        phase_frames[:, 0] = np.arange(phase_frames.shape[0])
        output = output_dir / f"{csv_path.stem}_phase{phase}_{phase_fps:g}hz.csv"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, phase_frames, delimiter=",", header=",".join(header), comments="")
        outputs.append(output)
    return outputs


def convert_path(input_path: Path, output_dir: Path, robot: str,
                 input_fps: float, stride: int, output_fps: float) -> None:
    """Process one CSV or recursively process all CSV files in a directory."""
    csv_files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found under {input_path}")
    phase_fps = input_fps / stride
    for source in csv_files:
        relative_parent = Path() if input_path.is_file() else source.relative_to(input_path).parent
        target_dir = output_dir / relative_parent
        for phase_csv in split_csv(source, target_dir, stride, phase_fps):
            phase_stem = phase_csv.stem.rsplit("_", 1)[0]
            npz_path = phase_csv.with_name(f"{phase_stem}_{output_fps:g}hz.npz")
            if npz_path.exists():
                raise FileExistsError(f"Refusing to overwrite {npz_path}")
            convert_csv_to_npz(
                phase_csv, npz_path, robot_type=robot,
                input_fps=phase_fps, output_fps=output_fps,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input CSV file or directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--robot", required=True, choices=pipeline_utils.get_registered_targets())
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output-fps", type=float, default=50.0)
    args = parser.parse_args()
    convert_path(args.input, args.output_dir, args.robot, args.input_fps, args.stride, args.output_fps)


if __name__ == "__main__":
    main()
