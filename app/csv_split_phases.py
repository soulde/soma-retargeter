#!/usr/bin/env python3
"""Split a CSV motion into one decimated CSV for each sampling phase."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from soma_retargeter.utils.frame_sampling import split_frame_phases


def split_csv(csv_path: Path, output_dir: Path, input_fps: float = 120.0,
              stride: int = 4) -> list[Path]:
    """Write one decimated CSV for every phase while preserving the header."""
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    frames = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    output_fps = input_fps / stride
    outputs = []
    for phase, phase_frames in enumerate(split_frame_phases(frames, stride)):
        if phase_frames.shape[0] == 0:
            continue
        phase_frames[:, 0] = np.arange(phase_frames.shape[0])
        output = output_dir / f"{csv_path.stem}_phase{phase}_{output_fps:g}hz.csv"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, phase_frames, delimiter=",", header=",".join(header), comments="")
        outputs.append(output)
    return outputs


def split_path(input_path: Path, output_dir: Path, input_fps: float,
               stride: int) -> None:
    """Split one CSV or all CSV files below a directory."""
    csv_files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found under {input_path}")
    for source in csv_files:
        relative_parent = Path() if input_path.is_file() else source.relative_to(input_path).parent
        split_csv(source, output_dir / relative_parent, input_fps, stride)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input CSV file or directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    split_path(args.input, args.output_dir, args.input_fps, args.stride)


if __name__ == "__main__":
    main()
