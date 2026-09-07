"""Frame sampling helpers shared by motion conversion tools."""

from __future__ import annotations

import numpy as np


def split_frame_phases(frames: np.ndarray, stride: int) -> list[np.ndarray]:
    """Split frames into one decimated sequence for each phase offset."""
    frames = np.asarray(frames)
    if frames.ndim < 1:
        raise ValueError("frames must have at least one dimension")
    if stride <= 0:
        raise ValueError("stride must be positive")
    return [frames[phase::stride].copy() for phase in range(stride)]
