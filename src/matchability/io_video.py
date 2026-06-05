"""Extract left/right frames from MV-HEVC stereo videos (e.g. Apple Vision Pro).

Both eyes are stored as separate views inside one file; we pull each view with
ffmpeg and decode it in-memory. View 0 is the left eye, view 1 the right (verified
on the AVP data: horizontal disparity x_L - x_R is positive).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to extract video frames")
    return exe


def _extract_view(path: str | Path, view: int, frame_index: int) -> NDArray[np.uint8]:
    select = f"select=eq(n\\,{frame_index})"
    cmd = [
        _ffmpeg(),
        "-v", "error",
        "-i", str(path),
        "-map", f"0:v:view:{view}",
        "-vf", select,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    buffer = np.frombuffer(result.stdout, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to decode view {view} frame {frame_index} from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_stereo_frame(
    path: str | Path,
    frame_index: int = 0,
    *,
    view_left: int = 0,
    view_right: int = 1,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Extract one ``(left, right)`` RGB frame pair from an MV-HEVC stereo video."""
    left = _extract_view(path, view_left, frame_index)
    right = _extract_view(path, view_right, frame_index)
    return left, right
