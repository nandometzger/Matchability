"""Coerce arbitrary image inputs to canonical ``(H, W, 3)`` uint8 RGB arrays.

Accepts file paths, PIL images, numpy arrays (uint8 or float, gray/RGB/RGBA), and
torch tensors (``CHW`` or ``HWC``, float in ``[0, 1]`` or uint8). This is what
makes the public API forgiving about input types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

ImageLike = Any


def resize_max_side(image: NDArray[np.uint8], max_side: int) -> NDArray[np.uint8]:
    """Downscale so the longer side equals ``max_side`` (never upscales)."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _from_array(arr: NDArray) -> NDArray[np.uint8]:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(arr, initial=0.0)) <= 1.0 + 1e-6 else 1.0
        arr = np.clip(np.round(arr * scale), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"cannot interpret array of shape {arr.shape} as an image")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _from_torch(tensor: Any) -> NDArray[np.uint8]:
    arr = tensor.detach().to("cpu").numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    return _from_array(arr)


def load_image(src: ImageLike, max_side: int | None = None) -> NDArray[np.uint8]:
    """Load ``src`` as a canonical ``(H, W, 3)`` uint8 RGB array.

    Args:
        src: path, PIL image, numpy array, or torch tensor.
        max_side: if given, downscale so the longer side is at most this many pixels.
    """
    if isinstance(src, (str, Path)):
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {src}")
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif type(src).__module__.split(".")[0] == "torch":
        out = _from_torch(src)
    elif type(src).__module__.startswith("PIL"):
        out = _from_array(np.array(src.convert("RGB")))
    else:
        out = _from_array(src)

    if max_side is not None:
        out = resize_max_side(out, max_side)
    return out
