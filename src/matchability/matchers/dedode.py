"""The DeDoDe v2 matcher backend (the faithful default).

Uses kornia's DeDoDe with the ``L-C4-v2`` detector (DeDoDe v2, arXiv 2404.08928 --
reference [9] in Elastic3D) and the ``G-upright`` descriptor. Keypoints are matched
with a dual-softmax mutual nearest neighbour. The model and its weights are
auto-downloaded and memoised; the device is auto-selected (mps > cuda > cpu) with
Apple-Silicon MPS handled as a first-class citizen.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

from matchability.matchers.base import Matcher
from matchability.types import LeftToRightMatches

_MODEL_CACHE: dict[tuple, object] = {}


def _select_device(device: str | None):
    import torch

    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DeDoDeV2Matcher(Matcher):
    """DeDoDe v2 keypoints + descriptors with dual-softmax mutual matching."""

    def __init__(
        self,
        device: str | None = None,
        *,
        detector_weights: str = "L-C4-v2",
        descriptor_weights: str = "G-upright",
        n_right: int = 10000,
        inv_temperature: float = 20.0,
        min_confidence: float = 0.0,
    ) -> None:
        import torch

        self._torch = torch
        self.device = _select_device(device)
        if self.device.type == "mps":
            # Let any op not yet implemented on MPS fall back to CPU instead of erroring.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # float16 is for CUDA; CPU/MPS need float32 (per kornia's DeDoDe docs).
        self._amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._detector_weights = detector_weights
        self._descriptor_weights = descriptor_weights
        self.n_right = n_right
        self.inv_temperature = inv_temperature
        self.min_confidence = min_confidence
        self._model = self._load_model()

    def _load_model(self):
        key = (
            str(self.device),
            self._detector_weights,
            self._descriptor_weights,
            str(self._amp_dtype),
        )
        if key not in _MODEL_CACHE:
            import kornia.feature as KF

            model = KF.DeDoDe.from_pretrained(
                detector_weights=self._detector_weights,
                descriptor_weights=self._descriptor_weights,
                amp_dtype=self._amp_dtype,
            )
            _MODEL_CACHE[key] = model.to(self.device).eval()
        return _MODEL_CACHE[key]

    def _prep(self, image: NDArray[np.uint8]):
        """Pad to a multiple of the ViT patch size (14) and return (tensor, H_pad, W_pad).

        DeDoDe's descriptor requires divisible-by-14 inputs. Padding is edge-replicated
        on the bottom/right so original pixel coordinates are unchanged; the padded dims
        are then used consistently for (de)normalising keypoints.
        """
        torch = self._torch
        h, w = image.shape[:2]
        pad_h, pad_w = (14 - h % 14) % 14, (14 - w % 14) % 14
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        t = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).float().div_(255.0)
        return t.permute(2, 0, 1).unsqueeze(0), image.shape[0], image.shape[1]

    def detect(self, image: NDArray[np.uint8], n: int | None = None) -> NDArray[np.floating]:
        from kornia.geometry.conversions import denormalize_pixel_coordinates

        tensor, h_pad, w_pad = self._prep(image)
        keypoints, _ = self._model.detect(tensor, n=n or self.n_right)
        xy = denormalize_pixel_coordinates(keypoints[0], h_pad, w_pad)
        return xy.detach().cpu().numpy().astype(float)

    def _left_descriptors(self, image_left: NDArray[np.uint8], keypoints_left: NDArray):
        """Normalised descriptors for the fixed left keypoints, cached across calls.

        The left side is identical for every distortion of a pair, so caching it
        removes one of the three ViT forwards per match.
        """
        torch = self._torch
        from kornia.geometry.conversions import normalize_pixel_coordinates

        flat = np.asarray(image_left).reshape(-1)
        key = (
            id(image_left),
            id(keypoints_left),
            image_left.shape,
            len(keypoints_left),
            int(flat[0]),
            int(flat[-1]),
        )
        cached = getattr(self, "_left_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        tensor_left, hl, wl = self._prep(image_left)
        kpt_left = torch.as_tensor(
            np.asarray(keypoints_left, dtype=float), device=self.device, dtype=torch.float32
        )
        norm_left = normalize_pixel_coordinates(kpt_left, hl, wl).unsqueeze(0)  # (1, K, 2)
        with torch.inference_mode():
            desc_left = self._model.describe(tensor_left, keypoints=norm_left)[0]  # (K, D)
            desc_left = torch.nn.functional.normalize(desc_left, dim=-1)
        self._left_cache = (key, desc_left)
        return desc_left

    def match(
        self,
        image_left: NDArray[np.uint8],
        keypoints_left: NDArray[np.floating],
        image_right: NDArray[np.uint8],
    ) -> LeftToRightMatches:
        torch = self._torch
        from kornia.geometry.conversions import denormalize_pixel_coordinates

        k = len(keypoints_left)
        if k == 0:
            return LeftToRightMatches(np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool))

        desc_left = self._left_descriptors(image_left, keypoints_left)  # cached, normalised (K, D)
        tensor_right, hr, wr = self._prep(image_right)

        with torch.inference_mode():
            kpt_right_norm, _ = self._model.detect(tensor_right, n=self.n_right)  # (1, M, 2)
            desc_right = self._model.describe(tensor_right, keypoints=kpt_right_norm)[0]  # (M, D)
            desc_right = torch.nn.functional.normalize(desc_right, dim=-1)
            logits = self.inv_temperature * (desc_left @ desc_right.T)  # (K, M)
            prob = logits.softmax(dim=1) * logits.softmax(dim=0)  # dual softmax

            best_right = prob.argmax(dim=1)  # (K,)
            confidence = prob.gather(1, best_right[:, None]).squeeze(1)
            best_left_for_right = prob.argmax(dim=0)  # (M,)
            mutual = best_left_for_right[best_right] == torch.arange(k, device=self.device)
            xy_right = denormalize_pixel_coordinates(kpt_right_norm[0], hr, wr)  # (M, 2)
            right_xy = xy_right[best_right]
            valid = mutual & (confidence >= self.min_confidence)

        return LeftToRightMatches(
            right_xy.detach().cpu().numpy().astype(float),
            confidence.detach().cpu().numpy().astype(float),
            valid.detach().cpu().numpy().astype(bool),
        )
