"""The high-level, easy-to-use API: :class:`Matchability` and :func:`matchability_error`.

The default backend is DeDoDe v2 (loaded lazily so importing this package stays
cheap and the classical/mock backends work without torch). The model and its
weights are auto-downloaded and memoised on first use.
"""

from __future__ import annotations

from matchability.imageio_util import ImageLike, load_image
from matchability.matchers.base import Matcher
from matchability.metric import compute_matchability
from matchability.types import MatchabilityResult


class Matchability:
    """Reusable Matchability Error estimator.

    Example:
        >>> metric = Matchability()                       # default DeDoDe v2
        >>> res = metric(left, right_gt, right_pred)
        >>> res.error_pct                                 # doctest: +SKIP

    Args:
        matcher: matcher backend. ``None`` (default) lazily builds DeDoDe v2.
        tau: epipolar vertical-disparity threshold, in pixels at the working resolution.
        n_keypoints: size of the fixed reference keypoint set in the left image.
        conf_threshold: minimum match confidence to count a correspondence.
        working_resolution: longer-side resize applied to every image (``None`` keeps native).
        disparity_bounds: optional ``(d_min, d_max)`` horizontal-disparity bound.
        device: torch device for the default DeDoDe backend (``"mps"``/``"cuda"``/``"cpu"``).
    """

    def __init__(
        self,
        matcher: Matcher | None = None,
        *,
        tau: float = 2.0,
        n_keypoints: int = 5000,
        conf_threshold: float = 0.0,
        working_resolution: int | None = 768,
        disparity_bounds: tuple[float, float] | None = None,
        device: str | None = None,
    ) -> None:
        self._matcher = matcher
        self.tau = tau
        self.n_keypoints = n_keypoints
        self.conf_threshold = conf_threshold
        self.working_resolution = working_resolution
        self.disparity_bounds = disparity_bounds
        self._device = device

    @property
    def matcher(self) -> Matcher:
        """The matcher backend, building the default DeDoDe v2 on first access."""
        if self._matcher is None:
            from matchability.matchers.dedode import DeDoDeV2Matcher

            self._matcher = DeDoDeV2Matcher(device=self._device)
        return self._matcher

    def __call__(
        self, left: ImageLike, right_gt: ImageLike, right_pred: ImageLike
    ) -> MatchabilityResult:
        left_img = load_image(left, self.working_resolution)
        right_gt_img = load_image(right_gt, self.working_resolution)
        right_pred_img = load_image(right_pred, self.working_resolution)
        if not (left_img.shape == right_gt_img.shape == right_pred_img.shape):
            raise ValueError(
                "left, right_gt and right_pred must share a shape after loading, got "
                f"{left_img.shape}, {right_gt_img.shape}, {right_pred_img.shape}"
            )
        return compute_matchability(
            left_img,
            right_gt_img,
            right_pred_img,
            matcher=self.matcher,
            tau=self.tau,
            conf_threshold=self.conf_threshold,
            disparity_bounds=self.disparity_bounds,
            n_keypoints=self.n_keypoints,
        )


def matchability_error(
    left: ImageLike, right_gt: ImageLike, right_pred: ImageLike, **kwargs
) -> MatchabilityResult:
    """One-shot Matchability Error for a ``(left, right_gt, right_pred)`` triplet.

    Keyword arguments are forwarded to :class:`Matchability`. For repeated calls,
    construct a :class:`Matchability` once and reuse it (the DeDoDe model is then
    loaded a single time).
    """
    return Matchability(**kwargs)(left, right_gt, right_pred)
