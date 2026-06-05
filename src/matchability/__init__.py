"""Matchability Error — a DeDoDe-v2 stereoscopic-fidelity metric (Elastic3D).

Public API is intentionally tiny::

    from matchability import matchability_error
    res = matchability_error(left, right_gt, right_pred)
    print(res.error, res.tp, res.fp, res.fn)

See ``docs/metric.md`` for the precise definition.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("matchability")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
