"""Matcher backends for the Matchability metric.

A matcher detects a fixed reference keypoint set in the left image and matches it
into a right view. The metric core is agnostic to which backend is used.
"""

from matchability.matchers.base import Matcher

__all__ = ["Matcher"]
