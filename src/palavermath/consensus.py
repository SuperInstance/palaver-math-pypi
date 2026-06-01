"""Consensus computation — geometric center and weighted aggregation.

Participants are expressed as sequences of float coordinates.  A *participant*
is any ``Sequence[float]`` (list, tuple, etc.) where the length equals the
number of dimensions.  All participants in a single call must share the same
dimensionality.
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_participants(participants: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert to list-of-lists and validate uniform dimensionality."""
    if not participants:
        raise ValueError("At least one participant is required")
    pts = [list(p) for p in participants]
    dim = len(pts[0])
    if dim == 0:
        raise ValueError("Participants must have at least one dimension")
    for i, p in enumerate(pts):
        if len(p) != dim:
            raise ValueError(
                f"Participant {i} has {len(p)} dimensions, expected {dim}"
            )
    return pts, dim  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_center(participants: Sequence[Sequence[float]]) -> list[float]:
    """Return the geometric centroid (unweighted mean) of *participants*.

    Parameters
    ----------
    participants:
        A non-empty sequence of equal-length coordinate sequences.

    Returns
    -------
    list[float]
        The centroid as a list of floats with the same dimensionality.

    Examples
    --------
    >>> compute_center([[0, 0], [2, 2]])
    [1.0, 1.0]
    """
    pts, dim = _validate_participants(participants)
    n = len(pts)
    return [sum(p[d] for p in pts) / n for d in range(dim)]


def consensus_distance(participants: Sequence[Sequence[float]]) -> float:
    """Return the average Euclidean distance from each participant to the center.

    This gives a scalar measure of how spread out the participants are —
    lower values indicate stronger consensus.

    Parameters
    ----------
    participants:
        A non-empty sequence of equal-length coordinate sequences.

    Returns
    -------
    float
        The mean distance to the centroid.
    """
    pts, _ = _validate_participants(participants)
    center = compute_center(pts)
    total = sum(
        math.sqrt(sum((p[d] - center[d]) ** 2 for d in range(len(center))))
        for p in pts
    )
    return total / len(pts)


def influence_weighted_center(
    participants: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
) -> list[float]:
    """Return a weighted centroid where each participant may carry influence.

    If *weights* is ``None``, every participant receives equal weight (equivalent
    to :func:`compute_center`).  Otherwise *weights* must be the same length
    as *participants* and all values must be non-negative with at least one
    positive entry.

    Parameters
    ----------
    participants:
        A non-empty sequence of equal-length coordinate sequences.
    weights:
        Optional per-participant influence weights.

    Returns
    -------
    list[float]
        The weighted centroid.
    """
    pts, dim = _validate_participants(participants)
    n = len(pts)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("weights must match the number of participants")
    if all(w <= 0 for w in weights):
        raise ValueError("At least one weight must be positive")
    total_w = sum(weights)
    return [
        sum(weights[i] * pts[i][d] for i in range(n)) / total_w
        for d in range(dim)
    ]
