"""Convergence analysis — rate measurement and prediction.

Functions here operate on *history* — a list of snapshots where each snapshot
is a list of participant positions (``list[list[list[float]]]``).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from palavermath.consensus import compute_center, consensus_distance


def convergence_rate(history: Sequence[Sequence[Sequence[float]]]) -> float:
    """Return the average per-step change in consensus distance.

    A negative value means the group is converging; positive means diverging.
    Zero means no movement.

    Parameters
    ----------
    history:
        Snapshots of participant positions over time.

    Returns
    -------
    float
        Average change in consensus distance per step.
    """
    if len(history) < 2:
        return 0.0

    distances = [consensus_distance(list(h)) for h in history]
    diffs = [distances[i + 1] - distances[i] for i in range(len(distances) - 1)]
    return sum(diffs) / len(diffs)


def predict_consensus(
    session: Any,
    max_rounds: int = 200,
) -> list[float]:
    """Predict where a :class:`~palavermath.session.PalaverSession` will converge.

    Runs the session's convergence loop (up to *max_rounds*) on a copy of
    participant data and returns the predicted consensus position.

    Parameters
    ----------
    session:
        A :class:`PalaverSession` instance (or compatible object) that exposes
        ``_participants`` and ``threshold``.
    max_rounds:
        Upper bound on prediction iterations.

    Returns
    -------
    list[float]
        Predicted consensus position.
    """
    # Work on a snapshot to avoid mutating the real session.
    participants = [
        {"position": list(p["position"]), "influence": p["influence"]}
        for p in session._participants
    ]
    if not participants:
        return []

    threshold = getattr(session, "threshold", 0.01)

    for _ in range(max_rounds):
        positions = [p["position"] for p in participants]
        n = len(positions)
        dim = len(positions[0])

        # Weighted center.
        total_inf = sum(p["influence"] for p in participants)
        center = [
            sum(participants[i]["influence"] * positions[i][d] for i in range(n))
            / total_inf
            for d in range(dim)
        ]

        # Average distance.
        avg_dist = (
            sum(
                math.sqrt(sum((positions[i][d] - center[d]) ** 2 for d in range(dim)))
                for i in range(n)
            )
            / n
        )

        if avg_dist <= threshold:
            break

        # Nudge.
        for p in participants:
            openness = 1.0 / (1.0 + p["influence"])
            p["position"] = [
                p["position"][d] + openness * (center[d] - p["position"][d])
                for d in range(dim)
            ]

    final_positions = [p["position"] for p in participants]
    return compute_center(final_positions)


def is_converging(history: Sequence[Sequence[Sequence[float]]]) -> bool:
    """Return ``True`` if the group is trending toward consensus.

    The check is simple: the consensus distance in the last snapshot must be
    less than or equal to that in the first snapshot.

    Parameters
    ----------
    history:
        Snapshots of participant positions over time.

    Returns
    -------
    bool
    """
    if len(history) < 2:
        return True

    first = consensus_distance(list(history[0]))
    last = consensus_distance(list(history[-1]))
    return last <= first
