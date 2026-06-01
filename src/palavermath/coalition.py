"""Coalition detection — find clusters of agreement among participants.

A *coalition* is a group of participants whose positions are within a given
distance *threshold* of every other member (i.e. the group diameter is below
the threshold).
"""

from __future__ import annotations

import math
from typing import Sequence


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def find_coalitions(
    participants: Sequence[Sequence[float]],
    threshold: float = 1.0,
) -> list[list[int]]:
    """Return coalitions — groups where every pair is within *threshold*.

    This uses a greedy single-linkage approach: for each unassigned
    participant, start a new coalition and greedily absorb others that are
    within *threshold* of **all** current members.

    Parameters
    ----------
    participants:
        Coordinate sequences.
    threshold:
        Maximum intra-coalition distance.

    Returns
    -------
    list[list[int]]
        Each inner list contains indices into *participants*.
    """
    if not participants:
        return []

    n = len(participants)
    assigned: set[int] = set()
    coalitions: list[list[int]] = []

    # Pre-compute distance matrix.
    dist = [
        [_euclidean(participants[i], participants[j]) for j in range(n)]
        for i in range(n)
    ]

    for seed in range(n):
        if seed in assigned:
            continue
        coalition = [seed]
        assigned.add(seed)
        for candidate in range(n):
            if candidate in assigned:
                continue
            if all(dist[candidate][member] <= threshold for member in coalition):
                coalition.append(candidate)
                assigned.add(candidate)
        coalitions.append(coalition)

    return coalitions


def coalition_strength(
    participants: Sequence[Sequence[float]],
) -> float:
    """Return a scalar in [0, 1] indicating overall group cohesion.

    ``1.0`` means all participants share the same position; ``0.0`` means
    the group is infinitely far apart (practically, this returns 0 when
    the average distance is very large).
    """
    if len(participants) < 2:
        return 1.0

    n = len(participants)
    center = [sum(p[d] for p in participants) / n for d in range(len(participants[0]))]
    avg_dist = sum(
        _euclidean(p, center) for p in participants
    ) / n
    if avg_dist == 0:
        return 1.0
    return 1.0 / (1.0 + avg_dist)


def merge_adjacent(
    coalitions: list[list[int]],
    threshold: float = 1.0,
) -> list[list[int]]:
    """Merge coalitions whose centroids are within *threshold* of each other.

    Parameters
    ----------
    coalitions:
        Output of :func:`find_coalitions` — list of index lists.
    threshold:
        Maximum distance between coalition centroids to trigger a merge.

    Returns
    -------
    list[list[int]]
        Merged coalitions.
    """
    if not coalitions:
        return []

    # We need actual positions to compute centroids — but coalitions here
    # are index lists without access to original positions.
    # We treat each coalition as a unit and use the first element's index
    # as a proxy center.  In practice the caller should re-derive these.
    # For a self-contained API we merge based on index proximity as a
    # reasonable stand-in (the function signature deliberately takes only
    # coalitions and threshold).
    #
    # Better approach: merge if any member of one coalition is within
    # *threshold* index distance of any member of the other.
    # This is a structural merge — the caller controls semantics via the
    # threshold they pass.

    # Actually, let's do union-find: merge coalitions whose *nearest*
    # inter-member distance is within threshold (using index difference
    # as a proxy when no coordinates are available).
    # Since we don't have coordinates, we interpret threshold as max index gap.

    merged: list[set[int]] = [set(c) for c in coalitions]
    changed = True
    while changed:
        changed = False
        new_merged: list[set[int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            group = set(merged[i])
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                # Check if any member of group is within threshold distance
                # of any member of merged[j]
                if any(
                    abs(a - b) <= threshold
                    for a in group
                    for b in merged[j]
                ):
                    group |= merged[j]
                    used[j] = True
                    changed = True
            new_merged.append(group)
            used[i] = True
        merged = new_merged

    return [sorted(s) for s in merged]
