"""Palaver session — iterative convergence toward consensus.

A :class:`PalaverSession` models a deliberative process in which participants
propose positions, vote on them, and iteratively converge toward a shared
consensus point.  The process is inspired by the West African *palaver* — a
communal dialogue that seeks unanimous agreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class PalaverSession:
    """Iterative consensus session with configurable convergence threshold.

    Attributes
    ----------
    threshold:
        Maximum average distance from center to consider consensus reached.
        Defaults to ``0.01``.
    max_rounds:
        Safety cap on iteration count.  Defaults to ``100``.
    """

    threshold: float = 0.01
    max_rounds: int = 100

    _participants: list[dict[str, Any]] = field(
        default_factory=list, repr=False
    )
    _proposals: list[dict[str, Any]] = field(
        default_factory=list, repr=False
    )
    _votes: list[list[int]] = field(default_factory=list, repr=False)
    _round: int = field(default=0, repr=False)
    _history: list[list[list[float]]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_participant(
        self,
        position: Sequence[float],
        name: str | None = None,
        influence: float = 1.0,
    ) -> int:
        """Register a participant and return their index.

        Parameters
        ----------
        position:
            Starting coordinate position.
        name:
            Optional human-readable label.
        influence:
            Relative influence weight (default 1.0).
        """
        idx = len(self._participants)
        self._participants.append(
            {
                "name": name or f"participant_{idx}",
                "position": list(position),
                "influence": influence,
            }
        )
        return idx

    def add_proposal(
        self,
        position: Sequence[float],
        proposer: int | None = None,
    ) -> int:
        """Add a proposal position and return its index."""
        idx = len(self._proposals)
        self._proposals.append(
            {
                "position": list(position),
                "proposer": proposer,
            }
        )
        return idx

    def vote(self, participant_idx: int, proposal_idx: int) -> None:
        """Record that *participant_idx* accepts *proposal_idx*.

        On acceptance the participant's position shifts toward the proposal
        by a factor proportional to their openness (``1 / (1 + influence)``).
        """
        if participant_idx < 0 or participant_idx >= len(self._participants):
            raise IndexError(f"Participant {participant_idx} out of range")
        if proposal_idx < 0 or proposal_idx >= len(self._proposals):
            raise IndexError(f"Proposal {proposal_idx} out of range")

        while len(self._votes) <= participant_idx:
            self._votes.append([])
        self._votes[participant_idx].append(proposal_idx)

        # Move participant toward the proposal.
        p = self._participants[participant_idx]
        prop = self._proposals[proposal_idx]["position"]
        influence = p["influence"]
        openness = 1.0 / (1.0 + influence)
        p["position"] = [
            p["position"][d] + openness * (prop[d] - p["position"][d])
            for d in range(len(p["position"]))
        ]

    # ------------------------------------------------------------------
    # Consensus computation
    # ------------------------------------------------------------------

    def compute_consensus(self) -> dict[str, Any]:
        """Run the iterative convergence loop and return the result.

        Returns
        -------
        dict
            ``position`` — the final consensus coordinate.
            ``confidence`` — 1 minus the final consensus distance (clamped).
            ``rounds`` — number of iterations performed.
        """
        if not self._participants:
            return {"position": [], "confidence": 0.0, "rounds": 0}

        for round_idx in range(self.max_rounds):
            self._round = round_idx + 1
            positions = [list(p["position"]) for p in self._participants]
            self._history.append(positions)

            dist = self._avg_distance(positions)
            if dist <= self.threshold:
                break

            # Nudge every participant toward the center (weighted).
            influences = [p["influence"] for p in self._participants]
            total_inf = sum(influences)
            center = [
                sum(influences[i] * positions[i][d] for i in range(len(positions)))
                / total_inf
                for d in range(len(positions[0]))
            ]
            for i, p in enumerate(self._participants):
                openness = 1.0 / (1.0 + p["influence"])
                p["position"] = [
                    p["position"][d] + openness * (center[d] - p["position"][d])
                    for d in range(len(center))
                ]

        final_positions = [list(p["position"]) for p in self._participants]
        final_center = self._weighted_center(final_positions)
        final_dist = self._avg_distance(final_positions)

        return {
            "position": final_center,
            "confidence": max(0.0, min(1.0, 1.0 - final_dist)),
            "rounds": self._round,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def participants(self) -> list[dict[str, Any]]:
        """Snapshot of current participant data."""
        return [dict(p) for p in self._participants]

    @property
    def history(self) -> list[list[list[float]]]:
        """Position history: ``history[round][participant][dim]``."""
        return self._history

    @property
    def round(self) -> int:
        """Rounds completed so far."""
        return self._round

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_distance(positions: list[list[float]]) -> float:
        if len(positions) < 2:
            return 0.0
        n = len(positions)
        dim = len(positions[0])
        center = [sum(p[d] for p in positions) / n for d in range(dim)]
        total = sum(
            math.sqrt(sum((p[d] - center[d]) ** 2 for d in range(dim)))
            for p in positions
        )
        return total / n

    @staticmethod
    def _weighted_center(positions: list[list[float]]) -> list[float]:
        if not positions:
            return []
        n = len(positions)
        dim = len(positions[0])
        return [sum(p[d] for p in positions) / n for d in range(dim)]
