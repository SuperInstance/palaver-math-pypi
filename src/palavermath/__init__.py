"""Palaver Math — Consensus mathematics from West African dialogue traditions."""

from palavermath.consensus import (
    compute_center,
    consensus_distance,
    influence_weighted_center,
)
from palavermath.session import PalaverSession
from palavermath.coalition import (
    find_coalitions,
    coalition_strength,
    merge_adjacent,
)
from palavermath.convergence import (
    convergence_rate,
    predict_consensus,
    is_converging,
)
from palavermath.dialogue import DialogueNode, build_dialogue_tree, find_consensus_path

__all__ = [
    "compute_center",
    "consensus_distance",
    "influence_weighted_center",
    "PalaverSession",
    "find_coalitions",
    "coalition_strength",
    "merge_adjacent",
    "convergence_rate",
    "predict_consensus",
    "is_converging",
    "DialogueNode",
    "build_dialogue_tree",
    "find_consensus_path",
]
