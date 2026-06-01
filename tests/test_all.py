"""Comprehensive tests for palavermath."""

import math
import pytest

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


# ===================================================================
# consensus.py
# ===================================================================

class TestComputeCenter:
    def test_simple_2d(self):
        assert compute_center([[0, 0], [2, 2]]) == [1.0, 1.0]

    def test_simple_1d(self):
        assert compute_center([[0], [4]]) == [2.0]

    def test_single_participant(self):
        assert compute_center([[3, 5, 7]]) == [3.0, 5.0, 7.0]

    def test_three_participants_2d(self):
        result = compute_center([[0, 0], [3, 0], [0, 3]])
        assert result == pytest.approx([1.0, 1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_center([])

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError):
            compute_center([[]])

    def test_mismatched_dims_raises(self):
        with pytest.raises(ValueError):
            compute_center([[0, 0], [1]])

    def test_negative_coords(self):
        assert compute_center([[-2, -2], [2, 2]]) == [0.0, 0.0]


class TestConsensusDistance:
    def test_zero_distance(self):
        assert consensus_distance([[1, 1], [1, 1]]) == pytest.approx(0.0)

    def test_known_distance(self):
        # Two points at (0,0) and (2,0) -> center (1,0), distances 1 each -> avg 1
        assert consensus_distance([[0, 0], [2, 0]]) == pytest.approx(1.0)

    def test_single_participant(self):
        assert consensus_distance([[5, 5]]) == pytest.approx(0.0)

    def test_spread_reduces_with_more_participants_at_center(self):
        d1 = consensus_distance([[0, 0], [10, 10]])
        d2 = consensus_distance([[0, 0], [5, 5], [10, 10]])
        assert d2 < d1


class TestInfluenceWeightedCenter:
    def test_equal_weights_matches_center(self):
        pts = [[0, 0], [4, 4]]
        assert influence_weighted_center(pts) == compute_center(pts)

    def test_single_weighted(self):
        result = influence_weighted_center([[0, 0], [4, 0]], weights=[3, 1])
        assert result == pytest.approx([1.0, 0.0])

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            influence_weighted_center([[1, 1]], weights=[0])

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            influence_weighted_center([[1, 1]], weights=[1, 2])

    def test_default_weights(self):
        result = influence_weighted_center([[2, 2], [4, 4]])
        assert result == [3.0, 3.0]


# ===================================================================
# session.py
# ===================================================================

class TestPalaverSession:
    def test_basic_convergence(self):
        s = PalaverSession(threshold=0.1, max_rounds=50)
        s.add_participant([0, 0])
        s.add_participant([10, 10])
        result = s.compute_consensus()
        assert result["confidence"] > 0.0
        assert result["rounds"] > 0

    def test_already_consensus(self):
        s = PalaverSession(threshold=0.01)
        s.add_participant([5, 5])
        s.add_participant([5, 5])
        result = s.compute_consensus()
        assert result["position"] == pytest.approx([5.0, 5.0])
        assert result["rounds"] <= 1

    def test_single_participant(self):
        s = PalaverSession()
        s.add_participant([3, 3])
        result = s.compute_consensus()
        assert result["position"] == [3.0, 3.0]
        assert result["confidence"] == pytest.approx(1.0)

    def test_empty_session(self):
        s = PalaverSession()
        result = s.compute_consensus()
        assert result["position"] == []
        assert result["rounds"] == 0

    def test_add_participant_returns_index(self):
        s = PalaverSession()
        assert s.add_participant([1, 1]) == 0
        assert s.add_participant([2, 2]) == 1

    def test_add_proposal_returns_index(self):
        s = PalaverSession()
        assert s.add_proposal([5, 5]) == 0

    def test_vote_moves_position(self):
        s = PalaverSession()
        s.add_participant([0, 0])
        s.add_proposal([10, 10])
        s.vote(0, 0)
        pos = s.participants[0]["position"]
        assert pos[0] > 0  # Moved toward proposal

    def test_vote_out_of_range(self):
        s = PalaverSession()
        s.add_participant([0, 0])
        with pytest.raises(IndexError):
            s.vote(5, 0)
        with pytest.raises(IndexError):
            s.vote(0, 5)

    def test_history_tracks_positions(self):
        s = PalaverSession(threshold=0.5, max_rounds=5)
        s.add_participant([0, 0])
        s.add_participant([10, 10])
        s.compute_consensus()
        assert len(s.history) > 0
        assert len(s.history[0]) == 2

    def test_max_rounds_respected(self):
        s = PalaverSession(threshold=1e-15, max_rounds=5)
        s.add_participant([0, 0])
        s.add_participant([100, 100])
        result = s.compute_consensus()
        assert result["rounds"] <= 5

    def test_named_participant(self):
        s = PalaverSession()
        s.add_participant([1, 1], name="Alice")
        assert s.participants[0]["name"] == "Alice"

    def test_influence_affects_convergence(self):
        # High-influence participant moves less.
        s = PalaverSession(threshold=0.01, max_rounds=100)
        s.add_participant([0, 0], influence=100)  # barely moves
        s.add_participant([10, 10], influence=1)
        result = s.compute_consensus()
        # Center should be closer to [0,0] than [5,5]
        assert result["position"][0] < 5.0


# ===================================================================
# coalition.py
# ===================================================================

class TestFindCoalitions:
    def test_all_close(self):
        coalitions = find_coalitions([[0, 0], [0.5, 0.5]], threshold=1.0)
        assert len(coalitions) == 1

    def test_all_far(self):
        coalitions = find_coalitions([[0, 0], [100, 100]], threshold=1.0)
        assert len(coalitions) == 2

    def test_three_two_close_one_far(self):
        pts = [[0, 0], [0.5, 0.5], [50, 50]]
        coalitions = find_coalitions(pts, threshold=1.0)
        assert len(coalitions) == 2

    def test_empty(self):
        assert find_coalitions([]) == []

    def test_single(self):
        assert find_coalitions([[1, 1]]) == [[0]]

    def test_threshold_zero(self):
        coalitions = find_coalitions([[0, 0], [0, 0]], threshold=0.0)
        # Identical points should be in same coalition
        assert len(coalitions) == 1


class TestCoalitionStrength:
    def test_perfect_agreement(self):
        assert coalition_strength([[1, 1], [1, 1]]) == pytest.approx(1.0)

    def test_single_participant(self):
        assert coalition_strength([[0, 0]]) == 1.0

    def test_spread_reduces_strength(self):
        s1 = coalition_strength([[0, 0], [1, 1]])
        s2 = coalition_strength([[0, 0], [10, 10]])
        assert s1 > s2

    def test_strength_between_zero_and_one(self):
        s = coalition_strength([[0, 0], [5, 5]])
        assert 0.0 < s < 1.0


class TestMergeAdjacent:
    def test_no_merge_needed(self):
        coalitions = [[0], [5], [10]]
        result = merge_adjacent(coalitions, threshold=1)
        assert len(result) == 3

    def test_merge_close(self):
        coalitions = [[0, 1], [5]]
        result = merge_adjacent(coalitions, threshold=2)
        # 0 and 1 are within threshold 2 of each other, 5 is not
        assert len(result) == 2

    def test_empty(self):
        assert merge_adjacent([]) == []

    def test_merge_all(self):
        coalitions = [[0], [1], [2]]
        result = merge_adjacent(coalitions, threshold=2)
        assert len(result) == 1


# ===================================================================
# convergence.py
# ===================================================================

class TestConvergenceRate:
    def test_converging(self):
        history = [
            [[0, 0], [10, 10]],
            [[2, 2], [8, 8]],
            [[4, 4], [6, 6]],
        ]
        rate = convergence_rate(history)
        assert rate < 0  # Negative means converging

    def test_static(self):
        history = [
            [[0, 0], [10, 10]],
            [[0, 0], [10, 10]],
        ]
        rate = convergence_rate(history)
        assert rate == pytest.approx(0.0)

    def test_single_snapshot(self):
        assert convergence_rate([[[1, 1]]]) == 0.0

    def test_empty(self):
        assert convergence_rate([]) == 0.0


class TestPredictConsensus:
    def test_basic_prediction(self):
        s = PalaverSession(threshold=0.01, max_rounds=200)
        s.add_participant([0, 0])
        s.add_participant([10, 10])
        predicted = predict_consensus(s)
        assert len(predicted) == 2
        # Should converge near center
        assert predicted[0] == pytest.approx(5.0, abs=0.5)

    def test_empty_session(self):
        s = PalaverSession()
        assert predict_consensus(s) == []

    def test_does_not_mutate_session(self):
        s = PalaverSession()
        s.add_participant([0, 0])
        pos_before = list(s.participants[0]["position"])
        predict_consensus(s)
        assert s.participants[0]["position"] == pos_before


class TestIsConverging:
    def test_converging(self):
        history = [
            [[0, 0], [10, 10]],
            [[3, 3], [7, 7]],
            [[4.5, 4.5], [5.5, 5.5]],
        ]
        assert is_converging(history) is True

    def test_diverging(self):
        history = [
            [[4, 4], [6, 6]],
            [[0, 0], [10, 10]],
        ]
        assert is_converging(history) is False

    def test_single_snapshot(self):
        assert is_converging([[[1, 1]]]) is True

    def test_equal_snapshots(self):
        history = [
            [[1, 1], [3, 3]],
            [[1, 1], [3, 3]],
        ]
        assert is_converging(history) is True


# ===================================================================
# dialogue.py
# ===================================================================

class TestDialogueNode:
    def test_is_leaf(self):
        node = DialogueNode(statement="hello")
        assert node.is_leaf()

    def test_not_leaf(self):
        node = DialogueNode(
            statement="hello",
            responses=[DialogueNode(statement="world")],
        )
        assert not node.is_leaf()

    def test_depth_single(self):
        assert DialogueNode(statement="x").depth() == 0

    def test_depth_chain(self):
        root = DialogueNode(
            statement="a",
            responses=[DialogueNode(
                statement="b",
                responses=[DialogueNode(statement="c")],
            )],
        )
        assert root.depth() == 2


class TestBuildDialogueTree:
    def test_simple_chain(self):
        stmts = [
            {"statement": "root", "speaker": "A", "parent": None},
            {"statement": "reply", "speaker": "B", "parent": 0},
        ]
        tree = build_dialogue_tree(stmts)
        assert tree.statement == "root"
        assert len(tree.responses) == 1
        assert tree.responses[0].statement == "reply"

    def test_multiple_replies(self):
        stmts = [
            {"statement": "root", "speaker": "A", "parent": None},
            {"statement": "r1", "speaker": "B", "parent": 0},
            {"statement": "r2", "speaker": "C", "parent": 0},
        ]
        tree = build_dialogue_tree(stmts)
        assert len(tree.responses) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_dialogue_tree([])

    def test_no_root_raises(self):
        # Circular parents — no root, causes IndexError when resolving parent
        with pytest.raises((ValueError, IndexError)):
            build_dialogue_tree([
                {"statement": "a", "parent": 1},
                {"statement": "b", "parent": 0},
            ])

    def test_multiple_roots_raises(self):
        with pytest.raises(ValueError, match="Multiple root"):
            build_dialogue_tree([
                {"statement": "a", "parent": None},
                {"statement": "b", "parent": -1},
            ])


class TestFindConsensusPath:
    def test_single_node(self):
        node = DialogueNode(statement="done")
        path = find_consensus_path(node)
        assert len(path) == 1
        assert path[0].statement == "done"

    def test_shortest_path(self):
        tree = DialogueNode(
            statement="root",
            responses=[
                DialogueNode(statement="short"),  # leaf — shortest
                DialogueNode(
                    statement="long",
                    responses=[DialogueNode(statement="end")],
                ),
            ],
        )
        path = find_consensus_path(tree)
        assert len(path) == 2
        assert path[-1].statement == "short"

    def test_deep_tree(self):
        # Linear chain of 5 nodes
        nodes = [DialogueNode(statement=f"n{i}") for i in range(5)]
        for i in range(4):
            nodes[i].responses.append(nodes[i + 1])
        path = find_consensus_path(nodes[0])
        assert len(path) == 5

    def test_branched_tree_picks_shortest(self):
        """With two branches of different lengths, BFS returns the shorter."""
        tree = DialogueNode(
            statement="root",
            responses=[
                DialogueNode(statement="a"),  # leaf (depth 1)
                DialogueNode(
                    statement="b",
                    responses=[
                        DialogueNode(statement="c"),  # leaf (depth 2)
                    ],
                ),
                DialogueNode(
                    statement="d",
                    responses=[
                        DialogueNode(
                            statement="e",
                            responses=[DialogueNode(statement="f")],  # leaf (depth 3)
                        ),
                    ],
                ),
            ],
        )
        path = find_consensus_path(tree)
        assert len(path) == 2
        assert path[-1].statement == "a"
