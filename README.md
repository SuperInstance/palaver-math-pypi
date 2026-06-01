# palaver-math

> Dialogue and consensus mathematics — West African palaver tradition as iterative convergence, coalition formation, and dialogue trees.

## What This Does

`palaver-math` models group decision-making as iterative consensus convergence, inspired by the West African palaver (community dialogue under a baobab tree). It provides sessions where participants hold position vectors, vote on proposals, and gradually converge toward consensus. It detects convergence rates, predicts final consensus positions, finds coalitions (clusters of nearby positions), and builds dialogue trees. Use it for multi-agent consensus, deliberative democracy simulations, collaborative filtering, or any scenario where distributed actors need to agree.

## The Cultural Root

In many West African traditions, a palaver is a community gathering under a tree where disputes are resolved through dialogue until consensus is reached. There is no vote — discussion continues until everyone agrees. The mathematical insight: **consensus is an iterative convergence process** where each participant's position drifts toward the group center weighted by influence. This is identical to distributed averaging algorithms in computer science and opinion dynamics in statistical physics.

## Install

```bash
pip install palaver-math
```

## Quick Start

```python
from palavermath.session import PalaverSession
from palavermath.consensus import compute_center, consensus_distance, influence_weighted_center
from palavermath.convergence import convergence_rate, is_converging
from palavermath.coalition import find_coalitions, coalition_strength
from palavermath.dialogue import build_dialogue_tree, find_consensus_path

# Start a consensus session
session = PalaverSession(threshold=0.5, step_size=0.1)

# Add participants with positions (e.g., 2D policy space)
p1 = session.add_participant(position=[1.0, 0.0])   # Conservative
p2 = session.add_participant(position=[0.0, 1.0])   # Progressive
p3 = session.add_participant(position=[0.5, 0.5])   # Moderate

# Add proposals and vote
prop = session.add_proposal(position=[0.4, 0.4])
session.vote(participant_idx=0, proposal_idx=0)  # Accepts → shifts position

# Run the full consensus loop
result = session.compute_consensus()
print(f"Consensus reached: {result['consensus_reached']}")
print(f"Center: {result['center']}")
print(f"Distance: {result['distance']}")
print(f"Rounds: {result['rounds']}")

# Check convergence
history = session.history()
rate = convergence_rate(history)
print(f"Converging: {is_converging(history)}")

# Find coalitions
participants = session.participants()
coalitions = find_coalitions(participants, threshold=0.8)
for c in coalitions:
    print(f"Coalition of {len(c)} participants, strength: {coalition_strength(c):.2f}")

# Dialogue tree
statements = [
    {"statement": "We should build the road", "parent_id": None},
    {"statement": "But what about the river?", "parent_id": 0},
    {"statement": "We can build a bridge", "parent_id": 1},
]
tree = build_dialogue_tree(statements)
path = find_consensus_path(tree)
```

## API Reference

### `session` module

#### `PalaverSession`
```python
class PalaverSession:
    def __init__(self, threshold=1.0, step_size=0.1, max_rounds=1000)
    def add_participant(self, position) → int  # Returns participant index
    def add_proposal(self, position) → int     # Returns proposal index
    def vote(self, participant_idx, proposal_idx) → None
    def compute_consensus(self) → dict  # {consensus_reached, center, distance, rounds, positions}
    def participants(self) → list[dict]
    def history(self) → list  # history[round][participant][dim]
    def round(self) → int
```

### `consensus` module

#### `compute_center(participants) → list[float]`
Geometric centroid of participant positions.

#### `consensus_distance(participants) → float`
Average Euclidean distance from each participant to the center. Measures spread: 0 = perfect agreement.

#### `influence_weighted_center(participants, weights=None) → list[float]`
Weighted centroid where participants may carry different influence.

### `convergence` module

#### `convergence_rate(session_history) → float`
Average per-step change in consensus distance. Negative = converging.

#### `predict_consensus(session, max_steps=100) → list[float]`
Predict where the session will converge by running the convergence loop.

#### `is_converging(session_history) → bool`
True if the last few rounds show decreasing consensus distance.

### `coalition` module

#### `find_coalitions(participants, threshold) → list[list]`
Groups where every pair is within `threshold` distance (single-linkage clustering).

#### `coalition_strength(coalition) → float`
0–1 cohesion score. 1.0 = everyone shares the same position.

#### `merge_adjacent(coalitions, threshold) → list[list]`
Merge coalitions whose centroids are within threshold.

### `dialogue` module

#### `DialogueNode`
```python
@dataclass
class DialogueNode:
    statement: str
    responses: list[DialogueNode]
    depth: int
```

#### `build_dialogue_tree(statements) → DialogueNode`
Build a tree from flat statement records with `statement`, `parent_id`, `responses`.

#### `find_consensus_path(tree) → list[DialogueNode]`
Shortest root-to-leaf path (BFS). Consensus = reaching a leaf.

## How It Works

**Consensus Session:** Each participant holds a position vector. When a participant accepts a proposal, their position shifts toward the proposal by `step_size`. The session runs rounds until consensus distance drops below `threshold` or `max_rounds` is hit.

**Convergence:** Tracks the average distance from all participants to the geometric center over time. If this distance is decreasing, the group is converging.

**Coalitions:** Uses single-linkage clustering — two participants are in the same coalition if any path of within-threshold hops connects them.

**Dialogue Trees:** Statements form a tree. Consensus path = shortest root-to-leaf path, representing the most efficient route to a conclusion.

## The Math

**Consensus Distance:** D = (1/n) Σᵢ ||pᵢ − c|| where c = (1/n) Σᵢ pᵢ is the centroid.

**Position Update:** pᵢ(t+1) = pᵢ(t) + α · (q − pᵢ(t)) where q is the accepted proposal and α is the step size. This is gradient descent toward the proposal.

**Convergence Rate:** r = (1/T) Σₜ (D(t+1) − D(t)). Negative r means converging.

**Coalition Strength:** s = 1 − (1/(k·d_max)) Σᵢ<ⱼ ||pᵢ − pⱼ|| normalized to [0, 1].

## License

MIT
