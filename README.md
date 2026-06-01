# palaver-math

Consensus mathematics from West African dialogue traditions.

## Installation

```bash
pip install palaver-math
```

## Usage

```python
from palavermath import PalaverSession, compute_center, find_coalitions

# Geometric consensus
center = compute_center([[0, 0], [2, 4], [4, 0]])
# [2.0, 1.333...]

# Iterative palaver session
session = PalaverSession(threshold=0.01)
session.add_participant([0, 0], name="Elder A", influence=2.0)
session.add_participant([10, 10], name="Elder B")
result = session.compute_consensus()
# {'position': [...], 'confidence': 0.99, 'rounds': 42}

# Coalition detection
coalitions = find_coalitions([[0,0], [0.5,0.5], [50,50]], threshold=1.0)
# [[0, 1], [2]]
```

## Modules

- **consensus** — `compute_center`, `consensus_distance`, `influence_weighted_center`
- **session** — `PalaverSession` with iterative convergence
- **coalition** — `find_coalitions`, `coalition_strength`, `merge_adjacent`
- **convergence** — `convergence_rate`, `predict_consensus`, `is_converging`
- **dialogue** — `DialogueNode`, `build_dialogue_tree`, `find_consensus_path`

## License

MIT
