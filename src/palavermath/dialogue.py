"""Dialogue tree — model deliberation as a branching conversation.

A :class:`DialogueNode` is a single statement in the tree, linked to its
responses (children).  :func:`build_dialogue_tree` constructs a tree from a
flat list of annotated statements, and :func:`find_consensus_path` finds the
shortest root-to-leaf path using BFS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class DialogueNode:
    """A single node in a dialogue tree.

    Attributes
    ----------
    statement:
        The text (or data) of the statement made.
    speaker:
        Identifier for the speaker (name, index, etc.).
    responses:
        Child nodes — responses to this statement.
    """

    statement: Any
    speaker: Any = None
    responses: list["DialogueNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        """Return ``True`` if this node has no responses."""
        return len(self.responses) == 0

    def depth(self) -> int:
        """Return the depth of the subtree rooted here (root = 0 for a leaf)."""
        if not self.responses:
            return 0
        return 1 + max(r.depth() for r in self.responses)


def build_dialogue_tree(
    statements: Sequence[dict[str, Any]],
) -> DialogueNode:
    """Build a dialogue tree from a flat list of statement records.

    Each statement dict should contain:

    - ``"statement"`` — the content.
    - ``"speaker"`` — who said it.
    - ``"parent"`` — the index of the parent statement, or ``None``/``-1``
      for the root.

    Parameters
    ----------
    statements:
        Ordered list of statement records.

    Returns
    -------
    DialogueNode
        The root of the tree.

    Raises
    ------
    ValueError
        If no root statement is found.
    """
    if not statements:
        raise ValueError("At least one statement is required")

    nodes: list[DialogueNode] = []
    root: DialogueNode | None = None

    for i, rec in enumerate(statements):
        node = DialogueNode(
            statement=rec.get("statement", ""),
            speaker=rec.get("speaker"),
        )
        nodes.append(node)
        parent_idx = rec.get("parent")
        if parent_idx is None or parent_idx < 0:
            if root is not None:
                raise ValueError("Multiple root statements found")
            root = node
        else:
            nodes[parent_idx].responses.append(node)

    if root is None:
        raise ValueError("No root statement found")

    return root


def find_consensus_path(tree: DialogueNode) -> list[DialogueNode]:
    """Find the shortest root-to-leaf path via BFS.

    In palaver tradition, consensus is reached when the dialogue arrives at a
    terminal statement that everyone can accept.  The *shortest* such path
    represents the most efficient route to agreement.

    Parameters
    ----------
    tree:
        The root of the dialogue tree.

    Returns
    -------
    list[DialogueNode]
        Nodes along the shortest path from root to a leaf.
    """
    if tree.is_leaf():
        return [tree]

    # BFS — track the path as we go.
    from collections import deque

    queue: deque[list[DialogueNode]] = deque()
    queue.append([tree])

    while queue:
        path = queue.popleft()
        node = path[-1]
        if node.is_leaf():
            return path
        for child in node.responses:
            queue.append(path + [child])

    # Fallback (should not happen with a well-formed tree).
    return [tree]
