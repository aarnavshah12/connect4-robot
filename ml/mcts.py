"""Small neural-guided Monte Carlo tree search for Connect 4.

The policy supplies move priors and the value head estimates newly explored
positions. Terminal wins/draws use the actual game rules. This does not call
the old minimax engine.
"""

import math

import numpy as np
import torch

import board as B
from ml.positions import encode


class Node:
    def __init__(self, board, side, terminal=None):
        self.board, self.side, self.terminal = board, side, terminal
        self.prior = None
        self.legal = None
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0


def expand(node, model, device):
    """Get priors and an estimate from the current side's perspective."""
    with torch.inference_mode():
        x = torch.from_numpy(encode(node.board, node.side)).unsqueeze(0).to(device)
        logits, value = model(x)
        legal = [c for c in range(7) if B.landing_row(node.board, c) is not None]
        allowed = torch.tensor([c in legal for c in range(7)], device=device)
        node.legal = legal
        node.prior = logits[0].masked_fill(~allowed, -torch.inf).softmax(dim=0).cpu().numpy()
    return float(value.item())


def select(node, exploration):
    total = math.sqrt(max(node.visits, 1))
    def score(col):
        child = node.children.get(col)
        visits = 0 if child is None else child.visits
        q = 0.0 if not visits else -child.value_sum / visits
        return q + exploration * float(node.prior[col]) * total / (1 + visits)
    return max(node.legal, key=score)


def descend(node, col):
    row = B.landing_row(node.board, col)
    board = [line[:] for line in node.board]
    board[row][col] = node.side
    terminal = -1.0 if B.check_win(board, node.side) else 0.0 if B.is_full(board) else None
    child = Node(board, 3 - node.side, terminal)
    node.children[col] = child
    return child


def search(model, board, side, device="cpu", simulations=64, exploration=1.5):
    if simulations < 1:
        raise ValueError("simulations must be positive")
    if B.is_full(board):
        raise ValueError("cannot search a full board")
    root = Node(board, side)
    expand(root, model, device)
    for _ in range(simulations):
        node, path = root, [root]
        while node.terminal is None and node.prior is not None:
            col = select(node, exploration)
            node = node.children.get(col) or descend(node, col)
            path.append(node)
        value = node.terminal if node.terminal is not None else expand(node, model, device)
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value
    counts = np.array([root.children[c].visits if c in root.children else 0 for c in range(7)])
    return int(counts.argmax()), counts / counts.sum(), root.value_sum / root.visits
