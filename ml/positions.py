"""Board representation shared by data generation, training, and inference."""

import numpy as np


def encode(board, player):
    board = np.asarray(board)
    if board.shape != (6, 7) or player not in (1, 2) or not np.isin(board, [0, 1, 2]).all():
        raise ValueError("expected a 6x7 board with cells 0/1/2 and player 1/2")
    # Player-relative inputs let one network play either color.
    return np.stack((board == player, board == 3 - player)).astype(np.float32)


def legal_mask(board):
    return np.asarray(board)[5] == 0


def position_key(board, player):
    """Group a position and its mirror to prevent train/evaluation overlap."""
    planes = encode(board, player).astype(np.uint8)
    return min(planes.tobytes(), planes[:, :, ::-1].tobytes())
