"""Evaluate held-out prediction agreement, tactics, and actual games.

No tactical overrides are applied: these results measure the neural player.
"""

import argparse
import json
import random

import torch
from torch.utils.data import DataLoader

import board as B
import engine
from ml.player import TorchPlayer
from ml.train_player import PositionDataset, metrics


def winning_moves(board, player):
    moves = []
    for col in range(7):
        row = B.landing_row(board, col)
        if row is None:
            continue
        trial = [r[:] for r in board]
        trial[row][col] = player
        if B.check_win(trial, player):
            moves.append(col)
    return moves


def evaluate(player, games=40, opponent="random", depth=6, seed=123):
    if games < 2 or depth < 2:
        raise ValueError("games and depth must be >= 2")
    rng = random.Random(seed)
    results = {"wins": 0, "draws": 0, "losses": 0,
               "immediate_wins_available": 0, "immediate_wins_taken": 0,
               "single_threat_blocks_available": 0, "single_threat_blocks_played": 0}
    old_depth, engine.DEPTH = engine.DEPTH, depth
    try:
        for game_id in range(games):
            board, turn, neural_side = B.empty_board(), 1, 1 + game_id % 2
            # At most six opening drops: no position can have a winner yet.
            for _ in range(rng.randrange(7)):
                col = rng.randrange(7)
                board[B.landing_row(board, col)][col] = turn
                turn = 3 - turn
            while True:
                allowed = [c for c in range(7) if B.landing_row(board, c) is not None]
                if turn == neural_side:
                    col, _ = player(board, turn)
                    wins, threats = winning_moves(board, turn), winning_moves(board, 3 - turn)
                    if wins:
                        results["immediate_wins_available"] += 1
                        results["immediate_wins_taken"] += int(col in wins)
                    elif len(threats) == 1:
                        results["single_threat_blocks_available"] += 1
                        results["single_threat_blocks_played"] += int(col == threats[0])
                elif opponent == "minimax":
                    col, _ = engine.best_move(board, turn)
                else:
                    col = rng.choice(allowed)
                if col not in allowed:
                    raise ValueError(f"player returned illegal column {col}")
                board[B.landing_row(board, col)][col] = turn
                if B.check_win(board, turn):
                    results["wins" if turn == neural_side else "losses"] += 1
                    break
                if B.is_full(board):
                    results["draws"] += 1
                    break
                turn = 3 - turn
    finally:
        engine.DEPTH = old_depth
    return {"games": games, "opponent": opponent, "opponent_depth": depth if opponent == "minimax" else None,
            "seed": seed, "both_sides": True, "random_opening_drops": "0..6", **results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/player/best.pt")
    parser.add_argument("--data", default="data/games.npz")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--opponent", choices=["random", "minimax"], default="random")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output")
    args = parser.parse_args()
    player = TorchPlayer(args.checkpoint, device=args.device)
    result = {"held_out": metrics(player.model, DataLoader(PositionDataset(args.data, "test"), batch_size=128),
                                  player.device),
              "matches": evaluate(player, args.games, args.opponent, args.depth, args.seed)}
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(rendered)


if __name__ == "__main__":
    main()
