"""Create player training examples using legal simulated games and minimax.

The policy label is the teacher's preferred move. The value label is the
actual outcome of that simulated game, from the recorded player's viewpoint.
Exploration diversifies the positions, so outcomes describe this behavior
mixture rather than perfect play.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

import board as B
import engine
from ml.positions import position_key


def generate(output, games=150, teacher_depth=6, exploration=0.35, seed=42):
    output = Path(output)
    if output.exists():
        raise ValueError(f"dataset already exists; choose a new output path: {output}")
    if games < 10 or teacher_depth < 2 or not 0 <= exploration <= 1:
        raise ValueError("need games >= 10, teacher_depth >= 2, and exploration in [0, 1]")
    rng = random.Random(seed)
    old_depth, engine.DEPTH = engine.DEPTH, teacher_depth
    completed = []
    try:
        for game_id in range(games):
            board, player, positions = B.empty_board(), 1, []
            winner = 0
            while True:
                move, _ = engine.best_move(board, player)
                positions.append(([row[:] for row in board], player, move))
                allowed = [c for c in range(7) if B.landing_row(board, c) is not None]
                action = rng.choice(allowed) if rng.random() < exploration else move
                board[B.landing_row(board, action)][action] = player
                if B.check_win(board, player):
                    winner = player
                    break
                if B.is_full(board):
                    break
                player = 3 - player
            records = [(board, player, move, 0 if not winner else 1 if player == winner else -1,
                        game_id) for board, player, move in positions]
            completed.append(records)
            if (game_id + 1) % 25 == 0:
                print(f"generated {game_id + 1}/{games} games", flush=True)
    finally:
        engine.DEPTH = old_depth
    rng.shuffle(completed)
    n_train, n_val = int(0.8 * games), max(1, int(0.1 * games))
    groups = {"train": completed[:n_train], "val": completed[n_train:n_train + n_val],
              "test": completed[n_train + n_val:]}
    arrays, prior_keys, counts = {}, set(), {}
    for split, grouped_games in groups.items():
        records = [record for game in grouped_games for record in game
                   if position_key(record[0], record[1]) not in prior_keys]
        if not records:
            raise ValueError(f"no unique {split} positions; generate more games")
        # A mirror of a training position also stays out of val/test.
        prior_keys.update(position_key(record[0], record[1]) for record in records)
        boards, players, moves, values, game_ids = zip(*records)
        arrays.update({f"{split}_boards": np.asarray(boards, dtype=np.int8),
                       f"{split}_players": np.asarray(players, dtype=np.int8),
                       f"{split}_moves": np.asarray(moves, dtype=np.int64),
                       f"{split}_values": np.asarray(values, dtype=np.float32),
                       f"{split}_game_ids": np.asarray(game_ids, dtype=np.int32)})
        counts[split] = len(records)
    metadata = {"games": games, "teacher_depth": teacher_depth, "exploration": exploration,
                "seed": seed, "counts": counts, "value_target": "mixed-play terminal outcome"}
    arrays["metadata"] = np.array(json.dumps(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/games.npz")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--teacher-depth", type=int, default=6)
    parser.add_argument("--exploration", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.games, args.teacher_depth, args.exploration, args.seed)


if __name__ == "__main__":
    main()
