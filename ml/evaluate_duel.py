"""Play two PyTorch checkpoints against each other with balanced sides."""

import argparse
import json
import random

import board as B
from ml.player import TorchPlayer


def duel(first, second, games=100, seed=2027):
    if games < 2:
        raise ValueError("games must be >= 2")
    rng = random.Random(seed)
    result = {"first_wins": 0, "draws": 0, "second_wins": 0}
    for game_id in range(games):
        board, turn = B.empty_board(), 1
        first_side = 1 + game_id % 2
        for _ in range(rng.randrange(7)):
            col = rng.choice([c for c in range(7) if B.landing_row(board, c) is not None])
            board[B.landing_row(board, col)][col] = turn
            turn = 3 - turn
        while True:
            player = first if turn == first_side else second
            col, _ = player(board, turn)
            row = B.landing_row(board, col)
            if row is None:
                raise ValueError(f"checkpoint returned illegal column {col}")
            board[row][col] = turn
            if B.check_win(board, turn):
                result["first_wins" if turn == first_side else "second_wins"] += 1
                break
            if B.is_full(board):
                result["draws"] += 1
                break
            turn = 3 - turn
    return {"games": games, "seed": seed, "alternating_sides": True,
            "random_opening_drops": "0..6", **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="checkpoint being tested")
    parser.add_argument("--second", required=True, help="comparison checkpoint")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--search-simulations", type=int, default=64)
    parser.add_argument("--tactical-guard", action="store_true")
    args = parser.parse_args()
    options = {"search_simulations": args.search_simulations,
               "tactical_guard": args.tactical_guard}
    first, second = TorchPlayer(args.first, **options), TorchPlayer(args.second, **options)
    print(json.dumps({"first": args.first, "second": args.second, **options,
                      **duel(first, second, args.games, args.seed)}, indent=2))


if __name__ == "__main__":
    main()
