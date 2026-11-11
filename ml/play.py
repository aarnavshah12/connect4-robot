"""Play the trained neural player in the terminal; no camera or arm."""

import argparse

import board as B
from ml.player import TorchPlayer


def show(board):
    print("\n" + "\n".join(" ".join(".RY"[cell] for cell in row) for row in reversed(board)))
    print("0 1 2 3 4 5 6\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/player/best.pt")
    parser.add_argument("--robot-first", action="store_true")
    parser.add_argument("--tactical-guard", action="store_true")
    parser.add_argument("--search-simulations", type=int, default=0)
    args = parser.parse_args()
    player = TorchPlayer(args.checkpoint, tactical_guard=args.tactical_guard,
                         search_simulations=args.search_simulations)
    board, turn = B.empty_board(), B.ROBOT if args.robot_first else B.HUMAN
    print("You are yellow (Y). The neural player is red (R). Enter q to quit.")
    try:
        while True:
            show(board)
            if turn == B.HUMAN:
                answer = input("Your column: ").strip()
                if answer.lower() == "q":
                    return
                if answer not in "0123456" or len(answer) != 1:
                    print("Enter one column number from 0 to 6.")
                    continue
                col = int(answer)
                if B.landing_row(board, col) is None:
                    print("That column is full.")
                    continue
            else:
                considered = []
                col, value = player(board, B.ROBOT, considered)
                print(f"Neural player: column {col}; learned advantage estimate {value:+.2f}")
                print("Move shares: " + ", ".join(f"{c}: {p:.0%}" for c, p in considered))
            board[B.landing_row(board, col)][col] = turn
            if B.check_win(board, turn):
                show(board)
                print("You win." if turn == B.HUMAN else "Neural player wins.")
                return
            if B.is_full(board):
                show(board)
                print("Draw.")
                return
            turn = 3 - turn
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
