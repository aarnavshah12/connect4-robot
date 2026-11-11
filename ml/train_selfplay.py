"""Fine-tune the Connect 4 policy/value network with self-play PPO.

The learner samples legal moves against a mixture of frozen PyTorch policies,
random play, and a shallow minimax opponent. Only actual win/draw/loss outcomes
provide training targets. No camera, arm, Roboflow data, or minimax labels are
needed for this training stage.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

import board as B
import engine
from ml.player import PolicyValueNet
from ml.positions import encode
from ml.utils import load_checkpoint, setup


def legal_columns(board):
    return [c for c in range(7) if B.landing_row(board, c) is not None]


def policy_move(model, board, side, rng, temperature=1.0, sample=True):
    """Return a legal action, its log probability, and the predicted value."""
    position = torch.from_numpy(encode(board, side)).unsqueeze(0)
    mask = torch.tensor([B.landing_row(board, c) is not None for c in range(7)])
    with torch.inference_mode():
        logits, value = model(position)
        logits = (logits[0] / temperature).masked_fill(~mask, -1e9)
        distribution = Categorical(logits=logits)
        if sample:
            probabilities = distribution.probs.numpy()
            action = rng.choices(range(7), weights=probabilities, k=1)[0]
        else:
            action = int(logits.argmax())
        log_probability = float(distribution.log_prob(torch.tensor(action)))
    return action, log_probability, float(value.item())


def one_game(learner, opponent, rng, opening_drops=6, opponent_depth=4):
    """Collect only the learner's actions and assign its terminal result."""
    board, side = B.empty_board(), 1
    learner_side = rng.choice((1, 2))
    for _ in range(rng.randrange(opening_drops + 1)):
        col = rng.choice(legal_columns(board))
        board[B.landing_row(board, col)][col] = side
        side = 3 - side
    trajectory = []
    while True:
        if side == learner_side:
            state = encode(board, side)
            mask = np.asarray([B.landing_row(board, c) is not None for c in range(7)])
            col, log_probability, value = policy_move(learner, board, side, rng)
            trajectory.append((state, mask, col, log_probability, value))
        elif opponent == "random":
            col = rng.choice(legal_columns(board))
        elif opponent == "minimax":
            old_depth, engine.DEPTH = engine.DEPTH, opponent_depth
            try:
                col, _ = engine.best_move(board, side)
            finally:
                engine.DEPTH = old_depth
        else:
            # Greedy frozen opponents expose mistakes that sampled opponents
            # often forgive. The learner still samples for on-policy PPO.
            col, _, _ = policy_move(opponent, board, side, rng, sample=False)
        board[B.landing_row(board, col)][col] = side
        if B.check_win(board, side):
            outcome = 1.0 if side == learner_side else -1.0
            break
        if B.is_full(board):
            outcome = 0.0
            break
        side = 3 - side
    return trajectory, outcome


def ppo_update(model, optimizer, experiences, rng, epochs=3, batch_size=256,
               clip=0.2, value_weight=0.5, entropy_weight=0.01):
    """Improve sampled actions without letting a single update move too far."""
    states, masks, actions, old_logp, old_values, returns = zip(*experiences)
    states = torch.from_numpy(np.stack(states))
    masks = torch.from_numpy(np.stack(masks))
    actions = torch.tensor(actions, dtype=torch.long)
    old_logp = torch.tensor(old_logp, dtype=torch.float32)
    advantages = torch.tensor(returns, dtype=torch.float32) - torch.tensor(old_values)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-4)
    returns = torch.tensor(returns, dtype=torch.float32)
    losses = []
    model.train()
    for _ in range(epochs):
        order = list(range(len(actions)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            logits, values = model(states[indices])
            distribution = Categorical(logits=logits.masked_fill(~masks[indices], -1e9))
            ratio = (distribution.log_prob(actions[indices]) - old_logp[indices]).exp()
            advantage = advantages[indices]
            surrogate = torch.minimum(ratio * advantage,
                                      ratio.clamp(1 - clip, 1 + clip) * advantage)
            policy_loss = -surrogate.mean()
            value_loss = (values - returns[indices]).square().mean()
            loss = policy_loss + value_weight * value_loss - entropy_weight * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
    model.eval()
    return float(np.mean(losses))


def train(start_checkpoint, output, iterations=10, games_per_iteration=384,
          ppo_epochs=3, batch_size=256, lr=2e-4, seed=2026,
          opponent_depth=4):
    output = Path(output)
    if (output / "last.pt").exists():
        raise ValueError(f"checkpoint exists; choose a new output directory: {output}")
    if min(iterations, games_per_iteration, ppo_epochs, batch_size) < 1 or lr <= 0 or opponent_depth < 2:
        raise ValueError("invalid training settings")
    setup(seed, "cpu")
    rng = random.Random(seed)
    starting = load_checkpoint(start_checkpoint, "player")
    if starting.get("encoding") != "current-opponent-bottom-first":
        raise ValueError("unsupported player checkpoint encoding")
    model, baseline, previous = PolicyValueNet(), PolicyValueNet(), PolicyValueNet()
    for network in (model, baseline, previous):
        network.load_state_dict(starting["state_dict"])
        network.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for iteration in range(1, iterations + 1):
        experiences = []
        outcomes = []
        opponents = {"baseline": 0, "previous": 0, "random": 0, "minimax": 0}
        for _ in range(games_per_iteration):
            kind = rng.choices(tuple(opponents), weights=(3, 3, 2, 2), k=1)[0]
            opponents[kind] += 1
            opponent = {"baseline": baseline, "previous": previous,
                        "random": "random", "minimax": "minimax"}[kind]
            trajectory, outcome = one_game(model, opponent, rng, opponent_depth=opponent_depth)
            experiences.extend((*record, outcome) for record in trajectory)
            outcomes.append(outcome)
        loss = ppo_update(model, optimizer, experiences, rng, epochs=ppo_epochs,
                          batch_size=batch_size)
        previous.load_state_dict(model.state_dict())
        previous.eval()
        row = {"iteration": iteration, "games": games_per_iteration,
               "positions": len(experiences), "opponents": opponents,
               "wins": outcomes.count(1.0), "draws": outcomes.count(0.0),
               "losses": outcomes.count(-1.0), "mean_ppo_loss": loss}
        history.append(row)
        torch.save({"format_version": 1, "kind": "player", "state_dict": model.state_dict(),
                    "encoding": "current-opponent-bottom-first", "training": "self-play PPO",
                    "iteration": iteration, "start_checkpoint": str(start_checkpoint),
                    "seed": seed, "opponent_counts": opponents,
                    "training_config": {"iterations": iterations,
                                        "games_per_iteration": games_per_iteration,
                                        "ppo_epochs": ppo_epochs, "batch_size": batch_size,
                                        "lr": lr, "opponent_depth": opponent_depth,
                                        "opponent_weights": [3, 3, 2, 2]}},
                   output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"iteration={iteration}/{iterations} positions={len(experiences)} "
              f"W/D/L={row['wins']}/{row['draws']}/{row['losses']} loss={loss:.3f}",
              flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", default="artifacts/player/supervised.pt")
    parser.add_argument("--output", default="artifacts/player-selfplay")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=384)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--opponent-depth", type=int, default=4)
    args = parser.parse_args()
    train(args.start_checkpoint, args.output, args.iterations, args.games_per_iteration,
          args.ppo_epochs, args.batch_size, args.lr, args.seed, args.opponent_depth)


if __name__ == "__main__":
    main()
