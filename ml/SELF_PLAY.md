# Learn self-play reinforcement learning

This branch starts from the supervised PyTorch player and improves it through
simulated games. No camera, images, or arm are involved. The selected RL weights
are `artifacts/player/best.pt`; the earlier supervised weights are preserved at
`artifacts/player/supervised.pt`.

## What the training loop does

Read `train_selfplay.py` in this order: `one_game`, `ppo_update`, then `train`.
For each learner move, the script records a player-relative board tensor, the
legal columns, the sampled action, its old log probability, and the network's
value estimate. At the end of the game, it gives those moves +1 for a win, 0
for a draw, or -1 for a loss. The learner faces frozen supervised and previous
network versions, random play, and depth-4 minimax as an **opponent**. Minimax
does not supply move labels or run in the deployed PyTorch player.

PPO compares `new_probability / old_probability`. Its clipped policy loss
limits how much a batch changes the policy. The value head learns the terminal
result, while an entropy term keeps some exploration. Legal-move masking is
applied when moves are sampled and when PPO recomputes probabilities. The
opponent networks play greedily so they expose weak moves; the learner samples
its moves to explore. This is outcome-based reinforcement learning, not another
round of supervised minimax imitation.

Reproduce the training experiment from this worktree:

```bash
.venv/bin/python -m ml.train_selfplay --start-checkpoint artifacts/player/supervised.pt --iterations 20 --games-per-iteration 512 --ppo-epochs 3 --opponent-depth 4 --seed 2031 --output artifacts/my-selfplay
```

Use a new output directory for each run. The script writes `last.pt` and a
`history.json`. The win rate in that history is against a changing mixture of
opponents and is **not** a final strength test.

## Why search is in the player

`mcts.py` runs 64 neural-guided Monte Carlo tree search simulations per move
in the recommended configuration. It uses policy probabilities to choose
branches, the learned value to evaluate new positions, and the game rules for
terminal wins and draws. It never calls the old minimax engine. Search lets the
network examine consequences before moving. `player_tactical_guard: true`
also takes an immediate win or blocks a single immediate threat.

Set `player_search_simulations: 0` to measure the raw RL policy. On the tested
midgame board, 64 simulations took about 12 ms per move on this machine's CPU
after loading. Camera capture, vision debounce, and the arm were not timed.

Evaluate your new checkpoint with the same settings as the included one:

```bash
.venv/bin/python -m ml.evaluate_duel --first artifacts/my-selfplay/last.pt --second artifacts/player/supervised.pt --games 100 --seed 2027 --search-simulations 64 --tactical-guard
.venv/bin/python -m ml.evaluate_player --checkpoint artifacts/my-selfplay/last.pt --data data/games-v2.npz --opponent minimax --depth 6 --games 40 --seed 123 --search-simulations 64 --tactical-guard
.venv/bin/python -m ml.play --checkpoint artifacts/player/best.pt --search-simulations 64 --tactical-guard
```

The original simulated-position dataset is local and ignored by Git. To run
`evaluate_player` after a fresh clone, regenerate it using the command in
`ml/README.md`, or omit that command and use `evaluate_duel`, which needs only
the two included checkpoints. Use new opening seeds for a final comparison;
the published experiment used seeds 2027 and 4040 for head-to-head games.
See `ml/RESULTS.md` for the measured results and limitations.
