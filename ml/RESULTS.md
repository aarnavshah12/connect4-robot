# PyTorch experiments — 2026-11-10

## Self-play RL branch: `pytorch-self-play`

Vision weights and the five-photo vision results below are unchanged. This
branch adds an outcome-based PPO player trained for **20 iterations × 512
simulated games**, yielding 98,919 learner positions, from the supervised
checkpoint. The learner sampled moves
against frozen greedy supervised and previous policies, random play, and
depth-4 minimax as an opponent. Only final win/draw/loss results trained PPO;
the minimax opponent did not label moves. Training ran on CPU with seed 2031.
The selected RL checkpoint is the final iteration. The old supervised weights
remain at `artifacts/player/supervised.pt`.

At runtime the recommended player uses **64 neural MCTS simulations** per
move plus the optional immediate win/block rule. MCTS uses the PyTorch policy
and value heads and the Connect 4 rules, not the old minimax engine. On one
midgame board, its median move time was **11.7 ms** on this machine's CPU
after warmup (30 repeated calls); live end-to-end timing was not measured.

Paired 40-game matches versus depth-6 minimax alternated player sides and
used the same 0–6-drop random openings for each checkpoint:

| Opening seed | Supervised + 64 search W/D/L | RL + 64 search W/D/L |
|---|---:|---:|
| 123 | 9/9/22 | **14/3/23** |
| 8675309 | 5/9/26 | **14/5/21** |

Direct head-to-head matches with the same search and guard also favored RL:
**49 wins, 17 draws, 34 losses** on seed 2027 and **46 wins, 18 draws, 36
losses** on seed 4040, each over 100 games. Against random play, the RL player
with search won **80/80** on seed 123.

The search is material. Without it, the guarded RL policy beat the guarded
supervised policy **65–26** head-to-head with 9 draws on seed 2027, but scored
only **2/1/37** versus depth-6 minimax on seed 123, compared with **7/3/30**
for the guarded supervised policy. PPO's value MSE against the *old minimax
search target* rose from 0.188 to 0.274; that target is no longer what the RL
value head was trained to predict, so it is not a measure of outcome-value
quality. These results support the combined RL-plus-search configuration, not
a claim that the raw RL network solves Connect 4. The opening samples are
small, and no camera or arm was used.

## Improved branch: `pytorch-models-v2`

The selected v2 checkpoints shipped with that branch. The 55-image COCO export
was read without changing its raw files. `prepare_board_cells.py` found 49
complete annotated boards: 34 train, 10 validation, and 5 test. It combined
full calibrated-cell crops with the original training object crops (3,063
training examples), then evaluated on full cells only (420 validation and
210 test examples). Six source photos with missing or inconsistent geometry
were skipped. Five yellow test chips annotated as red and one red training chip
annotated as yellow were corrected in the derived dataset via the explicit,
visually reviewed `label_fixes.json`. One oversized validation chip box was
discarded by the geometry filter. The test label corrections followed inspection
of the first model's errors, so this is an audited comparison rather than a
fresh blind test.

The new cell CNN used brightness/contrast and mild blur augmentation, with
extra loss weight for less frequent chip classes. The checkpoint was selected
at epoch 3 using clean, dim, and bright validation F1. On the five corrected
test photos, both the old and new models classified **210/210 clean cells**.

| Controlled test condition | Old CNN | New CNN |
|---|---:|---:|
| Clean cells correct | 210/210 | 210/210 |
| 55% brightness cells correct | 194/210 | **210/210** |
| 140% brightness cells correct | 183/210 | **210/210** |
| Complete clean boards accepted at 0.90 confidence | 3/5 | **5/5** |
| Complete dim boards accepted | 2/5 | 3/5 |

These are controlled brightness changes of the same five photos. They show
better robustness on this probe, not proven performance on a new camera or
occluded board. The export has no hand/arm occlusion labels. Keep using the
existing board legality and robot placement checks during future live tests.

The new player used **1,200 simulated games**, with the starting color chosen
randomly and **55% exploratory moves** mixed with depth-6 minimax labels. The
dataset contains **19,631 train**, **1,604 validation**, and **1,506 test**
positions, with game and mirrored-position separation. Its value target is
the teacher's search score scaled with `tanh(score / 400)` from the side to
move. The first dataset's terminal-outcome target was unhelpful because its
first mover won every generated game. Tactical positions received four times
the policy loss weight. The selected checkpoint is epoch 11.

On the new held-out positions, move agreement with the teacher is **49.1%**
and value MSE is **0.188**, compared with **0.293** for a constant-zero value
prediction on those same positions. Because the training target and dataset
changed, the old and new value MSE values are not directly comparable.

Matches used identical seed 123, 0–6 random opening moves, and alternated
which color the network played. The optional guard is a one-move win/block
rule. It uses no minimax call at runtime, and the pure network results are
listed separately.

| Player | Versus random (40 games) | Versus depth-6 minimax (20 games) |
|---|---:|---:|
| Old pure network | 31 wins, 9 losses | 2 wins, 18 losses |
| New pure network | **40 wins** | 2 wins, 1 draw, 17 losses |
| New network + tactical guard | **40 wins** | 2 wins, 2 draws, 16 losses |

The guarded player blocked **20/20** single immediate threats encountered
against minimax, versus **13/18** for the pure new network. This is an
improvement in basic tactical reliability. The network remains substantially
weaker than depth-6 minimax; it needs a stronger training curriculum or search
before being described as competitive with that engine.

The v2 checkpoint paths were `artifacts/vision/best.pt` and
`artifacts/player/best.pt`. In this self-play branch the same vision weights
remain at that path; the v2 player is `artifacts/player/supervised.pt` and the
RL player is `artifacts/player/best.pt`. Raw images, derived crops, game
datasets, and experiment histories remain local and ignored by Git.

## Earlier baseline: `pytorch-models`

These are learning baselines. Both networks were trained from random weights on
CPU using the existing Python environment (PyTorch 2.13.0). No camera, serial
connection, or physical arm was used. Datasets and checkpoint files are local
artifacts, separate from the source commit.

### Cell classifier

Source: the supplied `Connect4-Forked on 8-17-2026.coco (1)` export. Its files
were read in place without modification. Derived crops live in `data/cells`.

| Split | Source images | Empty crops | Red crops | Yellow crops |
|---|---:|---:|---:|---:|
| Train | 39 | 1,228 | 215 | 192 |
| Validation | 11 | 329 | 69 | 64 |
| Test | 5 | 134 | 44 | 32 |

Some source cells are not annotated; the converter exports only explicit
annotations. None of these splits has occlusion examples. It preserved the
provided image splits and found no identical source-image bytes across them;
recording/session independence has not been established.

Trained for 20 epochs, Adam learning rate 0.001, batch size 64, seed 42.
Checkpoint selected by validation macro F1: epoch 5, validation accuracy 99.8%,
validation macro F1 0.997.

Held-out **cell** test accuracy: **96.7% (203/210)**, macro F1 **0.943**.

| True class | Predicted empty | Predicted red | Predicted yellow |
|---|---:|---:|---:|
| Empty | 134 | 0 | 0 |
| Red | 0 | 39 | 5 |
| Yellow | 2 | 0 | 30 |

The small test set represents five photos, not 210 independent scenes. This
does not measure exact-board accuracy or prove reliable calibrated-cell
inference. Tight COCO crops may differ from full cells in the application.
There is no measured RF-DETR comparison in this experiment.

Artifacts: `artifacts/vision/best.pt`, `history.json`, `evaluation-test.json`.

### Neural player

Generated 150 games with depth-6 minimax move labels, 35% random exploration,
seed 42. After game splits and removing cross-split position/mirror overlap:
2,723 training positions, 295 validation positions, 212 test positions.

Trained for 30 epochs, Adam learning rate 0.001, batch size 128, seed 42.
Selected epoch 24 by validation move agreement: **32.9%**. Held-out test move
agreement: **44.3%**. Agreement is not the same as optimal-move accuracy because
multiple moves can be equally good and the teacher is depth limited.

Test value MSE: **2.029**, worse than a constant-zero prediction's MSE of 1
on these win/loss examples. The value head is not yet a useful advantage
estimate. Training loss decreased while generalization remained weak; simply
adding epochs is not a sufficient fix.

Matches used seed 123, alternated neural sides, and started with 0–6 random
opening drops. No tactical overrides or minimax fallback helped the network.

| Opponent | Wins | Draws | Losses |
|---|---:|---:|---:|
| Random, 40 games | 31 | 0 | 9 |
| Depth-6 minimax, 20 games | 2 | 0 | 18 |

Against random play, it took 31/101 available immediate winning moves and
blocked 6/33 single threats. These are encountered positions, not an independent
tactical benchmark: repeatedly missing a win can count several opportunities
within one game. Against minimax, those figures were 2/2 and 9/15.

Artifacts: `artifacts/player/best.pt`, `history.json`,
`evaluation-random.json`, `evaluation-minimax.json`; dataset: `data/games.npz`.

### Next learning experiments from the baseline

1. Keep the 55 images for the initial lesson. Add new recording sessions with
   balanced red/yellow coverage and the lighting that caused these mistakes.
   A practical next collection target is 200–500 diverse board photos, adjusted
   according to measured errors. Reserve entire sessions for evaluation.
2. Add hand/arm occlusion labels and actual calibrated-cell crops. Evaluate
   complete boards and scan rejection, not only individual objects.
3. Generate more varied game positions and a separate win/block curriculum.
   Compare pure policy behavior before and after that change.
4. Diagnose the value targets/distribution and compare value training against
   simple baselines. A larger game sample or a separate value-training target
   is a better next experiment than trusting this first head.
5. Keep the original engine/detector as reference backends. The existing state
   machine and hardware steps remain the application integration points.
