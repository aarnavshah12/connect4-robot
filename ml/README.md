# Learn PyTorch with the Connect 4 robot

This branch builds two small neural networks with ordinary Python training
scripts. You can read every layer and every optimizer step. Start with the
player: its data comes from simulation, so it needs no hardware or images.

The new branch lives in a separate worktree. The original `main` checkout's
tracked files have not been edited. Run the commands below from this worktree.

## 1. Understand the pieces

| Idea | Meaning here |
|---|---|
| Tensor | A multidimensional array: an image, a board, or a batch of examples |
| Model | An `nn.Module` that maps inputs to predictions using learned weights |
| Logits | Raw class/move scores before softmax |
| Loss | A number measuring prediction error on labeled examples |
| Backpropagation | Computing how each weight affects the loss |
| Optimizer | Updating weights using those gradients |
| Batch | A group of examples processed in one step |
| Epoch | One pass through the training examples |
| Validation set | Separate data used to select a checkpoint and tune experiments |
| Test set | Held-out data used to assess the selected experiment |
| Checkpoint | Saved weights plus the metadata needed to load them correctly |

In `train_vision.py`, find the five lines from `optimizer.zero_grad()` through
`optimizer.step()`. That is the core PyTorch training loop. Both tasks use it.
We feed raw logits into `CrossEntropyLoss`; it already handles the necessary
logarithms and normalization. Softmax is used when displaying predictions.

Training uses `model.train()` and gradient tracking. Evaluation uses
`model.eval()` and `torch.inference_mode()`. The first selects layer behavior;
the second turns off autograd bookkeeping. They have different jobs.

## 2. Environment

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-ml.txt
```

An existing compatible Python environment also works. The initial experiments
used the original checkout's existing environment through a local, ignored
`.venv` symlink. No packages were installed into that environment.
The training dependency list needs no cloud keys. Running the full robot,
including its optional services, uses `requirements.txt`.

## 3. Train the game player

Read `positions.py`, `player.py`, and then `train_player.py` in that order.

Input shape is **[batch, 2, 6, 7]**. Plane 0 marks the current player's pieces;
plane 1 marks their opponent's. Empty cells are zero in both. Row 0 is the
bottom, matching `board.py`. This encoding lets one network play either color.

Two convolution layers extract patterns. A shared feature vector feeds two heads:

- **Policy:** seven logits, one per column. Full columns are masked out.
- **Value:** one number in [-1, 1], estimating the expected outcome for the
  current player: win +1, draw 0, loss -1.

Generate legal games using a mixture of minimax moves and random exploration:

```bash
.venv/bin/python -m ml.generate_games --games 150 --teacher-depth 6 --output data/games.npz
.venv/bin/python -m ml.train_player --data data/games.npz --epochs 30 --device cpu --output artifacts/player
```

The policy label is what minimax preferred at the recorded position. The value
label is how that generated game actually ended. These are different targets:
exploration can throw away a winning position, so the value is learning outcomes
under the data-generation mixture. It is not learning a perfect solver's value.

Loss = **cross-entropy(policy, teacher move) + MSE(value, game outcome)**.
Games are split before training, and any position shared with an earlier split
is excluded from later splits. Mirrored positions are grouped too. Training
augmentation mirrors columns and changes a move label `c` to `6-c`.

Evaluate and try the player yourself:

```bash
.venv/bin/python -m ml.evaluate_player --opponent random --games 40 --output artifacts/player/evaluation-random.json
.venv/bin/python -m ml.evaluate_player --opponent minimax --depth 6 --games 20 --output artifacts/player/evaluation-minimax.json
.venv/bin/python -m ml.play
```

Evaluation alternates the neural player's side and uses 0–6 random opening
drops. It reports move agreement, value error, actual wins/draws/losses, and
whether immediate wins and single threats were handled. It applies no hidden
tactical overrides. A move different from the teacher's can still be equally
good; match results and tactical failures provide additional evidence.

**First exercises:**

1. Build a board with one red and one yellow piece. Print `encode(board, 1)`
   and then `encode(board, 2)`. Identify what changes.
2. Count parameters with `sum(p.numel() for p in model.parameters())`.
3. Predict why masking a full column only after softmax can be misleading.
4. Compare train loss, validation agreement, and value error in `history.json`.
   More epochs can improve train loss while hurting generalization.
5. Generate 1,000 games with a new output filename, train into a new output
   directory, and compare against the first checkpoint on the same match seeds.
   Keep a fresh final test set for conclusions after tuning.
6. Add a training curriculum emphasizing immediate wins and blocks. Measure
   those tactical cases separately before trusting overall win rate.

The original depth-8 engine remains available as a baseline and data teacher.
The PyTorch backend chooses moves with the network at runtime. A later lesson
can add MCTS/self-play; this version is supervised learning, not reinforcement
learning.

## 4. Train the vision model

Read `cells.py`, `prepare_cells.py`, and `train_vision.py`.

For a fixed camera and board, the existing calibration can straighten the play
area and divide it into 42 cells. The CNN learns the cell contents. Board
geometry is supplied by calibration; there is no learned board locator here.

Input shape is **[batch, 3, 64, 64]**: RGB float32 values in [0, 1]. OpenCV loads
BGR, so preprocessing explicitly changes channel order. The classifier outputs
four logits: **empty, red, yellow, occluded**, in that order. The first three IDs
match the game board's cell values. Loss is cross-entropy against cell labels.

Export the existing dataset from Roboflow as **COCO JSON**, unzip it, and point
the converter at the folder containing `train`, `valid`, and `test`:

```bash
.venv/bin/python -m ml.prepare_cells --source /path/to/COCO-export --output data/cells
.venv/bin/python -m ml.train_vision --data data/cells --epochs 20 --output artifacts/vision
.venv/bin/python -m ml.evaluate_vision --data data/cells --checkpoint artifacts/vision/best.pt
```

The converter maps `No Piece` to empty, crops red/yellow boxes, ignores `Board`,
and preserves the export's existing splits. It requires explicit empty-cell
annotations; missing piece annotations cannot safely stand in for empty labels.
It also accepts an `Occluded` category for hand/arm examples. If the export
does not contain those examples, add them before live use; a fourth output
class without training examples cannot learn to recognize occlusions.

The converter rejects exact source-image duplicates across splits. It cannot
detect all related video frames or augmented versions: audit the original
splits by recording/session before trusting the test score. The 42 crops of
one frame are related examples, not 42 independent board configurations.

COCO object boxes bootstrap the classifier. Tight object-box crops can differ
from full calibrated cell crops at runtime. Add labeled rectified cell crops
from the actual camera to the appropriate split/class directories when that
domain difference causes failures. Keep each recording in one split.

The saved checkpoint is selected by validation macro F1 so a common empty
class cannot dominate selection. The evaluation prints a confusion matrix
(rows=true, columns=predicted) and per-class precision/recall.

**Vision exercises:**

1. Inspect one red, yellow, and empty crop. Compare them to the actual cell
   crops from the calibrated camera.
2. Run one batch through the model and check its input/output shapes.
3. Inspect the red/yellow confusion cells. Add examples of the lighting that
   causes confusion, then train a new experiment.
4. Tune the confidence threshold on validation recordings. Record both how
   often the model accepts a scan and how often accepted boards are correct.
5. Check exact 42-cell board matches and false stable boards on held-out
   recordings. Strong cell accuracy alone does not establish reliable board reads.

## 5. Connect the models to the application

After both checkpoints exist, copy `config.pytorch.example.yaml` to `config.yaml`
**inside this worktree**. Supply a calibrated camera file for this rig, then run
`debug_vision.py` and `game.py --no-arm` to check the vision integration.
Calibration and poses are rig-specific; the code does not invent new ones.

The two backend settings are independent. For comparisons you can select
`vision_backend: roboflow` with `player_backend: torch`, or the converse.
RF-DETR stays available as the reference detector while the small CNN handles
the main PyTorch vision experiment. Its relative accuracy has to be measured.

The PyTorch reader batches all 42 crops in the inference process. Any low-
confidence/occluded cell rejects the scan. `None` means unknown and `[]` means
confidently empty. Confidence is not a guarantee of correctness. The game counts
five distinct usable inference results on the PyTorch path, with unknown scans
resetting its streak. The reference detector retains its existing debounce behavior.
Existing gravity, one-move legality, and robot-placement verification still apply.

Neural values use their own [-1, 1] display scale and commentary wording. A
value near +1 is a learned estimate; it never triggers minimax's proven-win
announcement. Column highlights represent model preferences for the neural
backend, and search candidates for minimax.

## 6. Keep experiments reproducible

Datasets/checkpoints/logs are ignored by Git. Re-running into an existing dataset
or checkpoint path raises an error; use a new output name for each experiment.
`history.json` records each epoch. Seeds make CPU experiments repeatable in this
environment; different library versions/devices can still change results.

Run `python -m pytest tests/ -q` for offline checks. Synthetic vision tests
exercise preprocessing, coordinates, training, and process plumbing. Real
camera accuracy and sustained live overlay performance remain separate checks.

References: [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html),
[vision training](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html),
[standalone RF-DETR training](https://rfdetr.roboflow.com/learn/train/), and
[AlphaZero's policy/value and self-play approach](https://arxiv.org/html/1712.01815v1).
