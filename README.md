# Connect 4 Robot

A robot arm that plays Connect 4 against you — and talks trash while doing it.

A webcam watches the board, an RF-DETR model (trained on
[Roboflow](https://roboflow.com)) reads the pieces, a minimax engine picks the
move, a HiWonder MaxArm picks a red piece off a feeder stack with a suction cup
and drops it into the chosen column, and Gemini writes board-aware roasts that
an ElevenLabs voice delivers out loud. A fullscreen overlay shows the live
feed, detections, the robot's "mental model" of the board, an advantage bar,
and the commentary ticker.

```
webcam -> RF-DETR (local, ~175ms/frame) -> homography -> 7x6 board state
       -> state machine -> minimax (depth 8, ~12ms) -> serial -> MaxArm
                        -> overlay window (the demo surface)
                        -> Gemini commentary + ElevenLabs voice (async)
```

**The LLM never picks moves.** Minimax plays; the LLM is only the mouth.

## Hardware

- HiWonder MaxArm (ESP32, MicroPython) over USB serial, running HiWonder's
  `MaxArm_micropython_microUSB` serial firmware (factory firmware backed up in
  `maxarm_factory_backup/`)
- Suction-cup end effector; feeder = a vertical stack of red pieces
- Any webcam at 720p+ pointed at the board
- Human plays YELLOW, robot plays RED

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cp config.example.yaml config.yaml   # then fill in your keys
```

Keys in `config.yaml` (gitignored): Roboflow (or via `ROBOFLOW_API_KEY` /
`.claude/settings.local.json`), Gemini for trash talk, ElevenLabs for the
voice. Everything degrades gracefully: no Gemini key = canned lines, no
ElevenLabs = macOS `say`.

## Calibration (once per rig setup)

1. **Arm poses** — `python jog.py` (full teacher) or `python nudge.py`
   (simple pad: arc left/right, up/down, suction test). Teach the feeder pick
   point (`e`, on top of a FULL stack) and the 7 column drop slots (`0`-`6`).
   `poses.json` stores them; `piece_thickness` makes the pick sink as the
   stack empties.
2. **Camera** — `python calibrate.py`: click the 4 corners of the play area,
   press `x` if the grid comes up rotated, `s` to save.
3. **Verify** — `python run_test.py --spot 1 --pieces 3` (arm cycles without
   the game), `python debug_vision.py` (live window + per-detection log of
   exactly what the board parser sees and rejects).

## Run

```bash
.venv/bin/python game.py           # the real thing
.venv/bin/python game.py --no-arm  # rehearsal: you place the robot's pieces
```

Drop a yellow piece when the banner says it's your turn. Clearing the board
after a win starts a new game. Every spoken line is logged to
`roast_log.jsonl`.

## Repo map

| File | What it is |
|---|---|
| `game.py` | State machine: WAIT_HUMAN → SCANNING → THINKING → ROBOT_MOVING → VERIFYING |
| `engine.py` | Bitboard negamax, depth 8, alpha-beta + transposition table |
| `vision.py` | Camera thread + RF-DETR in a separate process over shared memory |
| `board.py` | Homography snap, debounce, physics check, legality, win detection |
| `overlay.py` | The demo window (also runs standalone with a mocked state machine) |
| `maxarm.py` | Serial driver for the arm (validated protocol quirks documented inside) |
| `commentary.py` / `voice.py` | Gemini roasts (board-aware) + ElevenLabs speech |
| `jog.py` / `nudge.py` / `calibrate.py` / `run_test.py` / `debug_vision.py` | Calibration & bring-up tools |
| `connect4-robot-plan-v2.md` | The original build plan |

## Tests

```bash
.venv/bin/python -m pytest tests/   # 41 tests, no hardware needed
```

Driver bytes vs a fake serial port, engine never missing win-in-1/block-in-1,
board parsing/debounce, vision plumbing across real process boundaries,
overlay frame budget, commentary style rules, full state-machine turns.
