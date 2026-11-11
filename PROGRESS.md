# Progress

## Self-play RL player (2026-11-10, `pytorch-self-play`)
- Kept the v2 vision checkpoint unchanged and documented saved-photo and
  camera-only vision checks; no arm was run.
- Trained an outcome-based PPO player for 10,240 simulated games (98,919
  learner positions) against frozen PyTorch, random, and depth-4 minimax
  opponents. Preserved the supervised checkpoint for direct comparison.
- Added neural-guided MCTS with 64 simulations per move and no old minimax call
  at runtime. A warmed midgame move took about 12 ms on this machine's CPU.
- With the same search and tactical guard, RL beat the supervised checkpoint
  49–34 and 46–36 in two 100-game direct comparisons, with 17 and 18 draws.
  Against depth-6 minimax, it scored 14/3/23 and 14/5/21 (W/D/L) in two
  40-game sets; see `ml/RESULTS.md` for matched baselines and limitations.
- Offline checks: 56 tests passed.

## PyTorch model improvements (2026-11-10, `pytorch-models-v2`)
- Audited the supplied COCO export without changing raw files; six explicit
  chip-color corrections live in `ml/label_fixes.json`.
- Added complete-board crop preparation and 55%/140% brightness checks.
  On the five audited test photos, the improved vision CNN classified
  210/210 clean, dim, and bright cells; the first CNN got 210/210, 194/210,
  and 183/210 respectively. The new checkpoint accepted 5/5 clean boards
  at 0.90 confidence, versus 3/5 previously.
- Rebuilt the player dataset from 1,200 simulated games with either color
  starting, depth-6 search values, and tactical loss weighting. On the
  1,506-position test set, teacher agreement is 49.1% and value MSE 0.188.
- Same-seed matches: pure player 40/40 vs random and 2 wins/1 draw in 20
  games vs depth-6 minimax. Optional one-move tactical guard blocked all
  20/20 single threats encountered vs minimax and gained one more draw.
- See `ml/RESULTS.md` for methods and limits. No arm or live camera was used.

## PyTorch learning branch (2026-11-10)
- Isolated `pytorch-models` worktree; original main checkout's tracked files unchanged.
- Added small cell CNN and policy/value network with explicit Python training loops.
- Added COCO conversion, simulated-game generation, held-out evaluation, terminal play,
  backend selection, and a hands-on walkthrough in `ml/README.md`.
- Trained vision on the supplied 55-image export: 203/210 held-out cell crops correct.
- Trained player on 150 simulated games: 31/40 wins vs random, 2/20 vs depth-6 minimax.
  Value estimates remain poor. Full details and limitations: `ml/RESULTS.md`.
- No physical hardware used. Existing arm, pose teaching, calibration, and voice
  driver files remain unchanged. Model adapters preserve the existing game stages.
- Pending: larger/independent vision evaluation, occlusion examples, stronger
  neural-player training, exact-board accuracy, and a measured RF-DETR comparison.
- Offline checks: 52 tests passed; all 6 affected game/integration checks also
  passed after the final reference-backend isolation edit.

## Done (2026-08-17)
- **Arm bring-up** (physical arm, morning session): installed HiWonder
  `MaxArm_micropython_microUSB` serial firmware (factory files backed up in
  `maxarm_factory_backup/`), fixed `test.py`, full read/suction/motion test passed
  on `/dev/cu.usbserial-310` @ 9600.
- **Hardware layer**: `maxarm.py` (driver, validated protocol incl. reply-checksum
  quirk + boot wait), `jog.py` (pose teacher), `poses.json` (PLACEHOLDERS,
  `calibrated: false` — pick_and_drop refuses to run until jog.py stamps it true).
  Pick cycle per owner spec: feeder hover → down → pump on → lift → over column →
  release → home. Robot plays RED.
- **engine.py**: bitboard negamax, depth 8, TT with bound flags. 12 ms/move.
- **board.py**: homography snap, half-cell reject, 5-read debounce, physics check,
  diff/legality/win. Detections are `(x, y, w, h, class, conf)` project-wide.
- **calibrate.py**: 4-corner click → homography + cell centers + inference crop box.
- **vision.py**: camera thread + RF-DETR in a separate process over shared memory
  (GIL-proof); MPS verified available (torch 2.13). Crop-before-infer wired in.
- **overlay.py**: banner/twin/history/advantage-bar/ticker, ghost drop, considered-
  column flash, win line, fps counter. `python overlay.py` = phase-3 mock (SPACE
  cycles states). compose() = 1.1 ms/frame.
- **game.py**: full state machine (occlusion-aware debounce, ERROR recovery,
  VERIFYING with expected-board match, board-cleared → new game). `--no-arm`
  rehearsal mode. Workspace-clear confirmation before first motion (safety rule).
- **commentary.py + voice.py**: Gemini one-liners (see below), banned-phrase
  checker enforced pre-speech, 12+ canned lines per trigger for offline, ~60%
  speak rate, ElevenLabs voice with cache + depth-1 speech queue, `say -v
  Zarvox` offline fallback. Stale lines dropped (newest trigger wins); exit
  never blocks on an in-flight API call.
- **Tests: 40 passing** (driver bytes, engine gate, board logic, vision plumbing
  across real process boundaries, overlay fps proxy, commentary spec + stale-line
  ordering, state machine turns incl. error recovery + wins).

## Verified live against real services (2026-08-17 afternoon)
- **Roboflow**: key auto-read from `.claude/settings.local.json` (never committed;
  `.claude/` is gitignored). The plan's 3-part model id is the dashboard checkpoint
  name; the local `inference` package needs **`connect4-kewhf/1`** (rfdetr-small,
  trained 2026-08-17 10:10am). Model downloads, loads (8s), and infers at
  **~175 ms/frame** via CoreML — plenty for the debounce loop; overlay unaffected.
- **Model classes are `Board / No Piece / Red Piece / Yellow Piece`** — human
  plays YELLOW, not blue. Mapping + overlay colors + tests all updated;
  Board/No Piece detections are ignored by construction.
- **Commentary is now Gemini** (owner request): `gemini-3.5-flash-lite`,
  measured 0.6-0.9 s/line live with the real key (in gitignored config.yaml).
  Google rejects <10 s client deadlines, so the plan's 3 s budget is enforced
  client-side (future.result(3.0) -> canned fallback; late lines discarded).

## Gate status
1. Vision (50 identical reads, static board): **owner-run** — needs camera + board.
2. Engine (win-in-1/block-in-1 across 20 games): **PASSED** (automated, tests/test_engine.py).
3. Overlay 30+ fps: compose is 1.1 ms (proxy PASSED); live-camera confirmation owner-run.
4. 5 unattended games: needs the physical rig.
5. Voice: ElevenLabs live-verified (Callum premade; owner's library voice needs a paid tier), cached playback works; `say -v Zarvox` remains the offline fallback.

## Demo-day checklist (owner)
1. config.yaml is DONE (Roboflow auto-read from .claude, Gemini + ElevenLabs keys
   in, model id verified). Only optional tweak: `camera_index` if using an
   external webcam.
2. Plug in arm + camera. Close any VS Code Serial Monitor tab (it steals the port).
   Human plays YELLOW pieces; feeder stocked with RED. Light the board evenly
   (the model's one weakness is red↔yellow confusion in poor light).
3. Teach poses: `.venv/bin/python jog.py` (h=home, e=feeder pick, 0-6=columns,
   t=travel height, y=test cycle, q=save).
4. Calibrate camera: `.venv/bin/python calibrate.py` (click 4 corners, s=save).
5. Rehearse without arm: `.venv/bin/python game.py --no-arm`.
6. The real thing: `.venv/bin/python game.py`. Board cleared = new game.
7. Set `show_fps: false` in config.yaml for the final recording.

## Blocked on owner (physical-rig steps only)
- Pose teaching (jog.py) and camera calibration (calibrate.py) on the real rig.
- Everything key/model/API-related is resolved and live-verified.

## Notes
- Python: `.venv/` (3.12, uv-managed). Run everything with `.venv/bin/python`.
- Arm DISCONNECTED during this build; all hardware code unit-tested against a
  fake serial port. First physical run: confirm workspace clear when prompted.
