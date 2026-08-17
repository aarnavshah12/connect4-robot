# Progress

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
- **commentary.py + voice.py**: Anthropic API (opus-5, thinking off, effort low,
  3 s timeout, no retry), banned-phrase checker enforced pre-speech, 12+ canned
  lines per trigger for offline, ~60% speak rate, `say -v Zarvox` default with
  ElevenLabs behind the same speak() interface, depth-1 speech queue.
- **Tests: 38 passing** (driver bytes, engine gate, board logic, vision plumbing
  across real process boundaries, overlay fps proxy, commentary spec, state
  machine turns incl. error recovery + wins).

## Gate status
1. Vision (50 identical reads, static board): **owner-run** — needs camera + board.
2. Engine (win-in-1/block-in-1 across 20 games): **PASSED** (automated, tests/test_engine.py).
3. Overlay 30+ fps: compose is 1.1 ms (proxy PASSED); live-camera confirmation owner-run.
4. 5 unattended games: needs the physical rig.
5. Voice: built; audition ElevenLabs voice (owner), `say -v Zarvox` works now.

## Demo-day checklist (owner)
1. Copy `config.example.yaml` → `config.yaml`; fill Roboflow API key (+ Anthropic
   key for live trash talk; optional ElevenLabs). Confirm model version `/1` on
   the dashboard.
2. Plug in arm + camera. Close any VS Code Serial Monitor tab (it steals the port).
3. Teach poses: `.venv/bin/python jog.py` (h=home, e=feeder pick, 0-6=columns,
   t=travel height, y=test cycle, q=save).
4. Calibrate camera: `.venv/bin/python calibrate.py` (click 4 corners, s=save).
5. Rehearse without arm: `.venv/bin/python game.py --no-arm`.
6. The real thing: `.venv/bin/python game.py`. Board cleared = new game.
7. Set `show_fps: false` in config.yaml for the final recording.

## Blocked on owner
- Roboflow API key; model version confirmation (Roboflow MCP unauthenticated here).
- Pose teaching, camera calibration, ElevenLabs audition — all physical-rig steps.
- End-to-end model smoke test (needs the API key): first run of game.py will
  print the MPS + model-load log lines from the vision worker — check them.

## Notes
- Python: `.venv/` (3.12, uv-managed). Run everything with `.venv/bin/python`.
- Arm DISCONNECTED during this build; all hardware code unit-tested against a
  fake serial port. First physical run: confirm workspace clear when prompted.
