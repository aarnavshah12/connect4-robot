# Progress

## Done
- Arm bring-up (2026-08-17 morning, physical arm connected): installed HiWonder
  `MaxArm_micropython_microUSB` serial firmware on the arm (factory files backed up in
  `maxarm_factory_backup/`), fixed `test.py` (port, reply checksum, boot wait), full
  read/suction/motion test passed on `/dev/cu.usbserial-310` @ 9600.
- Repo initialized.

## In progress
- Hardware layer: `maxarm.py`, `jog.py`, placeholder `poses.json`.

## Next
- engine.py → board.py → calibrate.py → vision.py → overlay.py → game.py → commentary/voice.

## Gate status
1. Vision (50 identical reads of static board): NOT RUN — needs camera + board (owner setup).
2. Engine (never miss win-in-1/block-in-1, 20 games): pending build.
3. Overlay 30+ fps all states: pending build (will verify with synthetic feed; final check needs live camera).
4. 5 unattended games: needs physical rig (owner session).
5. Voice: pending build.

## Blocked on owner
- Roboflow API key (none found on this machine) → goes in `config.yaml`.
- Model version number: dashboard check not possible from this session (Roboflow MCP
  unauthenticated); using `/1` until confirmed.
- Pose calibration at demo time: run `jog.py` to teach feeder/col0–col6/home (current
  poses.json is placeholders, marked `"calibrated": false`).
- Camera calibration clicks (`calibrate.py`) against the physical rig.
- Anthropic API key for commentary + optional ElevenLabs key for voice.

## Notes
- Arm currently DISCONNECTED (owner unplugged everything). All hardware code is
  unit-tested against a fake serial port; first physical run must confirm workspace clear.
- Robot plays RED pieces (default).
