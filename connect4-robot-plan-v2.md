# Connect 4 Robot: Build Plan v2

Handoff spec for an LLM coding agent. Hardware: HiWonder MaxArm on macOS over USB serial (driver already written: maxarm.py, poses taught via jog.py into poses.json). Vision: Roboflow RF-DETR model. Move engine: minimax. LLM: live commentary only, not move selection. The deliverable is a demo that looks great on video.

## Architecture

```
webcam -> RF-DETR (local, real time) -> detections
       -> homography -> 7x6 grid -> board state
       -> state machine -> minimax move -> serial -> MaxArm
                        -> overlay window (the demo surface)
                        -> LLM commentary (async, non-blocking)
```

## Vision: the trained model

Model: aarnavs-space/connect4-kewhf-1-rfdetr-small-t1 (append the version number, e.g. /1, check the Roboflow dashboard for the latest).

Run the model natively in-process on Apple Silicon's GPU (MPS). Do NOT use the Docker inference server on macOS: Docker on Mac is a CPU-only Linux VM, which makes inference slow and starves the draw loop of CPU at the same time. No HTTP client either; pass the raw numpy frame directly:

```python
# pip install inference   (no Docker, no server)
from inference import get_model

model = get_model(
    model_id="aarnavs-space/connect4-kewhf-1-rfdetr-small-t1/1",
    api_key=API_KEY,
)
result = model.infer(frame)[0]  # frame: numpy array from OpenCV, verify torch is using mps
```

Verify at startup that torch reports the mps device is available and in use, and log which device the model landed on. Expected on an M-series Mac: RF-DETR small in the 15 to 30 fps range on MPS versus low single digits on CPU.

Two mandatory speedups on top:

1. Crop before inferring. The calibration defines exactly where the board sits in the frame. Send only that crop (plus a small margin) to the model, never the full frame. Fewer pixels means faster inference, and the pieces fill far more of the model's input resolution, which improves detection quality too.
2. Inference runs in a separate PROCESS, not a thread. Python threads share the GIL, so model CPU spikes still stutter a threaded draw loop. Use multiprocessing: the worker reads the newest frame from shared memory, runs the model, and publishes predictions back through a queue or shared structure. The draw loop must be completely isolated from inference compute. This isolation, not raw model speed, is what guarantees the 30 fps floor.

Target inference rate: as fast as the model goes on MPS, typically 15 to 30 fps with the crop. There is no need to throttle it; the debounce logic in Grid Mapping consumes results at whatever rate they arrive. Fallback if MPS setup fights back: Roboflow's serverless endpoint via inference-sdk works with an api_url swap, at the cost of per-call latency; it is acceptable for early bring-up only, not for the demo recording.

Demo clarity is a hard requirement, not a preference. The overlay window is what gets screen-recorded, so it must hold a minimum of 30 fps at all times, in every state, including while inference, minimax, the arm, the LLM, and TTS are all running at once. Treat any drop below 30 fps as a bug and fix it before moving on. The way to guarantee this: the draw loop owns the window and does nothing but draw. Inference lives in its own process (see above) and publishes its latest predictions; the draw loop reads whatever is newest and renders it over the current camera frame. Same rule for everything else: minimax result, commentary lines, and game state are all published to the draw loop, never computed inside it. If the camera supports 60 fps at 720p, prefer it and render at 60; the fps floor is 30, not the target.

Also for clarity on video: capture at 1280x720 minimum, lock the camera's exposure and white balance after setup (auto-exposure hunting makes detections flicker and looks bad on film), light the board evenly, and render the overlay window at the camera's native resolution so nothing is upscaled and soft.

## Grid mapping

Same as v1: one-time calibration clicking the 4 corners of the play area, cv2.getPerspectiveTransform, 7x6 cell centers saved to disk. At runtime, warp detection centroids and snap to nearest cell. Reject detections farther than half a cell from any center. Debounce: identical parsed board on 5 consecutive inference results before it counts. Physics check: no floating pieces, else rescan.

## Move engine (engine.py)

Minimax with alpha-beta pruning, depth 8, standard Connect 4 evaluation (center column weighting plus open 2s and 3s). Runs in under 100 ms in plain Python at depth 8 with move ordering (search center columns first). Never returns an illegal move by construction. Expose one function:

```python
def best_move(board, player) -> tuple[int, int]  # (column, eval_score)
```

The eval score feeds the overlay's advantage bar. If an unbeatable robot is wanted later, swap in a perfect solver behind the same signature.

## Trash talk with voice (commentary.py + voice.py)

The LLM never picks moves. It is the robot's mouth. On each trigger, fire an async request with the board, the move, the eval score, and the recent banter history (so it doesn't repeat itself), asking for one spoken line under 12 words, cocky but friendly, PG. The line goes to the overlay ticker and to TTS simultaneously.

Triggers, each with its own prompt flavor:

- game start: an opening taunt
- human moved, eval barely changed: light jab
- human blundered (eval swings 300+ toward robot): pounce on it, reference the actual mistake ("column 2? bold. wrong, but bold")
- human played the best move: grudging respect
- robot sets up a fork or win-in-2: ominous foreshadowing without spoiling the column
- robot wins: gloat
- human wins: salty but gracious
- human takes over 20 s to move: impatience line

Line style, so it doesn't sound like an AI. Bake these into the system prompt and enforce with a banned-phrase check before speaking; on violation, use a fallback line instead of retrying:

- Mostly 3 to 8 words. Fragments are good. Sometimes one word ("cute." "no."). Hard cap 12.
- Deadpan, not hype. Periods, never exclamation marks. No "Oh," "Ah," "Well well well," "Ooh."
- Banned outright: "bold move", "interesting choice", "let's see", "calculating", "beep boop", "as an AI", "game on", "bring it", "fascinating", "delightful", "shall we", anything with an em dash, any rhetorical "The question is...".
- Reference concrete things only: the column number, the seconds they took, the piece they just hung. Never generic gamer talk.
- Contractions always. Lowercase energy.
- Silence is a weapon: speak on roughly 60% of triggers, chosen randomly. A robot that comments on everything sounds scripted; one that occasionally says nothing and just plays sounds confident.
- Vary rhythm across the game: if the last line was long, the next is short.

Voice pipeline (voice.py):

- Voice: ElevenLabs. Pick a dry, low-energy voice from the library (audition for deadpan, not "announcer"; smug reads better slightly slow and quiet). Use their expressive model rather than the low-latency one: lines are short and async so latency doesn't matter, delivery does.
- Delivery is steered by the text itself: periods force downward inflection, an ellipsis buys a pause ("column four... obviously"), lowercase keeps it flat. Have the LLM write lines with this punctuation in mind.
- Keep macOS `say -v Zarvox` as the config-swap fallback for offline demos, behind the same speak(text) interface.
- Cache generated audio by line hash so repeated fallback lines don't re-bill and play instantly.
- Speech runs on its own queue thread. Max queue depth 1: if a new line arrives while one is pending, drop the old pending line (stale trash talk is worse than none). Never interrupt a line mid-sentence.
- Mute during ROBOT_MOVING arm motion only if the mic/audio setup for the demo video makes overlap messy; otherwise talking while the arm moves is peak robot.

Rules: total LLM timeout 3 s, then fall back to a local list of ~20 canned lines per trigger so the robot still talks offline. The game loop never waits on commentary or speech; both are fire-and-forget.

## The overlay (overlay.py): this is the demo

One fullscreen OpenCV window (or pygame if easier to make pretty) showing the live camera feed with layers drawn on top. Use the supervision library for detection annotation, it looks professional out of the box. Performance gate from the Vision section applies here: 30 fps minimum in every state. Render an fps counter in a corner during development (hidden behind a config flag for the final recording) so regressions are visible immediately. Prefer cheap draw calls: pre-render static elements (panel background, grid lines, banner frames) once and blit them, rather than redrawing shapes every frame.

Layout:

- Live feed with bounding boxes on every detected piece, red and blue, with confidence.
- Grid lines drawn over the board via the calibration homography, so viewers see the machine's mental model aligned to the physical board.
- Right side panel: a rendered digital twin of the board (clean 7x6 graphic), move history list, and a vertical advantage bar driven by the minimax eval that swings toward whoever is winning.
- Bottom ticker: LLM commentary lines.
- Top center: the status banner, driven by the state machine below. Big type, color coded.

Status banner states:

| State          | Banner text                  | Visual                          |
|----------------|------------------------------|---------------------------------|
| WAIT_HUMAN     | YOUR TURN, DROP A PIECE      | pulsing green                   |
| SCANNING       | READING BOARD                | subtle spinner                  |
| THINKING       | THINKING                     | animated dots, eval bar updates |
| ROBOT_MOVING   | ROBOT PLAYING COLUMN n       | target column highlighted       |
| VERIFYING      | (no banner, brief)           |                                 |
| HUMAN_WIN      | YOU WIN                      | winning 4 highlighted on feed   |
| ROBOT_WIN      | ROBOT WINS                   | winning 4 highlighted on feed   |
| ILLEGAL/ERROR  | BOARD LOOKS WRONG, FIX IT    | red, shows what it expected     |

Extra demo flourishes, in priority order:

1. When the robot picks a column, draw a ghost piece dropping down that column on the digital twin before the arm moves. Machine announces intent, then the physical arm does it. This is the money shot.
2. Highlight the winning four with a drawn line on the live feed when the game ends.
3. During THINKING, flash the columns minimax is evaluating on the digital twin. It is honest (you can emit them from the search root) and looks great.
4. Piece-count and turn-number chips in a corner.

## Game state machine (game.py)

```
WAIT_HUMAN: banner on. Watch for a stable board with exactly one new human piece.
  Ignore all frames where a hand/arm occludes the grid (detection count drops).
SCANNING -> confirm stable -> check human win -> THINKING
THINKING: engine.best_move (instant, but hold the banner ~1.5 s for drama,
  during which the LLM commentary request fires)
ROBOT_MOVING: ghost drop animation, then arm.pick_and_drop(col) via maxarm.py
  poses from poses.json: feeder -> col{n} -> release -> home. Do not scan
  while the arm is in frame; resume after the arm reports home.
VERIFYING: rescan, confirm the expected piece appeared in the expected cell.
  Mismatch -> ERROR state with expected vs seen.
-> check robot win -> WAIT_HUMAN
```

## Repo layout

```
connect4-robot/
  maxarm.py        # done: serial driver
  jog.py           # done: pose teaching -> poses.json
  poses.json       # done: taught poses (feeder, col0..col6)
  calibrate.py     # click corners, save homography + cell centers
  vision.py        # camera thread + local RF-DETR inference thread
  board.py         # grid snap, debounce, legality, win detection, diff
  engine.py        # minimax alpha-beta depth 8, returns (col, eval)
  commentary.py    # async LLM trash talk, triggers, 3 s timeout, fallbacks
  voice.py         # speak(text): macOS say by default, ElevenLabs optional
  overlay.py       # the fullscreen demo window
  game.py          # state machine tying it all together
  config.yaml      # api key, model id, port (/dev/cu.usbserial-*), camera index
```

## Build order

1. calibrate.py + vision.py: live window, boxes, grid, printed board state. Gate: 50 identical reads of a static board.
2. engine.py: beat it yourself in a terminal. Gate: never misses a win-in-1 or block-in-1 across 20 games.
3. overlay.py with a mocked state machine cycling through states on keypress, so the visuals can be polished without hardware. Gate: 30+ fps sustained in every state with live camera and inference running.
4. game.py integration with the real arm. Gate: 5 full games unattended, overlay never dips below 30 fps at any point in any game.
5. commentary.py + voice.py last. Garnish, must never block anything. Test voice standalone first: `say -v Zarvox "column four. as if you had a choice"` in a terminal tells you in 5 seconds whether the default voice lands or you want ElevenLabs.
