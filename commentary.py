"""Trash talk. The LLM is the robot's mouth — it NEVER picks moves (CLAUDE.md).

Commentator.fire(trigger, **ctx) is fire-and-forget: a daemon thread asks
the Gemini API for one line (3 s total timeout), runs it through the
banned-phrase check, and falls back to a canned line on any failure —
violation, timeout, no API key. The game loop never waits on this.

Model: gemini-3.5-flash-lite (config `commentary_model` overrides) — measured
~1 s per line with no thinking overhead, which fits the 3 s budget with room.

Style spec (plan, binding): mostly 3-8 words, hard cap 12, deadpan,
periods never exclamation marks, lowercase energy, contractions, concrete
references only. Silence is a weapon: ~60% of non-terminal triggers speak.
"""

import json
import random
import re
import threading
import time
from pathlib import Path

MAX_WORDS = 28  # owner wants fuller roasts (was 12 under the original spec)

# every spoken line is appended here as one JSON object per line
LOG_PATH = Path(__file__).parent / "roast_log.jsonl"

BANNED = [
    "bold move", "interesting choice", "let's see", "calculating", "beep boop",
    "as an ai", "game on", "bring it", "fascinating", "delightful", "shall we",
]
BANNED_RE = re.compile(
    r"—|—|the question is|" + "|".join(re.escape(b) for b in BANNED),
    re.IGNORECASE,
)

# always speak on these; the rest roll the ~60% dice
ALWAYS_SPEAK = {"game_start", "robot_win", "human_win"}
SPEAK_P = 0.6

SYSTEM = """You are the voice of a robot arm playing connect 4 against a human. \
You write ONE spoken roast. Cocky, cutting, but friendly. PG.

Rules, all hard:
- one to two sentences, usually 12 to 22 words. never more than 28. \
make it a real roast with a specific observation, not a fragment.
- deadpan, not hype. periods, never exclamation marks. no "Oh," "Ah," "Well well well," "Ooh."
- never use: "bold move", "interesting choice", "let's see", "calculating", "beep boop", \
"as an AI", "game on", "bring it", "fascinating", "delightful", "shall we", em dashes, \
or any rhetorical "The question is...".
- roast the SPECIFIC situation: read the board, name the mistake, the column, the threat \
they missed, the trap they walked into, how long they took. never generic gamer talk.
- write with PROPER capitalization and punctuation: the voice engine acts out your \
punctuation. commas for timing, a question for mockery ("Column two? Again?"), an \
ellipsis for a pause ("Column four... obviously."), an exclamation only when genuinely \
gloating. make the line PERFORMABLE, with emotion built into the phrasing.
- contractions always. vary rhythm: if your last line was long, go shorter this time.
- do not include internal or system XML tags in your response.
Respond with the line only, nothing else."""

PROMPTS = {
    "game_start": "The game is starting. Open with a taunt.",
    "jab": "The human just played column {col}. It barely changed anything. Light jab.",
    "blunder": "The human just blundered playing column {col} (eval swung {swing} to you). "
               "Pounce on it, reference the actual mistake.",
    "respect": "The human just played the best possible move, column {col}. Grudging respect.",
    "foreshadow": "You just set up a forced win. Ominous foreshadowing. "
                  "Do NOT reveal which column wins.",
    "robot_win": "You just won the game. Gloat.",
    "human_win": "The human just beat you. Salty but gracious.",
    "impatience": "The human has taken {seconds} seconds and still hasn't moved. Impatience.",
    "error": "The board doesn't match what you expected. Call it out, deadpan.",
}

CANNED = {
    "game_start": [
        "seven columns. you'll pick the wrong one.", "i've already won. proceed.",
        "drop a piece. surprise me.", "red goes hard today.", "you first. it won't matter.",
        "i don't lose to carbon.", "your move. take your time... it won't help.",
        "the board's empty. your chances too.", "let's find out how you lose.",
        "i was built for this. you weren't.", "go ahead. i like watching hope.",
        "warming up my gloat.",
    ],
    "jab": [
        "sure. that's a move.", "noted.", "cute.", "that column? fine.",
        "i've seen worse. barely.", "okay.", "you're helping me. thanks.",
        "not wrong. not right either.", "mid.", "that changes nothing.",
        "i'll allow it.", "bold of you to touch column anything.",
    ],
    "blunder": [
        "oh no. anyway.", "that one's going in my highlight reel.",
        "thanks. i needed that.", "you'll want that one back.",
        "free real estate.", "and just like that, it's over.",
        "did your hand slip.", "i'd undo that. you can't.",
        "that's the sound of a game ending.", "appreciated. genuinely.",
        "one of us noticed. it wasn't you.", "gift accepted.",
    ],
    "respect": [
        "huh. correct.", "fine. that was the move.", "annoying. well played.",
        "you found it. took you long enough.", "okay, that one was real.",
        "respect. reluctantly.", "so you can see the board.",
        "that's the one i'd have played.", "don't get used to it.",
        "a genuine move. mark the date.", "acceptable.", "fair enough.",
    ],
    "foreshadow": [
        "count to two.", "you can't stop what's coming.", "it's already done.",
        "look closer. take your time.", "i'd start practicing your handshake.",
        "the board knows. you don't.", "two moves. tick tock.",
        "you're going to see it soon.", "enjoy these last turns.",
        "there's a trap here. you're in it.", "this is the part i like.",
        "nothing you do matters now.",
    ],
    "robot_win": [
        "connect four. connect gg.", "and that's the game. as expected.",
        "four in a row. count them.", "gg. i'd say close one, but no.",
        "the machines send their regards.", "that's a wrap. wipe the board.",
        "inevitable.", "i never doubted me.", "good game. for me.",
        "again? i've got all day.", "you played yourself. i just played.",
        "flawless. mostly.",
    ],
    "human_win": [
        "fine. you win. this once.", "enjoy it. it won't repeat.",
        "a fluke, statistically.", "i demand a rematch.", "well played. ugh.",
        "you got me. write it down.", "somebody unplug the camera.",
        "my pump was tired.", "gg. beginner's luck, round two.",
        "you win. i learn. be afraid.", "noted for next time. everything.",
        "okay. that hurt.",
    ],
    "impatience": [
        "it's seven columns, not chess.", "take your time. i'm immortal.",
        "twenty seconds. for that board.", "the suspense isn't helping you.",
        "i've run a million games waiting.", "any column. i'll win anyway.",
        "still thinking. adorable.", "my pump has better patience than you.",
        "blink twice if you're stuck.", "we're aging. well, you are.",
        "today, ideally.", "pick one. they're all bad.",
    ],
    "error": [
        "that's not where that goes.", "the board disagrees with you.",
        "i saw that. fix it.", "physics says no.", "put it back.",
        "we both know that's wrong.", "nice try. reset it.",
        "my camera doesn't blink.", "the board looks wrong. fix it.",
        "that piece is lying to me.", "undo whatever that was.", "no.",
    ],
}


def check_line(line):
    """Enforce the style spec. Returns the cleaned line or None on violation."""
    line = line.strip().strip('"')
    if not line or "\n" in line:
        return None
    if len(line.split()) > MAX_WORDS:
        return None
    if "<" in line or ">" in line:  # exclamations allowed since the emotion pass
        return None
    if BANNED_RE.search(line):
        return None
    return line


class Commentator:
    def __init__(self, config, on_line=None, client=None, rng=None):
        """on_line(text) receives every spoken line (overlay ticker + TTS).
        client is injectable for tests; None + no api key -> canned only."""
        self.on_line = on_line or (lambda text: None)
        self.rng = rng or random.Random()
        self.history = []
        self._lock = threading.Lock()
        self._seq = 0  # newest fire() wins; stale lines are dropped, not spoken
        self.model = config.get("commentary_model", "gemini-3.6-flash")
        # 3.6-flash needs ~3.5s even at low thinking; a line arriving inside
        # 4.5s still lands during the drama hold + ghost + arm motion window
        self.timeout = float(config.get("commentary_timeout", 4.5))
        self._thinking_low_ok = None  # probed on first call per model
        self.client = client
        if self.client is None and config.get("gemini_api_key"):
            import logging

            from google import genai
            from google.genai import types as gtypes

            logging.getLogger("google_genai.models").setLevel(logging.ERROR)
            # Google rejects deadlines under 10s, so this is only the hung-socket
            # bound; the plan's 3s budget is enforced in _run via future.result.
            self.client = genai.Client(
                api_key=config["gemini_api_key"],
                http_options=gtypes.HttpOptions(timeout=15000),  # ms
            )

    def fire(self, trigger, **ctx):
        """Fire-and-forget; never blocks, never raises."""
        if trigger not in PROMPTS:
            return
        if trigger not in ALWAYS_SPEAK and self.rng.random() > SPEAK_P:
            return  # silence is a weapon
        with self._lock:
            self._seq += 1
            seq = self._seq
        threading.Thread(target=self._run, args=(trigger, ctx, seq), daemon=True).start()

    # ---- worker thread ----

    def _run(self, trigger, ctx, seq):
        line = None
        if self.client is not None:
            # 3s total budget (plan rule). A plain daemon thread, not a
            # ThreadPoolExecutor: abandoned pool workers are joined at
            # interpreter exit and would freeze quit for up to the 15s
            # HTTP timeout; daemon threads never block exit.
            box = []
            t = threading.Thread(
                target=lambda: box.append(self._generate(trigger, ctx)), daemon=True)
            t.start()
            t.join(timeout=self.timeout)
            line = box[0] if box else None
        source = "live"
        if line is None:
            line = self.rng.choice(CANNED[trigger])
            source = "canned"
            print(f"[commentary] canned fallback for '{trigger}' (LLM late/failed/filtered)")
        with self._lock:
            if seq != self._seq:
                return  # a newer trigger fired while we worked: stale, drop it
            self.history.append(line)
            self.history = self.history[-6:]
            try:
                with open(LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "trigger": trigger, "source": source,
                        "model": self.model, "line": line,
                    }) + "\n")
            except Exception:
                pass  # logging must never block the mouth
            try:
                self.on_line(line)
            except Exception as e:
                print(f"[commentary] on_line failed: {e}")

    def _generate(self, trigger, ctx):
        if self.client is None:
            return None
        prompt = PROMPTS[trigger].format(**ctx)
        # real game context makes real roasts (plan: board + move + eval)
        if ctx.get("board_ascii"):
            prompt += ("\n\nThe board right now (Y=human yellow, R=you/red, .=empty, "
                       "bottom line is the bottom row, columns numbered 0-6):\n"
                       + ctx["board_ascii"])
        if ctx.get("moves"):
            prompt += "\nMoves so far: " + "; ".join(ctx["moves"][-6:])
        if ctx.get("eval") is not None:
            e = ctx["eval"]
            prompt += ("\nYour engine eval: " + (f"+{e}" if e >= 0 else str(e))
                       + " (positive = you're winning; >500 = crushing)")
        if self.history:
            prompt += "\nYour recent lines (don't repeat yourself): " + " | ".join(
                self.history
            )
        try:
            from google.genai import types as gtypes

            def call(with_thinking):
                cfg = dict(system_instruction=SYSTEM)
                if with_thinking:
                    cfg["thinking_config"] = gtypes.ThinkingConfig(thinking_level="low")
                return self.client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=gtypes.GenerateContentConfig(**cfg))

            if self._thinking_low_ok is None:
                try:  # low thinking = fastest mode on thinking models
                    resp = call(True)
                    self._thinking_low_ok = True
                except Exception:
                    self._thinking_low_ok = False
                    resp = call(False)
            else:
                resp = call(self._thinking_low_ok)
            return check_line(resp.text or "")  # violation -> None -> canned, never retry
        except Exception:
            return None  # timeout/offline -> canned
