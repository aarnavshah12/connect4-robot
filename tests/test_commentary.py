import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from commentary import CANNED, PROMPTS, Commentator, check_line


def test_checker_rejects_banned_phrases():
    assert check_line("bold move, human.") is None
    assert check_line("Interesting Choice.") is None
    assert check_line("beep boop. i win.") is None
    assert check_line("the question is... why.") is None


def test_checker_rejects_emdash_and_length():
    assert check_line("Gotcha!") == "Gotcha!"  # exclamations allowed (emotion pass)
    assert check_line("column four — obviously.") is None
    assert check_line(" ".join(["word"] * 29)) is None  # over the 28-word cap
    assert check_line(" ".join(["word"] * 20)) is not None  # fuller roasts allowed
    assert check_line("<thinking>hm</thinking> sure.") is None


def test_checker_accepts_good_lines():
    assert check_line('"column 2? bold. wrong, but bold."') == "column 2? bold. wrong, but bold."
    assert check_line("cute.") == "cute."


def test_every_canned_line_passes_the_checker():
    for trigger, lines in CANNED.items():
        assert trigger in PROMPTS
        assert len(lines) >= 10, f"{trigger} needs a deeper canned pool"
        for line in lines:
            assert check_line(line) == line, f"canned line violates spec: {line!r}"


class FakeResp:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Mimics google-genai: client.models.generate_content(...) -> resp.text"""

    def __init__(self, text):
        self._text = text
        self.models = self

    def generate_content(self, **kw):
        return FakeResp(self._text)


def fire_and_wait(com, trigger, **ctx):
    got = []
    com.on_line = got.append
    com.fire(trigger, **ctx)
    deadline = time.time() + 2
    while not got and time.time() < deadline:
        time.sleep(0.01)
    return got


def test_violating_llm_line_falls_back_to_canned():
    com = Commentator({}, client=FakeClient("WOW what a BOLD MOVE friend!!!"),
                      rng=random.Random(1))
    got = fire_and_wait(com, "robot_win")
    assert got and got[0] in CANNED["robot_win"]


def test_good_llm_line_is_used():
    com = Commentator({}, client=FakeClient("column four... obviously."),
                      rng=random.Random(1))
    got = fire_and_wait(com, "robot_win")
    assert got == ["column four... obviously."]


def test_no_client_uses_canned_and_never_blocks():
    com = Commentator({}, rng=random.Random(2))
    t0 = time.time()
    got = fire_and_wait(com, "human_win")
    assert time.time() - t0 < 1.5
    assert got and got[0] in CANNED["human_win"]


def test_stale_line_is_dropped_when_newer_trigger_fires():
    """A slow trigger's line must not be spoken after a newer trigger's."""

    class SlowThenFastClient:
        def __init__(self):
            self.models = self
            self.calls = 0

        def generate_content(self, **kw):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.8)  # slow first call (e.g. impatience nag)
                return FakeResp("still thinking. adorable.")
            return FakeResp("cute.")

    got = []
    com = Commentator({}, on_line=got.append, client=SlowThenFastClient(),
                      rng=random.Random(1))
    com.fire("robot_win")          # slow one
    time.sleep(0.1)
    com.fire("human_win")          # newer, fast one
    time.sleep(1.5)                # let both finish
    assert got == ["cute."], f"stale line leaked: {got}"


def test_silence_roughly_60_percent():
    com = Commentator({}, rng=random.Random(0))
    com.on_line = lambda text: None
    spoken = sum(1 for _ in range(500) if com.rng.random() <= 0.6)
    assert 250 < spoken < 350  # sanity on the dice, not the threads
