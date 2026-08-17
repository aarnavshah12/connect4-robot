"""speak(text): TTS behind one interface (plan §voice pipeline).

- Engine picked by config: "say" (macOS, offline, default) or "elevenlabs".
- Speech runs on its own queue thread, depth 1: a new line replaces a
  pending one (stale trash talk is worse than none); the line currently
  playing is never interrupted.
- ElevenLabs audio is cached by line hash so repeats don't re-bill.
"""

import hashlib
import queue
import subprocess
import threading
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "audio_cache"


class Voice:
    def __init__(self, config):
        self.cfg = config
        self.engine = config.get("voice", "say")
        self._q = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._el_client = None
        if self.engine == "elevenlabs" and config.get("elevenlabs_api_key"):
            from elevenlabs.client import ElevenLabs

            self._el_client = ElevenLabs(api_key=config["elevenlabs_api_key"])
            CACHE_DIR.mkdir(exist_ok=True)

    def speak(self, text):
        """Fire-and-forget. Replaces any line still waiting in the queue.
        Loops because the player thread can dequeue between our put and get —
        the new line must always land, never be silently dropped."""
        while True:
            try:
                self._q.put_nowait(text)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass  # consumer beat us to it; queue is now free

    def _loop(self):
        while True:
            text = self._q.get()
            try:
                self._play(text)
            except Exception as e:  # a TTS hiccup must never kill the thread
                print(f"[voice] {e}")

    def _play(self, text):
        if self._el_client is not None:
            path = CACHE_DIR / (hashlib.sha1(text.encode()).hexdigest() + ".mp3")
            if not path.exists():
                audio = self._el_client.text_to_speech.convert(
                    voice_id=self.cfg.get("elevenlabs_voice_id"),
                    text=text,
                    model_id="eleven_multilingual_v2",  # expressive > low-latency (plan)
                )
                path.write_bytes(b"".join(audio))
            subprocess.run(["afplay", str(path)], check=False)
        else:
            subprocess.run(["say", "-v", "Zarvox", text], check=False)
