"""MaxArm serial driver.

Protocol facts below were validated against the physical arm on 2026-08-17
(HiWonder MaxArm_micropython_microUSB firmware, installed by us; factory
files backed up in maxarm_factory_backup/):

  - 9600 baud on /dev/cu.usbserial-310. A plain port open does NOT reset the
    board; only a DTR/RTS pulse does.
  - Request frame:  AA 55 | func | len | data | ~sum(func,len,data) & 0xFF
  - Reply frame:    AA 55 | func | 06  | data6 | checksum over ALL preceding
    bytes INCLUDING the AA 55 header (off by one vs the request convention).
  - After a reset/power-on the firmware sleeps 10 s (reflash window) then
    homes the arm (~3 s) before it answers: poll for up to ~25 s.
  - Suction data byte: 1 = pump on, 2 = pump off + vent valve, 3 = valve close.

Pick/place cycle (pick_and_drop): hover over feeder -> down -> pump on ->
up -> over target column -> release -> home. Poses come from poses.json,
taught on the physical arm with jog.py. The robot plays RED pieces.
"""

import glob
import json
import struct
import time
from pathlib import Path

import serial

DEFAULT_PORT = "/dev/cu.usbserial-310"
BAUD = 9600
POSES_PATH = Path(__file__).parent / "poses.json"

SET_ANGLE, SET_XYZ, SET_PWM, SUCTION, READ_ANGLE, READ_XYZ = (
    0x01, 0x03, 0x05, 0x07, 0x11, 0x13,
)

PUMP_ON, PUMP_VENT, VALVE_CLOSE = 1, 2, 3


def checksum(payload: bytes) -> int:
    return (~sum(payload)) & 0xFF


class UncalibratedError(RuntimeError):
    """poses.json still holds placeholder coordinates — run jog.py first."""


def find_port(preferred=DEFAULT_PORT):
    """The configured port if it exists, else the sole usbserial device.
    The arm can enumerate under a different number on a different USB socket."""
    if preferred and Path(preferred).exists():
        return preferred
    candidates = sorted(glob.glob("/dev/cu.usbserial-*"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"no MaxArm serial port found (looked for {preferred} and "
            "/dev/cu.usbserial-*) — is the arm plugged in and powered?")
    raise SystemExit(f"several usbserial ports: {candidates} — set `port` in config.yaml")


class MaxArm:
    def __init__(self, port=DEFAULT_PORT, baud=BAUD, poses_path=POSES_PATH, ser=None):
        self.ser = ser if ser is not None else serial.Serial(find_port(port), baud, timeout=1)
        self._sleep = time.sleep  # injectable for tests
        self.poses_path = Path(poses_path)
        self.poses = json.loads(self.poses_path.read_text())
        self._workspace_confirmed = False
        self._sleep(0.1)
        self.ser.reset_input_buffer()

    # ---- protocol ----

    def _send(self, func, data=b""):
        payload = bytes([func, len(data)]) + data
        self.ser.write(b"\xaa\x55" + payload + bytes([checksum(payload)]))

    def _read6(self, func):
        raw = self.ser.read(15)
        i = raw.find(b"\xaa\x55")
        if i < 0 or len(raw) < i + 11:
            return None
        f = raw[i : i + 11]
        # reply checksum covers the AA 55 header too (unlike requests)
        if f[2] != func or f[3] != 6 or checksum(f[:10]) != f[10]:
            return None
        return f[4:10]

    def _query(self, func):
        self.ser.reset_input_buffer()
        self._send(func)
        self._sleep(0.1)
        d = self._read6(func)
        return struct.unpack("<hhh", d) if d else None

    def read_xyz(self):
        return self._query(READ_XYZ)

    def read_angles(self):
        return self._query(READ_ANGLE)

    def wait_ready(self, timeout=25):
        """Poll until the arm answers. After a reset the firmware sleeps 10 s
        (reflash window) then homes, so first contact can take ~15 s."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            xyz = self.read_xyz()
            if xyz is not None:
                return xyz
        return None

    # ---- motion primitives ----

    def set_xyz(self, x, y, z, ms=1000):
        self._send(SET_XYZ, struct.pack("<hhhH", int(x), int(y), int(z), int(ms)))

    def pump_on(self):
        self._send(SUCTION, bytes([PUMP_ON]))

    def release(self):
        self._send(SUCTION, bytes([PUMP_VENT]))
        self._sleep(0.2)
        self._send(SUCTION, bytes([VALVE_CLOSE]))

    def move_to(self, xyz, ms=None, settle=0.2):
        """Blocking move: command, wait the move duration plus a settle margin."""
        ms = ms if ms is not None else self.poses.get("speed_ms", 1200)
        self.set_xyz(*xyz, ms)
        self._sleep(ms / 1000 + settle)

    # ---- poses / safety ----

    def confirm_workspace(self):
        """Ask once per session before the first physical motion (CLAUDE.md rule)."""
        if self._workspace_confirmed:
            return
        ans = input("About to MOVE THE ARM. Workspace clear of hands/objects? [y/N] ")
        if ans.strip().lower() != "y":
            raise SystemExit("aborted: workspace not confirmed clear")
        self._workspace_confirmed = True

    def _require_calibrated(self):
        if not self.poses.get("calibrated"):
            raise UncalibratedError(
                "poses.json holds PLACEHOLDER coordinates. Run jog.py on the "
                "physical rig to teach feeder/col0-col6/home before playing."
            )

    def go_home(self, ms=None):
        self.confirm_workspace()
        self.move_to(self.poses["home"], ms)

    # ---- the game-facing cycle ----

    def _move_verified(self, xyz, ms=700):
        """Move and confirm arrival via servo feedback (False = refused/blocked)."""
        self.set_xyz(*xyz, ms)
        self._sleep(ms / 1000 + 0.3)
        real = self.read_xyz()
        return real is not None and max(
            abs(real[i] - round(xyz[i])) for i in range(3)) <= 8

    def pick_and_drop(self, col: int, pick_index: int = 0):
        """Feeder -> suction on -> lift -> over column `col` -> release -> home.

        Blocking; returns once the arm is back at home (safe to rescan).
        This is the motion pattern proven on the rig 2026-08-20 (run_test.py):
        staged slow descents at both ends, drop approached from a few mm above
        the taught slot (never transiting below slot height), settled release.

        pick_index: pieces already taken this game; the pick sinks by
        poses["piece_thickness"] per piece (feeder_pick = top of a FULL stack).
        """
        if not 0 <= col <= 6:
            raise ValueError(f"column out of range: {col}")
        self._require_calibrated()
        self.confirm_workspace()

        p = self.poses
        ztr = p["travel_z"]                       # pickup-side transit height
        pick = list(p["feeder_pick"])
        pick[2] = round(pick[2] - pick_index * p.get("piece_thickness", 0))
        drop = p["columns"][col]

        self.move_to([pick[0], pick[1], ztr])     # over the feeder
        self.move_to([pick[0], pick[1], pick[2] + 20])  # just above the piece
        self.move_to(pick, ms=700)                # short slow straight descent
        self.pump_on()
        self._sleep(0.6)                          # let the seal form
        self.move_to([pick[0], pick[1], ztr])     # lift with the piece
        # approach the slot from above; taught slots can sit ABOVE travel height
        placed = False
        for dz in (12, 6, 0):
            if self._move_verified([drop[0], drop[1], drop[2] + dz]):
                placed = True
                break
        if placed and dz > 0:
            self.move_to(drop, ms=800)            # slow final onto the slot
        self._sleep(0.5)                          # settle so nothing swings
        self.release()
        self._sleep(0.4)
        if placed:
            self.move_to([drop[0], drop[1], drop[2] + max(dz, 6)], ms=700)
        self.move_to(p["home"])                   # out of the camera's view
        return True
