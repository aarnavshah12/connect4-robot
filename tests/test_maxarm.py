"""Driver tests against a fake serial port — byte-level protocol as validated
on the physical arm 2026-08-17 (see maxarm.py docstring)."""

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from maxarm import READ_XYZ, SET_XYZ, SUCTION, MaxArm, checksum


class FakeSerial:
    def __init__(self):
        self.written = b""
        self.reply = b""

    def write(self, data):
        self.written += data

    def read(self, n):
        out, self.reply = self.reply[:n], self.reply[n:]
        return out

    def reset_input_buffer(self):
        pass


def firmware_reply(func, data6: bytes) -> bytes:
    """Build a reply exactly the way the MicroPython firmware does:
    checksum over the WHOLE frame including the AA 55 header."""
    frame = b"\xaa\x55" + bytes([func, 6]) + data6
    return frame + bytes([checksum(frame)])


@pytest.fixture
def arm(tmp_path):
    poses = {
        "calibrated": True,
        "speed_ms": 100,
        "travel_z": 160,
        "home": [0, -140, 160],
        "feeder_pick": [140, -120, 85],
        "columns": [[-120 + 40 * i, -180, 150] for i in range(7)],
    }
    p = tmp_path / "poses.json"
    p.write_text(json.dumps(poses))
    a = MaxArm(poses_path=p, ser=FakeSerial())
    a._sleep = lambda s: None
    a._workspace_confirmed = True
    return a


def test_set_xyz_frame_bytes(arm):
    arm.ser.written = b""
    arm.set_xyz(120, -180, 85, 1000)
    # known-good frame from HiWonder's own docs example
    assert arm.ser.written == bytes.fromhex("aa5503087800 4cff 5500 e803 f1".replace(" ", ""))


def test_read_xyz_accepts_firmware_checksum(arm):
    arm.ser.reply = firmware_reply(READ_XYZ, struct.pack("<hhh", -1, -162, 212))
    assert arm.read_xyz() == (-1, -162, 212)


def test_read_xyz_rejects_request_style_checksum(arm):
    # a reply checksummed WITHOUT the header (the request convention) is invalid
    body = bytes([READ_XYZ, 6]) + struct.pack("<hhh", 1, 2, 3)
    arm.ser.reply = b"\xaa\x55" + body + bytes([checksum(body)])
    assert arm.read_xyz() is None


def test_read_xyz_skips_leading_noise(arm):
    arm.ser.reply = b"\x00\x7f" + firmware_reply(READ_XYZ, struct.pack("<hhh", 5, 6, 7))
    assert arm.read_xyz() == (5, 6, 7)


def test_pick_and_drop_sequence(arm):
    arm.ser.written = b""
    arm.pick_and_drop(3)
    w = arm.ser.written
    # suction: pump on happens before vent+close
    on = w.find(bytes([0xAA, 0x55, SUCTION, 1, 1]))
    vent = w.find(bytes([0xAA, 0x55, SUCTION, 1, 2]))
    close = w.find(bytes([0xAA, 0x55, SUCTION, 1, 3]))
    assert 0 < on < vent < close
    # motion targets in order: hover, pick, hover, transit, drop, transit, home
    moves = []
    i = 0
    while (i := w.find(bytes([0xAA, 0x55, SET_XYZ]), i)) != -1:
        moves.append(struct.unpack("<hhhH", w[i + 4 : i + 12])[:3])
        i += 1
    assert moves == [
        (140, -120, 160), (140, -120, 85), (140, -120, 160),
        (0, -180, 160), (0, -180, 150), (0, -180, 160), (0, -140, 160),
    ]
    # the pickup happens between pump-on being sent and the lift
    assert on < w.find(bytes([0xAA, 0x55, SET_XYZ]), on)


def test_pick_and_drop_refuses_placeholder_poses(arm):
    arm.poses["calibrated"] = False
    with pytest.raises(Exception, match="jog.py"):
        arm.pick_and_drop(0)


def test_pick_and_drop_rejects_bad_column(arm):
    with pytest.raises(ValueError):
        arm.pick_and_drop(7)
