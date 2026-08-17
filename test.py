"""One-file MaxArm test. Edit PORT below, then:  python test_maxarm.py

Requires: pip install pyserial
The arm must be running HiWonder's MaxArm_Serial_Communication firmware
(microUSB variant) from section 10 of their docs.

What it does, in order:
  1. reads current XYZ and servo angles (proves 2-way comms)
  2. turns the suction pump on for 2s, then releases
  3. moves to two coordinates and back
"""

import struct
import time

import serial

PORT = "/dev/cu.usbserial-310"          # <-- change me ("/dev/ttyUSB0" on Linux/Mac)
BAUD = 9600

# protocol: AA 55 | func | len | data | checksum(~sum & 0xFF)
SET_ANGLE, SET_XYZ, SET_PWM, SUCTION, READ_ANGLE, READ_XYZ = (
    0x01, 0x03, 0x05, 0x07, 0x11, 0x13,
)


def checksum(payload: bytes) -> int:
    return (~sum(payload)) & 0xFF


class MaxArm:
    def __init__(self, port, baud=BAUD):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(0.1)
        self.ser.reset_input_buffer()

    def _send(self, func, data=b""):
        payload = bytes([func, len(data)]) + data
        self.ser.write(b"\xaa\x55" + payload + bytes([checksum(payload)]))

    def _read6(self, func):
        """Read reply frame AA 55 func 06 <6 bytes> check -> the 6 data bytes."""
        raw = self.ser.read(15)
        i = raw.find(b"\xaa\x55")
        if i < 0 or len(raw) < i + 11:
            return None
        f = raw[i : i + 11]
        # reply checksum covers the AA 55 header too (unlike requests)
        if f[2] != func or f[3] != 6 or checksum(f[:10]) != f[10]:
            return None
        return f[4:10]

    # movement (x, y, z in mm; the firmware does the inverse kinematics)
    def set_xyz(self, x, y, z, ms=1000):
        self._send(SET_XYZ, struct.pack("<hhhH", x, y, z, ms))

    # suction
    def pump_on(self):
        self._send(SUCTION, b"\x01")

    def release(self):
        self._send(SUCTION, b"\x02")   # vent valve (drops the piece)
        time.sleep(0.2)
        self._send(SUCTION, b"\x03")   # close valve

    def wait_ready(self, timeout=25):
        """Poll until the arm answers. After a reset the firmware sleeps 10s
        (reflash window) and then homes the arm, so first contact can take ~15s."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            xyz = self.read_xyz()
            if xyz is not None:
                return xyz
        return None

    # feedback
    def read_xyz(self):
        self.ser.reset_input_buffer()
        self._send(READ_XYZ)
        time.sleep(0.1)
        d = self._read6(READ_XYZ)
        return struct.unpack("<hhh", d) if d else None

    def read_angles(self):
        self.ser.reset_input_buffer()
        self._send(READ_ANGLE)
        time.sleep(0.1)
        d = self._read6(READ_ANGLE)
        return struct.unpack("<hhh", d) if d else None


if __name__ == "__main__":
    arm = MaxArm(PORT)

    print("1) reading position (up to ~25s if the arm is still booting)...")
    xyz = arm.wait_ready()
    print("   xyz:", xyz)
    print("   angles:", arm.read_angles())
    if xyz is None:
        raise SystemExit(
            "No reply from the arm. Check: right port? serial firmware flashed "
            "(microUSB variant)? nothing else holding the port open?"
        )

    print("2) suction test: pump on 2s, then release")
    arm.pump_on()
    time.sleep(2)
    arm.release()
    time.sleep(1)

    print("3) motion test (coordinates from HiWonder's own example)")
    arm.set_xyz(120, -180, 85, 1000)
    time.sleep(1.5)
    arm.set_xyz(-120, -180, 85, 1000)
    time.sleep(1.5)
    arm.set_xyz(*xyz, 1000)  # back to where it started
    time.sleep(1.5)

    print("done. final position:", arm.read_xyz())