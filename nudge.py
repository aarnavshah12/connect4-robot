"""Minimal feeder-calibration pad. Directions are YOUR view of the arm:
left/right swing along the arc it rotates on, up/down is pure height.

    .venv/bin/python nudge.py

  a = left        d = right        (arc steps, distance stays the same)
  w = up          s = down
  i = out (away from base)   k = in (toward base)
  1 / 2 / 3 = step size 2 / 10 / 25 mm
  x = suction on/off (test the grab: seat the cup, x, then w to lift)
  p = print position          e = SAVE this spot as the pickup pose
  q = quit

Refused moves (out of reach) are reported and the tool stays in sync.
"""

import json
import math
import select
import sys
import termios
import time
import tty

from maxarm import POSES_PATH, MaxArm

STEPS = {"1": 2, "2": 10, "3": 25}


def getch(timeout=0.05):
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def main():
    arm = MaxArm()
    print("waiting for arm...")
    xyz = arm.wait_ready()
    if xyz is None:
        raise SystemExit("no reply from the arm — powered? jog/nudge already running?")
    arm.confirm_workspace()

    pos = list(xyz)
    step = 10
    pumping = False
    print(f"arm at {pos}. step {step}mm. a/d=left/right w/s=up/down i/k=out/in "
          f"x=suction e=save pickup q=quit")

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            k = getch()
            if k is None:
                continue
            if k in STEPS:
                step = STEPS[k]
                print(f"step {step}mm")
            elif k in "adwsik":
                x, y, z = pos
                r, th = math.hypot(x, y), math.atan2(y, x)
                if k in "ad":                      # arc step at constant radius
                    dth = (step / max(r, 1)) * (-1 if k == "a" else 1)
                    x, y = r * math.cos(th + dth), r * math.sin(th + dth)
                elif k in "ws":
                    z += step if k == "w" else -step
                else:                              # radial: out/in
                    r += step if k == "i" else -step
                    x, y = r * math.cos(th), r * math.sin(th)
                target = [round(x), round(y), round(z)]
                arm.set_xyz(*target, 300)
                time.sleep(0.35)
                real = arm.read_xyz()
                if real is None:
                    print(f"no reply — still at {pos}?")
                elif max(abs(real[i] - target[i]) for i in range(3)) > 8:
                    print(f"can't go there (limit) — at {list(real)}")
                    pos = list(real)
                else:
                    pos = list(real)
            elif k == "x":
                pumping = not pumping
                arm.pump_on() if pumping else arm.release()
                print("suction ON" if pumping else "released")
            elif k == "p":
                real = arm.read_xyz()
                print(f"position: {list(real) if real else pos}")
                if real:
                    pos = list(real)
            elif k == "e":
                real = arm.read_xyz() or tuple(pos)
                arm.poses["feeder_pick"] = list(real)
                POSES_PATH.write_text(json.dumps(arm.poses, indent=2) + "\n")
                print(f"SAVED pickup pose: {list(real)}")
            elif k == "q":
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        if pumping:
            arm.release()


if __name__ == "__main__":
    main()
