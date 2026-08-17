"""Pose teacher for the MaxArm. Run on the physical rig before a demo.

Jog the arm with the keyboard, park it where you want, and stamp that
position into a named pose slot. Saves to poses.json and flips
"calibrated" to true so maxarm.pick_and_drop() will run.

Keys:
  w/s   y away / toward you        1/2/3  step size 2 / 10 / 25 mm
  a/d   x left / right             p      print current position (from arm)
  r/f   z up / down                o      suction toggle (pump on / release)
  h     save HOME here             e      save FEEDER PICK here
  0-6   save that column's drop pose here
  t     set travel height (transit z) to current z
  g     cycle: go to each saved pose (visual check)
  y     full pick_and_drop test into a column you type
  q     save poses.json and quit   Q      quit WITHOUT saving
"""

import json
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
    print("waiting for arm (up to 25s if it just powered on)...")
    xyz = arm.wait_ready()
    if xyz is None:
        raise SystemExit("no reply from the arm — check port/power")
    arm.confirm_workspace()

    pos = list(xyz)
    poses = arm.poses
    step = 10
    pumping = False
    print(f"arm at {pos}. step {step}mm. keys: see docstring. q saves+quits.")

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
            elif k in "wsadrf":
                dx = {"a": -step, "d": step}.get(k, 0)
                dy = {"s": -step, "w": step}.get(k, 0)
                dz = {"f": -step, "r": step}.get(k, 0)
                pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
                arm.set_xyz(*pos, 250)
                time.sleep(0.28)
            elif k == "p":
                real = arm.read_xyz()
                print(f"target {pos}  arm reports {real}")
                if real:
                    pos = list(real)
            elif k == "o":
                pumping = not pumping
                arm.pump_on() if pumping else arm.release()
                print("pump ON" if pumping else "released")
            elif k == "h":
                poses["home"] = list(pos)
                print(f"home = {pos}")
            elif k == "e":
                poses["feeder_pick"] = list(pos)
                print(f"feeder_pick = {pos}")
            elif k in "0123456":
                poses["columns"][int(k)] = list(pos)
                print(f"col{k} drop = {pos}")
            elif k == "t":
                poses["travel_z"] = pos[2]
                print(f"travel_z = {pos[2]}")
            elif k == "g":
                print("touring saved poses...")
                for name in ["home", "feeder_pick"] + [f"col{i}" for i in range(7)]:
                    p = poses["columns"][int(name[3])] if name.startswith("col") else poses[name]
                    print(f"  -> {name} {p}")
                    arm.move_to(p)
                pos = list(p)
            elif k == "y":
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
                col = int(input("test drop into column (0-6): "))
                poses["calibrated"] = True  # trust the operator mid-session
                arm.pick_and_drop(col)
                pos = list(poses["home"])
                tty.setcbreak(sys.stdin.fileno())
            elif k == "q":
                poses["calibrated"] = True
                POSES_PATH.write_text(json.dumps(poses, indent=2) + "\n")
                print(f"saved {POSES_PATH} (calibrated=true)")
                break
            elif k == "Q":
                print("quit without saving")
                break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
