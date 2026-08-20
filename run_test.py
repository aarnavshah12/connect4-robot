"""Feeder/board test harness — run it yourself, Ctrl+C is your pause button.

    .venv/bin/python run_test.py --spot 1 --pieces 7    # 7 pieces into spot 1
    .venv/bin/python run_test.py --all                  # one piece into each of the 7 spots
    .venv/bin/python run_test.py --spot 4 --pieces 2    # 2 pieces into spot 4

Uses poses.json: feeder_pick (top-of-full-stack grab), piece_thickness
(pick sinks per piece), columns[0..6] (spots 1..7, taught 2026-08-20).
"""

import argparse
import json
import time
from pathlib import Path

from maxarm import MaxArm

ZTR = 150  # transit height on the pickup side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", type=int, help="spot number 1-7 (all pieces go here)")
    ap.add_argument("--pieces", type=int, default=7, help="how many pieces (default 7)")
    ap.add_argument("--all", action="store_true", help="one piece into each spot 1..7")
    args = ap.parse_args()
    if not args.all and not args.spot:
        ap.error("pick --spot N or --all")

    poses = json.loads((Path(__file__).parent / "poses.json").read_text())
    px, py, pz0 = poses["feeder_pick"]
    thick = poses.get("piece_thickness", 6.5)
    cols = poses["columns"]

    if args.all:
        plan = [(i, cols[i]) for i in range(7)]
    else:
        plan = [(args.spot - 1, cols[args.spot - 1])] * args.pieces

    arm = MaxArm()
    print("waiting for arm...")
    if arm.wait_ready() is None:
        raise SystemExit("no reply — is nudge/jog still open, or arm off?")
    arm.confirm_workspace()
    x, y, z = arm.read_xyz()

    def goto(tx, ty, tz, ms=700, label=""):
        arm.set_xyz(round(tx), round(ty), round(tz), ms)
        time.sleep(ms / 1000 + 0.3)
        real = arm.read_xyz()
        if label:
            print(f"    {label}: -> {real}")
        ok = real is not None and max(
            abs(real[i] - (round(tx), round(ty), round(tz))[i]) for i in range(3)) <= 8
        return ok

    def above_col(col):
        for dz in (12, 6, 0):
            if goto(col[0], col[1], col[2] + dz, 700, f"above spot (+{dz})"):
                return dz
        return None

    goto(x, y, min(z + 25, 200), 600, "clear upward")
    goto(px, py, ZTR, 900, "to pickup side")

    try:
        for n, (ci, col) in enumerate(plan):
            zpick = round(pz0 - thick * n)
            print(f"--- piece {n + 1}/{len(plan)} -> spot {ci + 1}  (pick z={zpick})")
            goto(px, py, zpick + 20, 700, "above the piece")
            if not goto(px, py, zpick, 700, f"slow down to z={zpick}"):
                print("    pick unreachable — stopping")
                break
            arm.pump_on(); time.sleep(0.6)
            goto(px, py, ZTR, 800, "lift with piece")
            dz = above_col(col)
            if dz is None:
                print("    can't get above the spot — releasing here")
                arm.release(); time.sleep(0.4)
                goto(px, py, ZTR, 800, "back to pickup side")
                continue
            if dz > 0:
                goto(*col, 800, "SLOW final to the spot")
            time.sleep(0.5)
            arm.release(); time.sleep(0.4)
            print(f"    released at spot {ci + 1}")
            goto(col[0], col[1], col[2] + max(dz, 6), 700, "up off the slot")
            goto(px, py, ZTR, 900, "back to pickup side")
        print("done — parked on the pickup side")
    except KeyboardInterrupt:
        print("\npaused by you — lifting clear and stopping")
        arm.release()
        xx, yy, zz = arm.read_xyz() or (px, py, ZTR)
        goto(xx, yy, min(zz + 30, 200), 800)


if __name__ == "__main__":
    main()
