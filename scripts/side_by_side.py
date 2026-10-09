"""Headline video: my phone clip | Panda | G1, time-aligned, same demo.

Needs outputs/perception/<demo>.mp4 (from perceive.py). The sim replays use the
torque-rate retargeting; the robots' 1 s approach from home is cut so t=0 matches.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.config import LAMBDA_OFFLINE, Measurements  # noqa: E402
from ego2sim.demo import HumanDemo  # noqa: E402
from ego2sim.deploy import set_layout  # noqa: E402
from ego2sim.execute import run  # noqa: E402
from ego2sim.paths import DATA, OUT  # noqa: E402
from ego2sim.retarget import retarget  # noqa: E402
from ego2sim.perception import Track  # noqa: E402
from ego2sim.scene import EMBODIMENTS, build, phone_camera  # noqa: E402
from scripts.retarget_eval import task_objects  # noqa: E402

H = 360


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(img, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def fit(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * H / h), H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--cam", default="front", help="'phone' = sim camera at my real phone pose (occluded by the robot)")
    args = ap.parse_args()
    meas = Measurements.load()
    from ego2sim.frames import SHIFT, demo_to_task
    raw_demo = HumanDemo.load(DATA / "demos" / f"{args.demo}.npz")
    demo = demo_to_task(raw_demo)
    phone = imageio.mimread(OUT / "perception" / f"{args.demo}.mp4", memtest=False)
    fps = 1 / demo.dt
    tr_ = Track.load(DATA / "tracks" / f"{args.demo}.npz")
    calib = json.loads((DATA / "calib.json").read_text())
    T_med = tr_.T_cam_table[tr_.marker_ok]
    T = T_med[len(T_med) // 2]
    # camera pose w.r.t. the marker frame -> w.r.t. the sim task frame (180 deg about z + shift)
    T_mt = np.eye(4); T_mt[:3, :3] = np.diag([-1.0, -1.0, 1.0]); T_mt[:2, 3] = SHIFT  # task <- marker
    T = T @ np.linalg.inv(T_mt)
    phone_cam = phone_camera(T, calib["K"], tr_.size)
    # the demo starts where the hand track starts; the robots first spend 1 s coming from home
    s0 = max(0, raw_demo.qa["start_frame"] - int(1.0 * fps))
    cols = [[label(fit(f), "my phone video") for f in phone[s0:]]]
    for rname in ["panda", "g1"]:
        emb = EMBODIMENTS[rname]()
        m, d = build(emb, task_objects(meas), phone_cam=phone_cam)
        set_layout(m, d, emb, demo.cube_xy, demo.bowl_xy, meas.cube, demo.cube_yaw)
        tr = retarget(demo, m, emb, meas.cube, method="torque", lam=LAMBDA_OFFLINE[rname])
        r = mujoco.Renderer(m, H, int(H * 4 / 3))
        ro = run(m, d, emb, tr.q, tr.grip, tr.dt, render=r, render_every=1 / fps, cam=args.cam)
        skip = 0
        name = "Franka Panda" if rname == "panda" else "Unitree G1 humanoid"
        cols.append([label(f, f"{name} - success: {ro.success}") for f in ro.frames[skip:]])
        r.close()
    n = min(len(c) for c in cols)
    frames = [np.concatenate([c[i] for c in cols], 1) for i in range(n)]
    out = OUT / "side_by_side"
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out / f"{args.demo}.mp4", frames, fps=fps, macro_block_size=8)
    small = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)) for f in frames[::3]]
    imageio.mimsave(out / f"{args.demo}.gif", small, duration=3 / fps, loop=0)
    print("wrote", out / f"{args.demo}.mp4")


if __name__ == "__main__":
    main()
