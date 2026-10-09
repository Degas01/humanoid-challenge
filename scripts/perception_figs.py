"""Perception figures: bird's-eye view of every recorded layout + the three hand trajectories, and a
metric top-down rectification of one phone frame (what the scene parser actually looks at)."""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.config import Measurements  # noqa: E402
from ego2sim.demo import HumanDemo  # noqa: E402
from ego2sim.paths import DATA, OUT  # noqa: E402
from ego2sim.perception import Track  # noqa: E402
from ego2sim.scene_parse import rectify  # noqa: E402


def main():
    meas = Measurements.load()
    out = OUT / "perception"; out.mkdir(parents=True, exist_ok=True)
    L = json.loads((DATA / "layouts.json").read_text())
    demos = [HumanDemo.load(p) for p in sorted((DATA / "demos").glob("*.npz"))]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axs[0]
    s = meas.cube
    ax.add_patch(plt.Rectangle((-0.06, -0.06), 0.12, 0.12, fc="none", ec="#52514e", lw=1.2, ls="--"))
    ax.text(0, 0.075, "ArUco marker", ha="center", fontsize=8, color="#52514e")
    bowls = []
    for name, v in L.items():
        c = np.array(v["cube_xy"]); yaw = v["cube_yaw"]
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        sq = (R @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * s / 2).T).T + c
        col = "#d1495b" if v["has_hand_demo"] else "#edae49"
        ax.add_patch(plt.Polygon(sq, fc=col, ec="k", lw=0.6, alpha=0.85))
        ax.text(c[0], c[1], name[-2:], ha="center", va="center", fontsize=7, color="white", weight="bold")
        bowls.append(v["bowl_xy"])
    b = np.median(bowls, 0)
    for bb in bowls:
        ax.add_patch(plt.Circle(bb, meas.bowl_diam_mm / 2000, fc="none", ec="#00798c", lw=0.6, alpha=0.5))
    ax.add_patch(plt.Circle(b, meas.bowl_diam_mm / 2000, fc="#00798c", alpha=0.15))
    ax.text(b[0], b[1], "mug", ha="center", va="center", fontsize=9, color="#00798c")
    for d in demos:
        ax.plot(d.pinch[:, 0], d.pinch[:, 1], lw=1.4, color="#30638e")
        ax.plot(*d.pinch[d.i_grasp, :2], "o", ms=4, color="#30638e")
    ax.set_aspect("equal"); ax.set_xlim(-0.42, 0.22); ax.set_ylim(-0.6, 0.1)
    ax.set_xlabel("x [m] (marker frame)"); ax.set_ylabel("y [m]")
    ax.set_title("Cube layouts of my clips (11 of 12; not found in clip 08), metric top view\nred: hand demo used, yellow: layout only, blue: my pinch point",
                 fontsize=10)
    # rectified frame
    name = "demo_01"
    tr = Track.load(DATA / "tracks" / f"{name}.npz")
    calib = json.loads((DATA / "calib.json").read_text())
    cap = cv2.VideoCapture(str(DATA / "raw" / f"{name}.mp4")); cap.set(1, 3); ok, fr = cap.read(); cap.release()
    img, xs, ys = rectify(fr, calib["K"], calib["dist"], tr.T_cam_table[3], meas.cube,
                          x_rng=(-0.42, 0.22), y_rng=(-0.6, 0.1), res=0.001)
    axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]])
    v = L[name]
    c = np.array(v["cube_xy"]); yaw = v["cube_yaw"]
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    sq = (R @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * s / 2).T).T + c
    axs[1].plot(sq[:, 0], sq[:, 1], color="#5fd35f", lw=2)
    th = np.linspace(0, 2 * np.pi, 100)
    axs[1].plot(v["bowl_xy"][0] + 0.06 * np.cos(th), v["bowl_xy"][1] + 0.06 * np.sin(th), color="#ff9f1c", lw=2)
    axs[1].set_title(f"{name}, first frame re-projected onto the plane z = 6 cm\n"
                     "(1 px = 1 mm; green = 60 mm cube found, orange = 120 mm mug rim)", fontsize=10)
    axs[1].set_xlabel("x [m]")
    fig.tight_layout()
    fig.savefig(out / "layouts.png", dpi=140, bbox_inches="tight")
    print("wrote", out / "layouts.png")


if __name__ == "__main__":
    main()
