"""Step 4 - closed-loop evaluation of the policies on both robots.

Same random layouts (cube, bowl) for every variant:
  policy  in {real, aug}
  smooth  in {none, ema, torque}
  robot   in {panda, g1}
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.config import LAMBDA_ONLINE, Measurements  # noqa: E402
from ego2sim.deploy import Deployer, set_layout  # noqa: E402
from ego2sim.paths import OUT  # noqa: E402
from ego2sim.policy import ChunkPolicy, KNNPolicy  # noqa: E402
from ego2sim.scene import EMBODIMENTS, build  # noqa: E402
from scripts.retarget_eval import task_objects  # noqa: E402


def episode_metrics(tau, dt=0.002):
    from scipy.signal import butter, filtfilt
    rate = np.diff(tau, axis=0) / dt
    b, a = butter(2, 5.0 / (0.5 / dt), btype="high")
    hf = filtfilt(b, a, tau, axis=0)
    return {"tau_rate_rms": float(np.sqrt((rate ** 2).mean())),
            "tau_hf_rms": float(np.sqrt((hf ** 2).mean())),
            "tau_peak": float(np.abs(tau).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", default="panda,g1")
    ap.add_argument("--policies", default="knn_real,knn_aug,mlp_aug")
    ap.add_argument("--smooth", default="none,ema,torque")
    ap.add_argument("--n", type=int, default=20, help="total layouts: my 11 real ones + random")
    ap.add_argument("--gifs", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    meas = Measurements.load()
    pdir = OUT / ("policy_synthetic" if args.synthetic else "policy")
    meta = json.loads((pdir / "meta.json").read_text())
    rng = np.random.default_rng(123)
    layouts, names = [], []
    if not args.synthetic:
        # (1) the layouts of my own recordings (cube/bowl as seen in each clip's first frame)
        from scripts.retarget_eval import load_layouts
        for k, (c, yaw, b, has_demo) in load_layouts().items():
            layouts.append((c, b, yaw)); names.append(k + ("*" if has_demo else ""))
    # (2) random layouts in the region my clips cover
    while len(layouts) < args.n:
        c = rng.uniform(meta["cube_lo"], meta["cube_hi"])
        b = rng.uniform(meta["bowl_lo"], meta["bowl_hi"])
        if np.linalg.norm(c - b) > 0.11:
            layouts.append((c, b, float(rng.uniform(-0.4, 0.4)))); names.append(f"rand_{len(layouts):02d}")
    rest = np.array(meta["rest_h"])
    rows = []
    for rname in args.robots.split(","):
        emb = EMBODIMENTS[rname]()
        m, d = build(emb, task_objects(meas))
        renderer = mujoco.Renderer(m, 360, 480) if args.gifs else None
        for ptag in args.policies.split(","):
            if ptag.startswith("knn"):
                pol = KNNPolicy.load(pdir / f"{ptag}.npz")
            else:
                pol = ChunkPolicy(); pol.load_state_dict(torch.load(pdir / f"{ptag}.pt")); pol.eval()
            for sm in args.smooth.split(","):
                dep = Deployer(m, d, emb, meas.cube, lam=LAMBDA_ONLINE[rname], smooth=sm)
                succ = []
                for k, (c, b, yaw) in enumerate(layouts):
                    set_layout(m, d, emb, c, b, meas.cube, yaw)
                    gif = renderer if (k < args.gifs and ptag == "knn_aug" and sm in ("torque", "none")) else None
                    t0 = time.time()
                    r = dep.run(pol, rest, render=gif, cam="front")
                    row = {"robot": rname, "policy": ptag, "smooth": sm, "layout": k, "layout_name": names[k],
                           "cube_x": c[0], "cube_y": c[1], "success": int(r["success"]),
                           "grasped": int(r["grasped"]), "ep_time": r["time"], **episode_metrics(r["tau"])}
                    rows.append(row); succ.append(r["success"])
                    if gif is not None:
                        imageio.mimsave(pdir / f"rollout_{rname}_{ptag}_{sm}_{k}.gif", r["frames"][::2],
                                        duration=1 / 15, loop=0)
                    print(f"{rname:5s} {ptag:4s} {sm:6s} #{k:02d} ok={r['success']} "
                          f"rate={row['tau_rate_rms']:7.1f} ({time.time() - t0:.1f}s)")
                print(f"==> {rname} {ptag} {sm}: success {np.mean(succ):.2f}")
    with open(pdir / f"eval_{args.robots.replace(',', '_')}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
