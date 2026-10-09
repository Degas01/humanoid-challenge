"""Step 2 - replay every human demo on both robots, three retargeting methods.

Outputs
  outputs/retarget/metrics.csv        one row per (demo, robot, method[, param])
  outputs/retarget/summary.md         mean +- std table
  outputs/retarget/torque_trace_*.png what the shoulder/elbow motors feel
  outputs/retarget/pareto_*.png       smoothness vs contact accuracy sweep (--sweep)
  outputs/retarget/replay_*.gif       sim replays (--render N)
"""
import argparse
import csv
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.config import LAMBDA_OFFLINE, Measurements  # noqa: E402
from ego2sim.demo import HumanDemo  # noqa: E402
from ego2sim.deploy import set_layout  # noqa: E402
from ego2sim.execute import run  # noqa: E402
from ego2sim.metrics import tcp_jerk, torque_metrics, tracking_metrics  # noqa: E402
from ego2sim.paths import DATA, OUT  # noqa: E402
from ego2sim.retarget import retarget, synthetic_demo  # noqa: E402
from ego2sim.scene import EMBODIMENTS, TaskObjects, build  # noqa: E402


def load_demos(synthetic=False):
    if synthetic:
        rng = np.random.default_rng(0)
        return [synthetic_demo(cube_xy=tuple(rng.uniform([-0.1, -0.15], [0.1, 0.0])), seed=i)
                for i in range(6)]
    from ego2sim.frames import demo_to_task
    return [demo_to_task(HumanDemo.load(p)) for p in sorted((DATA / "demos").glob("*.npz"))]


def load_layouts():
    """Cube/bowl layout of every real clip (task frame): {name: (cube_xy, yaw, bowl_xy, has_demo)}."""
    import json
    from ego2sim.demo import fold_yaw
    from ego2sim.frames import to_task_xy
    L = json.loads((DATA / "layouts.json").read_text())
    return {k: (to_task_xy(v["cube_xy"]), float(fold_yaw(v["cube_yaw"])), to_task_xy(v["bowl_xy"]), v["has_hand_demo"])
            for k, v in L.items()}


def task_objects(meas):
    return TaskObjects(cube_size=meas.cube, bowl_diam=meas.bowl_diam_mm / 1000,
                       bowl_height=meas.bowl_height_mm / 1000, marker_size=meas.marker_mm / 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", default="panda,g1")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--render", type=int, default=0, help="render GIFs for the first N demos")
    ap.add_argument("--synthetic", action="store_true", help="smoke test without real data")
    args = ap.parse_args()
    meas = Measurements.load()
    demos = load_demos(args.synthetic)
    out = OUT / ("retarget_synthetic" if args.synthetic else "retarget")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(demos)} demos")
    rows = []
    for rname in args.robots.split(","):
        emb = EMBODIMENTS[rname]()
        m, d = build(emb, task_objects(meas))
        lam0 = LAMBDA_OFFLINE[rname]
        cfgs = [("raw", None), ("lowpass", 2.5), ("torque", lam0)]
        if args.sweep:
            cfgs += [("lowpass", c) for c in (1.0, 1.5, 4.0, 6.0)]
            cfgs += [("torque", lam0 * f) for f in (0.01, 0.1, 10.0, 100.0)]
        renderer = mujoco.Renderer(m, 360, 480) if args.render else None
        for k, demo in enumerate(demos):
            set_layout(m, d, emb, demo.cube_xy, demo.bowl_xy, meas.cube, demo.cube_yaw)
            for meth, par in cfgs:
                kw = {"lam": par} if meth == "torque" else ({"cutoff_hz": par} if meth == "lowpass" else {})
                tr = retarget(demo, m, emb, meas.cube, method=meth, **kw)
                do_render = renderer is not None and k < args.render and par in (None, 2.5, lam0)
                ro = run(m, d, emb, tr.q, tr.grip, tr.dt, render=renderer if do_render else None,
                         cam="front")
                row = {"demo": demo.name, "robot": rname, "method": meth, "param": par,
                       "success": int(ro.success), **torque_metrics(ro), **tracking_metrics(ro, tr),
                       "tcp_jerk": tcp_jerk(ro)}
                rows.append(row)
                print(f"{rname:5s} {demo.name:14s} {meth:8s} {str(par):8s} ok={ro.success} "
                      f"rate={row['tau_rate_rms']:7.1f} hf={row['tau_hf_rms']:.3f} "
                      f"grasp_err={row['track_at_grasp_mm']:.1f}mm")
                if do_render:
                    imageio.mimsave(out / f"replay_{rname}_{demo.name}_{meth}.gif", ro.frames[::2],
                                    duration=1 / 15, loop=0)
                if k == 0 and par in (None, 2.5, lam0):
                    np.savez_compressed(out / f"trace_{rname}_{meth}.npz", t=ro.t, tau=ro.tau)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
