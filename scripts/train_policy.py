"""Step 3 - train the human-frame chunk policy.

Two policies with the same optimisation budget:
  real   only my N recorded demos
  aug    my N demos + object-centric re-anchored copies (new cube/bowl layouts)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.augment import reanchor, sample_layout  # noqa: E402
from ego2sim.config import Measurements  # noqa: E402
from ego2sim.paths import OUT  # noqa: E402
from ego2sim.policy import P_START, active_range, train, with_reach  # noqa: E402
from scripts.retarget_eval import load_demos, load_layouts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-per-demo", type=int, default=30)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    meas = Measurements.load()
    demos = load_demos(args.synthetic)
    if not args.synthetic:
        demos = [with_reach(d) for d in demos]      # approach = prior, the rest is my recording
    out = OUT / ("policy_synthetic" if args.synthetic else "policy")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # The re-anchoring region is where I actually put the cube and bowl in ALL my clips, including
    # the ones whose hand track failed: the scene parser gives those layouts from the first frame.
    if args.synthetic:
        regs = demos
        C = np.array([d.cube_xy for d in regs]); B = np.array([d.bowl_xy for d in regs])
    else:
        L = load_layouts()
        C = np.array([v[0] for v in L.values()]); B = np.array([v[2] for v in L.values()])
    cl, ch, bl, bh = C.min(0) - 0.03, C.max(0) + 0.03, B.min(0) - 0.03, B.max(0) + 0.03   # the bowl never moved: +-3 cm

    def draw():
        for _ in range(1000):
            c, b = rng.uniform(cl, ch), rng.uniform(bl, bh)
            if np.linalg.norm(c - b) > 0.09:      # cube outside the bowl
                return c, b
        return c, b

    aug = [reanchor(d, *draw(), cube_yaw=float(rng.uniform(-0.6, 0.6)))
           for d in demos for _ in range(args.aug_per_demo)]
    rest = P_START if not args.synthetic else np.mean([d.pinch[active_range(d)[0]] for d in demos], 0)
    meta = {"n_real": len(demos), "n_aug": len(aug), "rest_h": np.asarray(rest).tolist(),
            "cube_lo": cl.tolist(), "cube_hi": ch.tolist(), "bowl_lo": bl.tolist(), "bowl_hi": bh.tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    # retrieval policies (no training): just the object-relative states and the chunks that followed
    from ego2sim.policy import build_dataset
    for tag, data in [("knn_real", demos), ("knn_aug", demos + aug)]:
        X, Yp, Yg = build_dataset(data, meas.cube, obs_noise=0.0, copies=1)
        np.savez_compressed(out / f"{tag}.npz", X=X.numpy(), Yp=Yp.numpy(), Yg=Yg.numpy())
        print(f"retrieval memory '{tag}': {len(X)} states from {len(data)} demos")
    curves = {}
    for tag, data in [("mlp_real", demos), ("mlp_aug", demos + aug)]:
        print(f"training '{tag}' on {len(data)} demos")
        pol, hist = train(data, meas.cube, steps=args.steps)
        torch.save(pol.state_dict(), out / f"{tag}.pt")
        curves[tag] = hist
    np.savez_compressed(out / "loss_curves.npz", **{k: np.array(v) for k, v in curves.items()})


if __name__ == "__main__":
    main()
