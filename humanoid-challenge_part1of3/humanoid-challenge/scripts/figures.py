"""Step 5 - figures and tables for the README from the CSVs written by steps 2-4."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.config import LAMBDA_OFFLINE  # noqa: E402
from ego2sim.paths import OUT  # noqa: E402

COL = {"raw": "#2a78d6", "lowpass": "#eb6834", "torque": "#1baf7a",
       "none": "#2a78d6", "ema": "#eb6834"}
LABEL = {"raw": "raw IK", "lowpass": "zero-phase low-pass 2.5 Hz", "torque": "torque-rate layer (ours)",
         "none": "no smoothing", "ema": "causal low-pass 3 Hz"}
ROBOT = {"panda": "Franka Panda", "g1": "Unitree G1 (humanoid)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "axes.edgecolor": "#8a8984",
                     "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "white", "savefig.dpi": 150})


def retarget_figs(rdir):
    df = pd.read_csv(rdir / "metrics.csv")
    main = df[(df.method == "raw") | ((df.method == "lowpass") & (df.param == 2.5)) |
              df.apply(lambda r: r.method == "torque" and np.isclose(r.param, LAMBDA_OFFLINE[r.robot]), axis=1)]
    robots = [r for r in ROBOT if r in set(main.robot)]
    metrics = [("tau_rate_rms", "torque rate RMS [N·m/s]  (log)", True),
               ("tau_hf_rms", "torque energy > 5 Hz, RMS [N·m]", False),
               ("track_at_grasp_mm", "TCP error at the grasp [mm]", False),
               ("success", "task success [%]", False)]
    fig, axs = plt.subplots(len(robots), 4, figsize=(13, 3.1 * len(robots)), squeeze=False)
    meths = ["raw", "lowpass", "torque"]
    for i, rb in enumerate(robots):
        sub = main[main.robot == rb]
        for j, (col, lab, log) in enumerate(metrics):
            ax = axs[i, j]
            vals = [sub[sub.method == mth][col] * (100 if col == "success" else 1) for mth in meths]
            mu = [v.mean() for v in vals]
            sd = [v.std() if col != "success" else 0 for v in vals]
            ax.bar(range(3), mu, yerr=sd, color=[COL[mth] for mth in meths], width=0.62,
                   capsize=3, error_kw={"ecolor": "#52514e", "lw": 1})
            for x, v in enumerate(mu):
                ax.text(x, v * (1.15 if log else 1) + (0 if log else max(mu) * 0.03),
                        f"{v:.0f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            if log:
                ax.set_yscale("log")
            ax.set_xticks(range(3)); ax.set_xticklabels(["raw", "low-pass", "ours"])
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(lab, fontsize=10, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(ROBOT[rb], fontsize=11, color="#0b0b0b")
    n_demos = len(main[(main.robot == robots[0]) & (main.method == "raw")])
    fig.suptitle(f"Replaying my {n_demos} recorded demos on two robots (500 Hz physics)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(rdir / "retarget_bars.png", bbox_inches="tight")
    plt.close(fig)

    # torque traces for the first demo
    fig, axs = plt.subplots(1, len(robots), figsize=(6.5 * len(robots), 3.2), squeeze=False)
    for i, rb in enumerate(robots):
        ax = axs[0, i]
        jt = 1 if rb == "panda" else 0
        traces = {mth: np.load(rdir / f"trace_{rb}_{mth}.npz") for mth in meths
                  if (rdir / f"trace_{rb}_{mth}.npz").exists()}
        for mth, z in traces.items():
            ax.plot(z["t"], z["tau"][:, jt], color=COL[mth], lw=1.4 if mth != "raw" else 0.9,
                    label=LABEL[mth], alpha=0.9 if mth != "raw" else 0.7)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("shoulder torque [N·m]")
        ax.set_title(f"{ROBOT[rb]} - what the shoulder motor feels", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(rdir / "torque_traces.png", bbox_inches="tight")
    plt.close(fig)

    # sweep: smoothness vs contact accuracy
    sw = df[df.method != "raw"]
    if sw.groupby(["robot", "method"]).param.nunique().max() > 1:
        fig, axs = plt.subplots(1, len(robots), figsize=(6 * len(robots), 3.6), squeeze=False)
        for i, rb in enumerate(robots):
            ax = axs[0, i]
            for mth in ["lowpass", "torque"]:
                g = sw[(sw.robot == rb) & (sw.method == mth)].groupby("param")
                pts = g[["tau_rate_rms", "track_at_grasp_mm", "success"]].mean().sort_index()
                ax.plot(pts.tau_rate_rms, pts.track_at_grasp_mm, "-o", color=COL[mth], ms=7,
                        label=LABEL[mth].split(" 2.5")[0], lw=2)
                for p, r in pts.iterrows():
                    lab = f"{p:g} Hz" if mth == "lowpass" else f"λ={p:.0e}"
                    ax.annotate(lab + ("" if r.success == 1 else f"  ({r.success * 100:.0f}% ok)"),
                                (r.tau_rate_rms, r.track_at_grasp_mm), fontsize=7, color="#52514e",
                                xytext=(5, 4), textcoords="offset points")
            xs_ = sw[sw.robot == rb].groupby(["method", "param"]).tau_rate_rms.mean()
            if xs_.max() / xs_.min() > 3:
                ax.set_xscale("log")
            ax.set_xlabel("torque rate RMS [N·m/s]  (smoother ←)")
            ax.set_ylabel("TCP error at grasp [mm]  (more precise ↓)")
            ax.set_title(ROBOT[rb], fontsize=10)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(rdir / "pareto.png", bbox_inches="tight")
        plt.close(fig)

    tab = main.groupby(["robot", "method"]).agg(
        success=("success", "mean"), tau_rate=("tau_rate_rms", "mean"), tau_hf=("tau_hf_rms", "mean"),
        tau_peak=("tau_peak", "mean"), grasp_err=("track_at_grasp_mm", "mean"),
        track_rms=("track_rms_mm", "mean"))
    tab["success"] = (tab["success"] * 100).round(0)
    (rdir / "summary.md").write_text(tab.round(2).to_markdown() + "\n")
    print(tab.round(2).to_markdown())


PNAME = {"knn_real": "retrieval\nmy 3 demos", "knn_aug": "retrieval\n+ re-anchored",
         "mlp_real": "MLP\nmy 3 demos", "mlp_aug": "MLP\n+ re-anchored"}


def policy_figs(pdir):
    df = pd.concat([pd.read_csv(f) for f in sorted(pdir.glob("eval_*.csv"))])
    df.to_csv(pdir / "eval.csv", index=False)
    robots = [r for r in ROBOT if r in set(df.robot)]
    pols = [p for p in PNAME if p in set(df.policy)]
    smooths = [s for s in ["none", "ema", "torque"] if s in set(df.smooth)]
    fig, axs = plt.subplots(2, len(robots), figsize=(6.5 * len(robots), 6.6), squeeze=False)
    w = 0.8 / len(smooths)
    for i, rb in enumerate(robots):
        for k, sm in enumerate(smooths):
            sub = [df[(df.robot == rb) & (df.policy == p) & (df.smooth == sm)] for p in pols]
            succ = [x.success.mean() * 100 for x in sub]
            rate = [x.tau_rate_rms.median() for x in sub]
            x = np.arange(len(pols)) + (k - (len(smooths) - 1) / 2) * w
            axs[0, i].bar(x, succ, w * 0.92, color=COL[sm], label=LABEL[sm])
            axs[1, i].bar(x, rate, w * 0.92, color=COL[sm], label=LABEL[sm])
            for xi, v in zip(x, succ):
                axs[0, i].text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
        for ax in axs[:, i]:
            ax.set_xticks(range(len(pols))); ax.set_xticklabels([PNAME[p] for p in pols], fontsize=8.5)
            ax.grid(axis="x", visible=False)
        axs[0, i].set_title(ROBOT[rb], fontsize=11)
        axs[0, i].set_ylim(0, 110)
        axs[1, i].set_yscale("log")
    axs[0, 0].set_ylabel("closed-loop success [%]")
    axs[1, 0].set_ylabel("torque rate, median RMS\n[N·m/s] (log)")
    n = df.layout.nunique()
    fig.suptitle(f"One policy, two bodies - {n} layouts per bar (11 from my own clips + 9 random)",
                 fontsize=12, y=1.04)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(pdir / "policy_eval.png", bbox_inches="tight")
    plt.close(fig)
    tab = df.groupby(["robot", "policy", "smooth"]).agg(
        success=("success", "mean"), grasped=("grasped", "mean"), tau_rate_median=("tau_rate_rms", "median"),
        tau_hf=("tau_hf_rms", "median"), tau_peak=("tau_peak", "median"), ep_time=("ep_time", "mean"))
    tab[["success", "grasped"]] = (tab[["success", "grasped"]] * 100).round(0)
    real = df[df.layout_name.str.startswith("demo")]
    t2 = real.groupby(["robot", "policy", "smooth"]).success.mean().mul(100).round(0).rename("success_on_my_11_layouts")
    tab = tab.join(t2)
    (pdir / "summary.md").write_text(tab.round(2).to_markdown() + "\n")
    print(tab.round(2).to_markdown())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    sfx = "_synthetic" if a.synthetic else ""
    if (OUT / f"retarget{sfx}" / "metrics.csv").exists():
        retarget_figs(OUT / f"retarget{sfx}")
    if list((OUT / f"policy{sfx}").glob("eval_*.csv")):
        policy_figs(OUT / f"policy{sfx}")
