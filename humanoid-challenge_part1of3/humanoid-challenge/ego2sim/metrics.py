"""What the motors feel: torque-rate and chattering metrics from 500 Hz rollouts."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def torque_metrics(ro, dt=0.002, hf_cut=5.0, skip_s=0.1):
    """ro: execute.Rollout. Skips the first `skip_s` (servo transient at start)."""
    k = int(skip_s / dt)
    tau = ro.tau[k:]
    dtau = np.diff(tau, axis=0) / dt
    b, a = butter(2, hf_cut / (0.5 / dt), btype="high")
    tau_hf = filtfilt(b, a, tau, axis=0)
    # share of torque signal energy above hf_cut (chattering) - scale free
    var = np.var(tau, axis=0).sum() + 1e-12
    hf_share = float((tau_hf ** 2).mean(0).sum() / (var))
    return {
        "tau_rate_rms": float(np.sqrt(np.mean(dtau ** 2))),          # N m / s
        "tau_rate_p99": float(np.percentile(np.abs(dtau), 99)),
        "tau_hf_rms": float(np.sqrt(np.mean(tau_hf ** 2))),          # N m above 5 Hz
        "tau_peak": float(np.abs(tau).max()),
        "chatter_energy_pct": 100 * hf_share,
    }


def tracking_metrics(ro, traj, dt_phys=0.002):
    """Sim TCP vs the commanded task-space target, sampled at control rate.
    (rollout logs start after the object-settling phase, so t=0 is aligned)"""
    idx = (np.arange(len(traj.p_target)) * traj.dt / dt_phys).astype(int)
    idx = idx[idx < len(ro.tcp)]
    err = np.linalg.norm(ro.tcp[idx] - traj.p_target[: len(idx)], axis=1)
    ig = min(traj.i_grasp, len(idx) - 1)
    return {"track_rms_mm": float(np.sqrt(np.mean(err ** 2)) * 1000),
            "track_at_grasp_mm": float(err[ig] * 1000)}


def tcp_jerk(ro, dt=0.002, skip_s=0.1):
    k = int(skip_s / dt)
    p = ro.tcp[k:]
    j = np.diff(p, 3, axis=0) / dt ** 3
    return float(np.sqrt(np.mean(np.sum(j ** 2, 1))))
