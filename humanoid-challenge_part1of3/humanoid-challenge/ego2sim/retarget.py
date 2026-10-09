"""HumanDemo  ->  joint trajectories for any embodiment, three ways.

  raw       per-frame IK on the perceived pinch point (what a naive pipeline does)
  lowpass   zero-phase Butterworth on the task-space targets, then IK (the usual fix)
  torque    per-frame IK, then the TorqueRateLayer (ours)

Shared by all three:
  * workspace map human -> robot (identity for the Panda, scaled for the G1)
  * event anchoring: around the grasp the trajectory is shifted so that the
    pinch point sits exactly at the cube centre height; perception depth error
    is removed where it matters and nowhere else
  * a 1 s minimum-jerk approach from the robot's home posture to the first
    human sample, and safety clamps (table clearance)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from .demo import HumanDemo
from .kinematics import IK
from .torque_rate import regularize, upsample


@dataclass
class RobotTraj:
    method: str
    dt: float
    q: np.ndarray        # (N, 7)
    grip: np.ndarray     # (N,)
    p_target: np.ndarray # (N, 3) task-space target the robot was asked to track
    yaw: np.ndarray
    i_grasp: int
    i_release: int


def _replace(demo, **kw):
    import copy
    d = copy.copy(demo)
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def minjerk(a, b, n):
    s = np.linspace(0, 1, n)
    s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return a[None] + (np.asarray(b) - a)[None] * s[:, None]


def task_targets(demo: HumanDemo, emb, cube_size: float, anchor=True, z_min=None):
    p = emb.map_point(demo.pinch)
    yaw = demo.yaw.copy()
    if anchor:
        # shift z by the grasp-height error, faded in/out over 0.5 s around the hold
        w = int(0.5 / demo.dt)
        dz = cube_size / 2 + emb_grasp_z_offset(emb) - np.median(p[demo.i_grasp:demo.i_grasp + 3, 2])
        prof = np.zeros(len(p))
        a, b = demo.i_grasp, demo.i_release
        prof[a:b + 1] = 1
        prof[max(0, a - w):a] = np.linspace(0, 1, a - max(0, a - w))
        prof[b + 1:min(len(p), b + 1 + w)] = np.linspace(1, 0, min(len(p), b + 1 + w) - (b + 1))
        p[:, 2] += dz * prof
    if z_min is None:
        z_min = 0.012 + emb_grasp_z_offset(emb)
    cube_r = emb.map_point([*demo.cube_xy, 0])[:2]
    bowl_r = emb.map_point([*demo.bowl_xy, 0])[:2]
    p[:, 2] = np.maximum(p[:, 2], table_floor(p[:, :2], cube_r, bowl_r, z_min))
    # yaw is only meaningful near the cube; elsewhere hold the grasp yaw (avoids wrist spinning)
    yaw[:] = np.where(np.arange(len(yaw)) < demo.i_grasp, yaw, demo.cube_yaw)
    return p, yaw


def table_floor(p_xy, cube_xy, bowl_xy, z_contact, z_free=0.06):
    """Lowest allowed TCP height. My hand rests flat on the table between demos;
    a gripper doing the same scrapes the table and the servo fights the contact
    (it shows up as large torque-rate spikes). Only near the objects may the TCP
    go down to grasp height; elsewhere it keeps `z_free` clearance."""
    dist = np.minimum(np.linalg.norm(p_xy - cube_xy, axis=-1), np.linalg.norm(p_xy - bowl_xy, axis=-1))
    s = np.clip((dist - 0.04) / 0.06, 0, 1)
    return z_contact + (z_free - z_contact) * s


def emb_grasp_z_offset(emb):
    # the G1 grasps with the middle finger below the tcp: keep it off the table
    return 0.012 if emb.name == "g1" else 0.0


def _ik_sequence(ik, emb, P, YAW, q0):
    Q, errs, q = [], [], q0.copy()
    for p, y in zip(P, YAW):
        q, e = ik.solve(p, emb.tcp_target_rot(y), q, iters=40)
        Q.append(q.copy()); errs.append(e)
    return np.array(Q), np.array(errs)


def retarget(demo: HumanDemo, model, emb, cube_size: float, method: str = "torque",
             lam: float = 2e-7, cutoff_hz: float = 2.5, approach_s: float = 1.0,
             tail_s: float = 0.5, up: int = 3, dwell=(0.5, 0.4)) -> RobotTraj:
    """All methods are delivered to the joint servos at `up` x the camera rate
    (linear interpolation of the per-frame solution for raw / lowpass; the
    torque-rate solve itself runs at the higher rate)."""
    ik = IK(model, emb)
    dt = demo.dt
    P, YAW = task_targets(demo, emb, cube_size)
    # Contact retiming: a servo gripper closes in ~0.3 s, a human hand in ~0.1 s.  The pinch point
    # is held still for `dwell` seconds at the grasp (closing, then lift-off) and at the release
    # (opening, then retreat); the rest of the human timing is kept.
    n_g, n_r = int(dwell[0] / dt), int(dwell[1] / dt)
    ig0, ir0, N0 = demo.i_grasp, demo.i_release, len(P)
    idx = np.r_[np.arange(0, ig0 + 1), np.full(n_g, ig0), np.arange(ig0 + 1, ir0 + 1),
                np.full(n_r, ir0), np.arange(ir0 + 1, N0)]
    P, YAW = P[idx], YAW[idx]
    grip0 = np.r_[demo.grip[:ig0 + 1], np.ones(n_g), demo.grip[ig0 + 1:ir0 + 1],
                  np.zeros(n_r), demo.grip[ir0 + 1:]]
    demo = _replace(demo, grip=grip0, i_grasp=ig0 + n_g, i_release=ir0 + n_g + 1)
    if method == "lowpass":
        b, a = butter(2, cutoff_hz / (0.5 / dt))
        P = filtfilt(b, a, P, axis=0)
        YAW = filtfilt(b, a, YAW)
    # approach from home + tail hold
    p_home, _ = ik.fk(emb.home_q)
    n_app = int(approach_s / dt)
    app = minjerk(p_home, P[0], n_app)
    P_all = np.vstack([app, P, np.repeat(P[-1:], int(tail_s / dt), 0)])
    YAW_all = np.r_[np.full(n_app, YAW[0]), YAW, np.full(int(tail_s / dt), YAW[-1])]
    G_all = np.r_[np.zeros(n_app), demo.grip, np.full(int(tail_s / dt), demo.grip[-1])]
    ig, ir = demo.i_grasp + n_app, demo.i_release + n_app
    Q, errs = _ik_sequence(ik, emb, P_all, YAW_all, emb.home_q)
    Q = upsample(Q, up)
    P_all = upsample(P_all, up)
    YAW_all = upsample(YAW_all[:, None], up)[:, 0]
    G_all = np.repeat(G_all, up)[: len(Q)]
    ig, ir, dt = ig * up, ir * up, dt / up
    if method == "torque":
        # track tightly around the contact events, let the smoother work elsewhere
        w = np.ones(len(Q))
        k = int(0.25 / dt)
        for i in (ig, ir):
            w[max(0, i - k):i + k] = 6.0
        Q = regularize(model, emb, Q, dt, lam=lam, weights=w, q_start=emb.home_q)
        Q = np.clip(Q, ik.lo, ik.hi)
    return RobotTraj(method, dt, Q, G_all, P_all, YAW_all, ig, ir)


# ----------------------------------------------------------------------------
# A synthetic "human" for development and unit tests ONLY. All results in the
# README come from real recordings.
# ----------------------------------------------------------------------------
def synthetic_demo(cube_xy=(0.03, -0.08), bowl_xy=(0.0, 0.17), cube_size=0.05, fps=30.0,
                   noise_mm=4.0, seed=0) -> HumanDemo:
    rng = np.random.default_rng(seed)
    rest = np.array([-0.15, -0.20, 0.03])
    c = np.r_[cube_xy, cube_size / 2]
    bw = np.r_[bowl_xy, 0.14]
    segs = [(rest, rest, 0.8, 0), (rest, c + [0, 0, 0.10], 0.9, 0), (c + [0, 0, 0.10], c, 0.5, 0),
            (c, c, 0.4, 1), (c, c + [0, 0, 0.12], 0.5, 1), (c + [0, 0, 0.12], bw, 0.9, 1),
            (bw, bw, 0.3, 0), (bw, rest, 1.0, 0), (rest, rest, 0.6, 0)]
    P, G = [], []
    for a, b, T, g in segs:
        n = int(T * fps)
        P.append(minjerk(np.asarray(a, float), b, n)); G.append(np.full(n, g, float))
    P = np.vstack(P); G = np.concatenate(G)
    # MediaPipe-like noise: white + slowly varying bias + rare spikes
    noise = rng.normal(0, noise_mm / 1000, P.shape)
    drift = np.cumsum(rng.normal(0, 0.0008, P.shape), 0); drift -= drift.mean(0)
    P_noisy = P + noise + 0.3 * drift
    N = len(P)
    i_g = int(np.argmax(G > 0.5)); i_r = int(N - 1 - np.argmax(G[::-1] > 0.5))
    yaw = 0.15 + rng.normal(0, 0.05, N)
    ap = np.where(G > 0.5, cube_size + 0.02, 0.10)
    return HumanDemo("synthetic", 1 / fps, np.arange(N) / fps, P_noisy, ap, yaw, G, i_g, i_r,
                     np.array(cube_xy), 0.15, np.array(bowl_xy), 1.0, {"synthetic": True})
