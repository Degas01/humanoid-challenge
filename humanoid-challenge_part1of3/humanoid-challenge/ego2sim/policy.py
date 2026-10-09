"""A small action-chunking policy trained on *human* hand trajectories.

The policy lives in the human/table frame: it never sees robot joints. It maps
(hand-or-gripper position, grip state, cube position, bowl position) to the
next K waypoints of the pinch point plus open/close. Deployment on a robot is
then embodiment-specific only through the workspace map, IK and the
TorqueRateLayer - so the *same* weights drive the Panda and the G1.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .demo import HumanDemo

K = 15          # chunk length (0.5 s at 30 Hz)
OBS_DIM = 1 + 3 + 2 + 1 + 1


def cube_track(demo: HumanDemo, cube_size: float):
    """Where the cube is at every frame of a human demo (it moves with the hand)."""
    N = len(demo.pinch)
    c = np.zeros((N, 3))
    c[:, :2] = demo.cube_xy
    c[:, 2] = cube_size / 2
    ig, ir = demo.i_grasp, demo.i_release
    c[ig:ir + 1] = demo.pinch[ig:ir + 1]
    c[ir + 1:, :2] = demo.bowl_xy
    c[ir + 1:, 2] = cube_size / 2 + 0.006
    return c


def make_obs(tcp, grip, cube, bowl_xy, t):
    """t = seconds since the grasp (0 before it). An *event* clock, not a wall clock: it lets the policy
    leave the dwell at the grasp, and it is the same for a fast hand and a slow robot."""
    tcp, cube, bowl_xy = np.atleast_2d(tcp), np.atleast_2d(cube), np.atleast_2d(bowl_xy)
    grip = np.atleast_1d(grip)[:, None]
    t = np.clip(np.atleast_1d(t), 0, 4)[:, None]
    # object-centric and translation-equivariant: no absolute positions, so the policy cannot
    # memorise where my 3 demos happened to be and has to use the cube / bowl it is looking at
    return np.concatenate([grip, cube - tcp, bowl_xy - tcp[:, :2], tcp[:, 2:3], t], 1)


def anchor(grasped, cube_start, bowl_xy):
    """Reference point the policy's waypoints are relative to."""
    return np.r_[bowl_xy, 0.0] if grasped else np.asarray(cube_start, float)


def active_range(demo: HumanDemo, v_thr=0.05):
    """Drop the motionless rest at the start/end of a clip (it only serves the
    hand-scale calibration) - otherwise the policy learns to wait forever."""
    from scipy.signal import savgol_filter
    w = max(5, int(0.4 / demo.dt) | 1)
    p = savgol_filter(demo.pinch, w, 2, axis=0)
    v = np.linalg.norm(np.gradient(p, demo.dt, axis=0), axis=1)
    moving = np.where(v > v_thr)[0]
    if len(moving) == 0:
        return 0, len(v)
    return max(0, moving[0] - 2), min(len(v), moving[-1] + 3)


def build_dataset(demos, cube_size, obs_noise=0.01, copies=4, seed=0):
    rng = np.random.default_rng(seed)
    X, Yp, Yg = [], [], []
    for d in demos:
        a, b = active_range(d)
        b = min(b, d.i_release + int(0.3 / d.dt))   # the policy's job ends at the release
        cube = cube_track(d, cube_size)[a:b]
        P, G = d.pinch[a:b], d.grip[a:b]
        N = len(P)
        Pp = np.vstack([P, np.repeat(P[-1:], K, 0)])
        Gp = np.r_[G, np.repeat(G[-1:], K)]
        for c in range(copies):
            noise = rng.normal(0, obs_noise, (N, 3)) * (c > 0)
            tcp = P + noise
            obs = make_obs(tcp, G, cube, np.repeat(d.bowl_xy[None], N, 0), np.clip(np.arange(a, b) - d.i_grasp, 0, None) * d.dt)
            # actions are *object-relative targets*: relative to the cube until the grasp, to the bowl
            # afterwards. Errors cannot accumulate and the release point is wherever the bowl is.
            anc = np.where((np.arange(a, b) < d.i_grasp)[:, None], np.r_[d.cube_xy, cube_size / 2][None], np.r_[d.bowl_xy, 0.0][None])
            fut = np.stack([Pp[i + 1:i + 1 + K] - anc[i] for i in range(N)])   # (N,K,3)
            fg = np.stack([Gp[i + 1:i + 1 + K] for i in range(N)])
            X.append(obs); Yp.append(fut); Yg.append(fg)
    return (torch.tensor(np.concatenate(X), dtype=torch.float32),
            torch.tensor(np.concatenate(Yp), dtype=torch.float32),
            torch.tensor(np.concatenate(Yg), dtype=torch.float32))


class ChunkPolicy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, K * 4))
        self.register_buffer("mu", torch.zeros(OBS_DIM))
        self.register_buffer("sd", torch.ones(OBS_DIM))

    def forward(self, obs):
        out = self.net((obs - self.mu) / self.sd).view(-1, K, 4)
        return out[..., :3] * 0.2, out[..., 3]          # object-relative targets [m], grip logits

    @torch.no_grad()
    def act(self, obs_np):
        dp, g = self(torch.tensor(obs_np, dtype=torch.float32))
        return dp.numpy(), torch.sigmoid(g).numpy()


def train(demos, cube_size, steps=15000, lr=1e-3, seed=0, bs=512, log=print):
    """Fixed number of gradient steps, so policies trained on 20 or 600 demos get the
    same optimisation budget."""
    torch.manual_seed(seed)
    X, Yp, Yg = build_dataset(demos, cube_size, seed=seed)
    pol = ChunkPolicy()
    pol.mu.copy_(X.mean(0)); pol.sd.copy_(X.std(0) + 1e-3)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = len(X)
    hist = []
    g = torch.Generator().manual_seed(seed)
    for it in range(steps):
        b = torch.randint(0, n, (bs,), generator=g)
        dp, gl = pol(X[b])
        loss = (dp - Yp[b]).abs().mean() * 100 + nn.functional.binary_cross_entropy_with_logits(gl, Yg[b])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
        if it % 2500 == 0 or it == steps - 1:
            log(f"  step {it:6d}  loss {np.mean(hist[-200:]):.4f}   ({n} samples)")
    return pol, hist


P_START = np.array([0.06, -0.04, 0.18])      # nominal hover pose every episode starts from (task frame)


def with_reach(demo: HumanDemo, p_start=P_START, t_up=0.9, t_down=0.5) -> HumanDemo:
    """My hand is only tracked from the hover above the cube onwards (it enters the frame from the
    edge and is out of view / lost during the approach), so the approach is NOT observed.  It is
    added as a min-jerk prior (start pose -> above the cube -> grasp pose); everything from the
    hover onwards (grasp, carry, release, retreat) is the real recording."""
    import copy
    from .retarget import minjerk
    p0 = demo.pinch[0]
    above = np.r_[demo.cube_xy, 0.13]
    a = minjerk(p_start, above, int(t_up / demo.dt))
    b = minjerk(above, p0, int(t_down / demo.dt))
    pre = np.vstack([a, b[1:]])
    n = len(pre)
    d = copy.copy(demo)
    d.pinch = np.vstack([pre, demo.pinch])
    d.grip = np.r_[np.zeros(n), demo.grip]
    d.yaw = np.r_[np.full(n, demo.yaw[0]), demo.yaw]
    d.aperture = np.r_[np.full(n, demo.aperture[0]), demo.aperture]
    d.t = np.arange(len(d.pinch)) * demo.dt
    d.i_grasp = demo.i_grasp + n
    d.i_release = demo.i_release + n
    d.name = demo.name + "_reach"
    return d


class KNNPolicy:
    """Retrieval policy (VINN-style): the next chunk is the distance-weighted average of the chunks
    that followed the k most similar *object-relative* states in the demonstrations. No training,
    nothing to overfit with 3 demos, and the output is always a blend of real human motion."""
    SCALE = np.array([0.08, 0.04, 0.04, 0.04, 0.25, 0.25, 0.04, 0.2])   # per-feature tolerance [m, s]

    def __init__(self, k=8):
        self.k = k

    def fit(self, demos, cube_size):
        X, Yp, Yg = build_dataset(demos, cube_size, obs_noise=0.0, copies=1)
        self.X = (X.numpy() / self.SCALE).astype(np.float32)
        self.Yp, self.Yg = Yp.numpy(), Yg.numpy()
        return self

    @staticmethod
    def load(path, k=8):
        z = np.load(path)
        p = KNNPolicy(k)
        p.X = (z["X"] / p.SCALE).astype(np.float32); p.Yp, p.Yg = z["Yp"], z["Yg"]
        return p

    def act(self, obs_np):
        o = np.atleast_2d(obs_np) / self.SCALE
        d = np.linalg.norm(self.X - o[0], axis=1)
        idx = np.argpartition(d, self.k)[: self.k]
        w = 1.0 / (d[idx] + 1e-2)
        w /= w.sum()
        return (w[:, None, None] * self.Yp[idx]).sum(0)[None], (w[:, None] * self.Yg[idx]).sum(0)[None]
