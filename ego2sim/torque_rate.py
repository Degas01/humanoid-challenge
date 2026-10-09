"""Torque-rate regularization as a differentiable layer.

Human video is jittery (MediaPipe noise is a few mm frame-to-frame at 30 Hz).
Feed that straight into a joint-position servo and every 33 ms the setpoint
kinks, which the motor sees as a torque step: chattering. A low-pass filter
removes the jitter but also rounds off the moments that matter (the grasp).

Instead, we look for the joint trajectory Q that

    min_Q   sum_t || W_t^1/2 J_t (q_t - q_ref,t) ||^2            task-space tracking
          + lam   sum_t || M_t (q_{t+1} - 3 q_t + 3 q_{t-1} - q_{t-2}) / dt^3 ||^2 dt^?

The second term is the rate of change of the inertial torque M(q) q_ddot,
i.e. a torque-rate penalty expressed in the robot's own mass matrix: heavy
proximal joints are smoothed more than the light wrist, and the shoulder of a
G1 is treated differently from the shoulder of a Panda without any re-tuning.
J_t and M_t come from MuJoCo at the reference configuration (one
re-linearisation is usually enough).

The problem is quadratic, so the solution is one linear solve; written in
PyTorch it is differentiable end to end and can sit at the output of a policy
(see policy.py), projecting each predicted action chunk onto smooth-torque
trajectories during both training and inference.
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from .scene import arm_dadr, arm_qadr


def third_diff_matrix(N: int) -> torch.Tensor:
    """(N-3, N) finite-difference operator for the third derivative (jerk)."""
    D = torch.zeros(N - 3, N, dtype=torch.float64)
    for i in range(N - 3):
        D[i, i:i + 4] = torch.tensor([-1.0, 3.0, -3.0, 1.0], dtype=torch.float64)
    return D


class TorqueRateLayer(torch.nn.Module):
    """Differentiable projection of a joint trajectory onto low torque-rate trajectories.

    forward(q_ref (B,N,n), J (B,N,6,n) or None, M (B,N,n,n) or None, w (B,N)) -> q (B,N,n)
    """

    def __init__(self, lam: float = 1e-6, dt: float = 1 / 30, anchor: int = 2,
                 anchor_weight: float = 1e3, mu: float = 1e-2):
        super().__init__()
        self.lam, self.dt, self.anchor, self.anchor_weight = lam, dt, anchor, anchor_weight
        # small joint-space term: the task metric J^T J is blind to the null space of a
        # 7-DoF arm; without it the solution drifts there and the linearisation breaks
        self.mu = mu

    def forward(self, q_ref, J=None, M=None, w=None, q_start=None):
        B, N, n = q_ref.shape
        dtype = torch.float64
        q_ref = q_ref.to(dtype)
        # --- tracking metric A_t = J_t^T W J_t (falls back to identity in joint space)
        if J is None:
            A = torch.eye(n, dtype=dtype).expand(B, N, n, n)
        else:
            J = J.to(dtype)
            A = J.transpose(-1, -2) @ J + self.mu * torch.eye(n, dtype=dtype)
        if w is not None:
            A = A * w.to(dtype)[..., None, None]
        # --- inertia-weighted jerk -> torque rate
        if M is None:
            M = torch.eye(n, dtype=dtype).expand(B, N, n, n)
        M = M.to(dtype)
        D = third_diff_matrix(N) / self.dt ** 3                      # (N-3, N)
        # block operator: (D kron I) then per-row multiply by M at the centre sample
        Mc = M[:, 1:N - 2]                                           # (B, N-3, n, n)
        # build dense H = blockdiag(A) + lam * dt * (D kron I)^T Mc^T Mc (D kron I)
        H = torch.zeros(B, N * n, N * n, dtype=dtype)
        for t in range(N):
            H[:, t * n:(t + 1) * n, t * n:(t + 1) * n] += A[:, t]
        MtM = Mc.transpose(-1, -2) @ Mc                               # (B, N-3, n, n)
        # (D^T diag(MtM) D) in block form: sum_i D[i,a] D[i,b] MtM_i
        for i in range(N - 3):
            idx = range(i, i + 4)
            for a in idx:
                for b in idx:
                    c = D[i, a] * D[i, b] * self.lam * self.dt
                    H[:, a * n:(a + 1) * n, b * n:(b + 1) * n] += c * MtM[:, i]
        rhs = (A @ q_ref[..., None]).reshape(B, N * n)
        # pin the first samples to the current robot state (no jump at t=0)
        if q_start is not None:
            for t in range(self.anchor):
                sl = slice(t * n, (t + 1) * n)
                H[:, sl, sl] += self.anchor_weight * torch.eye(n, dtype=dtype)
                rhs[:, sl] += self.anchor_weight * q_start.to(dtype)
        q = torch.linalg.solve(H, rhs[..., None])[..., 0]
        return q.reshape(B, N, n)


def linearize(model, emb, Q: np.ndarray):
    """Task Jacobians (6 x n) and arm mass matrices (n x n) along a joint trajectory."""
    d = mujoco.MjData(model)
    qa, da = arm_qadr(model, emb), arm_dadr(model, emb)
    site = model.site(emb.tcp_site).id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    Mfull = np.zeros((model.nv, model.nv))
    Js, Ms = [], []
    w_rot = emb.rot_weight
    for q in Q:
        d.qpos[qa] = q
        mujoco.mj_forward(model, d)
        mujoco.mj_jacSite(model, d, jp, jr, site)
        Js.append(np.vstack([jp[:, da], w_rot * jr[:, da]]))
        mujoco.mj_fullM(model, d, Mfull)
        Ms.append(Mfull[np.ix_(da, da)].copy())
    return np.array(Js), np.array(Ms)


def _assemble_sparse(J, M, w, lam, dt, q_ref, q_start=None, anchor=2, anchor_weight=1e3,
                     mu=1e-2):
    """Same quadratic program as TorqueRateLayer, assembled as a sparse block-banded
    system so a whole demonstration (thousands of samples) is solved at once."""
    import scipy.sparse as sp
    N, n = q_ref.shape
    A = (np.einsum("tki,tkj->tij", J, J) + mu * np.eye(n)) * w[:, None, None]
    rows, cols, vals = [], [], []

    def add_block(r, c, B):
        ii, jj = np.meshgrid(np.arange(n) + r * n, np.arange(n) + c * n, indexing="ij")
        rows.append(ii.ravel()); cols.append(jj.ravel()); vals.append(B.ravel())

    for t in range(N):
        add_block(t, t, A[t])
    d = np.array([-1.0, 3.0, -3.0, 1.0]) / dt ** 3
    for i in range(N - 3):
        Mc = M[i + 1]
        MtM = Mc.T @ Mc * lam * dt
        for x in range(4):
            for y in range(4):
                add_block(i + x, i + y, d[x] * d[y] * MtM)
    rhs = np.einsum("tij,tj->ti", A, q_ref).ravel()
    if q_start is not None:
        for t in range(anchor):
            add_block(t, t, anchor_weight * np.eye(n))
            rhs[t * n:(t + 1) * n] += anchor_weight * q_start
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N * n, N * n))
    return H, rhs


def regularize(model, emb, Q_ref: np.ndarray, dt: float, lam: float, weights=None,
               q_start=None, relinearize: int = 1):
    """Offline (non-differentiable) solve of the TorqueRateLayer problem on a full
    trajectory. Re-linearises J and M around the solution `relinearize` times."""
    from scipy.sparse.linalg import spsolve
    N, n = Q_ref.shape
    w = np.ones(N) if weights is None else np.asarray(weights, float)
    Q_lin = Q_ref.copy()
    for _ in range(relinearize + 1):
        J, M = linearize(model, emb, Q_lin)
        H, rhs = _assemble_sparse(J, M, w, lam, dt, Q_ref, q_start)
        Q_lin = spsolve(H.tocsc(), rhs).reshape(N, n)
    return Q_lin


def upsample(Q: np.ndarray, factor: int) -> np.ndarray:
    """Linear resampling of a (N, n) trajectory to `factor` times the rate."""
    N = len(Q)
    t = np.arange(N)
    tt = np.linspace(0, N - 1, (N - 1) * factor + 1)
    return np.stack([np.interp(tt, t, Q[:, j]) for j in range(Q.shape[1])], 1)
