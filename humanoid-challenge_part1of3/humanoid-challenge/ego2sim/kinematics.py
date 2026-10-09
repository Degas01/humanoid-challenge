"""Damped-least-squares IK on a scratch MjData (never touches the sim state)."""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import arm_dadr, arm_qadr


class IK:
    def __init__(self, model, emb, damping=0.03, null_gain=0.05):
        self.m, self.emb = model, emb
        self.d = mujoco.MjData(model)
        self.qadr = arm_qadr(model, emb)
        self.dadr = arm_dadr(model, emb)
        jids = [model.joint(j).id for j in emb.arm_joints]
        self.lo = model.jnt_range[jids, 0] + 1e-3
        self.hi = model.jnt_range[jids, 1] - 1e-3
        self.site = model.site(emb.tcp_site).id
        self.damping, self.null_gain = damping, null_gain
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self, q):
        self.fk(q)
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
        return self.jacp[:, self.dadr].copy(), self.jacr[:, self.dadr].copy()

    def solve(self, p_des, R_des, q0, iters=60, tol=1e-4, rot_weight=None):
        w = self.emb.rot_weight if rot_weight is None else rot_weight
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        for _ in range(iters):
            p, R = self.fk(q)
            ep = p_des - p
            er = np.zeros(3)
            if R_des is not None and w > 0:
                # rotation error as axis-angle of R_des R^T
                qe = np.zeros(4); qd = np.zeros(4); qc = np.zeros(4)
                mujoco.mju_mat2Quat(qd, R_des.flatten()); mujoco.mju_mat2Quat(qc, R.flatten())
                mujoco.mju_negQuat(qc, qc); mujoco.mju_mulQuat(qe, qd, qc)
                mujoco.mju_quat2Vel(er, qe, 1.0)
            err = np.concatenate([ep, w * er])
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
                break
            Jp, Jr = self.jac(q)
            J = np.vstack([Jp, w * Jr])
            JJt = J @ J.T + self.damping ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            # null-space pull toward home posture keeps solutions consistent across frames
            N = np.eye(len(q)) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (self.null_gain * (self.emb.home_q - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, _ = self.fk(q)
        return q, float(np.linalg.norm(p_des - p))
