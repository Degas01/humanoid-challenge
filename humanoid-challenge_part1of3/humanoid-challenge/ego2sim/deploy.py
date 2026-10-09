"""Closed-loop deployment of the human-frame policy on a robot in MuJoCo.

Every 1/6 s:  observe -> policy chunk (K human-frame waypoints) -> workspace map
-> IK -> [TorqueRateLayer, anchored to the last two commanded joint samples so
velocity is continuous across replans] -> 90 Hz joint setpoints -> 500 Hz physics.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .demo import fold_yaw
from .execute import task_success
from .kinematics import IK
from .policy import anchor, make_obs
from .retarget import emb_grasp_z_offset, minjerk, table_floor
from .scene import arm_act, arm_qadr, reset
from .torque_rate import _assemble_sparse, linearize, upsample


def set_layout(model, data, emb, cube_xy_h, bowl_xy_h, cube_size, cube_yaw=0.0):
    """Place cube and bowl given *human-frame* positions."""
    c = emb.map_point([*cube_xy_h, 0])[:2]
    b = emb.map_point([*bowl_xy_h, 0])[:2]
    model.body_pos[model.body("bowl").id][:2] = b
    j = model.joint("cube_free")
    a = j.qposadr[0]
    data.qpos[a:a + 3] = [c[0], c[1], cube_size / 2 + 0.001]
    data.qpos[a + 3:a + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
    model.qpos0[a:a + 7] = data.qpos[a:a + 7]      # survives mj_resetData
    data.qvel[j.dofadr[0]:j.dofadr[0] + 6] = 0
    mujoco.mj_forward(model, data)


class Deployer:
    def __init__(self, model, data, emb, cube_size, lam=1e-5, smooth="torque", up=3, replan=5,
                 policy_dt=1 / 30, ema_hz=3.0, gate=True, gate_r=0.035, clearance_yaw=True, obs_cmd=True):
        """smooth: "none" | "ema" (causal 1st-order low-pass on joint setpoints, the
        standard online fix) | "torque" (TorqueRateLayer on every chunk)."""
        if smooth is True:
            smooth = "torque"
        if smooth is False:
            smooth = "none"
        self.m, self.d, self.emb = model, data, emb
        self.cube_size, self.lam, self.smooth, self.up = cube_size, lam, smooth, up
        self.ema_alpha = 1 - np.exp(-2 * np.pi * ema_hz * policy_dt / up)
        self.replan, self.pdt = replan, policy_dt
        self.gate, self.gate_r, self.clearance_yaw, self.obs_cmd = gate, gate_r, clearance_yaw, obs_cmd
        self.ik = IK(model, emb)
        self.act = arm_act(model, emb)
        self.qadr = arm_qadr(model, emb)
        self.zoff = emb_grasp_z_offset(emb)

    # human <-> robot task frames
    def to_robot(self, p):
        r = np.array(self.emb.map_point(p), float)
        floor = table_floor(r[..., :2], self.d.body("cube").xpos[:2],
                            self.m.body_pos[self.m.body("bowl").id][:2], 0.012 + self.zoff)
        r[..., 2] = np.maximum(r[..., 2] + self.zoff, floor)
        return r

    def to_human(self, p):
        h = (np.asarray(p) - self.emb.ws_offset) / self.emb.ws_scale
        h[..., 2] -= self.zoff
        return h

    def _run_cmds(self, Q, G, dt):
        """FOH-interpolate joint setpoints Q (n,7) at spacing dt and step physics."""
        m, d = self.m, self.d
        T = (len(Q) - 1) * dt
        nst = int(round(T / m.opt.timestep))
        tq = np.arange(len(Q)) * dt
        for k in range(1, nst + 1):
            t = k * m.opt.timestep
            d.ctrl[self.act] = [np.interp(t, tq, Q[:, j]) for j in range(Q.shape[1])]
            self.emb.gripper_ctrl(m, d, float(np.interp(t, tq, G)))
            mujoco.mj_step(m, d)
            self.log_tau.append(d.actuator_force[self.act].copy())
            self.log_tcp.append(d.site(self.emb.tcp_site).xpos.copy())
            if self.render is not None and d.time >= self.next_frame:
                self.render.update_scene(d, self.cam)
                self.frames.append(self.render.render().copy())
                self.next_frame += 1 / 30

    def _smooth(self, Qc):
        """TorqueRateLayer on the new chunk; the first three samples are pinned to the
        last three commanded setpoints, so position, velocity and acceleration are
        continuous across replans (the jerk stencil spans the seam)."""
        if self.smooth == "none":
            return Qc
        if self.smooth == "ema":
            out = Qc.copy()
            y = self.hist[-1].copy()
            for i in range(3, len(Qc)):
                y = y + self.ema_alpha * (Qc[i] - y)
                out[i] = y
            return out
        dt = self.pdt / self.up
        J, M = linearize(self.m, self.emb, Qc)
        w = np.ones(len(Qc))
        H, rhs = _assemble_sparse(J, M, w, self.lam, dt, Qc, None)
        n = Qc.shape[1]
        H = H.tolil()
        for t, qh in enumerate(self.hist[-3:]):
            for i in range(n):
                H[t * n + i, t * n + i] += 1e4
                rhs[t * n + i] += 1e4 * qh[i]
        from scipy.sparse.linalg import spsolve
        return np.clip(spsolve(H.tocsc(), rhs).reshape(-1, n), self.ik.lo, self.ik.hi)

    def run(self, policy, rest_h, T_max=14.0, render=None, cam="front", rng=None):
        m, d, emb = self.m, self.d, self.emb
        self.render, self.cam, self.frames, self.next_frame = render, cam, [], 0.0
        self.log_tau, self.log_tcp = [], []
        # settle objects
        cube_q = d.qpos.copy()
        bowl_pos = m.body_pos[m.body("bowl").id].copy()
        reset(m, d, emb)
        m.body_pos[m.body("bowl").id] = bowl_pos
        jc = m.joint("cube_free").qposadr[0]
        d.qpos[jc:jc + 7] = cube_q[jc:jc + 7]
        mujoco.mj_forward(m, d)
        for _ in range(300):
            mujoco.mj_step(m, d)
        d.time = 0.0
        # scripted move: home -> my average rest pose (the policy starts where I started)
        p0, _ = self.ik.fk(emb.home_q)
        P = minjerk(p0, self.to_robot(rest_h), 30)
        q = emb.home_q.copy(); Qs = []
        for p in P:
            q, _ = self.ik.solve(p, emb.tcp_target_rot(0.0), q, iters=40); Qs.append(q)
        Qs = upsample(np.array(Qs), self.up)
        self._run_cmds(Qs, np.zeros(len(Qs)), self.pdt / self.up)
        self.hist = [Qs[-3], Qs[-2], Qs[-1]]
        self.t0 = d.time
        cmd_h = np.asarray(rest_h, float).copy()   # last commanded pinch point (human frame)
        grip_cmd, grasped, t_release, t_grasp = 0.0, False, None, None
        self.yaw = 0.0
        dt_up = self.pdt / self.up
        while d.time < T_max:
            meas_h = self.to_human(d.site(emb.tcp_site).xpos)
            # proprioception = the *commanded* pinch point (as in most visuomotor policies): a robot that
            # is held back by contact or by its smaller workspace still 'arrives' where the human would
            tcp_h = cmd_h.copy() if self.obs_cmd else meas_h
            cube_h = self.to_human(d.body("cube").xpos.copy())
            cube_h[2] += self.zoff  # cube height is physical, not mapped
            # same representation as the human data: a held object moves with the hand
            if grip_cmd > 0.5 and np.linalg.norm(d.body("cube").xpos - d.site(emb.tcp_site).xpos) < 0.06:
                cube_h = tcp_h.copy()
            bowl_h = self.to_human(np.r_[m.body_pos[m.body("bowl").id][:2], 0])[:2]
            obs = make_obs(tcp_h, grip_cmd, cube_h, bowl_h, 0.0 if t_grasp is None else d.time - t_grasp)
            dp, g = policy.act(obs)
            way_h = anchor(grasped, np.r_[cube_h[:2], self.cube_size / 2], bowl_h) + dp[0]
            cmd_h = way_h[min(self.replan, len(way_h)) - 1].copy()
            way_r = self.to_robot(way_h)
            # gripper yaw follows the cube's (folded) yaw; the policy only does positions
            if not grasped:
                qc = d.body("cube").xquat
                y = 2 * np.arctan2(qc[3], qc[0])
                # equivalent cube yaw (mod 90 deg) closest to the current one: no flips
                self.yaw = float(self.yaw + fold_yaw(y - self.yaw))
                if self.clearance_yaw:
                    # clearance-aware grasp axis: of the cube's two grasp axes (90 deg apart), close the
                    # fingers across the cube->bowl direction, so no finger has to fit between them
                    u = m.body_pos[m.body("bowl").id][:2] - d.body("cube").xpos[:2]
                    u = np.r_[u / (np.linalg.norm(u) + 1e-9), 0]
                    cands = [self.yaw, self.yaw + np.pi / 2, self.yaw - np.pi / 2]
                    cost = [abs(emb.tcp_target_rot(c)[:, emb.closing_col] @ u) + 0.3 * abs(c - self.yaw) / np.pi
                            for c in cands]
                    self.yaw = float(cands[int(np.argmin(cost))])
            yaw = self.yaw
            Qc, q = [self.hist[-1]], self.hist[-1].copy()
            for p in way_r:
                q, _ = self.ik.solve(p, emb.tcp_target_rot(yaw), q, iters=30); Qc.append(q.copy())
            Qc = upsample(np.array(Qc), self.up)
            Qc = np.vstack([self.hist[-3][None], self.hist[-2][None], Qc])
            Qc = self._smooth(Qc)[2:]
            close_cmd = g[0] > 0.5
            if grasped and self.gate and np.linalg.norm(bowl_h - tcp_h[:2]) > self.gate_r:
                close_cmd = np.ones_like(close_cmd, bool)    # skill precondition: release only above the bowl
            Gc = np.r_[grip_cmd, close_cmd.astype(float)]
            Gc = np.repeat(Gc, self.up)[: len(Qc)]
            n_exec = self.replan * self.up + 1
            self._run_cmds(Qc[:n_exec], Gc[:n_exec], dt_up)
            self.hist = [Qc[n_exec - 3], Qc[n_exec - 2], Qc[n_exec - 1]]
            new_grip = Gc[n_exec - 1]
            if new_grip > 0.5:
                grasped = True      # from here on the wrist yaw is frozen
                if t_grasp is None:
                    t_grasp = d.time
            if grasped and new_grip < 0.5 and t_release is None:
                t_release = d.time
            grip_cmd = new_grip
            if t_release is not None:
                break
        if t_release is not None:
            # the policy's job ends at the release; wait for the hand to open (the Dex3 fingers take
            # ~0.5 s; lifting earlier flicked the cube out of the bowl), then a scripted 8 cm lift
            hold = np.repeat(self.hist[-1][None], int(0.6 / dt_up) + 1, 0)
            self._run_cmds(hold, np.zeros(len(hold)), dt_up)
            p_now = d.site(emb.tcp_site).xpos.copy()
            P = minjerk(p_now, p_now + [0, 0, 0.08], 20)
            q = self.hist[-1].copy(); Qs = [q.copy()]
            for p in P[1:]:
                q, _ = self.ik.solve(p, emb.tcp_target_rot(self.yaw), q, iters=30); Qs.append(q.copy())
            Qs = upsample(np.array(Qs), self.up)
            self._run_cmds(Qs, np.zeros(len(Qs)), dt_up)
            for _ in range(int(0.6 / m.opt.timestep)):
                mujoco.mj_step(m, d)
        return {"success": task_success(m, d, emb), "grasped": grasped,
                "time": float(d.time), "tau": np.array(self.log_tau),
                "tcp": np.array(self.log_tcp), "frames": self.frames}
