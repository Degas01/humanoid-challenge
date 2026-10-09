"""Run a joint-space trajectory in MuJoCo at 500 Hz and log what the motors feel."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import arm_act, arm_qadr, reset


@dataclass
class Rollout:
    t: np.ndarray            # physics time stamps [s] (500 Hz)
    tau: np.ndarray          # arm actuator forces/torques [N m], (T, 7)
    q: np.ndarray            # arm joint positions, (T, 7)
    tcp: np.ndarray          # tcp position, (T, 3)
    cube: np.ndarray         # cube position, (T, 3)
    success: bool
    frames: list | None = None


def task_success(model, data, emb) -> bool:
    cube = data.body("cube").xpos
    bowl = data.body("bowl").xpos
    r = model.geom("bowl_base").size[0]
    in_xy = np.linalg.norm(cube[:2] - bowl[:2]) < r * 0.9
    in_z = cube[2] < 0.12
    # released: cube is not moving with the tcp
    return bool(in_xy and in_z)


def run(model, data, emb, q_traj, grip_traj, ctrl_dt, settle=0.6, render=None,
        render_every=1 / 30, cam="front", hold_end=0.8):
    """q_traj: (N,7) joint targets at ctrl_dt; grip_traj: (N,) in [0,1]."""
    reset(model, data, emb)
    act = arm_act(model, emb)
    qadr = arm_qadr(model, emb)
    data.qpos[qadr] = q_traj[0]
    data.ctrl[act] = q_traj[0]
    emb.gripper_ctrl(model, data, grip_traj[0])
    mujoco.mj_forward(model, data)
    # settle objects on the table
    n_settle = int(settle / model.opt.timestep)
    for _ in range(n_settle):
        mujoco.mj_step(model, data)

    dt = model.opt.timestep
    T_total = (len(q_traj) - 1) * ctrl_dt + hold_end
    n = int(T_total / dt)
    t_ctrl = np.arange(len(q_traj)) * ctrl_dt
    taus, qs, tcps, cubes, ts, frames = [], [], [], [], [], []
    next_frame = 0.0
    site = model.site(emb.tcp_site).id
    for k in range(n):
        t = k * dt
        # first-order hold between control samples (what a real joint interface does)
        qk = np.array([np.interp(t, t_ctrl, q_traj[:, j]) for j in range(q_traj.shape[1])])
        gk = np.interp(t, t_ctrl, grip_traj)
        data.ctrl[act] = qk
        emb.gripper_ctrl(model, data, gk)
        mujoco.mj_step(model, data)
        taus.append(data.actuator_force[act].copy())
        qs.append(data.qpos[qadr].copy())
        tcps.append(data.site_xpos[site].copy())
        cubes.append(data.body("cube").xpos.copy())
        ts.append(t)
        if render is not None and t >= next_frame:
            render.update_scene(data, cam)
            frames.append(render.render().copy())
            next_frame += render_every
    return Rollout(np.array(ts), np.array(taus), np.array(qs), np.array(tcps), np.array(cubes),
                   task_success(model, data, emb), frames if render is not None else None)
