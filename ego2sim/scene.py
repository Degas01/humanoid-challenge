"""MuJoCo scenes: one tabletop, two embodiments.

World frame == the *table frame* recovered from the phone video: origin at the
centre of the printed ArUco marker, x pointing away from the demonstrator,
y to their left, z up, table surface at z = 0. Because the human data are
metric and expressed in this same frame, a cube that the person grasped at
(x, y) in their kitchen is spawned at (x, y) in simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .paths import ASSETS, menagerie

FLOOR_Z = -0.76


@dataclass
class TaskObjects:
    cube_size: float = 0.05            # edge length [m]
    cube_xy: tuple = (0.0, -0.05)
    cube_yaw: float = 0.0
    bowl_diam: float = 0.16            # outer diameter [m]
    bowl_height: float = 0.06
    bowl_xy: tuple = (0.0, 0.18)
    marker_size: float = 0.12


def _add_table(spec: mujoco.MjSpec, obj: TaskObjects):
    wb = spec.worldbody
    spec.option.timestep = 0.002            # 500 Hz physics
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.quality.shadowsize = 4096

    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.88, 0.92], rgb2=[0.25, 0.28, 0.33], width=512, height=512)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.32, 0.34, 0.38], rgb2=[0.27, 0.29, 0.33], width=512, height=512)
    spec.add_material(name="floor", textures=["", "floor"], texrepeat=[6, 6])
    spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[0.78, 0.66, 0.50], rgb2=[0.70, 0.58, 0.44], width=64, height=64)
    spec.add_material(name="table", rgba=[0.62, 0.52, 0.40, 1], specular=0.1)
    spec.add_texture(name="aruco", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=str(ASSETS / "table_marker_id0.png"))
    spec.add_material(name="aruco", textures=["", "aruco"])

    wb.add_light(pos=[0.3, -0.6, 1.6], dir=[-0.2, 0.4, -1], castshadow=True,
                 diffuse=[0.55, 0.55, 0.55], specular=[0.1, 0.1, 0.1])
    wb.add_light(pos=[-0.8, 0.8, 1.4], dir=[0.5, -0.5, -1], castshadow=False,
                 diffuse=[0.25, 0.25, 0.28], specular=[0, 0, 0])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05],
                pos=[0, 0, FLOOR_Z], material="floor")
    # table: top surface at z = 0
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.37, 0.55, 0.02],
                pos=[0.15, 0.0, -0.02], material="table", friction=[1.0, 0.005, 0.0001])
    for sx in (-0.19, 0.49):
        for sy in (-0.5, 0.5):
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, (0.75 - 0.04) / 2],
                        pos=[sx, sy, FLOOR_Z / 2 - 0.02], rgba=[0.55, 0.45, 0.35, 1],
                        contype=0, conaffinity=0)
    # printed marker (visual only); texture "up" = +x (away from demonstrator)
    half = obj.marker_size / 2 + 0.01
    wb.add_geom(name="marker", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.0005],
                pos=[0, 0, 0.0005], quat=_quat_z(np.pi / 2), material="aruco",
                contype=0, conaffinity=0)


def _quat_z(a):
    return [np.cos(a / 2), 0, 0, np.sin(a / 2)]


def _add_objects(spec: mujoco.MjSpec, obj: TaskObjects):
    wb = spec.worldbody
    h = obj.cube_size / 2
    cube = wb.add_body(name="cube", pos=[obj.cube_xy[0], obj.cube_xy[1], h + 0.001],
                       quat=_quat_z(obj.cube_yaw))
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h],
                  rgba=[0.86, 0.24, 0.20, 1], mass=0.06, friction=[1.5, 0.01, 0.0002],
                  condim=4, solref=[0.01, 1])

    bowl = wb.add_body(name="bowl", pos=[obj.bowl_xy[0], obj.bowl_xy[1], 0])
    r, H, t = obj.bowl_diam / 2, obj.bowl_height, 0.006
    col = [0.20, 0.45, 0.80, 1]
    bowl.add_geom(name="bowl_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, t / 2],
                  pos=[0, 0, t / 2], rgba=col)
    n = 20
    seg = 2 * r * np.tan(np.pi / n) * 1.05
    for i in range(n):
        a = 2 * np.pi * i / n
        bowl.add_geom(name=f"bowl_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[t / 2, seg / 2, H / 2],
                      pos=[(r - t / 2) * np.cos(a), (r - t / 2) * np.sin(a), H / 2],
                      quat=_quat_z(a), rgba=col)
    bowl.add_site(name="bowl_center", pos=[0, 0, t], size=[0.005, 0, 0])


def _add_cameras(spec: mujoco.MjSpec, ego_from: np.ndarray | None = None):
    wb = spec.worldbody
    # A "pseudo-egocentric" view similar to the phone placement in the recordings
    eye = np.array([-0.55, -0.35, 0.55]) if ego_from is None else np.asarray(ego_from)
    _look_at_camera(wb, "ego", eye, np.array([0.05, 0.02, 0.0]), fovy=55)
    _look_at_camera(wb, "front", np.array([1.15, -0.05, 0.55]), np.array([0.0, 0.0, 0.05]), fovy=45)
    _look_at_camera(wb, "side", np.array([0.15, -1.15, 0.45]), np.array([0.0, 0.05, 0.05]), fovy=50)


def _look_at_camera(wb, name, eye, target, fovy=45):
    f = target - eye
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    # MuJoCo camera looks along -z, x right, y up
    R = np.stack([r, u, -f], axis=1)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten())
    wb.add_camera(name=name, pos=eye.tolist(), quat=q.tolist(), fovy=fovy)


# ----------------------------------------------------------------------------
# Embodiments
# ----------------------------------------------------------------------------
@dataclass
class Embodiment:
    """Everything the retargeter needs to know about a robot."""
    name: str
    arm_joints: list
    arm_actuators: list
    tcp_site: str
    home_q: np.ndarray
    # human-table-frame -> robot-table-frame affine map (per-axis scale, offset)
    ws_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    ws_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # weight of orientation in IK
    rot_weight: float = 0.3
    closing_col: int = 1          # column of tcp_target_rot along which the fingers close
    extra_qpos: dict = field(default_factory=dict)

    def map_point(self, p_human):
        return np.asarray(p_human) * self.ws_scale + self.ws_offset

    # set by attach()
    def gripper_ctrl(self, model, data, closed: float):
        raise NotImplementedError

    def tcp_target_rot(self, yaw: float) -> np.ndarray:
        raise NotImplementedError


class Panda(Embodiment):
    BASE = np.array([-0.52, 0.0, 0.0])

    def __init__(self):
        super().__init__(
            name="panda",
            arm_joints=[f"panda/joint{i}" for i in range(1, 8)],
            arm_actuators=[f"panda/actuator{i}" for i in range(1, 8)],
            tcp_site="panda/tcp",
            home_q=np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]),
            rot_weight=0.35,
        )

    def attach(self, spec: mujoco.MjSpec):
        child = mujoco.MjSpec.from_file(menagerie("franka_emika_panda/panda.xml"))
        hand = child.body("hand")
        hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.006, 0, 0], rgba=[0, 1, 0, 0.0])
        # better finger friction for a small cube
        for g in child.geoms:
            if g.parent.name in ("left_finger", "right_finger"):
                g.friction = [1.6, 0.01, 0.0002]
        for k in list(child.keys):
            child.delete(k)
        # pedestal from the floor up to the table height
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.11, 0.11, -FLOOR_Z / 2],
                                pos=[self.BASE[0], 0, FLOOR_Z / 2], rgba=[0.2, 0.22, 0.25, 1])
        frame = spec.worldbody.add_frame(pos=self.BASE.tolist())
        spec.attach(child, prefix="panda/", frame=frame)

    def gripper_ctrl(self, model, data, closed: float):
        aid = model.actuator("panda/actuator8").id
        data.ctrl[aid] = 255.0 * (1.0 - np.clip(closed, 0, 1))

    def tcp_target_rot(self, yaw: float) -> np.ndarray:
        # gripper pointing down (tcp z = -world z), fingers' closing axis set by yaw
        c, s = np.cos(yaw), np.sin(yaw)
        x = np.array([c, s, 0.0])
        z = np.array([0, 0, -1.0])
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)


class G1Humanoid(Embodiment):
    """Unitree G1 with Dex3 hands, pelvis welded behind the table.

    The humanoid's right arm is much shorter than a Panda and its shoulder
    sits low relative to the table, so the human workspace is mapped into the
    G1's reachable volume with an anisotropic scale + offset. The *scene*
    (cube, bowl) is mapped with the same transform, so the task stays
    geometrically consistent - this is the embodiment-specific part of the
    retargeting, everything else is shared with the Panda.
    """
    PELVIS = np.array([-0.34, 0.0, 0.03 - 0.793])  # pelvis body itself sits 0.793 above its frame

    HAND = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
    # right Dex3 hand: open / closed joint targets (found empirically, see scripts/tune_g1_hand.py)
    OPEN = np.array([0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    CLOSED = np.array([0.0, -0.45, -1.5, 0.8, 0.8, 0.8, 0.8])

    def __init__(self):
        arm = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
               "wrist_roll", "wrist_pitch", "wrist_yaw"]
        super().__init__(
            name="g1",
            arm_joints=[f"g1/right_{j}_joint" for j in arm],
            arm_actuators=[f"g1/right_{j}_joint" for j in arm],
            tcp_site="g1/tcp",
            home_q=np.array([0.36, -0.36, -0.20, -0.29, 0.42, 0.0, 0.17]),
            # task-frame workspace of the recordings (x in [-0.11, 0.24], y in [-0.27, 0.21]) ->
            # G1 reach. A tighter fit to the measured reach box (0.49/0.52) was tried and was
            # worse (33 % success): the arm needs the slack, see README.
            ws_scale=np.array([0.55, 0.58, 1.0]),
            ws_offset=np.array([-0.085, -0.145, 0.0]),
            rot_weight=0.25,
            closing_col=0,
        )

    def attach(self, spec: mujoco.MjSpec):
        child = mujoco.MjSpec.from_file(menagerie("unitree_g1/g1_with_hands.xml"))
        child.delete(child.joint("floating_base_joint"))
        for k in list(child.keys):
            child.delete(k)
        palm = child.body("right_wrist_yaw_link")
        palm.add_site(name="tcp", pos=[0.118, 0.05, 0.008], size=[0.006, 0, 0], rgba=[0, 1, 0, 0.0])
        for g in child.geoms:
            if g.parent.name.startswith("right_hand"):
                g.friction = [1.8, 0.01, 0.0002]
        frame = spec.worldbody.add_frame(pos=self.PELVIS.tolist())
        spec.attach(child, prefix="g1/", frame=frame)

    def hand_actuators(self, model):
        return [model.actuator(f"g1/right_hand_{j}_joint").id for j in self.HAND]

    def gripper_ctrl(self, model, data, closed: float):
        c = np.clip(closed, 0, 1)
        data.ctrl[self.hand_actuators(model)] = (1 - c) * self.OPEN + c * self.CLOSED

    def tcp_target_rot(self, yaw: float) -> np.ndarray:
        # "karate-chop" lateral grasp: fingers point forward (wrist x), palm faces
        # the robot's left (wrist +y), index above middle (wrist z = up). The thumb
        # opposes the curled fingers along wrist x, so the cube is pinched
        # front/back - the natural way a right hand grabs a block on its left.
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


EMBODIMENTS = {"panda": Panda, "g1": G1Humanoid}


def phone_camera(T_cam_table: np.ndarray, K, image_size):
    """Sim camera placed exactly where my phone was (from the ArUco pose) with the
    calibrated vertical field of view, so sim renders line up with the real video."""
    R = T_cam_table[:3, :3]
    t = T_cam_table[:3, 3]
    pos = -R.T @ t
    R_wc = R.T @ np.diag([1.0, -1.0, -1.0])      # OpenCV (x r, y down, z fwd) -> MuJoCo camera
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_wc.flatten())
    fovy = float(np.degrees(2 * np.arctan(image_size[1] / 2 / np.asarray(K)[1][1])))
    return pos, q, fovy


def build(embodiment: Embodiment, obj: TaskObjects, ego_from=None, phone_cam=None):
    spec = mujoco.MjSpec()
    spec.modelname = f"ego2sim_{embodiment.name}"
    _add_table(spec, obj)
    _add_objects(spec, obj)
    _add_cameras(spec, ego_from)
    if phone_cam is not None:
        pos, q, fovy = phone_cam
        spec.worldbody.add_camera(name="phone", pos=list(pos), quat=list(q), fovy=fovy)
    embodiment.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    reset(model, data, embodiment)
    return model, data


def reset(model, data, emb: Embodiment):
    mujoco.mj_resetData(model, data)
    # hold every actuated joint at its current (zero) position, then arm at home
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            j = model.actuator_trnid[i, 0]
            data.ctrl[i] = data.qpos[model.jnt_qposadr[j]]
    set_arm_q(model, data, emb, emb.home_q)
    emb.gripper_ctrl(model, data, 0.0)
    if isinstance(emb, G1Humanoid):
        # let the hand settle to its open pose
        for jn, v in zip(emb.HAND, emb.OPEN):
            data.qpos[model.joint(f"g1/right_hand_{jn}_joint").qposadr[0]] = v
        # relaxed left arm (otherwise it points straight ahead)
        for jn, v in [("left_shoulder_pitch", 0.25), ("left_shoulder_roll", 0.25), ("left_elbow", 0.9)]:
            j = model.joint(f"g1/{jn}_joint")
            data.qpos[j.qposadr[0]] = v
            data.ctrl[model.actuator(f"g1/{jn}_joint").id] = v
    mujoco.mj_forward(model, data)


def arm_qadr(model, emb):
    return np.array([model.joint(j).qposadr[0] for j in emb.arm_joints])


def arm_dadr(model, emb):
    return np.array([model.joint(j).dofadr[0] for j in emb.arm_joints])


def arm_act(model, emb):
    return np.array([model.actuator(a).id for a in emb.arm_actuators])


def set_arm_q(model, data, emb, q):
    data.qpos[arm_qadr(model, emb)] = q
    data.ctrl[arm_act(model, emb)] = q
