"""Marker (table) frame of the phone recordings  ->  task frame of the simulation.

The ArUco marker defines the recording frame (origin at the marker, bowl 45 cm away along -y).
The simulated robots were built around a task frame with the bowl on +y at about (0, 0.18), so
the real data is rotated by 180 deg about the vertical and shifted: a rigid motion (distances,
handedness and heights are preserved)."""
import numpy as np

SHIFT = np.array([-0.04, -0.25])


def to_task_xy(p):
    p = np.asarray(p, float)
    return np.stack([-p[..., 0], -p[..., 1]], -1) + SHIFT


def to_task(p):
    p = np.asarray(p, float)
    return np.concatenate([to_task_xy(p[..., :2]), p[..., 2:]], -1)


def demo_to_task(d):
    """Return a copy of a HumanDemo expressed in the task frame."""
    import copy
    e = copy.copy(d)
    e.pinch = to_task(d.pinch)
    e.cube_xy = to_task_xy(d.cube_xy)
    e.bowl_xy = to_task_xy(d.bowl_xy)
    return e                                  # yaw: a 180 deg rotation is invisible mod 90 deg
