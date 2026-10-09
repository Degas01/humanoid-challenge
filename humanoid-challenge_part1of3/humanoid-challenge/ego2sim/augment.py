"""Object-centric re-anchoring of human demonstrations (in the spirit of MimicGen).

A demo is split at the grasp and release events into three segments:
  reach    rest -> cube        follows the cube
  carry    cube -> bowl        starts at the cube, ends at the bowl
  retreat  bowl -> rest        leaves the bowl, returns to the original rest
For a new layout (cube', bowl') each segment is offset by a smooth blend of the
object displacements, so the hand still closes exactly on the new cube and
opens exactly over the new bowl, while keeping the timing and the
idiosyncrasies of my real motion (approach curvature, lift height, speed).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .demo import HumanDemo


def _ramp(n):
    s = np.linspace(0, 1, max(n, 1))
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def reanchor(demo: HumanDemo, cube_xy, bowl_xy, cube_yaw=None) -> HumanDemo:
    N = len(demo.pinch)
    ig, ir = demo.i_grasp, demo.i_release
    dc = np.r_[np.asarray(cube_xy) - demo.cube_xy, 0.0]
    db = np.r_[np.asarray(bowl_xy) - demo.bowl_xy, 0.0]
    off = np.zeros((N, 3))
    off[:ig] = _ramp(ig)[:, None] * dc
    off[ig:ir + 1] = dc + _ramp(ir + 1 - ig)[:, None] * (db - dc)
    off[ir + 1:] = (1 - _ramp(N - ir - 1))[:, None] * db
    yaw = demo.yaw.copy()
    cy = demo.cube_yaw if cube_yaw is None else cube_yaw
    if cube_yaw is not None:
        yaw[:ig + 1] += _ramp(ig + 1) * (cube_yaw - demo.cube_yaw)
        yaw[ig + 1:] += cube_yaw - demo.cube_yaw
    return replace(demo, name=demo.name + "_aug", pinch=demo.pinch + off, yaw=yaw,
                   cube_xy=np.asarray(cube_xy, float), bowl_xy=np.asarray(bowl_xy, float),
                   cube_yaw=float(cy))


def sample_layout(rng, demos, margin=0.03, min_sep=0.13):
    """Sample a new (cube, bowl) layout inside the region my demos covered."""
    C = np.array([d.cube_xy for d in demos])
    B = np.array([d.bowl_xy for d in demos])
    lo_c, hi_c = C.min(0) - margin, C.max(0) + margin
    lo_b, hi_b = B.min(0) - margin, B.max(0) + margin
    for _ in range(1000):
        c = rng.uniform(lo_c, hi_c)
        b = rng.uniform(lo_b, hi_b)
        if np.linalg.norm(c - b) > min_sep:
            return c, b
    return c, b
