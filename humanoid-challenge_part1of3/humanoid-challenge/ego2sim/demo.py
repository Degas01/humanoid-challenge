"""Table-frame hand landmarks  ->  a robot-agnostic demonstration.

A demonstration is just: pinch point p(t) in metres, gripper "closedness" g(t)
and a grasp yaw. The object poses are NOT detected by any vision model:
the cube is wherever the fingers closed, the bowl is wherever they opened.
The hand is the object detector.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import medfilt

from .perception import INDEX_TIP, PALM, THUMB_TIP, Track, estimate_hand_scale, to_table


@dataclass
class HumanDemo:
    name: str
    dt: float
    t: np.ndarray          # (N,)
    pinch: np.ndarray      # (N,3) midpoint of thumb & index tips, table frame [m]
    aperture: np.ndarray   # (N,)  thumb-index distance [m]
    yaw: np.ndarray        # (N,)  pinch-axis yaw folded to [-pi/4, pi/4] (cube symmetry)
    grip: np.ndarray       # (N,)  0 open / 1 closed
    i_grasp: int
    i_release: int
    cube_xy: np.ndarray
    cube_yaw: float
    bowl_xy: np.ndarray
    hand_scale: float
    qa: dict               # physical-consistency checks (see README)

    def to_npz(self, path):
        d = asdict(self)
        qa = d.pop("qa")
        np.savez_compressed(path, **d, qa=np.array([json.dumps(qa)]))

    @staticmethod
    def load(path):
        z = np.load(path)
        kw = {k: z[k] for k in z.files if k != "qa"}
        for k in ("name",):
            kw[k] = str(kw[k])
        for k in ("dt", "cube_yaw", "hand_scale"):
            kw[k] = float(kw[k])
        for k in ("i_grasp", "i_release"):
            kw[k] = int(kw[k])
        return HumanDemo(**kw, qa=json.loads(str(z["qa"][0])))


def _fill_nan(x):
    x = x.copy()
    idx = np.arange(len(x))
    for k in range(x.shape[1]):
        good = ~np.isnan(x[:, k])
        if good.sum() >= 2:
            x[:, k] = np.interp(idx, idx[good], x[good, k])
    return x


def _hampel(x, k=5, n_sigma=3.0):
    """Remove isolated perception glitches (single-frame jumps) but keep jitter:
    the jitter is exactly what the retargeting stage is meant to handle."""
    y = x.copy()
    for j in range(x.shape[1]):
        med = medfilt(x[:, j], 2 * k + 1)
        mad = medfilt(np.abs(x[:, j] - med), 2 * k + 1) * 1.4826 + 1e-6
        bad = np.abs(x[:, j] - med) > n_sigma * mad
        y[bad, j] = med[bad]
    return y


def fold_yaw(a):
    """Cube has 90-degree symmetry and a parallel gripper 180: fold into [-pi/4, pi/4]."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def segment(name: str, track: Track, cube_size: float, rest_s: float = 0.8) -> HumanDemo:
    scale, resid = estimate_hand_scale(track, rest_s=rest_s)
    L = to_table(track, scale)
    ok = track.hand_ok
    L[~ok] = np.nan
    flat = L.reshape(len(L), -1)
    valid_frac = float(ok.mean())
    flat = _fill_nan(flat)
    flat = _hampel(flat)
    L = flat.reshape(-1, 21, 3)

    thumb, index = L[:, THUMB_TIP], L[:, INDEX_TIP]
    pinch = 0.5 * (thumb + index)
    ap = np.linalg.norm(thumb - index, axis=1)
    ax = index - thumb
    yaw = fold_yaw(np.arctan2(ax[:, 1], ax[:, 0]) - np.pi / 2)  # gripper closes along pinch axis

    # --- grasp / release from the aperture, with hysteresis ----------------------
    ap_s = medfilt(ap, 7)
    close_thr = cube_size + 0.030      # fingertip thickness on both sides of the cube
    open_thr = cube_size + 0.045
    closed = np.zeros(len(ap), bool)
    state = False
    for i, a in enumerate(ap_s):
        state = (a < close_thr) if not state else (a < open_thr)
        closed[i] = state
    # keep the closed interval during which the pinch point is actually lifted
    segs, start = [], None
    for i, c in enumerate(np.r_[closed, False]):
        if c and start is None:
            start = i
        if not c and start is not None:
            segs.append((start, i - 1)); start = None
    lift = pinch[:, 2] - np.nanmin(pinch[:, 2])
    segs = [s for s in segs if lift[s[0]:s[1] + 1].max() > 0.04 and s[1] - s[0] > 0.3 * track.fps]
    if not segs:
        raise RuntimeError(f"{name}: no grasp found (min aperture {ap_s.min():.3f} m)")
    i_g, i_r = max(segs, key=lambda s: lift[s[0]:s[1] + 1].max() * (s[1] - s[0]))
    grip = np.zeros(len(ap)); grip[i_g:i_r + 1] = 1.0

    # --- the hand as object detector --------------------------------------------
    w = max(1, int(0.1 * track.fps))
    cube_xy = np.median(pinch[i_g:i_g + w, :2], axis=0)
    cube_yaw = float(np.median(yaw[i_g:i_g + w]))
    bowl_xy = np.median(pinch[max(0, i_r - w):i_r + 1, :2], axis=0)

    # --- physical consistency checks (no ground truth needed) -------------------
    qa = {
        "valid_hand_frames": round(valid_frac, 3),
        "marker_frames": round(float(track.marker_ok.mean()), 3),
        "hand_scale": round(scale, 3),
        "rest_palm_residual_mm": round(resid * 1000, 1),
        # fingers should close around the cube's centre height
        "grasp_height_err_mm": round(float((np.median(pinch[i_g:i_g + w, 2]) - cube_size / 2) * 1000), 1),
        # thumb-index distance while holding ~ cube + 2 fingertip half-thicknesses
        "hold_aperture_mm": round(float(np.median(ap[i_g:i_r + 1]) * 1000), 1),
        "max_lift_mm": round(float(lift[i_g:i_r + 1].max() * 1000), 1),
        "duration_s": round(len(ap) / track.fps, 2),
    }
    return HumanDemo(name, 1 / track.fps, track.t.copy(), pinch, ap, yaw, grip, int(i_g), int(i_r),
                     cube_xy, cube_yaw, bowl_xy, scale, qa)


# =============================================================================================
# Real phone data: the hand is partly out of frame and monocular PnP depth is unreliable
# (errors > 10 cm), so the demonstration is *grounded* in what the video measures well:
#   - pixel ray of the thumb/index pinch  -> table x,y at a known height
#   - cube / bowl poses from the hand-free first frame (scene_parse)  -> events + endpoints
#   - height follows the task structure: grasp at cube/2, carry arc over the rim, release above
#     the bowl floor.  QA reports how well the independent hand track agrees with the scene.
# =============================================================================================
def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def longest_run(ok, max_gap=8):
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return None
    runs, s, p = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - p > max_gap:
            runs.append((s, p)); s = k
        p = k
    runs.append((s, p))
    return max(runs, key=lambda r: r[1] - r[0])


def pinch_rays(track: Track, sel):
    """Unit camera-frame rays through the thumb/index midpoint, and the table pose per frame."""
    pin = 0.5 * (track.lm_cam[sel, THUMB_TIP] + track.lm_cam[sel, INDEX_TIP])
    pin = _fill_nan(pin)                      # gaps (<= 8 frames) inside the tracked run
    d = pin / np.linalg.norm(pin, axis=1, keepdims=True)
    return d, track.T_cam_table[sel, :3, :3], track.T_cam_table[sel, :3, 3]


def ray_to_table(d, R, t, z):
    """Intersect camera rays with the horizontal planes z (per frame) of the table frame."""
    R2 = R[:, :, 2]
    lam = (z + np.einsum("ti,ti->t", R2, t)) / np.einsum("ti,ti->t", R2, d)
    return np.einsum("tji,tj->ti", R, lam[:, None] * d - t)


def height_profile(n, ig, ir, cube, bowl_h, z_hover=0.12, arc=0.07):
    """Task-structured pinch height for frames 0..n-1 with grasp at ig and release at ir."""
    z = np.zeros(n)
    zg, zr = cube / 2, bowl_h + 0.035
    k = np.arange(n)
    pre = k <= ig
    z[pre] = zg + (z_hover - zg) * (1 - _smoothstep(k[pre] / max(ig, 1)))
    mid = (k > ig) & (k <= ir)
    u = (k[mid] - ig) / max(ir - ig, 1)
    z[mid] = zg + (zr - zg) * _smoothstep(u) + arc * np.sin(np.pi * u) ** 0.5 * (1 - 0.0 * u)
    post = k > ir
    z[post] = zr + (z_hover - zr) * _smoothstep((k[post] - ir) / max(n - 1 - ir, 1))
    return z


def segment_grounded(name: str, track: Track, objects: dict, cube_size: float, bowl_h: float) -> HumanDemo:
    """Demo from a real clip. `objects` = scene_parse.find_objects(...) (+ bowl fallback)."""
    if "cube_xy" not in objects:
        raise RuntimeError(f"{name}: cube not found in the first frame")
    run = longest_run(track.hand_ok)
    if run is None or run[1] - run[0] < 0.8 * track.fps:
        raise RuntimeError(f"{name}: hand tracked for <0.8 s")
    a, b = run
    sel = np.arange(a, b + 1)
    cube_xy, bowl_xy = np.asarray(objects["cube_xy"]), np.asarray(objects["bowl_xy"])
    d, R, t = pinch_rays(track, sel)
    d = _hampel(d, k=4)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    P0 = ray_to_table(d, R, t, np.full(len(sel), cube_size / 2))       # provisional, at grasp height
    dc = np.linalg.norm(P0[:, :2] - cube_xy, axis=1)
    db = np.linalg.norm(P0[:, :2] - bowl_xy, axis=1)
    if dc.min() > 0.05:
        raise RuntimeError(f"{name}: hand never reaches the cube (closest {dc.min()*1000:.0f} mm)")
    ir = int(np.argmin(db))
    if db[ir] > 0.05 or ir < 0.5 * track.fps:
        raise RuntimeError(f"{name}: hand track ends before the bowl (closest {db[ir]*1000:.0f} mm)")
    near = np.where(dc[:ir] < max(dc[:ir].min() + 0.015, 0.03))[0]
    ig = int(near[-1])                                               # last frame at the cube = lift-off
    z = height_profile(len(sel), ig, ir, cube_size, bowl_h)
    P = ray_to_table(d, R, t, z)
    P = np.c_[medfilt(P[:, 0], 5), medfilt(P[:, 1], 5), P[:, 2]]
    P[:, 2] = z
    # object-anchored correction: the hand's *shape* of motion is kept, but the two contact events
    # are snapped to the cube / bowl seen in the scene (the hand track is only ~2-4 cm accurate,
    # a 6 cm cube and a 12 cm bowl are not forgiving).  Offsets are blended smoothly in time.
    off_g = cube_xy - P[ig, :2]
    off_r = bowl_xy - P[ir, :2]
    k = np.arange(len(P))
    w_pre = _smoothstep(k / max(ig, 1))
    w_mid = _smoothstep((k - ig) / max(ir - ig, 1))
    off = np.where((k <= ig)[:, None], w_pre[:, None] * off_g,
                   np.where((k <= ir)[:, None], (1 - w_mid)[:, None] * off_g + w_mid[:, None] * off_r, off_r))
    P[:, :2] += off
    ap = np.linalg.norm(_fill_nan(track.lm_cam[sel, THUMB_TIP] - track.lm_cam[sel, INDEX_TIP]), axis=1)
    grip = np.zeros(len(sel)); grip[ig:ir + 1] = 1.0
    if dc[ig] > 0.05 or db[ir] > 0.035 or (ir - ig) < 1.2 * track.fps:
        raise RuntimeError(f"{name}: track does not cleanly cover grasp and release "
                           f"(cube {dc[ig]*1000:.0f} mm, bowl {db[ir]*1000:.0f} mm, carry {(ir-ig)/track.fps:.1f} s)")
    cube_yaw = float(fold_yaw(objects.get("cube_yaw_raw", 0.0)))
    yaw = np.full(len(sel), cube_yaw)
    qa = {
        "tracked_frames": int(len(sel)), "tracked_s": round(len(sel) / track.fps, 2),
        "hand_frames_total": round(float(track.hand_ok.mean()), 3),
        "pinch_vs_cube_mm": round(float(dc[ig] * 1000), 1),           # hand track vs the scene, at lift-off
        "pinch_vs_bowl_mm": round(float(db[ir] * 1000), 1),           # ... and at release
        "cube_side_seen_mm": round(float(objects.get("cube_side_est", np.nan) * 1000), 1),
        "bowl_diam_seen_mm": round(float(objects.get("bowl_d_est", np.nan) * 1000), 1),
        "hold_aperture_mm": round(float(np.median(ap[ig:ir + 1]) * 1000), 1),
        "carry_s": round((ir - ig) / track.fps, 2),
        "marker_frames": round(float(track.marker_ok.mean()), 3),
        "start_frame": int(a),
    }
    return HumanDemo(name, 1 / track.fps, track.t[sel] - track.t[a], P, ap, yaw, grip, ig, ir,
                     cube_xy.copy(), cube_yaw, bowl_xy.copy(), 1.0, qa)
