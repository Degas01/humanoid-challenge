"""First frame of a clip (no hand yet) -> where the cube and bowl are, in the table frame.

The table plane is known from the ArUco marker, so a frame can be re-projected onto any plane
parallel to the table (here: the cube's top face / the bowl's rim, both ~6 cm up) and the white
objects segmented in metric coordinates: a 60 mm square and a 120 mm circle.  This is
independent of the hand, so it is used (a) for the layouts of clips where the hand tracker
fails and (b) as ground truth to validate the hand-derived grasp/release points.
"""
from __future__ import annotations

import cv2
import numpy as np


def rectify(frame, K, dist, T_cam_table, z, x_rng=(-0.55, 0.45), y_rng=(-0.8, 0.5), res=0.002):
    xs = np.arange(*x_rng, res); ys = np.arange(*y_rng, res)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
    R, t = T_cam_table[:3, :3], T_cam_table[:3, 3]
    rv, _ = cv2.Rodrigues(R)
    px, _ = cv2.projectPoints(P, rv, t, np.asarray(K, float), np.asarray(dist, float))
    px = px.reshape(X.shape + (2,)).astype(np.float32)
    img = cv2.remap(frame, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderValue=0)
    return img, xs, ys


def _white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # under this phone's white balance the white box/bowl look blue-ish (hue ~105), wood is hue ~12
    h = hsv[..., 0]
    glare = (hsv[..., 2] > 236) & (hsv[..., 1] < 90)          # the lamp's specular blob
    m = ((h > 85) & (h < 135) & (hsv[..., 2] > 120) & ~glare).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_objects(frame, K, dist, T_cam_table, cube=0.06, bowl_d=0.12, bowl_h=0.065, marker=0.12,
                 frame_last=None, T_last=None):
    """Returns dict(cube_xy, cube_yaw, bowl_xy, bowl_d_est, cube_side_est, debug images)."""
    out = {}
    # ---- bowl: rim circle on the z = bowl_h plane -----------------------------------------
    img_b, xs, ys = rectify(frame, K, dist, T_cam_table, bowl_h)
    res = xs[1] - xs[0]
    wb = _white(img_b)
    # kill the printed marker sheet (white paper) around the marker origin
    r = 0.088
    i0, i1 = int((-r - xs[0]) / res), int((r - xs[0]) / res)
    j0, j1 = int((-r - ys[0]) / res), int((r - ys[0]) / res)
    wb[j0:j1, i0:i1] = 0
    wb = cv2.morphologyEx(wb, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(wb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c) * res * res
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        circ = cv2.contourArea(c) / (np.pi * rad * rad + 1e-9)
        d = 2 * rad * res
        score = abs(d - bowl_d) + 0.05 * (1 - circ)
        if 0.09 < d < 0.16 and circ > 0.6 and (best is None or score < best[0]):
            best = (score, cx, cy, d, circ)
    if best:
        out["bowl_xy"] = np.array([xs[0] + best[1] * res, ys[0] + best[2] * res])
        out["bowl_d_est"] = best[3]
    # ---- cube: top face on the z = cube plane ---------------------------------------------
    img_c, xs, ys = rectify(frame, K, dist, T_cam_table, cube)
    wc = _white(img_c)
    if frame_last is not None:
        # what was white at the start but is not any more = the cube at its start position
        img_l, _, _ = rectify(frame_last, K, dist, T_last, cube)
        diff = cv2.absdiff(cv2.GaussianBlur(cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY), (5, 5), 0),
                           cv2.GaussianBlur(cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), (5, 5), 0))
        gone = (diff > 35).astype(np.uint8) * 255
        gone = cv2.dilate(gone, np.ones((5, 5), np.uint8))
        wc = cv2.bitwise_and(wc, gone)
    wc[j0:j1, i0:i1] = 0
    wc[:, :int((-0.40 - xs[0]) / res)] = 0; wc[:, int((0.25 - xs[0]) / res):] = 0   # table region only
    wc[:int((-0.62 - ys[0]) / res)] = 0; wc[int((0.30 - ys[0]) / res):] = 0
    if "bowl_xy" in out:            # blank out the bowl
        bi = int((out["bowl_xy"][0] - xs[0]) / res); bj = int((out["bowl_xy"][1] - ys[0]) / res)
        cv2.circle(wc, (bi, bj), int((bowl_d / 2 + 0.008) / res), 0, -1)
    wc = cv2.morphologyEx(wc, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(wc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bestc = None
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        w, h = w * res, h * res
        fill = cv2.contourArea(c) * res * res / (w * h + 1e-9)
        score = abs(w - cube) + abs(h - cube) + 0.05 * (1 - fill)
        if 0.035 < w < 0.085 and 0.035 < h < 0.085 and (bestc is None or score < bestc[0]):
            bestc = (score, cx, cy, ang, (w + h) / 2)
    if bestc:
        out["cube_xy"] = np.array([xs[0] + bestc[1] * res, ys[0] + bestc[2] * res])
        out["cube_yaw_raw"] = float(np.deg2rad(bestc[3]))
        out["cube_side_est"] = bestc[4]
    out["_dbg"] = (img_b, img_c, wb, wc, (xs[0], ys[0], res))
    return out
