"""Phone video  ->  metric 3D hand trajectory in the table frame.

Pipeline per frame
  1. ArUco marker (id 0)            -> camera pose w.r.t. the table (solvePnP, IPPE_SQUARE)
  2. MediaPipe Hands                -> 21 image keypoints + 21 *metric* hand-centric 3D points
  3. PnP(hand-centric 3D, image 2D) -> where the hand is in the camera frame (absolute depth
                                       from a single RGB camera, using the hand as its own ruler)
  4. camera -> table transform      -> landmarks in the table frame (x fwd, y left, z up, metres)

MediaPipe's metric hand model has a canonical size, so the depth from step 3 is
off by the ratio between my hand and the canonical one. That single scale is
recovered for free from the first second of every clip, when the hand rests flat
on the table: the table plane is a known metric calibration target.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# OpenCV marker frame (x right, y "up" on the page, z out of the table)
# -> table frame (x away from the demonstrator, y to their left, z up)
A_MARKER_TO_TABLE = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)

THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 4, 8, 12
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
PALM = [0, 5, 9, 13, 17]


# ----------------------------------------------------------------------------
# Camera calibration from the ChArUco clip
# ----------------------------------------------------------------------------
def calibrate(video: str, square_m: float = 0.030, marker_m: float = 0.022,
              every: int = 5, max_frames: int = 80) -> dict:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 7), square_m, marker_m, d)
    det = cv2.aruco.CharucoDetector(board)
    cap = cv2.VideoCapture(video)
    obj_pts, img_pts, size, i = [], [], None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % every:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        cc, ci, _, _ = det.detectBoard(g)
        if ci is None or len(ci) < 8:
            continue
        op, ip = board.matchImagePoints(cc, ci)
        obj_pts.append(op); img_pts.append(ip)
    cap.release()
    if len(obj_pts) > max_frames:
        idx = np.linspace(0, len(obj_pts) - 1, max_frames).astype(int)
        obj_pts = [obj_pts[k] for k in idx]; img_pts = [img_pts[k] for k in idx]
    if len(obj_pts) < 8:
        raise RuntimeError(f"only {len(obj_pts)} usable calibration views in {video}")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "size": list(size),
            "rms_px": float(rms), "n_views": len(obj_pts)}


def calibrate_images(images, square_m: float = 0.030, marker_m: float = 0.022) -> dict:
    """ChArUco calibration from still photos (the board seen from several angles)."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 7), square_m, marker_m, d)
    det = cv2.aruco.CharucoDetector(board)
    obj_pts, img_pts, size = [], [], None
    for f in images:
        g = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        cc, ci, _, _ = det.detectBoard(g)
        if ci is None or len(ci) < 8:
            continue
        op, ip = board.matchImagePoints(cc, ci)
        obj_pts.append(op); img_pts.append(ip)
    if len(obj_pts) < 3:
        raise RuntimeError(f"only {len(obj_pts)} usable calibration photos")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None,
                                             flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "size": list(size),
            "rms_px": float(rms), "n_views": len(obj_pts)}


def rescale_intrinsics(calib: dict, w: int, h: int) -> dict:
    """Photo (4:3) -> video (16:9) of the same sensor: same field of view along the long
    side, shorter side cropped. Scale by long-side ratio; principal point centred."""
    K = np.array(calib["K"]); (cw, ch) = calib["size"]
    s = max(w, h) / max(cw, ch)
    K2 = np.array([[K[0, 0] * s, 0, w / 2], [0, K[1, 1] * s, h / 2], [0, 0, 1]])
    return {**calib, "K": K2.tolist(), "size": [w, h], "from_photo_size": [cw, ch]}


def intrinsics_guess(w: int, h: int, hfov_deg: float = 69.0) -> dict:
    """Fallback if no calibration clip: typical phone main camera ~ 69 deg HFOV."""
    f = (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    return {"K": K.tolist(), "dist": [0, 0, 0, 0, 0], "size": [w, h], "rms_px": None, "n_views": 0}


# ----------------------------------------------------------------------------
# Per-frame tracking
# ----------------------------------------------------------------------------
@dataclass
class Track:
    fps: float
    t: np.ndarray            # (T,)
    lm_cam: np.ndarray       # (T, 21, 3) landmarks in camera frame [m] (NaN if missing)
    lm_img: np.ndarray       # (T, 21, 2) pixels
    hand_ok: np.ndarray      # (T,) bool
    T_cam_table: np.ndarray  # (T, 4, 4) table->camera transform (interpolated over gaps)
    marker_ok: np.ndarray    # (T,) bool
    size: tuple

    def save(self, path):
        np.savez_compressed(path, fps=self.fps, t=self.t, lm_cam=self.lm_cam, lm_img=self.lm_img,
                            hand_ok=self.hand_ok, T_cam_table=self.T_cam_table,
                            marker_ok=self.marker_ok, size=np.array(self.size))

    @staticmethod
    def load(path):
        z = np.load(path)
        return Track(float(z["fps"]), z["t"], z["lm_cam"], z["lm_img"], z["hand_ok"],
                     z["T_cam_table"], z["marker_ok"], tuple(z["size"]))


def _marker_pose(gray, K, dist, marker_m, detector):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or 0 not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(0)].reshape(4, 2)
    h = marker_m / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float32), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R_cm, _ = cv2.Rodrigues(rvec)               # marker -> camera
    T = np.eye(4)
    T[:3, :3] = R_cm @ A_MARKER_TO_TABLE.T       # table -> camera
    T[:3, 3] = tvec.ravel()
    return T


def track_video(video: str, calib: dict, marker_m: float = 0.12, hand="right",
                max_frames: int | None = None) -> Track:
    import mediapipe as mp

    K = np.array(calib["K"], float)
    dist = np.array(calib["dist"], float)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                     model_complexity=1, min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lm_cam, lm_img, hand_ok, Ts, mk_ok = [], [], [], [], []
    prev_rt = None
    size = None
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        n += 1
        h, w = fr.shape[:2]
        size = (w, h)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        T = _marker_pose(gray, K, dist, marker_m, det)
        mk_ok.append(T is not None)
        Ts.append(T if T is not None else np.full((4, 4), np.nan))

        res = hands.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        L3 = np.full((21, 3), np.nan); L2 = np.full((21, 2), np.nan); okh = False
        if res.multi_hand_landmarks:
            # rear camera => MediaPipe's handedness label is mirrored; pick the larger,
            # most confident hand (only the right hand should be in frame anyway)
            k = int(np.argmax([hw.classification[0].score for hw in res.multi_handedness]))
            img = np.array([[p.x * w, p.y * h] for p in res.multi_hand_landmarks[k].landmark])
            wld = np.array([[p.x, p.y, p.z] for p in res.multi_hand_world_landmarks[k].landmark])
            if prev_rt is not None:
                ok_p, rv, tv = cv2.solvePnP(wld, img, K, dist, rvec=prev_rt[0].copy(),
                                            tvec=prev_rt[1].copy(), useExtrinsicGuess=True,
                                            flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                ok_p, rv, tv = cv2.solvePnP(wld, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
            if ok_p and tv[2, 0] > 0.05:
                R, _ = cv2.Rodrigues(rv)
                L3 = (R @ wld.T).T + tv.ravel()
                L2 = img
                okh = True
                prev_rt = (rv, tv)
        lm_cam.append(L3); lm_img.append(L2); hand_ok.append(okh)
    cap.release()
    hands.close()
    Ts = _fill_poses(np.array(Ts), np.array(mk_ok))
    t = np.arange(len(hand_ok)) / fps
    return Track(fps, t, np.array(lm_cam), np.array(lm_img), np.array(hand_ok), Ts,
                 np.array(mk_ok), size)


def _fill_poses(Ts, ok):
    """The hand regularly occludes the marker. With a (mostly) static phone the camera
    pose barely changes, so gaps are filled by nearest-valid / linear interpolation of
    translation and nearest rotation, and the whole sequence is lightly smoothed."""
    if ok.sum() == 0:
        raise RuntimeError("table marker never detected - is it in frame and printed at 100%?")
    idx = np.arange(len(Ts))
    good = idx[ok]
    out = Ts.copy()
    for k in range(3):
        out[:, k, 3] = np.interp(idx, good, Ts[good, k, 3])
    nearest = good[np.abs(idx[:, None] - good[None]).argmin(1)]
    out[:, :3, :3] = Ts[nearest, :3, :3]
    out[:, 3] = [0, 0, 0, 1]
    return out


# ----------------------------------------------------------------------------
# Camera frame -> table frame, with the "table is a ruler" scale fix
# ----------------------------------------------------------------------------
def to_table(track: Track, scale: float = 1.0) -> np.ndarray:
    """Landmarks in the table frame. `scale` multiplies camera-frame depth (hand size fix)."""
    L = track.lm_cam * scale
    R = track.T_cam_table[:, :3, :3]
    t = track.T_cam_table[:, :3, 3]
    # p_table = R^T (p_cam - t)
    return np.einsum("tji,tkj->tki", R, L - t[:, None, :])


def estimate_hand_scale(track: Track, rest_s: float = 0.8, palm_z: float = 0.012) -> tuple:
    """Find s such that the palm, resting flat on the table during the first `rest_s`
    seconds, lies `palm_z` above the table plane. Returns (scale, residual_m)."""
    n = max(3, int(rest_s * track.fps))
    sel = np.where(track.hand_ok[:n])[0]
    if len(sel) < 3:
        return 1.0, float("nan")
    sub = Track(track.fps, track.t[sel], track.lm_cam[sel], track.lm_img[sel],
                track.hand_ok[sel], track.T_cam_table[sel], track.marker_ok[sel], track.size)
    ss = np.linspace(0.6, 1.6, 401)
    errs = [abs(np.nanmedian(to_table(sub, s)[:, PALM, 2]) - palm_z) for s in ss]
    k = int(np.argmin(errs))
    return float(ss[k]), float(errs[k])


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))
