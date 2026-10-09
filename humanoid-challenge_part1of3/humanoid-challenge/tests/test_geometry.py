"""Synthetic check of the perception geometry: render the table marker into an image
from a known camera, project a known 3D hand, and recover it in the table frame."""
import sys, numpy as np, cv2
sys.path.insert(0, '.')
from ego2sim.perception import (A_MARKER_TO_TABLE, Track, _marker_pose, to_table,
                                estimate_hand_scale, PALM)

W, H = 1920, 1080
K = np.array([[1400, 0, W/2], [0, 1400, H/2], [0, 0, 1.]]); dist = np.zeros(5)
MARK = 0.12

def cam_pose(eye, target):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.]); r /= np.linalg.norm(r); u = np.cross(r, f)
    R_wc = np.stack([r, -u, f], 1)          # OpenCV camera axes in table frame
    T = np.eye(4); T[:3, :3] = R_wc.T; T[:3, 3] = -R_wc.T @ eye   # table -> camera
    return T

def render_marker(T):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = cv2.aruco.generateImageMarker(d, 0, 400, borderBits=1)
    m = cv2.copyMakeBorder(m, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    h = MARK / 2 * 500 / 400
    # marker corners in marker frame -> table frame
    cm = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    ct = (A_MARKER_TO_TABLE @ cm.T).T
    pc = (T[:3, :3] @ ct.T).T + T[:3, 3]
    px = (K @ pc.T).T; px = px[:, :2] / px[:, 2:]
    src = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
    Hm = cv2.getPerspectiveTransform(src, px.astype(np.float32))
    img = np.full((H, W), 120, np.uint8)
    warped = cv2.warpPerspective(m, Hm, (W, H), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(m, 255), Hm, (W, H))
    img[mask > 0] = warped[mask > 0]
    return img

T = cam_pose(np.array([-0.45, -0.30, 0.55]), np.array([0.05, 0.0, 0.0]))
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())
T_est = _marker_pose(render_marker(T), K, dist, MARK, det)
err_t = np.linalg.norm(T_est[:3, 3] - T[:3, 3]); err_R = np.linalg.norm(T_est[:3, :3] - T[:3, :3])
print(f'marker pose: translation err {err_t*1000:.2f} mm, rotation err {err_R:.4f}')
assert err_t < 0.005 and err_R < 0.02

# hand: 21 pts, palm flat 12 mm above table at (-0.05,-0.1); MediaPipe-like canonical
# model is 1.15x bigger than the "true" hand -> PnP depth is 1.15x too far
rng = np.random.default_rng(0)
true_hand = np.c_[rng.uniform(-0.05, 0.08, 21), rng.uniform(-0.04, 0.04, 21), np.zeros(21)]
true_hand[PALM, 2] = 0.0
true_table = true_hand + [-0.05, -0.10, 0.012]
true_cam = (T[:3, :3] @ true_table.T).T + T[:3, 3]
# simulate monocular scale ambiguity
est_cam = true_cam * 1.15
tr = Track(30., np.zeros(30), np.repeat(est_cam[None], 30, 0), np.zeros((30, 21, 2)),
           np.ones(30, bool), np.repeat(T_est[None], 30, 0), np.ones(30, bool), (W, H))
s, res = estimate_hand_scale(tr)
rec = to_table(tr, s)[0]
print(f'recovered hand scale {s:.3f} (true {1/1.15:.3f}), mean 3D err {np.linalg.norm(rec-true_table,axis=1).mean()*1000:.1f} mm')
assert abs(s - 1/1.15) < 0.01
print('OK')

# the sim "phone" camera must see the marker centre where the real camera did
import mujoco
from ego2sim.scene import phone_camera, TaskObjects, build, Panda
import os
if os.environ.get("EGO2SIM_MENAGERIE"):
    pos, q, fovy = phone_camera(T, K, (W, H))
    m, d = build(Panda(), TaskObjects(), phone_cam=(pos, q, fovy))
    cid = m.camera("phone").id
    cam_pos = d.cam_xpos[cid]; cam_R = d.cam_xmat[cid].reshape(3, 3)
    # marker origin in mujoco camera coords: looking along -z
    pc = cam_R.T @ (np.zeros(3) - cam_pos)
    u = W / 2 + K[0, 0] * pc[0] / -pc[2]; v = H / 2 - K[1, 1] * pc[1] / -pc[2]
    px = K @ (T[:3, 3]); px = px[:2] / px[2]
    print(f'phone camera reprojection of marker origin: {np.hypot(u-px[0], v-px[1]):.2f} px')
    assert np.hypot(u - px[0], v - px[1]) < 1.0
    print('OK phone camera')
