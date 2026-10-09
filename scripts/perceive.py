"""Step 1 - phone videos -> scene layouts + human demonstrations.

  data/calib_photos/*.jpg       ChArUco photos -> data/calib.json   (intrinsics, rescaled to the video)
  data/raw/demo_XX.mp4          clips
     -> data/tracks/*.npz       per-frame MediaPipe hand (+ marker pose)
     -> data/layouts.json       cube / bowl pose of EVERY clip, from its hand-free first frame
     -> data/demos/*.npz        demonstrations (clips where the hand track covers grasp AND release)
     -> outputs/perception/*.mp4, qa.md, layouts.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.config import Measurements  # noqa: E402
from ego2sim.demo import segment_grounded  # noqa: E402
from ego2sim.paths import DATA, OUT  # noqa: E402
from ego2sim.perception import (INDEX_TIP, THUMB_TIP, Track, calibrate, calibrate_images,  # noqa: E402
                                intrinsics_guess, rescale_intrinsics, save_json, track_video)
from ego2sim.scene_parse import find_objects  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]


def overlay(video, track: Track, demo, calib, out_mp4, obj, scale_px=540):
    K = np.array(calib["K"]); dist = np.array(calib["dist"])
    cap = cv2.VideoCapture(str(video))
    frames, i, trail = [], 0, []
    a = demo.qa["start_frame"] if demo is not None else None
    while True:
        ok, fr = cap.read()
        if not ok or i >= len(track.t):
            break
        T = track.T_cam_table[i]
        rv, _ = cv2.Rodrigues(T[:3, :3])
        cv2.drawFrameAxes(fr, K, dist, rv, T[:3, 3], 0.08, 4)
        for key, col in (("cube_xy", (0, 255, 0)), ("bowl_xy", (0, 140, 255))):    # what the scene parser found
            if key in obj:
                px, _ = cv2.projectPoints(np.array([[*obj[key], 0.03]]), rv, T[:3, 3], K, dist)
                cv2.circle(fr, tuple(px[0, 0].astype(int)), 28, col, 5, cv2.LINE_AA)
        state = "hand not tracked"
        if track.hand_ok[i]:
            L = track.lm_img[i].astype(int)
            grip = demo is not None and a <= i < a + len(demo.grip) and demo.grip[i - a] > 0.5
            col = (60, 60, 230) if grip else (230, 200, 60)
            for u, v in HAND_EDGES:
                cv2.line(fr, tuple(L[u]), tuple(L[v]), col, 3, cv2.LINE_AA)
            trail.append(tuple(((L[THUMB_TIP] + L[INDEX_TIP]) / 2).astype(int)))
            state = "carrying" if grip else "hand tracked"
        for p, q in zip(trail[:-1], trail[1:]):
            cv2.line(fr, p, q, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(fr, state, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(fr, state, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (scale_px, int(h * scale_px / w) // 2 * 2))
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    imageio.mimsave(out_mp4, frames, fps=track.fps, macro_block_size=2)


def read_frame(video, k):
    cap = cv2.VideoCapture(str(video))
    if k < 0:
        k = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + k
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, fr = cap.read(); cap.release()
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DATA / "raw"))
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()
    meas = Measurements.load()
    raw = Path(args.raw)
    for d in ("tracks", "demos"):
        (DATA / d).mkdir(parents=True, exist_ok=True)
    (OUT / "perception").mkdir(parents=True, exist_ok=True)

    calib_path = DATA / "calib.json"
    if not calib_path.exists():
        photos = sorted((DATA / "calib_photos").glob("*.jpg"))
        cv = sorted(raw.glob("calib*"))
        first = sorted(raw.glob("demo_*"))[0]
        cap = cv2.VideoCapture(str(first)); w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
        if photos:
            pc = calibrate_images(photos, meas.board_square_mm / 1000, meas.board_marker_mm / 1000)
            print(f"calibration from {len(photos)} board photos: RMS {pc['rms_px']:.2f} px over {pc['n_views']} views")
            calib = rescale_intrinsics(pc, w, h)
        elif cv:
            calib = calibrate(str(cv[0]), meas.board_square_mm / 1000, meas.board_marker_mm / 1000)
        else:
            calib = intrinsics_guess(w, h)
            print("[warn] no calibration data - generic phone intrinsics")
        save_json(calib, calib_path)
    calib = json.loads(calib_path.read_text())
    K, dist = calib["K"], calib["dist"]

    layouts, qa_rows, objs = {}, [], {}
    for vid in sorted(raw.glob("demo_*")):
        name = vid.stem
        tp = DATA / "tracks" / f"{name}.npz"
        if tp.exists():
            tr = Track.load(tp)
        else:
            print(f"tracking {vid.name} ...")
            tr = track_video(str(vid), calib, meas.marker_mm / 1000)
            tr.save(tp)
        f0, fl = read_frame(vid, 3), read_frame(vid, -4)
        obj = find_objects(f0, K, dist, tr.T_cam_table[3], cube=meas.cube, bowl_d=meas.bowl_diam_mm / 1000,
                           bowl_h=meas.bowl_height_mm / 1000, frame_last=fl, T_last=tr.T_cam_table[-4])
        obj.pop("_dbg", None)
        objs[name] = obj
    bowls = np.array([o["bowl_xy"] for o in objs.values() if "bowl_xy" in o])
    bowl_med = np.median(bowls, 0)          # the bowl never moved: use it where the scene parser lost it
    for name, o in objs.items():
        o.setdefault("bowl_xy", bowl_med)
        o["bowl_from_fallback"] = "bowl_d_est" not in o

    for vid in sorted(raw.glob("demo_*")):
        name = vid.stem
        tr = Track.load(DATA / "tracks" / f"{name}.npz")
        o = objs[name]
        row = {"name": name}
        demo = None
        try:
            demo = segment_grounded(name, tr, o, meas.cube, meas.bowl_height_mm / 1000)
            demo.to_npz(DATA / "demos" / f"{name}.npz")
            row.update(demo.qa); row["status"] = "demo"
        except RuntimeError as e:
            row["status"] = "layout only"; row["why"] = str(e).split(": ", 1)[-1]
            row["hand_frames_total"] = round(float(tr.hand_ok.mean()), 3)
        row["cube_side_seen_mm"] = round(float(o.get("cube_side_est", np.nan)) * 1000, 1)
        row["bowl_diam_seen_mm"] = round(float(o.get("bowl_d_est", np.nan)) * 1000, 1)
        if "cube_xy" in o:
            layouts[name] = {"cube_xy": np.asarray(o["cube_xy"]).tolist(),
                             "cube_yaw": float(o.get("cube_yaw_raw", 0.0)),
                             "bowl_xy": np.asarray(o["bowl_xy"]).tolist(),
                             "has_hand_demo": demo is not None}
        qa_rows.append(row)
        print(row)
        if not args.no_overlay:
            overlay(vid, tr, demo, calib, OUT / "perception" / f"{name}.mp4", o)
    (DATA / "layouts.json").write_text(json.dumps(layouts, indent=1))

    keys = ["name", "status", "hand_frames_total", "tracked_s", "pinch_vs_cube_mm", "pinch_vs_bowl_mm",
            "cube_side_seen_mm", "bowl_diam_seen_mm", "hold_aperture_mm", "carry_s", "why"]
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in qa_rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    (OUT / "perception" / "qa.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
