#!/usr/bin/env bash
# Full pipeline: phone videos in data/raw/ + board photos in data/calib_photos/  ->  outputs/
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}      # headless rendering (use osmesa if egl is unavailable)
python scripts/fetch_assets.py
python scripts/perceive.py                            # 1. calibration, scene layouts, hand demos, QA
python scripts/perception_figs.py                     #    bird's-eye figure
python scripts/retarget_eval.py --sweep --render 1    # 2. replay on Panda + G1, 3 methods + sweep
python scripts/train_policy.py                        # 3. retrieval memories + MLP policies
python scripts/eval_policy.py --robots panda          # 4. closed-loop eval (20 layouts) ...
python scripts/eval_policy.py --robots g1             #    ... on both bodies
python scripts/figures.py                             # 5. plots + tables
python scripts/side_by_side.py demo_01                # headline video
