"""Where third-party robot models live.

Robot models come from MuJoCo Menagerie (Apache-2.0 / BSD-3). They are not
vendored in this repo; run `python scripts/fetch_assets.py` once, or point
EGO2SIM_MENAGERIE at an existing checkout.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MENAGERIE = Path(os.environ.get("EGO2SIM_MENAGERIE", REPO / "third_party" / "mujoco_menagerie"))
ASSETS = REPO / "assets"
DATA = REPO / "data"
OUT = REPO / "outputs"


def menagerie(rel: str) -> str:
    p = MENAGERIE / rel
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python scripts/fetch_assets.py` or set EGO2SIM_MENAGERIE.")
    return str(p)
