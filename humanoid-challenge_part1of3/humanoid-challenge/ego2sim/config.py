"""Physical measurements of my recording setup (data/measurements.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .paths import DATA

# per-embodiment strength of the TorqueRateLayer. The penalty is weighted by each
# robot's own mass matrix, and the G1 arm is ~10x lighter than the Panda's, so the
# same lambda would barely touch it. Values picked by the sweeps in results/.
LAMBDA_OFFLINE = {"panda": 1e-5, "g1": 1e-3}   # retargeting whole demos (90 Hz)
LAMBDA_ONLINE = {"panda": 1e-8, "g1": 1e-5}    # receding-horizon policy chunks


@dataclass
class Measurements:
    marker_mm: float = 120.0
    board_square_mm: float = 30.0
    board_marker_mm: float = 22.0
    cube_mm: float = 50.0
    bowl_diam_mm: float = 160.0
    bowl_height_mm: float = 60.0

    @property
    def cube(self):
        return self.cube_mm / 1000

    @staticmethod
    def load(path=None):
        p = DATA / "measurements.json" if path is None else path
        if not p.exists():
            print(f"[warn] {p} missing - using nominal printed sizes")
            return Measurements()
        return Measurements(**json.loads(p.read_text()))
