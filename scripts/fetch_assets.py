"""Fetch the two robot models from MuJoCo Menagerie (sparse checkout, ~40 MB)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ego2sim.paths import MENAGERIE  # noqa: E402

URL = "https://github.com/google-deepmind/mujoco_menagerie"
DIRS = ["franka_emika_panda", "unitree_g1"]

if (MENAGERIE / "unitree_g1" / "g1_with_hands.xml").exists():
    print("menagerie already present:", MENAGERIE)
    sys.exit(0)
MENAGERIE.parent.mkdir(parents=True, exist_ok=True)
run = lambda *a: subprocess.run(a, check=True)  # noqa: E731
run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", URL, str(MENAGERIE))
run("git", "-C", str(MENAGERIE), "sparse-checkout", "set", *DIRS)
print("ok:", MENAGERIE)
