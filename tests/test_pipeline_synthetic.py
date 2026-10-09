"""Smoke test: synthetic hand trajectory -> both robots, all three retargeting methods."""
import sys
sys.path.insert(0, '.')
from ego2sim.config import LAMBDA_OFFLINE
from ego2sim.deploy import set_layout
from ego2sim.execute import run
from ego2sim.retarget import retarget, synthetic_demo
from ego2sim.scene import EMBODIMENTS, TaskObjects, build

demo = synthetic_demo(seed=3)
for name in ["panda", "g1"]:
    emb = EMBODIMENTS[name]()
    m, d = build(emb, TaskObjects())
    set_layout(m, d, emb, demo.cube_xy, demo.bowl_xy, 0.05, demo.cube_yaw)
    for meth in ["raw", "lowpass", "torque"]:
        tr = retarget(demo, m, emb, 0.05, method=meth, lam=LAMBDA_OFFLINE[name])
        ro = run(m, d, emb, tr.q, tr.grip, tr.dt)
        print(name, meth, "success", ro.success)
        assert ro.success, (name, meth)
print("OK")
