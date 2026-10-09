"""Sanity check: a hand-scripted pick-and-place on each embodiment (no human data)."""
import sys, numpy as np, mujoco, imageio
sys.path.insert(0, '.')
from ego2sim.scene import EMBODIMENTS, TaskObjects, build
from ego2sim.kinematics import IK
from ego2sim.execute import run

def minjerk(a, b, n):
    s = np.linspace(0, 1, n); s = 10*s**3 - 15*s**4 + 6*s**5
    return a[None] + (b - a)[None] * s[:, None]

name = sys.argv[1] if len(sys.argv) > 1 else 'panda'
emb = EMBODIMENTS[name]()
cube_h = np.array([0.02, -0.08]); bowl_h = np.array([0.0, 0.17]); cs = 0.05
cube = emb.map_point([*cube_h, 0])[:2]; bowl = emb.map_point([*bowl_h, 0])[:2]
obj = TaskObjects(cube_size=cs, cube_xy=tuple(cube), bowl_xy=tuple(bowl))
m, d = build(emb, obj)
ik = IK(m, emb)
p0, _ = ik.fk(emb.home_q)
gz = cs / 2 + (0.0 if name == 'panda' else 0.012)
W = [p0, np.r_[cube, gz + 0.12], np.r_[cube, gz], np.r_[cube, gz], np.r_[cube, gz + 0.15],
     np.r_[bowl, gz + 0.15], np.r_[bowl, gz + 0.10], np.r_[bowl, gz + 0.10], p0]
G = [0, 0, 0, 1, 1, 1, 1, 0, 0]
N = [40, 40, 25, 25, 45, 25, 20, 40]
dt = 0.02
P, Gr = [W[0][None]], [np.array([0.0])]
for i in range(len(N)):
    P.append(minjerk(W[i], W[i+1], N[i])[1:]); Gr.append(np.linspace(G[i], G[i+1], N[i])[1:])
P = np.concatenate(P); Gr = np.concatenate(Gr)
q = emb.home_q.copy(); Q = []; errs = []
R = emb.tcp_target_rot(0.0)
for p in P:
    q, e = ik.solve(p, R, q); Q.append(q); errs.append(e)
print('max IK err', max(errs))
r = mujoco.Renderer(m, 240, 320)
ro = run(m, d, emb, np.array(Q), Gr, dt, render=r, cam=sys.argv[2] if len(sys.argv) > 2 else 'front')
print(name, 'success', ro.success, 'cube end', ro.cube[-1].round(3), 'max cube z', ro.cube[:,2].max().round(3))
imageio.mimsave(f'/tmp/scripted_{name}.gif', ro.frames[::2], duration=1/15)
imageio.imwrite(f'/tmp/scripted_{name}_mid.png', np.concatenate([ro.frames[len(ro.frames)*k//6] for k in (1,2,3,4)],1))
