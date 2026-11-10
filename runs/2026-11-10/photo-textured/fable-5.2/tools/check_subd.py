import sys, os
sys.path.insert(0, os.getcwd())
import bpy, numpy as np
from fable.base import Body
from fable import util
bpy.ops.wm.read_factory_settings(use_empty=True)
b = Body('input')
Vs, Fs, _ = util.subdivided(b.V, b.F, 2)
n = len(b.V)
d = np.linalg.norm(Vs[:n] - b.V, axis=1)
print('SUBD n', len(Vs), 'first-n dist mean %.5f max %.5f' % (d.mean(), d.max()))
# brute-force: nearest subdivided vertex to each cage vertex
from mathutils import kdtree
kd = kdtree.KDTree(len(Vs))
for i, p in enumerate(Vs): kd.insert(p, i)
kd.balance()
nn = np.array([kd.find(p)[1] for p in b.V[:200]])
print('nearest idx of first 20 cage verts', nn[:20])
print('identity fraction (first 200):', float((nn == np.arange(200)).mean()))
