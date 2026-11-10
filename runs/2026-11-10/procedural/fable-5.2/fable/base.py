"""Assemble the rest-pose body from the supplied MakeHuman files, plus small numpy/bmesh utilities."""
import json
import os
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

Z_CUT = 1.0968  # same crop the organisers used for upper-body.obj (source y >= 2.8)


def load_obj(path):
    V, VT, F, FT, G = [], [], [], [], []
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('vt '):
                VT.append([float(x) for x in line.split()[1:3]])
            elif line.startswith('g '):
                cur = line.split()[1]
            elif line.startswith('f '):
                idx, tidx = [], []
                for tok in line.split()[1:]:
                    p = tok.split('/')
                    idx.append(int(p[0]) - 1)
                    tidx.append(int(p[1]) - 1 if len(p) > 1 and p[1] else -1)
                F.append(idx); FT.append(tidx); G.append(cur)
    return np.array(V, dtype=np.float64), np.array(VT, dtype=np.float64), F, FT, G


def prepared_frame(B):
    """Same transform the organisers documented in environment.json (meters, Z up, face toward -Y)."""
    return np.stack([B[:, 0] * 0.1, -B[:, 2] * 0.1, (B[:, 1] + 8.168) * 0.1], 1)


class Body:
    """Rest-pose body data. V is edited in place by the sculpt stage."""

    def __init__(self, input_dir):
        B, BT, BF, BFT, BG = load_obj(os.path.join(input_dir, 'base.obj'))
        T = prepared_frame(B)
        body = [(f, ft) for f, ft, g in zip(BF, BFT, BG) if g == 'body']
        upper = [(f, ft) for f, ft in body if min(T[i, 2] for i in f) >= Z_CUT - 1e-9]
        rest = [(f, ft) for f, ft in body if not (min(T[i, 2] for i in f) >= Z_CUT - 1e-9)]
        # forearms + hands: the only body faces below the crop that sit laterally outside the hips
        arm = [(f, ft) for f, ft in rest if min(abs(T[i, 0]) for i in f) > 0.26 and min(T[i, 2] for i in f) > 0.85]
        faces = upper + arm
        used = sorted({i for f, _ in faces for i in f})
        remap = {v: k for k, v in enumerate(used)}
        self.base_index = np.array(used)
        self.V = T[used].copy()
        self.V0 = self.V.copy()
        self.F = [[remap[i] for i in f] for f, _ in faces]
        self.UV = [[tuple(BT[t]) for t in ft] for _, ft in faces]
        self.n_upper_faces = len(upper)
        self.n_arm_faces = len(arm)
        # cross-check against the supplied crop
        U, _, UF, _, _ = load_obj(os.path.join(input_dir, 'upper-body.obj'))
        assert len(UF) == len(upper), (len(UF), len(upper))
        key = lambda p: (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5))
        have = {key(p) for p in self.V}
        assert all(key(p) in have for p in U), 'upper-body.obj vertices must all be present'
        self.helpers = {}
        for g in ('helper-l-eye', 'helper-r-eye', 'helper-l-eyelashes-1', 'helper-l-eyelashes-2',
                  'helper-r-eyelashes-1', 'helper-r-eyelashes-2', 'helper-hair'):
            ids = sorted({i for f, gg in zip(BF, BG) if gg == g for i in f})
            self.helpers[g] = T[ids]
        with open(os.path.join(input_dir, 'environment.json')) as fh:
            env = json.load(fh)
        self.landmarks = {k: np.array(v) for k, v in env['landmarks'].items()}
        self._topology()

    def _topology(self):
        n = len(self.V)
        edges = set()
        for f in self.F:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                edges.add((min(a, b), max(a, b)))
        self.E = np.array(sorted(edges))
        self.adj = [[] for _ in range(n)]
        for a, b in self.E:
            self.adj[a].append(b); self.adj[b].append(a)
        # enclosed = inner cavities (eye sockets, mouth bag, nostrils): a short ray along the normal hits the mesh
        bm = self.to_bmesh(self.V0)
        tree = BVHTree.FromBMesh(bm)
        enc = np.zeros(n, bool)
        N = np.zeros((n, 3))
        for i, v in enumerate(bm.verts):
            N[i] = v.normal
            if tree.ray_cast(v.co + v.normal * 1e-4, v.normal, 0.06)[0] is not None:
                enc[i] = True
        bm.free()
        self.enclosed = enc
        self.N0 = N

    def to_bmesh(self, V=None):
        V = self.V if V is None else V
        bm = bmesh.new()
        bv = [bm.verts.new(p) for p in V]
        for f in self.F:
            bm.faces.new([bv[i] for i in f])
        bm.normal_update()
        bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
        return bm

    def normals(self, V=None):
        bm = self.to_bmesh(V)
        N = np.array([v.normal[:] for v in bm.verts])
        bm.free()
        return N

    def make_object(self, name, V=None, collection=None):
        V = self.V if V is None else V
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for p in V], [], self.F)
        uvl = me.uv_layers.new(name='UVMap')
        flat = np.array([uv for f in self.UV for uv in f], dtype=np.float32).ravel()
        uvl.data.foreach_set('uv', flat)
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
        me.update()
        ob = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(ob)
        return ob


# ----------------------------------------------------------------------------- numpy sculpt helpers

def bump(d, r):
    t = np.clip(d / r, 0.0, 1.0)
    return (1.0 - t * t) ** 2


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mirror_x(v):
    v = np.array(v, dtype=np.float64).copy()
    v[..., 0] *= -1
    return v
