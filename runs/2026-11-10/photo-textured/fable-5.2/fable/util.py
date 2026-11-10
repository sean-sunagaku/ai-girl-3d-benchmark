"""Blender helpers shared by the build modules."""
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


def link(ob, col):
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    col.objects.link(ob)
    return ob


def mesh_object(name, V, F, col, smooth=True, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in V], [], [list(map(int, f)) for f in F])
    if smooth:
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    if uvs is not None:
        uvl = me.uv_layers.new(name='UVMap')
        uvl.data.foreach_set('uv', np.asarray(uvs, dtype=np.float32).ravel())
    me.update()
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    return ob


def evaluated_arrays(ob):
    """Return (V, F) numpy arrays of the evaluated (modifier-applied) mesh; F is a list for mixed polygons."""
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    V = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get('co', V)
    V = V.reshape(-1, 3).astype(np.float64)
    F = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    return V, F


def subdivided(V, F, levels, uvs=None):
    """Catmull-Clark subdivide a pydata mesh with Blender's own modifier; returns V, F (quads) and per-loop UVs."""
    me = bpy.data.meshes.new('tmp_subd')
    me.from_pydata([tuple(map(float, p)) for p in V], [], [list(f) for f in F])
    if uvs is not None:
        uvl = me.uv_layers.new(name='UVMap')
        uvl.data.foreach_set('uv', np.asarray(uvs, dtype=np.float32).ravel())
    me.update()
    ob = bpy.data.objects.new('tmp_subd', me)
    bpy.context.scene.collection.objects.link(ob)
    m = ob.modifiers.new('s', 'SUBSURF'); m.levels = levels; m.render_levels = levels
    m.uv_smooth = 'NONE'
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    ev = ob.evaluated_get(dg); em = ev.to_mesh()
    n = len(em.vertices)
    Vs = np.empty(n * 3, dtype=np.float32); em.vertices.foreach_get('co', Vs); Vs = Vs.reshape(-1, 3).astype(np.float64)
    Fs = np.empty(len(em.polygons) * 4, dtype=np.int32); em.polygons.foreach_get('vertices', Fs); Fs = Fs.reshape(-1, 4)
    UVs = None
    if uvs is not None:
        UVs = np.empty(len(em.loops) * 2, dtype=np.float32); em.uv_layers[0].data.foreach_get('uv', UVs); UVs = UVs.reshape(-1, 4, 2).astype(np.float64)
    ev.to_mesh_clear()
    bpy.data.objects.remove(ob, do_unlink=True); bpy.data.meshes.remove(me)
    return Vs, Fs, UVs


def bvh_from(V, F):
    return BVHTree.FromPolygons([tuple(p) for p in V], [tuple(int(i) for i in f) for f in F])


def vertex_normals(V, F):
    F = np.asarray(F)
    n = np.zeros_like(V)
    if F.shape[1] == 4:
        tris = np.concatenate([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])
    else:
        tris = F
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    l = np.linalg.norm(n, axis=1); l[l == 0] = 1
    return n / l[:, None]


def new_image(name, arr, colorspace='sRGB', is_data=False):
    """arr: (H, W, 3|4) float array in the image's own (display) encoding, row 0 = bottom. Packed into the .blend."""
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones((h, w, 1), arr.dtype)], 2)
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=False, is_data=is_data)
    if not is_data:
        img.colorspace_settings.name = colorspace
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).ravel())
    img.update()
    img.pack()
    return img


def set_active(ob):
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def hash01(a, seed=0):
    """Cheap deterministic per-element hash in [0,1)."""
    x = np.sin(np.asarray(a, dtype=np.float64) * 12.9898 + seed * 78.233) * 43758.5453
    return x - np.floor(x)


def value_noise2(u, v, seed=0):
    """Smooth 2D value noise for numpy arrays (deterministic)."""
    u0 = np.floor(u); v0 = np.floor(v)
    fu = u - u0; fv = v - v0
    fu = fu * fu * (3 - 2 * fu); fv = fv * fv * (3 - 2 * fv)
    def h(i, j):
        x = np.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453
        return x - np.floor(x)
    return (h(u0, v0) * (1 - fu) + h(u0 + 1, v0) * fu) * (1 - fv) + (h(u0, v0 + 1) * (1 - fu) + h(u0 + 1, v0 + 1) * fu) * fv


def fbm2(u, v, octaves=4, seed=0, lac=2.0, gain=0.5):
    a = 1.0; f = 1.0; s = 0.0; n = 0.0
    for o in range(octaves):
        s += a * value_noise2(u * f, v * f, seed + o * 17)
        n += a; a *= gain; f *= lac
    return s / n


def value_noise3(p, seed=0):
    p = np.asarray(p, dtype=np.float64)
    i = np.floor(p); f = p - i
    f = f * f * (3 - 2 * f)
    def h(o):
        q = i + np.array(o)
        x = np.sin(q[..., 0] * 127.1 + q[..., 1] * 311.7 + q[..., 2] * 74.7 + seed * 19.19) * 43758.5453
        return x - np.floor(x)
    x00 = h((0, 0, 0)) * (1 - f[..., 0]) + h((1, 0, 0)) * f[..., 0]
    x10 = h((0, 1, 0)) * (1 - f[..., 0]) + h((1, 1, 0)) * f[..., 0]
    x01 = h((0, 0, 1)) * (1 - f[..., 0]) + h((1, 0, 1)) * f[..., 0]
    x11 = h((0, 1, 1)) * (1 - f[..., 0]) + h((1, 1, 1)) * f[..., 0]
    y0 = x00 * (1 - f[..., 1]) + x10 * f[..., 1]
    y1 = x01 * (1 - f[..., 1]) + x11 * f[..., 1]
    return y0 * (1 - f[..., 2]) + y1 * f[..., 2]


def fbm3(p, octaves=4, seed=0):
    a = 1.0; fr = 1.0; s = 0.0; n = 0.0
    for o in range(octaves):
        s += a * value_noise3(np.asarray(p) * fr, seed + o * 31)
        n += a; a *= 0.5; fr *= 2.0
    return s / n
