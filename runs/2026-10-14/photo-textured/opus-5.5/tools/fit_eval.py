"""Development helper: project the mesh landmarks of a saved scene through Cam_Beauty and
compare them with the hand-picked photo landmarks.

    Blender --background --factory-startup some.blend --python tools/fit_eval.py -- out.png
Red = photo landmark, blue = projected mesh landmark (drawn on the reference photo).
"""
import bpy, sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import photo_fit as pf

out = sys.argv[sys.argv.index("--") + 1]
body = bpy.data.objects["Body_Skin"]
me = body.data
n = len(me.vertices)


def attr(name):
    a = np.empty(n * 3, np.float32)
    me.attributes[name].data.foreach_get("vector", a)
    return a.reshape(-1, 3).astype(float)


co = np.empty(n * 3)
me.vertices.foreach_get("co", co)
co = co.reshape(-1, 3)
orig = attr("orig_co")
head_m = np.array(body.get("head_matrix", np.eye(4).ravel().tolist())).reshape(4, 4) if "head_matrix" in body else None
if head_m is None:
    # same head transform the build uses
    import importlib.util
    spec = importlib.util.spec_from_file_location("b", os.path.join(ROOT, "build.py"))
    b = importlib.util.module_from_spec(spec)
    os.chdir(ROOT)
    spec.loader.exec_module(b)
    head_m = np.array(b.pose_definition()["head"])
Hi = np.linalg.inv(head_m)
loc = co @ Hi[:3, :3].T + Hi[:3, 3]
me.calc_loop_triangles()
tris = np.empty(len(me.loop_triangles) * 3, np.int32)
me.loop_triangles.foreach_get("vertices", tris)
tris = tris.reshape(-1, 3)
ecs = {}
r_eye = None
for s in ("L", "R"):
    eb = bpy.data.objects["Eyeball." + s]
    ecs[s] = (Hi @ np.r_[np.array(eb.location), 1])[:3]
    r_eye = max(v.co.length for v in eb.data.vertices)
lm, _ = pf.mesh_landmarks(orig, loc, tris, ecs, r_eye)
for k_, v_ in lm.items(): print("LM", k_, None if v_ is None else np.round(v_, 4))
cam = bpy.data.objects["Cam_Beauty"]
R, t, k = pf.cam_params(cam, pf.PHOTO_SIZE)
names = [k_ for k_ in pf.PHOTO_LM if lm.get(k_) is not None]
Pw = np.array([lm[k_] for k_ in names]) @ head_m[:3, :3].T + head_m[:3, 3]
uv, z = pf.project(Pw, R, t, k, pf.PHOTO_SIZE)
err = []
for nm, p in zip(names, uv):
    q = np.array(pf.PHOTO_LM[nm])
    err.append(np.linalg.norm(p - q))
    print("%-12s photo %4d %4d  mesh %7.1f %7.1f  err %5.1f" % (nm, q[0], q[1], p[0], p[1], err[-1]))
print("mean err %.1f px" % np.mean(err))
img = bpy.data.images.load(os.path.join(ROOT, "input", "reference.png"))
W, H = img.size
px = np.array(img.pixels[:], np.float32).reshape(H, W, 4)[::-1].copy()


def dot(p, c, r=3):
    x, y = int(round(p[0])), int(round(p[1]))
    px[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1, :3] = c


for nm, p in zip(names, uv):
    dot(pf.PHOTO_LM[nm], (1, 0, 0))
    dot(p, (0, 0.3, 1))
x0, y0, x1, y1 = 380, 180, 880, 680
crop = px[y0:y1, x0:x1]
o = bpy.data.images.new("ov", x1 - x0, y1 - y0)
o.pixels[:] = crop[::-1].ravel()
o.filepath_raw = out
o.file_format = 'PNG'
o.save()

# ---- try a rigid camera refit for several focal lengths
tgt = np.array([pf.PHOTO_LM[k_] for k_ in names], float)
wts = np.array([pf.PHOTO_W[k_] for k_ in names])
face_c = Pw.mean(0)
for lens in (24, 28, 32, 35, 40, 50):
    kk = lens / 36.0 * max(pf.PHOTO_SIZE)
    R2, t2, e = pf.fit_camera(Pw, tgt, wts, R, t, kk, pf.PHOTO_SIZE)
    fwd = -R2[2]
    print("lens %d  mean %.1f max %.1f  cam %s dist %.3f  fwd %s" % (lens, e.mean(), e.max(), np.round(t2, 3), np.linalg.norm(t2 - face_c), np.round(fwd, 3)))
    if lens == 35:
        print("  per-point", dict(zip(names, np.round(e, 1))))
