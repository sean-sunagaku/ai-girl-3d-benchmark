"""Photo -> UV skin texture for Body_Skin (imported by build.py).

Pipeline (all numpy, inside Blender):
  1. load input/reference.png, build a skin-visibility mask in photo space (bright, low-saturation
     skin; lips added from the lip landmarks; eye openings, hair, sweater and background removed);
  2. "clean" the photo: masked pixels are filled by pull-push interpolation from neighbouring skin,
     then the low-frequency photo lighting is partly divided out (de-lighting);
  3. Cycles bakes world position / normal / coverage of the posed, subdivided Body_Skin into
     its SkinUV layout; every texel is projected through the fitted beauty camera (the same
     camera the face was fitted to), checked for self-occlusion and facing, and sampled from the
     cleaned photo through a small landmark morph field;
  4. result = RGBA texture: RGB photo colour, A = confidence (facing x visibility x mask).
     Texels the photo never saw keep A=0 and the shader shows the procedural skin layer there.
"""
import math
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def load_rgb(path):
    img = bpy.data.images.load(path)
    W, H = img.size
    px = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(H, W, 4)[::-1, :, :3].copy()          # top-left origin, sRGB-encoded 0..1


def blur(a, sigma):
    """Separable gaussian blur (edges clamped) for 2-D or 3-D arrays."""
    r = int(3 * sigma)
    if r < 1:
        return a.copy()
    x = np.arange(-r, r + 1)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode='edge')
        acc = np.zeros_like(out)
        for i, kv in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            acc += kv * p[tuple(sl)]
        out = acc
    return out


def pull_push(img, w, levels=9):
    """Fill pixels with weight 0 from weighted neighbours (multiresolution pull-push)."""
    pyr = [(img * w[..., None], w.copy())]
    for _ in range(levels):
        ci, cw = pyr[-1]
        H, W = cw.shape
        H2, W2 = (H + 1) // 2, (W + 1) // 2
        ci = np.pad(ci, ((0, H2 * 2 - H), (0, W2 * 2 - W), (0, 0)), mode='edge')
        cw = np.pad(cw, ((0, H2 * 2 - H), (0, W2 * 2 - W)), mode='edge')
        ni = ci.reshape(H2, 2, W2, 2, -1).sum((1, 3))
        nw = cw.reshape(H2, 2, W2, 2).sum((1, 3))
        pyr.append((ni, nw))
    ci, cw = pyr[-1]
    col = ci / np.maximum(cw, 1e-6)[..., None]
    for li in range(levels - 1, -1, -1):
        fi, fw = pyr[li]
        H, W = fw.shape
        up = np.repeat(np.repeat(col, 2, 0), 2, 1)[:H, :W]
        up = blur(up, 0.8)
        alpha = np.clip(fw, 0, 1)[..., None]
        col = (fi / np.maximum(fw, 1e-6)[..., None]) * alpha + up * (1 - alpha)
    return col


def ellipse_mask(shape, pts, grow=1.0):
    """Mask of the ellipse spanned by eye corners/lids (outer, inner, upper, lower)."""
    H, W = shape
    o, i, u, l = (np.asarray(p, float) for p in pts)
    c = (o + i) / 2
    ax = (o - i) / np.linalg.norm(o - i)
    nm = np.array([ax[1], -ax[0]])
    a = np.linalg.norm(o - i) / 2 * grow
    bu = abs((u - c) @ nm) * grow
    bl = abs((l - c) @ nm) * grow
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.stack([xx - c[0], yy - c[1]], -1)
    s = d @ ax
    t = d @ nm
    b = np.where(t * ((u - c) @ nm) > 0, bu, bl)
    return ((s / a) ** 2 + (t / b) ** 2) < 1.0


def lip_mask(shape, lm, grow=1.12):
    H, W = shape
    L, R = np.array(lm["mouth_L"], float), np.array(lm["mouth_R"], float)
    top, bot, seam = (np.array(lm[k], float) for k in ("lip_top", "lip_bottom", "lip_seam"))
    c = (L + R) / 2
    ax = (L - R) / np.linalg.norm(L - R)
    nm = np.array([ax[1], -ax[0]])
    if nm[1] > 0:
        nm = -nm                                   # up in the image
    half = np.linalg.norm(L - R) / 2 * grow
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.stack([xx - c[0], yy - c[1]], -1)
    s = (d @ ax) / half
    t = d @ nm
    seam_t = (seam - c) @ nm
    up = ((top - c) @ nm - seam_t) * grow
    dn = (seam_t - (bot - c) @ nm) * grow
    prof = np.clip(1 - s * s, 0, 1) ** 0.6
    inside_up = (t - seam_t) < up * prof
    inside_dn = (seam_t - t) < dn * prof
    return (np.abs(s) < 1) & inside_up & inside_dn


def photo_skin_layers(photo, lm2d):
    """Return (clean photo, confidence) in photo space."""
    H, W, _ = photo.shape
    mx, mn = photo.max(2), photo.min(2)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    val = mx
    r, g, b = photo[..., 0], photo[..., 1], photo[..., 2]
    # skin: bright, low saturation, red >= green >= blue
    skin = smoothstep(0.72, 0.84, val) * smoothstep(0.36, 0.26, sat) * (r >= g) * (g >= b * 0.95)
    # local darkness relative to the neighbourhood (thin hair strands over skin)
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    loc = blur(lum, 4.0)
    skin *= smoothstep(-0.07, -0.025, lum - loc)
    lips = lip_mask((H, W), lm2d)
    skin = np.maximum(skin, lips.astype(np.float32))
    eye_ring = np.zeros((H, W), bool)
    for s_ in ("L", "R"):
        pts = [lm2d["eye_%s_%s" % (s_, q)] for q in ("outer", "inner", "upper", "lower")]
        inner = ellipse_mask((H, W), pts, 0.9)
        skin *= ~inner
        eye_ring |= ellipse_mask((H, W), pts, 1.45) & ~inner
    # face/neck region of interest (keeps sweater-pink and background highlights out)
    yy, xx = np.mgrid[0:H, 0:W]
    roi = ((xx - 640) / 260.0) ** 2 + ((yy - 430) / 300.0) ** 2 < 1.0
    skin *= roi
    # face frame: above the brows the photo shows bangs (strand highlights read as skin) -> drop
    eL, eR = np.array(lm2d["eye_L_iris"], float), np.array(lm2d["eye_R_iris"], float)
    mid = (eL + eR) / 2
    ex = (eL - eR) / np.linalg.norm(eL - eR)
    dn = np.array([-ex[1], ex[0]])
    iod = np.linalg.norm(eL - eR)
    dd = (xx - mid[0]) * dn[0] + (yy - mid[1]) * dn[1]
    skin *= smoothstep(-0.30 * iod, -0.22 * iod, dd)
    # keep the photographed lash line / lid shading in a thin ring around each eye opening
    skin = np.maximum(skin, eye_ring * smoothstep(-0.30 * iod, -0.22 * iod, dd))
    # below the knit collar line
    skin *= smoothstep(628, 612, yy)
    # erode a little so mixed edge pixels (hair/skin) are not used
    skin = np.clip((blur(skin, 1.5) - 0.5) * 2.2, 0, 1)
    clean = pull_push(photo, (skin > 0.5).astype(np.float32))
    # partial de-lighting: divide out large-scale luminance variation inside the skin
    lum_c = 0.3 * clean[..., 0] + 0.59 * clean[..., 1] + 0.11 * clean[..., 2]
    # low-pass of the *skin* luminance only (normalised convolution), so dark lashes, brows
    # and lips do not drive the gain and get washed out
    wsk = ((skin > 0.5) & ~eye_ring & ~lips).astype(np.float32)
    low = blur(lum_c * wsk, 22.0) / np.maximum(blur(wsk, 22.0), 1e-3)
    low = np.where(blur(wsk, 22.0) > 0.02, low, blur(lum_c, 22.0))
    ref = np.median(lum_c[skin > 0.5])
    gain = (ref / np.maximum(low, 0.05)) ** 0.55
    gain = np.where(lips[..., None], 1.0, gain[..., None])
    clean = np.clip(clean * gain, 0, 1)
    return clean, skin, lips


def bilinear(img, x, y):
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    if img.ndim == 2:
        img = img[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return (a * (1 - fy) + b * fy).squeeze()


def tps_morph(src, dst, lam=1.0):
    """2-D thin-plate spline f with f(src) ~= dst - src (returns callable on Nx2)."""
    n = len(src)

    def U(r2):
        return np.where(r2 > 0, r2 * np.log(np.maximum(r2, 1e-12)), 0.0)
    K = U(((src[:, None] - src[None]) ** 2).sum(-1)) + lam * np.eye(n)
    P = np.c_[np.ones(n), src]
    A = np.zeros((n + 3, n + 3))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n] = dst - src
    sol = np.linalg.solve(A, rhs)

    def f(q):
        out = np.empty_like(q)
        for i in range(0, len(q), 200000):
            qq = q[i:i + 200000]
            Kq = U(((qq[:, None] - src[None]) ** 2).sum(-1))
            out[i:i + 200000] = Kq @ sol[:n] + np.c_[np.ones(len(qq)), qq] @ sol[n:]
        return out
    return f


# --------------------------------------------------------------------------- baking
def bake_attribute_maps(ob, uv_name, size):
    """Bake world position, world normal and coverage of ob's evaluated surface into float
    images laid out in uv_name.  Returns (P, N, cov) arrays of shape (size, size, ...)."""
    sc = bpy.context.scene
    saved_mats = [s.material for s in ob.material_slots]
    m = bpy.data.materials.new("_bake_tmp")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    tex = nt.nodes.new("ShaderNodeTexImage")
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = uv_name
    nt.links.new(uvn.outputs[0], tex.inputs[0])
    nt.nodes.active = tex
    for i in range(len(ob.material_slots)):
        ob.material_slots[i].material = m
    if not ob.material_slots:
        ob.data.materials.append(m)
    sc.render.engine = 'CYCLES'
    samples = sc.cycles.samples
    sc.cycles.samples = 1
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    res = {}
    for key, sock, margin in (("P", "Position", 16), ("N", "Normal", 16), ("C", None, 0)):
        img = bpy.data.images.new("_bake_" + key, size, size, float_buffer=True, alpha=False)
        img.colorspace_settings.name = 'Non-Color'
        tex.image = img
        for l in list(nt.links):
            if l.to_node == em and l.to_socket.name == "Color":
                nt.links.remove(l)
        if sock is None:
            em.inputs["Color"].default_value = (1, 1, 1, 1)
        else:
            nt.links.new(geo.outputs[sock], em.inputs["Color"])
        bpy.ops.object.bake(type='EMIT', margin=margin, use_clear=True, margin_type='EXTEND')
        a = np.empty(size * size * 4, np.float32)
        img.pixels.foreach_get(a)
        res[key] = a.reshape(size, size, 4)[..., :3].copy()
        bpy.data.images.remove(img)
    sc.cycles.samples = samples
    for i, mt in enumerate(saved_mats):
        ob.material_slots[i].material = mt
    bpy.data.materials.remove(m)
    return res["P"], res["N"], res["C"][..., 0] > 0.5


def visibility(ob, P, cam_pos, sel):
    """Self-occlusion test from the camera for texels in sel (bool mask)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    mw = ob.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    me.calc_loop_triangles()
    tris = [tuple(t.vertices) for t in me.loop_triangles]
    ev.to_mesh_clear()
    bvh = BVHTree.FromPolygons(verts, tris)
    vis = np.zeros(sel.shape, bool)
    idx = np.argwhere(sel)
    cp = Vector(cam_pos.tolist())
    for (i, j) in idx:
        p = P[i, j]
        d = Vector(p.tolist()) - cp
        L = d.length
        hit = bvh.ray_cast(cp, d / L, L - 0.0015)
        vis[i, j] = hit[0] is None
    return vis


def photo_iris_image(name, photo, ob, r_eye, iris_r, cam_proj, morph, eye_pts, size=512, gain=1.0):
    """Iris texture for one eyeball: planar map along the eyeball's local +Z (gaze) axis,
    sampled from the photo through the beauty camera.  Catchlights are removed (they come
    from the real cornea shader instead) and alpha limits the photo to the visible iris."""
    S = iris_r * 1.3
    g = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(g * S, g * S)                      # rows = v (bottom-up), cols = u
    rr = np.sqrt(X * X + Y * Y)
    Z = np.sqrt(np.maximum(r_eye * r_eye - rr * rr, 0))
    M = np.array(ob.matrix_world)
    Pl = np.stack([X, Y, Z], -1).reshape(-1, 3)
    Pw = Pl @ M[:3, :3].T + M[:3, 3]
    uv = cam_proj(Pw)
    uv = uv + morph(uv)
    col = bilinear(photo, uv[:, 0], uv[:, 1]).reshape(size, size, 3)
    lum = (0.3 * col[..., 0] + 0.59 * col[..., 1] + 0.11 * col[..., 2])
    t = rr / iris_r
    ok = (lum < 0.42) & (t < 1.08)                        # catchlight / glare removed
    ell = ellipse_mask((1306, 1205), eye_pts, 0.93)
    ui = np.clip(uv[:, 0].astype(int), 0, 1204)
    vi = np.clip(uv[:, 1].astype(int), 0, 1305)
    in_open = ell[vi, ui].reshape(size, size)
    filled = pull_push(col, ok.astype(np.float32), levels=7)
    alpha = smoothstep(1.04, 0.90, t) * in_open
    alpha = blur(alpha.astype(np.float32), 1.5)
    rgba = np.ones((size, size, 4), np.float32)
    lin = np.where(filled <= 0.04045, filled / 12.92, ((filled + 0.055) / 1.055) ** 2.4) * gain
    rgba[..., :3] = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.maximum(lin, 0) ** (1 / 2.4) - 0.055)
    rgba[..., 3] = alpha
    img = bpy.data.images.new(name, size, size, alpha=True)
    img.colorspace_settings.name = 'sRGB'
    img.alpha_mode = 'CHANNEL_PACKED'
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img, S
