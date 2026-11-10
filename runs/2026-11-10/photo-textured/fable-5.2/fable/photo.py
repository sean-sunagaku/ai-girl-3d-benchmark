"""Second pass: use the supplied reference photograph as *texture source* on the real 3D head.

Everything here feeds UV textures or scalar colours of the existing shaped mesh; the photo is never used as a
plane, a background or a composite. The steps:

  * the photograph is projected through the fitted selfie camera onto the posed, sculpted head and baked into
    the MakeHuman UV layout (only where the photo shows clean, camera-facing, unoccluded skin);
  * a thin-plate-spline warp driven by a dozen facial landmarks registers photo features to the geometry, so
    lips land on the lips and lid lines on the lids (no doubled features);
  * hair, background, the eye openings, the fringe-covered forehead, the ears and everything turned away
    from the camera are masked out and keep the tone-matched generated skin;
  * iris, hair and knit colours are sampled from clean photo regions.
"""
import math
import os
import numpy as np
import bpy
from mathutils import Vector
from . import util
from .base import smoothstep

# landmark pixel positions measured by hand on input/reference.png (x right, y down, 1205 x 1306)
PHOTO_LANDMARKS = {
    'pupil_R': (508, 391), 'pupil_L': (643, 315),
    'inner_R': (538, 391), 'outer_R': (468, 408), 'inner_L': (613, 343), 'outer_L': (686, 302),
    'nose_tip': (613, 445), 'subnasale': (625, 467), 'lip_top': (647, 478), 'stomion': (658, 493),
    'lip_bot': (667, 520), 'corner_R': (598, 515), 'corner_L': (715, 445), 'chin': (686, 574),
}
BASE_LANDMARK_VERTS = {       # nearest cage vertex to these rest-space (supplied base) positions
    'nose_tip': (0.0, -0.1681, 1.5077), 'subnasale': (0.0, -0.1563, 1.4957), 'lip_top': (0.0, -0.1597, 1.4864),
    'stomion': (0.0, -0.1553, 1.4776), 'lip_bot': (0.0, -0.1557, 1.4657), 'chin': (0.0, -0.1378, 1.4336),
    'corner_L': (0.0249, -0.1419, 1.4777), 'corner_R': (-0.0249, -0.1419, 1.4777),
    'inner_L': (0.0193, -0.1375, 1.5448), 'outer_L': (0.0437, -0.1355, 1.5450),
    'inner_R': (-0.0193, -0.1375, 1.5448), 'outer_R': (-0.0437, -0.1355, 1.5450),
}


def load_photo(path):
    """sRGB-encoded float array (H, W, 3), row 0 = top of the picture. The image datablock is removed again."""
    img = bpy.data.images.load(path)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1, :, :3].astype(np.float64)


class Camera:
    def __init__(self, spec, width=1205.0, height=1306.0):
        self.pos = np.asarray(spec['position'], float)
        f = np.asarray(spec['target'], float) - self.pos
        self.f = f / np.linalg.norm(f)
        r = np.cross(self.f, [0, 0, 1.0]); self.r = r / np.linalg.norm(r)
        self.up = np.cross(self.r, self.f)
        self.W = width; self.H = height
        self.f_px = (height / 2) / math.tan(math.radians(spec['fov_v']) / 2)

    def project(self, P):
        q = np.asarray(P, float) - self.pos
        depth = q @ self.f
        x = self.W / 2 + self.f_px * (q @ self.r) / depth
        y = self.H / 2 - self.f_px * (q @ self.up) / depth
        return np.stack([x, y], -1), depth


def tps(src, dst, lam=25.0):
    """Thin-plate spline g with g(src_i) ~= dst_i (2D -> 2D). Returns a callable for (n, 2) arrays."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    n = len(src)

    def U(r2):
        return np.where(r2 > 1e-12, r2 * np.log(np.maximum(r2, 1e-12)) * 0.5, 0.0)
    d2 = ((src[:, None] - src[None]) ** 2).sum(-1)
    K = U(d2) + lam * np.eye(n)
    Pm = np.concatenate([np.ones((n, 1)), src], 1)
    A = np.zeros((n + 3, n + 3)); A[:n, :n] = K; A[:n, n:] = Pm; A[n:, :n] = Pm.T
    b = np.zeros((n + 3, 2)); b[:n] = dst
    sol = np.linalg.solve(A, b)
    w = sol[:n]; a = sol[n:]

    def g(q):
        q = np.asarray(q, float)
        out = a[0] + q @ a[1:]
        step = 200000
        for s in range(0, len(q), step):
            qq = q[s:s + step]
            out[s:s + step] += U(((qq[:, None] - src[None]) ** 2).sum(-1)) @ w
        return out
    return g


def bilinear(img, xy):
    h, w = img.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001); y = np.clip(xy[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy


def _box_blur(a, r):
    out = a.astype(np.float64)
    for ax in (0, 1):
        c = np.cumsum(np.concatenate([np.zeros_like(np.take(out, [0], ax)), out], ax), ax)
        n = out.shape[ax]
        lo = np.clip(np.arange(n) - r, 0, n); hi = np.clip(np.arange(n) + r + 1, 0, n)
        shape = [-1 if i == ax else 1 for i in range(out.ndim)]
        out = (np.take(c, hi, ax) - np.take(c, lo, ax)) / (hi - lo).reshape(shape)
    return out


def _erode(mask, r):
    """Grey-scale erosion with a (2r+1) square window."""
    out = mask.copy()
    for ax in (0, 1):
        acc = out.copy()
        for k in range(1, r + 1):
            acc = np.minimum(acc, np.roll(out, k, ax)); acc = np.minimum(acc, np.roll(out, -k, ax))
        out = acc
    return out


def photo_skin_mask(photo):
    """1 where the photograph shows clean skin, 0 on hair / background / deep shadow (photo pixel space)."""
    R, G, B = photo[..., 0], photo[..., 1], photo[..., 2]
    lum = 0.2126 * R + 0.7152 * G + 0.0722 * B
    warm = R - B
    m = smoothstep(lum, 0.58, 0.72) * smoothstep(warm, 0.04, 0.10) * smoothstep(R, 0.78, 0.88)
    m = _erode(m, 3)
    return np.clip(_box_blur(_box_blur(m, 3), 3), 0, 1), lum


def landmark_pairs(body, V_posed, eye_centres_posed, eye_radius, cam):
    """Projected model landmarks (pixels) and the measured photo landmarks, in matching order."""
    model = []; photo = []
    V0 = body.V0
    for name, p in BASE_LANDMARK_VERTS.items():
        i = int(np.argmin(np.linalg.norm(V0 - np.array(p), axis=1)))
        model.append(cam.project(V_posed[i][None])[0][0]); photo.append(PHOTO_LANDMARKS[name])
    for name, c in (('pupil_L', eye_centres_posed[0]), ('pupil_R', eye_centres_posed[1])):
        d = cam.pos - c; p = c + d / np.linalg.norm(d) * eye_radius
        model.append(cam.project(p[None])[0][0]); photo.append(PHOTO_LANDMARKS[name])
    return np.array(model), np.array(photo, float)


def sample_colours(photo):
    """Median colours of a few clean regions (sRGB). Only these scalars leave this function."""
    def med(x0, y0, x1, y1, keep=None):
        reg = photo[y0:y1, x0:x1].reshape(-1, 3)
        if keep is not None:
            lum = reg @ np.array([0.2126, 0.7152, 0.0722])
            reg = reg[keep(lum)]
        return np.median(reg, 0)
    iris = np.median(np.concatenate([
        _ring(photo, PHOTO_LANDMARKS['pupil_R'], 7, 15), _ring(photo, PHOTO_LANDMARKS['pupil_L'], 7, 15)]), 0)
    return dict(
        iris=iris,
        hair=med(300, 300, 420, 520, lambda l: (l > 0.12) & (l < 0.55)),
        hair_light=med(430, 60, 620, 150, lambda l: (l > 0.2) & (l < 0.7)),
        knit=med(700, 800, 930, 1040, lambda l: l > 0.45),
        knit_shadow=med(700, 800, 930, 1040, lambda l: l < 0.62),
        skin=med(650, 370, 760, 450, lambda l: l > 0.7),
    )


def _ring(photo, c, r0, r1):
    cx, cy = c
    ys, xs = np.mgrid[cy - r1:cy + r1 + 1, cx - r1:cx + r1 + 1]
    d = np.hypot(xs - cx, ys - cy)
    reg = photo[cy - r1:cy + r1 + 1, cx - r1:cx + r1 + 1]
    lum = reg @ np.array([0.2126, 0.7152, 0.0722])
    sel = (d >= r0) & (d <= r1) & (lum < 0.62) & (lum > 0.12)      # skip catch-lights and the pupil/lash line
    return reg[sel]


def project_face(photo, cam, warp, texel_pos, texel_nrm, valid, bvh, enc, region, lips=None):
    """Bake the registered photograph into UV space.

    texel_pos / texel_nrm: (H, W, 3) posed positions and normals per texel; valid: texels to consider;
    region: (H, W) geometric weight of the central face. Returns colour (H, W, 3) and blend weight (H, W)."""
    Hh, Ww = valid.shape
    idx = np.where(valid & (region > 0.002))
    P = texel_pos[idx]; N = texel_nrm[idx]
    px, depth = cam.project(P)
    view = cam.pos[None] - P; dist = np.linalg.norm(view, axis=1); view /= dist[:, None]
    facing = np.sum(N * view, 1)
    inside = (px[:, 0] > 2) & (px[:, 0] < cam.W - 3) & (px[:, 1] > 2) & (px[:, 1] < cam.H - 3) & (depth > 0.05) & (facing > 0.12)
    # occlusion: anything of the body between the texel and the lens (nose over cheek, lids over lids)
    vis = np.zeros(len(P), bool)
    for k in np.where(inside)[0]:
        o = Vector(P[k] + N[k] * 0.0008)
        hit = bvh.ray_cast(o, Vector(view[k]), float(dist[k]))
        vis[k] = hit[0] is None
    src = warp(px)
    skin, lum = photo_skin_mask(photo)
    col = bilinear(photo, src)
    m_skin = bilinear(skin[..., None], src)[:, 0]
    if lips is not None:                                   # lips are darker than skin but are wanted from the photo
        m_lum = bilinear(lum[..., None], src)[:, 0]
        m_skin = np.maximum(m_skin, np.clip(lips[idx], 0, 1) * smoothstep(m_lum, 0.40, 0.52))
    w = vis * smoothstep(facing, 0.12, 0.42) * m_skin * region[idx] * (1 - np.clip(enc[idx], 0, 1))
    colour = np.zeros((Hh, Ww, 3)); weight = np.zeros((Hh, Ww))
    colour[idx] = col; weight[idx] = w
    # feather the weight so the hand-over to the generated skin is soft
    weight = np.minimum(weight, _box_blur(_box_blur(weight, 6), 6) * 1.25)
    return colour, np.clip(weight, 0, 1)
