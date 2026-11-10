"""Generated (hand-authored, non-photographic) skin maps painted in the MakeHuman UV layout + the skin shader."""
import math
import numpy as np
import bpy
from . import util
from .base import smoothstep
from .sculpt import EYE_C0 as EYE_C
from .eyes import BROW_PTS, lid_margin

MOUTH_Z = 1.4777
# vermilion borders measured on the supplied topology (x mm, dz mm from the mouth line), see README
LIP_UP = [(0, 8.0), (2.3, 8.2), (5.3, 9.3), (8.0, 10.3), (11.0, 9.9), (14.5, 8.8), (17.9, 7.2), (21.0, 5.2), (24.0, 4.0), (26.3, 2.0), (27.6, 0.3)]
LIP_LO = [(0, -10.8), (5.5, -10.4), (12.7, -8.8), (18.5, -6.2), (22.9, -4.6), (25.0, -3.0), (26.3, -1.8), (27.6, -0.3)]


def rasterize(uv, attr, size):
    """uv: (n,3,2) in [0,1]; attr: (n,3,k). Returns (size,size,k) buffer and a filled mask. Row 0 = v=0."""
    n, _, k = attr.shape
    buf = np.zeros((size, size, k), dtype=np.float32)
    filled = np.zeros((size, size), dtype=bool)
    P = uv * size - 0.5
    x0 = np.floor(P[:, :, 0].min(1)).astype(np.int64); x1 = np.ceil(P[:, :, 0].max(1)).astype(np.int64)
    y0 = np.floor(P[:, :, 1].min(1)).astype(np.int64); y1 = np.ceil(P[:, :, 1].max(1)).astype(np.int64)
    ext = np.maximum(x1 - x0, y1 - y0) + 1
    a = P[:, 0]; b = P[:, 1]; c = P[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) > 1e-12
    K = 2
    while True:
        sel = np.where(ok & (ext <= K) & (ext > (K // 2 if K > 2 else 0)))[0]
        if len(sel):
            g = np.arange(K)
            chunk = max(1, int(2_000_000 // (K * K)))
            for s in range(0, len(sel), chunk):
                t = sel[s:s + chunk]
                X = x0[t, None, None] + g[None, None, :]
                Y = y0[t, None, None] + g[None, :, None]
                X = np.broadcast_to(X, (len(t), K, K)); Y = np.broadcast_to(Y, (len(t), K, K))
                w0 = ((b[t, 1] - c[t, 1])[:, None, None] * (X - c[t, 0][:, None, None]) + (c[t, 0] - b[t, 0])[:, None, None] * (Y - c[t, 1][:, None, None])) / den[t][:, None, None]
                w1 = ((c[t, 1] - a[t, 1])[:, None, None] * (X - c[t, 0][:, None, None]) + (a[t, 0] - c[t, 0])[:, None, None] * (Y - c[t, 1][:, None, None])) / den[t][:, None, None]
                w2 = 1 - w0 - w1
                e = -0.02
                ins = (w0 >= e) & (w1 >= e) & (w2 >= e) & (X >= 0) & (X < size) & (Y >= 0) & (Y < size)
                ti, yy, xx = np.nonzero(ins)
                val = (w0[ti, yy, xx][:, None] * attr[t[ti], 0] + w1[ti, yy, xx][:, None] * attr[t[ti], 1] + w2[ti, yy, xx][:, None] * attr[t[ti], 2])
                buf[Y[ti, yy, xx], X[ti, yy, xx]] = val
                filled[Y[ti, yy, xx], X[ti, yy, xx]] = True
        if K >= ext.max():
            break
        K *= 2
    return buf, filled


def dilate(buf, filled, iters=12):
    buf = buf.copy(); filled = filled.copy()
    for _ in range(iters):
        acc = np.zeros_like(buf); cnt = np.zeros(filled.shape, dtype=np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.roll(buf, (dy, dx), axis=(0, 1)); sf = np.roll(filled, (dy, dx), axis=(0, 1))
            acc += sh * sf[..., None]; cnt += sf
        new = (~filled) & (cnt > 0)
        buf[new] = acc[new] / cnt[new][:, None]
        filled |= new
    return buf, filled


def _poly_dist(P, poly):
    """Distance from points P (m,3) to a 3D polyline (k,3)."""
    best = np.full(len(P), 1e9)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a; t = np.clip(((P - a) @ ab) / (ab @ ab + 1e-18), 0, 1)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def _blur(a, r):
    """Cheap separable box blur (applied twice) for 2D arrays."""
    if r <= 0:
        return a
    out = a.astype(np.float32)
    for _ in range(2):
        for ax in (0, 1):
            c = np.cumsum(np.concatenate([np.zeros_like(np.take(out, [0], ax)), out], ax), ax)
            n = out.shape[ax]
            lo = np.clip(np.arange(n) - r, 0, n); hi = np.clip(np.arange(n) + r + 1, 0, n)
            out = (np.take(c, hi, ax) - np.take(c, lo, ax)) / (hi - lo).reshape([-1 if i == ax else 1 for i in range(2)])
    return out


def paint_skin(body, size=2048, tone=None, photo=None):
    """Returns dict of numpy images (row 0 = v 0): color (sRGB), data (R rough, G coat, B bump height).

    photo (second pass only): dict(img, cam, warp, Mh, bvh) - the registered reference photograph is projected
    onto the posed head and blended in where it shows clean, camera-facing skin."""
    tone = tone or {}
    F = body.F
    uvs = np.array([uv for f in body.UV for uv in f], dtype=np.float64)
    V0s, Fs, UVs = util.subdivided(body.V0, F, 2, uvs)
    V1s, _, _ = util.subdivided(body.V, F, 2, None)
    A = np.stack([body.enclosed.astype(float), np.zeros(len(body.V)), np.zeros(len(body.V))], 1)
    As, _, _ = util.subdivided(A, F, 2, None)
    enc_s = As[:, 0]
    # two triangles per quad
    tri = np.concatenate([Fs[:, [0, 1, 2]], Fs[:, [0, 2, 3]]])
    tuv = np.concatenate([UVs[:, [0, 1, 2]], UVs[:, [0, 2, 3]]])
    N1s = util.vertex_normals(V1s, Fs)
    attr = np.concatenate([V0s, V1s, enc_s[:, None], N1s], 1).astype(np.float32)
    buf, filled = rasterize(tuv, attr[tri], size)
    buf, filled2 = dilate(buf, filled, 10)
    P0 = buf[..., 0:3].astype(np.float64); P1 = buf[..., 3:6].astype(np.float64); enc = np.clip(buf[..., 6], 0, 1)
    N1 = buf[..., 7:10].astype(np.float64)
    H, W = size, size
    x = P0[..., 0]; y = P0[..., 1]; z = P0[..., 2]; ax = np.abs(x)

    head = smoothstep(z, 1.40, 1.44)
    face = head * (1 - smoothstep(y, -0.075, -0.035))              # frontal half of the head
    noise_lo = util.fbm3(P0 * 55.0, 4, 1)
    noise_mid = util.fbm3(P0 * 260.0, 4, 2)
    noise_hi = util.fbm3(P0 * 1400.0, 3, 3)

    base = np.array(tone.get('base', (0.915, 0.762, 0.672)))
    col = np.ones((H, W, 3)) * base
    # soft warm/cool mottling
    col *= (1.0 + (noise_lo[..., None] - 0.5) * np.array([0.05, 0.075, 0.09]))
    col *= (1.0 + (noise_mid[..., None] - 0.5) * np.array([0.035, 0.06, 0.07]))
    col *= (1.0 + (noise_hi[..., None] - 0.5) * np.array([0.02, 0.035, 0.04]))

    def tint(mask, colour, amount):
        nonlocal col
        m = np.clip(mask * amount, 0, 1)[..., None]
        col = col * (1 - m) + np.array(colour) * m

    def blob(c, r, sx=1.0, sy=1.0, sz=1.0):
        d = np.sqrt(((ax - c[0]) * sx) ** 2 + ((y - c[1]) * sy) ** 2 + ((z - c[2]) * sz) ** 2)
        t = np.clip(d / r, 0, 1)
        return (1 - t * t) ** 2

    blush = blob((0.043, -0.123, 1.506), 0.040, 1.0, 0.6, 1.10) * face
    tint(blush, (0.940, 0.61, 0.60), 0.40 * tone.get('blush', 1.0))
    tint(blob((0.0, -0.166, 1.508), 0.013) * face, (0.93, 0.60, 0.55), 0.22)        # nose tip
    tint(blob((0.0, -0.145, 1.442), 0.022) * face, (0.93, 0.62, 0.57), 0.13)        # chin
    tint(blob((0.0, -0.140, 1.600), 0.060) * face, (0.94, 0.80, 0.72), 0.10)        # forehead a touch lighter
    ear = smoothstep(ax, 0.070, 0.078) * head * smoothstep(z, 1.485, 1.50) * (1 - smoothstep(z, 1.565, 1.58))
    tint(ear, (0.92, 0.60, 0.54), 0.30)
    # nose: soft contact shading in the alar creases and under the tip (keeps a small nose readable in flat light)
    alar = blob((0.0145, -0.150, 1.5035), 0.0075, 1.0, 0.7, 1.0) * face
    tint(alar, (0.80, 0.56, 0.50), 0.34)
    under = blob((0.0, -0.158, 1.4975), 0.0080, 1.0, 0.8, 1.6) * face
    tint(under, (0.78, 0.52, 0.47), 0.38)
    side = blob((0.0105, -0.152, 1.520), 0.012, 1.6, 0.5, 0.7) * face
    tint(side, (0.86, 0.66, 0.60), 0.16)
    # orbital area: soft warm shadow on the lid, lighter under-eye roll
    eye_c = np.array([abs(EYE_C[0]), EYE_C[1], EYE_C[2]])
    de = np.sqrt((ax - eye_c[0]) ** 2 + (z - eye_c[2]) ** 2)
    lid = (1 - smoothstep(de, 0.011, 0.020)) * smoothstep(z - eye_c[2], -0.002, 0.004) * face
    tint(lid, (0.80, 0.56, 0.50), 0.30)
    under = np.exp(-((z - (eye_c[2] - 0.0105)) / 0.004) ** 2) * (1 - smoothstep(np.abs(ax - eye_c[0] - 0.002), 0.010, 0.017)) * face
    tint(under, (0.95, 0.75, 0.70), 0.22)
    # lash line (eyeliner) and waterline from the true lid-margin curves of the supplied topology
    box = (ax > eye_c[0] - 0.022) & (ax < eye_c[0] + 0.026) & (np.abs(z - eye_c[2]) < 0.016) & (y < -0.115)
    pts = np.stack([ax[box], y[box], z[box]], 1)
    up_ids, lo_ids = lid_margin(body, 1.0)
    d_up = _poly_dist(pts, V0s[up_ids]); d_lo = _poly_dist(pts, V0s[lo_ids])
    m = np.zeros((H, W)); m[box] = (1 - smoothstep(d_up, 0.0004, 0.0017)); tint(m * (1 - enc), (0.20, 0.115, 0.085), 0.92)
    # soft double-eyelid fold a few millimetres above the lash line
    up_only = np.zeros((H, W)); up_only[box] = np.exp(-((d_up - 0.0031) / 0.0007) ** 2)
    tint(up_only * (z > eye_c[2] + 0.0035) * (1 - enc) * face, (0.74, 0.50, 0.45), 0.40)
    # faint tear trough and philtrum shading
    tint(blob((eye_c[0] - 0.012, -0.142, eye_c[2] - 0.011), 0.008, 1.0, 0.4, 1.6) * face, (0.86, 0.66, 0.63), 0.20)
    tint(blob((0.0, -0.157, 1.4935), 0.0045, 1.6, 0.5, 0.8) * face, (0.84, 0.60, 0.56), 0.22)
    wing = blob((eye_c[0] + 0.0150, -0.1335, eye_c[2] + 0.0012), 0.0042, 1.0, 0.3, 2.6) * face
    tint(wing, (0.22, 0.13, 0.10), 0.55)
    m = np.zeros((H, W)); m[box] = (1 - smoothstep(d_lo, 0.0003, 0.0013)); tint(m * (1 - enc), (0.62, 0.36, 0.32), 0.50)
    # brows: soft tint under the hair curves (sculpted-space positions)
    bp = np.array(BROW_PTS)
    bx = np.abs(P1[..., 0]); bz = P1[..., 2]
    bm = np.zeros((H, W))
    sel = (bz > 1.548) & (bz < 1.576) & (bx < 0.064) & (P1[..., 1] < -0.11)
    q = np.stack([bx[sel], bz[sel]], 1)
    best = np.full(len(q), 1e9); hw = np.zeros(len(q))
    for (a0, a1) in zip(bp[:-1], bp[1:]):
        ab = a1[:2] - a0[:2]; t = np.clip(((q - a0[:2]) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(q - (a0[:2] + t[:, None] * ab), axis=1)
        upd = d < best; best[upd] = d[upd]; hw[upd] = (a0[2] + t * (a1[2] - a0[2]))[upd]
    bm[sel] = 1 - smoothstep(best / np.maximum(hw, 1e-6), 0.55, 1.25)
    tint(bm, (0.60, 0.45, 0.38), 0.34)

    # lips ---------------------------------------------------------------------------------------
    up = np.array(LIP_UP) * 1e-3; lo = np.array(LIP_LO) * 1e-3
    dz = z - MOUTH_Z
    zu = np.interp(ax, up[:, 0], up[:, 1], right=-1); zl = np.interp(ax, lo[:, 0], lo[:, 1], right=1)
    edge = 0.0009
    lipm = smoothstep(zu - dz, -edge * 0.3, edge) * smoothstep(dz - zl, -edge * 0.3, edge)
    lipm *= (y < -0.128) * (1 - smoothstep(ax, 0.0262, 0.0282))
    lipm = _blur(lipm, 1)
    inner = np.exp(-(dz / 0.0030) ** 2)                                # deeper colour towards the mouth line
    crease = util.fbm2(ax * 2600.0, dz * 260.0, 3, 7)                  # vertical lip lines
    lip_col = np.array(tone.get('lip', (0.850, 0.375, 0.395)))[None, None] * (0.93 + 0.14 * crease[..., None])
    lip_col = lip_col * (1 - 0.16 * inner[..., None]) + np.array([0.76, 0.33, 0.35]) * (0.16 * inner[..., None])
    mm = np.clip(lipm, 0, 1)[..., None] * 0.92
    col = col * (1 - mm) + lip_col * mm

    # scalp under the hair: brown under-tone with an irregular, feathered hairline
    from .hair import scalp_depth
    flatP = P0.reshape(-1, 3)
    dep = scalp_depth(flatP)[0].reshape(H, W) + (noise_mid - 0.5) * 7.0 + (noise_hi - 0.5) * 5.0
    scalp = (0.45 * smoothstep(dep, -2.0, 4.0) + 0.55 * smoothstep(dep, 2.0, 14.0)) * head
    tint(scalp, (0.22, 0.125, 0.085), 0.88)

    # tiny moles / freckles (hand placed, restrained)
    for (mx, my, mz, r, a) in ((0.031, -0.128, 1.5165, 0.0007, 0.55), (-0.047, -0.118, 1.498, 0.0006, 0.45), (0.018, -0.150, 1.452, 0.0005, 0.4)):
        d = np.sqrt((x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2)
        tint(1 - smoothstep(d, r * 0.5, r), (0.45, 0.30, 0.24), a)
    fr = np.clip((noise_hi - 0.80) / 0.06, 0, 1) * blush
    tint(fr, (0.78, 0.55, 0.46), 0.18)

    # neck / chest / hands a little less rosy; knuckles and finger tips warmer
    body_m = 1 - head
    col = col * (1 - body_m[..., None] * 0.04)
    hand = smoothstep(ax, 0.36, 0.40) * (z < 1.12)
    tint(hand * np.clip((noise_mid - 0.45) * 2.5, 0, 1), (0.93, 0.66, 0.60), 0.18)

    # cavities ------------------------------------------------------------------------------------
    mouth_in = enc * (z < 1.515) * (z > 1.44) * (ax < 0.04) * (y > -0.158)
    eye_in = enc * (np.abs(z - eye_c[2]) < 0.02)
    tint(enc * (1 - np.clip(eye_in, 0, 1)), (0.55, 0.30, 0.27), 0.85)
    tint(mouth_in, (0.20, 0.06, 0.06), 0.9)
    tint(eye_in, (0.90, 0.64, 0.60), 0.75)          # waterline / lid margin: pale pink, never dark

    col = np.clip(col, 0, 1)
    info = {}; photo_weight = None
    if photo is not None:
        from . import photo as ph
        R = photo['Mh'][:3, :3]; t = photo['Mh'][:3, 3]
        Ppos = P1 @ R.T + t
        Npos = N1 @ R.T
        Npos /= np.linalg.norm(Npos, axis=2, keepdims=True) + 1e-12
        # central face only: below the fringe, inside the cheeks, above the jaw line; never ears, scalp or neck
        xmax = np.interp(z, [1.437, 1.452, 1.478, 1.500, 1.545, 1.566], [0.020, 0.034, 0.047, 0.056, 0.060, 0.056])
        # the photo's fringe and brows cover the skin above the eyes: stop below the upper lids there and only run
        # a little higher on the bridge of the nose
        top = 1.5430 + 0.0030 * (1 - smoothstep(ax, 0.004, 0.010))
        region = (1 - smoothstep(ax, xmax - 0.007, xmax)) * smoothstep(z, 1.432, 1.441) * (1 - smoothstep(z, top, top + 0.005))
        region *= (y < -0.085) * face
        # keep clear of the lid margins and the eye openings (the eyeballs are separate geometry)
        near_lid = np.zeros((H, W)); near_lid[box] = 1 - smoothstep(np.minimum(d_up, d_lo), 0.0012, 0.0030)
        region *= (1 - near_lid)
        valid = filled2 & (region > 0.002)
        pcol, pw = ph.project_face(photo['img'], photo['cam'], photo['warp'], Ppos, Npos, valid, photo['bvh'], enc, region, lips=_blur(lipm, 2))
        # The photograph is already lit and exposed, so its values are not albedo. Bring both sources to one common
        # level: the generated complexion's value with a little of the photographed skin's hue.
        cheek = (pw > 0.7) & (blob((0.040, -0.125, 1.500), 0.030) > 0.2) & (lipm < 0.05)
        if cheek.sum() > 500:
            med_photo = np.median(pcol[cheek], 0); med_proc = np.median(col[cheek], 0)
            hue = med_photo / med_photo.mean() / (med_proc / med_proc.mean())
            target = med_proc * (1.0 + 0.6 * (hue - 1.0))
            pcol = np.clip(pcol * (target / np.maximum(med_photo, 1e-3)), 0, 1)
            skin_only = (1 - np.clip(scalp, 0, 1) * 0.9)[..., None] * (1 - np.clip(lipm, 0, 1))[..., None]
            col = col * (1 - skin_only) + np.clip(col * (target / np.maximum(med_proc, 1e-3)), 0, 1) * skin_only
            info['photo_gain'] = [round(float(g), 4) for g in target / med_photo]
            info['generated_gain'] = [round(float(g), 4) for g in target / med_proc]
        # remove the remaining low-frequency difference towards the edge of the projected area
        rb = max(8, size // 64)
        solid = (pw > 0.25).astype(np.float64)
        den = np.maximum(_blur(solid, rb), 1e-4)
        lo_photo = np.stack([_blur(pcol[..., k] * solid, rb) for k in range(3)], 2) / den[..., None]
        lo_proc = np.stack([_blur(col[..., k] * solid, rb) for k in range(3)], 2) / den[..., None]
        ratio = np.clip(lo_proc / np.maximum(lo_photo, 1e-3), 0.85, 1.18)
        edge = (1 - smoothstep(pw, 0.35, 0.98))[..., None]
        pcol = np.clip(pcol * (1 + (ratio - 1) * edge), 0, 1)
        pw = np.clip(_blur(pw, max(3, size // 256)), 0, 1) * (pw > 0.01)
        col = col * (1 - pw[..., None]) + pcol * pw[..., None]
        info['photo_texels'] = int((pw > 0.5).sum()); info['photo_weight_max'] = round(float(pw.max()), 3)
        photo_weight = pw
        col = np.clip(col, 0, 1)
    # ---- data map ---------------------------------------------------------------------------------
    rough = np.full((H, W), 0.47)
    rough -= 0.10 * blob((0.0, -0.160, 1.515), 0.030) * face                 # nose / T-zone
    rough -= 0.05 * blob((0.0, -0.140, 1.600), 0.050) * face
    rough -= 0.06 * blob((0.043, -0.128, 1.512), 0.030) * face               # dewy cheek-bones
    rough += 0.05 * (noise_mid - 0.5)
    rough = rough * (1 - lipm) + (0.19 + 0.10 * crease) * lipm
    rough = np.where(enc > 0.5, 0.30, rough)
    coat = np.clip(lipm * (0.70 + 0.3 * (1 - crease)) + 0.06 * face * (1 - lipm), 0, 1)
    bump = 0.5 + 0.5 * (crease - 0.5) * lipm
    data = np.stack([np.clip(rough, 0.05, 1), coat, np.clip(bump, 0, 1)], 2)
    return dict(color=col.astype(np.float32), data=data.astype(np.float32), lip=lipm, filled=filled, info=info, photo_weight=photo_weight)


def skin_material(maps, sss_scale=0.0048):
    img_c = util.new_image('Skin_Color_Generated', maps['color'], 'sRGB')
    img_d = util.new_image('Skin_Data_Generated', maps['data'], is_data=True)
    m = bpy.data.materials.new('Skin'); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexImage'); tc.image = img_c
    td = nt.nodes.new('ShaderNodeTexImage'); td.image = img_d
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    nt.links.new(td.outputs['Color'], sep.inputs['Color'])
    nt.links.new(tc.outputs['Color'], p.inputs['Base Color'])
    nt.links.new(sep.outputs['Red'], p.inputs['Roughness'])
    nt.links.new(sep.outputs['Green'], p.inputs['Coat Weight'])
    p.inputs['Coat Roughness'].default_value = 0.16
    p.subsurface_method = 'RANDOM_WALK_SKIN'
    p.inputs['Subsurface Weight'].default_value = 1.0
    p.inputs['Subsurface Radius'].default_value = (1.0, 0.36, 0.20)
    p.inputs['Subsurface Scale'].default_value = sss_scale
    p.inputs['Subsurface IOR'].default_value = 1.40
    p.inputs['Subsurface Anisotropy'].default_value = 0.6
    p.inputs['IOR'].default_value = 1.42
    p.inputs['Specular IOR Level'].default_value = 0.45
    # restrained micro relief: two procedural scales plus the painted lip lines
    geo = nt.nodes.new('ShaderNodeTexCoord')
    n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = 2600.0; n1.inputs['Detail'].default_value = 2.0
    n2 = nt.nodes.new('ShaderNodeTexVoronoi'); n2.inputs['Scale'].default_value = 5200.0
    nt.links.new(geo.outputs['Object'], n1.inputs['Vector']); nt.links.new(geo.outputs['Object'], n2.inputs['Vector'])
    add = nt.nodes.new('ShaderNodeMath'); add.operation = 'ADD'
    nt.links.new(n1.outputs['Fac'], add.inputs[0])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'; mul.inputs[1].default_value = 0.6
    nt.links.new(n2.outputs['Distance'], mul.inputs[0]); nt.links.new(mul.outputs[0], add.inputs[1])
    b1 = nt.nodes.new('ShaderNodeBump'); b1.inputs['Strength'].default_value = 0.10; b1.inputs['Distance'].default_value = 0.00012
    nt.links.new(add.outputs[0], b1.inputs['Height'])
    b2 = nt.nodes.new('ShaderNodeBump'); b2.inputs['Strength'].default_value = 0.35; b2.inputs['Distance'].default_value = 0.0004
    nt.links.new(sep.outputs['Blue'], b2.inputs['Height']); nt.links.new(b1.outputs['Normal'], b2.inputs['Normal'])
    nt.links.new(b2.outputs['Normal'], p.inputs['Normal'])
    m.diffuse_color = (0.80, 0.55, 0.45, 1)
    return m
