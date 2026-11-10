"""Long chestnut hair: scalp roots, combed + gravity-settled locks with collision, clumped strands, fine children.

Pipeline (all generated here):
  1. signed distance grid of the posed body + sweater (narrow band, from Blender BVH queries)
  2. a few thousand "lock" strands integrated from the scalp (comb field -> gravity) sliding over that grid
  3. ~10k authored strands = lock + clump offsets + waves + fly-aways, stored as a native Curves object
  4. a Geometry Nodes modifier multiplies them into the final strand count at render time (editable)
  5. tapered ribbon meshes under the long locks (fills gaps; also what a mesh-only exporter can carry)
"""
import math
import numpy as np
import bpy
from mathutils import Vector
from . import util
from .base import smoothstep
from .eyes import curves_object, hair_material

HEAD_C = np.array([0.0, -0.035, 1.56])      # base-space reference point inside the skull
K_PTS = 40


# ------------------------------------------------------------------------------------------ distance field
class SDF:
    def __init__(self, lo, hi, h):
        self.lo = np.asarray(lo, float); self.h = h
        self.n = np.ceil((np.asarray(hi) - self.lo) / h).astype(int) + 1
        self.d = None; self.g = None

    def points(self, step=1):
        ax = [self.lo[k] + np.arange(0, self.n[k], step) * self.h for k in range(3)]
        return np.stack(np.meshgrid(*ax, indexing='ij'), -1)

    def build(self, bvhs, band=0.04, signed_within=None):
        """signed_within[i]: beyond this distance mesh i is treated as unsigned (open shells such as the sweater)."""
        signed_within = signed_within or [1e9] * len(bvhs)
        coarse = 4
        pc = self.points(coarse)
        shp = pc.shape[:3]
        dc = np.full(shp, 1.0)
        flat = pc.reshape(-1, 3)
        out = np.full(len(flat), 1.0)
        for bvh, lim in zip(bvhs, signed_within):
            for i, p in enumerate(flat):
                hit = bvh.find_nearest(Vector(p))
                if hit[0] is None:
                    continue
                d = (p - np.array(hit[0]))
                sd = hit[3] if (float(d @ np.array(hit[1])) >= 0 or hit[3] > lim) else -hit[3]
                if sd < out[i]:
                    out[i] = sd
        dc = out.reshape(shp)
        # upsample to the fine grid (trilinear)
        fine = self.points(1)
        idx = (fine - self.lo) / (self.h * coarse)
        d = _trilinear(dc, idx)
        need = np.abs(d) < band
        pts = fine[need]
        vals = np.full(len(pts), 1.0)
        for bvh, lim in zip(bvhs, signed_within):
            for i, p in enumerate(pts):
                hit = bvh.find_nearest(Vector(p), band * 1.6)
                if hit[0] is None:
                    continue
                dd = (p - np.array(hit[0]))
                sd = hit[3] if (float(dd @ np.array(hit[1])) >= 0 or hit[3] > lim) else -hit[3]
                if sd < vals[i]:
                    vals[i] = sd
        d[need] = np.where(vals < 0.99, vals, d[need])
        self.d = d
        gx, gy, gz = np.gradient(d, self.h)
        self.g = np.stack([gx, gy, gz], -1)

    def sample(self, P):
        idx = (P - self.lo) / self.h
        return _trilinear(self.d, idx)

    def grad(self, P):
        idx = (P - self.lo) / self.h
        g = np.stack([_trilinear(self.g[..., k], idx) for k in range(3)], -1)
        l = np.linalg.norm(g, axis=-1, keepdims=True); l[l < 1e-9] = 1
        return g / l


def _trilinear(G, idx):
    n = np.array(G.shape)
    idx = np.clip(idx, 0, n - 1.001)
    i0 = np.floor(idx).astype(int); f = idx - i0
    i1 = np.minimum(i0 + 1, n - 1)
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]; x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c00 = G[x0, y0, z0] * (1 - fx) + G[x1, y0, z0] * fx
    c10 = G[x0, y1, z0] * (1 - fx) + G[x1, y1, z0] * fx
    c01 = G[x0, y0, z1] * (1 - fx) + G[x1, y0, z1] * fx
    c11 = G[x0, y1, z1] * (1 - fx) + G[x1, y1, z1] * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz


# ------------------------------------------------------------------------------------------ scalp
def scalp_coords(P0):
    """Azimuth (0 front, 90 side, 180 back; unsigned) and elevation (deg) of base-space points about HEAD_C."""
    r = P0 - HEAD_C
    az = np.degrees(np.arctan2(np.abs(r[:, 0]), -r[:, 1]))
    el = np.degrees(np.arctan2(r[:, 2], np.hypot(r[:, 0], r[:, 1])))
    return az, el


HAIRLINE_AZ = [0, 20, 38, 52, 64, 74, 84, 100, 127, 150, 180]
HAIRLINE_EL = [37.5, 37.0, 34.0, 25.0, 11.0, 1.0, -5.0, -12.0, -36.0, -45.0, -48.0]


def scalp_depth(P0):
    """Degrees of elevation above the hairline (positive = hair-bearing scalp)."""
    az, el = scalp_coords(P0)
    d = el - np.interp(az, HAIRLINE_AZ, HAIRLINE_EL)
    ear = (np.abs(P0[:, 0]) > 0.0705) & (P0[:, 2] > 1.488) & (P0[:, 2] < 1.568) & (P0[:, 1] > -0.070) & (P0[:, 1] < 0.004)
    return np.where(ear, -30.0, d), az, el


def sample_roots(V, V0, F, n, rng, head_c, density_fn=None):
    """Area-weighted random points on scalp quads. Returns posed positions, base positions, posed normals."""
    d, az, el = scalp_depth(V0)
    fd = d[F].min(1)
    sel = np.where(fd > -1.0)[0]
    Fq = F[sel]
    a = V[Fq[:, 0]]; b = V[Fq[:, 1]]; c = V[Fq[:, 2]]; dd = V[Fq[:, 3]]
    area = 0.5 * (np.linalg.norm(np.cross(b - a, c - a), axis=1) + np.linalg.norm(np.cross(c - a, dd - a), axis=1))
    pick = rng.choice(len(sel), size=int(n * 1.6), p=area / area.sum())
    u = rng.random(len(pick)); v = rng.random(len(pick))
    def bil(X):
        q = X[Fq[pick]]
        return (q[:, 0] * ((1 - u) * (1 - v))[:, None] + q[:, 1] * (u * (1 - v))[:, None] + q[:, 2] * (u * v)[:, None] + q[:, 3] * ((1 - u) * v)[:, None])
    P = bil(V); P0 = bil(V0)
    Nn = np.cross(b - a, dd - a)[pick]; Nn /= np.linalg.norm(Nn, axis=1)[:, None]
    flip = np.sum(Nn * (P - head_c), 1) < 0
    Nn[flip] *= -1
    dep, az2, el2 = scalp_depth(P0)
    keep = dep > 0.0
    if density_fn is not None:
        keep &= rng.random(len(P)) < density_fn(P0, dep, az2, el2)
    idx = np.where(keep)[0][:n]
    return P[idx], P0[idx], Nn[idx]


# ------------------------------------------------------------------------------------------ lock integration
def comb_and_grow(roots, roots0, normals, R_head, sdf, rng, P):
    """Integrate lock centre lines. Returns (n, K, 3) points and per-lock info dict."""
    n = len(roots)
    r0 = roots0 - HEAD_C
    dep, az, el = scalp_depth(roots0)
    x_part = P['part_x'] + 0.0025 * np.sin(roots0[:, 1] * 70.0)
    side = np.where(roots0[:, 0] >= x_part, 1.0, -1.0)
    front = smoothstep(-roots0[:, 1], 0.02, 0.10)                 # 1 near the forehead
    back = smoothstep(roots0[:, 1], 0.00, 0.07)
    top = smoothstep(el, 35, 70)
    # ---- fringe (bangs): a band behind the frontal hairline
    tri = P['bang_depth'] * np.clip(1.0 - az / 60.0, 0, 1) ** 0.8          # fringe section: deep at the centre, vanishing at the temples
    bang_w = (1 - smoothstep(dep, tri * 0.75, tri + 0.01)) * (az < 60) * (roots0[:, 1] < -0.045)
    bang_w *= 0.25 + 0.75 * smoothstep(np.abs(roots0[:, 0] - x_part), 0.002, 0.010)   # a little forehead shows at the parting
    is_bang = bang_w > rng.random(n) * 0.75 + 0.05
    # ---- comb direction in base (head) space
    X = np.array([1.0, 0, 0]); Y = np.array([0, 1.0, 0]); Z = np.array([0, 0, 1.0])
    y_crown = P['crown_y']
    rx = roots0[:, 0] - P['part_x']; ry = roots0[:, 1] - y_crown
    radial = rx / np.sqrt(rx * rx + ry * ry + 1e-6)
    behind = smoothstep(roots0[:, 1], y_crown - 0.015, y_crown + 0.02)      # past the crown there is no parting
    lat = side * (1 - behind) + radial * behind
    sweep = np.where(side > 0, P['sweep_left'], P['sweep_right']) * (1 - behind) * (0.4 + 0.6 * front)
    comb = (lat[:, None] * X) * (0.50 + 0.60 * top)[:, None] + Y[None] * (behind * 0.9 + sweep)[:, None] - Z[None] * 0.55
    # her left side (the raised side): sweep the temple hair back over / behind the ear
    tuck = (side > 0) * smoothstep(az, 46, 64) * (1 - smoothstep(az, 100, 118)) * smoothstep(el, -8, 6) * (1 - smoothstep(el, 34, 48))
    comb = comb * (1 - tuck)[:, None] + (Y * 1.0 - Z * 0.22 + X * 0.10)[None] * tuck[:, None]
    bang_dir = (-Z * 0.75 - Y * 0.25)[None] + (side * (0.04 + 0.42 * smoothstep(az, 22, 50)) + rng.normal(0, 0.10, n))[:, None] * X[None]
    comb = np.where(is_bang[:, None], bang_dir, comb)
    comb = comb @ R_head.T                                         # to the posed frame
    comb -= normals * np.sum(comb * normals, 1)[:, None]            # tangent to the scalp
    comb /= np.linalg.norm(comb, axis=1)[:, None] + 1e-12
    near_part = (1 - smoothstep(np.abs(roots0[:, 0] - x_part), 0.004, 0.022)) * (1 - behind)
    crown = 1 - smoothstep(np.hypot(roots0[:, 0] - P['part_x'], roots0[:, 1] - y_crown), 0.012, 0.045)
    lift = np.where(is_bang, P['bang_lift'], P['root_lift'] * (1 - 0.75 * near_part) * (1 - 0.8 * crown))
    D = comb + normals * lift[:, None]; D /= np.linalg.norm(D, axis=1)[:, None]
    # ---- lengths
    L = P['length'] * (0.90 + 0.16 * rng.random(n))
    L *= 1.0 - 0.10 * smoothstep(el, 20, -40)                       # nape hair is a little shorter than the top layer
    frame = (~is_bang) * (1 - smoothstep(dep, 4, 16)) * (1 - smoothstep(az, 60, 80)) * (roots0[:, 1] < -0.03)   # face-framing layer
    L = np.where(frame > 0.5, P['frame_len'] * (0.75 + 0.55 * rng.random(n)), L)
    Lb = P['bang_len'] + 0.00175 * np.clip(dep, 0, 40) + P['bang_side_extra'] * smoothstep(az, 16, 50) ** 1.3 + rng.normal(0, 0.011, n)
    L = np.where(is_bang, np.clip(Lb, 0.05, 0.24), L)
    wisp = (side > 0) & (~is_bang) & (roots0[:, 1] < -0.004) & (roots0[:, 2] <= 1.582)          # in front of her exposed ear
    L = np.where(wisp, 0.022 + 0.045 * rng.random(n), L)
    # ---- layer offset: top-of-head hair lies over the hair growing lower down
    layer = np.clip((el + 48) / 138.0, 0, 1)
    off = 0.0030 + P['thickness'] * (0.25 + 0.75 * layer) * rng.random(n) ** 0.7
    off = np.where(is_bang, 0.0020 + 0.0045 * rng.random(n), off)
    # ---- her left side is styled back behind the ear: steer those locks through a way-point
    wp = P['ear_waypoint']
    tucked = (side > 0) & (roots0[:, 1] < 0.004) & (roots0[:, 2] > 1.582) & (el < 66) & ((~is_bang) | (az > 26))
    L = np.where(tucked & is_bang, np.maximum(L, 0.15), L)
    active = tucked.copy()
    neck_axis = P['neck_axis']
    down_face = (-Z * 0.94 + Y * 0.34) @ R_head.T
    wp_temple = P['temple_waypoint'][None] + np.stack([rng.normal(0, 0.003, n), rng.normal(0, 0.008, n), rng.normal(0, 0.008, n)], 1) @ R_head.T
    stage1 = tucked & (el > 30) & (az < 62)                                # top hair drapes over the temple before the ear
    wp = wp[None] + np.stack([rng.normal(0, 0.003, n), rng.normal(0, 0.008, n), np.abs(rng.normal(0, 0.008, n))], 1) @ R_head.T
    d_wp0 = np.linalg.norm(wp - roots, axis=1)
    # ---- integrate
    pts = np.zeros((n, K_PTS, 3)); pts[:, 0] = roots
    p = roots.copy()
    g = np.array([0, 0, -1.0])
    ds = (L / (K_PTS - 1))
    stiff = np.where(is_bang, P['bang_stiff'], P['stiff'])
    arc = np.zeros(n)
    for k in range(1, K_PTS):
        t = k / (K_PTS - 1)
        arc += ds
        rate = stiff * (0.35 + arc / 0.06)                             # gravity takes over a few cm from the root
        a = (1.0 - np.exp(-rate * ds / 0.0128))[:, None]
        D = D * (1 - a) + g[None] * a
        # fringe wisps curl gently back towards the face instead of hanging as straight sticks
        cb = (P['bang_curl'] * is_bang * ds / 0.002)[:, None]
        D = D * (1 - cb) + down_face[None] * cb
        if active.any():
            to1 = wp_temple - p; d1 = np.linalg.norm(to1, axis=1)
            stage1 &= (d1 > 0.014) & (arc < 0.12)
            to = np.where(stage1[:, None], to1, wp - p); dist = np.linalg.norm(to, axis=1)
            active &= stage1 | ((dist > 0.016) & (arc < d_wp0 * 1.5 + 0.03))
            gr = sdf.grad(p)
            to_t = to - gr * np.sum(to * gr, 1)[:, None]            # slide over the scalp, never through the skull
            to_t /= np.linalg.norm(to_t, axis=1)[:, None] + 1e-9
            b = (0.48 * active)[:, None]
            D = D * (1 - b) + to_t * b
        # long hair fans out a little over the shoulders and back instead of hanging as one rope
        out_xy = p - neck_axis[None]; out_xy[:, 2] = 0
        out_xy /= np.linalg.norm(out_xy, axis=1)[:, None] + 1e-9
        sp = (P['splay'] * smoothstep(arc, 0.10, 0.20) * (1 - smoothstep(arc, 0.30, 0.42)) * (off / 0.03))[:, None]
        D = D + out_xy * sp * (~is_bang)[:, None]
        # the raised (left) side falls behind the shoulder, the lowered (right) side in front of it
        yb = (P['fall_bias'] * side * smoothstep(arc, 0.10, 0.22) * (~is_bang))[:, None]
        D = D + np.array([0.0, 1.0, 0.0])[None] * yb
        # ...and the lowered side drifts outwards over that shoulder, leaving the collar visible
        xb = (P['fall_out'] * (side < 0) * smoothstep(arc, 0.08, 0.20) * (1 - smoothstep(arc, 0.30, 0.40)) * (~is_bang))[:, None]
        D = D + np.array([-1.0, 0.0, 0.0])[None] * xb
        D /= np.linalg.norm(D, axis=1)[:, None]
        q = p + D * ds[:, None]
        for _ in range(2):
            s = sdf.sample(q) - off
            m = s < 0
            if m.any():
                q[m] += sdf.grad(q[m]) * (-s[m])[:, None]
        # combed hair lies on the head: near the roots pull strands back down to their own layer
        adh = (1 - smoothstep(arc, P['adhesion'] * 0.6, P['adhesion'])) * np.where(is_bang, 0.35, 1.0)
        s = sdf.sample(q) - off
        m = (s > 0) & (adh > 0.01)
        if m.any():
            q[m] -= sdf.grad(q[m]) * (np.minimum(s[m], 0.02) * 0.75 * adh[m])[:, None]
        Dn = q - p; ln = np.linalg.norm(Dn, axis=1)[:, None]; ln[ln < 1e-9] = 1
        D = Dn / ln
        q = p + D * ds[:, None]
        p = q
        pts[:, k] = p
    info = dict(is_bang=is_bang, side=side, L=L, off=off, az=az, el=el, dep=dep, frame=frame > 0.5, tuck=tucked, near_part=near_part)
    return pts, info


def fringe_wisps(pts, info, roots0, rng, P):
    """Gather the fringe locks into a handful of wisps (see-through bangs): tips converge, gaps open between."""
    ib = np.where(info['is_bang'])[0]
    if len(ib) < 8:
        return pts
    x = roots0[ib, 0]
    nw = int(P['bang_wisps'])
    centres = np.sort(rng.uniform(x.min(), x.max(), nw) * 0.35 + np.linspace(x.min(), x.max(), nw) * 0.65)
    cid = np.argmin(np.abs(x[:, None] - centres[None]), 1)
    t = np.linspace(0, 1, pts.shape[1])[None, :, None]
    w = P['bang_gather'] * smoothstep(t, 0.15, 0.95)
    out = pts.copy()
    for c in range(nw):
        mem = ib[cid == c]
        if len(mem) < 2:
            continue
        mean = pts[mem].mean(0, keepdims=True)
        out[mem] = pts[mem] * (1 - w) + mean * w
    return out


def add_waves(pts, info, rng, P):
    """Loose S-waves that start below ear level; phase is shared per lock so locks stay coherent."""
    n, K, _ = pts.shape
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    s = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, 1)], 1)
    T = np.gradient(pts, axis=1); T /= np.linalg.norm(T, axis=2, keepdims=True) + 1e-12
    ref = np.array([0.33, 0.9, 0.2]); ref /= np.linalg.norm(ref)
    n1 = np.cross(T, ref[None, None]); n1 /= np.linalg.norm(n1, axis=2, keepdims=True) + 1e-12
    n2 = np.cross(T, n1)
    lam = P['wave_len'] * (0.90 + 0.2 * rng.random(n))[:, None]
    # neighbouring locks share their phase (loose waves, not crimps); a little jitter keeps it natural
    ph = (util.fbm3(pts[:, 0] * 9.0, 3, 21) * 4.0 * math.pi + rng.normal(0, 0.45, n))[:, None]
    amp = P['wave_amp'] * (0.6 + 0.8 * rng.random(n))[:, None]
    start = np.where(info['is_bang'], 0.035, P['wave_start'])[:, None]
    grow = smoothstep(s, start, start + 0.14)
    amp = np.where(info['is_bang'][:, None], amp * 0.22, amp)
    w1 = np.sin(2 * math.pi * s / lam + ph) * amp * grow
    w2 = np.cos(2 * math.pi * s / lam * 0.5 + ph * 1.7) * amp * 0.45 * grow
    return pts + n1 * w1[..., None] + n2 * w2[..., None]


def strands_from_locks(locks, info, n_per, rng, sdf, P, normals):
    """Clumped strands around each lock centre line with frizz and fly-aways. Returns (m, K, 3), lock ids."""
    n, K, _ = locks.shape
    per = np.where(info['is_bang'], np.maximum(2, int(n_per * P['bang_density'])), n_per)
    ids = np.repeat(np.arange(n), per)
    m = len(ids)
    base = locks[ids]
    t = np.linspace(0, 1, K)[None, :, None]
    # random offset in a disc roughly perpendicular to the lock
    T = np.gradient(locks, axis=1); T /= np.linalg.norm(T, axis=2, keepdims=True) + 1e-12
    v = rng.normal(0, 1, (m, 3))
    Tm = T[ids]
    v = v[:, None, :] - Tm * np.sum(v[:, None, :] * Tm, 2, keepdims=True)
    v /= np.linalg.norm(v, axis=2, keepdims=True) + 1e-12
    rad = np.sqrt(rng.random(m))[:, None, None]
    isb = info['is_bang'][ids][:, None, None]
    r_root = np.where(isb, P['bang_root_r'], P['root_r'])
    r_mid = np.where(isb, P['bang_mid_r'], P['mid_r'])
    r_tip = np.where(isb, P['bang_tip_r'], P['tip_r'])
    prof = r_root * (1 - t) ** 2 + r_mid * 4 * t * (1 - t) * 0.5 + r_tip * t ** 2
    offs = v * rad * prof
    vn = normals[ids][:, None, :]
    flat_off = offs - vn * np.sum(offs * vn, 2, keepdims=True)      # at the scalp the spread stays in the skin plane
    wroot = 1.0 - smoothstep(t, 0.0, 0.14)
    offs = offs * (1 - wroot) + flat_off * wroot
    out = base + offs
    # frizz: smooth per-strand noise growing towards the tip
    fz = util.fbm3(out * 38.0 + rng.random((m, 1, 3)) * 50.0, 3, 5) - 0.5
    fz2 = util.fbm3(out * 38.0 + rng.random((m, 1, 3)) * 50.0 + 17.0, 3, 9) - 0.5
    fz3 = util.fbm3(out * 38.0 + rng.random((m, 1, 3)) * 50.0 + 31.0, 3, 13) - 0.5
    amp = (P['frizz'] * (0.25 + 0.75 * t))
    fly = rng.random(m) < P['flyaway']
    amp = amp * np.where(fly, 7.0, 1.0)[:, None, None]
    out = out + np.stack([fz, fz2, fz3], -1)[..., :] * amp
    # shorten a few strands (natural uneven ends) by pulling points back along the strand
    cut = 1.0 - rng.random(m) ** 3 * np.where(info['is_bang'][ids], 0.50, 0.22)
    idxf = np.clip(np.linspace(0, K - 1, K)[None, :] * cut[:, None], 0, K - 1)
    i0 = np.floor(idxf).astype(int); f = (idxf - i0)[..., None]; i1 = np.minimum(i0 + 1, K - 1)
    rows = np.arange(m)[:, None]
    out = out[rows, i0] * (1 - f) + out[rows, i1] * f
    # keep everything outside the body / sweater
    flat = out.reshape(-1, 3)
    off = (info['off'][ids][:, None] * (0.55 + 0.45 * rng.random((m, 1)))).repeat(K, 1).reshape(-1)
    off = np.maximum(off * 0.8, 0.0012)
    for _ in range(2):
        s = sdf.sample(flat) - off
        mk = s < 0
        if mk.any():
            flat[mk] += sdf.grad(flat[mk]) * (-s[mk])[:, None]
    out = flat.reshape(m, K, 3)
    out[:, 0] = base[:, 0] + offs[:, 0]                              # roots stay on the scalp
    return out, ids, fly


def children_modifier(ob, amount, radius_mid, radius_tip, seed=1):
    """Geometry Nodes: duplicate every authored strand `amount` times with a small tapering offset."""
    ng = bpy.data.node_groups.new('Hair_Children', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    s_amount = ng.interface.new_socket('Children per strand', in_out='INPUT', socket_type='NodeSocketInt')
    s_amount.default_value = amount; s_amount.min_value = 1; s_amount.max_value = 64
    s_spread = ng.interface.new_socket('Spread', in_out='INPUT', socket_type='NodeSocketFloat')
    s_spread.default_value = radius_mid; s_spread.min_value = 0.0
    N = ng.nodes; Lk = ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    dup = N.new('GeometryNodeDuplicateElements'); dup.domain = 'SPLINE'
    Lk.new(gi.outputs['Geometry'], dup.inputs['Geometry']); Lk.new(gi.outputs['Children per strand'], dup.inputs['Amount'])
    # per-curve random vector, captured on the curve domain
    rnd = N.new('FunctionNodeRandomValue'); rnd.data_type = 'FLOAT_VECTOR'
    rnd.inputs['Min'].default_value = (-1, -1, -1); rnd.inputs['Max'].default_value = (1, 1, 1)
    rnd.inputs['Seed'].default_value = seed
    cap = N.new('GeometryNodeCaptureAttribute'); cap.domain = 'CURVE'
    cap.capture_items.new('VECTOR', 'offset')
    Lk.new(dup.outputs['Geometry'], cap.inputs['Geometry']); Lk.new(rnd.outputs['Value'], cap.inputs['offset'])
    sp = N.new('GeometryNodeSplineParameter')
    # profile: 0.15 at the root -> 1 in the middle -> tip ratio
    cr = N.new('ShaderNodeFloatCurve')
    cm = cr.mapping.curves[0]
    cm.points[0].location = (0.0, 0.32); cm.points[1].location = (1.0, max(0.02, radius_tip / max(radius_mid, 1e-9)))
    cm.points.new(0.45, 1.0)
    cr.mapping.update()
    Lk.new(sp.outputs['Factor'], cr.inputs['Value'])
    mul = N.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
    Lk.new(cr.outputs['Value'], mul.inputs[0]); Lk.new(gi.outputs['Spread'], mul.inputs[1])
    # keep the first copy of each strand exactly on the authored curve
    didx = dup.outputs['Duplicate Index']
    cmpn = N.new('FunctionNodeCompare'); cmpn.data_type = 'INT'; cmpn.operation = 'GREATER_THAN'
    Lk.new(didx, cmpn.inputs[2]); cmpn.inputs[3].default_value = 0
    capi = N.new('GeometryNodeCaptureAttribute'); capi.domain = 'CURVE'; capi.capture_items.new('BOOLEAN', 'moved')
    Lk.new(cap.outputs['Geometry'], capi.inputs['Geometry']); Lk.new(cmpn.outputs['Result'], capi.inputs['moved'])
    sc = N.new('ShaderNodeVectorMath'); sc.operation = 'SCALE'
    Lk.new(cap.outputs['offset'], sc.inputs[0]); Lk.new(mul.outputs[0], sc.inputs['Scale'])
    setp = N.new('GeometryNodeSetPosition')
    Lk.new(capi.outputs['Geometry'], setp.inputs['Geometry']); Lk.new(sc.outputs['Vector'], setp.inputs['Offset'])
    Lk.new(capi.outputs['moved'], setp.inputs['Selection'])
    Lk.new(setp.outputs['Geometry'], go.inputs['Geometry'])
    mod = ob.modifiers.new('Hair_Children', 'NODES'); mod.node_group = ng
    return mod


def ribbons(locks, info, col, mat, sdf, width=0.0036, every=3, name='Hair_Export_Proxy'):
    """Narrow ribbon mesh along every second authored strand.

    It is invisible to every Cycles ray (the renders use only the strand curves); it exists so that mesh-only
    exchange formats can still carry the hairstyle."""
    L = locks[::every]
    n, K, _ = L.shape
    idx = np.linspace(0, K - 1, 18).astype(int)
    L = L[:, idx]; K2 = L.shape[1]
    T = np.gradient(L, axis=1); T /= np.linalg.norm(T, axis=2, keepdims=True) + 1e-12
    flat = L.reshape(-1, 3)
    nrm = sdf.grad(flat).reshape(n, K2, 3)                        # away from the body
    side = np.cross(T, nrm); side /= np.linalg.norm(side, axis=2, keepdims=True) + 1e-12
    t = np.linspace(0, 1, K2)[None, :, None]
    w = width * (0.6 + 0.4 * smoothstep(t, 0.0, 0.2)) * (1 - 0.8 * smoothstep(t, 0.7, 1.0))
    centre = L
    A = centre - side * w * 0.5; B = centre + side * w * 0.5
    V = np.concatenate([A.reshape(-1, 3), B.reshape(-1, 3)])
    F = []; uv = []
    for i in range(n):
        for k in range(K2 - 1):
            a0 = i * K2 + k; a1 = a0 + 1; b0 = n * K2 + a0; b1 = b0 + 1
            F.append((a0, b0, b1, a1)); uv += [(0, k / (K2 - 1)), (1, k / (K2 - 1)), (1, (k + 1) / (K2 - 1)), (0, (k + 1) / (K2 - 1))]
    ob = util.mesh_object(name, V, F, col, uvs=uv)
    ob.data.materials.append(mat)
    for attr in ('visible_camera', 'visible_diffuse', 'visible_glossy', 'visible_transmission', 'visible_volume_scatter', 'visible_shadow'):
        setattr(ob, attr, False)
    ob.display_type = 'WIRE'
    return ob


def ribbon_material(color):
    m = bpy.data.materials.new('Hair_Underlayer'); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = 0.55
    p.inputs['Anisotropic'].default_value = 0.6
    p.inputs['Specular IOR Level'].default_value = 0.25
    uv = nt.nodes.new('ShaderNodeTexCoord')
    wv = nt.nodes.new('ShaderNodeTexNoise'); wv.inputs['Scale'].default_value = 6.0; wv.inputs['Detail'].default_value = 4.0
    mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (60.0, 0.6, 1.0)
    nt.links.new(uv.outputs['UV'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], wv.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.inputs['To Min'].default_value = 0.55; ramp.inputs['To Max'].default_value = 1.25
    nt.links.new(wv.outputs['Fac'], ramp.inputs['Value'])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'; mix.inputs['Factor'].default_value = 1.0
    mix.inputs['A'].default_value = (*color, 1); nt.links.new(ramp.outputs['Result'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], p.inputs['Base Color'])
    m.diffuse_color = (*color, 1)
    return m


DEFAULT = dict(
    part_x=0.004, crown_y=0.030, sweep_left=0.85, sweep_right=-0.10,
    length=0.50, frame_len=0.22, thickness=0.036, stiff=0.26, root_lift=0.34,
    bang_depth=30.0, bang_len=0.088, bang_side_extra=0.060, bang_lift=0.30, bang_stiff=0.045, bang_density=0.6, n_bang_locks=175, bang_curl=0.070,
    bang_wisps=13, bang_gather=0.80,
    splay=0.10, adhesion=0.11, fall_bias=0.035, fall_out=0.10,
    wave_len=0.140, wave_amp=0.0125, wave_start=0.10,
    root_r=0.0048, mid_r=0.0105, tip_r=0.0032, bang_root_r=0.0045, bang_mid_r=0.0030, bang_tip_r=0.0012,
    frizz=0.0022, flyaway=0.035,
    n_locks=1500, per_lock=7, children=9,
)


def build(ctx):
    body = ctx['body']; sk = ctx['sk']; M = ctx['M']; log = ctx['log']; rng = ctx['rng']; q = ctx['q']
    P = dict(DEFAULT, **ctx.get('hair_params', {}))
    col = util.collection('Hair', ctx['col_char'])
    # ---- posed surfaces
    V_posed = ctx['V_posed']
    Vp, Fp, _ = util.subdivided(V_posed, body.F, 2)
    V0s, _, _ = util.subdivided(body.V0, body.F, 2)
    bvhs = [util.bvh_from(*util.subdivided(V_posed, body.F, 1)[:2])]
    if ctx.get('ob_sweater') is not None:
        sv, sf = util.evaluated_arrays(ctx['ob_sweater'])
        bvhs.append(util.bvh_from(sv, sf))
    sdf = SDF((-0.40, -0.46, 0.92), (0.40, 0.28, 1.84), 0.006 if q < 1 else 0.005)
    sdf.build(bvhs, signed_within=[1e9, 0.03][:len(bvhs)])
    log('hair collision field %s' % (tuple(sdf.n),))
    R_head = M['head'][:3, :3]
    P['ear_waypoint'] = sk.apply(M['head'], np.array([0.0800, 0.0100, 1.5760]))     # just above/behind her left ear
    P['temple_waypoint'] = sk.apply(M['head'], np.array([0.0700, -0.0620, 1.5900]))  # left temple, in front of the hairline
    P['neck_axis'] = sk.apply(M['neck'], np.array([0.0, -0.01, 1.42]))
    # ---- locks
    n_locks = int(P['n_locks'] * (0.6 if q < 1 else 1.0))
    head_c = sk.apply(M['head'], HEAD_C)
    assert sdf.sample(np.array([[0.0, -0.45, 1.6]]))[0] > 0 > sdf.sample(head_c[None])[0], 'collision field sign'
    roots, roots0, normals = sample_roots(Vp, V0s, Fp, n_locks, rng, head_c)
    nb = int(P['n_bang_locks'] * (0.6 if q < 1 else 1.0))
    fringe = lambda P0, dep, az, el: ((dep < P['bang_depth'] * np.clip(1.0 - az / 60.0, 0, 1) ** 0.8) & (az < 60) & (P0[:, 1] < -0.045)).astype(float)
    rb_, rb0_, nb_ = sample_roots(Vp, V0s, Fp, nb * 12, rng, head_c, density_fn=fringe)
    roots = np.concatenate([roots, rb_[:nb]]); roots0 = np.concatenate([roots0, rb0_[:nb]]); normals = np.concatenate([normals, nb_[:nb]])
    locks, info = comb_and_grow(roots, roots0, normals, R_head, sdf, rng, P)
    locks = fringe_wisps(locks, info, roots0, rng, P)
    locks = add_waves(locks, info, rng, P)
    strands, ids, fly = strands_from_locks(locks, info, P['per_lock'], rng, sdf, P, normals)
    m = len(strands)
    t = np.linspace(0, 1, K_PTS)[None, :]
    rad = (0.000062 * (1 - 0.55 * t ** 2)) * np.ones((m, 1))
    rad = np.where(info['is_bang'][ids][:, None], rad * 0.85, rad)
    mat, hnode = hair_material('Hair_Chestnut', (0.20, 0.10, 0.055), roughness=0.22, radial=0.32, coat=0.15,
                               melanin=0.80, redness=0.58)
    hnode.inputs['Tint'].default_value = (1.0, 0.86, 0.78, 1)
    hnode.inputs['Random Color'].default_value = 0.06
    hnode.inputs['Random Roughness'].default_value = 0.08
    ob = curves_object('Hair', strands, rad, col, mat)
    children = max(2, int(P['children'] * (0.5 if q < 1 else 1.0)))
    children_modifier(ob, children, 0.0042, 0.0012)
    ob.data.attributes.new('lock_id', 'INT', 'CURVE').data.foreach_set('value', ids.astype(np.int32))
    log('hair: %d locks, %d authored strands x %d children' % (len(locks), m, children))
    rb = ribbons(strands, None, col, ribbon_material((0.085, 0.043, 0.024)), sdf)
    ctx['hair_points'] = strands.reshape(-1, 3)
    ctx['hair_info'] = dict(locks=int(len(locks)), authored_strands=int(m), children_per_strand=int(children), points_per_strand=K_PTS,
                            rendered_strands=int(m * children), fringe_locks=int(info['is_bang'].sum()))
    return ob
