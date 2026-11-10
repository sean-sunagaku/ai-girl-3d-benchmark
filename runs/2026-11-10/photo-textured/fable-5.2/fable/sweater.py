"""Chunky raglan knit sweater.

The garment is built as one structured quad mesh in the rest pose (yoke from the collar outwards, body tube,
two sleeve tubes that share the under-arm stitches), given stitch-unit UVs per knitted panel, then carried
into the selfie pose with the body's own skin weights and settled against the posed body.
"""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from . import util, knit
from .base import smoothstep

STITCH_H = 0.0050          # row height (m)
N_F = 62; N_B = 62; N_U = 12; N_SL = 62
N_S = N_SL - N_U
N_Y = N_F + N_B + 2 * N_S
N_BODY = N_F + N_B + 2 * N_U
C_BODY = 160; C_SLEEVE = 68; C_COLLAR = 96
ROWS_YOKE = 24; ROWS_BODY = 60; ROWS_SLEEVE = 78
ROWS_HEM = 10; ROWS_CUFF = 11; ROWS_COLLAR = 7


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def superellipse(theta, ax, ay, n=2.5):
    return (np.abs(np.sin(theta) / ax) ** n + np.abs(np.cos(theta) / ay) ** n) ** (-1.0 / n)


def torso_axis(z):
    return np.array([0.0, np.interp(z, [1.0, 1.2, 1.4], [-0.022, -0.030, -0.030]), z])


def box_radius(theta, z):
    ax = np.interp(z, [0.86, 0.93, 1.10, 1.27], [0.164, 0.176, 0.180, 0.184])
    ay = np.interp(z, [0.86, 0.93, 1.10, 1.27], [0.126, 0.134, 0.140, 0.142])
    return superellipse(theta, ax, ay)


def sleeve_radius(s, total):
    """Loose tube that blouses a little before the cuff."""
    t = s / total
    return float(np.interp(t, [0.0, 0.25, 0.55, 0.80, 0.865, 0.90, 1.0], [0.078, 0.073, 0.068, 0.068, 0.063, 0.041, 0.038]))


class Garment:
    def __init__(self):
        self.P = []            # rest positions
        self.faces = []        # (v0, v1, v2, v3)
        self.uv = []           # per face: 4 (u, v)
        self.mat = []          # per face material index
        self.tag = []          # per vertex: region name

    def add(self, p, tag):
        self.P.append(np.asarray(p, float)); self.tag.append(tag)
        return len(self.P) - 1


def _sleeve_frames(sk, side):
    L = sk.landmarks
    S = np.array(L['joint-%s-shoulder' % side]); E = np.array(L['joint-%s-elbow' % side]); Wr = np.array(L['joint-%s-hand' % side])
    au = unit(E - S); af = unit(Wr - E)
    A0 = S + au * 0.045
    end = Wr + af * 0.062
    key = np.array([A0, E, Wr, end])
    seg = np.linalg.norm(np.diff(key, axis=0), axis=1); cs = np.concatenate([[0], np.cumsum(seg)])
    total = cs[-1]
    s = np.linspace(0, total, ROWS_SLEEVE + 1)
    pts = np.stack([np.interp(s, cs, key[:, k]) for k in range(3)], 1)
    for _ in range(6):                                   # round the elbow
        pts[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
    T = np.gradient(pts, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]
    sx = 1.0 if side == 'l' else -1.0
    e1 = unit(np.array([-au[2] * sx, 0.0, au[0] * sx]) * 1.0)       # in the frontal plane, pointing to the arm pit
    if e1[2] > 0:
        e1 = -e1
    e1 = unit(e1 - T[0] * np.dot(e1, T[0]))
    E1 = [e1]
    for k in range(1, len(pts)):                         # parallel transport
        v = E1[-1] - T[k] * np.dot(E1[-1], T[k])
        E1.append(unit(v))
    E1 = np.array(E1)
    E2 = np.cross(T, E1)
    if side == 'r':
        E2 = -E2                                        # keep "phi grows toward the back" on both sides
    return pts, T, E1, E2, s, total


def build_rest(body, sk, bvh):
    g = Garment()
    cast = lambda o, d, dist=0.4: bvh.ray_cast(Vector(o), Vector(d), dist)

    # ------------------------------------------------------------------ sleeves: analytic rings
    sleeves = {}
    for side in ('l', 'r'):
        pts, T, E1, E2, s, total = _sleeve_frames(sk, side)
        rings = np.zeros((ROWS_SLEEVE + 1, N_SL, 3))
        for k in range(ROWS_SLEEVE + 1):
            r0 = sleeve_radius(s[k], total)
            bell = 1.0 + 0.045 * math.sin(2 * math.pi * s[k] / 0.058 + 0.7) * smoothstep(s[k], 0.03, 0.08) * (1 - smoothstep(s[k], total - 0.11, total - 0.07))
            for i in range(N_SL):
                phi = 2 * math.pi * i / N_SL
                d = math.cos(phi) * E1[k] + math.sin(phi) * E2[k]
                r = r0 * bell
                hit = cast(pts[k], d, 0.2)
                if hit[0] is not None and k > 3:
                    r = max(r, min(hit[3] + 0.012, r0 + 0.02))
                rings[k, i] = pts[k] + d * r
        sleeves[side] = dict(rings=rings, axis=pts, s=s, total=total)

    h = N_U // 2
    strip = {side: [sleeves[side]['rings'][0][(u - h) % N_SL] for u in range(N_U + 1)] for side in ('l', 'r')}
    PLf, PLb = strip['l'][0], strip['l'][N_U]
    z_split = float(PLf[2])

    # ------------------------------------------------------------------ yoke bottom ring
    c0 = torso_axis(z_split)
    th_f = math.atan2(PLf[0], -(PLf[1] - c0[1])); rad_f = math.hypot(PLf[0], PLf[1] - c0[1])
    th_b = math.atan2(PLb[0], -(PLb[1] - c0[1])); rad_b = math.hypot(PLb[0], PLb[1] - c0[1])

    def arc(th0, th1, r_end, n, front):
        out = []
        for k in range(n + 1):
            tau = -1 + 2 * k / n
            th = th0 + (th1 - th0) * k / n
            re = float(box_radius(th, z_split))
            hit = cast(c0, (math.sin(th), -math.cos(th), 0.0))
            if hit[0] is not None and abs(hit[0][0]) < 0.15:
                re = max(re, hit[3] + 0.016)
            w = abs(tau) ** 4
            rr = re * (1 - w) + r_end * w
            out.append(np.array([c0[0] + rr * math.sin(th), c0[1] - rr * math.cos(th), z_split]))
        return out
    F_arc = arc(-th_f, th_f, rad_f, N_F, True)                       # P_Rf ... P_Lf
    B_arc = arc(th_b, 2 * math.pi - th_b, rad_b, N_B, False)         # P_Lb ... P_Rb
    ring0L = sleeves['l']['rings'][0]; ring0R = sleeves['r']['rings'][0]
    yk = np.zeros((ROWS_YOKE + 1, N_Y), dtype=int)
    bottom = []
    for c in range(N_Y):
        if c <= N_F:
            p = F_arc[c]
        elif c <= N_F + N_S:
            i = h + (N_F + N_S - c); p = ring0L[i]
        elif c <= N_F + N_S + N_B:
            p = B_arc[c - (N_F + N_S)]
        else:
            m = c - (N_Y - N_S); i = h + m; p = ring0R[i]
        bottom.append(np.asarray(p))
    # pin the junction vertices exactly
    bottom[0] = np.asarray(strip['r'][0]); bottom[N_F] = np.asarray(PLf)
    bottom[N_F + N_S] = np.asarray(PLb); bottom[N_F + N_S + N_B] = np.asarray(strip['r'][N_U])

    # ------------------------------------------------------------------ neck ring
    cn = np.array([0.0, -0.012, 1.384]); tilt = math.radians(22.0)
    neck = []
    rads = []
    for c in range(N_Y):
        psi = 2 * math.pi * (c - N_F / 2) / N_Y
        d = np.array([math.sin(psi), -math.cos(psi) * math.cos(tilt), -math.cos(psi) * math.sin(tilt)])
        hit = cast(cn, d, 0.2)
        r = (hit[3] if hit[0] is not None else 0.060) + 0.0075
        rads.append(min(max(r, 0.056), 0.084))
    rads = np.array(rads)
    for _ in range(20):
        rads = 0.5 * rads + 0.25 * (np.roll(rads, 1) + np.roll(rads, -1))
    for c in range(N_Y):
        psi = 2 * math.pi * (c - N_F / 2) / N_Y
        d = np.array([math.sin(psi), -math.cos(psi) * math.cos(tilt), -math.cos(psi) * math.sin(tilt)])
        neck.append(cn + d * rads[c])

    for r in range(ROWS_YOKE + 1):
        t = r / ROWS_YOKE
        for c in range(N_Y):
            p = bottom[c] * (1 - t) + neck[c] * t
            p = p + np.array([0, 0, 0.022 * math.sin(math.pi * t)])          # start above the shoulders, then settle
            yk[r, c] = g.add(p, 'yoke_fixed' if r in (0, ROWS_YOKE) else 'yoke')
    for c in range(N_Y):
        g.P[yk[0, c]] = bottom[c].copy(); g.P[yk[ROWS_YOKE, c]] = neck[c].copy()

    # ------------------------------------------------------------------ under-arm strips
    us = {}
    for side in ('l', 'r'):
        ids = []
        for u in range(N_U + 1):
            if u == 0:
                ids.append(yk[0, N_F] if side == 'l' else yk[0, 0])
            elif u == N_U:
                ids.append(yk[0, N_F + N_S] if side == 'l' else yk[0, N_F + N_S + N_B])
            else:
                ids.append(g.add(strip[side][u], 'underarm'))
        us[side] = ids

    # ------------------------------------------------------------------ body tube
    bd = np.zeros((ROWS_BODY + 1, N_BODY), dtype=int)
    row0 = [yk[0, c] for c in range(0, N_F + 1)] + us['l'][1:N_U] + [yk[0, c] for c in range(N_F + N_S, N_F + N_S + N_B + 1)] + us['r'][N_U - 1:0:-1]
    assert len(row0) == N_BODY
    bd[0] = row0
    top = np.array([g.P[i] for i in row0])
    th_j = np.arctan2(top[:, 0], -(top[:, 1] - c0[1]))
    rho0 = np.hypot(top[:, 0], top[:, 1] - c0[1])
    z_hem = 0.885
    dz = (z_split - z_hem) / ROWS_BODY
    excess = np.zeros(N_BODY)            # how far the body pushes the knit beyond the boxy profile, carried below the crop
    for r in range(1, ROWS_BODY + 1):
        zr = z_split - r * dz
        w = smoothstep(r, 0, 9)
        cz = torso_axis(zr)
        hem_t = smoothstep(r, ROWS_BODY - ROWS_HEM - 3, ROWS_BODY - ROWS_HEM + 1)
        for j in range(N_BODY):
            box = float(box_radius(th_j[j], zr))
            hit = cast(cz, (math.sin(th_j[j]), -math.cos(th_j[j]), 0.0))
            if hit[0] is not None and abs(hit[0][0]) < 0.155 and zr > 1.125:
                excess[j] = max(0.0, hit[3] + 0.014 - box)
                re = box + excess[j]
            else:
                re = box + excess[j] * float(smoothstep(zr, 1.125 - 0.16, 1.125))
            re *= (1.0 - 0.085 * hem_t)
            rr = rho0[j] * (1 - w) + re * w
            z = (top[j, 2] - r * dz) * (1 - w) + zr * w
            bd[r, j] = g.add((cz[0] + rr * math.sin(th_j[j]), cz[1] - rr * math.cos(th_j[j]), z), 'body' if r > 6 else 'body_top')
    # rolled hem edge (gives the band a real thickness)
    bd_extra = []
    for k, (dr, dzz) in enumerate(((-0.004, -0.002), (-0.007, 0.004))):
        row = []
        for j in range(N_BODY):
            p = g.P[bd[ROWS_BODY, j]]; cz = torso_axis(p[2])
            d = unit(np.array([p[0] - cz[0], p[1] - cz[1], 0.0]))
            row.append(g.add(p + d * dr + np.array([0, 0, dzz]), 'body'))
        bd_extra.append(row)

    # ------------------------------------------------------------------ sleeve tubes
    sl = {}
    for side in ('l', 'r'):
        rings = sleeves[side]['rings']
        ids = np.zeros((ROWS_SLEEVE + 1, N_SL), dtype=int)
        for i in range(N_SL):
            if i <= h:
                ids[0, i] = us[side][i + h]
            elif i <= N_SL - h:
                m = i - h
                ids[0, i] = yk[0, N_F + N_S - m] if side == 'l' else yk[0, (N_Y - N_S + m) % N_Y]
            else:
                ids[0, i] = us[side][i - (N_SL - h)]
        for k in range(1, ROWS_SLEEVE + 1):
            for i in range(N_SL):
                ids[k, i] = g.add(rings[k, i], 'sleeve_' + side if k > 5 else 'sleeve_top_' + side)
        extra = []
        ax = sleeves[side]['axis']
        for (dr, da) in ((-0.004, 0.002), (-0.007, -0.004)):
            row = []
            for i in range(N_SL):
                p = g.P[ids[ROWS_SLEEVE, i]]; d = unit(p - ax[-1]); t = unit(ax[-1] - ax[-2])
                row.append(g.add(p + d * dr + t * da, 'sleeve_' + side))
            extra.append(row)
        sl[side] = dict(ids=ids, extra=extra)

    # ------------------------------------------------------------------ faces
    def quad(a, b, c, d):
        g.faces.append((int(a), int(b), int(c), int(d)))
    face_info = []          # (kind, r, c, side)
    for r in range(ROWS_YOKE):
        for c in range(N_Y):
            c2 = (c + 1) % N_Y
            quad(yk[r, c], yk[r, c2], yk[r + 1, c2], yk[r + 1, c]); face_info.append(('yoke', r, c, None))
    for r in range(ROWS_BODY):
        for j in range(N_BODY):
            j2 = (j + 1) % N_BODY
            quad(bd[r, j], bd[r + 1, j], bd[r + 1, j2], bd[r, j2]); face_info.append(('body', r, j, None))
    rows = [list(bd[ROWS_BODY])] + bd_extra
    for e in range(2):
        for j in range(N_BODY):
            j2 = (j + 1) % N_BODY
            quad(rows[e][j], rows[e + 1][j], rows[e + 1][j2], rows[e][j2]); face_info.append(('body', ROWS_BODY + e, j, None))
    for side in ('l', 'r'):
        ids = sl[side]['ids']
        for k in range(ROWS_SLEEVE):
            for i in range(N_SL):
                i2 = (i + 1) % N_SL
                quad(ids[k, i], ids[k, i2], ids[k + 1, i2], ids[k + 1, i]); face_info.append(('sleeve', k, i, side))
        rows = [list(ids[ROWS_SLEEVE])] + sl[side]['extra']
        for e in range(2):
            for i in range(N_SL):
                i2 = (i + 1) % N_SL
                quad(rows[e][i], rows[e][i2], rows[e + 1][i2], rows[e + 1][i]); face_info.append(('sleeve', ROWS_SLEEVE + e, i, side))
    g.grid = dict(yk=yk, bd=bd, bd_extra=bd_extra, sl=sl, us=us, face_info=face_info, sleeves=sleeves, z_split=z_split)
    return g


COLLAR_BAND_ROWS = 6


def add_collar(g):
    """The neck band is the last COLLAR_BAND_ROWS of the yoke (ribbed, slightly raised); here it gets a rolled rim."""
    yk = g.grid['yk']
    P = np.array(g.P)
    centre = P[yk[ROWS_YOKE]].mean(0)
    # raise the band a little above the body knit so the join reads as a sewn-on rib
    for r in range(ROWS_YOKE - COLLAR_BAND_ROWS, ROWS_YOKE + 1):
        w = smoothstep(r, ROWS_YOKE - COLLAR_BAND_ROWS - 0.5, ROWS_YOKE - COLLAR_BAND_ROWS + 1.0)
        for c in range(N_Y):
            a_ = P[yk[r, (c + 1) % N_Y]] - P[yk[r, c - 1]]
            b_ = P[yk[min(r + 1, ROWS_YOKE), c]] - P[yk[r - 1, c]]
            nrm = np.cross(a_, b_); nrm /= np.linalg.norm(nrm) + 1e-12
            if np.dot(nrm, P[yk[r, c]] - centre) < 0:
                nrm = -nrm
            g.P[yk[r, c]] = P[yk[r, c]] + nrm * 0.0016 * w
    P = np.array(g.P)
    ring = P[yk[ROWS_YOKE]]
    rows = []
    for (inward, up) in ((0.0026, 0.0022), (0.0058, 0.0012), (0.0074, -0.0026)):
        row = []
        for c in range(N_Y):
            p = ring[c]
            d_in = centre - p; d_in[2] = 0; d_in = unit(d_in)
            row.append(g.add(p + d_in * inward + np.array([0, 0, up]), 'collar'))
        rows.append(row)
    allrows = [list(yk[ROWS_YOKE])] + rows
    for e in range(len(rows)):
        for c in range(N_Y):
            c2 = (c + 1) % N_Y
            g.faces.append((int(allrows[e][c]), int(allrows[e][c2]), int(allrows[e + 1][c2]), int(allrows[e + 1][c])))
            g.grid['face_info'].append(('collar', e, c, None))
    g.grid['collar_rows'] = allrows


def relax(P, E, free, bvh, gap, iters=30, lam=0.55, z_floor=1.125):
    """Membrane smoothing of the free vertices with a minimum clearance from the body surface."""
    P = P.copy()
    n = len(P)
    deg = np.zeros(n); np.add.at(deg, E[:, 0], 1); np.add.at(deg, E[:, 1], 1)
    idx = np.where(free)[0]
    for it in range(iters):
        acc = np.zeros_like(P)
        np.add.at(acc, E[:, 0], P[E[:, 1]]); np.add.at(acc, E[:, 1], P[E[:, 0]])
        avg = acc / np.maximum(deg, 1)[:, None]
        P[idx] += lam * (avg[idx] - P[idx])
        if bvh is not None:
            for i in idx:
                hit = bvh.find_nearest(Vector(P[i]))
                if hit[0] is None:
                    continue
                loc = np.array(hit[0]); nrm = np.array(hit[1])
                d = float(np.dot(P[i] - loc, nrm))
                if d < gap and hit[3] < 0.08 and loc[2] > z_floor:
                    P[i] = loc + nrm * gap + (P[i] - loc - nrm * d) * 1.0
    return P


def edges_of(faces, n):
    s = set()
    for f in faces:
        for k in range(4):
            a, b = f[k], f[(k + 1) % 4]
            s.add((min(a, b), max(a, b)))
    return np.array(sorted(s))


def compute_uvs(g, P):
    """Per-face stitch-unit UVs for every knitted panel (see module docstring)."""
    G = g.grid; yk = G['yk']; bd = G['bd']; sl = G['sl']
    L = lambda a, b: float(np.linalg.norm(P[a] - P[b]))
    # body rows
    body_rows = [list(bd[r]) for r in range(ROWS_BODY + 1)] + G['bd_extra']
    Ub = []
    for row in body_rows:
        seg = [L(row[j], row[(j + 1) % N_BODY]) for j in range(N_BODY)]
        cum = np.concatenate([[0], np.cumsum(seg)])
        Ub.append(C_BODY * cum / cum[-1])
    jc = N_F // 2
    Vb = [0.0]
    for r in range(1, len(body_rows)):
        Vb.append(Vb[-1] - L(body_rows[r][jc], body_rows[r - 1][jc]) / STITCH_H)
    w_body = sum(L(body_rows[0][j], body_rows[0][(j + 1) % N_BODY]) for j in range(N_BODY)) / C_BODY
    # sleeves
    Us = {}; Vs = {}; w_sl = {}
    for side in ('l', 'r'):
        rows = [list(sl[side]['ids'][k]) for k in range(ROWS_SLEEVE + 1)] + sl[side]['extra']
        U = []
        for row in rows:
            seg = [L(row[i], row[(i + 1) % N_SL]) for i in range(N_SL)]
            cum = np.concatenate([[0], np.cumsum(seg)])
            U.append(C_SLEEVE * cum / cum[-1])
        it = N_SL // 2
        V = [0.0]
        for k in range(1, len(rows)):
            V.append(V[-1] - L(rows[k][it], rows[k - 1][it]) / STITCH_H)
        Us[side] = U; Vs[side] = V
        w_sl[side] = sum(L(rows[0][i], rows[0][(i + 1) % N_SL]) for i in range(N_SL)) / C_SLEEVE
    # yoke panels: (start column, end column, centre column, U at the centre, stitch width, sign)
    h = N_U // 2
    jB = N_F + N_U + N_B // 2
    panels = [
        (0, N_F, N_F // 2, Ub[0][N_F // 2], w_body, 1.0),
        (N_F, N_F + N_S, N_F + N_S // 2, Us['l'][0][N_SL // 2], w_sl['l'], -1.0),
        (N_F + N_S, N_F + N_S + N_B, N_F + N_S + N_B // 2, Ub[0][jB], w_body, 1.0),
        (N_F + N_S + N_B, N_Y, N_Y - N_S // 2, Us['r'][0][N_SL // 2], w_sl['r'], 1.0),
    ]
    Uy = {}; Vy = {}
    for pi, (ca, cb, cc, u0, w, sg) in enumerate(panels):
        Vc = [0.0]
        for r in range(1, ROWS_YOKE + 1):
            Vc.append(Vc[-1] + L(yk[r, cc % N_Y], yk[r - 1, cc % N_Y]) / STITCH_H)
        for r in range(ROWS_YOKE + 1):
            cols = list(range(ca, cb + 1))
            seg = [L(yk[r, cols[k] % N_Y], yk[r, cols[k + 1] % N_Y]) for k in range(len(cols) - 1)]
            cum = np.concatenate([[0], np.cumsum(seg)])
            cum = cum - cum[cc - ca]
            for k, c in enumerate(cols):
                Uy[(pi, r, c)] = u0 + sg * cum[k] / w
                Vy[(pi, r, c)] = Vc[r]
    def panel_of(c):
        for pi, (ca, cb, *_r) in enumerate(panels):
            if ca <= c < cb:
                return pi
        return 3
    # collar band: the last yoke rows followed by the rim rows
    col_rows = [list(yk[r]) for r in range(ROWS_YOKE - COLLAR_BAND_ROWS, ROWS_YOKE)] + G['collar_rows']
    Uc = []
    for row in col_rows:
        seg = [L(row[c], row[(c + 1) % N_Y]) for c in range(N_Y)]
        cum = np.concatenate([[0], np.cumsum(seg)])
        Uc.append(C_COLLAR * cum / cum[-1])
    Vc_ = [0.0]
    for e in range(1, len(col_rows)):
        Vc_.append(Vc_[-1] + L(col_rows[e][0], col_rows[e - 1][0]) / STITCH_H)
    uvs = []; mats = []
    for (kind, r, c, side), f in zip(G['face_info'], g.faces):
        if kind == 'yoke' and r >= ROWS_YOKE - COLLAR_BAND_ROWS:
            e = r - (ROWS_YOKE - COLLAR_BAND_ROWS)
            uvs.append([(Uc[e][c], Vc_[e]), (Uc[e][c + 1], Vc_[e]), (Uc[e + 1][c + 1], Vc_[e + 1]), (Uc[e + 1][c], Vc_[e + 1])])
            mats.append(1)
        elif kind == 'yoke':
            pi = panel_of(c)
            uvs.append([(Uy[(pi, r, c)], Vy[(pi, r, c)]), (Uy[(pi, r, c + 1)], Vy[(pi, r, c + 1)]),
                        (Uy[(pi, r + 1, c + 1)], Vy[(pi, r + 1, c + 1)]), (Uy[(pi, r + 1, c)], Vy[(pi, r + 1, c)])])
            mats.append(0)
        elif kind == 'body':
            uvs.append([(Ub[r][c], Vb[r]), (Ub[r + 1][c], Vb[r + 1]), (Ub[r + 1][c + 1], Vb[r + 1]), (Ub[r][c + 1], Vb[r])])
            mats.append(1 if r >= ROWS_BODY - ROWS_HEM else 0)
        elif kind == 'sleeve':
            U = Us[side]; V = Vs[side]
            uvs.append([(U[r][c], V[r]), (U[r][c + 1], V[r]), (U[r + 1][c + 1], V[r + 1]), (U[r + 1][c], V[r + 1])])
            mats.append(1 if r >= ROWS_SLEEVE - ROWS_CUFF else 0)
        else:
            e = r + COLLAR_BAND_ROWS
            uvs.append([(Uc[e][c], Vc_[e]), (Uc[e][c + 1], Vc_[e]), (Uc[e + 1][c + 1], Vc_[e + 1]), (Uc[e + 1][c], Vc_[e + 1])])
            mats.append(1)
    return uvs, mats


def transfer_weights(g, body, sk, W):
    """Nearest-surface skin weights from the body, restricted to torso/arm bones and smoothed over the knit."""
    bvh = util.bvh_from(body.V, body.F)
    P = np.array(g.P)
    n = len(P)
    Wg = np.zeros((n, W.shape[1]))
    F = body.F
    for i in range(n):
        loc, nrm, fi, dist = bvh.find_nearest(Vector(P[i]))
        f = F[fi]
        d = np.array([np.linalg.norm(body.V[v] - np.array(loc)) for v in f]) + 1e-5
        w = 1.0 / d; w /= w.sum()
        for v, ww in zip(f, w):
            Wg[i] += ww * W[v]
    idx = sk.index
    for side in ('l', 'r'):
        fa = idx['forearm.' + side]
        for name, bi in idx.items():
            if name.endswith('.' + side) and (name.startswith('hand') or name.startswith('f')) and not name.startswith('forearm'):
                Wg[:, fa] += Wg[:, bi]; Wg[:, bi] = 0
    ch = idx['chest']
    for name in ('neck', 'head'):
        Wg[:, ch] += Wg[:, idx[name]]; Wg[:, idx[name]] = 0
    # sleeves must not stick to the torso and vice versa
    tags = g.tag
    for i, t in enumerate(tags):
        if t.startswith('sleeve_') and not t.startswith('sleeve_top'):
            side = t[-1]
            keep = [idx['upperarm.' + side], idx['forearm.' + side], idx['clav.' + side]]
            mask = np.zeros(W.shape[1], bool); mask[keep] = True
            other = Wg[i, ~mask].sum()
            Wg[i, ~mask] = 0
            if Wg[i].sum() < 1e-6:
                Wg[i, idx['upperarm.' + side]] = 1.0
    Wg /= Wg.sum(1)[:, None]
    E = edges_of(g.faces, n)
    deg = np.zeros(n); np.add.at(deg, E[:, 0], 1); np.add.at(deg, E[:, 1], 1)
    for _ in range(14):
        acc = np.zeros_like(Wg)
        np.add.at(acc, E[:, 0], Wg[E[:, 1]]); np.add.at(acc, E[:, 1], Wg[E[:, 0]])
        Wg = 0.5 * Wg + 0.5 * acc / np.maximum(deg, 1)[:, None]
    Wg /= Wg.sum(1)[:, None]
    return Wg


def build(ctx):
    body = ctx['body']; sk = ctx['sk']; M = ctx['M']; W = ctx['W']; log = ctx['log']
    col = util.collection('Sweater', ctx['col_char'])
    rest_V, rest_F, _ = util.subdivided(body.V, body.F, 1)
    bvh_rest = util.bvh_from(rest_V, rest_F)
    g = build_rest(body, sk, bvh_rest)
    P = np.array(g.P)
    n = len(P)
    E = edges_of(g.faces, n)
    tags = np.array(g.tag)
    free = np.isin(tags, ['yoke', 'body_top', 'sleeve_top_l', 'sleeve_top_r', 'underarm'])
    P = relax(P, E, free, bvh_rest, 0.015, iters=26)
    g.P = [p for p in P]
    add_collar(g)
    P = np.array(g.P); n = len(P); tags = np.array(g.tag)
    log('sweater rest mesh: %d verts, %d faces' % (n, len(g.faces)))
    uvs, mats = compute_uvs(g, P)

    # ---- pose with the body's weights ------------------------------------------------------------------
    Wg = transfer_weights(g, body, sk, W)
    Pp = sk.skin(P, Wg, M)
    # gravity: loose sleeves hang from the top of the arm
    for side in ('l', 'r'):
        ids = g.grid['sl'][side]['ids']
        for k in range(ROWS_SLEEVE + 1):
            row = ids[k]
            w = smoothstep(k, 2, 12) * (1 - 0.75 * smoothstep(k, ROWS_SLEEVE - ROWS_CUFF - 4, ROWS_SLEEVE - ROWS_CUFF + 2))
            Pp[row, 2] -= 0.013 * w
        for row in g.grid['sl'][side]['extra']:
            Pp[row, 2] -= 0.013 * 0.25
    # settle against the posed body
    V_posed = ctx['V_posed']
    pv, pf, _ = util.subdivided(V_posed, body.F, 1)
    # the head and neck must not push the knit around (hair/neck are handled by the collar ring itself)
    bvh_pose = util.bvh_from(pv, pf)
    E = edges_of(g.faces, n)
    free2 = np.ones(n, bool)
    P2 = relax(Pp, E, free2, bvh_pose, 0.0075, iters=8, lam=0.35)
    # keep the stitch rows from shrinking: blend most of the smoothing away except where the pose crumpled the cloth
    crumple = np.clip(np.linalg.norm(P2 - Pp, axis=1) / 0.012, 0, 1)
    Pf = Pp + (P2 - Pp) * (0.35 + 0.65 * crumple)[:, None]
    Pf = relax(Pf, E, free2, bvh_pose, 0.0065, iters=2, lam=0.15)

    ob = util.mesh_object('Sweater', Pf, g.faces, col, uvs=[uv for f in uvs for uv in f])
    me = ob.data
    me.polygons.foreach_set('material_index', np.array(mats, dtype=np.int32))
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    vol = sum(f.calc_center_median().dot(f.normal) * f.calc_area() for f in bm.faces) / 3.0
    centre = sum((f.calc_center_median() * f.calc_area() for f in bm.faces), Vector()) / sum(f.calc_area() for f in bm.faces)
    vol = sum((f.calc_center_median() - centre).dot(f.normal) * f.calc_area() for f in bm.faces) / 3.0
    if vol < 0:                                           # open garment: make sure "outside" really is outside
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(me); bm.free()
    imgs = knit.knit_images(512)
    kw = {}
    sampled = ctx.get('sampled')
    if sampled is not None:                                  # second pass: yarn colours measured on the photograph
        lit = util.srgb_to_linear(np.asarray(sampled['knit'], float)); shade = util.srgb_to_linear(np.asarray(sampled['knit_shadow'], float))
        kw = dict(base=tuple(float(v) for v in np.clip(lit * 1.06, 0, 1)), deep=tuple(float(v) for v in shade * 0.36))
    me.materials.append(knit.knit_material('Knit_Stockinette', imgs['Knit_Stockinette'], **kw))
    me.materials.append(knit.knit_material('Knit_Rib', imgs['Knit_Rib'], bump=1.15, **kw))
    sub = ob.modifiers.new('Subdivision', 'SUBSURF'); sub.levels = 1; sub.render_levels = 1; sub.uv_smooth = 'NONE'
    ctx['ob_sweater'] = ob
    ctx['sweater_P'] = Pf
    ctx['garment'] = g
    try:
        fuzz(ob, col, ctx['q'])
    except Exception as exc:                                 # the halo is a refinement; never fail the build for it
        log('WARNING: knit fuzz skipped (%s)' % exc)
    return ob


def fuzz(sweater, col, q=1.0, density=260000.0, length=0.0052):
    """Mohair-like halo: short fine fibres scattered over the knit by a Geometry Nodes modifier (render-time)."""
    me = bpy.data.meshes.new('Sweater_Fuzz')
    ob = bpy.data.objects.new('Sweater_Fuzz', me); col.objects.link(ob)
    ng = bpy.data.node_groups.new('Knit_Fuzz', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    sd = ng.interface.new_socket('Density', in_out='INPUT', socket_type='NodeSocketFloat'); sd.default_value = density * (0.4 if q < 1 else 1.0); sd.min_value = 0
    sl = ng.interface.new_socket('Length', in_out='INPUT', socket_type='NodeSocketFloat'); sl.default_value = length; sl.min_value = 0
    N = ng.nodes; Lk = ng.links
    gi = N.new('NodeGroupInput'); go = N.new('NodeGroupOutput')
    oi = N.new('GeometryNodeObjectInfo'); oi.transform_space = 'RELATIVE'; oi.inputs['Object'].default_value = sweater
    dist = N.new('GeometryNodeDistributePointsOnFaces'); dist.distribute_method = 'RANDOM'
    Lk.new(oi.outputs['Geometry'], dist.inputs['Mesh']); Lk.new(gi.outputs['Density'], dist.inputs['Density'])
    curve = N.new('GeometryNodeCurveQuadraticBezier'); curve.inputs['Resolution'].default_value = 3
    curve.inputs['Start'].default_value = (0, 0, 0); curve.inputs['Middle'].default_value = (0.25, 0.0, 0.55); curve.inputs['End'].default_value = (0.75, 0.15, 1.0)
    inst = N.new('GeometryNodeInstanceOnPoints')
    Lk.new(dist.outputs['Points'], inst.inputs['Points']); Lk.new(curve.outputs['Curve'], inst.inputs['Instance'])
    # orientation: surface normal, spun randomly about it and tilted a little
    spin = N.new('FunctionNodeRandomValue'); spin.data_type = 'FLOAT'; spin.inputs[2].default_value = 0.0; spin.inputs[3].default_value = 6.2832
    aa = N.new('FunctionNodeAxisAngleToRotation'); aa.inputs['Axis'].default_value = (0, 0, 1); Lk.new(spin.outputs[1], aa.inputs['Angle'])
    rr = N.new('FunctionNodeRotateRotation'); rr.rotation_space = 'LOCAL'
    Lk.new(dist.outputs['Rotation'], rr.inputs['Rotation']); Lk.new(aa.outputs['Rotation'], rr.inputs['Rotate By'])
    Lk.new(rr.outputs['Rotation'], inst.inputs['Rotation'])
    rs = N.new('FunctionNodeRandomValue'); rs.data_type = 'FLOAT'; rs.inputs[2].default_value = 0.45; rs.inputs[3].default_value = 1.25; rs.inputs['Seed'].default_value = 7
    ms = N.new('ShaderNodeMath'); ms.operation = 'MULTIPLY'; Lk.new(rs.outputs[1], ms.inputs[0]); Lk.new(gi.outputs['Length'], ms.inputs[1])
    Lk.new(ms.outputs[0], inst.inputs['Scale'])
    real = N.new('GeometryNodeRealizeInstances'); Lk.new(inst.outputs['Instances'], real.inputs['Geometry'])
    rad = N.new('GeometryNodeSetCurveRadius'); rad.inputs['Radius'].default_value = 0.000016
    Lk.new(real.outputs['Geometry'], rad.inputs['Curve'])
    sm = N.new('GeometryNodeSetMaterial'); sm.inputs['Material'].default_value = knit.fuzz_material()
    Lk.new(rad.outputs['Curve'], sm.inputs['Geometry'])
    Lk.new(sm.outputs['Geometry'], go.inputs['Geometry'])
    mod = ob.modifiers.new('Knit_Fuzz', 'NODES'); mod.node_group = ng
    me.materials.append(sm.inputs['Material'].default_value)
    return ob
