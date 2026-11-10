"""Rest-pose facial/body sculpt of the MakeHuman base toward the reference look (all values in meters).

Everything here is a smooth, symmetric, analytic deformation of the supplied topology; no external
shape data is used. The reference photograph only informed the hand-chosen numbers.
"""
import numpy as np
from .base import bump, smoothstep

EYE_C0 = np.array([0.03078, -0.12454, 1.54521])  # MakeHuman helper eye centre (left); right is mirrored
EYE_R0 = 0.01495                                  # helper eyeball radius the lids were modelled around
EYE_SCALE = 1.085                                 # the whole eye (globe + lids) is enlarged about its centre
EYE_C = EYE_C0 + np.array([0.0, 0.0013, 0.0])     # ...and set back a little so the cornea does not bulge forward
EYE_R = EYE_R0 * EYE_SCALE


class Sculptor:
    def __init__(self, body):
        self.b = body
        self.V = body.V

    # -- primitive brushes -------------------------------------------------------------------
    def _sides(self, c, vec=None, sym=True):
        c = np.asarray(c, float)
        out = [(c, None if vec is None else np.asarray(vec, float))]
        if sym and abs(c[0]) > 1e-9:
            c2 = c.copy(); c2[0] *= -1
            v2 = None
            if vec is not None:
                v2 = np.asarray(vec, float).copy(); v2[0] *= -1
            out.append((c2, v2))
        return out

    def grab(self, c, r, vec, sym=True, mask=None):
        for cc, vv in self._sides(c, vec, sym):
            w = bump(np.linalg.norm(self.V - cc, axis=1), r)
            if mask is not None:
                w = w * mask
            self.V += w[:, None] * vv

    def scale(self, c, r, s, sym=True, mask=None):
        s = np.asarray(s, float)
        for cc, _ in self._sides(c, None, sym):
            w = bump(np.linalg.norm(self.V - cc, axis=1), r)
            if mask is not None:
                w = w * mask
            self.V += w[:, None] * ((self.V - cc) * (s - 1.0))

    def inflate(self, c, r, amt, sym=True, mask=None):
        N = self.b.normals(self.V)
        for cc, _ in self._sides(c, None, sym):
            w = bump(np.linalg.norm(self.V - cc, axis=1), r)
            if mask is not None:
                w = w * mask
            self.V += (w * amt)[:, None] * N

    def smooth(self, c, r, iters=4, strength=0.5, sym=True, mask=None):
        wts = np.zeros(len(self.V))
        for cc, _ in self._sides(c, None, sym):
            wts = np.maximum(wts, bump(np.linalg.norm(self.V - cc, axis=1), r))
        if mask is not None:
            wts = wts * mask
        E = self.b.E
        deg = np.zeros(len(self.V)); np.add.at(deg, E[:, 0], 1); np.add.at(deg, E[:, 1], 1)
        for _ in range(iters):
            acc = np.zeros_like(self.V)
            np.add.at(acc, E[:, 0], self.V[E[:, 1]]); np.add.at(acc, E[:, 1], self.V[E[:, 0]])
            avg = acc / np.maximum(deg, 1)[:, None]
            self.V += (wts * strength)[:, None] * (avg - self.V)

    # -- eye opening: move lid skin in angle space around the eyeball, keeping its radius ---------
    def eye_warp(self, moves, r_keep=0.0215 * 1.085, r_fade=0.030 * 1.085):
        """moves: list of (a_deg, e_deg, da_deg, de_deg, radius_deg) for the LEFT eye; mirrored for the right."""
        for side in (1.0, -1.0):
            c = EYE_C * np.array([side, 1, 1])
            d = self.V - c
            r = np.linalg.norm(d, axis=1)
            near = r < r_fade
            dd = d[near]; rr = r[near]
            a = np.degrees(np.arctan2(dd[:, 0] * side, -dd[:, 1]))
            e = np.degrees(np.arcsin(np.clip(dd[:, 2] / rr, -1, 1)))
            front = (-dd[:, 1] > -0.004)               # only the frontal cap (lids), not the back of the socket
            wr = (1.0 - smoothstep(rr, r_keep, r_fade)) * front
            da = np.zeros(len(rr)); de = np.zeros(len(rr))
            for (a0, e0, ma, me, rad) in moves:
                w = bump(np.hypot(a - a0, e - e0), rad) * wr
                da += w * ma; de += w * me
            a2 = np.radians(a + da); e2 = np.radians(e + de)
            nd = np.stack([np.sin(a2) * np.cos(e2) * side, -np.cos(a2) * np.cos(e2), np.sin(e2)], 1) * rr[:, None]
            self.V[near] = c + nd


def eye_radii(body):
    """Per-vertex distance from its eyeball centre (rest), used to re-seat lids after other sculpting."""
    out = {}
    for side in (1.0, -1.0):
        c = EYE_C0 * np.array([side, 1, 1])
        r = np.linalg.norm(body.V0 - c, axis=1)
        idx = np.where(r < 0.024)[0]
        out[side] = (idx, r[idx].copy() * EYE_SCALE)
    return out


def enlarge_eyes(body):
    """Uniformly scale each eye region (lids, socket) about the helper eye centre and move it to EYE_C."""
    for side in (1.0, -1.0):
        c0 = EYE_C0 * np.array([side, 1, 1]); c1 = EYE_C * np.array([side, 1, 1])
        d = body.V - c0
        r = np.linalg.norm(d, axis=1)
        w = 1.0 - smoothstep(r, 0.0200, 0.0360)
        body.V += w[:, None] * (d * (EYE_SCALE - 1.0) + (c1 - c0))


def reseat_lids(body, radii, extra=0.0):
    """Restore the original lid-to-eyeball clearance (lids keep wrapping the same sphere)."""
    for side, (idx, r0) in radii.items():
        c = EYE_C * np.array([side, 1, 1])
        d = body.V[idx] - c
        r = np.linalg.norm(d, axis=1)
        w = 1.0 - smoothstep(r0, 0.0185 * EYE_SCALE, 0.024 * EYE_SCALE)      # full restore near the globe, fading into the orbit
        target = r0 + extra
        newr = r + w * (target - r)
        # the inner (cavity-side) lid surface rides directly on the globe so no gap opens at the canthi
        inner = body.enclosed[idx] & (d[:, 1] < -0.002)
        newr = np.where(inner, np.minimum(newr, EYE_R + 0.00035), newr)
        body.V[idx] = c + d / r[:, None] * newr[:, None]


DEFAULT = dict(
    chin_up=0.0065, chin_back=0.0020, jaw_in=0.0, jowl_in=0.0, cheek=0.0026, cheek_wide=0.0040, cheek_low=0.0046, forehead=0.0020,
    midface=(0.0038, 0.0052, 0.0070, 0.0020),          # lift of nose tip / subnasale / mouth / chin (shorter mid-face)
    nose_width=0.74, nose_tip_back=0.0042, nose_tip_up=0.0014, bridge_back=0.0026,
    smile_out=0.0006, smile_up=0.0026, smile_back=0.0020,
    smile_left_extra=(0.0010, 0.0006, 0.0008), mouth_shift=0.0026,   # the reference smile is slightly lop-sided
    lip_upper_full=0.0022, lip_lower_full=0.0032,
    eye_outer=11.0, eye_outer_up=1.0, eye_inner=-3.0, eye_inner_down=-4.5, eye_upper=12.0, eye_lower=-4.2,
    neck_slim=0.94, crease=0.0007, aegyo=0.0008,
)


def sculpt_face(body, P=None):
    P = dict(DEFAULT, **(P or {}))
    s = Sculptor(body)
    radii = eye_radii(body)
    V = body.V

    # ---- overall head: softer, shorter lower face -------------------------------------------------
    s.grab((0, -0.135, 1.435), 0.050, (0, P['chin_back'], P['chin_up']))             # shorter chin
    s.grab((0, -0.150, 1.462), 0.022, (0, 0.0004, 0.0016))                           # tuck the mentolabial fold up
    s.grab((0.050, -0.060, 1.452), 0.045, (-P['jaw_in'], 0, 0.0015))                 # narrower jaw angle
    s.grab((0.030, -0.125, 1.445), 0.030, (-P['jowl_in'], 0.0005, 0.0008))           # V-line towards the chin
    s.grab((0, -0.150, 1.447), 0.020, (0, 0.0012, 0.0))                              # slightly less projecting chin
    s.grab((0.046, -0.122, 1.502), 0.030, (P['cheek'] * 0.4, -P['cheek'], P['cheek'] * 0.5))  # fuller front cheek
    s.grab((0.058, -0.085, 1.488), 0.048, (P['cheek_wide'], 0, 0))                   # fuller, rounder cheek outline
    s.grab((0.046, -0.100, 1.462), 0.040, (P['cheek_low'], -0.0006, 0))              # soft lower cheek / jaw line
    s.grab((0.020, -0.138, 1.447), 0.022, (0.0016, 0, 0.0004))                       # rounder, softer chin
    s.grab((0, -0.135, 1.615), 0.070, (0, -P['forehead'], 0))                        # rounder forehead
    s.grab((0.060, -0.085, 1.585), 0.040, (-0.0012, 0, 0))                           # softer temple width
    s.grab((0, -0.153, 1.572), 0.020, (0, 0.0010, 0))                                # flatter glabella

    # ---- shorter mid-face: lift nose, mouth and (less) chin towards the eyes ------------------------
    mt, ms, mm_, mc = P['midface']
    zk = np.array([1.380, 1.433, 1.4777, 1.4957, 1.5077, 1.5360, 1.70])
    dk = np.array([0.0, mc, mm_, ms, mt, 0.0, 0.0])
    front = 1.0 - smoothstep(V[:, 1], -0.095, -0.040)
    lift = np.interp(V[:, 2], zk, dk)
    # smooth the piecewise-linear profile a little so no crease appears at the knots
    lift = 0.5 * lift + 0.25 * np.interp(V[:, 2] + 0.004, zk, dk) + 0.25 * np.interp(V[:, 2] - 0.004, zk, dk)
    V[:, 2] += lift * front

    # ---- nose: narrower wings, smaller softly up-turned tip, lower bridge -------------------------
    w = bump(np.linalg.norm((V - np.array([0, -0.150, 1.505 + mt])) * np.array([1.0, 0.8, 1.0]), axis=1), 0.026)
    V[:, 0] *= 1.0 - w * (1.0 - P['nose_width'])
    s.grab((0, -0.168, 1.508 + mt), 0.015, (0, P['nose_tip_back'], P['nose_tip_up']))
    s.scale((0, -0.160, 1.507 + mt), 0.014, (0.90, 0.96, 0.92), sym=False)            # smaller, rounder tip
    s.grab((0, -0.157, 1.530), 0.022, (0, P['bridge_back'], 0))                      # low, soft bridge
    s.grab((0, -0.151, 1.547), 0.016, (0, P['bridge_back'] * 0.6, 0))
    s.smooth((0, -0.155, 1.528), 0.020, iters=3, strength=0.5, sym=False)

    # ---- mouth: closed-mouth smile ---------------------------------------------------------------
    zm = 1.4777 + mm_
    s.grab((0.0252, -0.141, zm), 0.019, (P['smile_out'], P['smile_back'], P['smile_up']))
    ex = P['smile_left_extra']
    s.grab((0.0262, -0.140, zm + 0.002), 0.020, ex, sym=False)                        # her left corner pulls further
    s.grab((0.0, -0.150, zm), 0.034, (P['mouth_shift'], 0, 0), sym=False)
    s.grab((0.036, -0.128, zm + 0.014), 0.022, (0.0008, -0.0012, 0.0016))             # cheek lifted by the smile
    s.grab((0, -0.160, zm + 0.0078), 0.010, (0, -P['lip_upper_full'], 0.0003), sym=False)
    s.grab((0.008, -0.158, zm + 0.0083), 0.007, (0, -P['lip_upper_full'] * 0.7, 0.0004))  # cupid's bow peaks
    s.grab((0, -0.1566, zm - 0.0062), 0.012, (0, -P['lip_lower_full'], -0.0004), sym=False)
    s.grab((0, -0.156, zm - 0.0110), 0.010, (0, -0.0012, -0.0024), sym=False)        # taller, fuller lower lip
    s.grab((0, -0.159, zm + 0.0082), 0.009, (0, -0.0010, 0.0016), sym=False)         # taller upper lip
    s.grab((0.0085, -0.158, zm + 0.0088), 0.007, (0, -0.0008, 0.0014))
    s.grab((0, -0.156, zm + 0.0128), 0.006, (0, 0.0006, 0), sym=False)                # philtrum dip

    # ---- eyes: larger, rounder opening that keeps wrapping the globe ---------------------------------
    enlarge_eyes(body)
    s.eye_warp([
        (49.5, 0.3, P['eye_outer'], P['eye_outer_up'], 24.0),
        (-41.5, -0.2, P['eye_inner'], P['eye_inner_down'], 16.0),
        (0.0, 15.8, 0.0, P['eye_upper'], 34.0),
        (6.0, -15.4, 0.0, P['eye_lower'], 30.0),
    ])
    # lower-lid fullness and a soft upper-lid fold
    s.inflate((0.031, -0.139, 1.5330), 0.0065, P['aegyo'], mask=(~body.enclosed).astype(float))
    s.inflate((0.031, -0.139, 1.5600), 0.0050, -P['crease'], mask=(~body.enclosed).astype(float))

    # ---- neck ------------------------------------------------------------------------------------
    wz = smoothstep(V[:, 2], 1.335, 1.375) * (1.0 - smoothstep(V[:, 2], 1.415, 1.445))
    wr = 1.0 - smoothstep(np.hypot(V[:, 0], V[:, 1] + 0.02), 0.060, 0.085)
    wn = wz * wr
    V[:, 0] *= 1.0 - wn * (1.0 - P['neck_slim'])
    V[:, 1] = -0.02 + (V[:, 1] + 0.02) * (1.0 - wn * (1.0 - P['neck_slim']) * 0.6)

    reseat_lids(body, radii)

    # ---- torso under the sweater: keep it a smooth, neutral form (it is only a collision/drape support) --------
    chest = np.where((V[:, 2] > 1.19) & (V[:, 2] < 1.33) & (np.abs(V[:, 0]) > 0.025) & (np.abs(V[:, 0]) < 0.11) & (V[:, 1] < -0.06))[0]
    if len(chest):
        apex = chest[np.argmin(V[chest, 1])]
        c = V[apex].copy(); c[0] = abs(c[0])
        s.smooth(c, 0.050, iters=14, strength=0.6)
        s.grab(c, 0.045, (0, 0.0045, 0))
    return P
