"""Photo landmark data + mesh landmark detection + camera projection helpers.

Imported by build.py.  PHOTO_LM are pixel positions (origin top-left) that I picked by eye
on gridded, magnified crops of input/reference.png (tools/crop_grid.py).  "L"/"R" are the
subject's own left/right (her left eye is on the image right).
"""
import math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

PHOTO_SIZE = (1205, 1306)

PHOTO_LM = {
    # lid points: picked on the lash line, then moved ~2 px toward the iris to the actual
    # lid/eyeball boundary that the 3-D measurement uses
    "eye_R_outer": (473, 403), "eye_R_inner": (537, 392), "eye_R_upper": (505, 381), "eye_R_lower": (510, 411),
    "eye_R_iris": (508, 395),
    "eye_L_outer": (670, 304), "eye_L_inner": (614, 343), "eye_L_upper": (640, 301), "eye_L_lower": (646, 331),
    "eye_L_iris": (643, 315),
    "nose_tip": (619, 441), "nostril_L": (645, 452), "nostril_R": (598, 463),
    "mouth_L": (712, 445), "mouth_R": (592, 513),
    "lip_top": (660, 481), "lip_seam": (651, 499), "lip_bottom": (645, 526),
    "chin": (690, 577),
}
# silhouette points of the face on the image-left side (subject's right cheek/jaw) and the
# visible (subject's left) ear; the right-side jaw is hidden by hair in the photo
PHOTO_CONTOUR = [(450, 450), (470, 475), (495, 500), (525, 525), (560, 550)]
PHOTO_EAR = {"ear_L": (803, 262), "earlobe_L": (800, 326)}

# front centre of the knit collar's top edge
PHOTO_COLLAR = (750, 660)

# relative confidence of each picked point
PHOTO_W = {k: 1.0 for k in PHOTO_LM}
PHOTO_W.update({"nostril_L": 0.5, "nostril_R": 0.5, "chin": 0.7, "eye_R_iris": 1.5, "eye_L_iris": 1.5})


# --------------------------------------------------------------------------- camera
def cam_params(cam, res):
    """(R_world_to_cam 3x3, cam location, k pixels per unit tan) for a perspective camera."""
    W, H = res
    d = cam.data
    M = np.array(cam.matrix_world.normalized())
    R = M[:3, :3].T
    sensor = d.sensor_width if d.sensor_fit in ('AUTO', 'HORIZONTAL') else d.sensor_height
    span = max(W, H) if d.sensor_fit == 'AUTO' else (W if d.sensor_fit == 'HORIZONTAL' else H)
    k = d.lens / sensor * span
    return R, M[:3, 3].copy(), k


def project(P, R, t, k, res):
    """world points -> pixel (x right, y down) in a res-sized frame, plus depth."""
    W, H = res
    c = (np.asarray(P, float) - t) @ R.T
    z = -c[:, 2]
    u = W / 2 + k * c[:, 0] / z
    v = H / 2 - k * c[:, 1] / z
    return np.stack([u, v], 1), z


def unproject(uv, z, R, t, k, res):
    W, H = res
    x = (uv[:, 0] - W / 2) / k * z
    y = -(uv[:, 1] - H / 2) / k * z
    c = np.stack([x, y, -z], 1)
    return c @ R + t


# --------------------------------------------------------------------------- mesh landmarks
def _front_hit(bvh, x, z, y0=-0.35):
    hit = bvh.ray_cast(Vector((x, y0, z)), Vector((0, 1, 0)), 0.5)
    return None if hit[0] is None else np.array(hit[0])


def lid_margin(bvh, ec, r_eye, phi, side_sign):
    """Point on the lid margin around eye centre ec (head-local), phi measured from lateral."""
    fwd = np.array([0, -1.0, 0])
    lat = np.array([side_sign, 0, 0.0])
    up = np.array([0, 0, 1.0])
    for a_deg in np.arange(4, 85, 0.25):
        a = math.radians(a_deg)
        d = math.cos(a) * fwd + math.sin(a) * (math.cos(phi) * lat + math.sin(phi) * up)
        hit = bvh.ray_cast(Vector(ec), Vector(d), r_eye + 0.004)
        if hit[0] is not None:
            a2 = a - math.radians(0.6)
            d2 = math.cos(a2) * fwd + math.sin(a2) * (math.cos(phi) * lat + math.sin(phi) * up)
            return ec + d2 * (r_eye + 0.0008)
    return None


def _bary_map(bvh, tris, src, dst, origin, direction):
    """Ray-cast in src-space mesh, return the same surface point in dst-space coords."""
    hit = bvh.ray_cast(Vector(origin), Vector(direction), 0.6)
    if hit[0] is None:
        return None
    p = np.array(hit[0])
    a, b, c = src[tris[hit[2]]]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    A, B, C = dst[tris[hit[2]]]
    return A * (1 - v - w) + B * v + C * w


def mouth_line_orig(x):
    """z of the lip seam on the unsculpted MakeHuman mesh (same as build.mouth_line)."""
    return 1.4776 + 0.0005 * np.clip(np.abs(x) / 0.0265, 0, 1.3) ** 2


def mesh_landmarks(orig, loc, tris, eye_centres, r_eye):
    """Landmarks in head-local coordinates.

    Nose/mouth points are defined on the canonical (unsculpted) MakeHuman surface and carried
    through the sculpt by barycentric mapping; lid points are ray-cast from the fitted eyeball
    centre in head-local space.
    """
    bvh_o = BVHTree.FromPolygons(orig.tolist(), tris.tolist())
    bvh = BVHTree.FromPolygons(loc.tolist(), tris.tolist())
    fwd = (0, 1, 0)
    m = lambda x, z: _bary_map(bvh_o, tris, orig, loc, (x, -0.35, z), fwd)
    out = {}
    for side, sg in (("L", 1), ("R", -1)):
        ec = np.asarray(eye_centres[side], float)
        # the eyes look into the selfie lens, so the iris centre projects onto the eyeball centre
        out["eye_%s_iris" % side] = ec.copy()
        ring = []
        for phi in np.radians(np.arange(0, 360, 4.0)):
            p = lid_margin(bvh, ec, r_eye, phi, sg)
            if p is not None:
                ring.append(p)
        out["eye_%s_ring" % side] = np.array(ring)
    best = None
    for z in np.arange(1.49, 1.535, 0.0005):
        h = _front_hit(bvh_o, 0.0, z)
        if h is not None and (best is None or h[1] < best[1]):
            best = h
    out["nose_tip"] = m(0.0, best[2])
    for side, sg in (("L", 1), ("R", -1)):
        out["nostril_" + side] = m(sg * 0.0095, best[2] - 0.0095)
        out["mouth_" + side] = m(sg * 0.0262, mouth_line_orig(0.0262))
    zm = mouth_line_orig(0.0)
    a, b = m(0.0, zm + 0.0016), m(0.0, zm - 0.0016)      # the seam itself is a slit
    out["lip_seam"] = None if a is None or b is None else (a + b) / 2
    out["lip_top"] = m(0.0, zm + 0.0086)
    out["lip_bottom"] = m(0.0, zm - 0.0098)
    return out, bvh


def chin_point(loc_world, normals_world, rest, cam_pos, R, t, k, res, down_dir):
    """Chin silhouette point: most 'face-down' projected point of the front chin region."""
    X, Y, Z = rest[:, 0], rest[:, 1], rest[:, 2]
    sel = (np.abs(X) < 0.012) & (Y < -0.105) & (Z > 1.40) & (Z < 1.47)
    idx = np.nonzero(sel)[0]
    uv, _ = project(loc_world[idx], R, t, k, res)
    tocam = cam_pos - loc_world[idx]
    tocam /= np.linalg.norm(tocam, axis=1)[:, None]
    facing = (normals_world[idx] * tocam).sum(1) > 0.0
    s = uv @ np.asarray(down_dir) - 1e6 * (~facing)
    return idx[int(np.argmax(s))]


# --------------------------------------------------------------------------- fitting
def rotvec_matrix(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def fit_camera(Pw, target, wts, R0, t0, k, res, iters=30, prior=None):
    """Gauss-Newton refinement of camera rotation/position so Pw projects onto target.

    prior: (weight, t_ref) soft-pins the camera position (selfie arm length) to t_ref.
    """
    R, t = R0.copy(), t0.copy()
    sw = np.sqrt(np.repeat(wts, 2))

    def resid(p):
        Rn = R @ rotvec_matrix(p[:3]).T
        tn = t + p[3:]
        uv, _ = project(Pw, Rn, tn, k, res)
        r = ((uv - target).ravel()) * sw
        if prior is not None:
            r = np.r_[r, prior[0] * (tn - prior[1])]
        return r, Rn, tn
    for _ in range(iters):
        r0, _, _ = resid(np.zeros(6))
        J = np.empty((len(r0), 6))
        for j in range(6):
            dp = np.zeros(6)
            dp[j] = 1e-5 if j < 3 else 1e-5
            J[:, j] = (resid(dp)[0] - r0) / 1e-5
        step = np.linalg.lstsq(J.T @ J + 1e-3 * np.eye(6), -J.T @ r0, rcond=None)[0]
        _, R, t = resid(step)
        if np.linalg.norm(step) < 1e-8:
            break
    uv, _ = project(Pw, R, t, k, res)
    return R, t, np.linalg.norm(uv - target, axis=1)


def wendland(r, h):
    q = np.clip(r / h, 0, 1)
    return (1 - q) ** 4 * (4 * q + 1)


def rbf_displacement(centres, disp, query, h=0.035, lam=1e-3):
    """Smooth, compactly supported interpolation of displacement vectors."""
    D = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    A = wendland(D, h) + lam * np.eye(len(centres))
    coef = np.linalg.solve(A, disp)
    out = np.zeros_like(query)
    for i in range(0, len(query), 20000):
        q = query[i:i + 20000]
        Kq = wendland(np.linalg.norm(q[:, None] - centres[None], axis=2), h)
        out[i:i + 20000] = Kq @ coef
    return out


def similarity_2d(src, dst):
    """Umeyama: dst ~ s * Rot(theta) @ src + t (2D)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        D[1, 1] = -1
    Rm = U @ D @ Vt
    sc = np.trace(np.diag(S) @ D) / ((a ** 2).sum() / len(src))
    theta = math.atan2(Rm[1, 0], Rm[0, 0])
    return sc, theta, md - sc * Rm @ ms


def eye_points_2d(ring_uv, iris_uv, side_sign_img):
    """Corners/upper/lower lid points of a projected lid-margin ring (image coords).

    side_sign_img: +1 if the outer corner lies toward image +x for this eye."""
    d = ring_uv - iris_uv
    # eye axis = principal direction of the ring
    c = np.cov(d.T)
    w, V = np.linalg.eigh(c)
    ax = V[:, 1]
    if ax[0] * side_sign_img < 0:
        ax = -ax
    t = d @ ax
    outer = ring_uv[int(np.argmax(t))]
    inner = ring_uv[int(np.argmin(t))]
    axis = (outer - inner) / np.linalg.norm(outer - inner)
    nrm = np.array([axis[1], -axis[0]])                 # perpendicular
    if nrm[1] > 0:                                       # make it point up in the image (y down)
        nrm = -nrm
    # lid points: the ring points closest to the perpendicular through the iris centre
    along = (ring_uv - iris_uv) @ axis
    perp = (ring_uv - iris_uv) @ nrm
    up_c = np.where(perp > 0, np.abs(along), 1e9)
    lo_c = np.where(perp < 0, np.abs(along), 1e9)
    return outer, inner, ring_uv[int(np.argmin(up_c))], ring_uv[int(np.argmin(lo_c))]


def visible_eye_opening(bvh, cam_l, Rl, k, res, ec, r_ball, centre_uv, half=48):
    """Pixels (photo frame) where a camera ray reaches the eyeball sphere before the skin.

    bvh/ec/cam_l/Rl are in head-local space; Rl maps camera-space directions to head-local."""
    W, H = res
    pts = []
    cu, cv = int(round(centre_uv[0])), int(round(centre_uv[1]))
    for v in range(cv - half, cv + half + 1):
        for u in range(cu - half, cu + half + 1):
            d = Rl @ np.array([(u - W / 2) / k, -(v - H / 2) / k, -1.0])
            d /= np.linalg.norm(d)
            oc = cam_l - ec
            b = oc @ d
            disc = b * b - (oc @ oc - r_ball * r_ball)
            if disc <= 0:
                continue
            ts = -b - math.sqrt(disc)
            hit = bvh.ray_cast(Vector(cam_l), Vector(d), ts + 1e-4)
            if hit[0] is None:
                pts.append((u, v))
    return np.array(pts, float)


def opening_points(px, iris_uv, side_sign_img):
    """outer/inner corners and upper/lower lid (at the iris) of a visible-opening pixel set."""
    d = px - iris_uv
    w, V = np.linalg.eigh(np.cov(d.T))
    ax = V[:, 1]
    if ax[0] * side_sign_img < 0:
        ax = -ax
    t = d @ ax
    lo_t, hi_t = np.percentile(t, 0.5), np.percentile(t, 99.5)
    outer = px[np.argmin(np.abs(t - hi_t))]
    inner = px[np.argmin(np.abs(t - lo_t))]
    axis = (outer - inner) / np.linalg.norm(outer - inner)
    nrm = np.array([axis[1], -axis[0]])
    if nrm[1] > 0:
        nrm = -nrm
    along = d @ axis
    perp = d @ nrm
    col = np.abs(along) < 2.5
    up = iris_uv + nrm * perp[col].max() if col.any() else iris_uv
    lo = iris_uv + nrm * perp[col].min() if col.any() else iris_uv
    return outer, inner, up, lo


def silhouette_vertex(uv, facing, cand, centre, target, band=7.0):
    """Index of the candidate vertex on the projected silhouette along centre->target."""
    d = target - centre
    L = np.linalg.norm(d)
    d /= L
    rel = uv[cand] - centre
    along = rel @ d
    perp = np.abs(rel @ np.array([-d[1], d[0]]))
    ok = (perp < band) & (np.abs(facing[cand]) < 0.45)
    if not ok.any():
        return None, None
    j = np.argmax(np.where(ok, along, -1e9))
    return cand[j], L - along[j]
