"""
sculpt.py - Feminine portrait sculpting, UV transfer, pose deformation, and 3D eyes
from the CC0 MakeHuman hm08 upper-body.obj base mesh.
"""
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion, Euler


# Canonical eyeball landmarks from input/environment.json (in meters, Z-up, -Y front)
EYE_CENTER_L = np.array([0.030775, -0.124535, 1.545215], dtype=np.float64)
EYE_CENTER_R = np.array([-0.030775, -0.124535, 1.545215], dtype=np.float64)
EYE_RADIUS = 0.01485  # 14.85mm eyeball radius for tight, natural lid wrap

# Head pose parameters (applied rigidly above upper neck, blended smoothly through neck)
# Roll -14.5 deg around +Y tilts top of head gracefully toward viewer's left (-X)
# Pitch +3.5 deg around +X gives gentle flattering chin tuck
# Yaw -6.0 deg around +Z turns face slightly toward viewer's left, revealing left ear (+X)
HEAD_PIVOT = Vector((0.0, -0.015, 1.462))
HEAD_EULER = Euler((math.radians(3.5), math.radians(-14.5), math.radians(-6.0)), 'XYZ')
HEAD_QUAT = HEAD_EULER.to_quaternion()

# Gentle forward upper-torso lean around waist
TORSO_PIVOT = Vector((0.0, -0.02, 1.05))
TORSO_EULER = Euler((math.radians(4.0), math.radians(-1.5), math.radians(-1.5)), 'XYZ')
TORSO_QUAT = TORSO_EULER.to_quaternion()


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gauss3d(pts, center, sigmas):
    d = (pts - center) / sigmas
    return np.exp(-0.5 * np.sum(d * d, axis=-1))


def load_base_uvs(base_obj_path="input/base.obj"):
    """Parse MakeHuman base.obj body UVs matching upper-body.obj face order."""
    vs = []
    vts = []
    face_uvs = []
    in_body = False
    with open(base_obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vs.append((float(p[1]) * 0.1, -float(p[3]) * 0.1, (float(p[2]) + 8.168) * 0.1))
            elif line.startswith("vt "):
                p = line.split()
                vts.append((float(p[1]), float(p[2])))
            elif line.startswith("g "):
                in_body = (line.split()[1] == "body")
            elif line.startswith("f ") and in_body:
                toks = line.split()[1:]
                v_idx = [int(t.split("/")[0]) - 1 for t in toks]
                vt_idx = [int(t.split("/")[1]) - 1 for t in toks]
                if all((vs[i][2] * 10.0 - 8.168) >= 2.8 - 1e-6 for i in v_idx):
                    face_uvs.append([vts[t] for t in vt_idx])
    return face_uvs


def transfer_uvs(mesh, base_obj_path="input/base.obj"):
    """Attach exact 1-to-1 UV coordinates from base.obj onto upper-body mesh."""
    face_uvs = load_base_uvs(base_obj_path)
    if len(face_uvs) != len(mesh.polygons):
        raise RuntimeError(f"UV face count mismatch: {len(face_uvs)} vs {len(mesh.polygons)}")
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly, uvs in zip(mesh.polygons, face_uvs):
        for loop_idx, uv in zip(poly.loop_indices, uvs):
            uv_layer.data[loop_idx].uv = uv


def get_head_matrix():
    """Return the exact 4x4 rigid world transform for the posed head."""
    t_mat = Matrix.Translation(TORSO_PIVOT) @ TORSO_QUAT.to_matrix().to_4x4() @ Matrix.Translation(-TORSO_PIVOT)
    h_mat = Matrix.Translation(HEAD_PIVOT) @ HEAD_QUAT.to_matrix().to_4x4() @ Matrix.Translation(-HEAD_PIVOT)
    return t_mat @ h_mat


def transform_head_points(pts_np):
    """Apply exact rigid head+torso pose transform to an (N, 3) numpy array of canonical head points."""
    mat = np.array(get_head_matrix(), dtype=np.float64)
    ones = np.ones((pts_np.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts_np, ones])
    return (hom @ mat.T)[:, :3]


def sculpt_canonical_vertices(verts_raw):
    """
    Deform MakeHuman hm08 upper-body vertices in canonical space (meters, Z-up, -Y front)
    into a youthful East Asian feminine portrait likeness matching input/reference.png.
    Returns sculpted canonical vertices and updated eye centers.
    """
    v = verts_raw.copy().astype(np.float64)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ax = np.abs(x)

    # ------------------------------------------------------------------
    # 1. OVERALL FEMININE PROPORTIONS (head shape, neck, shoulders)
    # ------------------------------------------------------------------
    # Slightly more delicate/youthful vertical face proportion (subtle 3.5% compacting of mid/lower face)
    head_mask = smoothstep(1.40, 1.45, z)
    # Compact forehead-to-chin subtly around eye level (z=1.545)
    v[:, 2] += head_mask * (1.545 - z) * 0.032

    # Slightly fuller, rounder cranium/temples, slightly narrower lower jaw (heart-oval face)
    jaw_mask = smoothstep(1.535, 1.455, z) * smoothstep(1.415, 1.445, z)
    # Taper lower jaw width inward cleanly for soft V-oval feminine jawline
    v[:, 0] *= (1.0 - 0.115 * jaw_mask * smoothstep(0.015, 0.055, ax))
    # Soften gonial jaw angle (back-side of jaw around y in [-0.11, -0.055], z in [1.455, 1.500])
    gonial_w = gauss3d(
        np.column_stack([ax, y, z]),
        np.array([0.056, -0.082, 1.472]),
        np.array([0.018, 0.025, 0.022]),
    )
    v[:, 0] -= np.sign(x) * 0.0075 * gonial_w
    v[:, 2] += 0.0045 * gonial_w

    # Petite, gently tapered rounded chin (lift chin very slightly & pull slightly narrower/softer)
    chin_w = gauss3d(v, np.array([0.0, -0.142, 1.442]), np.array([0.028, 0.026, 0.018]))
    v[:, 0] *= (1.0 - 0.14 * chin_w)
    v[:, 2] += 0.0045 * chin_w
    v[:, 1] += 0.0035 * chin_w  # Less protruding chin

    # Soften brow ridge bossing & smooth convex feminine forehead
    forehead_w = gauss3d(v, np.array([0.0, -0.148, 1.585]), np.array([0.052, 0.030, 0.028]))
    v[:, 1] -= 0.0028 * forehead_w  # Gently rounded convex forehead
    brow_boss_w = gauss3d(v, np.array([0.0, -0.153, 1.564]), np.array([0.036, 0.016, 0.010]))
    v[:, 1] += 0.0022 * brow_boss_w  # Flatten masculine glabella/brow ridge

    # Delicate, slightly smaller ears tucked slightly closer to head
    ear_w = smoothstep(0.068, 0.078, ax) * smoothstep(1.485, 1.505, z) * smoothstep(1.590, 1.565, z)
    v[:, 0] -= np.sign(x) * 0.0045 * ear_w
    v[:, 2] += 0.0030 * ear_w

    # Slender graceful feminine neck & softer narrower feminine shoulders
    neck_w = smoothstep(1.345, 1.385, z) * smoothstep(1.460, 1.420, z)
    v[:, 0] *= (1.0 - 0.145 * neck_w)
    v[:, 1] = v[:, 1] * (1.0 - 0.10 * neck_w) + (-0.035) * (0.10 * neck_w)
    # Narrow & soften shoulders/torso (body below neck will sit under sweater, keep slender)
    torso_w = smoothstep(1.410, 1.340, z)
    v[:, 0] *= (1.0 - 0.11 * torso_w)
    # Slight softening of trapezius ridge
    trap_w = gauss3d(
        np.column_stack([ax, y, z]),
        np.array([0.095, -0.005, 1.375]),
        np.array([0.045, 0.045, 0.030]),
    )
    v[:, 2] -= 0.009 * trap_w

    # ------------------------------------------------------------------
    # 2. YOUTHFUL SOFT CHEEKS & HIGH CHEEKBONES (with warm smile lift)
    # ------------------------------------------------------------------
    # Refresh coordinates for fine facial feature sculpting
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ax = np.abs(x)

    # Full soft apples of cheeks (lifted gently by smile)
    cheek_apple_w = gauss3d(
        np.column_stack([ax, y, z]),
        np.array([0.036, -0.142, 1.512]),
        np.array([0.022, 0.024, 0.019]),
    )
    v[:, 1] -= 0.0055 * cheek_apple_w  # Fuller forward cheek volume
    v[:, 2] += 0.0028 * cheek_apple_w  # Gentle smile cheek lift
    v[:, 0] += np.sign(x) * 0.0020 * cheek_apple_w

    # Soft lower cheek transition (prevent hollowing, youthful plump curve)
    lower_cheek_w = gauss3d(
        np.column_stack([ax, y, z]),
        np.array([0.038, -0.132, 1.482]),
        np.array([0.022, 0.022, 0.018]),
    )
    v[:, 1] -= 0.0032 * lower_cheek_w

    # ------------------------------------------------------------------
    # 3. PETITE, DELICATE FEMININE NOSE
    # ------------------------------------------------------------------
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ax = np.abs(x)

    # Slim nasal bridge and nostrils (pull |x| inward)
    nose_body_w = gauss3d(v, np.array([0.0, -0.160, 1.514]), np.array([0.020, 0.022, 0.024]))
    v[:, 0] *= (1.0 - 0.19 * nose_body_w)

    # Lower & soften upper nasal bridge (delicate gentle curve)
    bridge_w = gauss3d(v, np.array([0.0, -0.156, 1.534]), np.array([0.011, 0.015, 0.014]))
    v[:, 1] += 0.0032 * bridge_w

    # Petite, slightly lifted, softly rounded nose tip
    tip_w = gauss3d(v, np.array([0.0, -0.168, 1.507]), np.array([0.011, 0.014, 0.010]))
    v[:, 1] += 0.0024 * tip_w  # Slightly less long projection
    v[:, 2] += 0.0032 * tip_w  # Delicate slight upward lift

    # Slimmer, softer alae (nostril wings)
    ala_w = gauss3d(
        np.column_stack([ax, y, z]),
        np.array([0.0145, -0.155, 1.503]),
        np.array([0.0075, 0.010, 0.0075]),
    )
    v[:, 0] -= np.sign(x) * 0.0026 * ala_w
    v[:, 2] += 0.0018 * ala_w

    # ------------------------------------------------------------------
    # 4. LARGE EXPRESSIVE ALMOND-ROUND EYES, DOUBLE EYELIDS & AEGYO-SAL
    # ------------------------------------------------------------------
    # Re-evaluate eye centers after subtle vertical head compacting
    eye_c_l = EYE_CENTER_L.copy()
    eye_c_r = EYE_CENTER_R.copy()
    # Slightly move eyes very gently forward/outward (~0.5mm) for bright East Asian youthful orbital balance
    eye_c_l += np.array([0.0008, -0.0010, 0.0005])
    eye_c_r += np.array([-0.0008, -0.0010, 0.0005])

    for side, ec in [(1.0, eye_c_l), (-1.0, eye_c_r)]:
        dx = v[:, 0] - ec[0]
        dy = v[:, 1] - ec[1]
        dz = v[:, 2] - ec[2]
        sx = side * dx  # + is outer canthus, - is inner canthus

        # Front eyelid region only (exclude deep back of eye socket bag y > ec[1] - 0.003)
        front_eye = smoothstep(ec[1] + 0.002, ec[1] - 0.009, v[:, 1])
        horiz_eye = np.exp(-0.5 * (dx / 0.0165) ** 2)

        # Open upper eyelid gracefully (+3.4mm at center, smooth almond arch gently overlapping top of iris)
        upper_zone = front_eye * horiz_eye * smoothstep(0.0002, 0.0028, dz) * smoothstep(0.0165, 0.0068, dz)
        arch_shape = np.exp(-0.5 * ((sx - 0.0005) / 0.0125) ** 2)
        v[:, 2] += 0.0034 * upper_zone * (0.45 + 0.55 * arch_shape)

        # Gentle lower lid opening (-1.2mm at center, smooth youthful almond curve)
        lower_zone = front_eye * horiz_eye * smoothstep(-0.0002, -0.0028, dz) * smoothstep(-0.0145, -0.0060, dz)
        lower_arch = np.exp(-0.5 * ((sx - 0.0015) / 0.0120) ** 2)
        v[:, 2] -= 0.0012 * lower_zone * (0.35 + 0.65 * lower_arch)

        # Subtle outward widening of outer canthus (~1.4mm) + gentle almond tilt
        outer_w = front_eye * np.exp(
            -0.5 * (((sx - 0.0145) / 0.0070) ** 2 + (dz / 0.0070) ** 2)
        )
        v[:, 0] += side * 0.0014 * outer_w
        v[:, 2] += 0.0009 * outer_w

        # Double eyelid (supratarsal crease): soft indented fold right above upper tarsal plate
        crease_z = ec[2] + 0.0094 - 0.0020 * ((sx - 0.001) / 0.013) ** 2
        crease_w = (
            smoothstep(ec[1] - 0.008, ec[1] - 0.013, v[:, 1])
            * np.exp(-0.5 * ((sx - 0.001) / 0.0135) ** 2)
            * np.exp(-0.5 * ((v[:, 2] - crease_z) / 0.0019) ** 2)
        )
        v[:, 1] += 0.0021 * crease_w  # Tuck inward to create realistic double-eyelid fold

        # Soft eyelid hood overhang directly above the crease
        hood_z = crease_z + 0.0030
        hood_w = (
            smoothstep(ec[1] - 0.008, ec[1] - 0.013, v[:, 1])
            * np.exp(-0.5 * ((sx - 0.001) / 0.0145) ** 2)
            * np.exp(-0.5 * ((v[:, 2] - hood_z) / 0.0025) ** 2)
        )
        v[:, 1] -= 0.0014 * hood_w

        # Subtle youthful aegyo-sal (soft fullness directly under lower lash margin)
        aegyo_z = ec[2] - 0.0088 + 0.0015 * (sx / 0.012) ** 2
        aegyo_w = (
            smoothstep(ec[1] - 0.008, ec[1] - 0.013, v[:, 1])
            * np.exp(-0.5 * ((sx - 0.0005) / 0.0115) ** 2)
            * np.exp(-0.5 * ((v[:, 2] - aegyo_z) / 0.0023) ** 2)
        )
        v[:, 1] -= 0.0017 * aegyo_w

        # Soft subtle under-aegyo groove
        groove_z = aegyo_z - 0.0036
        groove_w = (
            smoothstep(ec[1] - 0.008, ec[1] - 0.013, v[:, 1])
            * np.exp(-0.5 * ((sx - 0.0005) / 0.0115) ** 2)
            * np.exp(-0.5 * ((v[:, 2] - groove_z) / 0.0022) ** 2)
        )
        v[:, 1] += 0.0007 * groove_w

        # ------------------------------------------------------------------
        # CRITICAL: Spherical eyeball wrap constraint!
        # Guarantee eyelids hug the 3D eyeball smoothly with ZERO clipping & ZERO gap.
        # ------------------------------------------------------------------
        rel = v - ec
        dist = np.linalg.norm(rel, axis=1)
        # Only constrain front lid skin & lid margins (y < ec[1] - 0.004) near the eyeball
        lid_wrap_mask = (
            (v[:, 1] < ec[1] - 0.0035)
            & (dist < EYE_RADIUS + 0.0065)
            & (np.abs(v[:, 0] - ec[0]) < 0.022)
            & (np.abs(v[:, 2] - ec[2]) < 0.018)
        )
        # Minimum safe radius: inner cornea + 0.55mm clearance
        min_r = EYE_RADIUS + 0.00055
        too_close = lid_wrap_mask & (dist < min_r) & (dist > 1e-6)
        v[too_close] = ec + rel[too_close] * (min_r / dist[too_close, None])

        # Also gently snug overly loose front eyelid rim vertices toward the sphere so there is no gap
        rim_zone = (
            (v[:, 1] < ec[1] - 0.011)
            & (np.abs(v[:, 0] - ec[0]) < 0.014)
            & (np.abs(v[:, 2] - ec[2]) < 0.0105)
        )
        rel2 = v - ec
        dist2 = np.linalg.norm(rel2, axis=1)
        max_rim_r = EYE_RADIUS + 0.0020
        too_far_rim = rim_zone & (dist2 > max_rim_r)
        if np.any(too_far_rim):
            target_r = 0.35 * dist2[too_far_rim] + 0.65 * max_rim_r
            v[too_far_rim] = ec + rel2[too_far_rim] * (target_r[:, None] / dist2[too_far_rim, None])

    # ------------------------------------------------------------------
    # 5. SHAPED LIPS & WARM GENTLE CLOSED-MOUTH SMILE
    # ------------------------------------------------------------------
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ax = np.abs(x)

    # Note: after 3.2% vertical compacting around 1.545, original lip seam 1.4795 moved to:
    # 1.4795 + (1.545 - 1.4795)*0.032 = 1.4816
    lip_z = 1.4816

    # Gentle closed-mouth smile curve: lift corners smoothly & curve lip line upward
    # Affect both outer lips and inner mouth bag smoothly so closed lips never separate/intersect
    mouth_zone = smoothstep(-0.125, -0.145, y) * np.exp(-0.5 * ((z - lip_z) / 0.022) ** 2)
    corner_factor = smoothstep(0.004, 0.028, ax) * np.exp(-0.5 * ((ax - 0.027) / 0.018) ** 2)
    # Upward smile lift at corners (+5.2mm)
    v[:, 2] += 0.0052 * mouth_zone * corner_factor
    # Pull corners gently back (+4.5mm) and very slightly wider (+1.6mm)
    v[:, 1] += 0.0046 * mouth_zone * corner_factor
    v[:, 0] += np.sign(x) * 0.0016 * mouth_zone * corner_factor

    # Subtle warm smiling cheek-dimple tuck just outside mouth corners
    dimple_w = gauss3d(
        np.column_stack([ax, y, z]),
        np.array([0.031, -0.146, lip_z + 0.003]),
        np.array([0.007, 0.010, 0.008]),
    )
    v[:, 1] += 0.0018 * dimple_w

    # Soft pillowy lower lip (refined natural profile projection)
    lower_lip_w = (
        smoothstep(-0.148, -0.155, y)
        * np.exp(-0.5 * (ax / 0.0155) ** 2)
        * np.exp(-0.5 * ((z - (lip_z - 0.0052)) / 0.0046) ** 2)
    )
    v[:, 1] -= 0.0014 * lower_lip_w
    v[:, 2] -= 0.0009 * lower_lip_w

    # Delicate shaped upper lip with soft Cupid's bow peaks at |x| ~ 0.005
    upper_lip_w = (
        smoothstep(-0.148, -0.156, y)
        * np.exp(-0.5 * (ax / 0.0145) ** 2)
        * np.exp(-0.5 * ((z - (lip_z + 0.0044)) / 0.0040) ** 2)
    )
    cupid_peaks = np.exp(-0.5 * ((ax - 0.0055) / 0.0045) ** 2)
    v[:, 1] -= 0.0009 * upper_lip_w * (0.65 + 0.35 * cupid_peaks)
    v[:, 2] += 0.0007 * upper_lip_w * cupid_peaks

    # Subtle indentation directly under lower lip (mentolabial fold) so chin transitions smoothly
    sulcus_w = gauss3d(v, np.array([0.0, -0.153, lip_z - 0.014]), np.array([0.016, 0.010, 0.005]))
    v[:, 1] += 0.0012 * sulcus_w

    # ------------------------------------------------------------------
    # 6. SEAL INNER MOUTH SLIT CLEANLY & ENSURE SMOOTH EYE SOCKET BACK
    # ------------------------------------------------------------------
    # (Done naturally because our smile field is smooth across both outer & inner lips)

    return v, eye_c_l, eye_c_r


def apply_pose_to_body(canonical_verts):
    """
    Smoothly apply head tilt + gentle torso lean to body vertices.
    Rigid above z=1.458 so head/face/eyes undergo zero distortion;
    smooth quaternion slerp through neck (z in [1.355, 1.458]).
    """
    v = canonical_verts.copy()
    n = len(v)
    z = v[:, 2]

    # Blend weight for head rotation through neck
    w_head = smoothstep(1.355, 1.458, z)
    w_torso = smoothstep(1.06, 1.24, z)

    id_q = Quaternion((1.0, 0.0, 0.0, 0.0))
    hp = np.array(HEAD_PIVOT[:], dtype=np.float64)
    tp = np.array(TORSO_PIVOT[:], dtype=np.float64)

    out = np.empty_like(v)
    for i in range(n):
        p = Vector(v[i])
        wh = float(w_head[i])
        if wh > 1e-6:
            qh = id_q.slerp(HEAD_QUAT, wh)
            p = HEAD_PIVOT + qh @ (p - HEAD_PIVOT)
        wt = float(w_torso[i])
        if wt > 1e-6:
            qt = id_q.slerp(TORSO_QUAT, wt)
            p = TORSO_PIVOT + qt @ (p - TORSO_PIVOT)
        out[i] = (p.x, p.y, p.z)
    return out


def extract_eye_rim_landmarks(canonical_verts, eye_c_l, eye_c_r):
    """
    Extract exact upper and lower eyelid margin points in canonical head space
    so eyelashes and tear meniscus lines attach with millimeter precision.
    """
    landmarks = {}
    for side_name, side, ec in [("L", 1.0, eye_c_l), ("R", -1.0, eye_c_r)]:
        # Find front eyelid margin vertices around the eye opening
        dx = canonical_verts[:, 0] - ec[0]
        dy = canonical_verts[:, 1] - ec[1]
        dz = canonical_verts[:, 2] - ec[2]
        dist = np.linalg.norm(canonical_verts - ec, axis=1)

        # Lid margin sits right at the front of the eyeball (y < ec[1] - 0.010, dist close to EYE_RADIUS)
        cand = (
            (canonical_verts[:, 1] < ec[1] - 0.0105)
            & (np.abs(dx) < 0.017)
            & (dist < EYE_RADIUS + 0.0038)
        )
        pts = canonical_verts[cand]
        # Bin along X from inner corner to outer corner to trace upper & lower lid rims
        xs = (pts[:, 0] - ec[0]) * side  # -inner to +outer
        upper_pts = []
        lower_pts = []
        bins = np.linspace(-0.0135, 0.0145, 22)
        for i in range(len(bins) - 1):
            m = (xs >= bins[i]) & (xs < bins[i + 1])
            bpts = pts[m]
            if len(bpts) == 0:
                continue
            up_b = bpts[bpts[:, 2] > ec[2] + 0.0005]
            lo_b = bpts[bpts[:, 2] < ec[2] - 0.0005]
            if len(up_b):
                # Lowest Z among upper lid verts near front = true upper lash margin
                idx = np.argmin(up_b[:, 2])
                upper_pts.append(up_b[idx])
            if len(lo_b):
                # Highest Z among lower lid verts near front = true lower lash margin
                idx = np.argmax(lo_b[:, 2])
                lower_pts.append(lo_b[idx])
        landmarks[f"upper_{side_name}"] = np.array(upper_pts) if upper_pts else np.zeros((0, 3))
        landmarks[f"lower_{side_name}"] = np.array(lower_pts) if lower_pts else np.zeros((0, 3))
    return landmarks


def create_eye_assembly(side_name, canonical_center, head_mat, aim_world_target, collection, mats):
    """
    Create anatomically layered 3D eye:
      1. Eyeball (sclera + slightly indented concave iris disc for real optical depth)
      2. Cornea (outer glossy refractive shell with forward corneal bulge)
      3. Caruncle (tiny moist pinkish inner-corner tissue)
    """
    objs = []
    # Compute eye world center & aim rotation looking directly toward beauty camera
    world_c = head_mat @ Vector(canonical_center)
    to_cam = (Vector(aim_world_target) - world_c).normalized()
    # Local eyeball faces -Y, up is +Z (rotated by head roll)
    head_up = (head_mat.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()
    # Build orthonormal basis where -Y = to_cam
    y_axis = -to_cam
    x_axis = head_up.cross(y_axis).normalized()
    z_axis = y_axis.cross(x_axis).normalized()
    eye_rot = Matrix((x_axis, y_axis, z_axis)).transposed().to_4x4()
    eye_mat = Matrix.Translation(world_c) @ eye_rot

    # --- 1. Inner Eyeball Mesh (UV sphere oriented with pole at -Y iris center) ---
    bm = bmesh.new()
    segs = 48
    rings = 32
    r_ball = EYE_RADIUS * 0.985
    # Create UV sphere with poles on Z first, then rotate so +Z pole points to -Y
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=r_ball)
    # Rotate +Z pole to -Y (rotate +90 deg around X)
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90.0), 3, 'X'))

    # Flatten & subtly dish inward the iris disc (angular radius < 27 deg from -Y axis)
    # so cornea refraction + catchlights float naturally in front of the iris!
    iris_angle = math.radians(26.5)
    for vert in bm.verts:
        p = vert.co
        r_xy = math.hypot(p.x, p.z)
        ang = math.atan2(r_xy, -p.y)
        if ang < iris_angle:
            t = 1.0 - (ang / iris_angle) ** 2
            # Flatten front of iris slightly inward (~1.2mm dish depth)
            vert.co.y += 0.0013 * t

    me_eye = bpy.data.meshes.new(f"Eyeball_{side_name}")
    bm.to_mesh(me_eye)
    bm.free()

    # Planar polar UV centered on the -Y iris axis (radius 0..pi mapped to 0..0.5 around (0.5, 0.5))
    uv_layer = me_eye.uv_layers.new(name="UVMap")
    for poly in me_eye.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            vco = me_eye.vertices[me_eye.loops[li].vertex_index].co
            r_xz = math.hypot(vco.x, vco.z)
            ang = math.atan2(r_xz, -vco.y)  # 0 at iris center, pi at back
            rad_uv = (ang / math.pi) * 0.495
            phi = math.atan2(vco.z, vco.x)
            u = 0.5 + rad_uv * math.cos(phi)
            v_uv = 0.5 + rad_uv * math.sin(phi)
            uv_layer.data[li].uv = (u, v_uv)

    obj_eye = bpy.data.objects.new(f"Eyeball_{side_name}", me_eye)
    obj_eye.matrix_world = eye_mat
    if "eyeball" in mats:
        obj_eye.data.materials.append(mats["eyeball"])
    collection.objects.link(obj_eye)
    objs.append(obj_eye)

    # --- 2. Outer Cornea Shell (smooth refractive tear-film dome with subtle bulge over iris) ---
    bm_c = bmesh.new()
    bmesh.ops.create_uvsphere(bm_c, u_segments=48, v_segments=32, radius=EYE_RADIUS * 1.005)
    bmesh.ops.rotate(bm_c, verts=bm_c.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90.0), 3, 'X'))
    bulge_ang = math.radians(28.0)
    for vert in bm_c.verts:
        p = vert.co
        ang = math.atan2(math.hypot(p.x, p.z), -p.y)
        if ang < bulge_ang:
            t = math.cos((ang / bulge_ang) * (math.pi * 0.5)) ** 2
            vert.co.y -= 0.00095 * t  # Realistic subtle corneal dome bulge

    me_c = bpy.data.meshes.new(f"Cornea_{side_name}")
    bm_c.to_mesh(me_c)
    bm_c.free()
    for poly in me_c.polygons:
        poly.use_smooth = True
    obj_c = bpy.data.objects.new(f"Cornea_{side_name}", me_c)
    obj_c.matrix_world = eye_mat
    if "cornea" in mats:
        obj_c.data.materials.append(mats["cornea"])
    # Disable shadow casting on cornea so full crisp light illuminates the iris texture beneath
    obj_c.visible_shadow = False
    collection.objects.link(obj_c)
    objs.append(obj_c)

    # --- 3. Inner Canthus Caruncle (tiny soft moist pinkish corner bead) ---
    side = 1.0 if side_name == "L" else -1.0
    car_canonical = Vector((
        canonical_center[0] - side * 0.0138,
        canonical_center[1] - 0.0122,
        canonical_center[2] - 0.0012,
    ))
    bm_car = bmesh.new()
    bmesh.ops.create_uvsphere(bm_car, u_segments=16, v_segments=12, radius=0.0022)
    bmesh.ops.scale(bm_car, verts=bm_car.verts, vec=(1.25, 0.75, 0.85))
    me_car = bpy.data.meshes.new(f"Caruncle_{side_name}")
    bm_car.to_mesh(me_car)
    bm_car.free()
    for poly in me_car.polygons:
        poly.use_smooth = True
    obj_car = bpy.data.objects.new(f"Caruncle_{side_name}", me_car)
    obj_car.matrix_world = head_mat @ Matrix.Translation(car_canonical)
    if "caruncle" in mats:
        obj_car.data.materials.append(mats["caruncle"])
    collection.objects.link(obj_car)
    objs.append(obj_car)

    return objs


def create_tear_meniscus(side_name, lower_lid_pts, head_mat, collection, mat_tear):
    """Build a thin wet glossy tear-line tube nestled right along the lower eyelid margin."""
    if len(lower_lid_pts) < 4:
        return None
    curve_data = bpy.data.curves.new(f"TearLine_{side_name}", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = 0.00055
    curve_data.bevel_resolution = 3
    curve_data.resolution_u = 8

    spline = curve_data.splines.new('POLY')
    spline.points.add(len(lower_lid_pts) - 1)
    for i, pt in enumerate(lower_lid_pts):
        # Nudge slightly inward/upward onto lid-eyeball seam
        p_can = Vector((pt[0], pt[1] + 0.0003, pt[2] + 0.00035))
        pw = head_mat @ p_can
        spline.points[i].co = (pw.x, pw.y, pw.z, 1.0)

    obj = bpy.data.objects.new(f"TearLine_{side_name}", curve_data)
    if mat_tear:
        obj.data.materials.append(mat_tear)
    obj.visible_shadow = False
    collection.objects.link(obj)
    return obj


def build_character_body(collection, mats, beauty_cam_pos=(0.0, -0.58, 1.56)):
    """
    Import upper-body.obj, transfer MakeHuman UVs, sculpt the feminine portrait likeness,
    apply pose, add Subdivision Surface, and construct the 3D eye assemblies.
    Returns a dict with body object, canonical sculpted vertices, eye centers, head matrix, and landmarks.
    """
    bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis='Y', up_axis='Z')
    body_obj = bpy.context.selected_objects[0]
    body_obj.name = "Character_Body"
    body_obj.data.name = "Character_Body_Mesh"

    # Move into target collection cleanly
    for col in list(body_obj.users_collection):
        col.objects.unlink(body_obj)
    collection.objects.link(body_obj)

    me = body_obj.data
    transfer_uvs(me, "input/base.obj")

    raw_verts = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    canonical_verts, eye_c_l, eye_c_r = sculpt_canonical_vertices(raw_verts)
    posed_verts = apply_pose_to_body(canonical_verts)

    # Write posed vertices back to mesh
    for i, v in enumerate(me.vertices):
        v.co = (posed_verts[i, 0], posed_verts[i, 1], posed_verts[i, 2])
    for poly in me.polygons:
        poly.use_smooth = True
    me.update()

    # Cleanly remove covered lower torso and bare T-pose arm faces well inside the sweater,
    # preserving the entire head, ears, full neck, and upper clavicle cushion down to z=1.305.
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    # Tag each vert with its original index so canonical_verts stays 1-to-1 after trimming
    orig_idx_layer = bm.verts.layers.int.new("orig_idx")
    for i, bv in enumerate(bm.verts):
        bv[orig_idx_layer] = i
    faces_to_del = []
    for f in bm.faces:
        c_pts = [canonical_verts[v[orig_idx_layer]] for v in f.verts]
        max_z = max(p[2] for p in c_pts)
        max_ax = max(abs(p[0]) for p in c_pts)
        # Keep head, full neck, and upper collar interior (within radius ~7.2cm of neck center)
        if max_z < 1.335 or (max_z < 1.395 and max_ax > 0.072):
            faces_to_del.append(f)
    if faces_to_del:
        bmesh.ops.delete(bm, geom=faces_to_del, context='FACES')
        # Remove loose verts left behind
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.verts.ensure_lookup_table()
    kept_indices = [bv[orig_idx_layer] for bv in bm.verts]
    canonical_verts = canonical_verts[kept_indices]
    posed_verts = posed_verts[kept_indices]
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    me.update()

    # Smooth Subdivision Surface (level 2 for silky-smooth skin & crisp eyelids/lips)
    subsurf = body_obj.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 2

    if "skin" in mats:
        body_obj.data.materials.append(mats["skin"])

    head_mat = get_head_matrix()
    eye_landmarks = extract_eye_rim_landmarks(canonical_verts, eye_c_l, eye_c_r)

    eye_objs = []
    eye_objs.extend(create_eye_assembly("L", eye_c_l, head_mat, beauty_cam_pos, collection, mats))
    eye_objs.extend(create_eye_assembly("R", eye_c_r, head_mat, beauty_cam_pos, collection, mats))
    t_l = create_tear_meniscus("L", eye_landmarks["lower_L"], head_mat, collection, mats.get("tear"))
    t_r = create_tear_meniscus("R", eye_landmarks["lower_R"], head_mat, collection, mats.get("tear"))
    if t_l:
        eye_objs.append(t_l)
    if t_r:
        eye_objs.append(t_r)

    return {
        "body_obj": body_obj,
        "canonical_verts": canonical_verts,
        "posed_verts": posed_verts,
        "eye_c_l": eye_c_l,
        "eye_c_r": eye_c_r,
        "head_mat": head_mat,
        "eye_landmarks": eye_landmarks,
        "eye_objs": eye_objs,
    }
