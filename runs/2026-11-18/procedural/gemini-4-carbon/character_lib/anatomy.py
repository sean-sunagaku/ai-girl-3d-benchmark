"""Anatomical base mesh assembly, continuous facial sculpting, realistic eyes, and armature rig."""

import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix


EYE_CENTER_L = Vector((0.0312, -0.1256, 1.5453))
EYE_CENTER_R = Vector((-0.0312, -0.1256, 1.5453))
EYE_RADIUS = 0.01415


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gauss3(x, y, z, cx, cy, cz, sx, sy, sz):
    return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2))


def load_upper_body_with_hands_and_uvs(upper_obj_path: str, base_obj_path: str, collection) -> bpy.types.Object:
    """
    Load input/upper-body.obj topology plus waist ring down to Z=1.04 inside the sweater hem,
    transferring exact MakeHuman UVs from supplied CC0 input/base.obj.
    """
    with open(upper_obj_path, "r", encoding="utf-8") as f:
        ub_lines = f.readlines()
    ub_verts = [tuple(round(float(c), 6) for c in l.split()[1:4]) for l in ub_lines if l.startswith("v ")]
    ub_face_keys = set()
    for l in ub_lines:
        if l.startswith("f "):
            idxs = [int(tok.split("/")[0]) - 1 for tok in l.split()[1:]]
            ub_face_keys.add(tuple(ub_verts[i] for i in idxs))

    b_vs, b_vts, b_body_faces = [], [], []
    cur_g = None
    with open(base_obj_path, "r", encoding="utf-8") as f:
        for l in f:
            if l.startswith("v "):
                sx, sy, sz = map(float, l.split()[1:4])
                b_vs.append((sx * 0.1, -sz * 0.1, (sy + 8.168) * 0.1))
            elif l.startswith("vt "):
                u, v = map(float, l.split()[1:3])
                b_vts.append((u, v))
            elif l.startswith("g "):
                cur_g = l.strip()[2:]
            elif l.startswith("f ") and cur_g == "body":
                toks = [t.split("/") for t in l.split()[1:]]
                b_body_faces.append([(int(p[0]) - 1, int(p[1]) - 1) for p in toks])

    selected_faces = []
    used_v_map = {}
    new_verts = []

    for f_pairs in b_body_faces:
        pts = [b_vs[vi] for vi, _ in f_pairs]
        key = tuple((round(p[0], 6), round(p[1], 6), round(p[2], 6)) for p in pts)
        in_ub = key in ub_face_keys
        in_waist = all(p[2] >= 1.04 and abs(p[0]) < 0.18 for p in pts)
        if in_ub or in_waist:
            remapped = []
            for vi, ti in f_pairs:
                if vi not in used_v_map:
                    used_v_map[vi] = len(new_verts)
                    new_verts.append(b_vs[vi])
                remapped.append((used_v_map[vi], b_vts[ti]))
            selected_faces.append(remapped)

    mesh = bpy.data.meshes.new("Character_Body_Mesh")
    face_v_indices = [[vi for vi, _ in fp] for fp in selected_faces]
    mesh.from_pydata(new_verts, [], face_v_indices)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_uvs = []
    for fp in selected_faces:
        for _, uv in fp:
            loop_uvs.extend([uv[0], uv[1]])
    uv_layer.data.foreach_set("uv", loop_uvs)

    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

    obj = bpy.data.objects.new("Character_Body", mesh)
    collection.objects.link(obj)
    return obj


def sculpt_feminine_likeness(body_obj: bpy.types.Object):
    """
    Continuous anatomical sculpting from neutral MakeHuman hm08 into the youthful East Asian
    feminine proportions of input/reference.png:
    - Shorter, sweeter neotenous midface & philtrum ratio anchored at eye level (Z=1.5453)
    - Larger, brighter almond eyes with snug wraparound eyelids, double-eyelid crease & subtle aegyo-sal
    - Smooth baby-soft cheeks (erasing hm08's adult nasolabial hollow), soft oval/heart jawline & chin
    - Small integrated nose & sealed, plump coral-rose lips in a gentle symmetrical closed-mouth smile
    """
    mesh = body_obj.data
    n = len(mesh.vertices)
    coords = np.zeros(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    P = coords.reshape((n, 3))

    uv_u = np.ones(n, dtype=np.float64) * 0.5
    uv_layer = mesh.uv_layers.get("UVMap")
    if uv_layer:
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                vi = mesh.loops[li].vertex_index
                uv = uv_layer.data[li].uv
                uv_u[vi] = min(uv_u[vi], uv[0]) if uv[0] < 0.28 else uv[0]

    x0 = P[:, 0].copy()
    y0 = P[:, 1].copy()
    z0 = P[:, 2].copy()
    ax0 = np.abs(x0)
    sx = np.sign(x0)

    # Topologically exact oral bag & lip masks computed on invariant hm08 coordinates BEFORE proportion warps!
    is_oral = (z0 > 1.462) & (z0 < 1.496) & (ax0 < 0.036) & ((uv_u < 0.30) | (y0 > -0.1445))
    ext = ~is_oral

    # Exact hm08 stomion curve: Z=1.4771 at X=0 rising to Z=1.4796 at |X|=0.024
    z_stom_hm08 = 1.4771 + 0.0025 * np.clip(ax0 / 0.024, 0.0, 1.2) ** 2
    cupid_arch_0 = 0.0078 + 0.0011 * np.exp(-(((ax0 - 0.0052) / 0.0030) ** 2)) - 0.0007 * np.exp(
        -((ax0 / 0.0024) ** 2)
    )
    lip_half_h0 = np.where(
        z0 >= z_stom_hm08,
        cupid_arch_0 * np.clip(1.0 - (ax0 / 0.0238) ** 1.85, 0.0, 1.0),
        0.0082 * np.clip(1.0 - (ax0 / 0.0228) ** 1.75, 0.0, 1.0),
    )
    d_lip0 = np.abs(z0 - z_stom_hm08) / np.maximum(lip_half_h0, 1e-5)
    lip_mask = (
        (1.0 - _smoothstep(0.66, 1.05, d_lip0))
        * (1.0 - _smoothstep(0.0205, 0.0242, ax0))
        * _smoothstep(-0.140, -0.1480, y0)
        * ext.astype(np.float64)
    )
    lip_line_mask = (
        np.exp(-(((z0 - z_stom_hm08) / 0.0011) ** 2))
        * (1.0 - _smoothstep(0.0195, 0.0238, ax0))
        * _smoothstep(-0.140, -0.1475, y0)
        * ext.astype(np.float64)
    )

    # ------------------------------------------------------------------
    # 1. Global Feminine Body, Neck & Neotenous Facial Proportion Harmonics
    # ------------------------------------------------------------------
    head_t = _smoothstep(1.365, 1.465, z0)
    width_scale = 0.855 * (1.0 - head_t) + 0.985 * head_t
    x_new = x0 * width_scale
    y_new = y0.copy()
    z_new = z0.copy()

    # Slenderize neck cross-section
    neck_band = _smoothstep(1.360, 1.395, z0) * (1.0 - _smoothstep(1.432, 1.462, z0))
    chin_guard = _smoothstep(-0.105, -0.128, y0) * _smoothstep(1.425, 1.442, z0)
    neck_w = neck_band * (1.0 - chin_guard)
    x_new *= 1.0 - 0.145 * neck_w
    neck_y_center = -0.022
    y_new = neck_y_center + (y_new - neck_y_center) * (1.0 - 0.14 * neck_w)

    # Slenderize arms
    arm_mask = _smoothstep(0.135, 0.185, ax0)
    arm_t = np.clip((ax0 - 0.165) / 0.255, 0.0, 1.1)
    arm_cy = -0.015 * (1.0 - arm_t) + (-0.105) * arm_t
    arm_cz = 1.340 * (1.0 - arm_t) + 1.105 * arm_t
    y_new = y_new - 0.17 * arm_mask * (y_new - arm_cy)
    z_new = z_new - 0.17 * arm_mask * (z_new - arm_cz)

    chest_broad = _gauss3(x_new, y_new, z_new, 0.0, -0.120, 1.225, 0.115, 0.075, 0.088)
    y_new -= 0.015 * chest_broad

    # ------------------------------------------------------------------
    # 2. Youthful East Asian Facial Proportions: Shorten Long Midface & Philtrum!
    # ------------------------------------------------------------------
    # Anchor eye level at Z=1.5453; lift nose (+3.2 mm), mouth (+6.2 mm), and chin (+8.5 mm) smoothly!
    ant_face = _smoothstep(-0.035, -0.095, y_new) * _smoothstep(1.410, 1.438, z_new) * (
        1.0 - _smoothstep(1.535, 1.548, z_new)
    )
    # Stronger lift on lower face & philtrum so eye-to-mouth ratio matches reference.png!
    midface_lift = 0.0072 * ant_face * _smoothstep(1.545, 1.455, z_new)
    philtrum_short = 0.0022 * _gauss3(np.abs(x_new), y_new, z_new, 0.0, -0.155, 1.482, 0.042, 0.035, 0.026)
    z_new += midface_lift + philtrum_short

    # ------------------------------------------------------------------
    # 3. Smooth Baby-Soft Cheeks, Oval/Heart Jawline, Forehead & Delicate Nose
    # ------------------------------------------------------------------
    ax_n = np.abs(x_new)

    # Soften supraorbital brow bossing & round feminine forehead
    y_new += 0.0026 * _gauss3(ax_n, y_new, z_new, 0.022, -0.146, 1.564, 0.036, 0.025, 0.014)
    y_new -= 0.0024 * _gauss3(ax_n, y_new, z_new, 0.0, -0.146, 1.600, 0.054, 0.035, 0.026)

    # Broad, smooth youthful apple & buccal cheeks (wide sigma so zero localized ridges ever form!)
    apple_cheek = _gauss3(ax_n, y_new, z_new, 0.038, -0.136, 1.522, 0.028, 0.028, 0.024)
    buccal_soft = _gauss3(ax_n, y_new, z_new, 0.036, -0.132, 1.492, 0.030, 0.030, 0.026) * ext
    y_new -= 0.0036 * apple_cheek + 0.0026 * buccal_soft
    z_new += 0.0018 * apple_cheek
    x_new += sx * (0.0012 * apple_cheek + 0.0008 * buccal_soft)

    # Smooth V-oval feminine jawline & softly rounded heart chin
    jaw_zone = (
        _smoothstep(1.425, 1.452, z_new)
        * (1.0 - _smoothstep(1.495, 1.528, z_new))
        * _smoothstep(-0.020, -0.075, y_new)
    )
    gonion_outer = _smoothstep(0.024, 0.052, ax_n)
    x_new *= 1.0 - 0.145 * jaw_zone * gonion_outer
    z_new += 0.0042 * jaw_zone * gonion_outer

    chin_zone = _gauss3(ax_n, y_new, z_new, 0.0, -0.150, 1.460, 0.032, 0.034, 0.022)
    x_new *= 1.0 - 0.145 * chin_zone
    y_new -= 0.0020 * _gauss3(ax_n, y_new, z_new, 0.0, -0.151, 1.458, 0.020, 0.018, 0.013) * ext

    # Delicate ear angle
    ear_zone = _smoothstep(0.066, 0.078, ax_n) * _smoothstep(1.495, 1.515, z_new) * (
        1.0 - _smoothstep(1.565, 1.585, z_new)
    )
    x_new = sx * (ax_n - 0.15 * ear_zone * np.maximum(ax_n - 0.065, 0.0))

    # Small, delicate integrated feminine nose
    ax_n = np.abs(x_new)
    bridge_zone = _gauss3(ax_n, y_new, z_new, 0.0, -0.154, 1.535, 0.018, 0.020, 0.018)
    x_new *= 1.0 - 0.23 * bridge_zone
    y_new += 0.0018 * _gauss3(ax_n, y_new, z_new, 0.0, -0.151, 1.546, 0.015, 0.015, 0.010)

    alar_zone = _gauss3(ax_n, y_new, z_new, 0.011, -0.158, 1.511, 0.014, 0.016, 0.010)
    x_new *= 1.0 - 0.185 * alar_zone

    tip_zone = _gauss3(ax_n, y_new, z_new, 0.0, -0.167, 1.513, 0.012, 0.013, 0.009)
    y_new += 0.0032 * tip_zone
    z_new += 0.0016 * tip_zone

    # ------------------------------------------------------------------
    # 4. Large, Warm, Bright Almond Eyes, Double-Eyelid & Subtle Aegyo-sal
    # ------------------------------------------------------------------
    ax_n = np.abs(x_new)
    eye_cx_old = 0.0298
    eye_cx = 0.0312
    eye_cy = -0.1256
    eye_cz = 1.5453

    # Gently widen & enlarge palpebral eye region (+6% horizontal, +14% vertical) so outer corner hugs eyeball snugly!
    eye_region_g = _gauss3(ax_n, y_new, z_new, eye_cx_old, -0.138, eye_cz, 0.0175, 0.016, 0.0115) * ext
    dx_from_eye = ax_n - eye_cx_old
    dz_from_eye = z_new - eye_cz
    x_new = sx * (ax_n + (0.0010 + 0.06 * dx_from_eye) * eye_region_g)
    z_new = z_new + (0.14 * dz_from_eye) * eye_region_g

    ax_n = np.abs(x_new)
    eye_front = _smoothstep(-0.132, -0.1385, y_new) * ext
    eye_x_bell = np.exp(-(((ax_n - eye_cx) / 0.0128) ** 2))

    # Open upper and lower eyelids into bright, warm rounded-almond eyes matching reference.png!
    upper_open = eye_x_bell * eye_front * np.exp(-(((z_new - 1.5508) / 0.0028) ** 2))
    lower_open = eye_x_bell * eye_front * np.exp(-(((z_new - 1.5398) / 0.0026) ** 2))
    z_new += 0.00145 * upper_open
    z_new -= 0.00085 * lower_open

    # Gentle lateral almond arch
    outer_corner_g = _gauss3(ax_n, y_new, z_new, 0.0445, -0.137, 1.5458, 0.0055, 0.007, 0.006)
    z_new += 0.0009 * outer_corner_g

    # Soft East Asian double-eyelid crease (~2.8 mm above raised upper lash margin)
    crease_z_curve = 1.5552 - 0.0024 * (((ax_n - eye_cx) / 0.0135) ** 2)
    crease_band = (
        np.exp(-(((ax_n - eye_cx) / 0.0135) ** 2))
        * _smoothstep(-0.135, -0.1395, y_new)
        * np.exp(-(((z_new - crease_z_curve) / 0.00095) ** 2))
    )
    y_new += 0.00125 * crease_band

    pretarsal_band = (
        np.exp(-(((ax_n - eye_cx) / 0.0130) ** 2))
        * _smoothstep(-0.136, -0.140, y_new)
        * np.exp(-(((z_new - (crease_z_curve - 0.0016)) / 0.0010) ** 2))
    )
    y_new -= 0.00055 * pretarsal_band

    # Subtle youthful aegyo-sal under lower lashes
    aegyo_z = 1.5366 + 0.0014 * (((ax_n - eye_cx) / 0.0135) ** 2)
    aegyo_roll = (
        np.exp(-(((ax_n - eye_cx) / 0.0130) ** 2))
        * _smoothstep(-0.134, -0.1388, y_new)
        * np.exp(-(((z_new - aegyo_z) / 0.0016) ** 2))
    )
    y_new -= 0.00090 * aegyo_roll

    # Snugly wrap eyelid margins around 3D eyeball sphere (EYE_RADIUS = 0.01415)
    for side in (-1.0, 1.0):
        ecx = side * eye_cx
        in_lid_zone = (
            (np.abs(x_new - ecx) < 0.0180)
            & (z_new > 1.5365)
            & (z_new < 1.5555)
            & (y_new < -0.1340)
        )
        dx_e = x_new[in_lid_zone] - ecx
        dy_e = y_new[in_lid_zone] - eye_cy
        dz_e = z_new[in_lid_zone] - eye_cz
        r_e = np.sqrt(dx_e * dx_e + dy_e * dy_e + dz_e * dz_e)
        ang_front = np.clip(-dy_e / np.maximum(r_e, 1e-6), 0.0, 1.0)
        min_r = EYE_RADIUS + 0.00095 * (ang_front ** 6.0) + 0.00018
        push_out = np.maximum(min_r - r_e, 0.0)
        y_new[in_lid_zone] -= push_out * 1.02

    # ------------------------------------------------------------------
    # 5. Soft, Plump, Sealed Lips & Gentle Sweet Closed-Mouth Smile!
    # ------------------------------------------------------------------
    y_new[is_oral] += 0.0055  # Keep oral bag well behind sealed lips!
    ax_e = np.abs(x_new)

    # Slightly delicate feminine mouth width (~43 mm)
    mouth_x_zone = _gauss3(ax_e, y_new, z_new, 0.0, -0.153, 1.484, 0.032, 0.022, 0.016) * ext
    x_new *= 1.0 - 0.075 * mouth_x_zone
    ax_e = np.abs(x_new)

    # Press upper & lower lip inner stomion edges firmly together (+0.6 mm overlap seal so ZERO dark slit appears!)
    above_s0 = z0 >= z_stom_hm08
    seal_band = (
        np.exp(-((ax0 / 0.022) ** 2))
        * _smoothstep(-0.142, -0.150, y0)
        * np.exp(-(((z0 - z_stom_hm08) / 0.0022) ** 2))
        * ext
    )
    z_new += np.where(above_s0, -0.00045, 0.00045) * seal_band
    y_new -= 0.0006 * seal_band  # Keep contact line plump and forward rather than pinched inward!

    # Plump, hydrated lower & upper lip fullness
    lower_plump = (
        np.exp(-((ax0 / 0.016) ** 2))
        * _smoothstep(-0.143, -0.151, y0)
        * np.exp(-(((z0 - 1.4725) / 0.0038) ** 2))
        * ext
    )
    upper_plump = (
        np.exp(-((ax0 / 0.017) ** 2))
        * _smoothstep(-0.143, -0.151, y0)
        * np.exp(-(((z0 - 1.4820) / 0.0036) ** 2))
        * ext
    )
    y_new -= 0.0018 * lower_plump + 0.0010 * upper_plump

    # GENTLE, SWEET closed-mouth smile (+1.5 mm corner lift, smooth perioral transition)
    smile_lat = _smoothstep(0.0050, 0.0235, ax0)
    smile_corner = _gauss3(ax_e, y_new, z_new, 0.0210, -0.146, 1.485, 0.018, 0.020, 0.018) * ext
    z_new += 0.0016 * smile_lat * smile_corner
    x_new += sx * 0.0006 * smile_lat * smile_corner

    P_sculpt = np.stack([x_new, y_new, z_new], axis=-1)

    # Selective Laplacian relaxation strictly on cheek/nasolabial skin (erases hm08 adult nasolabial crease
    # while keeping lips, nose, eyes, and ears 100% crisp!)
    adj = [set() for _ in range(n)]
    for poly in mesh.polygons:
        vs_p = poly.vertices
        m_p = len(vs_p)
        for k in range(m_p):
            a_i, b_i = vs_p[k], vs_p[(k + 1) % m_p]
            adj[a_i].add(b_i)
            adj[b_i].add(a_i)
    adj = [list(s) for s in adj]

    ax_s = np.abs(P_sculpt[:, 0])
    cheek_relax_w = (
        _smoothstep(0.018, 0.025, ax_s)
        * (1.0 - _smoothstep(0.054, 0.068, ax_s))
        * _smoothstep(1.444, 1.462, P_sculpt[:, 2])
        * (1.0 - _smoothstep(1.520, 1.534, P_sculpt[:, 2]))
        * _smoothstep(-0.080, -0.120, P_sculpt[:, 1])
        * (1.0 - np.clip(lip_mask * 1.5, 0.0, 1.0))
        * ext.astype(np.float64)
    ) * 0.58

    for _ in range(24):
        nxt_P = P_sculpt.copy()
        for vi in np.where(cheek_relax_w > 1e-3)[0]:
            nbrs = adj[vi]
            if nbrs:
                w_rlx = cheek_relax_w[vi]
                nxt_P[vi] = P_sculpt[vi] * (1.0 - w_rlx) + np.mean(P_sculpt[nbrs], axis=0) * w_rlx
        P_sculpt = nxt_P

    x_new, y_new, z_new = P_sculpt[:, 0], P_sculpt[:, 1], P_sculpt[:, 2]
    ax_e = np.abs(x_new)
    mesh.vertices.foreach_set("co", P_sculpt.astype(np.float32).ravel())
    mesh.update()

    # ------------------------------------------------------------------
    # 6. Store Seamless 3D Rest-Pose & Exact Invariant Masks
    # ------------------------------------------------------------------
    rest_attr = mesh.attributes.new(name="rest_pos", type="FLOAT_VECTOR", domain="POINT")
    rest_attr.data.foreach_set("vector", P_sculpt.astype(np.float32).ravel())

    ear_cutout = _gauss3(ax_e, y_new, z_new, 0.076, -0.038, 1.538, 0.020, 0.027, 0.036)
    front_scalp = _smoothstep(1.606, 1.622, z_new) * _smoothstep(-0.152, -0.055, y_new)
    temple_scalp = (
        _smoothstep(0.045, 0.059, ax_e)
        * _smoothstep(-0.130, -0.035, y_new)
        * _smoothstep(1.548, 1.572, z_new)
    )
    back_scalp = _smoothstep(-0.025, 0.010, y_new) * _smoothstep(1.420, 1.455, z_new)
    top_crown = _smoothstep(1.608, 1.624, z_new)
    scalp_mask = np.clip(
        np.maximum.reduce([front_scalp, back_scalp, temple_scalp, top_crown])
        * (1.0 - np.clip(ear_cutout * 2.4, 0.0, 1.0)),
        0.0,
        1.0,
    )

    for attr_name, arr in [
        ("lip_mask", lip_mask),
        ("lip_line_mask", lip_line_mask),
        ("scalp_mask", scalp_mask),
    ]:
        fa = mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        fa.data.foreach_set("value", arr.astype(np.float32))


def build_realistic_eyes(collection) -> dict:
    """Create realistic two-layer eyeballs, subtle medial caruncles, and lower-lid tear meniscus strips."""
    created = {}
    iris_r_world = 0.00680  # 13.6 mm warm expressive brown iris matching reference.png!

    for side_label, center in [("L", EYE_CENTER_L), ("R", EYE_CENTER_R)]:
        sign_x = 1.0 if side_label == "L" else -1.0

        bm_in = bmesh.new()
        bmesh.ops.create_uvsphere(bm_in, u_segments=64, v_segments=40, radius=EYE_RADIUS)
        bmesh.ops.rotate(bm_in, verts=bm_in.verts, matrix=Matrix.Rotation(math.pi * 0.5, 3, "X"))

        uv_lay = bm_in.loops.layers.uv.new("UVMap")
        for v in bm_in.verts:
            p = v.co.copy()
            rad_xy = math.hypot(p.x, p.z)
            if p.y < 0.0 and rad_xy < iris_r_world * 1.06:
                t_iris = max(0.0, 1.0 - (rad_xy / (iris_r_world * 1.04)) ** 2)
                flat_y = -math.sqrt(max(EYE_RADIUS * EYE_RADIUS - iris_r_world * iris_r_world, 1e-6))
                v.co.y = p.y * (1.0 - t_iris) + (flat_y + 0.00085 * t_iris) * t_iris

        for f in bm_in.faces:
            f.smooth = True
            for loop in f.loops:
                co = loop.vert.co
                if co.y < 0.0:
                    scale = 0.915 / iris_r_world
                    u = 0.5 + 0.5 * (co.x * scale)
                    v_uv = 0.5 + 0.5 * (co.z * scale)
                else:
                    u, v_uv = 0.02, 0.02
                loop[uv_lay].uv = (u, v_uv)

        mesh_in = bpy.data.meshes.new(f"Eyeball_Inner_{side_label}_Mesh")
        bm_in.to_mesh(mesh_in)
        bm_in.free()
        obj_in = bpy.data.objects.new(f"Eyeball_Inner_{side_label}", mesh_in)
        obj_in.location = center.copy()
        collection.objects.link(obj_in)

        bm_out = bmesh.new()
        cornea_r = EYE_RADIUS + 0.00014
        bmesh.ops.create_uvsphere(bm_out, u_segments=64, v_segments=40, radius=cornea_r)
        bmesh.ops.rotate(bm_out, verts=bm_out.verts, matrix=Matrix.Rotation(math.pi * 0.5, 3, "X"))
        for v in bm_out.verts:
            p = v.co
            rad_xy = math.hypot(p.x, p.z)
            if p.y < 0.0 and rad_xy < iris_r_world * 1.12:
                dome = math.cos(0.5 * math.pi * min(rad_xy / (iris_r_world * 1.12), 1.0)) ** 1.6
                p.y -= 0.00095 * dome
        for f in bm_out.faces:
            f.smooth = True

        mesh_out = bpy.data.meshes.new(f"Eyeball_Cornea_{side_label}_Mesh")
        bm_out.to_mesh(mesh_out)
        bm_out.free()
        obj_out = bpy.data.objects.new(f"Eyeball_Cornea_{side_label}", mesh_out)
        obj_out.location = center.copy()
        obj_out.visible_shadow = False
        collection.objects.link(obj_out)

        bm_car = bmesh.new()
        bmesh.ops.create_uvsphere(bm_car, u_segments=16, v_segments=12, radius=0.0017)
        for v in bm_car.verts:
            v.co.x *= 1.10
            v.co.y *= 0.55
            v.co.z *= 0.78
        for f in bm_car.faces:
            f.smooth = True
        mesh_car = bpy.data.meshes.new(f"Eye_Caruncle_{side_label}_Mesh")
        bm_car.to_mesh(mesh_car)
        bm_car.free()
        obj_car = bpy.data.objects.new(f"Eye_Caruncle_{side_label}", mesh_car)
        obj_car.location = Vector((sign_x * 0.0176, -0.1378, 1.5444))
        collection.objects.link(obj_car)

        bm_tear = bmesh.new()
        segs = 22
        r_strip = 0.00038
        ring_verts = []
        for si in range(segs + 1):
            t = si / segs
            lx = sign_x * (0.0180 + t * 0.0255)
            lz = 1.5436 - 0.0046 * math.sin(math.pi * (t ** 0.88)) + 0.0010 * t
            dx_c = lx - center.x
            dz_c = lz - center.z
            ly = center.y - math.sqrt(max((cornea_r + 0.00015) ** 2 - dx_c * dx_c - dz_c * dz_c, 1e-6))
            c_pt = Vector((lx, ly, lz))
            ring = []
            for k in range(6):
                ang = 2.0 * math.pi * k / 6.0
                off = Vector((0.0, r_strip * math.cos(ang), r_strip * math.sin(ang)))
                ring.append(bm_tear.verts.new(c_pt + off))
            ring_verts.append(ring)
        for si in range(segs):
            for k in range(6):
                k2 = (k + 1) % 6
                f = bm_tear.faces.new(
                    [ring_verts[si][k], ring_verts[si + 1][k], ring_verts[si + 1][k2], ring_verts[si][k2]]
                )
                f.smooth = True
        mesh_tear = bpy.data.meshes.new(f"Eye_TearLine_{side_label}_Mesh")
        bm_tear.to_mesh(mesh_tear)
        bm_tear.free()
        obj_tear = bpy.data.objects.new(f"Eye_TearLine_{side_label}", mesh_tear)
        obj_tear.visible_shadow = False
        collection.objects.link(obj_tear)

        created[side_label] = {
            "inner": obj_in,
            "cornea": obj_out,
            "caruncle": obj_car,
            "tear": obj_tear,
        }
    return created


def _point_to_segment_dist_sq(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-12:
        diff = P - a[None, :]
        return np.sum(diff * diff, axis=1)
    t = np.clip(np.dot(P - a[None, :], ab) / ab_len2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    diff = P - proj
    return np.sum(diff * diff, axis=1)


BONE_DEFS = [
    ("spine_lower", (0.0, -0.020, 1.060), (0.0, 0.004, 1.245), None, 0.135),
    ("spine_chest", (0.0, 0.004, 1.245), (0.0, -0.008, 1.362), "spine_lower", 0.125),
    ("neck", (0.0, -0.008, 1.362), (0.0, -0.018, 1.442), "spine_chest", 0.062),
    ("head", (0.0, -0.018, 1.442), (0.0, -0.018, 1.655), "neck", 0.105),
    ("eye.L", tuple(EYE_CENTER_L), (EYE_CENTER_L.x, EYE_CENTER_L.y - 0.025, EYE_CENTER_L.z), "head", 0.012),
    ("eye.R", tuple(EYE_CENTER_R), (EYE_CENTER_R.x, EYE_CENTER_R.y - 0.025, EYE_CENTER_R.z), "head", 0.012),
    ("clavicle.L", (0.024, -0.052, 1.336), (0.143, -0.015, 1.341), "spine_chest", 0.085),
    ("upper_arm.L", (0.143, -0.015, 1.341), (0.268, -0.013, 1.168), "clavicle.L", 0.082),
    ("forearm.L", (0.268, -0.013, 1.168), (0.358, -0.105, 1.102), "upper_arm.L", 0.082),
    ("clavicle.R", (-0.024, -0.052, 1.336), (-0.143, -0.015, 1.341), "spine_chest", 0.085),
    ("upper_arm.R", (-0.143, -0.015, 1.341), (-0.268, -0.013, 1.168), "clavicle.R", 0.082),
    ("forearm.R", (-0.268, -0.013, 1.168), (-0.358, -0.105, 1.102), "upper_arm.R", 0.082),
]


def bind_mesh_to_armature(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object, sharpness: float = 2.4):
    """Assign smooth C2 anatomical skinning weights while keeping skull, jaw, and face 100% rigid to head."""
    mesh = mesh_obj.data
    n = len(mesh.vertices)
    coords = np.zeros(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    P = coords.reshape((n, 3))

    cranium_rigid = _smoothstep(1.438, 1.468, P[:, 2])
    jaw_face_rigid = _smoothstep(1.420, 1.436, P[:, 2]) * _smoothstep(-0.050, -0.080, P[:, 1])
    head_lock = np.clip(np.maximum(cranium_rigid, jaw_face_rigid), 0.0, 1.0)

    skin_bones = [b for b in BONE_DEFS if not b[0].startswith("eye.")]
    weights = np.zeros((n, len(skin_bones)), dtype=np.float64)

    for bi, (bname, head_pt, tail_pt, _, sigma) in enumerate(skin_bones):
        a = np.array(head_pt, dtype=np.float64)
        b = np.array(tail_pt, dtype=np.float64)
        d2 = _point_to_segment_dist_sq(P, a, b)
        w = np.exp(-((np.sqrt(d2) / sigma) ** sharpness))

        if bname.endswith(".L"):
            w *= _smoothstep(-0.015, 0.035, P[:, 0])
            w *= 1.0 - _smoothstep(1.395, 1.435, P[:, 2]) * (1.0 - _smoothstep(0.08, 0.14, P[:, 0]))
            w *= 1.0 - head_lock
        elif bname.endswith(".R"):
            w *= _smoothstep(-0.015, 0.035, -P[:, 0])
            w *= 1.0 - _smoothstep(1.395, 1.435, P[:, 2]) * (1.0 - _smoothstep(0.08, 0.14, -P[:, 0]))
            w *= 1.0 - head_lock
        elif bname == "head":
            w = w * _smoothstep(1.400, 1.445, P[:, 2]) * (1.0 - head_lock) + head_lock * 25.0
        elif bname == "neck":
            w *= (1.0 - head_lock) * _smoothstep(1.315, 1.368, P[:, 2])
            w *= 1.0 - _smoothstep(0.095, 0.155, np.abs(P[:, 0]))
        elif bname.startswith("spine"):
            w *= 1.0 - head_lock
            w *= 1.0 - _smoothstep(1.415, 1.455, P[:, 2])
            w *= 1.0 - _smoothstep(0.155, 0.230, np.abs(P[:, 0]))

        weights[:, bi] = w

    w_sum = np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
    weights /= w_sum

    for bi, (bname, *_) in enumerate(skin_bones):
        vg = mesh_obj.vertex_groups.get(bname) or mesh_obj.vertex_groups.new(name=bname)
        col = weights[:, bi]
        nz = np.where(col > 1e-4)[0]
        for vi in nz:
            vg.add([int(vi)], float(col[vi]), "REPLACE")

    # Create 'body_smooth_only' group (0.0 on rigid head so CorrectiveSmooth NEVER blurs lips/eyelids!)
    vg_sm = mesh_obj.vertex_groups.get("body_smooth_only") or mesh_obj.vertex_groups.new(name="body_smooth_only")
    sm_col = 1.0 - head_lock
    for vi in np.where(sm_col > 1e-3)[0]:
        vg_sm.add([int(vi)], float(sm_col[vi]), "REPLACE")

    mod = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj
    mod.use_deform_preserve_volume = True
    return mod


def _set_world_rot_on_pose_bone(arm_obj: bpy.types.Object, bone_name: str, world_rot_3x3: Matrix):
    bpy.context.view_layer.update()
    pb = arm_obj.pose.bones[bone_name]
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    cur_3x3 = pb.matrix.to_3x3()
    rest_3x3 = pb.bone.matrix_local.to_3x3()
    desired_3x3 = world_rot_3x3 @ rest_3x3
    pb.rotation_quaternion = (cur_3x3.inverted() @ desired_3x3).to_quaternion()
    bpy.context.view_layer.update()


def _aim_pose_bone_world(arm_obj: bpy.types.Object, bone_name: str, target_dir_world: Vector):
    bpy.context.view_layer.update()
    pb = arm_obj.pose.bones[bone_name]
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    cur_3x3 = pb.matrix.to_3x3()
    cur_y_axis = (cur_3x3 @ Vector((0.0, 1.0, 0.0))).normalized()
    tgt_dir = target_dir_world.normalized()
    delta_world = cur_y_axis.rotation_difference(tgt_dir).to_matrix()
    desired_3x3 = delta_world @ cur_3x3
    pb.rotation_quaternion = (cur_3x3.inverted() @ desired_3x3).to_quaternion()
    bpy.context.view_layer.update()


def build_and_pose_armature(
    body_obj: bpy.types.Object,
    eye_objs: dict,
    collection,
    beauty_cam_loc: Vector,
) -> bpy.types.Object:
    """Create Character_Armature, bind Character_Body and eyes, and pose cleanly."""
    arm_data = bpy.data.armatures.new("Character_Armature_Data")
    arm_obj = bpy.data.objects.new("Character_Armature", arm_data)
    collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = arm_data.edit_bones
    for bname, head_pt, tail_pt, parent_name, _ in BONE_DEFS:
        eb = ebones.new(bname)
        eb.head = Vector(head_pt)
        eb.tail = Vector(tail_pt)
        eb.roll = 0.0
        if parent_name:
            eb.parent = ebones[parent_name]
            if eb.head == ebones[parent_name].tail:
                eb.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    bind_mesh_to_armature(body_obj, arm_obj)

    for side_label in ("L", "R"):
        for part_key in ("caruncle", "tear"):
            pobj = eye_objs[side_label][part_key]
            vg = pobj.vertex_groups.new(name="head")
            vg.add(list(range(len(pobj.data.vertices))), 1.0, "REPLACE")
            m = pobj.modifiers.new(name="Armature", type="ARMATURE")
            m.object = arm_obj

        bname = f"eye.{side_label}"
        for sph_key in ("inner", "cornea"):
            sobj = eye_objs[side_label][sph_key]
            wmat = sobj.matrix_world.copy()
            sobj.parent = arm_obj
            sobj.parent_type = "BONE"
            sobj.parent_bone = bname
            bone_tail_mat = arm_obj.matrix_world @ Matrix.Translation(
                arm_data.bones[bname].tail_local
            ) @ arm_data.bones[bname].matrix_local.to_4x4().to_3x3().to_4x4()
            sobj.matrix_parent_inverse = bone_tail_mat.inverted()
            sobj.matrix_world = wmat

    R_chest = (
        Matrix.Rotation(math.radians(3.0), 3, "Z")
        @ Matrix.Rotation(math.radians(-2.5), 3, "Y")
        @ Matrix.Rotation(math.radians(4.0), 3, "X")
    )
    _set_world_rot_on_pose_bone(arm_obj, "spine_chest", R_chest)

    R_neck = (
        Matrix.Rotation(math.radians(-7.5), 3, "Y")
        @ Matrix.Rotation(math.radians(3.0), 3, "X")
        @ Matrix.Rotation(math.radians(-1.5), 3, "Z")
    )
    _set_world_rot_on_pose_bone(arm_obj, "neck", R_neck)

    R_head = (
        Matrix.Rotation(math.radians(-17.5), 3, "Y")
        @ Matrix.Rotation(math.radians(4.0), 3, "X")
        @ Matrix.Rotation(math.radians(-2.0), 3, "Z")
    )
    _set_world_rot_on_pose_bone(arm_obj, "head", R_head)

    _aim_pose_bone_world(arm_obj, "clavicle.R", Vector((-0.88, 0.08, 0.30)))
    _aim_pose_bone_world(arm_obj, "upper_arm.R", Vector((-0.24, -0.90, -0.34)))
    _aim_pose_bone_world(arm_obj, "forearm.R", Vector((-0.05, -0.98, -0.14)))

    _aim_pose_bone_world(arm_obj, "clavicle.L", Vector((0.95, 0.14, 0.05)))
    _aim_pose_bone_world(arm_obj, "upper_arm.L", Vector((0.26, -0.86, -0.43)))
    _aim_pose_bone_world(arm_obj, "forearm.L", Vector((0.06, -0.97, -0.22)))

    # Center irises warmly in palpebral openings aimed right at camera
    gaze_target = beauty_cam_loc + Vector((0.0, 0.0, 0.042))
    for side_label in ("L", "R"):
        bname = f"eye.{side_label}"
        pb = arm_obj.pose.bones[bname]
        eye_world_pos = arm_obj.matrix_world @ pb.head
        gaze_dir = (gaze_target - eye_world_pos).normalized()
        _aim_pose_bone_world(arm_obj, bname, gaze_dir)

    cs = body_obj.modifiers.new(name="CorrectiveSmooth", type="CORRECTIVE_SMOOTH")
    cs.iterations = 5
    cs.factor = 0.45
    cs.smooth_type = "LENGTH_WEIGHTED"
    cs.use_pin_boundary = True
    cs.vertex_group = "body_smooth_only"  # Protects 100% of sculpted facial features!

    sub = body_obj.modifiers.new(name="Subdivision", type="SUBSURF")
    sub.levels = 2
    sub.render_levels = 2
    sub.quality = 3

    return arm_obj
