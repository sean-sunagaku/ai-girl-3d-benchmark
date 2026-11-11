import os
import math
from collections import deque
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix


# Canonical head/neck pose parameters (shared across Head, Eyes, Brows, Lashes, Hair)
NECK_PIVOT = np.array([0.0, -0.020, 1.380], dtype=np.float64)
HEAD_ROLL_Y_DEG = -18.5   # tilts crown toward -X (viewer's left / character's right shoulder)
HEAD_PITCH_X_DEG = 5.0    # slight forward nod so she gazes warmly up through lashes
HEAD_YAW_Z_DEG = -3.0     # subtle turn exposing character's left (+X) ear cleanly


def _head_rotation_matrix(weight=1.0):
    rx = math.radians(HEAD_PITCH_X_DEG * weight)
    ry = math.radians(HEAD_ROLL_Y_DEG * weight)
    rz = math.radians(HEAD_YAW_Z_DEG * weight)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def pose_point(pt, neck_weight=None):
    p = np.asarray(pt, dtype=np.float64)
    if neck_weight is None:
        t = np.clip((p[2] - 1.335) / (1.438 - 1.335), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
    else:
        w = float(neck_weight)
    if w <= 1e-6:
        return p.copy()
    R = _head_rotation_matrix(w)
    shift = np.array([-0.004 * w, -0.003 * w, 0.0], dtype=np.float64)
    return NECK_PIVOT + R @ (p - NECK_PIVOT) + shift


def pose_points_array(pts, rigid_head=False):
    pts = np.asarray(pts, dtype=np.float64)
    if rigid_head:
        R = _head_rotation_matrix(1.0)
        shift = np.array([-0.004, -0.003, 0.0], dtype=np.float64)
        return NECK_PIVOT + (pts - NECK_PIVOT) @ R.T + shift

    out = np.empty_like(pts)
    for i in range(len(pts)):
        out[i] = pose_point(pts[i])
    return out


def _load_upper_body_with_uvs(work_dir):
    """
    Load input/upper-body.obj and recover exact UV coordinates via topology inspection
    of input/base.obj. Keep the complete continuous Head, Ears, Throat, and Neck down to
    z >= 1.305 (|x| <= 0.095 below z=1.40), tucking the base of the neck cleanly inside
    the knitted crew-neck sweater collar so no bare chest overlaps the sweater.
    """
    base_path = os.path.join(work_dir, "input", "base.obj")
    ub_path = os.path.join(work_dir, "input", "upper-body.obj")

    base_v = []
    base_vt = []
    base_body_faces = {}
    cur_g = ""
    with open(base_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = (float(t) for t in line.split()[1:4])
                base_v.append((round(x * 0.1, 5), round(-z * 0.1, 5), round((y + 8.168) * 0.1, 5)))
            elif line.startswith("vt "):
                base_vt.append(tuple(float(t) for t in line.split()[1:3]))
            elif line.startswith("g "):
                cur_g = line.strip()[2:]
            elif line.startswith("f ") and cur_g == "body":
                items = [t.split("/") for t in line.split()[1:]]
                v_idx = tuple(int(it[0]) - 1 for it in items)
                vt_idx = [int(it[1]) - 1 for it in items]
                base_body_faces[v_idx] = vt_idx

    v_lookup = {co: idx for idx, co in enumerate(base_v)}

    ub_verts = []
    ub_faces = []
    with open(ub_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                ub_verts.append(tuple(float(t) for t in line.split()[1:4]))
            elif line.startswith("f "):
                ub_faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])

    ub_to_base = [v_lookup[(round(x, 5), round(y, 5), round(z, 5))] for (x, y, z) in ub_verts]

    def keep_coord(c):
        cx, cy, cz = c
        if cz >= 1.415:
            return True
        # Below z=1.415, keep neck & upper collarbone cylinder down to z=1.305 (tucked inside crew collar!)
        return (cz >= 1.305) and (abs(cx) <= 0.095) and (cy >= -0.098)

    keep_v = set()
    kept_faces = []
    kept_face_uvs = []
    for f_v in ub_faces:
        if all(keep_coord(ub_verts[vi]) for vi in f_v):
            base_key = tuple(ub_to_base[vi] for vi in f_v)
            uv_indices = base_body_faces[base_key]
            kept_faces.append(f_v)
            kept_face_uvs.append([base_vt[uvi] for uvi in uv_indices])
            keep_v.update(f_v)

    old_to_new = {}
    new_verts = []
    for old_i in sorted(keep_v):
        old_to_new[old_i] = len(new_verts)
        new_verts.append(ub_verts[old_i])

    remapped_faces = [[old_to_new[vi] for vi in f_v] for f_v in kept_faces]

    mesh = bpy.data.meshes.new("Head_Neck_Base")
    mesh.from_pydata(new_verts, [], remapped_faces)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loop_idx = 0
    for f_uv in kept_face_uvs:
        for uv in f_uv:
            uv_layer.data[loop_idx].uv = uv
            loop_idx += 1
    mesh.update()

    obj = bpy.data.objects.new("Character_Head_Neck", mesh)
    bpy.context.collection.objects.link(obj)

    sub = obj.modifiers.new(name="BaseRefine", type='SUBSURF')
    sub.levels = 1
    sub.render_levels = 1
    sub.uv_smooth = 'PRESERVE_BOUNDARIES'
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)
    refined_mesh = bpy.data.meshes.new_from_object(obj_eval, preserve_all_data_layers=True, depsgraph=deps)
    refined_mesh.name = "Character_Head_Neck_Mesh"
    obj.modifiers.clear()
    old_m = obj.data
    obj.data = refined_mesh
    bpy.data.meshes.remove(old_m)
    return obj


def _find_oral_cavity_mask(mesh, co):
    """Topological BFS flood-fill from deep throat/mouth-bag center so outer cheeks are NEVER flagged!"""
    n = len(co)
    adj = [[] for _ in range(n)]
    for e in mesh.edges:
        a, b = e.vertices[0], e.vertices[1]
        adj[a].append(b)
        adj[b].append(a)

    # Seed at deep posterior oral cavity wall near (0.0, -0.065, 1.485)
    seed = int(np.argmin((co[:, 0] ** 2) + ((co[:, 1] + 0.065) ** 2) + ((co[:, 2] - 1.485) ** 2)))
    is_oral = np.zeros(n, dtype=bool)
    q = deque([seed])
    is_oral[seed] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not is_oral[v]:
                vx, vy, vz = co[v]
                # Stay strictly inside internal mouth bag (behind lip roll line!)
                lip_roll_y = -0.1525 + 18.0 * (vx * vx)
                if 1.455 < vz < 1.520 and abs(vx) < 0.032 and vy > lip_roll_y:
                    is_oral[v] = True
                    q.append(v)
    return is_oral


def _gauss(val, center, sigma):
    return np.exp(-(((val - center) / sigma) ** 2))


def _gauss3(pts, cx, cy, cz, sx, sy, sz):
    return np.exp(
        -(((pts[:, 0] - cx) / sx) ** 2 + ((pts[:, 1] - cy) / sy) ** 2 + ((pts[:, 2] - cz) / sz) ** 2)
    )


def _smooth_vertex_scalar(mesh, values, iterations=2):
    n = len(mesh.vertices)
    adj = [[] for _ in range(n)]
    for e in mesh.edges:
        a, b = e.vertices[0], e.vertices[1]
        adj[a].append(b)
        adj[b].append(a)
    cur = values.copy()
    for _ in range(iterations):
        nxt = cur.copy()
        for i in range(n):
            if adj[i]:
                nxt[i] = 0.5 * cur[i] + 0.5 * np.mean(cur[adj[i]])
        cur = nxt
    return cur


def sculpt_head_and_face(obj):
    """
    Sculpt attractive, natural young East Asian feminine portrait proportions matching reference.png:
    - Compact youthful facial thirds, soft heart-oval cheek fullness, delicate chin & slender neck
    - Expressive warm almond eyes with subtle double-eyelid fold & aegyo-sal hugging 13.5mm eyeballs
    - Small integrated nose with delicate bridge and soft rounded tip
    - Warm closed-mouth smile with lifted corners and hydrated rose-coral lips
    """
    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty((n, 3), dtype=np.float64)
    mesh.vertices.foreach_get("co", co.ravel())

    orig = co.copy()
    is_oral = _find_oral_cavity_mask(mesh, orig)

    x = orig[:, 0]
    y = orig[:, 1]
    z = orig[:, 2]
    ax = np.abs(x)
    sx = np.sign(x)
    is_deep_socket = (z > 1.532) & (z < 1.560) & (ax > 0.014) & (ax < 0.048) & (y > -0.128)

    # 1. Balanced, compact youthful head & face proportions (avoid elongated midface/skull, keep natural eye IPD!)
    above_jaw = np.clip((z - 1.430) / 0.045, 0.0, 1.0)
    co[:, 2] -= 0.045 * (z - 1.525) * above_jaw
    # Add soft lateral cheek/cranial fullness ONLY outside the eye sockets (|x| > 0.040) so eye IPD stays natural!
    cheek_wide = np.clip((ax - 0.036) / 0.032, 0.0, 1.0) * _gauss(z, 1.528, 0.055)
    co[:, 0] += sx * 0.0028 * cheek_wide

    # Lift lower jaw/chin slightly for compact feminine lower third
    front_head = np.clip((-0.015 - y) / 0.085, 0.0, 1.0)
    above_neck = np.clip((z - 1.405) / 0.032, 0.0, 1.0)
    below_mid = np.clip((1.525 - z) / 0.085, 0.0, 1.0)
    co[:, 2] += 0.0058 * front_head * above_neck * below_mid

    # Smooth V-oval taper on square mandibular angle (gonion) ONLY below cheeks (z ~ 1.462)
    gonion_z = _gauss(z, 1.462, 0.028) * np.clip((-0.01 - y) / 0.09 + 0.35, 0.0, 1.0)
    gonion_x = np.clip((ax - 0.024) / 0.042, 0.0, 1.0)
    co[:, 0] -= sx * 0.0072 * gonion_z * gonion_x

    # Soft petite chin & clean submental tuck
    chin_tip = _gauss3(orig, 0.0, -0.150, 1.448, 0.022, 0.022, 0.015)
    co[:, 0] *= (1.0 - 0.14 * chin_tip)
    co[:, 1] -= 0.0010 * chin_tip
    under_chin = _gauss3(orig, 0.0, -0.132, 1.434, 0.028, 0.025, 0.012)
    co[:, 2] += 0.0028 * under_chin
    co[:, 1] += 0.0018 * under_chin

    # 2. Smooth feminine forehead & soft radiant apple cheeks (matching reference.png!)
    brow_boss = _gauss(z, 1.564, 0.012) * _gauss(ax, 0.024, 0.028) * np.clip((-0.132 - y) / 0.020, 0.0, 1.0)
    co[:, 1] += 0.0021 * brow_boss
    forehead_round = _gauss(z, 1.602, 0.026) * _gauss(ax, 0.0, 0.055) * np.clip((-0.115 - y) / 0.035, 0.0, 1.0)
    co[:, 1] -= 0.0018 * forehead_round

    malar = (
        _gauss3(orig, 0.036, -0.138, 1.518, 0.024, 0.022, 0.021)
        + _gauss3(orig, -0.036, -0.138, 1.518, 0.024, 0.022, 0.021)
    )
    co[:, 1] -= 0.0036 * malar
    co[:, 0] += sx * 0.0014 * malar
    co[:, 2] += 0.0014 * malar

    # 3. Slender graceful neck
    neck_band = _gauss(z, 1.385, 0.048)
    neck_cy = -0.018
    co[:, 0] -= x * 0.140 * neck_band
    co[:, 1] -= (y - neck_cy) * 0.130 * neck_band

    ear_mask = np.clip((ax - 0.070) / 0.016, 0.0, 1.0) * _gauss(z, 1.534, 0.038)
    co[:, 0] -= sx * 0.0040 * ear_mask
    co[:, 2] += 0.0015 * ear_mask

    # 4. Small, delicate feminine nose & nostrils
    bridge = _gauss(z, 1.535, 0.016) * _gauss(ax, 0.0, 0.014) * np.clip((-0.142 - y) / 0.016, 0.0, 1.0)
    co[:, 1] += 0.0018 * bridge
    co[:, 0] *= (1.0 - 0.14 * bridge)

    alar = _gauss(z, 1.506, 0.011) * np.clip((ax - 0.005) / 0.013, 0.0, 1.0) * np.clip((-0.145 - y) / 0.016, 0.0, 1.0)
    co[:, 0] -= sx * 0.0022 * alar
    ntip = _gauss3(orig, 0.0, -0.1675, 1.5085, 0.012, 0.012, 0.010)
    co[:, 1] += 0.0018 * ntip
    co[:, 2] += 0.0014 * ntip

    # 5. Expressive warm almond eyes, double-eyelid crease & subtle aegyo-sal (natural 60.4mm IPD!)
    eye_L = np.array([0.0302, -0.1256, 1.5443], dtype=np.float64)
    eye_R = np.array([-0.0302, -0.1256, 1.5443], dtype=np.float64)
    R_eye = 0.0136

    acx = np.abs(co[:, 0])
    scx = np.sign(co[:, 0])
    eye_front = np.clip((-0.131 - co[:, 1]) / 0.009, 0.0, 1.0) * (1.0 - is_deep_socket.astype(np.float64))

    # Open upper eyelid gracefully while keeping lower eyelid in a soft, warm smiling cradle (no lower sclera stare!)
    u_arch_x = _gauss(acx, 0.0298, 0.0105) * np.clip((acx - 0.0168) / 0.004, 0.0, 1.0) * np.clip((0.0472 - acx) / 0.005, 0.0, 1.0)
    u_lid = u_arch_x * _gauss(co[:, 2], 1.5490, 0.0034) * eye_front
    co[:, 2] += 0.0016 * u_lid

    l_arch_x = _gauss(acx, 0.0310, 0.0105) * np.clip((acx - 0.0182) / 0.004, 0.0, 1.0) * np.clip((0.0472 - acx) / 0.005, 0.0, 1.0)
    l_lid = l_arch_x * _gauss(co[:, 2], 1.5405, 0.0030) * eye_front
    co[:, 2] += 0.0004 * l_lid  # subtle Duchenne smile lift of lower eyelid!

    # Widen outer eye corner gently for graceful almond width
    lat_canthus = (
        _gauss3(co, 0.0445, -0.136, 1.5445, 0.0065, 0.0065, 0.0050)
        + _gauss3(co, -0.0445, -0.136, 1.5445, 0.0065, 0.0065, 0.0050)
    )
    co[:, 0] += scx * 0.0016 * lat_canthus
    co[:, 2] += 0.0006 * lat_canthus

    # Delicate double-eyelid supratarsal crease
    crease_z_target = 1.5532 - 0.0025 * (((acx - 0.0298) / 0.015) ** 2)
    crease_span_x = np.clip((acx - 0.0172) / 0.004, 0.0, 1.0) * np.clip((0.0468 - acx) / 0.005, 0.0, 1.0)
    crease_groove = _gauss(co[:, 2] - crease_z_target, 0.0, 0.00085) * crease_span_x * eye_front
    fold_overhang = _gauss(co[:, 2] - (crease_z_target + 0.0013), 0.0, 0.0011) * crease_span_x * eye_front
    co[:, 1] += 0.00095 * crease_groove - 0.00055 * fold_overhang

    # Subtle youthful aegyo-sal fullness beneath lower lashes
    aegyo_z_target = 1.5364 + 0.0016 * (((acx - 0.0315) / 0.014) ** 2)
    aegyo_span_x = np.clip((acx - 0.0185) / 0.004, 0.0, 1.0) * np.clip((0.046 - acx) / 0.005, 0.0, 1.0)
    aegyo_puff = _gauss(co[:, 2] - aegyo_z_target, 0.0, 0.0016) * aegyo_span_x * eye_front
    aegyo_line = _gauss(co[:, 2] - (aegyo_z_target - 0.0024), 0.0, 0.0009) * aegyo_span_x * eye_front
    co[:, 1] -= 0.00095 * aegyo_puff
    co[:, 1] += 0.00038 * aegyo_line

    # Wrap inner eyelids tightly around the 13.6mm spherical eyeballs
    for ec in (eye_L, eye_R):
        sign_side = 1.0 if ec[0] > 0 else -1.0
        in_side = (co[:, 0] * sign_side) > 0.012
        dvec = co - ec[None, :]
        dist = np.linalg.norm(dvec, axis=1)
        aperture_w = (
            in_side.astype(np.float64)
            * _gauss(co[:, 0], ec[0], 0.0155)
            * _gauss(co[:, 2], ec[2], 0.0088)
            * np.clip((-0.1325 - co[:, 1]) / 0.0055, 0.0, 1.0)
        )
        min_r = R_eye + 0.00020
        need_push = (dist < min_r) & (aperture_w > 0.05)
        if np.any(need_push):
            unit = dvec[need_push] / np.maximum(dist[need_push, None], 1e-6)
            target_pt = ec[None, :] + unit * min_r
            blend = np.clip(aperture_w[need_push] * 1.25, 0.0, 1.0)[:, None]
            co[need_push] = co[need_push] * (1.0 - blend) + target_pt * blend

    # 6. Warm Closed-Mouth Smile (matching reference.png's sweet expression!)
    mouth_base_z = 1.4828
    smile_corner = (
        _gauss3(co, 0.0268, -0.1415, mouth_base_z, 0.0145, 0.0160, 0.0130)
        + _gauss3(co, -0.0268, -0.1415, mouth_base_z, 0.0145, 0.0160, 0.0130)
    )
    co[:, 2] += 0.0042 * smile_corner
    co[:, 0] += scx * 0.0022 * smile_corner
    co[:, 1] += 0.0016 * smile_corner

    smile_cheek = (
        _gauss3(co, 0.0365, -0.1375, 1.4935, 0.0160, 0.0150, 0.0145)
        + _gauss3(co, -0.0365, -0.1375, 1.4935, 0.0160, 0.0150, 0.0145)
    )
    co[:, 1] -= 0.0016 * smile_cheek
    co[:, 2] += 0.0018 * smile_cheek

    outer_lip = (~is_oral).astype(np.float64) * np.clip((-0.143 - co[:, 1]) / 0.008, 0.0, 1.0)
    cupid = _gauss(acx, 0.0055, 0.0040) * _gauss(co[:, 2], 1.4900, 0.0026) * outer_lip
    philtrum_dip = _gauss(acx, 0.0, 0.0026) * _gauss(co[:, 2], 1.4905, 0.0022) * outer_lip
    co[:, 2] += 0.00070 * cupid - 0.00045 * philtrum_dip
    co[:, 1] -= 0.00075 * cupid

    lower_plump = _gauss(acx, 0.0, 0.0145) * _gauss(co[:, 2], 1.4775, 0.0035) * outer_lip
    co[:, 1] -= 0.0013 * lower_plump

    # Keep true internal mouth bag safely behind lips
    if np.any(is_oral):
        co[is_oral, 1] = np.maximum(co[is_oral, 1], -0.1460)

    # 7. Paint clean, seamless 3D facial shading attributes (ZERO cheek smudges!)
    acx = np.abs(co[:, 0])
    smile_z_shift = 0.0042 * np.clip(acx / 0.0275, 0.0, 1.2) ** 1.75
    lip_center_z = 1.4836 + smile_z_shift
    cupid_top = 0.0011 * _gauss(acx, 0.0056, 0.0042) - 0.0007 * _gauss(acx, 0.0, 0.0026)
    upper_border = lip_center_z + 0.0068 * np.clip(1.0 - (acx / 0.0290) ** 1.65, 0.0, 1.0) + cupid_top
    lower_border = lip_center_z - 0.0080 * np.clip(1.0 - (acx / 0.0280) ** 1.55, 0.0, 1.0)

    in_lip_x = np.clip((0.0292 - acx) / 0.0038, 0.0, 1.0)
    in_lip_u = np.clip((upper_border - co[:, 2]) / 0.0022, 0.0, 1.0)
    in_lip_l = np.clip((co[:, 2] - lower_border) / 0.0026, 0.0, 1.0)
    # Outer surface curve check so only outer lip vermilion gets lip tint
    cheek_curve_y = -0.155 + 18.0 * (acx ** 2)
    in_lip_front = np.clip((cheek_curve_y + 0.0055 - co[:, 1]) / 0.0045, 0.0, 1.0) * (~is_oral).astype(np.float64)
    lip_mask = _smooth_vertex_scalar(mesh, in_lip_x * in_lip_u * in_lip_l * in_lip_front, iterations=2)

    # Warm rosy blush on apple cheeks, nose tip, ears, and subtle eyelid warmth
    cheek_blush = (
        0.46 * _gauss3(co, 0.039, -0.138, 1.517, 0.025, 0.023, 0.021)
        + 0.46 * _gauss3(co, -0.039, -0.138, 1.517, 0.025, 0.023, 0.021)
    )
    nose_warmth = 0.25 * _gauss3(co, 0.0, -0.164, 1.509, 0.015, 0.014, 0.012)
    ear_warmth = 0.34 * np.clip((acx - 0.068) / 0.012, 0.0, 1.0) * _gauss(co[:, 2], 1.535, 0.036)
    waterline_pink = 0.35 * (
        _gauss3(co, 0.0195, -0.139, 1.5435, 0.004, 0.004, 0.004)
        + _gauss3(co, -0.0195, -0.139, 1.5435, 0.004, 0.004, 0.004)
    )
    blush_mask = _smooth_vertex_scalar(
        mesh, np.clip(cheek_blush + nose_warmth + ear_warmth + waterline_pink, 0.0, 1.0), iterations=2
    )

    # Subtle dark espresso upper-eyelid lash-line tightline (gives warm, defined eyes!)
    u_lash_z = 1.5495 + 0.0020 * _gauss(acx, 0.0298, 0.0105) - 0.0012 * (((acx - 0.0298) / 0.015) ** 2)
    lashline_raw = (
        _gauss(co[:, 2] - u_lash_z, 0.0, 0.0012)
        * np.clip((acx - 0.0175) / 0.0035, 0.0, 1.0)
        * np.clip((0.0462 - acx) / 0.0035, 0.0, 1.0)
        * np.clip((-0.1365 - co[:, 1]) / 0.0035, 0.0, 1.0)
    )
    lashline_mask = _smooth_vertex_scalar(mesh, 0.65 * np.clip(lashline_raw, 0.0, 1.0), iterations=1)

    # Soft feathered eyebrow skin under-shading (subtle & delicate!)
    bu_norm = np.clip((acx - 0.0155) / 0.0345, 0.0, 1.0)
    brow_arch_z = 1.5582 + 0.0014 * np.sin(math.pi * (bu_norm ** 0.95)) - 0.0008 * (bu_norm ** 2.4)
    brow_raw = (
        _gauss(co[:, 2] - brow_arch_z, 0.0, 0.0022)
        * np.clip((acx - 0.0155) / 0.006, 0.0, 1.0)
        * np.clip((0.0495 - acx) / 0.008, 0.0, 1.0)
        * np.clip((-0.134 - co[:, 1]) / 0.010, 0.0, 1.0)
    )
    brow_mask = _smooth_vertex_scalar(mesh, 0.22 * np.clip(brow_raw, 0.0, 1.0), iterations=2)

    # Clean cavity AO ONLY inside the two small nostril openings and thin lip contact line!
    nostril_ao = 0.82 * (
        _gauss3(co, 0.0092, -0.1580, 1.5052, 0.0044, 0.0042, 0.0028)
        + _gauss3(co, -0.0092, -0.1580, 1.5052, 0.0044, 0.0042, 0.0028)
    )
    mouth_slit_ao = 0.34 * _gauss(co[:, 2] - (lip_center_z - 0.0015), 0.0, 0.00065) * in_lip_x * in_lip_front
    cavity_ao = np.clip(nostril_ao + mouth_slit_ao, 0.0, 1.0)

    # Soft scalp root shadow strictly behind y > -0.105 (never on forehead, temples, or ears!)
    crown_top = np.clip((co[:, 2] - 1.615) / 0.020, 0.0, 1.0) * np.clip((co[:, 1] + 0.105) / 0.025, 0.0, 1.0)
    occiput_back = np.clip((co[:, 1] + 0.025) / 0.028, 0.0, 1.0) * np.clip((co[:, 2] - 1.445) / 0.035, 0.0, 1.0) * np.clip((0.072 - acx) / 0.012, 0.0, 1.0)
    scalp_mask = _smooth_vertex_scalar(
        mesh, 0.62 * np.clip(np.maximum(crown_top, occiput_back), 0.0, 1.0), iterations=2
    )

    attr_rc = mesh.attributes.new(name="rest_co", type='FLOAT_VECTOR', domain='POINT')
    attr_rc.data.foreach_set("vector", co.astype(np.float32).ravel())

    for aname, arr in [
        ("lip_mask", lip_mask),
        ("blush_mask", blush_mask),
        ("cavity_ao", cavity_ao),
        ("scalp_mask", scalp_mask),
        ("lashline_mask", lashline_mask),
        ("brow_mask", brow_mask),
    ]:
        at = mesh.attributes.new(name=aname, type='FLOAT', domain='POINT')
        at.data.foreach_set("value", arr.astype(np.float32))

    rest_sculpted = co.copy()
    posed_co = pose_points_array(co, rigid_head=False)

    # Guarantee posed neck base below Z=1.382 sits cleanly inside the crew-neck collar ring (zero collar-corner poke-through!)
    tuck_w = np.clip((1.382 - posed_co[:, 2]) / 0.052, 0.0, 1.0) ** 1.20
    posed_co[:, 0] = posed_co[:, 0] * (1.0 - 0.56 * tuck_w)
    posed_co[:, 1] = -0.026 + (posed_co[:, 1] + 0.026) * (1.0 - 0.56 * tuck_w)

    mesh.vertices.foreach_set("co", posed_co.astype(np.float32).ravel())
    mesh.update()

    for poly in mesh.polygons:
        poly.use_smooth = True
    sub_final = obj.modifiers.new(name="SkinSubsurf", type='SUBSURF')
    sub_final.levels = 1
    sub_final.render_levels = 1
    sub_final.uv_smooth = 'PRESERVE_BOUNDARIES'

    return {
        "eye_L_rest": eye_L,
        "eye_R_rest": eye_R,
        "eye_radius": R_eye,
        "rest_verts": rest_sculpted,
        "posed_verts": posed_co,
    }


def _create_single_eyeball(side_tag, center_rest, radius, target_world, mat_outer, mat_iris, mat_tear):
    """
    Build a realistic 3-part anatomical human eye with 12.7mm warm chestnut-brown iris diameter
    (iris_r = 6.35mm, matching East Asian youthful portrait proportions in input/reference.png).
    """
    center_posed = pose_point(center_rest, neck_weight=1.0)

    head_up = _head_rotation_matrix(1.0) @ np.array([0.0, 0.0, 1.0])
    fwd = np.asarray(target_world, dtype=np.float64) - center_posed
    fwd /= np.linalg.norm(fwd)
    y_ax = -fwd
    x_ax = np.cross(y_ax, head_up)
    x_ax /= np.linalg.norm(x_ax)
    z_ax = np.cross(x_ax, y_ax)
    R_eye = np.column_stack([x_ax, y_ax, z_ax])

    bm_out = bmesh.new()
    bmesh.ops.create_uvsphere(bm_out, u_segments=48, v_segments=32, radius=radius)
    bmesh.ops.rotate(bm_out, verts=bm_out.verts, matrix=Matrix.Rotation(math.pi * 0.5, 3, 'X'))

    iris_r = 0.00652
    cornea_masks = []
    for v in bm_out.verts:
        vx, vy, vz = v.co.x, v.co.y, v.co.z
        rad_xz = math.hypot(vx, vz)
        if vy < 0.0 and rad_xz < iris_r * 1.05:
            norm_r = rad_xz / (iris_r * 1.05)
            bulge = 0.00085 * ((1.0 - norm_r * norm_r) ** 1.35)
            v.co.y -= bulge
            cm = float(np.clip((1.02 - norm_r) / 0.11, 0.0, 1.0))
        else:
            cm = 0.0
        cornea_masks.append(cm)
        p_world = center_posed + R_eye @ np.array([v.co.x, v.co.y, v.co.z])
        v.co = Vector(p_world)

    mesh_out = bpy.data.meshes.new(f"Eye_Outer_{side_tag}_Mesh")
    bm_out.to_mesh(mesh_out)
    bm_out.free()
    for p in mesh_out.polygons:
        p.use_smooth = True
    attr_cm = mesh_out.attributes.new(name="cornea_mask", type='FLOAT', domain='POINT')
    attr_cm.data.foreach_set("value", np.array(cornea_masks, dtype=np.float32))
    obj_out = bpy.data.objects.new(f"Eye_Outer_{side_tag}", mesh_out)
    obj_out.data.materials.append(mat_outer)

    # Inner Concave Iris + Pupil Disc
    bm_ir = bmesh.new()
    rings = 16
    segs = 48
    y_limbus = -math.sqrt(max(1e-8, radius * radius - iris_r * iris_r)) + 0.00012
    ring_verts = []
    for r_i in range(rings + 1):
        frac = r_i / rings
        rr = frac * iris_r * 1.02
        yy = y_limbus + 0.00072 * (1.0 - frac * frac)
        row = []
        for s_i in range(segs):
            ang = 2.0 * math.pi * s_i / segs
            lx = rr * math.cos(ang)
            lz = rr * math.sin(ang)
            pw = center_posed + R_eye @ np.array([lx, yy, lz])
            vert = bm_ir.verts.new(pw)
            row.append((vert, frac * math.cos(ang), frac * math.sin(ang)))
        ring_verts.append(row)

    uv_lay = bm_ir.loops.layers.uv.new("UVMap")
    for r_i in range(rings):
        for s_i in range(segs):
            s_n = (s_i + 1) % segs
            v00, u00, w00 = ring_verts[r_i][s_i]
            v01, u01, w01 = ring_verts[r_i][s_n]
            v11, u11, w11 = ring_verts[r_i + 1][s_n]
            v10, u10, w10 = ring_verts[r_i + 1][s_i]
            if r_i == 0:
                f = bm_ir.faces.new((v00, v10, v11))
                uvs = [(u00, w00), (u10, w10), (u11, w11)]
            else:
                f = bm_ir.faces.new((v00, v10, v11, v01))
                uvs = [(u00, w00), (u10, w10), (u11, w11), (u01, w01)]
            f.smooth = True
            for loop, (uu, ww) in zip(f.loops, uvs):
                loop[uv_lay].uv = (0.5 + 0.49 * uu, 0.5 + 0.49 * ww)

    mesh_ir = bpy.data.meshes.new(f"Eye_Iris_{side_tag}_Mesh")
    bm_ir.to_mesh(mesh_ir)
    bm_ir.free()
    obj_ir = bpy.data.objects.new(f"Eye_Iris_{side_tag}", mesh_ir)
    obj_ir.data.materials.append(mat_iris)

    # Wet Lower-Lid Tear Meniscus
    bm_tm = bmesh.new()
    n_tm = 26
    tm_rows = []
    for k in range(2):
        row = []
        for i in range(n_tm):
            t = (i / (n_tm - 1)) * 2.0 - 1.0
            ang_x = t * 0.80
            lx = (radius + 0.00018 + 0.00035 * k) * math.sin(ang_x)
            lz = -0.0048 + 0.0014 * (t * t) - 0.00045 * k
            ly = -math.sqrt(max(1e-6, (radius + 0.00025) ** 2 - lx * lx - lz * lz)) - 0.00015 * k
            pw = center_posed + _head_rotation_matrix(1.0) @ np.array([lx, ly, lz])
            row.append(bm_tm.verts.new(pw))
        tm_rows.append(row)
    for i in range(n_tm - 1):
        f = bm_tm.faces.new((tm_rows[0][i], tm_rows[0][i + 1], tm_rows[1][i + 1], tm_rows[1][i]))
        f.smooth = True
    mesh_tm = bpy.data.meshes.new(f"Tear_Meniscus_{side_tag}_Mesh")
    bm_tm.to_mesh(mesh_tm)
    bm_tm.free()
    obj_tm = bpy.data.objects.new(f"Tear_Meniscus_{side_tag}", mesh_tm)
    obj_tm.data.materials.append(mat_tear)

    return [obj_out, obj_ir, obj_tm]


def _make_hair_curves_object(name, curves_pts_list, curves_radii_list, material):
    hc = bpy.data.hair_curves.new(name + "_Data")
    lengths = [len(pts) for pts in curves_pts_list]
    hc.add_curves(lengths)
    flat_co = np.concatenate(curves_pts_list, axis=0).astype(np.float32).ravel()
    flat_rad = np.concatenate(curves_radii_list, axis=0).astype(np.float32).ravel()
    hc.position_data.foreach_set("vector", flat_co)
    hc.points.foreach_set("radius", flat_rad)
    hc.set_types(type='CATMULL_ROM')
    hc.materials.append(material)
    return bpy.data.objects.new(name, hc)


def build_brows_and_lashes(anat_info, head_obj, mat_brow, mat_lash):
    """
    Create soft, natural straight-arch feminine eyebrows snapped directly to the evaluated
    subdivided brow skin via BVHTree, plus lush curved upper & lower eyelashes.
    """
    from mathutils.bvhtree import BVHTree

    deps = bpy.context.evaluated_depsgraph_get()
    head_bvh = BVHTree.FromObject(head_obj, deps)

    rng = np.random.default_rng(303)
    eye_L = anat_info["eye_L_rest"]
    R_eye = anat_info["eye_radius"]

    # 1. Soft, Straighter, Natural Feminine Eyebrows (ray-cast strictly along head +Y!)
    ray_dir_w = Vector(_head_rotation_matrix(1.0) @ np.array([0.0, 1.0, 0.0]))
    brow_pts, brow_rad = [], []
    for sign_s in (1.0, -1.0):
        for i in range(540):
            u = rng.beta(1.2, 1.3)
            bx = sign_s * (0.0158 + u * 0.0335)
            # Soft, gentle East Asian straight-arch brow matching reference.png!
            arch = 0.0014 * math.sin(math.pi * (u ** 0.95)) - 0.0008 * (u ** 2.4)
            half_thick = 0.0028 * (1.0 - 0.45 * (u ** 1.7)) * (0.65 + 0.35 * math.sin(math.pi * min(1.0, u * 3.2)))
            v_off = rng.triangular(-1.0, 0.0, 1.0) * half_thick
            bz = 1.5582 + arch + v_off

            ang = math.radians(32.0 * (1.0 - u) ** 1.8 - 8.0 * (u ** 1.5) - 6.0 * (v_off / max(1e-4, half_thick)))
            hlen = rng.uniform(0.0050, 0.0072) * (0.85 + 0.20 * math.sin(math.pi * u))
            n_seg = 6
            pts_w = np.zeros((n_seg, 3), dtype=np.float64)
            valid = True
            for k in range(n_seg):
                t = k / (n_seg - 1)
                px = bx + sign_s * math.cos(ang) * hlen * t
                pz = bz + math.sin(ang) * hlen * t
                orig_w = Vector(pose_point([px, -0.220, pz], neck_weight=1.0))
                loc, nrm, _, _ = head_bvh.ray_cast(orig_w, ray_dir_w)
                if loc is None:
                    valid = False
                    break
                loft = 0.00010 + 0.00055 * math.sin(math.pi * (t ** 0.65)) + 0.00012 * t
                p_surf = loc + nrm * loft
                pts_w[k] = [p_surf.x, p_surf.y, p_surf.z]
            if valid:
                brow_pts.append(pts_w)
                r0 = rng.uniform(0.000038, 0.000054)
                brow_rad.append(r0 * np.array([0.80, 1.0, 0.90, 0.70, 0.40, 0.08]))

    obj_brows = _make_hair_curves_object("Character_Eyebrows", brow_pts, brow_rad, mat_brow)

    # 2. Lush, Delicate Curved Eyelashes (aligned with eye_L = 0.0302)
    lash_pts, lash_rad = [], []
    for sign_s in (1.0, -1.0):
        ec = np.array([sign_s * eye_L[0], eye_L[1], eye_L[2]], dtype=np.float64)
        n_up = 150
        clump_centers = np.linspace(0.05, 0.96, 22)
        for i in range(n_up):
            u = rng.uniform(0.03, 0.98)
            nearest_c = clump_centers[np.argmin(np.abs(clump_centers - u))]
            lx = sign_s * (0.0180 + u * 0.0276)
            u_arch = math.exp(-(((abs(lx) - 0.0298) / 0.0105) ** 2))
            lz = 1.5468 + 0.0039 * u_arch - 0.0011 * ((u - 0.45) ** 2) + rng.uniform(-0.00016, 0.00016)
            dx = lx - ec[0]
            dz = lz - ec[2]
            r_lid = R_eye + 0.00085
            dy = -math.sqrt(max(1e-6, r_lid * r_lid - dx * dx - dz * dz))
            root = np.array([lx, ec[1] + dy, lz])

            llen = (0.0042 + 0.0044 * (math.sin(math.pi * (u ** 0.72)) ** 0.8)) * rng.uniform(0.88, 1.12)
            splay_x = sign_s * (0.20 * (u - 0.30) + 0.42 * (nearest_c - u))
            n_seg = 7
            pts = np.zeros((n_seg, 3), dtype=np.float64)
            for k in range(n_seg):
                t = k / (n_seg - 1)
                fwd_d = llen * (0.90 * t)
                curl_z = llen * (-0.08 * t + 0.68 * (t ** 2.1))
                lat_x = llen * splay_x * (t ** 1.25)
                pts[k] = root + np.array([lat_x, -fwd_d, curl_z])
            lash_pts.append(pose_points_array(pts, rigid_head=True))
            r0 = rng.uniform(0.000088, 0.000118)
            lash_rad.append(r0 * np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.20, 0.04]))

        for i in range(56):
            u = rng.uniform(0.08, 0.96)
            lx = sign_s * (0.0195 + u * 0.0252)
            l_arch = math.exp(-(((abs(lx) - 0.0318) / 0.0105) ** 2))
            lz = 1.5408 - 0.0026 * l_arch + 0.0010 * ((u - 0.5) ** 2)
            dx = lx - ec[0]
            dz = lz - ec[2]
            r_lid = R_eye + 0.00082
            dy = -math.sqrt(max(1e-6, r_lid * r_lid - dx * dx - dz * dz))
            root = np.array([lx, ec[1] + dy, lz])
            llen = rng.uniform(0.0024, 0.0038) * math.sin(math.pi * (0.12 + 0.78 * u))
            n_seg = 5
            pts = np.zeros((n_seg, 3), dtype=np.float64)
            for k in range(n_seg):
                t = k / (n_seg - 1)
                pts[k] = root + np.array([sign_s * llen * 0.22 * (u - 0.35) * t, -llen * 0.70 * t, -llen * (0.35 * t + 0.40 * t * t)])
            lash_pts.append(pose_points_array(pts, rigid_head=True))
            r0 = rng.uniform(0.000050, 0.000068)
            lash_rad.append(r0 * np.array([1.0, 0.85, 0.60, 0.32, 0.06]))

    obj_lashes = _make_hair_curves_object("Character_Eyelashes", lash_pts, lash_rad, mat_lash)
    return obj_brows, obj_lashes


def build_character_anatomy(work_dir, char_col, mat_skin, mat_eye_out, mat_eye_iris, mat_tear, mat_brow, mat_lash, gaze_target):
    head_obj = _load_upper_body_with_uvs(work_dir)
    for col in list(head_obj.users_collection):
        col.objects.unlink(head_obj)
    char_col.objects.link(head_obj)
    head_obj.data.materials.append(mat_skin)

    anat_info = sculpt_head_and_face(head_obj)

    eyes_L = _create_single_eyeball(
        "L", anat_info["eye_L_rest"], anat_info["eye_radius"], gaze_target, mat_eye_out, mat_eye_iris, mat_tear
    )
    eyes_R = _create_single_eyeball(
        "R", anat_info["eye_R_rest"], anat_info["eye_radius"], gaze_target, mat_eye_out, mat_eye_iris, mat_tear
    )
    brows_obj, lashes_obj = build_brows_and_lashes(anat_info, head_obj, mat_brow, mat_lash)

    for ob in eyes_L + eyes_R + [brows_obj, lashes_obj]:
        char_col.objects.link(ob)

    anat_info["head_obj"] = head_obj
    anat_info["eye_objs"] = eyes_L + eyes_R
    anat_info["brows_obj"] = brows_obj
    anat_info["lashes_obj"] = lashes_obj
    return anat_info
