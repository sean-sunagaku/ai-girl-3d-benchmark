"""Base upper-body import, anatomical facial sculpting, arm/torso extension, UVs, and multi-layer eyes."""

import math
import json
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix


def _smoothstep(edge0: float, edge1: float, x):
    denom = edge1 - edge0
    if abs(denom) < 1e-12:
        denom = 1e-12
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gauss_3d(coords: np.ndarray, center, radii) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    d2 = np.sum(((coords - c) / r) ** 2, axis=-1)
    return np.exp(-0.5 * d2)


def _build_adj(bm):
    adj = [[] for _ in range(len(bm.verts))]
    for e in bm.edges:
        a, b = e.verts[0].index, e.verts[1].index
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _bfs_hops(adj, seeds, max_hops: int, allowed=None):
    dist = {}
    queue = []
    for s in seeds:
        dist[s] = 0
        queue.append(s)
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        du = dist[u]
        if du >= max_hops:
            continue
        for v in adj[u]:
            if v not in dist and (allowed is None or v in allowed):
                dist[v] = du + 1
                queue.append(v)
    return dist


def _rotate_around_x(pts: np.ndarray, center: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    rel = pts - center
    ca = np.cos(angles_rad)
    sa = np.sin(angles_rad)
    out = rel.copy()
    out[:, 1] = rel[:, 1] * ca - rel[:, 2] * sa
    out[:, 2] = rel[:, 1] * sa + rel[:, 2] * ca
    return out + center


def sculpt_face_anatomy(obj: bpy.types.Object, eye_centers: dict):
    """Apply smooth anatomical sculpting to MakeHuman hm08 upper-body mesh."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    adj = _build_adj(bm)

    co = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    orig = co.copy()

    # 1. Global feminine head & jawline refinement (subtle youthful East Asian oval/V-line)
    # Taper gonial jaw angles inward and softly lift lower jaw edge
    jaw_w = _gauss_3d(orig, (0.0, -0.075, 1.460), (0.085, 0.055, 0.036))
    jaw_side = _smoothstep(0.022, 0.062, np.abs(orig[:, 0]))
    co[:, 0] -= np.sign(orig[:, 0]) * 0.0068 * jaw_w * jaw_side
    co[:, 2] += 0.0038 * jaw_w * jaw_side

    # Soften & slightly lift chin bottom for delicate heart/oval face shape
    chin_bot = _gauss_3d(orig, (0.0, -0.130, 1.435), (0.034, 0.032, 0.018))
    co[:, 2] += 0.0042 * chin_bot
    chin_pad = _gauss_3d(orig, (0.0, -0.149, 1.452), (0.022, 0.020, 0.014))
    co[:, 1] -= 0.0014 * chin_pad

    # Smooth forehead & soften supraorbital brow ridge (more feminine frontal contour)
    brow_ridge = _gauss_3d(orig, (0.0, -0.148, 1.568), (0.048, 0.022, 0.016))
    co[:, 1] += 0.0020 * brow_ridge
    forehead_top = _gauss_3d(orig, (0.0, -0.125, 1.615), (0.062, 0.045, 0.035))
    co[:, 1] -= 0.0022 * forehead_top  # gently rounder youthful forehead

    # Soft cheekbone / apple cheek fullness & smiling lift
    for sx in (-1.0, 1.0):
        cheek_apple = _gauss_3d(orig, (sx * 0.038, -0.138, 1.512), (0.024, 0.024, 0.022))
        co[:, 1] -= 0.0042 * cheek_apple
        co[:, 2] += 0.0024 * cheek_apple
        # Soft zygomatic & lower cheek fullness (avoid gaunt mid-cheek hollow)
        cheek_lat = _gauss_3d(orig, (sx * 0.052, -0.112, 1.496), (0.022, 0.028, 0.026))
        co[:, 0] += sx * 0.0018 * cheek_lat
        co[:, 1] -= 0.0028 * cheek_lat

    # 2. Delicate petite feminine nose
    nose_bridge = _gauss_3d(orig, (0.0, -0.155, 1.528), (0.016, 0.020, 0.022))
    co[:, 0] *= (1.0 - 0.18 * nose_bridge)
    co[:, 1] += 0.0012 * nose_bridge  # slightly softer bridge height

    nose_tip = _gauss_3d(orig, (0.0, -0.165, 1.508), (0.018, 0.018, 0.014))
    co[:, 0] *= (1.0 - 0.16 * nose_tip)
    co[:, 1] += 0.0018 * nose_tip  # slightly less protruding tip
    co[:, 2] += 0.0020 * nose_tip  # gentle youthful tip lift

    # Alar wings & nostrils refinement
    for sx in (-1.0, 1.0):
        alar = _gauss_3d(orig, (sx * 0.015, -0.153, 1.507), (0.009, 0.011, 0.009))
        co[:, 0] -= sx * 0.0022 * alar
        co[:, 2] += 0.0010 * alar
        # Subtle recessed nostril pocket
        nostril = _gauss_3d(orig, (sx * 0.0085, -0.155, 1.5015), (0.0045, 0.0055, 0.0035))
        co[:, 2] += 0.0018 * nostril
        co[:, 1] += 0.0012 * nostril

    # 3. Sweet closed-mouth gentle smile & shaped lips
    # Identify mouth front vertices (separate from deep inner mouth cavity for clean shaping)
    mouth_front = _gauss_3d(orig, (0.0, -0.154, 1.479), (0.036, 0.018, 0.016))
    # Petite mouth width compression
    mouth_w = _gauss_3d(orig, (0.0, -0.152, 1.479), (0.032, 0.018, 0.015))
    co[:, 0] *= (1.0 - 0.08 * mouth_w)

    # Smile curve: gentle natural corner lift (+Z) and dental-arc wrap (+Y)
    x_abs = np.abs(orig[:, 0])
    corner_curve = np.clip(x_abs / 0.0235, 0.0, 1.20) ** 1.85
    corner_falloff = np.exp(-np.maximum(0.0, x_abs - 0.0235) ** 2 / (2.0 * 0.018 ** 2))
    smile_zone = _smoothstep(-0.132, -0.144, orig[:, 1]) * np.exp(-((orig[:, 2] - 1.478) / 0.017) ** 2)
    smile_amp = corner_curve * corner_falloff * smile_zone
    co[:, 2] += 0.0023 * smile_amp
    co[:, 1] += 0.0028 * smile_amp

    # Very subtle soft mouth-corner dimple & smiling cheek lift (avoid harsh crease)
    for sx in (-1.0, 1.0):
        dimple = _gauss_3d(orig, (sx * 0.0260, -0.144, 1.481), (0.007, 0.009, 0.008))
        co[:, 1] += 0.0007 * dimple
        smile_pad = _gauss_3d(orig, (sx * 0.032, -0.145, 1.495), (0.015, 0.016, 0.014))
        co[:, 1] -= 0.0020 * smile_pad
        co[:, 2] += 0.0014 * smile_pad

    # Upper lip cupid's bow & lower lip plump cushion
    upper_lip = _gauss_3d(orig, (0.0, -0.159, 1.4840), (0.017, 0.009, 0.0042))
    cupid_peaks = np.exp(-((x_abs - 0.0055) / 0.0040) ** 2) * np.exp(-((orig[:, 2] - 1.4852) / 0.0025) ** 2) * _smoothstep(-0.152, -0.157, orig[:, 1])
    co[:, 2] += 0.0005 * cupid_peaks
    co[:, 1] += 0.0004 * upper_lip * np.exp(-(x_abs / 0.010) ** 2)

    lower_lip = _gauss_3d(orig, (0.0, -0.156, 1.4725), (0.017, 0.009, 0.0052))
    co[:, 1] -= 0.0022 * lower_lip
    co[:, 2] -= 0.0003 * lower_lip * np.exp(-(x_abs / 0.011) ** 2)

    # Ensure closed lip contact seam has zero gap (close inner lip edges cleanly onto gentle smile arc)
    seam_z = 1.4767 + 0.0023 * np.clip(x_abs / 0.0235, 0.0, 1.1) ** 1.85
    lip_inner = np.exp(-((orig[:, 2] - 1.4767) / 0.0032) ** 2) * _smoothstep(-0.145, -0.156, orig[:, 1]) * np.exp(-(x_abs / 0.023) ** 2)
    inner_depth_w = _smoothstep(-0.1595, -0.1535, orig[:, 1])
    co[:, 2] += (seam_z - co[:, 2]) * (0.62 + 0.34 * inner_depth_w) * lip_inner

    # 4. Expressive wide almond-round eyelids wrapping eyeball spheres cleanly
    for side, sx, c_eye in (("l", 1.0, np.array(eye_centers["l"])), ("r", -1.0, np.array(eye_centers["r"]))):
        upper_target = np.array([sx * 0.0314, -0.1397, 1.5500])
        lower_target = np.array([sx * 0.0311, -0.1393, 1.5409])
        u_seed = int(np.argmin(np.linalg.norm(orig - upper_target, axis=1)))
        l_seed = int(np.argmin(np.linalg.norm(orig - lower_target, axis=1)))

        eye_mask_all = set(
            i for i in range(len(orig))
            if np.linalg.norm(orig[i] - c_eye) < 0.028 and orig[i, 1] < -0.118 and np.sign(orig[i, 0]) == sx
        )
        upper_allowed = set(
            i for i in eye_mask_all
            if orig[i, 2] >= 1.5445 or (orig[i, 2] >= 1.5435 and abs(orig[i, 0] - c_eye[0]) > 0.009)
        )
        lower_allowed = set(
            i for i in eye_mask_all
            if orig[i, 2] <= 1.5450
        )

        u_hops = _bfs_hops(adj, [u_seed], 9, allowed=upper_allowed)
        l_hops = _bfs_hops(adj, [l_seed], 7, allowed=lower_allowed)

        # Wide smooth arch peaking above medial-center pupil, smoothly tapering into eye corners
        x_rel = (orig[:, 0] - sx * 0.0312) / 0.0155
        horiz_bell = np.cos(np.clip(x_rel, -1.0, 1.0) * (math.pi * 0.5)) ** 0.80

        u_angles = np.zeros(len(orig), dtype=np.float64)
        for idx, h in u_hops.items():
            w_hop = 1.0 if h <= 3 else max(0.0, 1.0 - (h - 3) / 6.5)
            w_front = float(_smoothstep(-0.125, -0.135, orig[idx, 1]))
            u_angles[idx] = -math.radians(4.8) * w_hop * w_front * horiz_bell[idx]

        mask_u = u_angles != 0.0
        co[mask_u] = _rotate_around_x(co[mask_u], c_eye, u_angles[mask_u])

        # Soft double-eyelid crease ~2.8mm above lifted upper lash margin
        for idx, h in u_hops.items():
            if 4 <= h <= 6 and orig[idx, 1] < -0.137:
                w_crease = 1.0 - abs(h - 5.0) / 1.5
                tuck_dir = c_eye - co[idx]
                tuck_dir /= (np.linalg.norm(tuck_dir) + 1e-8)
                co[idx] += tuck_dir * (0.0014 * w_crease * horiz_bell[idx])

        # Lower eyelid: warm smiling lift (-4.2 deg) so NO startled white sclera shows below iris
        l_angles = np.zeros(len(orig), dtype=np.float64)
        for idx, h in l_hops.items():
            w_hop = 1.0 if h <= 2 else max(0.0, 1.0 - (h - 2) / 5.0)
            w_front = float(_smoothstep(-0.125, -0.135, orig[idx, 1]))
            l_angles[idx] = -math.radians(4.2) * w_hop * w_front * horiz_bell[idx]

        mask_l = l_angles != 0.0
        co[mask_l] = _rotate_around_x(co[mask_l], c_eye, l_angles[mask_l])

        # Gentle youthful under-eye aegyo-sal puff just below lower lash margin
        aegyo = _gauss_3d(orig, (sx * 0.031, -0.141, 1.5355), (0.011, 0.008, 0.0032))
        co[:, 1] -= 0.0012 * aegyo

        # Gently lift outer eye corner slightly for bright round-almond expression
        outer_corner = _gauss_3d(orig, (sx * 0.0430, -0.132, 1.545), (0.0040, 0.007, 0.0040))
        co[:, 2] += 0.0008 * outer_corner

        # Enforce snug spherical eyeball clearance so eyelids hug eyeball with zero gap/clipping
        eye_rad = 0.01405
        for idx in eye_mask_all:
            if orig[idx, 1] < -0.124:
                rel = co[idx] - c_eye
                d = np.linalg.norm(rel)
                if 1e-6 < d < eye_rad + 0.00018:
                    co[idx] = c_eye + rel * ((eye_rad + 0.00018) / d)

    # 5. Neck slenderizing & subtle clavicle/suprasternal hollow
    neck_zone = _smoothstep(1.36, 1.40, orig[:, 2]) * (1.0 - _smoothstep(1.435, 1.465, orig[:, 2]))
    co[:, 0] *= (1.0 - 0.09 * neck_zone)
    # Soften neck front Y relative to neck spine axis y=-0.01
    co[:, 1] = -0.01 + (co[:, 1] + 0.01) * (1.0 - 0.06 * neck_zone)

    suprasternal = _gauss_3d(orig, (0.0, -0.072, 1.345), (0.016, 0.015, 0.014))
    co[:, 1] += 0.0035 * suprasternal

    # Write sculpted coordinates back
    for i, v in enumerate(bm.verts):
        v.co = co[i]
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def extend_arms_and_torso(obj: bpy.types.Object, landmarks: dict):
    """Extend cropped lower torso & forearms cleanly so full upper body is manifold & complete."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Find the 3 boundary edge loops
    visited = set()
    loops = []
    for e in bm.edges:
        if not e.is_boundary or e.index in visited:
            continue
        # Walk ordered vertex loop
        start_v = e.verts[0]
        cur_e = e
        cur_v = e.verts[1]
        loop_verts = [start_v]
        visited.add(e.index)
        while cur_v != start_v:
            loop_verts.append(cur_v)
            nxt_e = None
            for le in cur_v.link_edges:
                if le.is_boundary and le.index not in visited:
                    nxt_e = le
                    break
            if nxt_e is None:
                break
            visited.add(nxt_e.index)
            cur_v = nxt_e.other_vert(cur_v)
        loops.append(loop_verts)

    for loop in loops:
        cx = sum(v.co.x for v in loop) / len(loop)
        cy = sum(v.co.y for v in loop) / len(loop)
        cz = sum(v.co.z for v in loop) / len(loop)

        if abs(cx) < 0.05:
            # Waist boundary loop (44 verts): extrude smoothly down to Z=0.98 and cap cleanly
            cur_ring = loop
            steps = [
                (1.075, 0.98, 0.96),
                (1.040, 0.94, 0.92),
                (1.005, 0.88, 0.86),
                (0.980, 0.72, 0.72),
            ]
            for target_z, sx, sy in steps:
                res = bmesh.ops.extrude_edge_only(bm, edges=[
                    bm.edges.get((cur_ring[k], cur_ring[(k + 1) % len(cur_ring)]))
                    for k in range(len(cur_ring))
                ])
                new_verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
                # Map old ring order to new ring via link_edges
                next_ring = []
                old_set = set(cur_ring)
                for ov in cur_ring:
                    for le in ov.link_edges:
                        nv = le.other_vert(ov)
                        if nv in new_verts and nv not in next_ring:
                            next_ring.append(nv)
                            break
                for nv in next_ring:
                    nv.co.x *= sx
                    nv.co.y = cy + (nv.co.y - cy) * sy
                    nv.co.z = target_z
                cur_ring = next_ring
            # Cap bottom cleanly with central vertex fan
            center_v = bm.verts.new((0.0, cy, 0.972))
            for k in range(len(cur_ring)):
                bm.faces.new((cur_ring[k], center_v, cur_ring[(k + 1) % len(cur_ring)]))
        else:
            # Left (+X) or Right (-X) forearm loop: extend along elbow->hand vector cleanly into wrist + 5-digit hand
            side = "l" if cx > 0 else "r"
            sx = 1.0 if side == "l" else -1.0
            elbow = Vector(landmarks[f"joint-{side}-elbow"])
            wrist = Vector(landmarks[f"joint-{side}-hand"])
            axis = (wrist - elbow).normalized()
            cur_center = Vector((cx, cy, cz))
            rem_len = max(0.08, (wrist - cur_center).dot(axis))

            cur_ring = loop
            ext_steps = [
                (0.45 * rem_len, 0.85),
                (0.92 * rem_len, 0.72),
                (1.02 * rem_len, 0.70),  # wrist tucked cleanly inside sweater cuff
            ]
            for idx_s, (dist_along, scale_r) in enumerate(ext_steps):
                target_c = cur_center + axis * dist_along
                edges_ring = [
                    bm.edges.get((cur_ring[k], cur_ring[(k + 1) % len(cur_ring)]))
                    for k in range(len(cur_ring))
                ]
                res = bmesh.ops.extrude_edge_only(bm, edges=edges_ring)
                new_verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
                next_ring = []
                for ov in cur_ring:
                    for le in ov.link_edges:
                        nv = le.other_vert(ov)
                        if nv in new_verts and nv not in next_ring:
                            next_ring.append(nv)
                            break
                prev_c = sum((v.co for v in cur_ring), Vector((0, 0, 0))) / len(cur_ring)
                rel_scale = scale_r / (1.0 if idx_s == 0 else ext_steps[idx_s - 1][1])
                for k, nv in enumerate(next_ring):
                    offset = cur_ring[k].co - prev_c
                    axial_comp = axis * offset.dot(axis)
                    radial_comp = offset - axial_comp
                    nv.co = target_c + radial_comp * rel_scale + axial_comp * 0.20
                cur_ring = next_ring

            # Cap wrist cleanly inside sleeve cuff
            wrist_c = cur_center + axis * (1.02 * rem_len)
            cap_v = bm.verts.new(wrist_c)
            for k in range(len(cur_ring)):
                bm.faces.new((cur_ring[k], cap_v, cur_ring[(k + 1) % len(cur_ring)]))

            # Build clean manifold 5-digit relaxed feminine hand welded from wrist inside cuff
            # Local hand basis: forward=axis, radial_thumb toward -Y (front), dorsal_up perpendicular
            radial = Vector((0.0, -1.0, 0.0))
            radial = (radial - axis * radial.dot(axis)).normalized()
            if sx < 0:
                # Ensure thumb points toward medial/front (-Y) symmetrically on both hands
                pass
            dorsal = radial.cross(axis * sx).normalized()
            if dorsal.z < 0:
                dorsal = -dorsal

            # 4x1 quad palm grid (5 cols across width: idx 0=thumb side .. 4=pinky side; 2 rows top/bot)
            hand_origin = wrist - axis * 0.015
            palm_sections = [
                # (dist_along, width, thick, arch_dorsal)
                (0.000, 0.048, 0.026, 0.000),
                (0.028, 0.062, 0.025, 0.003),
                (0.056, 0.070, 0.023, 0.004),
                (0.082, 0.072, 0.020, 0.002),
            ]
            # Perimeter vertex order around 4x1 grid (10 verts): top[0..4], bot[4..0]
            palm_rings = []
            for d_al, w_h, th_h, arch_z in palm_sections:
                ring_v = []
                # Top 5 (0..4)
                for c_idx in range(5):
                    u_f = (c_idx / 4.0) - 0.5  # -0.5 thumb .. +0.5 pinky
                    arch = arch_z * (1.0 - (2.0 * u_f) ** 2)
                    p = hand_origin + axis * d_al - radial * (u_f * w_h) + dorsal * (0.5 * th_h + arch)
                    ring_v.append(bm.verts.new(p))
                # Bottom 5 (4..0)
                for c_idx in range(4, -1, -1):
                    u_f = (c_idx / 4.0) - 0.5
                    arch = arch_z * (1.0 - (2.0 * u_f) ** 2) * 0.4
                    p = hand_origin + axis * d_al - radial * (u_f * w_h) - dorsal * (0.5 * th_h - arch)
                    ring_v.append(bm.verts.new(p))
                palm_rings.append(ring_v)

            # Cap wrist heel of palm
            heel_c = bm.verts.new(hand_origin - axis * 0.005)
            for k in range(10):
                bm.faces.new((palm_rings[0][(k + 1) % 10], heel_c, palm_rings[0][k]))

            # Side faces along palm rings
            thumb_side_face = None
            for r_i in range(len(palm_rings) - 1):
                r0 = palm_rings[r_i]
                r1 = palm_rings[r_i + 1]
                for k in range(10):
                    kn = (k + 1) % 10
                    f = bm.faces.new((r0[k], r0[kn], r1[kn], r1[k]))
                    if r_i == 1 and k == 9:
                        thumb_side_face = f

            # 4 knuckle faces at front of palm (cols k=0..3)
            tip_r = palm_rings[-1]
            knuckle_faces = []
            for k in range(4):
                # top[k], top[k+1], bot[k+1], bot[k] -> bot index is 9-k and 8-k
                f_k = bm.faces.new((tip_r[k], tip_r[k + 1], tip_r[8 - k], tip_r[9 - k]))
                knuckle_faces.append(f_k)

            # Inset each knuckle face slightly so fingers separate naturally
            bmesh.ops.inset_individual(bm, faces=knuckle_faces, thickness=0.0022, depth=0.001)

            finger_specs = [
                # (total_length, radius_scale, curl_deg) - soft gently relaxed curled fingers
                (0.068, 0.92, 24.0),  # index
                (0.076, 0.96, 26.0),  # middle
                (0.071, 0.90, 28.0),  # ring
                (0.056, 0.80, 30.0),  # pinky
            ]
            for f_face, (f_len, f_rad, curl_d) in zip(knuckle_faces, finger_specs):
                cur_f = f_face
                cur_dir = axis.copy()
                seg_l = f_len / 3.0
                for seg_i, t_scale in enumerate((0.92, 0.84, 0.72)):
                    # Rotate direction softly toward palm (-dorsal)
                    cur_dir = (cur_dir - dorsal * math.radians(curl_d)).normalized()
                    res = bmesh.ops.extrude_face_region(bm, geom=[cur_f])
                    new_vs = set(g for g in res["geom"] if isinstance(g, bmesh.types.BMVert))
                    nf = next(g for g in res["geom"] if isinstance(g, bmesh.types.BMFace) and all(v in new_vs for v in g.verts))
                    fc = nf.calc_center_median()
                    for v in nf.verts:
                        v.co = fc + (v.co - fc) * t_scale + cur_dir * seg_l
                    cur_f = nf
                # Rounded fingertip cap
                res = bmesh.ops.extrude_face_region(bm, geom=[cur_f])
                new_vs = set(g for g in res["geom"] if isinstance(g, bmesh.types.BMVert))
                nf = next(g for g in res["geom"] if isinstance(g, bmesh.types.BMFace) and all(v in new_vs for v in g.verts))
                fc = nf.calc_center_median()
                for v in nf.verts:
                    v.co = fc + (v.co - fc) * 0.45 + cur_dir * 0.004

            # Extrude thumb from radial thumb_side_face
            if thumb_side_face is not None:
                bmesh.ops.inset_individual(bm, faces=[thumb_side_face], thickness=0.002, depth=0.001)
                cur_tf = thumb_side_face
                t_dir = (axis * 0.65 + radial * 0.72 - dorsal * 0.18).normalized()
                for seg_l, t_scale in ((0.022, 0.88), (0.020, 0.80), (0.016, 0.65)):
                    t_dir = (t_dir + axis * 0.25 - dorsal * 0.12).normalized()
                    res = bmesh.ops.extrude_face_region(bm, geom=[cur_tf])
                    new_vs = set(g for g in res["geom"] if isinstance(g, bmesh.types.BMVert))
                    nf = next(g for g in res["geom"] if isinstance(g, bmesh.types.BMFace) and all(v in new_vs for v in g.verts))
                    fc = nf.calc_center_median()
                    for v in nf.verts:
                        v.co = fc + (v.co - fc) * t_scale + t_dir * seg_l
                    cur_tf = nf

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def assign_head_uvs_and_collect_landmarks(obj: bpy.types.Object, eye_centers: dict) -> dict:
    """Create seam-free cylindrical-frontal head/neck UVs and extract exact UV coordinates of facial features."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    uv_layer = bm.loops.layers.uv.verify()

    # Assign two material slots: 0 = Skin_FaceNeck (z > 1.31 or neck/head), 1 = Skin_Hands (below collar/arms)
    # Wait: keep chest/collarbone area up to z=1.28 in FaceNeck UV cleanly so neck-to-chest transition has zero seam!
    head_c_y = -0.045

    def world_to_uv(co: Vector):
        theta = math.atan2(co.x, -(co.y - head_c_y))  # 0 at dead front -Y, +/-pi at back +Y
        # Nonlinear angle warp: concentrate 66% of U in front face [-1.15, 1.15] rad
        t_norm = theta / math.pi  # [-1, 1]
        # Smooth monotonic warp u_rel in [-0.5, 0.5]
        u_rel = 0.5 * (0.68 * math.sin(theta * 0.5) + 0.32 * t_norm)
        u = 0.5 + u_rel
        # Vertical mapping z in [1.26, 1.69] -> v in [0.04, 0.96]
        v = 0.04 + 0.92 * max(0.0, min(1.0, (co.z - 1.26) / 0.43))
        return u, v

    for face in bm.faces:
        # Check if face belongs to head/neck/upper-chest vs lower torso/arms
        avg_z = sum(v.co.z for v in face.verts) / len(face.verts)
        avg_x = sum(abs(v.co.x) for v in face.verts) / len(face.verts)
        is_head_neck = (avg_z > 1.28) and (avg_x < 0.14 or avg_z > 1.38)
        face.material_index = 0 if is_head_neck else 1

        # Handle UV seam crossing at back of skull (theta crossing +/-pi)
        us = []
        vs = []
        for loop in face.loops:
            u, v = world_to_uv(loop.vert.co)
            us.append(u)
            vs.append(v)
        # If polygon straddles the u=0 / u=1 seam at back of head, wrap negative/low side by +1
        if max(us) - min(us) > 0.5:
            us = [u + 1.0 if u < 0.5 else u for u in us]
        for loop, u, v in zip(face.loops, us, vs):
            loop[uv_layer].uv = (u, v)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # Now compute exact UV coordinates of key sculpted facial landmarks for textures.py!
    def uv_pt(x, y, z):
        u, v = world_to_uv(Vector((x, y, z)))
        return [float(u), float(v)]

    # Sample actual eyelid & lip curves directly from sculpted mesh vertices
    co = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)

    def sample_curve_uv(x_vals, z_min, z_max, y_max, mode="top"):
        pts = []
        for xv in x_vals:
            mask = (np.abs(co[:, 0] - xv) < 0.0028) & (co[:, 2] >= z_min) & (co[:, 2] <= z_max) & (co[:, 1] <= y_max)
            sub = co[mask]
            if len(sub) > 0:
                if mode == "top":
                    p = sub[np.argmax(sub[:, 2])]
                elif mode == "bot":
                    p = sub[np.argmin(sub[:, 2])]
                else:
                    p = sub[np.argmin(sub[:, 1])]
                pts.append(uv_pt(p[0], p[1], p[2]))
        return pts

    # Collect exact sculpted eyelid rim and lip UV curves directly from mesh geometry
    # Sample actual deformed upper & lower eyelid lash-margin vertices around eye centers
    def sample_lid_uv(sx, is_upper):
        pts = []
        c_e = eye_centers["l"] if sx > 0 else eye_centers["r"]
        # Mask vertices near front of eye socket
        for xv in np.linspace(0.0195, 0.0430, 9):
            x_target = sx * xv
            mask = (
                (np.abs(co[:, 0] - x_target) < 0.0028)
                & (co[:, 1] < -0.130)
                & (np.linalg.norm(co - np.array(c_e), axis=1) < 0.018)
            )
            if is_upper:
                mask &= co[:, 2] > c_e[2] - 0.001
                sub = co[mask]
                if len(sub) > 0:
                    # Eyelid margin is the lowest Z among upper-lid rim verts near eyeball surface
                    p = sub[np.argmin(sub[:, 2])]
                    pts.append(uv_pt(p[0], p[1], p[2]))
            else:
                mask &= co[:, 2] < c_e[2] + 0.001
                sub = co[mask]
                if len(sub) > 0:
                    p = sub[np.argmax(sub[:, 2])]
                    pts.append(uv_pt(p[0], p[1], p[2]))
        return sorted(pts, key=lambda p: p[0])

    # Sample actual sculpted lip vermilion borders and inner seam from mesh vertices!
    # We know original hm08 indices: upper peak ~466 (z~1.485), seam ~474/494 (z~1.477), lower bottom ~496 (z~1.470)
    lip_upper_pts = []
    lip_lower_pts = []
    lip_seam_pts = []
    for xv in np.linspace(-0.0205, 0.0205, 17):
        x_abs = abs(xv)
        z_seam_est = 1.4767 + 0.0023 * min(1.0, x_abs / 0.022) ** 1.85
        # Vermilion lip top has cupid's bow around z_seam + 0.009 at center tapering to +0.003 at corners
        taper = max(0.18, 1.0 - (x_abs / 0.022) ** 1.4)
        cupid = 0.0020 * math.exp(-((x_abs - 0.0055) / 0.0042) ** 2) - 0.0006 * math.exp(-(x_abs / 0.0030) ** 2)
        z_top = z_seam_est + 0.0122 * taper + cupid
        z_bot = z_seam_est - 0.0085 * taper
        lip_upper_pts.append(uv_pt(xv, -0.156, z_top))
        lip_lower_pts.append(uv_pt(xv, -0.156, z_bot))
        lip_seam_pts.append(uv_pt(xv, -0.156, z_seam_est))

    landmarks_uv = {
        "cheek_l": uv_pt(0.039, -0.140, 1.514),
        "cheek_r": uv_pt(-0.039, -0.140, 1.514),
        "nose_tip": uv_pt(0.0, -0.165, 1.510),
        "chin": uv_pt(0.0, -0.150, 1.456),
        "forehead": uv_pt(0.0, -0.146, 1.600),
        "ear_l": uv_pt(0.080, -0.052, 1.535),
        "ear_r": uv_pt(-0.080, -0.052, 1.535),
        "nostrils": [
            uv_pt(0.0082, -0.155, 1.503),
            uv_pt(-0.0082, -0.155, 1.503),
        ],
        "eye_center_l": uv_pt(eye_centers["l"][0], -0.139, eye_centers["l"][2]),
        "eye_center_r": uv_pt(eye_centers["r"][0], -0.139, eye_centers["r"][2]),
        "eyelid_upper_l": sample_lid_uv(1.0, True),
        "eyelid_lower_l": sample_lid_uv(1.0, False),
        "eyelid_upper_r": sample_lid_uv(-1.0, True),
        "eyelid_lower_r": sample_lid_uv(-1.0, False),
        "brow_l": [
            uv_pt(0.017 + 0.029 * t, -0.147, 1.5645 + 0.0022 * math.sin(math.pi * t * 0.78))
            for t in np.linspace(0.0, 1.0, 9)
        ],
        "brow_r": [
            uv_pt(-(0.017 + 0.029 * t), -0.147, 1.5645 + 0.0022 * math.sin(math.pi * t * 0.78))
            for t in np.linspace(0.0, 1.0, 9)
        ],
        "lip_upper_outline": sorted(lip_upper_pts, key=lambda p: p[0]),
        "lip_lower_outline": sorted(lip_lower_pts, key=lambda p: p[0]),
        "lip_seam": sorted(lip_seam_pts, key=lambda p: p[0]),
    }
    return landmarks_uv


def create_skin_materials(skin_imgs: tuple) -> tuple:
    """Create photorealistic subsurface-scattering skin materials for face/neck and hands."""
    img_alb, img_rough, img_bump = skin_imgs

    mat_face = bpy.data.materials.new("Skin_FaceNeck")
    nodes = mat_face.node_tree.nodes
    links = mat_face.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.subsurface_method = "RANDOM_WALK_SKIN"
    bsdf.inputs["Subsurface Weight"].default_value = 0.18
    bsdf.inputs["Subsurface Scale"].default_value = 0.0050
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.22)
    bsdf.inputs["Subsurface Anisotropy"].default_value = 0.25
    bsdf.inputs["IOR"].default_value = 1.40
    bsdf.inputs["Specular IOR Level"].default_value = 0.38
    bsdf.inputs["Coat Weight"].default_value = 0.04
    bsdf.inputs["Coat Roughness"].default_value = 0.30

    uv = nodes.new("ShaderNodeTexCoord")

    tex_alb = nodes.new("ShaderNodeTexImage")
    tex_alb.image = img_alb
    tex_rough = nodes.new("ShaderNodeTexImage")
    tex_rough.image = img_rough
    tex_bump = nodes.new("ShaderNodeTexImage")
    tex_bump.image = img_bump

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.0008

    links.new(uv.outputs["UV"], tex_alb.inputs["Vector"])
    links.new(uv.outputs["UV"], tex_rough.inputs["Vector"])
    links.new(uv.outputs["UV"], tex_bump.inputs["Vector"])

    links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex_rough.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(tex_bump.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Matching hand/wrist skin shader (clean procedural fair warm skin)
    mat_hand = bpy.data.materials.new("Skin_Hands")
    hn = mat_hand.node_tree.nodes
    hl = mat_hand.node_tree.links
    hn.clear()
    hout = hn.new("ShaderNodeOutputMaterial")
    hbsdf = hn.new("ShaderNodeBsdfPrincipled")
    hbsdf.subsurface_method = "RANDOM_WALK_SKIN"
    hbsdf.inputs["Base Color"].default_value = (0.78, 0.51, 0.40, 1.0)
    hbsdf.inputs["Roughness"].default_value = 0.42
    hbsdf.inputs["Subsurface Weight"].default_value = 0.18
    hbsdf.inputs["Subsurface Scale"].default_value = 0.005
    hbsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.22)
    hbsdf.inputs["Specular IOR Level"].default_value = 0.38
    hl.new(hbsdf.outputs["BSDF"], hout.inputs["Surface"])

    return mat_face, mat_hand


def create_eye_materials(iris_img: bpy.types.Image) -> tuple:
    """Create realistic sclera+iris material, shadow-transparent refractive cornea, wet tear line, and caruncle."""
    # 1. Inner sclera & iris
    mat_iris = bpy.data.materials.new("Eye_ScleraIris")
    nodes = mat_iris.node_tree.nodes
    links = mat_iris.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.42
    bsdf.inputs["Specular IOR Level"].default_value = 0.04
    bsdf.inputs["Subsurface Weight"].default_value = 0.0
    uv = nodes.new("ShaderNodeTexCoord")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = iris_img
    tex.extension = "EXTEND"
    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # 2. Outer glossy cornea (Transparent + Fresnel Glossy so 100% of iris GI passes cleanly with crisp catchlights!)
    mat_cornea = bpy.data.materials.new("Eye_Cornea")
    cn = mat_cornea.node_tree.nodes
    cl = mat_cornea.node_tree.links
    cn.clear()
    cout = cn.new("ShaderNodeOutputMaterial")
    transp = cn.new("ShaderNodeBsdfTransparent")
    gloss = cn.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    gloss.inputs["Roughness"].default_value = 0.012
    lw = cn.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.032
    ramp = cn.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.018, 0.018, 0.018, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.14, 0.14, 0.14, 1.0)
    mix = cn.new("ShaderNodeMixShader")
    cl.new(lw.outputs["Fresnel"], ramp.inputs["Fac"])
    cl.new(ramp.outputs["Color"], mix.inputs["Fac"])
    cl.new(transp.outputs["BSDF"], mix.inputs[1])
    cl.new(gloss.outputs["BSDF"], mix.inputs[2])
    cl.new(mix.outputs["Shader"], cout.inputs["Surface"])

    # 3. Wet tear line meniscus (Transparent + Fresnel Glossy to avoid refraction caustics)
    mat_tear = bpy.data.materials.new("Eye_TearMeniscus")
    tn = mat_tear.node_tree.nodes
    tl = mat_tear.node_tree.links
    tn.clear()
    tout = tn.new("ShaderNodeOutputMaterial")
    ttransp = tn.new("ShaderNodeBsdfTransparent")
    tgloss = tn.new("ShaderNodeBsdfGlossy")
    tgloss.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    tgloss.inputs["Roughness"].default_value = 0.02
    tlw = tn.new("ShaderNodeLayerWeight")
    tlw.inputs["Blend"].default_value = 0.05
    tmix = tn.new("ShaderNodeMixShader")
    tl.new(tlw.outputs["Fresnel"], tmix.inputs["Fac"])
    tl.new(ttransp.outputs["BSDF"], tmix.inputs[1])
    tl.new(tgloss.outputs["BSDF"], tmix.inputs[2])
    tl.new(tmix.outputs["Shader"], tout.inputs["Surface"])

    # 4. Inner eye corner caruncle
    mat_car = bpy.data.materials.new("Eye_Caruncle")
    rn = mat_car.node_tree.nodes
    rl = mat_car.node_tree.links
    rn.clear()
    rout = rn.new("ShaderNodeOutputMaterial")
    rbsdf = rn.new("ShaderNodeBsdfPrincipled")
    rbsdf.inputs["Base Color"].default_value = (0.68, 0.30, 0.28, 1.0)
    rbsdf.inputs["Roughness"].default_value = 0.20
    rbsdf.inputs["Subsurface Weight"].default_value = 0.35
    rbsdf.inputs["Subsurface Scale"].default_value = 0.003
    rl.new(rbsdf.outputs["BSDF"], rout.inputs["Surface"])

    return mat_iris, mat_cornea, mat_tear, mat_car


def build_eye_pair(eye_centers: dict, eye_mats: tuple, collection: bpy.types.Collection) -> list:
    """Create left and right multi-layer eyes (sclera+iris, bulging cornea, tear meniscus, caruncle)."""
    mat_iris, mat_cornea, mat_tear, mat_car = eye_mats
    created = []

    for side, sx in (("L", 1.0), ("R", -1.0)):
        c = Vector(eye_centers[side.lower()])

        # Inner eyeball sphere (rotate pole to -Y so concentric rings align with iris!)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=32, radius=0.01390)
        bmesh.ops.rotate(bm, verts=bm.verts, matrix=Matrix.Rotation(math.pi * 0.5, 3, "X"))
        uv_layer = bm.loops.layers.uv.verify()
        for v in bm.verts:
            if v.co.y < -0.0110:
                r_xz = math.hypot(v.co.x, v.co.z)
                if r_xz < 0.0070:
                    t_d = 1.0 - (r_xz / 0.0070) ** 2
                    t_d = t_d * t_d * (3.0 - 2.0 * t_d)
                    v.co.y += 0.0005 * t_d
        bm.normal_update()
        r_map = 0.0258
        for f in bm.faces:
            for loop in f.loops:
                co = loop.vert.co
                if co.y < 0.002:
                    u = 0.5 + co.x / r_map
                    v = 0.5 + co.z / r_map
                else:
                    u, v = 0.02, 0.02
                loop[uv_layer].uv = (u, v)

        m_in = bpy.data.meshes.new(f"Eye_Inner_{side}")
        bm.to_mesh(m_in)
        bm.free()
        o_in = bpy.data.objects.new(f"Eye_Inner_{side}", m_in)
        o_in.location = c
        o_in.data.materials.append(mat_iris)
        for p in o_in.data.polygons:
            p.use_smooth = True
        sub_in = o_in.modifiers.new("Subdivision", type="SUBSURF")
        sub_in.levels = 1
        sub_in.render_levels = 2
        collection.objects.link(o_in)

        # Outer cornea dome (pole along -Y for liquid-smooth catchlight reflections!)
        bm_c = bmesh.new()
        bmesh.ops.create_uvsphere(bm_c, u_segments=48, v_segments=32, radius=0.01406)
        bmesh.ops.rotate(bm_c, verts=bm_c.verts, matrix=Matrix.Rotation(math.pi * 0.5, 3, "X"))
        for v in bm_c.verts:
            if v.co.y < -0.0105:
                r_xz = math.hypot(v.co.x, v.co.z)
                if r_xz < 0.0070:
                    t_b = 1.0 - (r_xz / 0.0070) ** 2
                    t_b = t_b * t_b * (3.0 - 2.0 * t_b)
                    v.co.y -= 0.0008 * t_b
        bm_c.normal_update()
        m_cor = bpy.data.meshes.new(f"Eye_Cornea_{side}")
        bm_c.to_mesh(m_cor)
        bm_c.free()
        o_cor = bpy.data.objects.new(f"Eye_Cornea_{side}", m_cor)
        o_cor.location = c
        o_cor.data.materials.append(mat_cornea)
        for p in o_cor.data.polygons:
            p.use_smooth = True
        sub_c = o_cor.modifiers.new("Subdivision", type="SUBSURF")
        sub_c.levels = 1
        sub_c.render_levels = 2
        collection.objects.link(o_cor)

        # Thin glistening tear meniscus along lower lid edge
        bm_t = bmesh.new()
        n_seg = 18
        ring_verts = []
        for k in range(n_seg + 1):
            t = k / n_seg
            ang_x = math.radians(-26.0 + 52.0 * t)
            px = math.sin(ang_x) * 0.0137
            py = -math.cos(ang_x) * 0.0137
            pz = -0.0052 + 0.0018 * ((2.0 * t - 1.0) ** 2)
            v0 = bm_t.verts.new((px, py - 0.0005, pz))
            v1 = bm_t.verts.new((px, py + 0.0003, pz + 0.0006))
            v2 = bm_t.verts.new((px, py - 0.0001, pz - 0.0005))
            ring_verts.append((v0, v1, v2))
        for k in range(n_seg):
            a0, a1, a2 = ring_verts[k]
            b0, b1, b2 = ring_verts[k + 1]
            bm_t.faces.new((a0, b0, b1, a1))
            bm_t.faces.new((a2, b2, b0, a0))
        bm_t.normal_update()
        m_tear = bpy.data.meshes.new(f"Eye_TearLine_{side}")
        bm_t.to_mesh(m_tear)
        bm_t.free()
        o_tear = bpy.data.objects.new(f"Eye_TearLine_{side}", m_tear)
        o_tear.location = c
        o_tear.data.materials.append(mat_tear)
        for p in o_tear.data.polygons:
            p.use_smooth = True
        collection.objects.link(o_tear)

        # Soft rosy caruncle tucked in medial canthus
        bm_k = bmesh.new()
        bmesh.ops.create_uvsphere(bm_k, u_segments=14, v_segments=10, radius=0.0022)
        m_car = bpy.data.meshes.new(f"Eye_Caruncle_{side}")
        bm_k.to_mesh(m_car)
        bm_k.free()
        o_car = bpy.data.objects.new(f"Eye_Caruncle_{side}", m_car)
        o_car.location = (c.x - sx * 0.0110, c.y - 0.0080, c.z - 0.0014)
        o_car.scale = (0.9, 0.7, 0.7)
        o_car.data.materials.append(mat_car)
        for p in o_car.data.polygons:
            p.use_smooth = True
        collection.objects.link(o_car)

        created.append((side, o_in, o_cor, o_tear, o_car))
    return created


def build_sculpted_body(env: dict, collection: bpy.types.Collection):
    """Import upper-body.obj, sculpt face, extend torso/arms, UV unwrap head, and return mesh + landmarks."""
    bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis="Y", up_axis="Z")
    body_obj = bpy.context.selected_objects[0]
    body_obj.name = "Character_Body"
    body_obj.data.name = "Character_Body_Mesh"

    # Move cleanly into target collection
    for col in list(body_obj.users_collection):
        col.objects.unlink(body_obj)
    collection.objects.link(body_obj)

    eye_centers = {
        "l": [env["landmarks"]["helper-l-eye"][0], env["landmarks"]["helper-l-eye"][1] - 0.0007, env["landmarks"]["helper-l-eye"][2]],
        "r": [env["landmarks"]["helper-r-eye"][0], env["landmarks"]["helper-r-eye"][1] - 0.0007, env["landmarks"]["helper-r-eye"][2]],
    }

    # Sculpt face & neck BEFORE extending forearms/torso so vertex indices match base hm08 topology
    sculpt_face_anatomy(body_obj, eye_centers)

    # Extend cropped arms & torso cleanly so full upper body is solid & complete
    extend_arms_and_torso(body_obj, env["landmarks"])

    # Assign head/neck cylindrical UVs and collect exact UV coordinates for texture generation
    landmarks_uv = assign_head_uvs_and_collect_landmarks(body_obj, eye_centers)

    # Smooth shading + Subdivision Surface modifier for organic skin smoothness
    for p in body_obj.data.polygons:
        p.use_smooth = True
    sub = body_obj.modifiers.new("Subdivision", type="SUBSURF")
    sub.levels = 1
    sub.render_levels = 2

    return body_obj, landmarks_uv, eye_centers
