"""Genuine 3D knitted crew-neck sweater with relaxed fit, 3D folds, drop-shoulder seams, and ribbed trim."""

import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from .anatomy import bind_mesh_to_armature, _smoothstep


def _laplacian_smooth_selective(V: np.ndarray, adj: list, weights: np.ndarray, iterations: int = 18):
    cur = V.copy()
    for _ in range(iterations):
        nxt = cur.copy()
        for i, nbrs in enumerate(adj):
            w = weights[i]
            if w > 1e-4 and nbrs:
                mean_nb = np.mean(cur[nbrs], axis=0)
                nxt[i] = cur[i] * (1.0 - w) + mean_nb * w
        cur = nxt
    return cur


def build_knitted_sweater(
    body_obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    knit_height_img: bpy.types.Image,
    collection,
) -> bpy.types.Object:
    """Construct a genuine 3D editable knitted crew-neck pullover sweater over the rest-pose body."""
    src_mesh = body_obj.data
    n_src = len(src_mesh.vertices)

    rest_flat = np.zeros(n_src * 3, dtype=np.float64)
    src_mesh.attributes["rest_pos"].data.foreach_get("vector", rest_flat)
    P_body = rest_flat.reshape((n_src, 3))

    def neck_field(pt):
        px, py, pz = pt[0], pt[1], pt[2]
        t_fb = np.clip((py - (-0.085)) / 0.130, 0.0, 1.0)
        z_cut = 1.352 + 0.050 * t_fb + 0.012 * min((abs(px) / 0.065) ** 2, 1.2)
        return (abs(px) < 0.095) and (pz > z_cut)

    kept_polys = []
    v_map = {}
    sw_verts = []

    for poly in src_mesh.polygons:
        pts = [P_body[vi] for vi in poly.vertices]
        if any(neck_field(p) for p in pts):
            continue
        if any(p[2] > 1.418 for p in pts):
            continue
        remap = []
        for vi in poly.vertices:
            if vi not in v_map:
                v_map[vi] = len(sw_verts)
                sw_verts.append(P_body[vi].copy())
            remap.append(v_map[vi])
        kept_polys.append(remap)

    V = np.array(sw_verts, dtype=np.float64)
    n_v = len(V)

    adj = [set() for _ in range(n_v)]
    edge_counts = {}
    for f in kept_polys:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            adj[a].add(b)
            adj[b].add(a)
            ek = (a, b) if a < b else (b, a)
            edge_counts[ek] = edge_counts.get(ek, 0) + 1
    adj = [list(s) for s in adj]

    v_normals = np.zeros_like(V)
    for f in kept_polys:
        p0, p1, p2 = V[f[0]], V[f[1]], V[f[2]]
        fn = np.cross(p1 - p0, p2 - p0)
        nl = np.linalg.norm(fn)
        if nl > 1e-12:
            fn /= nl
        for vi in f:
            v_normals[vi] += fn
    v_normals /= np.maximum(np.linalg.norm(v_normals, axis=1, keepdims=True), 1e-8)

    boundary_verts = set()
    for (a, b), cnt in edge_counts.items():
        if cnt == 1:
            boundary_verts.add(a)
            boundary_verts.add(b)

    # ------------------------------------------------------------------
    # 1. Plush Relaxed Knitwear Loft + Strong Garment Span Relaxation
    # ------------------------------------------------------------------
    ax = np.abs(V[:, 0])
    neck_prox = _smoothstep(1.315, 1.385, V[:, 2]) * (1.0 - _smoothstep(0.065, 0.125, ax))
    cuff_prox = _smoothstep(0.315, 0.358, ax)
    sleeve_zone = _smoothstep(0.145, 0.215, ax) * (1.0 - cuff_prox)

    loft = (
        0.0195 * (1.0 - neck_prox) + 0.0095 * neck_prox
        + 0.0075 * sleeve_zone
        - 0.0025 * cuff_prox
    )
    underarm = np.exp(-(((ax - 0.145) / 0.055) ** 2)) * np.exp(-(((V[:, 2] - 1.255) / 0.065) ** 2))
    lower_drape = (1.0 - _smoothstep(1.06, 1.25, V[:, 2])) * (1.0 - _smoothstep(0.13, 0.18, ax))

    V_loft = V + v_normals * loft[:, None]
    V_loft[:, 2] -= 0.015 * underarm
    # Relax under-bust into straight cozy knit pullover hang!
    V_loft[:, 1] -= 0.012 * lower_drape * _smoothstep(0.02, -0.09, V[:, 1])

    smooth_w = np.ones(n_v, dtype=np.float64) * 0.46
    for bv in boundary_verts:
        smooth_w[bv] = 0.25
    V_relaxed = _laplacian_smooth_selective(V_loft, adj, smooth_w, iterations=28)

    diff_body = V_relaxed - V
    proj_n = np.sum(diff_body * v_normals, axis=1)
    min_clear = 0.0090 * (1.0 - 0.25 * neck_prox)
    deficit = np.maximum(min_clear - proj_n, 0.0)
    V_relaxed += v_normals * deficit[:, None]

    # ------------------------------------------------------------------
    # 2. Sculpt Authentic 3D Garment Seams, Compression Folds & Drape Waves
    # ------------------------------------------------------------------
    sx = np.sign(V_relaxed[:, 0])
    ax_r = np.abs(V_relaxed[:, 0])
    yr = V_relaxed[:, 1]
    zr = V_relaxed[:, 2]

    # Recessed Drop-Shoulder Seam Groove around |X| = 0.170
    seam_x = 0.170 + 0.012 * (1.32 - zr)
    seam_band = (
        np.exp(-(((ax_r - seam_x) / 0.0065) ** 2))
        * _smoothstep(1.18, 1.23, zr)
        * _smoothstep(0.11, 0.15, ax_r)
    )
    welt_band = (
        np.exp(-(((ax_r - (seam_x - 0.006)) / 0.0055) ** 2))
        * _smoothstep(1.18, 1.23, zr)
    )
    V_relaxed -= v_normals * (0.0028 * seam_band)[:, None]
    V_relaxed += v_normals * (0.0015 * welt_band)[:, None]

    # Anterior Shoulder Compression Folds
    sh_front = _smoothstep(-0.01, -0.09, yr) * np.exp(-(((ax_r - 0.165) / 0.065) ** 2)) * np.exp(
        -(((zr - 1.295) / 0.075) ** 2)
    )
    sh_phase = ax_r * 42.0 + zr * 34.0 + yr * 18.0
    sh_folds = 0.55 * np.sin(sh_phase) + 0.30 * np.sin(sh_phase * 1.85 + 0.9)
    V_relaxed += v_normals * (0.0042 * sh_front * sh_folds)[:, None]

    # Cozy Plush Sleeve Bunching Folds along Upper Arm, Elbow & Forearm (very prominent in reference.png!)
    sleeve_len = np.clip((ax_r - 0.170) / 0.185, 0.0, 1.0)
    sleeve_mask = _smoothstep(0.168, 0.200, ax_r)
    wave_1 = np.sin(sleeve_len * 7.0 * math.pi + 0.6 * sx + yr * 14.0)
    wave_2 = np.sin(sleeve_len * 12.5 * math.pi - 1.1 * sx + zr * 18.0)
    elbow_boost = 1.0 + 0.90 * np.exp(-(((sleeve_len - 0.52) / 0.24) ** 2)) * _smoothstep(0.01, -0.09, yr)
    sleeve_disp = 0.0045 * sleeve_mask * elbow_boost * (0.65 * wave_1 + 0.35 * wave_2)
    V_relaxed += v_normals * np.maximum(sleeve_disp, -0.0010)[:, None]

    # Soft Torso Drape Ripples
    torso_drape_m = (1.0 - _smoothstep(0.13, 0.17, ax_r)) * (1.0 - _smoothstep(1.12, 1.27, zr))
    drape_wave = (
        0.6 * np.sin(V_relaxed[:, 0] * 36.0 + zr * 16.0 + 0.4)
        + 0.4 * np.cos(V_relaxed[:, 0] * 22.0 - yr * 20.0)
    )
    V_relaxed += v_normals * np.maximum(0.0036 * torso_drape_m * drape_wave, -0.0010)[:, None]

    # ------------------------------------------------------------------
    # 3. Extrude Thick 3D Rolled Ribbed Collar, Cozy Long Cuffs, & Closed Waist Hem
    # ------------------------------------------------------------------
    bm = bmesh.new()
    bm_verts = [bm.verts.new(Vector(pt)) for pt in V_relaxed]
    bm.verts.ensure_lookup_table()

    for f_idxs in kept_polys:
        face = bm.faces.new([bm_verts[i] for i in f_idxs])
        face.smooth = True
        face.material_index = 0

    bm.edges.ensure_lookup_table()
    bm.normal_update()

    b_edges = [e for e in bm.edges if e.is_boundary]
    visited_e = set()
    loops_info = []

    for start_e in b_edges:
        if start_e in visited_e:
            continue
        loop_v = [start_e.verts[0], start_e.verts[1]]
        visited_e.add(start_e)
        while True:
            cur_v = loop_v[-1]
            nxt_e = None
            for e in cur_v.link_edges:
                if e.is_boundary and e not in visited_e:
                    nxt_e = e
                    break
            if nxt_e is None:
                break
            visited_e.add(nxt_e)
            other_v = nxt_e.other_vert(cur_v)
            if other_v == loop_v[0]:
                break
            loop_v.append(other_v)

        pts_l = np.array([[v.co.x, v.co.y, v.co.z] for v in loop_v])
        loops_info.append((loop_v, pts_l.mean(axis=0)))

    def ensure_loop_winding(loop_v, center_vec, ref_axis_vec):
        p0 = loop_v[0].co
        p1 = loop_v[1].co
        tang = (p1 - p0).normalized()
        rad = (p0 - center_vec).normalized()
        if tang.cross(ref_axis_vec).dot(rad) < 0.0:
            loop_v.reverse()

    tag_layer = bm.faces.layers.int.new("trim_tag")
    for f in bm.faces:
        f[tag_layer] = 0

    for loop_v, mean_p in loops_info:
        mx, my, mz = mean_p
        n_lp = len(loop_v)
        center_v = Vector((mx, my, mz))

        if mz > 1.30:
            # CREW-NECK RIBBED COLLAR (~3 cm chunky vertical 1x1 rib band!)
            for v in loop_v:
                for lf in v.link_faces:
                    lf.material_index = 1
                    lf[tag_layer] = 1
                    for v2 in lf.verts:
                        for lf2 in v2.link_faces:
                            if (lf2.calc_center_median() - v.co).length < 0.024:
                                lf2.material_index = 1
                                lf2[tag_layer] = 1

            neck_axis = Vector((0.0, -0.020, mz))
            ensure_loop_winding(loop_v, neck_axis, Vector((0.0, 0.0, 1.0)))

            ring_specs = [
                (0.0080, 0.89, 0.0024),
                (0.0135, 0.81, 0.0006),
                (0.0090, 0.72, -0.0025),
            ]
            prev_ring = loop_v
            for dz_s, rad_scale, out_off in ring_specs:
                new_ring = []
                for v0 in loop_v:
                    c_ax = Vector((0.0, -0.022, v0.co.z))
                    r_vec = v0.co - c_ax
                    r_dir = Vector((r_vec.x, r_vec.y, 0.0)).normalized()
                    np_co = Vector((
                        c_ax.x + r_vec.x * rad_scale + r_dir.x * out_off,
                        c_ax.y + r_vec.y * rad_scale + r_dir.y * out_off,
                        v0.co.z + dz_s,
                    ))
                    new_ring.append(bm.verts.new(np_co))
                for k in range(n_lp):
                    k2 = (k + 1) % n_lp
                    nf = bm.faces.new([prev_ring[k], prev_ring[k2], new_ring[k2], new_ring[k]])
                    nf.smooth = True
                    nf.material_index = 1
                    nf[tag_layer] = 1
                prev_ring = new_ring

        elif abs(mx) > 0.25:
            # COZY LONG SLEEVE EXTENSION + CHUNKY RIBBED CUFF + CLEAN CLOSED INTERIOR!
            is_left = mx > 0.0
            tag_id = 2 if is_left else 3
            sx_c = 1.0 if is_left else -1.0

            # Exact rest-pose forearm bone direction: (0.268,-0.013,1.168) -> (0.358,-0.105,1.102)
            arm_dir = Vector((sx_c * 0.090, -0.092, -0.066)).normalized()
            ensure_loop_winding(loop_v, center_v, arm_dir)

            # Extend sleeve smoothly +9.5 cm past wrist with plush cylindrical chunky ribbed cuffs!
            cuff_specs = [
                (0.032, 1.08, 0, tag_id),   # plush lower forearm fullness
                (0.064, 1.03, 1, tag_id),   # wide chunky ribbed cuff start
                (0.095, 0.99, 1, tag_id),   # chunky ribbed cuff outer edge
                (0.099, 0.91, 1, tag_id),   # thick rounded cuff lip
                (0.076, 0.78, 1, tag_id),   # folded inner cuff return
            ]
            prev_ring = loop_v
            for along_d, r_sc, mat_i, tg_i in cuff_specs:
                new_ring = []
                for v0 in loop_v:
                    rel = v0.co - center_v
                    perp = rel - arm_dir * rel.dot(arm_dir)
                    along_base = arm_dir * rel.dot(arm_dir)
                    np_co = center_v + along_base + arm_dir * along_d + perp * r_sc
                    new_ring.append(bm.verts.new(np_co))
                for k in range(n_lp):
                    k2 = (k + 1) % n_lp
                    nf = bm.faces.new([prev_ring[k], prev_ring[k2], new_ring[k2], new_ring[k]])
                    nf.smooth = True
                    nf.material_index = mat_i
                    nf[tag_layer] = tg_i if mat_i == 1 else 0
                prev_ring = new_ring

            # Recessed dark cuff interior cap so no hollow hole is ever visible
            cuff_cap = bm.verts.new(center_v + arm_dir * 0.058)
            for k in range(n_lp):
                k2 = (k + 1) % n_lp
                cf = bm.faces.new([prev_ring[k], prev_ring[k2], cuff_cap])
                cf.smooth = True
                cf.material_index = 2
                cf[tag_layer] = 5

        else:
            # WAISTBAND RIBBED HEM + CLOSED INTERIOR CAP
            for v in loop_v:
                for lf in v.link_faces:
                    lf.material_index = 1
                    lf[tag_layer] = 4

            down_dir = Vector((0.0, 0.0, -1.0))
            ensure_loop_winding(loop_v, center_v, down_dir)

            hem_specs = [
                (-0.026, 0.96),
                (-0.048, 0.91),
                (-0.051, 0.83),
                (-0.038, 0.72),
            ]
            prev_ring = loop_v
            for dz_h, r_sc in hem_specs:
                new_ring = []
                for v0 in loop_v:
                    c_xy = Vector((0.0, -0.045, v0.co.z))
                    rel = v0.co - c_xy
                    np_co = Vector((c_xy.x + rel.x * r_sc, c_xy.y + rel.y * r_sc, v0.co.z + dz_h))
                    new_ring.append(bm.verts.new(np_co))
                for k in range(n_lp):
                    k2 = (k + 1) % n_lp
                    nf = bm.faces.new([prev_ring[k], prev_ring[k2], new_ring[k2], new_ring[k]])
                    nf.smooth = True
                    nf.material_index = 1
                    nf[tag_layer] = 4
                prev_ring = new_ring

            cap_center = bm.verts.new(Vector((0.0, -0.045, mz - 0.032)))
            for k in range(n_lp):
                k2 = (k + 1) % n_lp
                cf = bm.faces.new([prev_ring[k], prev_ring[k2], cap_center])
                cf.smooth = True
                cf.material_index = 2
                cf[tag_layer] = 5

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # ------------------------------------------------------------------
    # 4. Garment Pattern UVs (Bodice Vertical Wales + Sleeve Lengthwise Wales + Rib Trim)
    # ------------------------------------------------------------------
    uv_lay = bm.loops.layers.uv.new("UVMap")

    for f in bm.faces:
        f_tag = f[tag_layer]
        f_center = f.calc_center_median()

        if f_tag == 0:
            seam_thresh = 0.170 + 0.012 * (1.32 - f_center.z)
            if abs(f_center.x) < seam_thresh:
                u_prev = None
                for loop in f.loops:
                    co = loop.vert.co
                    ang = math.atan2(co.x, -(co.y - (-0.035)))
                    u = 0.5 + (ang / (2.0 * math.pi)) * 1.48
                    if u_prev is not None and abs(u - u_prev) > 0.68:
                        u += 1.48 if u < u_prev else -1.48
                    u_prev = u
                    v_uv = (co.z - 1.00) * 2.15
                    loop[uv_lay].uv = (u, v_uv)
            else:
                sx_f = 1.0 if f_center.x >= 0.0 else -1.0
                sh_origin = Vector((sx_f * 0.155, -0.020, 1.330))
                wr_end = Vector((sx_f * 0.358, -0.105, 1.102))
                axis_u = (wr_end - sh_origin).normalized()
                up_ref = Vector((0.0, -1.0, 0.0))
                t1 = (up_ref - axis_u * up_ref.dot(axis_u)).normalized()
                t2 = axis_u.cross(t1).normalized()

                u_prev = None
                for loop in f.loops:
                    rel = loop.vert.co - sh_origin
                    along = rel.dot(axis_u)
                    ang = math.atan2(rel.dot(t2), rel.dot(t1))
                    u = 0.5 + (ang / (2.0 * math.pi)) * 0.74
                    if u_prev is not None and abs(u - u_prev) > 0.36:
                        u += 0.74 if u < u_prev else -0.74
                    u_prev = u
                    v_uv = 1.2 - along * 2.15
                    loop[uv_lay].uv = (u, v_uv)

        elif f_tag == 1:
            u_prev = None
            for loop in f.loops:
                co = loop.vert.co
                ang = math.atan2(co.x, -(co.y - (-0.025)))
                u = 0.5 + (ang / (2.0 * math.pi)) * 1.15
                if u_prev is not None and abs(u - u_prev) > 0.55:
                    u += 1.15 if u < u_prev else -1.15
                u_prev = u
                loop[uv_lay].uv = (u, (co.z - 1.33) * 2.8)

        elif f_tag in (2, 3):
            sx_f = 1.0 if f_tag == 2 else -1.0
            sh_origin = Vector((sx_f * 0.155, -0.020, 1.330))
            wr_end = Vector((sx_f * 0.358, -0.105, 1.102))
            axis_u = (wr_end - sh_origin).normalized()
            up_ref = Vector((0.0, -1.0, 0.0))
            t1 = (up_ref - axis_u * up_ref.dot(axis_u)).normalized()
            t2 = axis_u.cross(t1).normalized()
            u_prev = None
            for loop in f.loops:
                rel = loop.vert.co - sh_origin
                along = rel.dot(axis_u)
                ang = math.atan2(rel.dot(t2), rel.dot(t1))
                u = 0.5 + (ang / (2.0 * math.pi)) * 0.68
                if u_prev is not None and abs(u - u_prev) > 0.32:
                    u += 0.68 if u < u_prev else -0.68
                u_prev = u
                loop[uv_lay].uv = (u, 1.5 - along * 2.6)

        elif f_tag == 4:
            u_prev = None
            for loop in f.loops:
                co = loop.vert.co
                ang = math.atan2(co.x, -(co.y - (-0.045)))
                u = 0.5 + (ang / (2.0 * math.pi)) * 1.35
                if u_prev is not None and abs(u - u_prev) > 0.65:
                    u += 1.35 if u < u_prev else -1.35
                u_prev = u
                loop[uv_lay].uv = (u, (co.z - 0.98) * 2.6)
        else:
            for loop in f.loops:
                loop[uv_lay].uv = (0.5, 0.5)

    sw_mesh = bpy.data.meshes.new("Character_Sweater_Mesh")
    bm.to_mesh(sw_mesh)
    bm.free()

    sw_obj = bpy.data.objects.new("Character_Sweater", sw_mesh)
    collection.objects.link(sw_obj)

    bind_mesh_to_armature(sw_obj, arm_obj, sharpness=2.2)

    cs = sw_obj.modifiers.new(name="CorrectiveSmooth", type="CORRECTIVE_SMOOTH")
    cs.iterations = 8
    cs.factor = 0.50
    cs.smooth_type = "LENGTH_WEIGHTED"

    sub = sw_obj.modifiers.new(name="Subdivision", type="SUBSURF")
    sub.levels = 2
    sub.render_levels = 2
    sub.quality = 3

    tex_disp = bpy.data.textures.new("Sweater_Knit_Displace_Tex", type="IMAGE")
    tex_disp.image = knit_height_img
    tex_disp.extension = "REPEAT"

    disp = sw_obj.modifiers.new(name="KnitRelief", type="DISPLACE")
    disp.texture = tex_disp
    disp.texture_coords = "UV"
    disp.uv_layer = "UVMap"
    disp.mid_level = 0.45
    disp.strength = 0.0020  # 2.0 mm physical 3D chunky knit relief!

    bpy.context.view_layer.update()
    _ensure_zero_body_poke_through(body_obj)
    return sw_obj


def _ensure_zero_body_poke_through(body_obj: bpy.types.Object):
    """Slim strictly-covered torso/arm skin inward so posed sweater folds never clip."""
    bmesh_body = body_obj.data
    n = len(bmesh_body.vertices)
    rest_flat = np.zeros(n * 3, dtype=np.float64)
    bmesh_body.attributes["rest_pos"].data.foreach_get("vector", rest_flat)
    R = rest_flat.reshape((n, 3))

    cur_flat = np.zeros(n * 3, dtype=np.float64)
    bmesh_body.vertices.foreach_get("co", cur_flat)
    C = cur_flat.reshape((n, 3))

    norms = np.zeros(n * 3, dtype=np.float64)
    bmesh_body.vertices.foreach_get("normal", norms)
    N = norms.reshape((n, 3))

    ax = np.abs(R[:, 0])
    inside_torso = 1.0 - _smoothstep(1.315, 1.348, R[:, 2])
    inside_sleeve = _smoothstep(0.13, 0.17, ax)
    under_sweater = np.clip(np.maximum(inside_torso, inside_sleeve), 0.0, 1.0)

    C_safe = C - N * (0.0065 * under_sweater)[:, None]
    bmesh_body.vertices.foreach_set("co", C_safe.astype(np.float32).ravel())
    bmesh_body.update()
    bpy.context.view_layer.update()
