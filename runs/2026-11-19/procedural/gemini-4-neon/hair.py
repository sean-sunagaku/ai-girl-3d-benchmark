"""
hair.py - Multi-layer realistic 3D hair system using Blender 5.1 Curves (hair_curves):
  - Fitted inner scalp cap (zero bald gaps)
  - Scalp-hugging 360-degree layered wavy chestnut main hair with natural off-center part,
    smooth cranium contour wrap, left-ear tuck, and soft wavy back/shoulder drape
  - Signature organic wavy front-left hair cascade resting naturally over the dusty-pink sweater
  - Fine airy see-through wispy bangs emerging seamlessly from the crown + soft framing tendrils
  - Delicate sunlit flyaways
  - 3D anatomical eyebrows & curved upper/lower eyelashes rooted directly on eyelid margins
"""
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

from sculpt import get_head_matrix


def _build_bvh_from_objects(objs):
    """Build a combined world-space BVHTree from mesh objects (evaluated)."""
    bm = bmesh.new()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objs:
        if obj.type != 'MESH':
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        me = eval_obj.to_mesh()
        bm_tmp = bmesh.new()
        bm_tmp.from_mesh(me)
        bmesh.ops.transform(bm_tmp, matrix=eval_obj.matrix_world, verts=bm_tmp.verts)
        vmap = {}
        for v in bm_tmp.verts:
            vmap[v.index] = bm.verts.new(v.co)
        for f in bm_tmp.faces:
            try:
                bm.faces.new([vmap[v.index] for v in f.verts])
            except ValueError:
                pass
        bm_tmp.free()
        eval_obj.to_mesh_clear()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    return bvh


def _resolve_collisions_for_strands(strands_world, body_bvh, sweater_bvh,
                                    body_margin=0.0032, sweater_margin=0.0045,
                                    start_k=2, smooth_iters=2):
    """Push strand points cleanly outside body & sweater surfaces using BVHTree, with smoothing."""
    n_strands, n_pts, _ = strands_world.shape
    out = strands_world.copy()

    def push_pass():
        for i in range(n_strands):
            for k in range(start_k, n_pts):
                p = Vector(out[i, k])
                loc, norm, _, dist = body_bvh.find_nearest(p)
                if loc is not None and dist < 0.045:
                    if (p - loc).dot(norm) < body_margin:
                        p = loc + norm * body_margin
                if sweater_bvh is not None:
                    loc2, norm2, _, dist2 = sweater_bvh.find_nearest(p)
                    if loc2 is not None and dist2 < 0.055:
                        if (p - loc2).dot(norm2) < sweater_margin:
                            p = loc2 + norm2 * sweater_margin
                out[i, k] = (p.x, p.y, p.z)

    push_pass()
    for _ in range(smooth_iters):
        prev = out.copy()
        for k in range(max(1, start_k), n_pts - 1):
            out[:, k, :] = 0.25 * prev[:, k - 1, :] + 0.50 * prev[:, k, :] + 0.25 * prev[:, k + 1, :]
    push_pass()
    return out


def _create_curves_object(name, positions, radii, mat, collection):
    """Create a Blender 5.1 Curves (hair_curves) object from (N, K, 3) positions & (N, K) radii."""
    n_strands, n_pts, _ = positions.shape
    hd = bpy.data.hair_curves.new(f"{name}_Data")
    hd.add_curves([n_pts] * n_strands)

    hd.attributes["position"].data.foreach_set("vector", np.ascontiguousarray(positions, dtype=np.float32).ravel())
    rad_attr = hd.attributes.new("radius", 'FLOAT', 'POINT')
    rad_attr.data.foreach_set("value", np.ascontiguousarray(radii, dtype=np.float32).ravel())

    obj = bpy.data.objects.new(name, hd)
    if mat is not None:
        obj.data.materials.append(mat)
    collection.objects.link(obj)
    return obj


def _eval_polyline_smooth(ctrl_pts, n_samples):
    """Evaluate a smooth Catmull-Rom curve through 3D control points."""
    pts = np.asarray(ctrl_pts, dtype=np.float64)
    p_ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    n_seg = len(pts) - 1
    out = np.empty((n_samples, 3), dtype=np.float64)
    for k in range(n_samples):
        u = (k / (n_samples - 1)) * n_seg
        seg = min(int(math.floor(u)), n_seg - 1)
        t = u - seg
        p0, p1, p2, p3 = p_ext[seg], p_ext[seg + 1], p_ext[seg + 2], p_ext[seg + 3]
        t2, t3 = t * t, t * t * t
        out[k] = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    return out


def _build_scalp_cap(body_obj, canonical_verts, mat_scalp, collection):
    """Fitted matte dark-chestnut inner scalp cap strictly behind the hairline & above ears."""
    me_src = body_obj.data
    bm = bmesh.new()
    bm.from_mesh(me_src)
    bm.verts.ensure_lookup_table()

    keep_v = set()
    for i, v in enumerate(bm.verts):
        cx, cy, cz = canonical_verts[i]
        ax = abs(cx)
        front_hairline_z = 1.606 - 0.050 * min(1.0, (ax / 0.068) ** 2)
        in_top = (cz > front_hairline_z) and (cy > -0.142)
        in_back = (cy > -0.030) and (cz > 1.460) and (ax < 0.068 or cy > -0.015)
        in_temple = (cy > -0.095) and (cy <= -0.030) and (cz > 1.560) and (ax > 0.048)
        is_ear = (ax > 0.066) and (cz < 1.580) and (cy > -0.078) and (cy < -0.018)
        if (in_top or in_back or in_temple) and not is_ear:
            keep_v.add(v.index)

    del_faces = [f for f in bm.faces if not all(v.index in keep_v for v in f.verts)]
    bmesh.ops.delete(bm, geom=del_faces, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    # Seat scalp cap cleanly above skin (+1.0mm) and safely below hair roots (+2.2mm)
    for v in bm.verts:
        v.co += v.normal * 0.0010

    me = bpy.data.meshes.new("Hair_Scalp_Cap_Mesh")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True

    obj = bpy.data.objects.new("Hair_Scalp_Cap", me)
    if mat_scalp:
        obj.data.materials.append(mat_scalp)
    sub = obj.modifiers.new("Subsurf", 'SUBSURF')
    sub.levels = 1
    sub.render_levels = 1
    collection.objects.link(obj)
    return obj


def _sample_scalp_roots(canonical_verts, body_obj, n_roots, rng, mode="main"):
    """Sample jittered roots & outward normals directly from posed scalp triangles."""
    me = body_obj.data
    posed_v = np.array([v.co[:] for v in me.vertices], dtype=np.float64)
    tris = []
    areas = []
    for poly in me.polygons:
        vids = list(poly.vertices)
        cen = canonical_verts[vids].mean(axis=0)
        cx, cy, cz = cen
        ax = abs(cx)

        if mode == "main":
            front_z = 1.600 - 0.048 * min(1.0, (ax / 0.068) ** 2)
            in_top = (cz > front_z) and (cy > -0.144)
            in_back = (cy > -0.028) and (cz > 1.452)
            in_temple = (cy > -0.095) and (cy <= -0.028) and (cz > 1.548) and (ax > 0.046)
            is_ear = (ax > 0.066) and (cz < 1.580) and (cy > -0.078) and (cy < -0.018)
            ok = (in_top or in_back or in_temple) and not is_ear
        elif mode == "cascade_r":
            # Right front-side scalp (-X) feeding the signature front-chest wavy cascade
            ok = (cx < -0.012) and (cy > -0.128) and (cy < -0.015) and (cz > 1.540) and not (ax > 0.066 and cz < 1.578)
        elif mode == "cascade_l":
            # Left behind-ear scalp (+X) feeding soft drape behind exposed left ear onto left shoulder
            ok = (cx > 0.022) and (cy > -0.035) and (cy < 0.025) and (cz > 1.490) and not (ax > 0.066 and cz < 1.578)
        else:
            ok = False

        if not ok:
            continue

        sub_tris = [(vids[0], vids[1], vids[2])]
        if len(vids) == 4:
            sub_tris.append((vids[0], vids[2], vids[3]))
        for i0, i1, i2 in sub_tris:
            p0, p1, p2 = posed_v[i0], posed_v[i1], posed_v[i2]
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            if area > 1e-10:
                tris.append((i0, i1, i2))
                areas.append(area)

    tris = np.array(tris, dtype=np.int32)
    probs = np.array(areas, dtype=np.float64)
    probs /= probs.sum()

    chosen = rng.choice(len(tris), size=n_roots, p=probs)
    u = rng.random(n_roots)
    v = rng.random(n_roots)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v

    i0, i1, i2 = tris[chosen, 0], tris[chosen, 1], tris[chosen, 2]
    w_roots = w[:, None] * posed_v[i0] + u[:, None] * posed_v[i1] + v[:, None] * posed_v[i2]
    c_roots = w[:, None] * canonical_verts[i0] + u[:, None] * canonical_verts[i1] + v[:, None] * canonical_verts[i2]

    head_c = np.array(get_head_matrix() @ Vector((0.0, -0.04, 1.55)))
    raw_n = np.cross(posed_v[i1] - posed_v[i0], posed_v[i2] - posed_v[i0])
    raw_n /= (np.linalg.norm(raw_n, axis=1, keepdims=True) + 1e-12)
    outward = np.sum(raw_n * (w_roots - head_c), axis=1) < 0.0
    raw_n[outward] *= -1.0
    w_roots = w_roots + raw_n * 0.0022
    return w_roots, raw_n, c_roots


def _generate_main_volume_hair(body_obj, canonical_verts, body_bvh, sweater_bvh, mat_hair, collection):
    """
    Generate realistic scalp-hugging 360-degree long wavy chestnut hair (9,600 strands x 22 pts).
    Strands hug the natural cranium curvature closely (only ~7-12mm soft airy volume — never
    ballooned outward!), tuck cleanly behind the exposed left ear (+X) so the face profile is
    clear in side.png, and drape smoothly in layered chestnut S-waves over the shoulders & back.
    """
    rng = np.random.default_rng(1001)
    n_guides = 290
    children_per_guide = 40  # 11,600 fine silky strands
    n_pts = 22

    g_roots, g_norms, g_can = _sample_scalp_roots(canonical_verts, body_obj, n_guides, rng, mode="main")
    head_mat = get_head_matrix()
    head_inv = head_mat.inverted()
    head_rot = np.array(head_mat.to_3x3(), dtype=np.float64)

    # Canonical cranium ellipsoid (closely matches sculpted MakeHuman skull)
    c_skull = np.array([0.0, -0.042, 1.560], dtype=np.float64)
    r_skull = np.array([0.077, 0.098, 0.104], dtype=np.float64)

    # Subtle off-center natural part at canonical X = -0.012
    part_x = -0.012
    guides = np.zeros((n_guides, n_pts, 3), dtype=np.float64)
    t_arr = np.linspace(0.0, 1.0, n_pts)

    for g in range(n_guides):
        r0_w = g_roots[g]
        cx, cy, cz = g_can[g]

        side = 1.0 if (cx >= part_x) else -1.0
        # 0 at front hairline, 1 at back of skull
        back_w = max(0.0, min(1.0, (cy + 0.060) / 0.100))

        # Lush layered length: full mid-back cascade in back (~38-48cm), graceful shoulder length on sides
        length = rng.uniform(0.34, 0.46) * (0.86 + 0.22 * back_w)
        if cz < 1.50 and cy > 0.0:
            length *= rng.uniform(0.72, 0.92)

        # Gentle natural crown loft (5-11mm max — realistic airy volume!)
        loft = rng.uniform(0.0055, 0.0115)
        step_len = length / (n_pts - 1)

        p_w = r0_w.copy()
        guides[g, 0] = p_w
        vel_w = np.array([0.0, 0.0, -1.0])

        for k in range(1, n_pts):
            t = k / (n_pts - 1)
            p_can = np.array(head_inv @ Vector(p_w), dtype=np.float64)

            if p_can[2] > 1.505 and k < 10:
                # --- CRANIUM-HUGGING GEODESIC PHASE ---
                rel = p_can - c_skull
                surf_n = rel / (r_skull * r_skull)
                surf_n /= (np.linalg.norm(surf_n) + 1e-12)

                if back_w > 0.55:
                    # Back of head: glide smoothly downward & fan generously across full back width
                    des_can = np.array([cx * 2.1, 0.35, -1.0])
                elif side > 0:
                    # Viewer's right (+X): sweep gracefully OVER & BEHIND the left ear
                    behind_push = 0.82 if p_can[1] < -0.012 else 0.22
                    des_can = np.array([0.65 * (1.0 - 0.45 * back_w), behind_push, -0.78 - 0.35 * t])
                else:
                    # Viewer's left (-X): sweep smoothly down over right temple & cheek frame
                    des_can = np.array([-0.68 * (1.0 - 0.42 * back_w), 0.15, -0.82 - 0.35 * t])

                tan_can = des_can - np.dot(des_can, surf_n) * surf_n
                tan_can /= (np.linalg.norm(tan_can) + 1e-12)

                p_next_can = p_can + tan_can * step_len
                loft_k = loft * math.sin(min(1.0, k / 3.0) * math.pi * 0.5)
                rel_n = p_next_can - c_skull
                ell_d = np.linalg.norm(rel_n / r_skull)
                target_d = 1.0 + (loft_k / 0.095)
                if ell_d < target_d or k <= 5:
                    blend_shell = 0.85 if k <= 5 else 0.50
                    new_d = ell_d * (1.0 - blend_shell) + target_d * blend_shell
                    p_next_can = c_skull + rel_n * (new_d / (ell_d + 1e-8))

                p_next_w = np.array(head_mat @ Vector(p_next_can), dtype=np.float64)
                vel_w = p_next_w - p_w
                vel_w /= (np.linalg.norm(vel_w) + 1e-12)
                p_w = p_next_w
            else:
                # --- NATURAL GRAVITY DRAPE PHASE ---
                grav = np.array([0.0, 0.0, -1.0])
                if back_w > 0.35 or p_w[1] > -0.01:
                    # Gentle clear over back ribbed collar, then rest snugly along sweater back!
                    collar_clear = 0.22 if (1.35 < p_w[2] < 1.46) else -0.06
                    fan_x = 0.22 * math.tanh(cx * 14.0) * max(0.0, 1.0 - t * 1.1)
                    steer = np.array([fan_x, collar_clear, 0.0])
                elif side > 0:
                    steer = np.array([0.10, 0.26 if p_w[2] > 1.38 else 0.04, 0.0])
                else:
                    steer = np.array([-0.10, -0.03 if p_w[2] < 1.44 else 0.05, 0.0])

                target_dir = grav + steer
                target_dir /= (np.linalg.norm(target_dir) + 1e-12)
                vel_w = 0.52 * vel_w + 0.48 * target_dir
                vel_w /= (np.linalg.norm(vel_w) + 1e-12)
                p_w = p_w + vel_w * step_len

                # Compact back-collar clearance strictly at collar band (z in [1.34, 1.43]) so hair stays snug to sweater back
                if (back_w > 0.30 or side > 0) and p_w[1] > -0.015 and (1.34 < p_w[2] < 1.43):
                    min_back_y = 0.088 * math.cos(min(1.25, abs(p_w[0]) * 5.0)) + 0.008
                    if p_w[1] < min_back_y and abs(p_w[0]) < 0.165:
                        p_w[1] = p_w[1] * 0.30 + min_back_y * 0.70

            guides[g, k] = p_w

        # Add soft, natural chestnut S-waves below ear level (zero wave near scalp!)
        wave_env = np.clip((t_arr - 0.28) / 0.72, 0.0, 1.0) ** 1.35
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amp = rng.uniform(0.006, 0.014)
        freq = rng.uniform(1.8, 2.9)
        guides[g, :, 0] += np.sin(t_arr * freq * 2.0 * math.pi + phase) * amp * wave_env
        guides[g, :, 1] += np.cos(t_arr * freq * 1.6 * math.pi + phase * 1.2) * (amp * 0.65) * wave_env

    # Collision pass on guides
    guides = _resolve_collisions_for_strands(guides, body_bvh, sweater_bvh, body_margin=0.0045, sweater_margin=0.0065, start_k=2, smooth_iters=2)

    # Spawn silky clumped child strands around each guide
    n_total = n_guides * children_per_guide
    strands = np.empty((n_total, n_pts, 3), dtype=np.float64)
    radii = np.empty((n_total, n_pts), dtype=np.float64)

    idx = 0
    for g in range(n_guides):
        base = guides[g]
        tangs = np.gradient(base, axis=0)
        tangs /= (np.linalg.norm(tangs, axis=1, keepdims=True) + 1e-12)
        ref = np.array([1.0, 0.0, 0.0]) if abs(tangs[0, 0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        b1 = np.cross(tangs, ref)
        b1 /= (np.linalg.norm(b1, axis=1, keepdims=True) + 1e-12)
        b2 = np.cross(tangs, b1)

        for _ in range(children_per_guide):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            # Tight natural clumps (6-11mm at root, sleek mid-clump, feathered wavy tips)
            r_clump = rng.uniform(0.0015, 0.0105) * math.sqrt(rng.random())
            env = 0.60 * (1.0 - t_arr) ** 0.8 + 0.42 * np.sin(t_arr * math.pi) + 0.55 * (t_arr ** 2.2)
            twist = rng.uniform(-1.6, 1.6) * t_arr
            jitter = rng.uniform(0.0006, 0.0028) * np.sin(t_arr * rng.uniform(4.0, 8.0) + rng.uniform(0, 6.28)) * (t_arr ** 1.1)

            offset = b1 * (np.cos(ang + twist) * r_clump * env + jitter)[:, None] + b2 * (np.sin(ang + twist) * r_clump * env)[:, None]
            # Slight length feathering
            l_sc = rng.uniform(0.84, 1.0)
            t_sub = t_arr * l_sc * (n_pts - 1)
            k0 = np.clip(np.floor(t_sub).astype(np.int32), 0, n_pts - 2)
            frac = (t_sub - k0)[:, None]
            p_sub = base[k0] * (1.0 - frac) + base[k0 + 1] * frac

            strands[idx] = p_sub + offset
            # Fine realistic hair strand thickness (~0.24-0.36mm root tapering to 0.05mm fine tips)
            r0 = rng.uniform(0.00022, 0.00035)
            radii[idx] = r0 * (1.0 - 0.82 * (t_arr ** 1.5))
            idx += 1

    strands = _resolve_collisions_for_strands(strands, body_bvh, sweater_bvh, body_margin=0.0026, sweater_margin=0.0038, start_k=2, smooth_iters=1)
    return _create_curves_object("Hair_Main_Volume", strands, radii, mat_hair, collection)


def _generate_front_cascade(body_obj, canonical_verts, body_bvh, sweater_bvh, mat_hair, collection):
    """
    Signature flowing chestnut wavy front hair cascade draping naturally from the right side
    of the head (-X, viewer's left), curving softly over the right shoulder and sweeping
    in 6 organic wavy locks across the front-left of the dusty-pink sweater (matching reference.png!).
    """
    rng = np.random.default_rng(2026)
    n_pts = 24
    t_arr = np.linspace(0.0, 1.0, n_pts)
    all_strands = []
    all_radii = []

    # 5 organic wavy lock paths fanning gracefully over right shoulder & front-left chest
    lock_specs = [
        # (root_off_x, sh_x, sh_y, sh_z, mid_x, mid_y, mid_z, tip_x, tip_y, tip_z, n_guides, wave_amp)
        (-0.045, -0.152, -0.082, 1.376, -0.104, -0.155, 1.245, -0.072, -0.160, 1.095, 7, 0.016),
        (-0.038, -0.130, -0.098, 1.372, -0.074, -0.162, 1.235, -0.042, -0.164, 1.070, 9, 0.018),
        (-0.032, -0.110, -0.108, 1.368, -0.050, -0.166, 1.225, -0.018, -0.165, 1.055, 10, 0.017),
        (-0.025, -0.094, -0.114, 1.365, -0.034, -0.166, 1.215, -0.004, -0.160, 1.080, 8, 0.015),
        (-0.052, -0.168, -0.062, 1.372, -0.136, -0.135, 1.250, -0.114, -0.142, 1.135, 6, 0.014),
    ]

    g_roots_r, _, _ = _sample_scalp_roots(canonical_verts, body_obj, 45, rng, mode="cascade_r")
    g_roots_l, _, _ = _sample_scalp_roots(canonical_verts, body_obj, 12, rng, mode="cascade_l")
    r_idx = 0
    l_idx = 0

    for spec in lock_specs:
        rox, shx, shy, shz, mx, my, mz, tx, ty, tz, n_g, w_amp = spec
        is_left = rox > 0
        lock_guides = np.zeros((n_g, n_pts, 3), dtype=np.float64)
        base_phase = rng.uniform(0.0, 2.0 * math.pi)

        for g in range(n_g):
            if is_left:
                r0 = g_roots_l[l_idx % len(g_roots_l)]
                l_idx += 1
                # Stay cleanly behind exposed left ear before curving forward onto shoulder
                p1 = np.array([r0[0] + 0.024, max(r0[1] + 0.020, -0.008), r0[2] - 0.090])
            else:
                r0 = g_roots_r[r_idx % len(g_roots_r)]
                r_idx += 1
                # Clear right cheek cleanly so zero dark hair clips into cheekbone!
                p1 = np.array([r0[0] - 0.032, r0[1] + 0.002, r0[2] - 0.088])

            jit = rng.normal(0.0, 0.011, size=3)
            p2 = np.array([shx, shy, shz]) + jit * np.array([1.0, 0.7, 0.6])
            p3 = np.array([mx, my, mz]) + jit * np.array([1.2, 0.7, 0.9])
            p4 = np.array([tx, ty, tz]) + jit * np.array([1.4, 0.7, 1.5])

            pts = _eval_polyline_smooth(np.array([r0, p1, p2, p3, p4]), n_pts)
            # Soft romantic chestnut S-waves synchronized per lock with gentle per-guide phase variance
            ph = base_phase + rng.uniform(-0.45, 0.45)
            env = np.clip((t_arr - 0.16) / 0.84, 0.0, 1.0) ** 1.25
            pts[:, 0] += np.sin(t_arr * 2.5 * 2.0 * math.pi + ph) * w_amp * env
            pts[:, 1] += np.cos(t_arr * 2.1 * 2.0 * math.pi + ph) * (w_amp * 0.40) * env
            lock_guides[g] = pts

        lock_guides = _resolve_collisions_for_strands(lock_guides, body_bvh, sweater_bvh, body_margin=0.0080, sweater_margin=0.0065, start_k=2, smooth_iters=2)

        n_ch = 36 if not is_left else 28
        for g in range(n_g):
            base = lock_guides[g]
            tangs = np.gradient(base, axis=0)
            tangs /= (np.linalg.norm(tangs, axis=1, keepdims=True) + 1e-12)
            b1 = np.cross(tangs, np.array([0.0, 1.0, 0.0]))
            b1 /= (np.linalg.norm(b1, axis=1, keepdims=True) + 1e-12)
            b2 = np.cross(tangs, b1)
            for _ in range(n_ch):
                ang = rng.uniform(0.0, 2.0 * math.pi)
                r_c = rng.uniform(0.0012, 0.0085) * math.sqrt(rng.random())
                # Tapered silky lock ends (classic curled wavy hair lock silhouette!)
                env = 0.55 * (1.0 - t_arr) ** 0.5 + 0.55 * np.sin(t_arr * math.pi) + 0.25 * (t_arr ** 2)
                twist = rng.uniform(-2.4, 2.4) * t_arr
                offset = b1 * (np.cos(ang + twist) * r_c * env)[:, None] + b2 * (np.sin(ang + twist) * r_c * env)[:, None]

                l_sc = rng.uniform(0.80, 1.0)
                t_sub = t_arr * l_sc * (n_pts - 1)
                k0 = np.clip(np.floor(t_sub).astype(np.int32), 0, n_pts - 2)
                frac = (t_sub - k0)[:, None]
                p_sub = base[k0] * (1.0 - frac) + base[k0 + 1] * frac

                all_strands.append(p_sub + offset)
                r0_rad = rng.uniform(0.00021, 0.00033)
                all_radii.append(r0_rad * (1.0 - 0.85 * (t_arr ** 1.45)))

    strands = np.array(all_strands, dtype=np.float64)
    radii = np.array(all_radii, dtype=np.float64)
    strands = _resolve_collisions_for_strands(strands, body_bvh, sweater_bvh, body_margin=0.0055, sweater_margin=0.0045, start_k=2, smooth_iters=1)
    return _create_curves_object("Hair_Front_Cascade", strands, radii, mat_hair, collection)


def _generate_wispy_bangs_and_tendrils(body_obj, canonical_verts, body_bvh, mat_hair, collection):
    """
    Generate ultra-natural, delicate see-through wispy bangs emerging seamlessly from the
    crown/part (zero hairline gap!), hugging gently 4-7mm off the forehead and sweeping softly
    toward viewer's right across the forehead & left brow (matching input/reference.png!).
    """
    rng = np.random.default_rng(3030)
    head_mat = get_head_matrix()
    n_pts = 15
    t_arr = np.linspace(0.0, 1.0, n_pts)

    strands = []
    radii = []

    # Soft crown-to-hairline smooth drape layer + airy asymmetric see-through wispy bangs
    # Rooted high inside crown/part (z=1.630..1.648) so ZERO choppy horizontal hairline band ever appears!
    bang_clusters = [
        # (root_x, root_y, root_z, tip_x, tip_y, tip_z, n_strands, width)
        # Soft crown-part sweep left & right to smoothly blanket front hairline
        (-0.016, -0.102, 1.646, -0.068, -0.108, 1.525, 160, 0.0095),
        (-0.010, -0.100, 1.648,  0.072, -0.078, 1.520, 210, 0.0105),
        (-0.004, -0.108, 1.644,  0.068, -0.095, 1.515, 150, 0.0085),
        # Delicate airy see-through wispy bangs with varied lengths & soft rightward curve (matching reference.png!)
        (-0.036, -0.116, 1.632, -0.052, -0.146, 1.544, 34, 0.0036),
        (-0.025, -0.118, 1.636, -0.015, -0.160, 1.564, 28, 0.0030),
        (-0.017, -0.118, 1.638,  0.000, -0.163, 1.558, 32, 0.0030),
        (-0.011, -0.118, 1.638,  0.015, -0.162, 1.566, 26, 0.0028),
        (-0.005, -0.118, 1.638,  0.027, -0.158, 1.557, 34, 0.0032),
        ( 0.003, -0.117, 1.636,  0.041, -0.151, 1.549, 44, 0.0038),
        ( 0.012, -0.114, 1.632,  0.054, -0.139, 1.536, 56, 0.0044),
        ( 0.022, -0.108, 1.626,  0.064, -0.122, 1.522, 54, 0.0046),
    ]

    for rx, ry, rz, tx, ty, tz, n_ch, spread in bang_clusters:
        # Smooth natural forehead arch: lift only ~5-7mm off forehead curve
        mx1 = 0.72 * rx + 0.28 * tx
        my1 = min(ry, -0.140) - 0.007
        mz1 = 0.72 * rz + 0.28 * tz + 0.005
        mx2 = 0.35 * rx + 0.65 * tx
        my2 = min(ty, -0.150) - 0.005
        mz2 = 0.35 * rz + 0.65 * tz + 0.002
        ctrl_can = np.array([[rx, ry, rz], [mx1, my1, mz1], [mx2, my2, mz2], [tx, ty, tz]], dtype=np.float64)
        base_can = _eval_polyline_smooth(ctrl_can, n_pts)

        for _ in range(n_ch):
            dx_r = rng.normal(0.0, spread * 0.95)
            dy_r = rng.normal(0.0, spread * 0.30)
            dz_r = rng.normal(0.0, spread * 0.35)
            dx_t = rng.normal(0.0, spread * 0.55)
            dz_t = rng.normal(0.0, 0.008)  # Staggered airy wispy tips
            bow_y = rng.uniform(-0.0022, 0.0014)

            p_can = base_can.copy()
            p_can[:, 0] += dx_r * (1.0 - t_arr) + dx_t * t_arr
            p_can[:, 1] += dy_r * (1.0 - t_arr) + bow_y * np.sin(t_arr * math.pi)
            p_can[:, 2] += dz_r * (1.0 - t_arr) + dz_t * (t_arr ** 1.2)

            p_w = np.array([head_mat @ Vector(pt) for pt in p_can], dtype=np.float64)
            strands.append(p_w)
            # Whisper-fine silky bang strands (0.12-0.20mm root -> 0.028mm delicate tip)
            r0 = rng.uniform(0.00012, 0.00020)
            radii.append(r0 * (1.0 - 0.85 * (t_arr ** 1.35)))

    # Delicate cheek-framing side tendril on right side (-X), kept cleanly outside cheekbone
    tendril_r = _eval_polyline_smooth(
        np.array([[-0.062, -0.095, 1.595], [-0.080, -0.108, 1.530], [-0.078, -0.114, 1.465], [-0.068, -0.108, 1.410]]),
        n_pts,
    )
    for _ in range(38):
        off = rng.normal(0.0, 0.0030, size=(1, 3)) * np.stack([1.0 - 0.35 * t_arr] * 3, axis=-1)
        p_can = tendril_r + off
        p_can[:, 0] += np.sin(t_arr * 5.0 + rng.uniform(0, 6.28)) * 0.002 * t_arr
        p_w = np.array([head_mat @ Vector(pt) for pt in p_can], dtype=np.float64)
        strands.append(p_w)
        r0 = rng.uniform(0.00015, 0.00023)
        radii.append(r0 * (1.0 - 0.84 * (t_arr ** 1.4)))

    strands = np.array(strands, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)
    strands = _resolve_collisions_for_strands(strands, body_bvh, None, body_margin=0.0044, start_k=1, smooth_iters=1)
    return _create_curves_object("Hair_Bangs_Wispy", strands, radii, mat_hair, collection)


def _generate_flyaways(body_obj, canonical_verts, body_bvh, sweater_bvh, mat_hair, collection):
    """Generate ~220 subtle ultra-fine backlit flyaways rooted strictly on outer/back scalp."""
    rng = np.random.default_rng(4040)
    n_cand = 360
    n_pts = 14
    roots, norms, can = _sample_scalp_roots(canonical_verts, body_obj, n_cand, rng, mode="main")
    head_mat = get_head_matrix()
    head_inv = head_mat.inverted()
    c_skull = np.array([0.0, -0.042, 1.560])
    r_skull = np.array([0.088, 0.110, 0.114])

    strands_list = []
    radii_list = []
    t_arr = np.linspace(0.0, 1.0, n_pts)

    for i in range(n_cand):
        r0 = roots[i]
        cx, cy, cz = can[i]
        # Cleanly skip front-face forehead candidates instead of displacing into mid-air
        if cy < -0.085 and abs(cx) < 0.052:
            continue
        side = 1.0 if cx >= -0.012 else -1.0
        length = rng.uniform(0.14, 0.26)
        step = length / (n_pts - 1)
        pos = r0.copy()
        pts = np.zeros((n_pts, 3), dtype=np.float64)
        pts[0] = pos
        vel = np.array([side * 0.22, 0.22 if side > 0 else 0.0, -0.88])
        vel /= np.linalg.norm(vel)
        phase = rng.uniform(0.0, 6.28)
        freq = rng.uniform(3.0, 6.0)
        for k in range(1, n_pts):
            p_can = np.array(head_inv @ Vector(pos))
            if p_can[2] > 1.50 and k < 6:
                rel = p_can - c_skull
                ell_d = np.linalg.norm(rel / r_skull)
                if ell_d < 1.03:
                    p_can = c_skull + rel * (1.03 / (ell_d + 1e-8))
                pos = np.array(head_mat @ Vector(p_can))
            target = np.array([side * 0.10, 0.16 if side > 0 else 0.0, -0.96])
            target /= np.linalg.norm(target)
            vel = 0.65 * vel + 0.35 * target
            vel /= np.linalg.norm(vel)
            pos = pos + vel * step
            pts[k] = pos

        pts[:, 0] += np.sin(t_arr * freq + phase) * 0.005 * t_arr
        pts[:, 1] += np.cos(t_arr * freq * 0.8 + phase) * 0.004 * t_arr
        strands_list.append(pts)
        radii_list.append(rng.uniform(0.00008, 0.00013) * (1.0 - 0.80 * t_arr))

    strands = np.array(strands_list, dtype=np.float64)
    radii = np.array(radii_list, dtype=np.float64)
    strands = _resolve_collisions_for_strands(strands, body_bvh, sweater_bvh, body_margin=0.005, sweater_margin=0.007, start_k=2, smooth_iters=1)
    return _create_curves_object("Hair_Flyaways", strands, radii, mat_hair, collection)


def _generate_eyebrows(eye_c_l, eye_c_r, body_bvh, mat_brows, collection):
    """3D anatomical soft-arched eyebrows rooted directly onto the posed brow ridge surface."""
    rng = np.random.default_rng(5050)
    head_mat = get_head_matrix()
    head_rot = np.array(head_mat.to_3x3(), dtype=np.float64)
    n_pts = 6
    t_arr = np.linspace(0.0, 1.0, n_pts)

    strands = []
    radii = []

    for side, ec in [(1.0, eye_c_l), (-1.0, eye_c_r)]:
        n_hairs = 360
        for _ in range(n_hairs):
            u = rng.beta(1.30, 1.35)
            sx = -0.0155 + u * 0.0415  # Full graceful brow span extending softly past outer canthus
            arch_z = ec[2] + 0.0168 + 0.0020 * math.sin(u * math.pi * 0.88) - 0.0018 * max(u - 0.65, 0.0)
            half_h = 0.0029 * (1.0 - 0.55 * u) * min(1.0, u / 0.14 + 0.25) * min(1.0, (1.0 - u) / 0.16 + 0.12)
            v_off = rng.uniform(-1.0, 1.0)
            z_can = arch_z + v_off * half_h
            x_can = ec[0] + side * sx
            y_can = ec[1] - 0.025

            probe_w = head_mat @ Vector((x_can, y_can, z_can))
            loc, norm, _, _ = body_bvh.find_nearest(probe_w)
            if loc is None:
                continue
            root = np.array(loc + norm * 0.00022, dtype=np.float64)
            n_surf = np.array(norm, dtype=np.float64)

            if u < 0.22:
                dir_can = np.array([side * (0.45 + 1.5 * u), -0.08, 0.85 - 1.2 * u])
            elif u < 0.70:
                dir_can = np.array([side * 0.94, -0.04, 0.28 * (0.4 - v_off) + 0.10])
            else:
                dir_can = np.array([side * 0.92, 0.12, -0.26 - 0.12 * v_off])

            dir_w = head_rot @ dir_can
            dir_tan = dir_w - np.dot(dir_w, n_surf) * n_surf
            dir_tan /= (np.linalg.norm(dir_tan) + 1e-12)

            hair_len = rng.uniform(0.0044, 0.0074) * (1.0 - 0.26 * u)
            pts = np.empty((n_pts, 3), dtype=np.float64)
            for k in range(n_pts):
                tk = t_arr[k]
                lift = 0.00075 * math.sin(tk * math.pi * 0.82)
                p = root + dir_tan * (hair_len * tk) + n_surf * lift
                loc2, norm2, _, dist2 = body_bvh.find_nearest(Vector(p))
                if loc2 is not None and dist2 < 0.01:
                    if (Vector(p) - loc2).dot(norm2) < 0.00025:
                        p = np.array(loc2 + norm2 * 0.00025)
                pts[k] = p
            strands.append(pts)
            r0 = rng.uniform(0.000095, 0.000150)
            radii.append(r0 * (1.0 - 0.80 * (t_arr ** 1.3)))

    return _create_curves_object(
        "Eyebrows_Curves",
        np.array(strands, dtype=np.float64),
        np.array(radii, dtype=np.float64),
        mat_brows,
        collection,
    )


def _generate_eyelashes(eye_landmarks, eye_c_l, eye_c_r, mat_lashes, collection):
    """Curved 3D upper and lower eyelashes rooted precisely on extracted eyelid margins."""
    rng = np.random.default_rng(6060)
    head_mat = get_head_matrix()
    n_pts = 7
    t_arr = np.linspace(0.0, 1.0, n_pts)

    strands = []
    radii = []

    for side_name, side, ec in [("L", 1.0, eye_c_l), ("R", -1.0, eye_c_r)]:
        up_pts = eye_landmarks.get(f"upper_{side_name}", np.zeros((0, 3)))
        if len(up_pts) >= 4:
            order = np.argsort((up_pts[:, 0] - ec[0]) * side)
            up_pts = up_pts[order]
            n_up = 155
            for i in range(n_up):
                u = np.clip((i + rng.uniform(-0.35, 0.35)) / (n_up - 1), 0.02, 0.98)
                idx_f = u * (len(up_pts) - 1)
                k0 = min(int(math.floor(idx_f)), len(up_pts) - 2)
                frac = idx_f - k0
                root_can = up_pts[k0] * (1.0 - frac) + up_pts[k0 + 1] * frac
                root_can = root_can + np.array([0.0, -0.00025, 0.00010]) + rng.normal(0.0, 0.00014, 3)

                len_env = 0.55 + 0.45 * math.sin(min(1.0, u * 1.12) * math.pi * 0.85)
                lash_len = rng.uniform(0.0056, 0.0076) * len_env
                fan_x = side * (-0.22 + 0.78 * u) + rng.normal(0.0, 0.07)

                pts_can = np.empty((n_pts, 3), dtype=np.float64)
                p = root_can.copy()
                pts_can[0] = p
                curl_up = rng.uniform(0.75, 1.12)
                for k in range(1, n_pts):
                    tk = t_arr[k]
                    d = np.array([
                        fan_x * (0.45 + 0.55 * tk),
                        -(1.0 - 0.52 * tk * tk),
                        -0.16 + curl_up * (tk ** 1.25),
                    ])
                    d /= (np.linalg.norm(d) + 1e-12)
                    p = p + d * (lash_len / (n_pts - 1))
                    pts_can[k] = p

                p_w = np.array([head_mat @ Vector(pt) for pt in pts_can], dtype=np.float64)
                strands.append(p_w)
                r0 = rng.uniform(0.00010, 0.00015)
                radii.append(r0 * (1.0 - 0.85 * (t_arr ** 1.2)))

        lo_pts = eye_landmarks.get(f"lower_{side_name}", np.zeros((0, 3)))
        if len(lo_pts) >= 4:
            order = np.argsort((lo_pts[:, 0] - ec[0]) * side)
            lo_pts = lo_pts[order]
            n_lo = 72
            for i in range(n_lo):
                u = np.clip((i + rng.uniform(-0.35, 0.35)) / (n_lo - 1), 0.12, 0.96)
                idx_f = u * (len(lo_pts) - 1)
                k0 = min(int(math.floor(idx_f)), len(lo_pts) - 2)
                frac = idx_f - k0
                root_can = lo_pts[k0] * (1.0 - frac) + lo_pts[k0 + 1] * frac
                root_can = root_can + np.array([0.0, -0.00025, -0.00022])

                lash_len = rng.uniform(0.0025, 0.0038) * (0.65 + 0.35 * math.sin(u * math.pi))
                fan_x = side * (-0.15 + 0.65 * u) + rng.normal(0.0, 0.08)
                pts_can = np.empty((n_pts, 3), dtype=np.float64)
                p = root_can.copy()
                pts_can[0] = p
                for k in range(1, n_pts):
                    tk = t_arr[k]
                    d = np.array([fan_x * 0.55, -(1.0 - 0.35 * tk), -0.32 - 0.45 * tk])
                    d /= (np.linalg.norm(d) + 1e-12)
                    p = p + d * (lash_len / (n_pts - 1))
                    pts_can[k] = p

                p_w = np.array([head_mat @ Vector(pt) for pt in pts_can], dtype=np.float64)
                strands.append(p_w)
                r0 = rng.uniform(0.000065, 0.000095)
                radii.append(r0 * (1.0 - 0.85 * (t_arr ** 1.2)))

    return _create_curves_object(
        "Eyelashes_Curves",
        np.array(strands, dtype=np.float64),
        np.array(radii, dtype=np.float64),
        mat_lashes,
        collection,
    )


def build_all_hair(body_info, sweater_info, mats, collection):
    """Build scalp cap, scalp-hugging wavy hair, front cascade, wispy bangs, flyaways, brows, and lashes."""
    body_obj = body_info["body_obj"]
    canonical_verts = body_info["canonical_verts"]
    eye_c_l = body_info["eye_c_l"]
    eye_c_r = body_info["eye_c_r"]
    eye_landmarks = body_info["eye_landmarks"]

    body_bvh = _build_bvh_from_objects([body_obj])
    sweater_mesh_objs = [o for o in sweater_info["all"] if o.type == 'MESH']
    sweater_bvh = _build_bvh_from_objects(sweater_mesh_objs)

    scalp_cap = _build_scalp_cap(body_obj, canonical_verts, mats["scalp"], collection)
    main_hair = _generate_main_volume_hair(body_obj, canonical_verts, body_bvh, sweater_bvh, mats["hair"], collection)
    cascade = _generate_front_cascade(body_obj, canonical_verts, body_bvh, sweater_bvh, mats["hair"], collection)
    bangs = _generate_wispy_bangs_and_tendrils(body_obj, canonical_verts, body_bvh, mats["hair"], collection)
    flyaways = _generate_flyaways(body_obj, canonical_verts, body_bvh, sweater_bvh, mats["hair"], collection)
    brows = _generate_eyebrows(eye_c_l, eye_c_r, body_bvh, mats["brows"], collection)
    lashes = _generate_eyelashes(eye_landmarks, eye_c_l, eye_c_r, mats["lashes"], collection)

    return {
        "scalp_cap": scalp_cap,
        "main_hair": main_hair,
        "cascade": cascade,
        "bangs": bangs,
        "flyaways": flyaways,
        "brows": brows,
        "lashes": lashes,
        "all": [scalp_cap, main_hair, cascade, bangs, flyaways, brows, lashes],
    }
