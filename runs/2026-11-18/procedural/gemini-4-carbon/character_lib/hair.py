"""Stratified mesh-surface follicle sampling & collision-aware wavy 3D hair curves, airy bangs, brows, and lashes."""

import math
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _smoothstep_scalar(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-12)))
    return t * t * (3.0 - 2.0 * t)


def _create_hair_curves_object(
    name: str,
    strands: list,
    radii_list: list,
    collection,
    surface_obj: bpy.types.Object = None,
) -> bpy.types.Object:
    hc = bpy.data.hair_curves.new(f"{name}_Data")
    lengths = [len(s) for s in strands]
    hc.add_curves(lengths)
    hc.set_types(type="CATMULL_ROM")

    all_pts = np.concatenate(strands, axis=0).astype(np.float32)
    all_rad = np.concatenate(radii_list, axis=0).astype(np.float32)

    hc.points.foreach_set("position", all_pts.ravel())
    hc.points.foreach_set("radius", all_rad.ravel())

    if surface_obj is not None:
        hc.surface = surface_obj

    obj = bpy.data.objects.new(name, hc)
    collection.objects.link(obj)
    return obj


def _head_transform(arm_obj: bpy.types.Object):
    pb = arm_obj.pose.bones["head"]
    rest_m = arm_obj.matrix_world @ pb.bone.matrix_local
    pose_m = arm_obj.matrix_world @ pb.matrix
    return pose_m @ rest_m.inverted(), pose_m.to_3x3() @ pb.bone.matrix_local.to_3x3().inverted()


def _push_outside_colliders(
    pt: Vector,
    bvh_body: BVHTree,
    bvh_sw: BVHTree,
    clearance_body=0.0020,
    clearance_sw=0.0032,
) -> Vector:
    p = pt.copy()
    loc_b, norm_b, _, _ = bvh_body.find_nearest(p, 0.055)
    if loc_b is not None and norm_b.length > 0.5:
        signed_b = (p - loc_b).dot(norm_b)
        if signed_b < clearance_body:
            p = loc_b + norm_b * clearance_body

    loc_s, norm_s, _, _ = bvh_sw.find_nearest(p, 0.065)
    if loc_s is not None and norm_s.length > 0.5:
        signed_s = (p - loc_s).dot(norm_s)
        if signed_s < clearance_sw:
            p = loc_s + norm_s * clearance_sw
    return p


def _smooth_strand(pts: np.ndarray, passes: int = 2) -> np.ndarray:
    out = pts.copy()
    for _ in range(passes):
        nxt = out.copy()
        nxt[1:-1] = 0.25 * out[:-2] + 0.50 * out[1:-1] + 0.25 * out[2:]
        out = nxt
    return out


def _taper_radii(n_pts: int, r_root: float, r_tip: float, root_taper: float = 0.10) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
    r_in = np.clip(t / max(root_taper, 1e-4), 0.50, 1.0)
    taper = (1.0 - (t ** 1.4)) * r_root + (t ** 1.4) * r_tip
    return (taper * r_in).astype(np.float32)


def _resample_polyline(ctrl_pts: list, n_out: int) -> np.ndarray:
    arr = np.array([[p.x, p.y, p.z] for p in ctrl_pts], dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if cum[-1] < 1e-8:
        return np.repeat(arr[:1], n_out, axis=0).astype(np.float32)
    cum /= cum[-1]
    t_out = np.linspace(0.0, 1.0, n_out)
    out = np.stack([np.interp(t_out, cum, arr[:, k]) for k in range(3)], axis=-1)
    return out.astype(np.float32)


def _extract_scalp_triangles(ev_body: bpy.types.Object):
    """Extract all scalp triangles on posed Character_Body with exact cumulative CDF for stratified sampling."""
    em = ev_body.to_mesh()
    em.calc_loop_triangles()

    n_v = len(em.vertices)
    w_mat = ev_body.matrix_world
    world_co = np.array([tuple(w_mat @ v.co) for v in em.vertices], dtype=np.float64)
    rot_w = w_mat.to_3x3()
    world_no = np.array([tuple((rot_w @ v.normal).normalized()) for v in em.vertices], dtype=np.float64)

    rest_co = np.zeros(n_v * 3, dtype=np.float64)
    em.attributes["rest_pos"].data.foreach_get("vector", rest_co)
    rest_co = rest_co.reshape((n_v, 3))

    scalp_m = np.zeros(n_v, dtype=np.float64)
    em.attributes["scalp_mask"].data.foreach_get("value", scalp_m)

    tris_w, tris_n, tris_r, weights = [], [], [], []
    for lt in em.loop_triangles:
        v0, v1, v2 = lt.vertices
        sm = (scalp_m[v0] + scalp_m[v1] + scalp_m[v2]) / 3.0
        if sm < 0.20:
            continue
        p0, p1, p2 = world_co[v0], world_co[v1], world_co[v2]
        r0, r1, r2 = rest_co[v0], rest_co[v1], rest_co[v2]
        rc = (r0 + r1 + r2) / 3.0

        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if area < 1e-10:
            continue

        crown_boost = 1.0
        if rc[2] > 1.595:
            crown_boost = 2.2
        elif abs(rc[0]) > 0.044 and rc[2] > 1.545:
            crown_boost = 1.8

        tris_w.append((p0, p1, p2))
        tris_n.append((world_no[v0], world_no[v1], world_no[v2]))
        tris_r.append((r0, r1, r2))
        weights.append(area * sm * crown_boost)

    ev_body.to_mesh_clear()
    probs = np.array(weights, dtype=np.float64)
    probs /= np.sum(probs)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    return tris_w, tris_n, tris_r, cdf


def build_all_hair_brows_lashes(
    body_obj: bpy.types.Object,
    sw_obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    hair_collection,
) -> dict:
    """Build lush, natural-silhouette wavy chestnut hair with stratified zero-gap scalp coverage."""
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    ev_body = body_obj.evaluated_get(deps)
    ev_sw = sw_obj.evaluated_get(deps)

    bvh_body = BVHTree.FromObject(ev_body, deps)
    bvh_sw = BVHTree.FromObject(ev_sw, deps)

    M_head, R_head = _head_transform(arm_obj)
    M_head_inv = M_head.inverted()
    rng = np.random.default_rng(20261118)

    tris_w, tris_n, tris_r, tri_cdf = _extract_scalp_triangles(ev_body)
    part_x = -0.008

    def sample_stratified_follicles(count: int):
        """Generate `count` evenly-stratified follicles across every region of the scalp (zero random holes!)."""
        strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        t_idxs = np.searchsorted(tri_cdf, strata)
        out = []
        for ti in t_idxs:
            ti = min(int(ti), len(tris_w) - 1)
            u = float(rng.random())
            v = float(rng.random())
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            w = 1.0 - u - v
            p0, p1, p2 = tris_w[ti]
            n0, n1, n2 = tris_n[ti]
            r0, r1, r2 = tris_r[ti]
            pw = Vector(w * p0 + u * p1 + v * p2)
            nw = Vector(w * n0 + u * n1 + v * n2).normalized()
            rw = Vector(w * r0 + u * r1 + v * r2)
            out.append((pw - nw * 0.0004, nw, rw))
        return out

    def grow_guide_from_follicle(
        root_w: Vector,
        norm_w: Vector,
        root_r: Vector,
        slen: float,
        wave_ph: float,
        wave_amp: float,
        n_pts: int,
        part_jitter: float = 0.0,
    ):
        """Grow a natural skull-hugging, loose S-wave guide curve without crown splits or side puffiness."""
        eff_part_x = part_x + part_jitter
        side_s = -1.0 if root_r.x < eff_part_x else 1.0
        is_right_front = (side_s < 0.0) and (root_r.y < 0.008)
        is_left_side = (side_s > 0.0) and (root_r.y < 0.008)

        # Hug skull tangentially! Near the crown/top (z > 1.60), emerge softly forward/sideways along the skull
        # rather than shooting straight out along scalp normal (prevents wide puffy helmet AND prevents part canyon!)
        dist_part = abs(root_r.x - eff_part_x)
        near_part = math.exp(-((dist_part / 0.022) ** 2))
        # Tangential direction in head space: gentle downward-outward drape right from the root!
        tang_head = Vector((
            side_s * (0.28 + 0.32 * (1.0 - near_part)),
            (0.24 if is_left_side else -0.18) if root_r.y < 0.0 else 0.35,
            0.16 * near_part - 0.32 * (1.0 - near_part),
        )).normalized()
        init_tang = (norm_w * 0.16 + (R_head @ tang_head) * 0.84).normalized()

        pts_w = [root_w]
        cur = root_w.copy()

        for si in range(1, n_pts):
            t = si / (n_pts - 1)
            ds = slen / (n_pts - 1)

            if is_right_front:
                # Hug right temple closely, glide over raised right shoulder, then sweep in loose S-waves
                # across the front-left chest of the pink sweater!
                if t < 0.22:
                    flow = (R_head @ Vector((-0.34, -0.14, -0.92))).normalized()
                elif t < 0.50:
                    flow = Vector((-0.05, -0.46, -0.88)).normalized()
                else:
                    flow = Vector((0.15, -0.16, -0.97)).normalized()
            elif is_left_side:
                # Hug right/left temple closely, tuck behind visible left ear, and flow down behind/along left shoulder!
                if t < 0.26:
                    flow = (R_head @ Vector((0.30, 0.36, -0.88))).normalized()
                elif t < 0.52:
                    flow = Vector((0.10, 0.10, -0.99)).normalized()
                else:
                    flow = Vector((0.03, -0.06, -0.99)).normalized()
            else:
                # Back hair flowing smoothly down back of head and sweater with natural S-waves!
                lat = 0.08 * (root_r.x / 0.08) * (1.0 - t)
                flow = Vector((lat, 0.10 * (1.0 - t) + 0.02, -0.99)).normalized()

            w_grav = _smoothstep_scalar(0.02, 0.14, t)
            step_dir = ((1.0 - w_grav) * init_tang + w_grav * flow).normalized()

            # Prominent, romantic loose S-wave curls along mid-lengths & tips (matching reference.png!)
            w_env = _smoothstep_scalar(0.18, 0.50, t)
            s_off = Vector((
                wave_amp * math.sin(t * 4.8 * math.pi + wave_ph)
                + 0.40 * wave_amp * math.cos(t * 7.5 * math.pi - wave_ph * 0.6),
                wave_amp * 0.65 * math.cos(t * 4.8 * math.pi + wave_ph),
                0.006 * math.sin(t * 9.6 * math.pi + wave_ph),
            )) * w_env

            cur = cur + step_dir * ds + s_off * 0.42
            cur = _push_outside_colliders(cur, bvh_body, bvh_sw, 0.0020, 0.0036)

            cur_rest = M_head_inv @ cur
            # Keep main locks outside full cheek & jawline oval (|x| >= 0.070) so entire symmetrical face is visible!
            if 1.43 < cur_rest.z < 1.590 and cur_rest.y < -0.095 and abs(cur_rest.x) < 0.070:
                cur_rest.x = -0.072 if side_s < 0 else 0.072
                cur = M_head @ cur_rest
                cur = _push_outside_colliders(cur, bvh_body, bvh_sw, 0.0020, 0.0036)

            # Tuck behind visible left ear ONLY at ear pinna height (z < 1.554), keeping temple skin covered!
            if is_left_side and 1.43 < cur_rest.z < 1.554 and cur_rest.x > 0.046:
                if cur_rest.y < -0.024:
                    cur_rest.y = -0.024
                    cur = M_head @ cur_rest
                    cur = _push_outside_colliders(cur, bvh_body, bvh_sw, 0.0020, 0.0036)

            pts_w.append(cur.copy())

        return _smooth_strand(_resample_polyline(pts_w, n_pts), passes=2)

    # ==================================================================
    # 1. HAIR_UNDERLAYER (380 stratified scalp guides x 12 = 4,560 overlapping base strands!)
    # ==================================================================
    under_strands = []
    under_radii = []
    pts_under = 18

    for rw, nw, rr in sample_stratified_follicles(380):
        slen = float(rng.uniform(0.38, 0.49))
        p_jit = float(rng.normal(0.0, 0.009))  # Softly blurs part boundary so zero bald seam ever forms!
        g_arr = grow_guide_from_follicle(
            rw, nw, rr, slen, float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0.014, 0.022)), pts_under, p_jit
        )
        t_col = np.linspace(0.0, 1.0, pts_under, dtype=np.float32)[:, None]

        for ci in range(12):
            j_r = rng.normal(0.0, 0.0042, size=3).astype(np.float32)
            j_t = rng.normal(0.0, 0.0055, size=3).astype(np.float32)
            child = g_arr + j_r[None, :] * (1.0 - 0.3 * t_col) + j_t[None, :] * (t_col ** 1.2)
            for pi in range(1, pts_under):
                vp = _push_outside_colliders(Vector(child[pi]), bvh_body, bvh_sw, 0.0016, 0.0028)
                child[pi] = (vp.x, vp.y, vp.z)
            under_strands.append(_smooth_strand(child, passes=1))
            under_radii.append(_taper_radii(pts_under, rng.uniform(0.00032, 0.00044), 0.000080))

    obj_under = _create_hair_curves_object("Hair_Underlayer", under_strands, under_radii, hair_collection, body_obj)

    # ==================================================================
    # 2. HAIR_MAIN_WAVES (120 stratified guides x 52 = 6,240 glossy S-wave strands!)
    # ==================================================================
    main_strands = []
    main_radii = []
    pts_main = 24

    for rw, nw, rr in sample_stratified_follicles(120):
        slen = float(rng.uniform(0.46, 0.57)) if (rr.x < part_x and rr.y < 0.0) else float(rng.uniform(0.41, 0.52))
        p_jit = float(rng.normal(0.0, 0.006))
        g_arr = grow_guide_from_follicle(
            rw, nw, rr, slen, float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0.022, 0.035)), pts_main, p_jit
        )
        t_col = np.linspace(0.0, 1.0, pts_main, dtype=np.float32)[:, None]

        tangents = np.gradient(g_arr, axis=0)
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-5)
        ref_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]
        ribbon_dir = np.cross(tangents, ref_y)
        ribbon_dir /= np.maximum(np.linalg.norm(ribbon_dir, axis=1, keepdims=True), 1e-5)
        normal_dir = np.cross(ribbon_dir, tangents)

        for ci in range(52):
            w_rib = float(rng.normal(0.0, 0.0042))
            w_nrm = float(rng.normal(0.0, 0.0014))
            tip_wisp = rng.normal(0.0, 0.0032, size=3).astype(np.float32)
            clump_env = 0.88 - 0.20 * np.sin(np.pi * t_col) + 0.55 * (t_col ** 2.1)
            child = (
                g_arr
                + ribbon_dir * (w_rib * clump_env)
                + normal_dir * (w_nrm * clump_env)
                + tip_wisp[None, :] * (t_col ** 2.0)
            )
            for pi in range(1, pts_main):
                vp = _push_outside_colliders(Vector(child[pi]), bvh_body, bvh_sw, 0.0018, 0.0030)
                child[pi] = (vp.x, vp.y, vp.z)
            main_strands.append(_smooth_strand(child, passes=1))
            main_radii.append(_taper_radii(pts_main, rng.uniform(0.00011, 0.00015), 0.000026))

    obj_main = _create_hair_curves_object("Hair_Main_Waves", main_strands, main_radii, hair_collection, body_obj)

    # ==================================================================
    # 3. HAIR_WISPY_BANGS (~540 continuous feathery see-through strands - NO barcode clumps!)
    # ==================================================================
    bang_strands = []
    bang_radii = []
    pts_bang = 16

    for bi in range(540):
        # Continuous feathery distribution across forehead [-0.036 .. +0.034] with subtle natural micro-grouping
        u_b = (bi + rng.uniform(0.0, 1.0)) / 540.0
        bx_base = -0.036 + u_b * 0.070
        # Soft wispy variation (not rigid pencils!)
        bx = bx_base + 0.0018 * math.sin(u_b * 11.0 * math.pi) + float(rng.normal(0.0, 0.0015))

        rx_b = part_x * 0.32 + bx * 0.68
        rz_root = 1.630 - 0.006 * (abs(bx) / 0.036) ** 2 + float(rng.normal(0.0, 0.003))
        q_root = M_head @ Vector((rx_b, -0.130, rz_root))
        loc_r, norm_r, _, _ = bvh_body.find_nearest(q_root, 0.035)
        root_w = (loc_r - norm_r * 0.0003) if loc_r is not None else q_root
        root_rest = M_head_inv @ root_w

        # Gentle rightward sweep across forehead like reference.png!
        tip_sweep_x = bx + 0.008 + 0.005 * u_b + float(rng.normal(0.0, 0.0022))
        tip_rest_z = 1.554 + 0.005 * (abs(bx) / 0.036) + float(rng.uniform(-0.005, 0.005))

        s_pts = [root_w]
        for si in range(1, pts_bang):
            t = si / (pts_bang - 1)
            arch_lift_z = 0.0040 * math.sin(math.pi * (t ** 0.7)) * (1.0 - t)
            arch_fwd_y = -0.0048 * math.sin(math.pi * (t ** 0.8))
            px = root_rest.x * (1.0 - t) + tip_sweep_x * t
            py = (root_rest.y * (1.0 - t) + (-0.1518) * t) + arch_fwd_y
            pz = root_rest.z * (1.0 - t) + tip_rest_z * t + arch_lift_z
            pw = M_head @ Vector((px, py, pz))
            pw = _push_outside_colliders(pw, bvh_body, bvh_sw, 0.0022, 0.0030)
            s_pts.append(pw)

        b_arr = _smooth_strand(_resample_polyline(s_pts, pts_bang), passes=2)
        bang_strands.append(b_arr)
        # Whisper-fine airy radius (45-62 microns at root -> 12 microns at tip!)
        bang_radii.append(_taper_radii(pts_bang, rng.uniform(0.000045, 0.000062), 0.000012))

    obj_bangs = _create_hair_curves_object("Hair_Wispy_Bangs", bang_strands, bang_radii, hair_collection, body_obj)

    # ==================================================================
    # 4. HAIR_FRAMING_TENDRILS (Temple curtain + signature diagonal lock + wavy front locks!)
    # ==================================================================
    tendril_strands = []
    tendril_radii = []
    pts_tend = 22

    # First 2 paths in head rest space (right-temple curtain locks tucking at upper left ear);
    # next 5 paths transition smoothly into posed world space across the front-left chest of her pink sweater!
    temple_paths_rest = [
        [(-0.004, -0.126, 1.632), (0.024, -0.148, 1.604), (0.050, -0.132, 1.576), (0.070, -0.088, 1.558), (0.081, -0.036, 1.545), (0.086, -0.008, 1.490)],
        [(0.010, -0.124, 1.628), (0.038, -0.138, 1.600), (0.060, -0.118, 1.574), (0.075, -0.075, 1.558), (0.084, -0.028, 1.540), (0.088, -0.002, 1.472)],
    ]
    chest_wave_paths = [
        # Root & cheek in head space (first 3 pts), then world-space S-wave cascade over collar/chest of pink sweater!
        ([(-0.020, -0.126, 1.628), (-0.058, -0.138, 1.582), (-0.080, -0.112, 1.518)], [(-0.125, -0.125, 1.385), (-0.075, -0.185, 1.295), (-0.112, -0.205, 1.205), (-0.058, -0.198, 1.115), (-0.085, -0.188, 1.035)]),
        ([(-0.032, -0.118, 1.616), (-0.068, -0.128, 1.568), (-0.086, -0.102, 1.502)], [(-0.138, -0.122, 1.375), (-0.092, -0.188, 1.280), (-0.128, -0.205, 1.188), (-0.074, -0.198, 1.098), (-0.098, -0.188, 1.025)]),
        ([(-0.042, -0.105, 1.604), (-0.076, -0.112, 1.552), (-0.092, -0.090, 1.488)], [(-0.150, -0.115, 1.368), (-0.108, -0.182, 1.268), (-0.142, -0.200, 1.172), (-0.092, -0.194, 1.082), (-0.112, -0.185, 1.020)]),
        ([(-0.024, -0.102, 1.622), (-0.064, -0.118, 1.572), (-0.084, -0.095, 1.508)], [(-0.115, -0.130, 1.380), (-0.058, -0.182, 1.290), (-0.094, -0.202, 1.195), (-0.044, -0.195, 1.105), (-0.068, -0.185, 1.035)]),
    ]

    all_tendril_world_guides = []
    for c_pts_rest in temple_paths_rest:
        ctrl = np.array(c_pts_rest, dtype=np.float64)
        t_in = np.linspace(0.0, 1.0, len(ctrl))
        t_out = np.linspace(0.0, 1.0, pts_tend)
        interp = np.stack([np.interp(t_out, t_in, ctrl[:, k]) for k in range(3)], axis=-1)
        all_tendril_world_guides.append([M_head @ Vector(interp[si]) for si in range(pts_tend)])

    for head_pts, world_pts in chest_wave_paths:
        w_chain = [M_head @ Vector(p) for p in head_pts] + [Vector(p) for p in world_pts]
        ctrl = np.array([[p.x, p.y, p.z] for p in w_chain], dtype=np.float64)
        t_in = np.linspace(0.0, 1.0, len(ctrl))
        t_out = np.linspace(0.0, 1.0, pts_tend)
        interp = np.stack([np.interp(t_out, t_in, ctrl[:, k]) for k in range(3)], axis=-1)
        all_tendril_world_guides.append([Vector(interp[si]) for si in range(pts_tend)])

    for raw_w in all_tendril_world_guides:
        g_world = []
        for si, pw0 in enumerate(raw_w):
            pw = pw0.copy()
            if si == 0:
                loc, norm, _, _ = bvh_body.find_nearest(pw, 0.03)
                if loc is not None:
                    pw = loc - norm * 0.0003
            else:
                pw = _push_outside_colliders(pw, bvh_body, bvh_sw, 0.0022, 0.0040)
            g_world.append(pw)
        gw_arr = _smooth_strand(_resample_polyline(g_world, pts_tend), passes=3)
        t_col = np.linspace(0.0, 1.0, pts_tend, dtype=np.float32)[:, None]

        for ci in range(135):
            j_r = rng.normal(0.0, 0.0028, size=3).astype(np.float32)
            j_t = rng.normal(0.0, 0.0045, size=3).astype(np.float32)
            child = gw_arr + j_r[None, :] * (1.0 - 0.35 * t_col) + j_t[None, :] * (t_col ** 1.2)
            for pi in range(1, pts_tend):
                vp = _push_outside_colliders(Vector(child[pi]), bvh_body, bvh_sw, 0.0018, 0.0032)
                child[pi] = (vp.x, vp.y, vp.z)
            tendril_strands.append(_smooth_strand(child, passes=1))
            tendril_radii.append(_taper_radii(pts_tend, rng.uniform(0.000090, 0.000130), 0.000022))

    obj_tendrils = _create_hair_curves_object(
        "Hair_Framing_Tendrils", tendril_strands, tendril_radii, hair_collection, body_obj
    )

    # ==================================================================
    # 5. HAIR_FLYAWAYS (~650 delicate sunlit halo strands)
    # ==================================================================
    fly_strands = []
    fly_radii = []
    skull_c_world = M_head @ Vector((0.0, -0.055, 1.565))
    for fi in range(650):
        base_s = main_strands[int(rng.integers(0, len(main_strands)))].copy()
        n_p = len(base_s)
        t_col = np.linspace(0.0, 1.0, n_p, dtype=np.float32)
        ph = float(rng.uniform(0.0, 2.0 * math.pi))
        amp = float(rng.uniform(0.003, 0.010))
        outward = base_s - np.array([skull_c_world.x, skull_c_world.y, base_s[0, 2]], dtype=np.float32)[None, :]
        out_norm = outward / np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-4)
        wiggle = (
            out_norm * (amp * np.sin(t_col * math.pi * rng.uniform(1.1, 2.4)) ** 2)[:, None]
            + np.stack([
                0.0035 * np.sin(t_col * 8.0 + ph),
                0.0035 * np.cos(t_col * 7.0 - ph),
                0.0025 * np.sin(t_col * 6.0 + ph * 0.5),
            ], axis=-1) * (t_col[:, None] ** 0.9)
        )
        fly = base_s + wiggle.astype(np.float32)
        cut_n = int(rng.integers(12, n_p + 1))
        fly = fly[:cut_n]
        for pi in range(1, cut_n):
            vp = _push_outside_colliders(Vector(fly[pi]), bvh_body, bvh_sw, 0.0022, 0.0032)
            fly[pi] = (vp.x, vp.y, vp.z)
        fly_strands.append(_smooth_strand(fly, passes=1))
        fly_radii.append(_taper_radii(cut_n, rng.uniform(0.000050, 0.000075), 0.000014))

    obj_fly = _create_hair_curves_object("Hair_Flyaways", fly_strands, fly_radii, hair_collection, body_obj)

    # ==================================================================
    # 6. EYEBROWS (~720 soft, natural straight-gently-arched East Asian brows!)
    # ==================================================================
    brow_strands = []
    brow_radii = []
    pts_brow = 7

    for side_s in (-1.0, 1.0):
        for bi in range(460):
            u = float(rng.uniform(0.0, 1.0) ** 0.92)
            bx_abs = 0.0150 + u * 0.0375
            arch_z = 1.5602 + 0.0011 * math.sin(math.pi * u) - 0.0018 * (u ** 2.2)
            thick_z = 0.0034 * (1.0 - 0.45 * (u ** 2.4)) * (0.82 + 0.18 * math.sin(math.pi * min(u * 3.0, 1.0)))
            v_off = rng.triangular(-1.0, 0.0, 1.0) * thick_z

            pw_query = M_head @ Vector((side_s * bx_abs, -0.152, arch_z + v_off))
            loc, norm, _, _ = bvh_body.find_nearest(pw_query, 0.03)
            if loc is None:
                continue
            root_b = loc - norm * 0.00015

            ang_deg = 28.0 * (1.0 - u) ** 2.2 + 4.0 * math.sin(math.pi * u) - 12.0 * (u ** 1.8)
            ang_deg -= (v_off / max(thick_z, 1e-5)) * 12.0
            ang_rad = math.radians(ang_deg + rng.normal(0.0, 3.0))

            dir_rest = Vector((side_s * math.cos(ang_rad), -0.18, math.sin(ang_rad))).normalized()
            dir_world = (R_head @ dir_rest).normalized()
            tang_w = (dir_world - norm * dir_world.dot(norm)).normalized()

            hlen = rng.uniform(0.0052, 0.0078) * (0.78 + 0.30 * math.sin(math.pi * u))
            b_pts = []
            for si in range(pts_brow):
                tb = si / (pts_brow - 1)
                lift = 0.00038 * math.sin(math.pi * (tb ** 0.8))
                pt_b = root_b + tang_w * (hlen * tb) + norm * lift
                b_pts.append([pt_b.x, pt_b.y, pt_b.z])

            brow_strands.append(np.array(b_pts, dtype=np.float32))
            brow_radii.append(_taper_radii(pts_brow, rng.uniform(0.000040, 0.000054), 0.000010))

    obj_brows = _create_hair_curves_object("Character_Eyebrows", brow_strands, brow_radii, hair_collection, body_obj)

    # ==================================================================
    # 7. EYELASHES (~400 curved delicate 3D upper & lower eyelashes)
    # ==================================================================
    lash_strands = []
    lash_radii = []
    pts_lash = 9

    rim_xs = np.array([0.0195, 0.0250, 0.0310, 0.0370, 0.0435])
    rim_up_zs = np.array([1.5459, 1.5492, 1.5517, 1.5514, 1.5486])
    rim_lo_zs = np.array([1.5426, 1.5418, 1.5414, 1.5421, 1.5442])

    for side_s in (-1.0, 1.0):
        n_upper = 145
        for li in range(n_upper):
            u = (li + rng.uniform(0.15, 0.85)) / n_upper
            lx_abs = 0.0195 + u * 0.0240
            arch_sin = math.sin(math.pi * (u ** 0.90))
            lz = float(np.interp(lx_abs, rim_xs, rim_up_zs)) + rng.normal(0.0, 0.00008)
            q_world = M_head @ Vector((side_s * lx_abs, -0.1435, lz))
            loc, _, _, _ = bvh_body.find_nearest(q_world, 0.010)
            root_l = loc if loc is not None else q_world

            clump_u = round(u * 15.0) / 15.0
            flare_x = side_s * (0.26 * (u - 0.32) + 0.12 * (clump_u - u) * 8.0)
            lash_len = (0.0038 + 0.0046 * (arch_sin ** 0.75) + 0.0010 * u) * rng.uniform(0.90, 1.10)

            l_pts = []
            for si in range(pts_lash):
                tl = si / (pts_lash - 1)
                arc_rest = Vector((
                    flare_x * lash_len * (tl ** 1.15),
                    -lash_len * (0.88 * tl),
                    lash_len * (-0.10 * tl + 0.58 * (tl ** 2.1)),
                ))
                pt_w = root_l + (R_head @ arc_rest)
                l_pts.append([pt_w.x, pt_w.y, pt_w.z])

            lash_strands.append(np.array(l_pts, dtype=np.float32))
            lash_radii.append(_taper_radii(pts_lash, rng.uniform(0.000050, 0.000066), 0.000010, root_taper=0.05))

        n_lower = 64
        for li in range(n_lower):
            u = (li + rng.uniform(0.1, 0.9)) / n_lower
            lx_abs = 0.0200 + u * 0.0230
            arch_low = math.sin(math.pi * (u ** 0.88))
            lz = float(np.interp(lx_abs, rim_xs, rim_lo_zs)) + rng.normal(0.0, 0.00008)
            q_world = M_head @ Vector((side_s * lx_abs, -0.1425, lz))
            loc, _, _, _ = bvh_body.find_nearest(q_world, 0.012)
            root_l = loc if loc is not None else q_world

            lash_len = (0.0020 + 0.0018 * arch_low) * rng.uniform(0.85, 1.15)
            flare_x = side_s * 0.24 * (u - 0.35)
            l_pts = []
            for si in range(pts_lash):
                tl = si / (pts_lash - 1)
                arc_rest = Vector((
                    flare_x * lash_len * tl,
                    -lash_len * 0.78 * tl,
                    -lash_len * (0.35 * tl + 0.45 * (tl ** 1.9)),
                ))
                pt_w = root_l + (R_head @ arc_rest)
                l_pts.append([pt_w.x, pt_w.y, pt_w.z])

            lash_strands.append(np.array(l_pts, dtype=np.float32))
            lash_radii.append(_taper_radii(pts_lash, rng.uniform(0.000032, 0.000044), 0.000008, root_taper=0.05))

    obj_lashes = _create_hair_curves_object("Character_Eyelashes", lash_strands, lash_radii, hair_collection, body_obj)

    return {
        "underlayer": obj_under,
        "main_waves": obj_main,
        "bangs": obj_bangs,
        "tendrils": obj_tendrils,
        "flyaways": obj_fly,
        "eyebrows": obj_brows,
        "eyelashes": obj_lashes,
    }
