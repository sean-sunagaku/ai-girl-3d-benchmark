"""Multi-layered procedural chestnut hair (volume cap, clumps, waves, sheer bangs, flyaways), 3D brows, and lashes."""

import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree


def create_hair_material(
    name: str,
    melanin: float = 0.91,
    redness: float = 0.38,
    roughness: float = 0.30,
    radial_roughness: float = 0.32,
    random_color: float = 0.14,
    coat: float = 0.08,
) -> bpy.types.Material:
    """Create rich dark-chestnut Principled Hair BSDF shader with warm sunlit sheen."""
    mat = bpy.data.materials.new(name)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfHairPrincipled")
    bsdf.parametrization = "MELANIN"
    bsdf.inputs["Melanin Redness"].default_value = redness
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Radial Roughness"].default_value = radial_roughness
    bsdf.inputs["Coat"].default_value = coat
    bsdf.inputs["IOR"].default_value = 1.55
    bsdf.inputs["Offset"].default_value = math.radians(2.0)
    bsdf.inputs["Random Color"].default_value = random_color
    bsdf.inputs["Random Roughness"].default_value = 0.15

    h_info = nodes.new("ShaderNodeHairInfo")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    m0 = min(0.98, melanin + 0.04)
    ramp.color_ramp.elements[0].color = (m0, m0, m0, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    m1 = max(0.76, melanin - 0.06)
    ramp.color_ramp.elements[1].color = (m1, m1, m1, 1.0)

    links.new(h_info.outputs["Intercept"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Melanin"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _get_head_pose_matrix(arm_obj: bpy.types.Object) -> Matrix:
    """Rigid transform from Head Rest World Space to Head Posed World Space."""
    pb = arm_obj.pose.bones["head"]
    return (arm_obj.matrix_world @ pb.matrix) @ pb.bone.matrix_local.inverted()


def _create_hair_curves_obj(
    name: str,
    strands_pts: np.ndarray,
    strands_rad: np.ndarray,
    mat: bpy.types.Material,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create a Blender 5.1 Hair Curves object from (N, K, 3) points & (N, K) radii."""
    n_curves, n_pts, _ = strands_pts.shape
    hc = bpy.data.hair_curves.new(f"{name}_Data")
    hc.add_curves([n_pts] * n_curves)
    hc.points.foreach_set("position", np.ascontiguousarray(strands_pts, dtype=np.float32).reshape(-1))
    rad_attr = hc.attributes.new(name="radius", type="FLOAT", domain="POINT")
    rad_attr.data.foreach_set("value", np.ascontiguousarray(strands_rad, dtype=np.float32).reshape(-1))
    hc.materials.append(mat)
    obj = bpy.data.objects.new(name, hc)
    collection.objects.link(obj)
    return obj


def _build_scalp_volume_cap(
    body_obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    m_head: Matrix,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Create dark-chestnut scalp volume cap covering hair-bearing skull cleanly behind hairline."""
    prev_pos = arm_obj.data.pose_position
    arm_obj.data.pose_position = "REST"
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    body_rest = body_obj.evaluated_get(depsgraph)
    bvh_rest = BVHTree.FromObject(body_rest, depsgraph)

    arm_obj.data.pose_position = prev_pos
    bpy.context.view_layer.update()

    head_c = Vector((0.0, -0.020, 1.560))
    bm = bmesh.new()
    nu, nv = 40, 18
    grid = [[None] * nu for _ in range(nv)]

    for i in range(nv):
        tv = i / (nv - 1)
        for j in range(nu):
            theta = 2.0 * math.pi * (j / nu)  # 0=+X, pi/2=+Y(back), pi=-X, 3pi/2=-Y(front)
            front_w = max(0.0, -math.sin(theta))
            back_w = max(0.0, math.sin(theta))

            # Front edge starts at +33 deg (natural hairline), side temples +2 deg (above ears), nape -48 deg
            elev_min = math.radians(33.0 * front_w - 48.0 * back_w + 2.0 * (1.0 - front_w - back_w))
            elev = elev_min + tv * (math.radians(88.0) - elev_min)

            dir_v = Vector((
                math.cos(elev) * math.cos(theta),
                math.cos(elev) * math.sin(theta),
                math.sin(elev),
            )).normalized()

            hit, _, _, _ = bvh_rest.ray_cast(head_c + dir_v * 0.20, -dir_v, 0.25)
            p_skin = hit if hit is not None else (head_c + dir_v * 0.085)

            # Keep cap recessed 1.0-1.8mm above skull (UNDER hair strand roots so cap NEVER pokes above hair!)
            offset = 0.0016 * math.sin(max(0.05, tv) * math.pi * 0.85) + 0.0004
            if front_w > 0.15 and tv < 0.25:
                offset -= 0.0022 * front_w * (1.0 - tv / 0.25)

            p_world = m_head @ (p_skin + dir_v * offset)
            grid[i][j] = bm.verts.new(p_world)

    for i in range(nv - 1):
        for j in range(nu):
            jn = (j + 1) % nu
            bm.faces.new((grid[i][j], grid[i][jn], grid[i + 1][jn], grid[i + 1][j]))

    top_p = m_head @ Vector((0.0, -0.015, 1.668))
    top_v = bm.verts.new(top_p)
    for j in range(nu):
        bm.faces.new((grid[nv - 1][j], grid[nv - 1][(j + 1) % nu], top_v))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    mesh = bpy.data.meshes.new("Hair_VolumeCap_Mesh")
    bm.to_mesh(mesh)
    bm.free()

    cap_obj = bpy.data.objects.new("Hair_VolumeCap", mesh)
    for p in cap_obj.data.polygons:
        p.use_smooth = True
    sub = cap_obj.modifiers.new("Subdivision", type="SUBSURF")
    sub.levels = 1
    sub.render_levels = 2

    mat_cap = bpy.data.materials.new("Hair_ScalpCap_Mat")
    nodes = mat_cap.node_tree.nodes
    links = mat_cap.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.020, 0.010, 0.007, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Specular IOR Level"].default_value = 0.06
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    cap_obj.data.materials.append(mat_cap)

    collection.objects.link(cap_obj)
    return cap_obj


def _flow_strand(
    root_w: Vector,
    init_vel_w: Vector,
    target_flow_fn,
    n_pts: int,
    seg_len: float,
    wave_freq: float,
    wave_amp: float,
    wave_phase: float,
    wave_axis_w: Vector,
    bvh_body: BVHTree,
    bvh_sw: BVHTree,
    face_center_w: Vector,
    min_clear_body: float = 0.0028,
    min_clear_sw: float = 0.0040,
    avoid_face: bool = True,
    shelf_bias_y: float = 0.25,
) -> np.ndarray:
    """March a single hair strand smoothly along scalp, gravity, waves, and BVH surfaces."""
    pts = np.zeros((n_pts, 3), dtype=np.float32)
    p = root_w.copy()
    pts[0] = p[:]
    vel = init_vel_w.normalized()

    for k in range(1, n_pts):
        t = k / (n_pts - 1)
        flow_dir = target_flow_fn(t, p).normalized()
        bend_rate = 0.45 if t < 0.16 else 0.54
        vel = vel.lerp(flow_dir, bend_rate).normalized()

        wave_env = math.sin(min(1.0, t / 0.35) * math.pi * 0.5) * (0.25 + 0.75 * t)
        wave_vel = wave_axis_w * (math.cos(t * wave_freq + wave_phase) * wave_amp * wave_env)
        p = p + (vel * seg_len) + wave_vel

        if avoid_face:
            rel_f = p - face_center_w
            rx, ry, rz = 0.068, 0.068, 0.072
            d_ell = math.sqrt((rel_f.x / rx) ** 2 + (rel_f.y / ry) ** 2 + (rel_f.z / rz) ** 2)
            if d_ell < 1.0 and rel_f.y < 0.025:
                push_dir = Vector((rel_f.x * 1.4, min(0.0, rel_f.y) * 0.2, rel_f.z * 0.15))
                if push_dir.length < 1e-5:
                    push_dir = Vector((1.0, 0.0, 0.0))
                push_dir.normalize()
                p = face_center_w + Vector((
                    rel_f.x / max(0.15, d_ell),
                    rel_f.y / max(0.35, d_ell),
                    rel_f.z / max(0.35, d_ell),
                ))
                vel = (vel - push_dir * min(0.0, vel.dot(push_dir))).normalized()

        # Slide smoothly over Body (head, ears, neck, face)
        loc_b, nrm_b, _, _ = bvh_body.find_nearest(p, 0.015)
        if loc_b is not None:
            sd_b = (p - loc_b).dot(nrm_b)
            if sd_b < min_clear_body:
                p = loc_b + nrm_b * min_clear_body
                if vel.dot(nrm_b) < 0:
                    tang_v = vel - nrm_b * vel.dot(nrm_b)
                    if tang_v.length < 0.25:
                        tang_v = Vector((0.0, shelf_bias_y, -0.8))
                    vel = tang_v.normalized()

        # Slide smoothly over Sweater (NEVER stall on horizontal collar/shoulder top!)
        loc_s, nrm_s, _, _ = bvh_sw.find_nearest(p, 0.020)
        if loc_s is not None:
            sd_s = (p - loc_s).dot(nrm_s)
            if sd_s < min_clear_sw:
                p = loc_s + nrm_s * min_clear_sw
                if vel.dot(nrm_s) < 0:
                    tang_v = vel - nrm_s * vel.dot(nrm_s)
                    # If hitting top shoulder/collar shelf, slide smoothly backward (+Y) or forward (-Y)!
                    if nrm_s.z > 0.45 or tang_v.length < 0.35:
                        tang_v = Vector((flow_dir.x * 0.35, shelf_bias_y, -0.75))
                        tang_v = tang_v - nrm_s * min(0.0, tang_v.dot(nrm_s))
                    if tang_v.length > 1e-4:
                        vel = tang_v.normalized()

        pts[k] = p[:]

    return pts


def build_hair_and_facial_details(
    body_obj: bpy.types.Object,
    sweater_obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    eye_centers: dict,
    collection: bpy.types.Collection,
    seed: int = 2026,
) -> list:
    """Build complete multi-layer dark-chestnut wavy hair, sheer bangs, flyaways, 3D brows, and lashes."""
    rng = np.random.default_rng(seed)
    m_head = _get_head_pose_matrix(arm_obj)
    m_head_rot = m_head.to_3x3()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    body_eval = body_obj.evaluated_get(depsgraph)
    sw_eval = sweater_obj.evaluated_get(depsgraph)
    bvh_body = BVHTree.FromObject(body_eval, depsgraph)
    bvh_sw = BVHTree.FromObject(sw_eval, depsgraph)

    prev_pos = arm_obj.data.pose_position
    arm_obj.data.pose_position = "REST"
    bpy.context.view_layer.update()
    body_rest = body_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    bvh_rest = BVHTree.FromObject(body_rest, bpy.context.evaluated_depsgraph_get())
    rest_co = np.array([v.co[:] for v in body_obj.data.vertices], dtype=np.float64)
    arm_obj.data.pose_position = prev_pos
    bpy.context.view_layer.update()

    face_center_w = m_head @ Vector((0.0, -0.092, 1.515))
    head_center_rest = Vector((0.0, -0.020, 1.560))

    mat_hair_main = create_hair_material("Hair_Chestnut_Main", melanin=0.91, redness=0.38, roughness=0.29, coat=0.08)
    mat_hair_bangs = create_hair_material("Hair_Chestnut_Bangs", melanin=0.93, redness=0.36, roughness=0.30, coat=0.07)
    mat_brow = create_hair_material("Eyebrow_Chestnut", melanin=0.84, redness=0.35, roughness=0.40, coat=0.02)
    mat_lash = create_hair_material("Eyelash_DarkBrown", melanin=0.96, redness=0.22, roughness=0.30, coat=0.06)

    created = []
    created.append(_build_scalp_volume_cap(body_obj, arm_obj, m_head, collection))

    def probe_scalp(theta, elev):
        d = Vector((math.cos(elev) * math.cos(theta), math.cos(elev) * math.sin(theta), math.sin(elev))).normalized()
        hit, nrm, _, _ = bvh_rest.ray_cast(head_center_rest + d * 0.20, -d, 0.25)
        if hit is None:
            return head_center_rest + d * 0.085, d
        return hit, nrm.normalized()

    def sample_elev(e_min_deg, e_max_deg):
        s0 = math.sin(math.radians(e_min_deg))
        s1 = math.sin(math.radians(e_max_deg))
        return math.asin(rng.uniform(s0, s1))

    # =========================================================================
    # 1. MAIN LONG WAVY HAIR (Full crown + Back + Left Temple/Behind-Ear + Right Chest Cascade)
    # =========================================================================
    n_clumps = 230
    strands_per_clump = 52
    n_pts_main = 24
    n_main = n_clumps * strands_per_clump  # 11,960 curves

    main_pts = np.zeros((n_main, n_pts_main, 3), dtype=np.float32)
    main_rad = np.zeros((n_main, n_pts_main), dtype=np.float32)

    part_x = 0.008
    idx_out = 0

    for c_i in range(n_clumps):
        u_zone = rng.random()
        if u_zone < 0.15:
            # Top-front crown: sweeps cleanly sideways & backward into side cascades
            theta = rng.uniform(math.radians(205.0), math.radians(335.0))
            elev = sample_elev(44.0, 88.0)
            zone = "top_crown"
        elif u_zone < 0.40:
            # Right front/temple cascade -> sweeps smoothly over right shoulder (-X) & diagonally across pink sweater front!
            theta = rng.uniform(math.radians(172.0), math.radians(248.0))
            elev = sample_elev(12.0, 84.0)
            zone = "right_front_cascade"
        elif u_zone < 0.54:
            # Left temple sweep (+X, front-left temple!): drapes gracefully across temple/outer brow then tucks behind left ear!
            theta = rng.uniform(math.radians(-68.0), math.radians(-22.0))
            elev = sample_elev(20.0, 80.0)
            zone = "left_temple"
        elif u_zone < 0.65:
            # Left side behind ear (+X)
            theta = rng.uniform(math.radians(-28.0), math.radians(45.0))
            elev = sample_elev(4.0, 84.0)
            zone = "left_side"
        elif u_zone < 0.74:
            # Right side behind shoulder
            theta = rng.uniform(math.radians(128.0), math.radians(176.0))
            elev = sample_elev(-12.0, 82.0)
            zone = "right_side"
        else:
            # Full lush back & crown-back hair draping smoothly down upper back of sweater!
            theta = rng.uniform(math.radians(15.0), math.radians(165.0))
            elev = sample_elev(-44.0, 88.0)
            zone = "back"

        p_root_r, nrm_r = probe_scalp(theta, elev)
        side_sign = 1.0 if p_root_r.x >= part_x else -1.0

        if zone == "top_crown":
            init_r = Vector((side_sign * 0.62, 0.32 if side_sign > 0 else -0.05, 0.15)).normalized()
            init_r = (nrm_r * 0.35 + init_r * 0.65).normalized()
        elif zone == "left_temple":
            init_r = Vector((0.62, 0.36, -0.32)).normalized()
            init_r = (nrm_r * 0.32 + init_r * 0.68).normalized()
        elif zone == "left_side":
            init_r = Vector((0.55, 0.45, -0.35)).normalized()
            init_r = (nrm_r * 0.32 + init_r * 0.68).normalized()
        elif zone == "right_front_cascade":
            init_r = Vector((-0.38, -0.16, -0.58)).normalized()
            init_r = (nrm_r * 0.28 + init_r * 0.72).normalized()
        else:
            init_r = (nrm_r * 0.30 + Vector((0.08 * math.cos(theta), 0.38, -0.88))).normalized()

        clump_phase = rng.uniform(0.0, 2.0 * math.pi)
        clump_wave_freq = rng.uniform(7.5, 11.5) if zone == "right_front_cascade" else rng.uniform(8.5, 13.5)
        clump_wave_amp = rng.uniform(0.0075, 0.0135) if zone == "right_front_cascade" else rng.uniform(0.0045, 0.0095)

        total_len = rng.uniform(0.42, 0.52)
        if elev < math.radians(0.0):
            total_len *= 0.85
        if zone == "right_front_cascade":
            total_len = rng.uniform(0.47, 0.57)
        seg_len = total_len / (n_pts_main - 1)

        shelf_bias = -0.38 if zone in ("right_front_cascade", "top_crown") and side_sign < 0 else 0.52
        if zone in ("left_side", "left_temple"):
            shelf_bias = 0.74

        def make_flow_fn(z_tag, s_sign, c_theta):
            def flow_fn(t, cur_p):
                if z_tag == "top_crown":
                    if s_sign > 0:
                        return Vector((0.40 if t < 0.35 else 0.14, 0.52 if t < 0.50 else 0.22, -0.76))
                    else:
                        if t < 0.38:
                            return Vector((-0.24, -0.20, -0.95))
                        else:
                            return Vector((0.08, -0.26, -0.96))
                elif z_tag == "left_temple":
                    # Sweep gracefully over left temple (+X) and cleanly tuck behind left ear & shoulder!
                    if t < 0.28:
                        return Vector((0.44, 0.42, -0.78))
                    elif t < 0.60:
                        return Vector((0.20, 0.56, -0.79))
                    else:
                        return Vector((0.08, 0.24, -0.96))
                elif z_tag == "right_front_cascade":
                    if t < 0.34:
                        return Vector((-0.06, -0.22, -0.97))
                    elif t < 0.72:
                        return Vector((0.22 if cur_p.x < -0.04 else -0.02, -0.28, -0.94))
                    else:
                        return Vector((0.14 if cur_p.x < -0.03 else -0.01, -0.10, -0.98))
                elif z_tag == "left_side":
                    if t < 0.42:
                        return Vector((0.25, 0.56, -0.79))
                    else:
                        return Vector((0.10, 0.28, -0.95))
                elif z_tag == "right_side":
                    if t < 0.28:
                        return Vector((-0.14, 0.10, -0.98))
                    else:
                        return Vector((0.05, -0.05, -0.99))
                else:
                    return Vector((0.05 * math.cos(c_theta), 0.36 if t < 0.42 else 0.12, -0.92))
            return flow_fn

        flow_fn = make_flow_fn(zone, side_sign, theta)
        wave_axis_w = Vector((math.sin(theta), math.cos(theta), 0.0)).normalized()

        for s_i in range(strands_per_clump):
            jit_theta = theta + rng.normal(0.0, 0.068)
            jit_elev = elev + rng.normal(0.0, 0.062)
            pr_r, nr_r = probe_scalp(jit_theta, jit_elev)
            root_w = m_head @ (pr_r + nr_r * 0.0022)
            init_w = (m_head_rot @ init_r).normalized()

            strand_amp = clump_wave_amp * rng.uniform(0.70, 1.25)
            strand_phase = clump_phase + rng.normal(0.0, 0.35)
            strand_len = seg_len * rng.uniform(0.90, 1.08)

            pts = _flow_strand(
                root_w=root_w,
                init_vel_w=init_w,
                target_flow_fn=flow_fn,
                n_pts=n_pts_main,
                seg_len=strand_len,
                wave_freq=clump_wave_freq,
                wave_amp=strand_amp,
                wave_phase=strand_phase,
                wave_axis_w=wave_axis_w,
                bvh_body=bvh_body,
                bvh_sw=bvh_sw,
                face_center_w=face_center_w,
                min_clear_body=0.0028,
                min_clear_sw=0.0040,
                avoid_face=True,
                shelf_bias_y=shelf_bias,
            )
            main_pts[idx_out] = pts
            r0 = rng.uniform(0.00018, 0.00026)
            t_arr = np.linspace(0.0, 1.0, n_pts_main)
            main_rad[idx_out] = r0 * (1.0 - 0.65 * (t_arr ** 1.5))
            idx_out += 1

    created.append(_create_hair_curves_obj("Hair_Main", main_pts, main_rad, mat_hair_main, collection))

    # =========================================================================
    # 2. CROWN-TO-FOREHEAD SEE-THROUGH WISPY BANGS
    # =========================================================================
    # (A) Dense upper-forehead/crown fringe (680 strands from crown part down to z~1.605) -> natural hairline!
    # (B) Delicate wispy see-through bang tips (360 strands in 9 locks ending gracefully at brows/upper-lids!)
    n_crown_fringe = 820
    n_bang_locks = 11
    strands_per_bang = 38
    n_pts_bang = 14
    n_bangs = n_crown_fringe + n_bang_locks * strands_per_bang  # 1,238 strands total
    bang_pts = np.zeros((n_bangs, n_pts_bang, 3), dtype=np.float32)
    bang_rad = np.zeros((n_bangs, n_pts_bang), dtype=np.float32)

    idx_b = 0
    # (A) Dense top-front crown layers sweeping smoothly across upper forehead & temples (no center bare V-part!)
    for k_f in range(n_crown_fringe):
        lx = rng.uniform(-0.070, 0.075)
        elev_b = math.radians(rng.uniform(36.0, 74.0))
        theta_b = math.radians(-90.0) + math.atan2(lx, 0.082)
        pr_r, _ = probe_scalp(theta_b, elev_b)

        target_end_z = rng.uniform(1.588, 1.612) - 0.015 * (abs(lx) / 0.072) ** 2
        drop_len = max(0.032, pr_r.z - target_end_z)
        swoop_x = 0.04 + 0.14 * (lx / 0.072) + rng.uniform(-0.04, 0.04)
        arch_out = rng.uniform(0.0020, 0.0036)
        for k in range(n_pts_bang):
            t = k / (n_pts_bang - 1)
            cur_x = pr_r.x + swoop_x * drop_len * (t ** 1.25)
            cur_z = pr_r.z + 0.0015 * math.sin(t * math.pi * 0.55) - drop_len * (t ** 1.04)
            hit_f, _, _, _ = bvh_rest.ray_cast(Vector((cur_x, -0.28, cur_z)), Vector((0.0, 1.0, 0.0)), 0.40)
            if hit_f is not None:
                clearance = arch_out * math.sin(t * math.pi) + 0.0018
                w_surf = min(1.0, t / 0.28)
                target_y = min(pr_r.y, hit_f.y - clearance)
                cur_y = pr_r.y * (1.0 - w_surf) + target_y * w_surf if k > 0 else pr_r.y
            else:
                cur_y = pr_r.y - 0.0015 * t
            bang_pts[idx_b, k] = (m_head @ Vector((cur_x, cur_y, cur_z)))[:]
        r0 = rng.uniform(0.00018, 0.00025)
        bang_rad[idx_b] = r0 * (1.0 - 0.65 * np.linspace(0, 1, n_pts_bang) ** 1.3)
        idx_b += 1

    # (B) Wispy see-through bangs arching gracefully across brows & temples!
    lock_x_centers = np.linspace(-0.056, 0.062, n_bang_locks) + rng.uniform(-0.002, 0.002, n_bang_locks)
    for l_i, lx in enumerate(lock_x_centers):
        side_t = abs(lx) / 0.050
        # Tips end right at brows/upper-lid crease (z~1.552..1.564 center, ~1.530 sides!)
        target_tip_z = 1.556 - 0.024 * (side_t ** 1.4) + rng.uniform(-0.003, 0.004)
        swoop_x = 0.20 + 0.16 * (lx / 0.05)
        lock_phase = rng.uniform(0.0, math.pi)

        for s_i in range(strands_per_bang):
            sx_r = lx + rng.normal(0.0, 0.0020)
            elev_b = math.radians(36.0 + rng.uniform(-2.0, 16.0))
            theta_b = math.radians(-90.0) + math.atan2(sx_r, 0.085)
            pr_r, _ = probe_scalp(theta_b, elev_b)

            drop_len = max(0.045, pr_r.z - target_tip_z) * rng.uniform(0.92, 1.06)
            arch_out = rng.uniform(0.0028, 0.0048)
            for k in range(n_pts_bang):
                t = k / (n_pts_bang - 1)
                cur_x = pr_r.x + swoop_x * drop_len * (t ** 1.35) + 0.0012 * math.sin(t * 3.8 + lock_phase)
                cur_z = pr_r.z + 0.0015 * math.sin(t * math.pi * 0.52) - drop_len * (t ** 1.05)
                hit_f, _, _, _ = bvh_rest.ray_cast(Vector((cur_x, -0.28, cur_z)), Vector((0.0, 1.0, 0.0)), 0.40)
                if hit_f is not None:
                    clearance = arch_out * math.sin(t * math.pi) + 0.0020
                    w_surf = min(1.0, t / 0.25)
                    target_y = min(pr_r.y, hit_f.y - clearance)
                    cur_y = pr_r.y * (1.0 - w_surf) + target_y * w_surf if k > 0 else pr_r.y
                else:
                    cur_y = pr_r.y - 0.0018 * t
                bang_pts[idx_b, k] = (m_head @ Vector((cur_x, cur_y, cur_z)))[:]

            r0 = rng.uniform(0.00013, 0.00018)
            bang_rad[idx_b] = r0 * (1.0 - 0.78 * np.linspace(0, 1, n_pts_bang) ** 1.3)
            idx_b += 1

    created.append(_create_hair_curves_obj("Hair_Bangs_Sheer", bang_pts, bang_rad, mat_hair_bangs, collection))

    # =========================================================================
    # 3. OUTER CHEEK & TEMPLE FACE-FRAMING TENDRILS (staying clear of eyes/nose/lips!)
    # =========================================================================
    n_frame = 260
    n_pts_frame = 16
    frame_pts = np.zeros((n_frame, n_pts_frame, 3), dtype=np.float32)
    frame_rad = np.zeros((n_frame, n_pts_frame), dtype=np.float32)

    for i in range(n_frame):
        is_left_side = i < 100
        if is_left_side:
            # Viewer's right (+X): subtle outer temple wisps curving down along cheekbone/ear edge (NOT across nose/lips!)
            theta = math.radians(rng.uniform(-45.0, -18.0))
            elev = math.radians(rng.uniform(22.0, 48.0))
            total_l = rng.uniform(0.14, 0.20)
            flow_fn = lambda t, p: Vector((0.12 if t < 0.5 else 0.02, 0.05, -0.98))
            wave_amp = rng.uniform(0.0025, 0.0050)
        else:
            # Viewer's left (-X): lush temple framing strands sweeping cleanly down left of collar (x < -0.04)
            theta = math.radians(rng.uniform(195.0, 226.0))
            elev = math.radians(rng.uniform(18.0, 48.0))
            total_l = rng.uniform(0.24, 0.34)
            flow_fn = lambda t, p: Vector((-0.14 + 0.08 * t, -0.24, -0.96))
            wave_amp = rng.uniform(0.0040, 0.0080)

        pr_r, nr_r = probe_scalp(theta, elev)
        root_w = m_head @ (pr_r - nr_r * 0.001)
        init_w = (m_head_rot @ (nr_r * 0.35 + Vector((0.35 if is_left_side else -0.35, -0.20, -0.60)))).normalized()
        wave_ax = Vector((0.3, 0.9, 0.0)).normalized()

        frame_pts[i] = _flow_strand(
            root_w=root_w,
            init_vel_w=init_w,
            target_flow_fn=flow_fn,
            n_pts=n_pts_frame,
            seg_len=total_l / (n_pts_frame - 1),
            wave_freq=rng.uniform(7.5, 11.0),
            wave_amp=wave_amp,
            wave_phase=rng.uniform(0.0, 6.28),
            wave_axis_w=wave_ax,
            bvh_body=bvh_body,
            bvh_sw=bvh_sw,
            face_center_w=face_center_w,
            min_clear_body=0.0032,
            min_clear_sw=0.0040,
            avoid_face=True,
            shelf_bias_y=0.25 if is_left_side else -0.30,
        )
        r0 = rng.uniform(0.00010, 0.00015)
        frame_rad[i] = r0 * (1.0 - 0.78 * np.linspace(0, 1, n_pts_frame) ** 1.4)

    created.append(_create_hair_curves_obj("Hair_FaceFraming", frame_pts, frame_rad, mat_hair_bangs, collection))

    # =========================================================================
    # 4. BACKLIT SILHOUETTE FLYAWAYS
    # =========================================================================
    n_fly = 600
    n_pts_fly = 12
    fly_pts = np.zeros((n_fly, n_pts_fly, 3), dtype=np.float32)
    fly_rad = np.zeros((n_fly, n_pts_fly), dtype=np.float32)

    for i in range(n_fly):
        theta = rng.uniform(-math.radians(45.0), math.radians(225.0))
        elev = sample_elev(-10.0, 85.0)
        pr_r, nr_r = probe_scalp(theta, elev)
        root_w = m_head @ pr_r
        nrm_w = (m_head_rot @ nr_r).normalized()
        init_w = (nrm_w * 0.40 + Vector((0, 0, -0.60))).normalized()
        total_l = rng.uniform(0.14, 0.28)
        p_phase = rng.uniform(0, 6.28)
        fly_pts[i] = _flow_strand(
            root_w=root_w,
            init_vel_w=init_w,
            target_flow_fn=lambda t, p, nw=nrm_w: (nw * (0.18 * (1.0 - t)) + Vector((0.0, 0.0, -1.0))).normalized(),
            n_pts=n_pts_fly,
            seg_len=total_l / (n_pts_fly - 1),
            wave_freq=rng.uniform(10.0, 16.0),
            wave_amp=rng.uniform(0.004, 0.009),
            wave_phase=p_phase,
            wave_axis_w=Vector((math.cos(p_phase), math.sin(p_phase), 0.1)).normalized(),
            bvh_body=bvh_body,
            bvh_sw=bvh_sw,
            face_center_w=face_center_w,
            min_clear_body=0.004,
            min_clear_sw=0.005,
            avoid_face=True,
            shelf_bias_y=0.30,
        )
        r0 = rng.uniform(0.000070, 0.000105)
        fly_rad[i] = r0 * (1.0 - 0.72 * np.linspace(0, 1, n_pts_fly))

    created.append(_create_hair_curves_obj("Hair_Flyaways", fly_pts, fly_rad, mat_hair_main, collection))

    # =========================================================================
    # 5. 3D FEATHERED EYEBROWS
    # =========================================================================
    n_brow_per_side = 240
    n_pts_brow = 5
    brow_pts = np.zeros((2 * n_brow_per_side, n_pts_brow, 3), dtype=np.float32)
    brow_rad = np.zeros((2 * n_brow_per_side, n_pts_brow), dtype=np.float32)
    idx_br = 0

    for sx in (-1.0, 1.0):
        for k in range(n_brow_per_side):
            t_b = rng.random()
            bx = sx * (0.0165 + 0.0295 * t_b)
            bz = 1.5642 + 0.0028 * math.sin(math.pi * t_b * 0.78) + rng.normal(0.0, 0.0018 * (1.0 - 0.3 * t_b))
            hit, nrm, _, _ = bvh_rest.ray_cast(Vector((bx, -0.22, bz)), Vector((0.0, 1.0, 0.0)), 0.15)
            p_base = hit if hit is not None else Vector((bx, -0.147, bz))
            n_surf = nrm.normalized() if nrm is not None else Vector((0.0, -1.0, 0.0))

            ang_up = 0.72 * (1.0 - t_b) + 0.15 * t_b - 0.15 * rng.random()
            dir_r = Vector((sx * 0.88, -0.12, ang_up)).normalized()
            h_len = rng.uniform(0.0055, 0.0090) * (1.0 - 0.25 * t_b)
            for s in range(n_pts_brow):
                ts = s / (n_pts_brow - 1)
                pr = p_base - n_surf * 0.0004 + dir_r * (h_len * ts) + n_surf * (0.0010 * math.sin(ts * math.pi * 0.8))
                brow_pts[idx_br, s] = (m_head @ pr)[:]
            r0 = rng.uniform(0.000090, 0.000125)
            brow_rad[idx_br] = r0 * (1.0 - 0.75 * np.linspace(0, 1, n_pts_brow))
            idx_br += 1

    created.append(_create_hair_curves_obj("Eyebrows_3D", brow_pts, brow_rad, mat_brow, collection))

    # =========================================================================
    # 6. 3D CURVED UPPER & LOWER EYELASHES
    # =========================================================================
    n_up_per_eye = 160
    n_lo_per_eye = 46
    n_pts_lash = 6
    tot_lash = 2 * (n_up_per_eye + n_lo_per_eye)
    lash_pts = np.zeros((tot_lash, n_pts_lash, 3), dtype=np.float32)
    lash_rad = np.zeros((tot_lash, n_pts_lash), dtype=np.float32)
    idx_l = 0

    for side_key, sx in (("l", 1.0), ("r", -1.0)):
        c_e = np.array(eye_centers[side_key], dtype=np.float64)
        for is_upper, n_lashes in ((True, n_up_per_eye), (False, n_lo_per_eye)):
            xs_sample = np.linspace(0.0205, 0.0422, 18)
            rim_pts = []
            for xv in xs_sample:
                xt = sx * xv
                mask = (
                    (np.abs(rest_co[:, 0] - xt) < 0.0022)
                    & (rest_co[:, 1] < -0.130)
                    & (np.linalg.norm(rest_co - c_e, axis=1) < 0.017)
                )
                if is_upper:
                    mask &= rest_co[:, 2] > c_e[2] - 0.001
                    sub = rest_co[mask]
                    if len(sub) > 0:
                        cand = sub[np.argsort(sub[:, 2])[:3]]
                        rim_pts.append(cand[np.argmin(cand[:, 1])])
                else:
                    mask &= rest_co[:, 2] < c_e[2] + 0.001
                    sub = rest_co[mask]
                    if len(sub) > 0:
                        cand = sub[np.argsort(sub[:, 2])[-3:]]
                        rim_pts.append(cand[np.argmin(cand[:, 1])])

            rim_arr = np.array(rim_pts, dtype=np.float64)
            ts_rim = np.linspace(0.0, 1.0, len(rim_arr))

            for k in range(n_lashes):
                t_l = (k + rng.uniform(0.1, 0.9)) / n_lashes
                px = float(np.interp(t_l, ts_rim, rim_arr[:, 0]))
                py = float(np.interp(t_l, ts_rim, rim_arr[:, 1]))
                pz = float(np.interp(t_l, ts_rim, rim_arr[:, 2]))
                p0 = Vector((px, py, pz))

                if is_upper:
                    l_len = (0.0038 + 0.0034 * math.sin(t_l * math.pi * 0.85)) * rng.uniform(0.88, 1.12)
                    dir_out = Vector((sx * (0.24 * t_l + 0.04), -0.96, 0.04)).normalized()
                    curl_up = Vector((0.0, 0.0, 1.0))
                    r0 = rng.uniform(0.000080, 0.000115)
                else:
                    l_len = (0.0022 + 0.0020 * math.sin(t_l * math.pi * 0.9)) * rng.uniform(0.85, 1.15)
                    dir_out = Vector((sx * (0.20 * t_l), -0.90, -0.32)).normalized()
                    curl_up = Vector((0.0, 0.0, -0.40))
                    r0 = rng.uniform(0.000055, 0.000075)

                for s in range(n_pts_lash):
                    ts = s / (n_pts_lash - 1)
                    p_cur = p0 + dir_out * (l_len * ts) + curl_up * (l_len * 0.34 * (ts ** 2))
                    lash_pts[idx_l, s] = (m_head @ p_cur)[:]
                lash_rad[idx_l] = r0 * (1.0 - 0.85 * np.linspace(0, 1, n_pts_lash))
                idx_l += 1

    created.append(_create_hair_curves_obj("Eyelashes_3D", lash_pts, lash_rad, mat_lash, collection))
    return created
