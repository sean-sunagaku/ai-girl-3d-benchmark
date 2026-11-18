"""Procedural dusty-pink knitted crew-neck sweater built cleanly around the posed character with BVH skin clearance, 3D ribbing, cloth folds, and wool fuzz."""

import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree


def _sample_body_radius(
    bvh: BVHTree,
    origin: Vector,
    direction: Vector,
    min_r: float,
    max_r: float,
    min_nrm_dot: float = -1.0,
    ray_pad: float = 0.08,
    fallback_r: float = None,
) -> float:
    """Raycast inward toward torso/arm centerline to find outermost outward-facing skin radius."""
    start = origin + direction * (max_r + ray_pad)
    hit, nrm, _, _ = bvh.ray_cast(start, -direction, max_r + ray_pad)
    if hit is not None and nrm.dot(direction) >= min_nrm_dot:
        r = (hit - origin).dot(direction)
        if fallback_r is not None:
            if min_r < r <= max_r + 0.005:
                return min(r, max_r)
            return fallback_r
        if r > min_r:
            return min(r, max_r)
    return fallback_r if fallback_r is not None else min_r


def create_sweater_materials(knit_imgs: dict) -> tuple:
    """Create dusty rose-pink stockinette knit & 2x2 ribbed knit materials with fuzzy sheen."""
    def _build_mat(name: str, img_a, img_h, img_r, bump_dist: float, bump_str: float):
        mat = bpy.data.materials.new(name)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        out = nodes.new("ShaderNodeOutputMaterial")
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Sheen Weight"].default_value = 0.88
        bsdf.inputs["Sheen Roughness"].default_value = 0.62
        bsdf.inputs["Sheen Tint"].default_value = (1.0, 0.90, 0.91, 1.0)
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
        bsdf.inputs["Subsurface Weight"].default_value = 0.06
        bsdf.inputs["Subsurface Scale"].default_value = 0.003
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.45)

        uv = nodes.new("ShaderNodeTexCoord")

        ta = nodes.new("ShaderNodeTexImage")
        ta.image = img_a
        th = nodes.new("ShaderNodeTexImage")
        th.image = img_h

        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_str
        bump.inputs["Distance"].default_value = bump_dist

        links.new(uv.outputs["UV"], ta.inputs["Vector"])
        links.new(uv.outputs["UV"], th.inputs["Vector"])
        links.new(ta.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(th.outputs["Color"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        if img_r is not None:
            tr = nodes.new("ShaderNodeTexImage")
            tr.image = img_r
            links.new(uv.outputs["UV"], tr.inputs["Vector"])
            links.new(tr.outputs["Color"], bsdf.inputs["Roughness"])
        else:
            bsdf.inputs["Roughness"].default_value = 0.82

        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat

    mat_main = _build_mat(
        "Sweater_Pink_Knit",
        knit_imgs["albedo"],
        knit_imgs["height"],
        knit_imgs["roughness"],
        bump_dist=0.0017,
        bump_str=0.52,
    )
    mat_rib = _build_mat(
        "Sweater_Pink_Ribbed",
        knit_imgs["rib_albedo"],
        knit_imgs["rib_height"],
        knit_imgs["roughness"],
        bump_dist=0.0022,
        bump_str=0.65,
    )
    return mat_main, mat_rib


def _pose_bone_world(arm_obj: bpy.types.Object, bone_name: str):
    pb = arm_obj.pose.bones[bone_name]
    mat = arm_obj.matrix_world @ pb.matrix
    head = mat.translation.copy()
    tail = mat @ Vector((0.0, pb.length, 0.0))
    return head, tail


def build_sweater(
    body_obj: bpy.types.Object,
    arm_obj: bpy.types.Object,
    landmarks: dict,
    knit_imgs: dict,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Construct tailored dusty-pink knit sweater around the POSED character with guaranteed skin clearance."""
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    body_eval = body_obj.evaluated_get(depsgraph)
    bvh = BVHTree.FromObject(body_eval, depsgraph)

    # Read posed skeleton centers
    spine_h, spine_t = _pose_bone_world(arm_obj, "spine_base")
    chest_h, chest_t = _pose_bone_world(arm_obj, "chest")
    neck_h, neck_t = _pose_bone_world(arm_obj, "neck")

    # Torso centerline frame in posed space
    torso_axis = (chest_t - spine_h).normalized()
    torso_right = Vector((1.0, 0.0, 0.0))
    torso_fwd = torso_right.cross(torso_axis).normalized()
    torso_right = torso_axis.cross(torso_fwd).normalized()  # +X (subject's left)
    # Note: torso_fwd points toward -Y (front) when chest leans forward

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.verify()

    def _make_face_uv(verts, uvs, mat_idx=0):
        f = bm.faces.new(verts)
        f.material_index = mat_idx
        for loop, uv in zip(f.loops, uvs):
            loop[uv_layer].uv = uv
        return f

    # =========================================================================
    # 1. TORSO + FULL SHOULDER PANEL + CREW-NECK COLLAR + WAIST HEM
    # =========================================================================
    NU = 56
    NV_BODY = 16
    NV_YOKE = 5
    NV_TOT = NV_BODY + NV_YOKE

    # Posed base and shoulder top Z heights along posed spine
    z_bot_c = spine_h + torso_axis * 0.015  # ~1.035m
    z_sh_c = chest_t - torso_axis * 0.022   # ~1.343m (relaxed natural shoulder level)
    collar_c = neck_h + torso_axis * 0.004  # centered cleanly on posed neck base!

    torso_verts = [[None] * NU for _ in range(NV_TOT)]

    for i in range(NV_TOT):
        for j in range(NU):
            phi = math.pi * 0.5 + 2.0 * math.pi * (j / NU)
            dir_v = (torso_right * math.cos(phi) - torso_fwd * math.sin(phi)).normalized()
            front_dip = 0.5 * (1.0 - math.sin(phi))  # 0 at back (+Y), 1 at front (-Y)

            if i < NV_BODY:
                tv = i / (NV_BODY - 1)
                origin = z_bot_c.lerp(z_sh_c, tv)
                origin = origin + torso_fwd * (0.025 + 0.014 * math.sin(tv * math.pi))
                # Keep torso sampling strictly inside ribcage/shoulder width so extended upper arms are NEVER grabbed!
                max_r = 0.124 * abs(math.cos(phi)) + 0.116 * abs(math.sin(phi))
                if tv > 0.78:
                    max_r = 0.138 * abs(math.cos(phi)) + 0.116 * abs(math.sin(phi))
                r_body = _sample_body_radius(bvh, origin, dir_v, min_r=0.075, max_r=max_r)
                r_body = min(r_body, max_r)

                front_w = max(0.0, -math.sin(phi)) ** 2
                ease = 0.013 + 0.007 * (1.0 - tv) + 0.010 * front_w * math.sin(tv * math.pi)
                r = r_body + ease
                r_min_env = (0.114 if abs(math.cos(phi)) > 0.6 else 0.098) * (1.0 - 0.08 * tv)
                if tv < 0.78:
                    r = max(r, r_min_env)
                co = origin + dir_v * r
                # Gentle natural shoulder slope while keeping full clearance above deltoid top
                co -= torso_axis * (0.005 * (abs(math.cos(phi)) ** 2) * (tv ** 2))
                if tv > 0.72:
                    hit, _, _, _ = bvh.ray_cast(co + torso_axis * 0.05, -torso_axis, 0.12)
                    if hit is not None:
                        min_h = hit.dot(torso_axis) + 0.010
                        if co.dot(torso_axis) < min_h:
                            co += torso_axis * (min_h - co.dot(torso_axis))
            else:
                # Shoulder yoke rows closing smoothly over clavicles/trapezius to crew-neck collar!
                ty = (i - (NV_BODY - 1)) / NV_YOKE
                p_start = torso_verts[NV_BODY - 1][j].co
                r_neck_x = 0.068
                r_neck_y = 0.064
                neck_drop = -0.028 * front_dip + 0.012 * (1.0 - front_dip)
                p_neck = (
                    collar_c
                    + torso_fwd * 0.020
                    + torso_right * (r_neck_x * math.cos(phi))
                    - torso_fwd * (r_neck_y * math.sin(phi))
                    + torso_axis * neck_drop
                )
                co = p_start.lerp(p_neck, ty)
                arch = math.sin(ty * math.pi) * (0.018 + 0.014 * (abs(math.cos(phi)) ** 2))
                co = co + torso_axis * arch

                # Probe downward ONLY 3.2cm above shoulder (never high enough to hit tilted jaw!)
                if ty < 0.88:
                    hit, _, _, _ = bvh.ray_cast(co + torso_axis * 0.032, -torso_axis, 0.09)
                    if hit is not None:
                        min_h = hit.dot(torso_axis) + 0.010
                        cur_h = co.dot(torso_axis)
                        if cur_h < min_h:
                            co += torso_axis * (min_h - cur_h)

            torso_verts[i][j] = bm.verts.new(co)

    # Smooth sternum valley on chest
    for _ in range(2):
        for i in range(4, NV_BODY):
            for j in range(21, 36):
                c_prev = torso_verts[i][(j - 1) % NU].co
                c_next = torso_verts[i][(j + 1) % NU].co
                mid = 0.5 * (c_prev + c_next)
                if torso_verts[i][j].co.y > mid.y:
                    torso_verts[i][j].co.y = 0.35 * torso_verts[i][j].co.y + 0.65 * mid.y

    for i in range(NV_TOT - 1):
        for j in range(NU):
            jn = (j + 1) % NU
            v00 = torso_verts[i][j]
            v01 = torso_verts[i][jn]
            v11 = torso_verts[i + 1][jn]
            v10 = torso_verts[i + 1][j]
            u0 = 11.0 * (j / NU)
            u1 = 11.0 * ((j + 1) / NU)
            vt0 = 5.0 * (i / (NV_TOT - 1))
            vt1 = 5.0 * ((i + 1) / (NV_TOT - 1))
            _make_face_uv((v00, v01, v11, v10), ((u0, vt0), (u1, vt0), (u1, vt1), (u0, vt1)), mat_idx=0)

    # Ribbed Crew-Neck Collar (lies smoothly along upper chest / base of neck, NOT stiff turtleneck!)
    collar_prev = [torso_verts[NV_TOT - 1][j] for j in range(NU)]
    collar_steps = [
        # (along_axis, radial_inward, rib_amp)
        (0.005, 0.96, 0.0005),
        (0.011, 0.89, 0.0011),
        (0.015, 0.83, 0.0011),
        (0.016, 0.78, 0.0006),  # soft rolled collar rim
        (0.010, 0.74, 0.0000),  # inner collar edge tucked onto collarbone
        (-0.004, 0.73, 0.0000),
    ]
    for cs_idx, (d_ax, r_in, rib_amp) in enumerate(collar_steps):
        nxt_c = []
        for j in range(NU):
            p0 = torso_verts[NV_TOT - 1][j].co
            c_ref = collar_c + torso_fwd * 0.020
            c_plane = c_ref + torso_axis * ((p0 - c_ref).dot(torso_axis))
            rel = p0 - c_plane
            rib = rib_amp * math.cos(j * 2.0 * math.pi * 26.0 / NU)
            p_new = c_plane + rel * r_in + rel.normalized() * rib + torso_axis * d_ax
            nxt_c.append(bm.verts.new(p_new))
        v0 = 0.25 * (cs_idx / len(collar_steps))
        v1 = 0.25 * ((cs_idx + 1) / len(collar_steps))
        for j in range(NU):
            jn = (j + 1) % NU
            u0 = 3.2 * (j / NU)
            u1 = 3.2 * ((j + 1) / NU)
            _make_face_uv(
                (collar_prev[j], collar_prev[jn], nxt_c[jn], nxt_c[j]),
                ((u0, v0), (u1, v0), (u1, v1), (u0, v1)),
                mat_idx=1,
            )
        collar_prev = nxt_c

    # Ribbed Bottom Waist Hem
    hem_prev = [torso_verts[0][j] for j in range(NU)]
    hem_steps = [
        (-0.012, 0.985, 0.0010),
        (-0.028, 0.965, 0.0013),
        (-0.040, 0.955, 0.0010),
        (-0.043, 0.925, 0.0000),
        (-0.036, 0.875, 0.0000),
        (-0.015, 0.870, 0.0000),
    ]
    for hs_idx, (d_ax, r_in, rib_amp) in enumerate(hem_steps):
        nxt_h = []
        for j in range(NU):
            p0 = torso_verts[0][j].co
            c_ref = z_bot_c + torso_fwd * 0.025
            c_plane = c_ref + torso_axis * ((p0 - c_ref).dot(torso_axis))
            rel = p0 - c_plane
            rib = rib_amp * math.cos(j * 2.0 * math.pi * 28.0 / NU)
            p_new = c_plane + rel * r_in + rel.normalized() * rib + torso_axis * d_ax
            nxt_h.append(bm.verts.new(p_new))
        v0 = 0.30 * (hs_idx / len(hem_steps))
        v1 = 0.30 * ((hs_idx + 1) / len(hem_steps))
        for j in range(NU):
            jn = (j + 1) % NU
            u0 = 5.0 * (j / NU)
            u1 = 5.0 * ((j + 1) / NU)
            _make_face_uv(
                (hem_prev[jn], hem_prev[j], nxt_h[j], nxt_h[jn]),
                ((u1, v0), (u0, v0), (u0, v1), (u1, v1)),
                mat_idx=1,
            )
        hem_prev = nxt_h

    # =========================================================================
    # 2. SET-IN POSED SLEEVES (zero-twist smooth tubes tucked cleanly into shoulder seams!)
    # =========================================================================
    for side in ("R", "L"):
        sx = 1.0 if side == "L" else -1.0
        sh_h, sh_t = _pose_bone_world(arm_obj, f"upper_arm.{side}")
        el_h, el_t = _pose_bone_world(arm_obj, f"forearm.{side}")
        ha_h, _ = _pose_bone_world(arm_obj, f"hand.{side}")

        # Start slightly inside the torso shoulder seam with full upper-back shoulder clearance
        sh_seam = sh_h - torso_right * (sx * 0.024) + torso_axis * 0.014 - torso_fwd * 0.008
        fore_dir = (ha_h - el_h).normalized()
        cuff_start = ha_h - fore_dir * 0.032

        # Quadratic-smoothed dense centerline along posed arm
        ctrl_pts = [
            sh_seam,
            sh_h + (el_h - sh_h) * 0.25,
            sh_h.lerp(el_h, 0.72),
            el_h + (el_h - sh_h) * 0.04 + fore_dir * 0.04,  # rounded elbow curve
            el_h.lerp(cuff_start, 0.50),
            cuff_start,
        ]
        # Subdivide polyline with Chaikin smoothing for silky elbow bend
        pts = list(ctrl_pts)
        for _ in range(3):
            npts = [pts[0]]
            for k in range(len(pts) - 1):
                npts.append(pts[k].lerp(pts[k + 1], 0.25))
                npts.append(pts[k].lerp(pts[k + 1], 0.75))
            npts.append(pts[-1])
            pts = npts

        # Resample to 22 evenly spaced rings along sleeve
        cum_len = [0.0]
        for k in range(1, len(pts)):
            cum_len.append(cum_len[-1] + (pts[k] - pts[k - 1]).length)
        tot_len = cum_len[-1]

        n_rings = 22
        sleeve_pts = []
        for r_i in range(n_rings):
            target_d = (r_i / (n_rings - 1)) * tot_len
            k = 0
            while k < len(cum_len) - 2 and cum_len[k + 1] < target_d:
                k += 1
            seg = max(1e-6, cum_len[k + 1] - cum_len[k])
            ft = (target_d - cum_len[k]) / seg
            sleeve_pts.append((r_i / (n_rings - 1), pts[k].lerp(pts[k + 1], ft)))

        N_SEG = 28
        # Build parallel-transported radial directions along sleeve
        tang0 = (sleeve_pts[1][1] - sleeve_pts[0][1]).normalized()
        bx0 = tang0.cross(Vector((0.0, 0.0, 1.0)))
        if bx0.length < 1e-4:
            bx0 = Vector((0.0, 1.0, 0.0))
        bx0.normalize()
        bz0 = bx0.cross(tang0).normalized()

        cur_dirs = [
            (bx0 * math.cos(2.0 * math.pi * k / N_SEG) + bz0 * math.sin(2.0 * math.pi * k / N_SEG)).normalized()
            for k in range(N_SEG)
        ]
        prev_tang = tang0

        sleeve_rings = []
        for r_i, (t_all, p_c) in enumerate(sleeve_pts):
            if r_i == 0:
                tang = tang0
            elif r_i == len(sleeve_pts) - 1:
                tang = (p_c - sleeve_pts[r_i - 1][1]).normalized()
            else:
                tang = (sleeve_pts[r_i + 1][1] - sleeve_pts[r_i - 1][1]).normalized()

            q_step = prev_tang.rotation_difference(tang)
            cur_dirs = [(q_step @ d).normalized() for d in cur_dirs]
            prev_tang = tang

            # Cozy slim-relaxed feminine sleeve radius: ~6.5cm at upper arm tapering to ~3.8cm wrist
            base_r = 0.065 * (1.0 - t_all) + 0.038 * t_all
            base_r += 0.005 * math.exp(-((t_all - 0.85) / 0.11) ** 2)
            if r_i == 0:
                base_r *= 0.94

            # Organic transverse sleeve bunching wrinkles along forearm/elbow (t_all in [0.35, 0.92])
            wrinkle_env = math.sin(max(0.0, min(1.0, (t_all - 0.28) / 0.66)) * math.pi)
            ring_v = []
            for k in range(N_SEG):
                ang = 2.0 * math.pi * k / N_SEG
                rad_dir = cur_dirs[k]
                # Sample actual posed arm skin radius along rad_dir without accidentally grabbing torso/elbow!
                r_skin = _sample_body_radius(
                    bvh,
                    p_c,
                    rad_dir,
                    min_r=0.022,
                    max_r=min(0.085, base_r + 0.014),
                    min_nrm_dot=0.25,
                    ray_pad=0.018,
                    fallback_r=base_r,
                )
                cap_r = base_r + (0.010 if r_i < 5 else 0.012)
                r_skin = min(r_skin, cap_r)
                # Ensure shoulder top & elbow clear skin cleanly everywhere!
                min_clear = 0.013 if r_i < 4 else 0.010
                wrinkle = 0.0038 * math.sin(t_all * 26.0 + math.sin(ang) * 1.4) * wrinkle_env
                r_use = max(base_r + wrinkle, r_skin + min_clear)
                ring_v.append(bm.verts.new(p_c + rad_dir * r_use))
            sleeve_rings.append(ring_v)

        # Cap inner shoulder seam inside torso (hidden inside torso shell so sleeve is manifold!)
        sh_cap_v = bm.verts.new(sleeve_pts[0][1] - tang0 * 0.015)
        for k in range(N_SEG):
            bm.faces.new((sleeve_rings[0][(k + 1) % N_SEG], sh_cap_v, sleeve_rings[0][k]))

        # Sleeve faces with V running lengthwise down arm
        for r_i in range(len(sleeve_rings) - 1):
            r0 = sleeve_rings[r_i]
            r1 = sleeve_rings[r_i + 1]
            v0 = 4.6 * (r_i / (len(sleeve_rings) - 1))
            v1 = 4.6 * ((r_i + 1) / (len(sleeve_rings) - 1))
            for k in range(N_SEG):
                kn = (k + 1) % N_SEG
                u0 = 4.2 * (k / N_SEG)
                u1 = 4.2 * ((k + 1) / N_SEG)
                _make_face_uv(
                    (r0[k], r0[kn], r1[kn], r1[k]),
                    ((u0, v0), (u1, v0), (u1, v1), (u0, v1)),
                    mat_idx=0,
                )

        # Snug 2x2 Ribbed Wrist Cuff
        q_end = prev_tang.rotation_difference(fore_dir)
        cur_dirs = [(q_end @ d).normalized() for d in cur_dirs]
        prev_c_ring = sleeve_rings[-1]
        cuff_steps = [
            (0.006, 0.036, 0.0005),
            (0.016, 0.032, 0.0011),
            (0.026, 0.031, 0.0011),
            (0.035, 0.031, 0.0007),
            (0.038, 0.028, 0.0000),
            (0.033, 0.022, 0.0000),
            (0.012, 0.021, 0.0000),
        ]
        for cs_idx, (dist_ax, r_cuff, rib_amp) in enumerate(cuff_steps):
            c_ring_pos = cuff_start + fore_dir * dist_ax
            nxt_ring = []
            for k in range(N_SEG):
                rib = rib_amp * math.cos(k * 2.0 * math.pi * 14.0 / N_SEG)
                nxt_ring.append(bm.verts.new(c_ring_pos + cur_dirs[k] * (r_cuff + rib)))
            v_prev = 0.35 * (cs_idx / len(cuff_steps))
            v_next = 0.35 * ((cs_idx + 1) / len(cuff_steps))
            for k in range(N_SEG):
                kn = (k + 1) % N_SEG
                u0 = 1.8 * (k / N_SEG)
                u1 = 1.8 * ((k + 1) / N_SEG)
                _make_face_uv(
                    (prev_c_ring[k], prev_c_ring[kn], nxt_ring[kn], nxt_ring[k]),
                    ((u0, v_prev), (u1, v_prev), (u1, v_next), (u0, v_next)),
                    mat_idx=1,
                )
            prev_c_ring = nxt_ring

    # =========================================================================
    # 3. FINAL TORSO CLOTH FOLDS & GUARANTEED BVH SKIN CLEARANCE PASS
    # =========================================================================
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    bm.verts.ensure_lookup_table()

    # Add subtle organic torso drape/armpit folds on torso vertices (first NU * NV_TOT verts)
    n_torso_v = NU * NV_TOT
    for v_idx in range(n_torso_v):
        v = bm.verts[v_idx]
        x, y, z = v.co.x, v.co.y, v.co.z
        # Fade folds out near collar & waist hem
        env = max(0.0, min(1.0, (z - 1.05) / 0.05)) * max(0.0, min(1.0, (1.33 - z) / 0.04))
        diag_l = math.sin(x * 20.0 + z * 26.0 + math.sin(y * 16.0)) * math.exp(-((x - 0.06) / 0.09) ** 2)
        diag_r = math.sin(-x * 20.0 + z * 26.0 + math.sin(y * 16.0)) * math.exp(-((x + 0.06) / 0.09) ** 2)
        sag = 0.5 * math.sin(z * 44.0 + math.cos(x * 14.0) * 1.4) * max(0.0, min(1.0, (1.16 - z) / 0.10))
        v.co += v.normal * ((0.0022 * (diag_l + diag_r) + 0.0025 * sag) * env)

    # Safety BVH skin push-out pass: prevent any body vertex from ever poking through sweater
    for v in bm.verts:
        loc, nrm, _, dist = bvh.find_nearest(v.co, 0.025)
        if loc is not None:
            signed_d = (v.co - loc).dot(nrm)
            if signed_d < 0.0105:
                if v.normal.dot(nrm) > 0.0:
                    v.co = loc + nrm * 0.0115

    bm.normal_update()
    mesh = bpy.data.meshes.new("Sweater_Mesh")
    bm.to_mesh(mesh)
    bm.free()

    sweater_obj = bpy.data.objects.new("Sweater", mesh)
    collection.objects.link(sweater_obj)
    for p in sweater_obj.data.polygons:
        p.use_smooth = True

    mat_main, mat_rib = create_sweater_materials(knit_imgs)
    sweater_obj.data.materials.append(mat_main)
    sweater_obj.data.materials.append(mat_rib)

    sub = sweater_obj.modifiers.new("Subdivision", type="SUBSURF")
    sub.levels = 1
    sub.render_levels = 2

    return sweater_obj


def build_sweater_fuzz(sweater_obj: bpy.types.Object, collection: bpy.types.Collection, count: int = 4000, seed: int = 123):
    """Scatter delicate 3D wool fuzz fibers across posed sweater surface for soft backlit knit halo."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    sw_eval = sweater_obj.evaluated_get(depsgraph)
    mesh = sw_eval.to_mesh()
    mesh.calc_loop_triangles()

    rng = np.random.default_rng(seed)
    tris = mesh.loop_triangles
    areas = np.array([t.area for t in tris], dtype=np.float64)
    probs = areas / np.sum(areas)
    chosen = rng.choice(len(tris), size=count, p=probs)

    verts_co = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
    verts_nrm = np.array([v.normal[:] for v in mesh.vertices], dtype=np.float64)

    n_pts = 3
    pos = np.zeros((count, n_pts, 3), dtype=np.float32)
    rad = np.zeros((count, n_pts), dtype=np.float32)

    for k, t_idx in enumerate(chosen):
        tri = tris[t_idx]
        i0, i1, i2 = tri.vertices
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1.0:
            r1, r2 = 1.0 - r1, 1.0 - r2
        r0 = 1.0 - r1 - r2

        p = r0 * verts_co[i0] + r1 * verts_co[i1] + r2 * verts_co[i2]
        n = r0 * verts_nrm[i0] + r1 * verts_nrm[i1] + r2 * verts_nrm[i2]
        n /= (np.linalg.norm(n) + 1e-8)

        rand_t = rng.normal(0.0, 1.0, size=3)
        rand_t -= n * np.dot(rand_t, n)
        rand_t /= (np.linalg.norm(rand_t) + 1e-8)

        length = rng.uniform(0.0035, 0.0075)
        pos[k, 0] = p - n * 0.0004
        pos[k, 1] = p + n * (length * 0.55) + rand_t * (length * 0.35)
        pos[k, 2] = p + n * (length * 0.85) + rand_t * (length * 0.80)

        r_base = rng.uniform(0.000045, 0.000075)
        rad[k, 0] = r_base
        rad[k, 1] = r_base * 0.65
        rad[k, 2] = r_base * 0.15

    sw_eval.to_mesh_clear()

    hc = bpy.data.hair_curves.new("Sweater_Fuzz_Curves")
    hc.add_curves([n_pts] * count)
    hc.points.foreach_set("position", pos.reshape(-1))
    rad_attr = hc.attributes.new(name="radius", type="FLOAT", domain="POINT")
    rad_attr.data.foreach_set("value", rad.reshape(-1))

    mat_fuzz = bpy.data.materials.new("Sweater_Wool_Fuzz")
    nodes = mat_fuzz.node_tree.nodes
    links = mat_fuzz.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.76, 0.50, 0.52, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.72
    bsdf.inputs["Subsurface Weight"].default_value = 0.35
    bsdf.inputs["Subsurface Scale"].default_value = 0.002
    bsdf.inputs["Transmission Weight"].default_value = 0.15
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    hc.materials.append(mat_fuzz)

    fuzz_obj = bpy.data.objects.new("Sweater_Fuzz", hc)
    collection.objects.link(fuzz_obj)
    return fuzz_obj
