"""
sweater.py - 3D chunky dusty-pink knitted crew-neck sweater with continuous
dropped-shoulder reaching sleeves, chunky ribbed collar/cuffs/hem, organic cloth folds,
3D knit displacement, and fine fuzzy wool halo fibers.
"""
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix

from sculpt import TORSO_PIVOT, TORSO_QUAT, HEAD_PIVOT, HEAD_QUAT


def get_torso_matrix():
    return (
        Matrix.Translation(TORSO_PIVOT)
        @ TORSO_QUAT.to_matrix().to_4x4()
        @ Matrix.Translation(-TORSO_PIVOT)
    )


def _catmull_rom_chain(ctrl_pts, n_samples):
    """Sample a smooth 3D Catmull-Rom spline through control points."""
    pts = np.asarray(ctrl_pts, dtype=np.float64)
    # Pad endpoints by reflection
    p_ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    n_seg = len(pts) - 1
    out = []
    for k in range(n_samples):
        u = (k / (n_samples - 1)) * n_seg
        seg = min(int(math.floor(u)), n_seg - 1)
        t = u - seg
        p0, p1, p2, p3 = p_ext[seg], p_ext[seg + 1], p_ext[seg + 2], p_ext[seg + 3]
        t2, t3 = t * t, t * t * t
        pos = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out.append(pos)
    return np.array(out, dtype=np.float64)


def _build_torso_mesh(mats, torso_mat):
    """
    Build sculpted cozy chunky-knit sweater torso (waist z=0.985 to neckline z=1.395)
    with natural bust fullness, relaxed knit drape, shoulder contour, and diagonal tension folds.
    """
    n_rings = 56
    n_segs = 72

    bm = bmesh.new()
    rings_verts = []

    # Height profile from bottom hem (v=0) to collar base (v=1)
    for ir in range(n_rings):
        v = ir / (n_rings - 1)
        # Smooth rise: waist (0.990) -> full shoulder yoke (1.355 at v=0.80) -> crew-neck rim (1.388 at v=1.0)
        if v < 0.80:
            s0 = v / 0.80
            z_base = 0.990 + s0 * 0.365
            rx = 0.144 + 0.028 * math.sin(s0 * math.pi * 0.80) + 0.024 * smoothstep_scalar(0.45, 1.0, s0)
            bust = math.exp(-0.5 * ((v - 0.50) / 0.18) ** 2)
            ry_f = 0.104 + 0.036 * bust + 0.008 * smoothstep_scalar(0.0, 0.3, v)
            ry_b = 0.098 + 0.018 * math.sin(s0 * math.pi)
            y_shift = -0.018 - 0.008 * bust
        else:
            # Plump continuous rounded shoulder yoke tapering cleanly into the collar band
            s = (v - 0.80) / 0.20  # 0 at shoulder line, 1 at collar seam
            # Convex arch so shoulder fabric stays comfortably above clavicles
            e = s * s * (3.0 - 2.0 * s)
            z_base = 1.355 + 0.033 * math.sin(s * math.pi * 0.5)
            rx = 0.184 * (1.0 - e) + 0.068 * e
            ry_f = 0.118 * (1.0 - e) + 0.068 * e
            ry_b = 0.106 * (1.0 - e) + 0.066 * e
            y_shift = -0.020 * (1.0 - e) + (-0.018) * e

        ring = []
        for iseg in range(n_segs):
            u = iseg / n_segs
            theta = u * 2.0 * math.pi  # 0 = +X, pi/2 = +Y (back), pi = -X, 3pi/2 = -Y (front)
            ct = math.cos(theta)
            st = math.sin(theta)

            ry = ry_b if st >= 0.0 else ry_f
            x_loc = rx * ct
            y_loc = y_shift + ry * st

            if st < 0.0:
                front_w = -st
                bust_z = math.exp(-0.5 * ((v - 0.50) / 0.17) ** 2)
                bust_side = math.exp(-0.5 * ((abs(ct) - 0.42) / 0.32) ** 2)
                y_loc -= 0.007 * front_w * bust_z * bust_side

            # Sculpted diagonal tension folds & relaxed vertical drape ripples
            fold_diag_l = math.sin((v * 9.5 - ct * 3.8 + 0.6) * math.pi) * math.exp(-0.5 * ((v - 0.46) / 0.28) ** 2)
            fold_diag_r = math.sin((v * 9.5 + ct * 3.8 + 1.4) * math.pi) * math.exp(-0.5 * ((v - 0.44) / 0.28) ** 2)
            fold_vert = math.sin(theta * 7.0 + v * 4.0) * 0.5 + math.cos(theta * 11.0 - v * 2.5) * 0.5
            fold_env = smoothstep_scalar(0.05, 0.22, v) * smoothstep_scalar(0.92, 0.68, v)
            dr = fold_env * (
                0.0040 * (0.5 * fold_diag_l + 0.5 * fold_diag_r)
                + 0.0026 * fold_vert
            )
            waist_fold = math.sin(v * 22.0) * math.exp(-0.5 * ((v - 0.22) / 0.14) ** 2) * 0.0028
            dr += waist_fold

            shoulder_flare = math.exp(-0.5 * ((v - 0.78) / 0.09) ** 2) * (ct * ct) * 0.008

            x_loc += ct * (dr + shoulder_flare)
            y_loc += st * dr

            # Crew-neck dip in front (-Y), slight rise in back (+Y), keeping shoulder yoke snug under collar
            neck_slope = smoothstep_scalar(0.76, 1.0, v)
            z_loc = z_base + neck_slope * (0.015 * st - 0.005 * (1.0 - abs(ct)) * max(-st, 0.0))
            if v > 0.70:
                sh_s = (v - 0.70) / 0.30
                z_loc -= 0.008 * (ct * ct) * math.sin(sh_s * math.pi)

            p = torso_mat @ Vector((x_loc, y_loc, z_loc))
            ring.append(bm.verts.new(p))
        rings_verts.append(ring)

    # Create quad faces with clean knit-aligned UVs
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for ir in range(n_rings - 1):
        v0 = ir / (n_rings - 1)
        v1 = (ir + 1) / (n_rings - 1)
        for iseg in range(n_segs):
            iseg_n = (iseg + 1) % n_segs
            u0 = iseg / n_segs
            u1 = (iseg + 1) / n_segs
            f = bm.faces.new((
                rings_verts[ir][iseg],
                rings_verts[ir][iseg_n],
                rings_verts[ir + 1][iseg_n],
                rings_verts[ir + 1][iseg],
            ))
            f.smooth = True
            # Material slot 0 = main stockinette knit, slot 1 = ribbed hem at bottom
            f.material_index = 1 if ir < 5 else 0
            uvs = [(u0 * 1.6, v0 * 1.0), (u1 * 1.6, v0 * 1.0), (u1 * 1.6, v1 * 1.0), (u0 * 1.6, v1 * 1.0)]
            for loop, uv in zip(f.loops, uvs):
                loop[uv_layer].uv = uv

    # Close bottom waist cleanly inward with a tucked hem rim so underside is solid
    bottom_center = torso_mat @ Vector((0.0, -0.02, 0.995))
    vc = bm.verts.new(bottom_center)
    for iseg in range(n_segs):
        iseg_n = (iseg + 1) % n_segs
        f = bm.faces.new((rings_verts[0][iseg_n], rings_verts[0][iseg], vc))
        f.smooth = True
        f.material_index = 1

    # Add inner collar tuck lip (2 rings going inward-downward inside neck so collar never looks paper-thin)
    inner_ring = []
    for iseg in range(n_segs):
        top_v = rings_verts[-1][iseg]
        u = iseg / n_segs
        theta = u * 2.0 * math.pi
        p_in = top_v.co - (torso_mat.to_3x3() @ Vector((0.012 * math.cos(theta), 0.012 * math.sin(theta), 0.020)))
        inner_ring.append(bm.verts.new(p_in))
    for iseg in range(n_segs):
        iseg_n = (iseg + 1) % n_segs
        f = bm.faces.new((
            rings_verts[-1][iseg],
            rings_verts[-1][iseg_n],
            inner_ring[iseg_n],
            inner_ring[iseg],
        ))
        f.smooth = True
        f.material_index = 1

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("Sweater_Torso_Mesh")
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new("Sweater_Torso", me)
    obj.data.materials.append(mats["sweater"])
    obj.data.materials.append(mats["sweater_rib"])
    return obj


def _build_collar_mesh(mats, torso_mat):
    """
    Build thick chunky 1x1 ribbed crew-neck collar band wrapping cleanly around the base of the neck,
    matching the prominent ~3.2cm chunky ribbed crew collar in input/reference.png.
    """
    n_around = 72
    n_profile = 18
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    rings = []
    for ia in range(n_around):
        u = ia / n_around
        theta = u * 2.0 * math.pi
        ct = math.cos(theta)
        st = math.sin(theta)

        # Centerline of collar tube in torso-local coords
        # Matches top neckline of torso closely, hugging base of neck with slight cozy ease
        cx = 0.072 * ct
        cy = -0.019 + 0.072 * st
        # Front dip (-Y -> st=-1) down to z=1.366, back rise (+Y -> st=+1) up to z=1.404
        cz = 1.384 + 0.017 * st - 0.008 * (1.0 - abs(ct)) * max(-st, 0.0)

        # Outward radial unit vector & tangent along collar slope
        rad_vec = Vector((ct, st, 0.0)).normalized()
        # Tilt band gently to follow neck/shoulder surface normal
        up_vec = Vector((-0.22 * ct, -0.18 * st, 1.0)).normalized()

        ring = []
        for ip in range(n_profile):
            phi = (ip / n_profile) * 2.0 * math.pi
            # Elliptical flat-ribbed collar band cross-section: 1.55cm tall along neck, 0.85cm thick
            # Add subtle 3D rib flutes directly in geometry around circumference!
            rib_bump = 0.0014 * math.cos(u * 42.0 * 2.0 * math.pi)
            thick = 0.0088 + rib_bump * max(0.0, math.cos(phi))
            height = 0.0175
            # Slight subtle seam tuck at bottom outer edge
            p_local = Vector((cx, cy, cz)) + rad_vec * (thick * math.cos(phi)) + up_vec * (height * math.sin(phi))
            p_world = torso_mat @ p_local
            ring.append(bm.verts.new(p_world))
        rings.append(ring)

    for ia in range(n_around):
        ia_n = (ia + 1) % n_around
        u0 = ia / n_around
        u1 = (ia + 1) / n_around
        for ip in range(n_profile):
            ip_n = (ip + 1) % n_profile
            v0 = ip / n_profile
            v1 = (ip + 1) / n_profile
            f = bm.faces.new((
                rings[ia][ip],
                rings[ia_n][ip],
                rings[ia_n][ip_n],
                rings[ia][ip_n],
            ))
            f.smooth = True
            # U runs around collar so vertical knit ribs in Knit_Rib_Height run perpendicular to collar!
            uvs = [(u0 * 2.2, v0 * 0.65), (u1 * 2.2, v0 * 0.65), (u1 * 2.2, v1 * 0.65), (u0 * 2.2, v1 * 0.65)]
            for loop, uv in zip(f.loops, uvs):
                loop[uv_layer].uv = uv

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("Sweater_Collar_Mesh")
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new("Sweater_Collar", me)
    obj.data.materials.append(mats["sweater_rib"])
    return obj


def _build_sleeve_mesh(side_name, side_sign, mats, torso_mat):
    """
    Build a continuous dropped-shoulder reaching sweater sleeve with lush concentric
    compression folds/ruching, naturally extending forward toward the selfie camera
    and finishing with a chunky ribbed cuff.
    side_sign: -1.0 = Character's Right Arm (viewer's left, foreground prominent)
               +1.0 = Character's Left Arm (viewer's right, slightly more compact)
    """
    s = side_sign
    if s < 0:
        # Character's right arm (viewer's left): tucked smoothly into dropped shoulder, reaching toward lower-left
        ctrl = [
            [-0.118, -0.018, 1.275],  # Deeply seated inside torso shoulder
            [-0.168, -0.034, 1.274],  # Rounded dropped-shoulder junction flush with torso
            [-0.218, -0.090, 1.228],  # Upper arm angling forward-down
            [-0.244, -0.180, 1.168],  # Soft elbow bend
            [-0.236, -0.292, 1.128],  # Forearm reaching forward
            [-0.212, -0.402, 1.110],  # Wrist cuff approaching lower-left frame
        ]
    else:
        # Character's left arm (viewer's right): tucked smoothly into dropped shoulder, reaching toward lower-right
        ctrl = [
            [0.118, -0.016, 1.278],
            [0.168, -0.030, 1.276],
            [0.216, -0.085, 1.234],
            [0.240, -0.172, 1.176],
            [0.232, -0.280, 1.136],
            [0.210, -0.386, 1.116],
        ]

    n_rings = 54
    n_segs = 44
    path = _catmull_rom_chain(ctrl, n_rings)

    # Compute smooth Bishop parallel-transport frame along the sleeve spine
    tangents = np.zeros_like(path)
    tangents[0] = path[1] - path[0]
    tangents[-1] = path[-1] - path[-2]
    tangents[1:-1] = path[2:] - path[:-2]
    tangents /= (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12)

    normals = np.zeros_like(path)
    binormals = np.zeros_like(path)
    up_ref = np.array([0.0, 0.0, 1.0])
    n0 = up_ref - np.dot(up_ref, tangents[0]) * tangents[0]
    normals[0] = n0 / (np.linalg.norm(n0) + 1e-12)
    binormals[0] = np.cross(tangents[0], normals[0])

    for i in range(1, n_rings):
        v = np.cross(tangents[i - 1], tangents[i])
        c = np.dot(tangents[i - 1], tangents[i])
        if np.linalg.norm(v) < 1e-8:
            normals[i] = normals[i - 1]
        else:
            axis = v / np.linalg.norm(v)
            ang = math.acos(max(-1.0, min(1.0, float(c))))
            rot = Matrix.Rotation(ang, 3, Vector(axis))
            normals[i] = np.array(rot @ Vector(normals[i - 1]))
        normals[i] -= np.dot(normals[i], tangents[i]) * tangents[i]
        normals[i] /= (np.linalg.norm(normals[i]) + 1e-12)
        binormals[i] = np.cross(tangents[i], normals[i])

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    rings = []

    for ir in range(n_rings):
        t = ir / (n_rings - 1)  # 0 shoulder -> 1 cuff
        # Natural cozy knit sleeve radius: full at dropped shoulder (7.0cm), relaxed elbow (6.0cm), ribbed cuff (4.8cm)
        base_r = (
            0.071 * (1.0 - t)
            + 0.049 * t
            + 0.005 * math.sin(t * math.pi)
            - 0.004 * smoothstep_scalar(0.86, 1.0, t)
        )
        # Smooth rounded dome taper at inner shoulder anchor so sleeve seats cleanly INSIDE torso
        shoulder_taper = 0.50 + 0.50 * smoothstep_scalar(0.0, 0.14, t)
        base_r *= shoulder_taper

        # Natural subtle knit cloth folds & gentle elbow bunching (never bulbous Michelin rings!)
        fold_env = smoothstep_scalar(0.14, 0.30, t) * smoothstep_scalar(0.92, 0.78, t)
        ruche = (
            0.0028 * math.sin(t * 22.0 + (0.0 if s < 0 else 1.3))
            + 0.0016 * math.sin(t * 39.0 - 0.7)
            + 0.0012 * math.cos(t * 13.0 + 0.5)
        ) * fold_env

        ring = []
        for iseg in range(n_segs):
            u = iseg / n_segs
            theta = u * 2.0 * math.pi
            ct = math.cos(theta)
            st = math.sin(theta)

            circ_mod = 1.0 + 0.35 * math.sin(theta * 2.0 + t * 10.0) + 0.22 * math.cos(theta * 3.0 - t * 7.0)
            r_loc = base_r + ruche * circ_mod
            sag = -0.0048 * max(0.0, -ct) * smoothstep_scalar(0.18, 0.50, t) * smoothstep_scalar(0.96, 0.80, t)

            r_n = r_loc * (1.08 if t < 0.20 else 1.01)
            r_b = r_loc * (0.95 if t < 0.20 else 0.99)

            pt_local = (
                path[ir]
                + normals[ir] * (r_n * ct + sag)
                + binormals[ir] * (r_b * st)
            )
            p_world = torso_mat @ Vector(pt_local)
            ring.append(bm.verts.new(p_world))
        rings.append(ring)

    # Faces + knit-aligned UVs (courses run lengthwise along sleeve!)
    for ir in range(n_rings - 1):
        v0 = ir / (n_rings - 1)
        v1 = (ir + 1) / (n_rings - 1)
        is_cuff = ir >= (n_rings - 7)
        for iseg in range(n_segs):
            iseg_n = (iseg + 1) % n_segs
            u0 = iseg / n_segs
            u1 = (iseg + 1) / n_segs
            f = bm.faces.new((
                rings[ir][iseg],
                rings[ir][iseg_n],
                rings[ir + 1][iseg_n],
                rings[ir + 1][iseg],
            ))
            f.smooth = True
            f.material_index = 1 if is_cuff else 0
            uvs = [(u0 * 0.95, v0 * 1.15), (u1 * 0.95, v0 * 1.15), (u1 * 0.95, v1 * 1.15), (u0 * 0.95, v1 * 1.15)]
            for loop, uv in zip(f.loops, uvs):
                loop[uv_layer].uv = uv

    # Close wrist cuff with thick rolled ribbed lip + deep shadowed hollow interior
    cuff_lip_ring = []
    cuff_deep_ring = []
    tan_w = torso_mat.to_3x3() @ Vector(tangents[-1])
    p_c_world = torso_mat @ Vector(path[-1])
    for iseg in range(n_segs):
        p_edge = rings[-1][iseg].co
        p_lip = p_edge.lerp(p_c_world, 0.22) - tan_w * 0.008
        p_deep = p_edge.lerp(p_c_world, 0.34) - tan_w * 0.055
        cuff_lip_ring.append(bm.verts.new(p_lip))
        cuff_deep_ring.append(bm.verts.new(p_deep))
    vc_end = bm.verts.new(p_c_world - tan_w * 0.068)

    for iseg in range(n_segs):
        iseg_n = (iseg + 1) % n_segs
        f1 = bm.faces.new((rings[-1][iseg], rings[-1][iseg_n], cuff_lip_ring[iseg_n], cuff_lip_ring[iseg]))
        f1.smooth = True
        f1.material_index = 1
        f2 = bm.faces.new((cuff_lip_ring[iseg], cuff_lip_ring[iseg_n], cuff_deep_ring[iseg_n], cuff_deep_ring[iseg]))
        f2.smooth = True
        f2.material_index = 2
        f3 = bm.faces.new((cuff_deep_ring[iseg], cuff_deep_ring[iseg_n], vc_end))
        f3.smooth = True
        f3.material_index = 2

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(f"Sweater_Sleeve_{side_name}_Mesh")
    bm.to_mesh(me)
    bm.free()

    obj = bpy.data.objects.new(f"Sweater_Sleeve_{side_name}", me)
    obj.data.materials.append(mats["sweater"])
    obj.data.materials.append(mats["sweater_rib"])
    obj.data.materials.append(mats["sweater_cuff_interior"])
    return obj


def _add_knit_modifiers(obj, knit_height_img, disp_strength=0.0022, uv_repeat=(4.6, 3.6)):
    """Add Subsurf (level 2) + subtle 3D knit-stitch Displace modifier for real silhouette texture."""
    sub = obj.modifiers.new(name="Subsurf", type='SUBSURF')
    sub.levels = 2
    sub.render_levels = 2

    tex_name = f"DispTex_{obj.name}"
    if tex_name in bpy.data.textures:
        bpy.data.textures.remove(bpy.data.textures[tex_name])
    tex = bpy.data.textures.new(name=tex_name, type='IMAGE')
    tex.image = knit_height_img
    tex.repeat_x = max(1, int(round(uv_repeat[0])))
    tex.repeat_y = max(1, int(round(uv_repeat[1])))

    disp = obj.modifiers.new(name="KnitDisplace", type='DISPLACE')
    disp.texture = tex
    disp.texture_coords = 'UV'
    disp.strength = disp_strength
    disp.mid_level = 0.5


def _build_wool_fuzz_curves(sweater_objs, mat_fuzz, collection, n_fibers=4200):
    """
    Scatter delicate short translucent wool halo micro-fibers across the sweater surface
    for authentic fuzzy mohair/wool sunlit softness along shoulders and sleeves.
    """
    rng = np.random.default_rng(777)
    all_pts = []
    all_norms = []
    for obj in sweater_objs:
        me = obj.data
        verts = np.array([obj.matrix_world @ v.co for v in me.vertices], dtype=np.float64)
        norms = np.array([(obj.matrix_world.to_3x3() @ v.normal).normalized() for v in me.vertices], dtype=np.float64)
        # Keep only outward-facing upper/visible sweater vertices, excluding recessed inner cuff cavities
        keep = (verts[:, 2] > 1.03) & (verts[:, 1] > -0.355)
        all_pts.append(verts[keep])
        all_norms.append(norms[keep])

    pts = np.vstack(all_pts)
    norms = np.vstack(all_norms)
    idx = rng.integers(0, len(pts), size=n_fibers)
    roots = pts[idx]
    ns = norms[idx]

    n_seg = 4
    hd = bpy.data.hair_curves.new("Sweater_WoolFuzz_Curves")
    hd.add_curves([n_seg] * n_fibers)

    pos_flat = np.empty((n_fibers, n_seg, 3), dtype=np.float32)
    rad_flat = np.empty((n_fibers, n_seg), dtype=np.float32)

    lengths = rng.uniform(0.0035, 0.0095, size=(n_fibers, 1))
    rand_dir = rng.normal(0.0, 1.0, size=(n_fibers, 3))
    rand_dir /= (np.linalg.norm(rand_dir, axis=1, keepdims=True) + 1e-8)
    curl_dir = rng.normal(0.0, 1.0, size=(n_fibers, 3))
    curl_dir /= (np.linalg.norm(curl_dir, axis=1, keepdims=True) + 1e-8)

    base_dir = ns * 0.65 + rand_dir * 0.35
    base_dir /= (np.linalg.norm(base_dir, axis=1, keepdims=True) + 1e-8)

    for k in range(n_seg):
        t = k / (n_seg - 1)
        p = roots + base_dir * (lengths * t) + curl_dir * (lengths * 0.28 * t * t)
        pos_flat[:, k, :] = p
        rad_flat[:, k] = 0.00016 * (1.0 - 0.65 * t)

    hd.attributes["position"].data.foreach_set("vector", pos_flat.ravel())
    rad_attr = hd.attributes.new("radius", 'FLOAT', 'POINT')
    rad_attr.data.foreach_set("value", rad_flat.ravel())

    obj = bpy.data.objects.new("Sweater_WoolFuzz", hd)
    if mat_fuzz:
        obj.data.materials.append(mat_fuzz)
    collection.objects.link(obj)
    return obj


def smoothstep_scalar(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0 + 1e-12)))
    return t * t * (3.0 - 2.0 * t)


def build_sweater(collection, mats, knit_imgs):
    """Construct complete dusty-pink chunky knitted crew-neck sweater with wool halo fuzz."""
    torso_mat = get_torso_matrix()

    torso_obj = _build_torso_mesh(mats, torso_mat)
    collar_obj = _build_collar_mesh(mats, torso_mat)
    sleeve_r_obj = _build_sleeve_mesh("R", -1.0, mats, torso_mat)
    sleeve_l_obj = _build_sleeve_mesh("L", 1.0, mats, torso_mat)

    for obj in (torso_obj, collar_obj, sleeve_r_obj, sleeve_l_obj):
        collection.objects.link(obj)

    # Real 3D knit stitch surface displacement + Catmull-Clark smoothing
    _add_knit_modifiers(torso_obj, knit_imgs["knit_height"], disp_strength=0.0020, uv_repeat=(5, 4))
    _add_knit_modifiers(sleeve_r_obj, knit_imgs["knit_height"], disp_strength=0.0022, uv_repeat=(5, 4))
    _add_knit_modifiers(sleeve_l_obj, knit_imgs["knit_height"], disp_strength=0.0022, uv_repeat=(5, 4))
    _add_knit_modifiers(collar_obj, knit_imgs["knit_rib_height"], disp_strength=0.0018, uv_repeat=(3, 1))

    fuzz_obj = _build_wool_fuzz_curves(
        [torso_obj, collar_obj, sleeve_r_obj, sleeve_l_obj],
        mats.get("sweater_fuzz"),
        collection,
        n_fibers=4200,
    )

    return {
        "torso": torso_obj,
        "collar": collar_obj,
        "sleeve_r": sleeve_r_obj,
        "sleeve_l": sleeve_l_obj,
        "fuzz": fuzz_obj,
        "all": [torso_obj, collar_obj, sleeve_r_obj, sleeve_l_obj, fuzz_obj],
    }
