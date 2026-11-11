import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from lib_anatomy import pose_point


def _create_knit_displacement_texture(height_img):
    tex_name = "Tex_Sweater_Knit_Displace"
    if tex_name in bpy.data.textures:
        bpy.data.textures.remove(bpy.data.textures[tex_name])
    tex = bpy.data.textures.new(tex_name, type='IMAGE')
    tex.image = height_img
    tex.extension = 'REPEAT'
    return tex


def _apply_sweater_modifiers(obj, disp_tex, uv_repeat_u, uv_repeat_v, disp_strength=0.0020, subsurf_levels=1):
    """Add Subdivision + UV-scaled 3D Knit Stitch Displacement so sweater has genuine 3D knit relief."""
    # Pre-scale a dedicated UV map "DisplaceUV" so Displace modifier tiles identically to the shader!
    mesh = obj.data
    base_uv = mesh.uv_layers["UVMap"]
    disp_uv = mesh.uv_layers.new(name="DisplaceUV")
    for i in range(len(base_uv.data)):
        u, v = base_uv.data[i].uv
        disp_uv.data[i].uv = (u * uv_repeat_u, v * uv_repeat_v)

    sub = obj.modifiers.new(name="SweaterSubsurf", type='SUBSURF')
    sub.levels = subsurf_levels
    sub.render_levels = subsurf_levels
    sub.uv_smooth = 'PRESERVE_BOUNDARIES'

    disp = obj.modifiers.new(name="KnitReliefDisplace", type='DISPLACE')
    disp.texture = disp_tex
    disp.texture_coords = 'UV'
    disp.uv_layer = "DisplaceUV"
    disp.mid_level = 0.45
    disp.strength = disp_strength


def _build_sweater_torso(mat_torso, disp_tex):
    """
    Build the dusty-pink knitted sweater torso from waist hem (Z=0.90) to crew-neck collar seam (Z=1.35),
    including natural feminine chest contour, slightly raised right (-X) shoulder, soft cloth folds,
    and closed turned-under ribbed bottom hem.
    """
    n_u = 88   # circumferential rings
    n_v = 56   # vertical rows from bottom hem tuck to neck opening

    bm = bmesh.new()
    uv_lay = bm.loops.layers.uv.new("UVMap")

    # Neck collar seam target center (matches posed neck at Z~1.362)
    neck_top = pose_point([0.0, -0.022, 1.362])

    verts_grid = []
    for iv in range(n_v + 1):
        t = iv / n_v  # 0 = inner hem tuck, 0.05 = outer bottom hem, 1.0 = crew-neck seam
        if t < 0.05:
            ht = t / 0.05
            z_base = 0.912 - 0.008 * math.sin(math.pi * (1.0 - ht))
            scale_in = 0.82 + 0.18 * ht
        else:
            vt = (t - 0.05) / 0.95
            z_base = 0.912 + vt * (1.368 - 0.912)
            scale_in = 1.0

        row = []
        for iu in range(n_u):
            u = iu / n_u
            # Put u=0 seam at BACK center (+Y) so the entire front chest & front collar (u=0.5) are 100% seamless!
            ang = 2.0 * math.pi * u + math.pi * 0.5
            ca = math.cos(ang)
            sa = math.sin(ang)

            vt = max(0.0, (t - 0.05) / 0.95)

            shoulder_bulge = math.exp(-(((vt - 0.82) / 0.15) ** 2))
            collar_taper = np.clip((vt - 0.82) / 0.18, 0.0, 1.0) ** 1.4
            waist_blouse = 0.009 * math.sin(math.pi * np.clip((vt - 0.04) / 0.24, 0.0, 1.0))

            rx = (0.138 + 0.022 * (vt ** 0.65) + 0.030 * shoulder_bulge + waist_blouse) * (1.0 - 0.60 * collar_taper)
            ry_front = (0.100 + 0.026 * math.exp(-(((vt - 0.62) / 0.25) ** 2)) + waist_blouse) * (1.0 - 0.36 * collar_taper)
            ry_back = (0.092 + 0.016 * math.exp(-(((vt - 0.68) / 0.26) ** 2)) + 0.6 * waist_blouse) * (1.0 - 0.40 * collar_taper)

            ry = ry_front if sa < 0.0 else ry_back
            se = (abs(ca) ** 2.2 + abs(sa) ** 2.2) ** (-1.0 / 2.2) if (abs(ca) + abs(sa)) > 0 else 1.0
            se = 0.72 + 0.28 * min(1.22, se)

            px = rx * ca * se * scale_in
            py = -0.022 + ry * sa * se * scale_in

            if sa < 0.0:
                bust_z = math.exp(-(((vt - 0.62) / 0.17) ** 2))
                bust_x = math.exp(-(((abs(px) - 0.074) / 0.060) ** 2)) + 0.35 * math.exp(-((px / 0.045) ** 2))
                py -= 0.017 * bust_z * bust_x * (-sa)

            drape1 = 0.0038 * math.sin(5.0 * ang + 2.2 * vt) * math.exp(-(((vt - 0.35) / 0.28) ** 2))
            drape2 = 0.0026 * math.cos(9.0 * ang - 3.8 * vt) * (1.0 - collar_taper) * np.clip(vt / 0.15, 0.0, 1.0)
            diag_fold = 0.0034 * math.sin(18.0 * (px * 0.7 + (z_base - 1.15))) * math.exp(-(((vt - 0.66) / 0.22) ** 2)) * max(0.0, -sa)
            radial_fold = (drape1 + drape2 + diag_fold) * scale_in * (1.0 - collar_taper)
            px += ca * radial_fold
            py += sa * radial_fold

            right_lift = 0.018 * np.clip(-px / 0.16, 0.0, 1.0) * np.clip((vt - 0.45) / 0.45, 0.0, 1.0)
            fwd_lean = -0.014 * (vt ** 1.2)
            py += fwd_lean

            neck_dip = -0.024 * max(0.0, -sa) ** 1.3 + 0.014 * max(0.0, sa) ** 1.2
            pz = z_base + right_lift * (1.0 - 0.85 * collar_taper) + neck_dip * (collar_taper ** 1.2)

            if collar_taper > 0.01:
                c_rx, c_ry = 0.074, 0.074
                cx = neck_top[0] + c_rx * ca
                cy = neck_top[1] - 0.008 + c_ry * sa
                cz = neck_top[2] - 0.025 * max(0.0, -sa) ** 1.35 + 0.014 * max(0.0, sa) + 0.004 * ca
                w_col = collar_taper ** 1.35
                px = px * (1.0 - w_col) + cx * w_col
                py = py * (1.0 - w_col) + cy * w_col
                pz = pz * (1.0 - w_col) + cz * w_col

            vert = bm.verts.new((px, py, pz))
            row.append((vert, u, t))
        verts_grid.append(row)

    # Cap inner bottom center at iv=0 so mesh is 100% watertight from below
    bottom_center = bm.verts.new((0.0, -0.022, 0.916))
    for iu in range(n_u):
        iu_n = (iu + 1) % n_u
        f_cap = bm.faces.new((bottom_center, verts_grid[0][iu_n][0], verts_grid[0][iu][0]))
        f_cap.smooth = True
        for lp in f_cap.loops:
            lp[uv_lay].uv = (0.5, 0.0)

    for iv in range(n_v):
        for iu in range(n_u):
            iu_n = (iu + 1) % n_u
            v00, u00, t00 = verts_grid[iv][iu]
            v01, u01, t01 = verts_grid[iv][iu_n]
            v11, u11, t11 = verts_grid[iv + 1][iu_n]
            v10, u10, t10 = verts_grid[iv + 1][iu]
            if iu_n == 0:
                u01 = 1.0
                u11 = 1.0
            f = bm.faces.new((v00, v01, v11, v10))
            f.smooth = True
            for lp, (uu, tt) in zip(f.loops, [(u00, t00), (u01, t01), (u11, t11), (u10, t10)]):
                lp[uv_lay].uv = (uu, tt)

    # Return top collar seam loop points for seamless collar stitching (extract before bm.free()!)
    top_ring = [np.array(verts_grid[n_v][iu][0].co, dtype=np.float64) for iu in range(n_u)]

    mesh = bpy.data.meshes.new("Sweater_Torso_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Sweater_Torso", mesh)
    obj.data.materials.append(mat_torso)
    _apply_sweater_modifiers(obj, disp_tex, uv_repeat_u=4.0, uv_repeat_v=3.2, disp_strength=0.0014, subsurf_levels=1)

    return obj, top_ring


def _build_crew_collar(top_ring, mat_collar, disp_tex):
    """
    Build plush 3.2cm-wide ribbed crew-neck collar stitched directly from `top_ring`
    upward and inward to hug the posed skin neck seamlessly (with u=0 seam at the back +Y!).
    """
    n_u = len(top_ring)
    n_v = 18
    bm = bmesh.new()
    uv_lay = bm.loops.layers.uv.new("UVMap")

    neck_inner = pose_point([0.0, -0.022, 1.378])
    grid = []
    for iv in range(n_v + 1):
        s = iv / n_v
        row = []
        for iu in range(n_u):
            u = iu / n_u
            ang = 2.0 * math.pi * u + math.pi * 0.5
            ca, sa = math.cos(ang), math.sin(ang)

            p_outer = top_ring[iu]
            p_inner = np.array([
                neck_inner[0] + 0.044 * ca,
                neck_inner[1] - 0.005 + 0.046 * sa,
                neck_inner[2] - 0.021 * max(0.0, -sa) ** 1.3 + 0.014 * max(0.0, sa) + 0.005 * ca,
            ], dtype=np.float64)

            if s <= 0.65:
                # Outer ribbed band rising smoothly from 3mm below torso seam up to plush collar rim (watertight overlap!)
                k = s / 0.65
                p_start = p_outer.copy()
                p_start[2] -= 0.0035
                pt = p_start * (1.0 - k) + p_inner * k
                plush = 0.0072 * math.sin(math.pi * (k ** 0.72)) + 0.0015 * (1.0 - k)
                pt[0] += ca * plush
                pt[1] += sa * plush
                pt[2] += 0.0060 * math.sin(math.pi * k)
            else:
                # Rolled over top edge turning 1.2cm downward inside against neck skin (zero gap!)
                k = (s - 0.65) / 0.35
                pt = p_inner.copy()
                pt[0] *= (1.0 - 0.06 * k)
                pt[1] = neck_inner[1] + (pt[1] - neck_inner[1]) * (1.0 - 0.06 * k)
                pt[2] -= 0.014 * k
                plush = 0.0035 * (1.0 - k)
                pt[0] += ca * plush
                pt[1] += sa * plush

            row.append((bm.verts.new(pt), u, s))
        grid.append(row)

    for iv in range(n_v):
        for iu in range(n_u):
            iu_n = (iu + 1) % n_u
            v00, u00, s00 = grid[iv][iu]
            v01, u01, s01 = grid[iv][iu_n]
            v11, u11, s11 = grid[iv + 1][iu_n]
            v10, u10, s10 = grid[iv + 1][iu]
            if iu_n == 0:
                u01 = 1.0
                u11 = 1.0
            f = bm.faces.new((v00, v01, v11, v10))
            f.smooth = True
            for lp, (uu, ss) in zip(f.loops, [(u00, s00), (u01, s01), (u11, s11), (u10, s10)]):
                lp[uv_lay].uv = (uu, ss)

    mesh = bpy.data.meshes.new("Sweater_Collar_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Sweater_Collar", mesh)
    obj.data.materials.append(mat_collar)
    _apply_sweater_modifiers(obj, disp_tex, uv_repeat_u=4.5, uv_repeat_v=0.45, disp_strength=0.0018, subsurf_levels=1)
    return obj


def _catmull_rom_spline(ctrl_pts, n_samples):
    ctrl = np.asarray(ctrl_pts, dtype=np.float64)
    # Pad endpoints
    pts = np.vstack([2 * ctrl[0] - ctrl[1], ctrl, 2 * ctrl[-1] - ctrl[-2]])
    out = []
    segs = len(ctrl) - 1
    for i in range(n_samples):
        u_tot = (i / (n_samples - 1)) * segs
        idx = min(segs - 1, int(math.floor(u_tot)))
        t = u_tot - idx
        p0, p1, p2, p3 = pts[idx], pts[idx + 1], pts[idx + 2], pts[idx + 3]
        t2, t3 = t * t, t * t * t
        val = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out.append(val)
    return np.array(out, dtype=np.float64)


def _build_single_sleeve(side_tag, ctrl_pts, radii_profile, mat_sleeve, disp_tex):
    """
    Build a dropped-shoulder knitted sweater sleeve extending forward toward the selfie camera,
    complete with dropped-shoulder seam ridge, realistic elbow/forearm compression folds,
    and rolled-in plush ribbed cuff.
    """
    n_s = 54
    n_u = 48
    centers = _catmull_rom_spline(ctrl_pts, n_s)

    # Parallel-transport / Bishop frame along the sleeve spline so UVs never twist!
    tangents = np.gradient(centers, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)
    up0 = np.array([0.0, 0.0, 1.0])
    b0 = np.cross(tangents[0], up0)
    b0 /= np.linalg.norm(b0)
    n0 = np.cross(b0, tangents[0])
    normals[0], binormals[0] = n0, b0

    for i in range(1, n_s):
        v = np.cross(tangents[i - 1], tangents[i])
        s_len = np.linalg.norm(v)
        c_ang = np.clip(np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0)
        if s_len > 1e-6:
            axis = v / s_len
            ang = math.atan2(s_len, c_ang)
            R = np.array(Matrix.Rotation(ang, 3, Vector(axis)))
            normals[i] = R @ normals[i - 1]
        else:
            normals[i] = normals[i - 1]
        normals[i] -= np.dot(normals[i], tangents[i]) * tangents[i]
        normals[i] /= np.linalg.norm(normals[i])
        binormals[i] = np.cross(tangents[i], normals[i])

    bm = bmesh.new()
    uv_lay = bm.loops.layers.uv.new("UVMap")
    grid = []

    for iseg in range(n_s):
        t = iseg / (n_s - 1)  # 0 = deep inside shoulder torso, 0.14 = dropped shoulder seam, 0.92 = cuff rim, 1.0 = cuff interior
        c = centers[iseg]
        N = normals[iseg]    # roughly +Z (top of sleeve)
        B = binormals[iseg]  # lateral

        # Interpolate base radius along sleeve
        if t <= 0.92:
            st = t / 0.92
            r_base = radii_profile[0] * (1.0 - st) + radii_profile[1] * st
            # Subtle cozy fullness in mid-forearm before ribbed cuff taper
            r_base += 0.005 * math.sin(math.pi * (st ** 0.85))
            # Dropped-shoulder seam trough + welt at st=0.15
            seam = -0.0032 * math.exp(-(((st - 0.15) / 0.025) ** 2)) + 0.0024 * math.exp(-(((st - 0.18) / 0.028) ** 2))
            # Cuff rib gather starting at st=0.80
            cuff_pinch = -0.006 * np.clip((st - 0.78) / 0.16, 0.0, 1.0)
            r_cur = r_base + seam + cuff_pinch
            z_push = 0.0
        else:
            # Cuff rim rolls inward 2.2cm so opening has plush yarn thickness!
            ct = (t - 0.92) / 0.08
            r_cur = (radii_profile[1] - 0.006) * (1.0 - 0.34 * ct)
            z_push = -0.022 * ct

        row = []
        for iu in range(n_u):
            u = iu / n_u
            ang = 2.0 * math.pi * u
            ca, sa = math.cos(ang), math.sin(ang)

            # Soft gravity sag on underside of sleeve (ca < 0) + accordion knit folds in elbow/forearm!
            sag = 0.006 * max(0.0, -ca) * math.sin(math.pi * min(1.0, t / 0.92))
            fold1 = 0.0042 * math.sin(26.0 * t + 1.4 * math.sin(2.0 * ang)) * math.exp(-(((t - 0.52) / 0.26) ** 2))
            fold2 = 0.0028 * math.cos(42.0 * t + 3.0 * ang) * math.exp(-(((t - 0.68) / 0.22) ** 2))
            r_local = r_cur + fold1 + fold2

            # Slightly oval arm + knit drape cross-section
            pt = c + tangents[iseg] * z_push + N * (r_local * 1.04 * ca - sag) + B * (r_local * 0.96 * sa)
            row.append((bm.verts.new(pt), u, min(1.0, t / 0.92)))
        grid.append(row)

    for iseg in range(n_s - 1):
        for iu in range(n_u):
            iu_n = (iu + 1) % n_u
            v00, u00, t00 = grid[iseg][iu]
            v01, u01, t01 = grid[iseg][iu_n]
            v11, u11, t11 = grid[iseg + 1][iu_n]
            v10, u10, t10 = grid[iseg + 1][iu]
            if iu_n == 0:
                u01 = 1.0
                u11 = 1.0
            f = bm.faces.new((v00, v01, v11, v10))
            f.smooth = True
            for lp, (uu, tt) in zip(f.loops, [(u00, t00), (u01, t01), (u11, t11), (u10, t10)]):
                # Note: along sleeve, knit wales run lengthwise (along tt), so swap (tt, uu) or keep wales along sleeve!
                # Look at reference.png: on the sleeves, the rib columns run LENGTHWISE from shoulder to cuff!
                # In our knit texture, wales run along V (second UV coord), and rib columns repeat along U!
                # So (uu, tt) makes wales run lengthwise along `tt` from shoulder to cuff! Exact match!!
                lp[uv_lay].uv = (uu, tt)

    # Cap deep interior cuff backplate so sleeve interior is dark/closed
    cuff_end_center = bm.verts.new(centers[-1] - tangents[-1] * 0.024)
    for iu in range(n_u):
        iu_n = (iu + 1) % n_u
        f_end = bm.faces.new((grid[-1][iu][0], grid[-1][iu_n][0], cuff_end_center))
        f_end.smooth = True
        for lp in f_end.loops:
            lp[uv_lay].uv = (0.5, 1.0)

    mesh = bpy.data.meshes.new(f"Sweater_Sleeve_{side_tag}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(f"Sweater_Sleeve_{side_tag}", mesh)
    obj.data.materials.append(mat_sleeve)
    _apply_sweater_modifiers(obj, disp_tex, uv_repeat_u=2.2, uv_repeat_v=2.8, disp_strength=0.0022, subsurf_levels=1)

    cuff_frame = (centers[-3], tangents[-3], normals[-3], binormals[-3])
    return obj, cuff_frame


def _build_cuff_hand(side_tag, sign_x, cuff_frame, mat_skin):
    """
    Build graceful, softly curled feminine hand/fingers emerging naturally ~3.5cm from
    the cozy oversized sweater cuff (as if holding the selfie camera/phone just outside
    the beauty frame, so diagnostic turnaround views show clean anatomical hands!).
    """
    c_pos, T, N, B = cuff_frame
    bm = bmesh.new()

    # Palm base tucked inside cuff opening
    palm_c = c_pos + T * 0.008
    # Create smooth palm + 4 softly curled fingers + thumb using subdivided metaball-free quad capsules
    def add_tapered_limb(pts_world, radii):
        segs = len(pts_world)
        ring_n = 12
        tang = np.gradient(pts_world, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        rings = []
        up_ref = N
        for i in range(segs):
            bi = np.cross(tang[i], up_ref)
            if np.linalg.norm(bi) < 1e-4:
                bi = B
            else:
                bi /= np.linalg.norm(bi)
            ni = np.cross(bi, tang[i])
            rr = radii[i]
            row = []
            for k in range(ring_n):
                a = 2.0 * math.pi * k / ring_n
                p = pts_world[i] + ni * (rr * 0.92 * math.cos(a)) + bi * (rr * 1.05 * math.sin(a))
                row.append(bm.verts.new(p))
            rings.append(row)
        for i in range(segs - 1):
            for k in range(ring_n):
                kn = (k + 1) % ring_n
                f = bm.faces.new((rings[i][k], rings[i][kn], rings[i + 1][kn], rings[i + 1][k]))
                f.smooth = True
        # Smooth rounded fingertip cap
        tip_v = bm.verts.new(pts_world[-1] + tang[-1] * (radii[-1] * 0.75))
        base_v = bm.verts.new(pts_world[0] - tang[0] * (radii[0] * 0.5))
        for k in range(ring_n):
            kn = (k + 1) % ring_n
            f1 = bm.faces.new((rings[-1][k], rings[-1][kn], tip_v))
            f2 = bm.faces.new((base_v, rings[0][kn], rings[0][k]))
            f1.smooth = True
            f2.smooth = True

    # Inward vector (toward midline X=0, holding camera edge)
    inward = np.array([-sign_x, 0.0, 0.0])
    inward = inward - np.dot(inward, T) * T
    if np.linalg.norm(inward) > 1e-5:
        inward /= np.linalg.norm(inward)
    else:
        inward = B * (-sign_x)
    dorsal = np.cross(T, inward) * (-sign_x)
    if dorsal[2] < 0:
        dorsal = -dorsal

    # Soft palm pad tucked inside oversized cuff
    palm_pts = np.array([
        palm_c - T * 0.018,
        palm_c - T * 0.004,
        palm_c + T * 0.012,
        palm_c + T * 0.024 + inward * 0.004,
    ])
    add_tapered_limb(palm_pts, [0.026, 0.028, 0.029, 0.025])

    # Four graceful softly-curled fingers peeking out of cozy cuff ("oversized sweater sleeve" style)
    finger_specs = [
        (-0.018, 0.044, 0.0068),  # index
        (-0.005, 0.048, 0.0070),  # middle
        (0.008, 0.044, 0.0066),   # ring
        (0.020, 0.036, 0.0058),   # pinky
    ]
    for lat_off, flen, frad in finger_specs:
        base_f = palm_c + T * 0.018 + dorsal * lat_off * 0.85 - dorsal * 0.002
        f_pts = np.array([
            base_f,
            base_f + T * (flen * 0.38) + inward * (flen * 0.10),
            base_f + T * (flen * 0.70) + inward * (flen * 0.34),
            base_f + T * (flen * 0.88) + inward * (flen * 0.62),
        ])
        add_tapered_limb(f_pts, [frad * 1.08, frad * 0.96, frad * 0.84, frad * 0.68])

    # Soft thumb opposing gently below/inward
    th_base = palm_c + T * 0.006 + inward * 0.018 - N * 0.012
    th_pts = np.array([
        th_base,
        th_base + T * 0.018 + inward * 0.012 + N * 0.004,
        th_base + T * 0.032 + inward * 0.020 + N * 0.010,
    ])
    add_tapered_limb(th_pts, [0.0088, 0.0076, 0.0064])

    mesh = bpy.data.meshes.new(f"Character_Hand_{side_tag}_Mesh")
    bm.to_mesh(mesh)
    bm.free()

    # Add UVMap + rest_co / neutral skin attributes so Mat_Character_Skin shades hands seamlessly!
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for lp in uv_layer.data:
        lp.uv = (0.25, 0.25)

    n = len(mesh.vertices)
    rc = np.tile(np.array([0.04, -0.13, 1.51], dtype=np.float32), (n, 1))
    attr_rc = mesh.attributes.new(name="rest_co", type='FLOAT_VECTOR', domain='POINT')
    attr_rc.data.foreach_set("vector", rc.ravel())
    for aname, val in [("lip_mask", 0.0), ("blush_mask", 0.14), ("cavity_ao", 0.0), ("scalp_mask", 0.0), ("lashline_mask", 0.0), ("brow_mask", 0.0)]:
        at = mesh.attributes.new(name=aname, type='FLOAT', domain='POINT')
        at.data.foreach_set("value", np.full(n, val, dtype=np.float32))

    obj = bpy.data.objects.new(f"Character_Hand_{side_tag}", mesh)
    obj.data.materials.append(mat_skin)
    sub = obj.modifiers.new(name="HandSubsurf", type='SUBSURF')
    sub.levels = 1
    sub.render_levels = 1
    return obj


def _build_sweater_fuzz_curves(sweater_objs, mat_fuzz):
    """Scatter delicate 2.5mm pastel-pink wool halo fibers across sweater shoulders & sleeves."""
    rng = np.random.default_rng(515)
    pts_list = []
    rad_list = []

    for ob in sweater_objs:
        mesh = ob.data
        nv = len(mesh.vertices)
        co = np.empty((nv, 3), dtype=np.float64)
        no = np.empty((nv, 3), dtype=np.float64)
        mesh.vertices.foreach_get("co", co.ravel())
        mesh.vertices.foreach_get("normal", no.ravel())

        count = 950 if "Sleeve" in ob.name else (900 if "Torso" in ob.name else 250)
        idxs = rng.integers(0, nv, size=count)
        for idx in idxs:
            p0 = co[idx]
            nrm = no[idx]
            nlen = np.linalg.norm(nrm)
            if nlen < 1e-5:
                continue
            nrm = nrm / nlen
            tangent_jit = rng.normal(0.0, 1.0, size=3)
            tangent_jit /= max(1e-5, np.linalg.norm(tangent_jit))
            flen = rng.uniform(0.0018, 0.0038)
            p1 = p0 + nrm * (flen * 0.55) + tangent_jit * (flen * 0.25)
            p2 = p0 + nrm * (flen * 0.85) + tangent_jit * (flen * 0.65)
            pts_list.append(np.array([p0, p1, p2], dtype=np.float64))
            r0 = rng.uniform(0.000045, 0.000070)
            rad_list.append(np.array([r0, r0 * 0.65, r0 * 0.15], dtype=np.float64))

    hc = bpy.data.hair_curves.new("Sweater_Fuzz_Data")
    hc.add_curves([3] * len(pts_list))
    hc.position_data.foreach_set("vector", np.concatenate(pts_list, axis=0).astype(np.float32).ravel())
    hc.points.foreach_set("radius", np.concatenate(rad_list, axis=0).astype(np.float32).ravel())
    hc.set_types(type='CATMULL_ROM')
    hc.materials.append(mat_fuzz)
    return bpy.data.objects.new("Sweater_Wool_Fuzz", hc)


def build_character_sweater(char_col, textures, mat_skin, mat_fuzz):
    from lib_materials import build_sweater_materials

    disp_tex = _create_knit_displacement_texture(textures["knit_height"])
    mat_torso = build_sweater_materials(
        textures, uv_scale=(11.5, 11.5, 6.8), name="Mat_Sweater_Torso_Knit", use_box_projection=True
    )
    mat_collar = build_sweater_materials(textures, uv_scale=(4.5, 0.45, 1.0), name="Mat_Sweater_Collar_Knit")
    mat_sleeve = build_sweater_materials(textures, uv_scale=(2.2, 2.8, 1.0), name="Mat_Sweater_Sleeve_Knit")

    torso_obj, top_ring = _build_sweater_torso(mat_torso, disp_tex)
    collar_obj = _build_crew_collar(top_ring, mat_collar, disp_tex)

    # Right Sleeve (-X, viewer's left in beauty.png): natural raised shoulder -> gentle elbow dip -> forward selfie forearm!
    ctrl_R = [
        [-0.120, -0.028, 1.300],  # rooted safely inside upper right shoulder
        [-0.174, -0.068, 1.308],  # raised right dropped-shoulder seam
        [-0.214, -0.215, 1.215],  # natural downward elbow bend!
        [-0.228, -0.365, 1.245],  # forearm rising softly toward selfie camera
        [-0.232, -0.495, 1.288],  # cozy oversized cuff framing lower-left of beauty.png
    ]
    sleeve_R, cuff_R = _build_single_sleeve("R", ctrl_R, (0.055, 0.044), mat_sleeve, disp_tex)

    # Left Sleeve (+X, viewer's right in beauty.png): relaxed shoulder -> gentle elbow dip -> forward selfie forearm!
    ctrl_L = [
        [0.120, -0.028, 1.278],
        [0.176, -0.065, 1.278],
        [0.218, -0.205, 1.178],
        [0.232, -0.355, 1.204],
        [0.236, -0.485, 1.242],
    ]
    sleeve_L, cuff_L = _build_single_sleeve("L", ctrl_L, (0.054, 0.043), mat_sleeve, disp_tex)

    hand_R = _build_cuff_hand("R", -1.0, cuff_R, mat_skin)
    hand_L = _build_cuff_hand("L", +1.0, cuff_L, mat_skin)

    fuzz_obj = _build_sweater_fuzz_curves([torso_obj, collar_obj, sleeve_R, sleeve_L], mat_fuzz)

    objs = [torso_obj, collar_obj, sleeve_R, sleeve_L, hand_R, hand_L, fuzz_obj]
    for ob in objs:
        char_col.objects.link(ob)

    return {
        "torso": torso_obj,
        "collar": collar_obj,
        "sleeve_R": sleeve_R,
        "sleeve_L": sleeve_L,
        "hands": [hand_R, hand_L],
        "fuzz": fuzz_obj,
        "all_sweater_meshes": [torso_obj, collar_obj, sleeve_R, sleeve_L],
    }
