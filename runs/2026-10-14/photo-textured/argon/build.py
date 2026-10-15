#!/usr/bin/env python3
"""
Reproducible Blender 5.1.1 build script for the 3D upper-body portrait.
Run with:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
"""

import os
import sys
import json
import math
import time
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler
from mathutils.kdtree import KDTree

WORKDIR = os.path.dirname(os.path.abspath(__file__))
PREVIEW_MODE = os.environ.get("PREVIEW", "0") == "1" or ("--preview" in sys.argv)


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)


def parse_obj_with_uvs():
    """
    Loads input/upper-body.obj and matches its vertices/faces to input/base.obj
    to recover exact UV coordinates while strictly keeping only upper-body geometry.
    """
    base_v = []
    base_vt = []
    base_face_uvs = {}
    with open(os.path.join(WORKDIR, "input", "base.obj"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                sx, sy, sz = [float(x) for x in line.split()[1:4]]
                base_v.append((sx * 0.1, -sz * 0.1, (sy + 8.168) * 0.1))
            elif line.startswith("vt "):
                u, v = [float(x) for x in line.split()[1:3]]
                base_vt.append((u, v))
            elif line.startswith("f "):
                toks = line.split()[1:]
                fv = tuple(int(t.split("/")[0]) - 1 for t in toks)
                fvt = tuple(int(t.split("/")[1]) - 1 for t in toks)
                base_face_uvs[fv] = fvt

    kd = KDTree(len(base_v))
    for i, co in enumerate(base_v):
        kd.insert(co, i)
    kd.balance()

    ub_v = []
    ub_f = []
    with open(os.path.join(WORKDIR, "input", "upper-body.obj"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                ub_v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                ub_f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])

    ub_to_base = [kd.find(co)[1] for co in ub_v]
    ub_f_uvs = []
    for fv in ub_f:
        bf = tuple(ub_to_base[vi] for vi in fv)
        fvt = base_face_uvs.get(bf, None)
        if fvt is not None:
            ub_f_uvs.append([base_vt[ti] for ti in fvt])
        else:
            ub_f_uvs.append([(0.5, 0.5) for _ in fv])

    return np.array(ub_v, dtype=np.float64), ub_f, ub_f_uvs


def sculpt_face_canonical(verts):
    """
    Sculpts the canonical MakeHuman hm08 head/face vertices to match the young,
    softly smiling feminine appearance in input/reference.png:
    - Softer, rounder lower face and delicate jawline/chin
    - Full, youthful apple cheeks
    - Smaller, integrated nose with narrower alae and defined nostrils
    - Almond-shaped eyes wrapping 3D eyeballs, subtle double eyelid + aegyo-sal
    - Closed-mouth gentle smile with shaped Cupid's bow and lifted mouth corners
    """
    v = verts.copy()
    x = v[:, 0].copy()
    y = v[:, 1].copy()
    z = v[:, 2].copy()
    ax = np.abs(x)
    sx = np.sign(x)

    # 1. Overall feminine head proportions: slightly narrower jaw, softer temples, compact lower face
    lower_compact = smoothstep(1.515, 1.425, z)
    v[:, 2] += 0.0055 * lower_compact
    jaw_zone = smoothstep(1.42, 1.470, z) * (1.0 - smoothstep(1.498, 1.545, z))
    v[:, 0] *= 1.0 - 0.135 * jaw_zone * smoothstep(0.024, 0.065, ax)

    # Slightly shorten and round the chin tip (z in 1.43..1.468, y < -0.12)
    chin_zone = gauss(z, 1.444, 0.015) * gauss(ax, 0.0, 0.032) * smoothstep(-0.10, -0.135, y)
    v[:, 2] += 0.0052 * chin_zone
    v[:, 1] += 0.0024 * chin_zone
    v[:, 0] *= 1.0 - 0.12 * chin_zone

    # Youthful cheek fullness (apple cheeks / malar pad lifted by soft smile)
    cheek_zone = (
        gauss(ax, 0.038, 0.018)
        * gauss(z, 1.516, 0.020)
        * smoothstep(-0.08, -0.125, y)
    )
    v[:, 1] -= 0.0044 * cheek_zone
    v[:, 2] += 0.0024 * cheek_zone
    v[:, 0] += sx * 0.0016 * cheek_zone

    # Smooth transition along lower cheek / buccal region
    buccal_zone = (
        gauss(ax, 0.045, 0.016)
        * gauss(z, 1.482, 0.020)
        * smoothstep(-0.07, -0.12, y)
    )
    v[:, 1] -= 0.0030 * buccal_zone
    v[:, 0] += sx * 0.0018 * buccal_zone

    # Soften brow ridge projection for a smoother feminine forehead/glabella
    brow_ridge = (
        gauss(z, 1.558, 0.012)
        * gauss(ax, 0.024, 0.026)
        * smoothstep(-0.12, -0.145, y)
    )
    v[:, 1] += 0.0018 * brow_ridge
    forehead_dome = (
        gauss(z, 1.595, 0.024)
        * gauss(ax, 0.0, 0.048)
        * smoothstep(-0.10, -0.14, y)
    )
    v[:, 1] -= 0.0024 * forehead_dome

    # 2. Small, delicate, integrated nose with defined nostrils
    bridge_zone = gauss(z, 1.528, 0.014) * gauss(ax, 0.0, 0.020) * smoothstep(-0.135, -0.150, y)
    v[:, 0] *= 1.0 - 0.24 * bridge_zone
    v[:, 1] += 0.0014 * bridge_zone

    alae_zone = gauss(z, 1.506, 0.009) * gauss(ax, 0.016, 0.012) * smoothstep(-0.138, -0.150, y)
    v[:, 0] -= sx * 0.0050 * alae_zone
    v[:, 2] += 0.0012 * alae_zone

    tip_zone = gauss(z, 1.508, 0.007) * gauss(ax, 0.0, 0.010) * smoothstep(-0.152, -0.162, y)
    v[:, 0] *= 1.0 - 0.25 * tip_zone
    v[:, 2] += 0.0015 * tip_zone
    v[:, 1] += 0.0020 * tip_zone

    nostril_cav = (
        gauss(ax, 0.0075, 0.0030)
        * gauss(z, 1.5028, 0.0028)
        * gauss(y, -0.1575, 0.0040)
    )
    v[:, 2] += 0.0028 * nostril_cav
    v[:, 1] += 0.0018 * nostril_cav

    # 3. Eyes & Eyelids: relaxed, softly smiling almond shape with gentle outer downturn & zero upper scleral show
    lid_surface = smoothstep(-0.129, -0.135, y)
    for side in [-1.0, 1.0]:
        ex = side * 0.030775
        ey = -0.124535
        ez = 1.545215

        dx = x - ex
        dy = y - ey
        dz = z - ez

        u_lid = (
            gauss(dx, -side * 0.0010, 0.0095)
            * gauss(z, 1.5502, 0.0034)
            * lid_surface
        )
        # Gently lower the upper eyelid margin (-2.2mm) to form a warm, relaxed, smiling almond eye
        # that covers the top 25% of the iris with zero upper scleral show
        outer_slant = smoothstep(0.026, 0.043, ax)
        v[:, 2] -= (0.0022 + 0.0010 * outer_slant) * u_lid
        v[:, 1] -= 0.0015 * u_lid

        l_lid = (
            gauss(dx, 0.0, 0.0092)
            * gauss(z, 1.5412, 0.0028)
            * lid_surface
        )
        # Gently lifted smiling lower lid margin (+0.7mm)
        v[:, 2] += 0.0007 * l_lid
        v[:, 1] -= 0.0011 * l_lid

        outer_canthus = (
            gauss(x, side * 0.0435, 0.0048)
            * gauss(z, 1.5450, 0.0045)
            * lid_surface
        )
        # Soft gentle downturn at outer canthus matching reference.png
        v[:, 2] -= 0.0006 * outer_canthus
        v[:, 0] += side * 0.0009 * outer_canthus

        aegyo = (
            gauss(dx, 0.0, 0.0098)
            * gauss(z, 1.5368, 0.0023)
            * lid_surface
        )
        v[:, 1] -= 0.0015 * aegyo

        socket_back = (
            (np.abs(dx) < 0.016)
            & (np.abs(dz) < 0.014)
            & (y > -0.1335)
            & (y < -0.105)
        )
        if np.any(socket_back):
            v[socket_back, 1] += 0.0050

        ec = np.array([ex, -0.1282, 1.5462])
        rel = v - ec
        dist = np.linalg.norm(rel, axis=1)
        rim_mask = (dist < 0.01355) & (v[:, 1] <= -0.132)
        if np.any(rim_mask):
            dirs = rel[rim_mask] / (dist[rim_mask, None] + 1e-12)
            v[rim_mask] = ec + dirs * 0.01355

    # 4. Shaped lips & closed-mouth smile
    y_oral_thresh = -0.1518 + 16.0 * (x ** 2)
    outer_lip_mask = smoothstep(y_oral_thresh + 0.0025, y_oral_thresh - 0.0010, y)
    inner_bag_mask = (1.0 - outer_lip_mask) * gauss(z, 1.480, 0.014) * (ax < 0.032)

    v[:, 1] += 0.0050 * inner_bag_mask

    lower_lip_seal = (
        outer_lip_mask
        * gauss(z, 1.4762, 0.0030)
        * gauss(ax, 0.0, 0.018)
    )
    v[:, 1] -= 0.0028 * lower_lip_seal
    v[:, 2] += 0.0012 * lower_lip_seal

    lower_lip_plush = (
        outer_lip_mask
        * gauss(z, 1.4742, 0.0028)
        * gauss(ax, 0.0, 0.014)
    )
    v[:, 1] -= 0.0016 * lower_lip_plush

    cupid_peaks = (
        outer_lip_mask
        * gauss(z, 1.4855, 0.0022)
        * gauss(ax, 0.0048, 0.0028)
    )
    v[:, 2] += 0.0009 * cupid_peaks
    v[:, 1] -= 0.0010 * cupid_peaks

    # Upper lip slightly deeper rose, lower lip plush warm pink
    upper_tubercle = (
        outer_lip_mask
        * gauss(z, 1.4815, 0.0024)
        * gauss(ax, 0.0, 0.012)
    )
    v[:, 1] -= 0.0014 * upper_tubercle
    v[:, 2] -= 0.0005 * upper_tubercle

    smile_curve = (
        gauss(z, 1.4795, 0.0095)
        * smoothstep(0.006, 0.025, ax)
        * (1.0 - smoothstep(0.030, 0.045, ax))
        * smoothstep(-0.125, -0.139, y)
    )
    v[:, 2] += 0.0042 * smile_curve
    v[:, 0] += sx * 0.0018 * smile_curve
    v[:, 1] -= 0.0010 * smile_curve

    smile_fold = (
        gauss(ax, 0.0285, 0.0055)
        * gauss(z, 1.486, 0.010)
        * smoothstep(-0.125, -0.138, y)
    )
    v[:, 1] -= 0.0012 * smile_fold
    v[:, 2] += 0.0018 * smile_fold

    # Re-apply eye socket clearance around the flush eyeball center (y = -0.1282, r = 0.0132)
    for side in [-1.0, 1.0]:
        ex = side * 0.030775
        ey = -0.1282
        ez = 1.545215
        dx = v[:, 0] - ex
        dz = v[:, 2] - ez
        # Push internal eye-socket back wall behind the eyeball so it never clips or shadows the iris
        inner_wall = (
            (np.abs(dx) < 0.0145)
            & (np.abs(dz) < 0.0105)
            & (v[:, 1] > -0.1385)
            & (v[:, 1] < -0.110)
        )
        if np.any(inner_wall):
            v[inner_wall, 1] = np.maximum(v[inner_wall, 1], -0.1220)

        # Ensure outer eyelid rim wraps cleanly around the r=0.0132 eyeball/cornea surface
        ec = np.array([ex, ey, ez])
        rel = v - ec
        dist = np.linalg.norm(rel, axis=1)
        rim_mask = (dist < 0.01345) & (v[:, 1] <= -0.134)
        if np.any(rim_mask):
            dirs = rel[rim_mask] / (dist[rim_mask, None] + 1e-12)
            v[rim_mask] = ec + dirs * 0.01345

    return v


def get_head_pose_matrix():
    theta_roll = math.radians(21.5)
    pitch_head = math.radians(3.2)
    yaw_head = math.radians(-3.2)
    cos_r, sin_r = math.cos(theta_roll), math.sin(theta_roll)
    rot_y = np.array([
        [cos_r, 0.0, -sin_r],
        [0.0,   1.0,  0.0],
        [sin_r, 0.0,  cos_r],
    ], dtype=np.float64)
    rot_xz = np.array(Euler((pitch_head, 0.0, yaw_head), "XYZ").to_matrix(), dtype=np.float64)
    return np.array([0.0, -0.016, 1.385], dtype=np.float64), rot_y @ rot_xz


def pose_point_array(pts, include_arms=True):
    """
    Applies the unified 3D character pose to any array of canonical 3D points (N, 3):
    1. Both arms extending forward (-Y) and slightly inward toward the selfie camera (when include_arms=True).
    2. Gentle head & neck tilt around neck pivot.
    """
    p = pts.copy()

    # A. Natural selfie arm pose from 45-deg A-pose: adduct arms downward and pitch slightly forward
    # so sleeves flow smoothly past the bottom edge of the beauty frame and terminate at the bust base (z~0.96m)
    if include_arms:
        for side in [-1.0, 1.0]:
            sh_pivot = np.array([side * 0.152, -0.018, 1.328])
            sx_val = side * p[:, 0]
            w_arm = smoothstep(0.125, 0.215, sx_val) * (1.0 - smoothstep(1.355, 1.400, p[:, 2]))
            if np.any(w_arm > 1e-5):
                pitch_ang = math.radians(-9.5)
                roll_ang = math.radians(side * 22.0)
                yaw_ang = math.radians(-side * 8.5)
                rot_mat = np.array(
                    Euler((pitch_ang, roll_ang, yaw_ang), "XYZ").to_matrix(),
                    dtype=np.float64,
                )
                rel = p - sh_pivot
                rot_rel = rel @ rot_mat.T
                p = p + w_arm[:, None] * (sh_pivot + rot_rel - p)

    # B. Gentle head & neck tilt
    neck_pivot, head_mat = get_head_pose_matrix()
    w_head = smoothstep(1.342, 1.435, pts[:, 2]) * (1.0 - smoothstep(0.11, 0.16, np.abs(pts[:, 0])))
    rel_h = p - neck_pivot
    rot_h = rel_h @ head_mat.T
    p = p + w_head[:, None] * (neck_pivot + rot_h - p)

    return p


def pose_hair_point_array(pts):
    """
    Applies the gentle head & neck tilt to canonical hair coordinates without applying
    forearm rotation, preventing shoulder-blade hair from pitching into the back of the sweater.
    """
    return pose_point_array(pts, include_arms=False)


def pose_head_rigid_point(pt):
    neck_pivot, head_mat = get_head_pose_matrix()
    arr = np.array(pt, dtype=np.float64)
    return neck_pivot + (arr - neck_pivot) @ head_mat.T, head_mat


def compute_face_cyl_uv(cx, cy, cz):
    """
    Computes distortion-free cylindrical UV coordinates for the canonical head/neck
    around the central vertical axis (x=0, y=-0.055), with seam at the back of the head (+Y).
    """
    theta = np.arctan2(cx, -(cy - (-0.055)))  # 0 at front (-Y), +pi/2 at left (+X), -pi/2 at right (-X)
    u = 0.5 + theta / (2.0 * math.pi)
    v = np.clip((cz - 1.320) / (1.680 - 1.320), 0.0, 1.0)
    return u, v


def build_head_and_sweater_meshes(char_col, mat_skin, mat_sweater, mat_collar, knit_disp_tex):
    """
    Constructs CharacterHead, SweaterBody, SweaterCollar, and CharacterHands
    from input/upper-body.obj with exact UVs and sculpted 3D detail.
    """
    ub_v, ub_f, ub_f_uvs = parse_obj_with_uvs()
    sculpted_v = sculpt_face_canonical(ub_v)

    # --- 1. Build CharacterHead (head, face, ears, and neck down to z=1.335) ---
    head_faces = []
    head_uvs = []
    for fi, fv in enumerate(ub_f):
        fc = sculpted_v[fv].mean(axis=0)
        if abs(fc[0]) < 0.135 and fc[2] > 1.335:
            head_faces.append(fv)
            head_uvs.append(ub_f_uvs[fi])

    used_v = sorted(list(set(vi for fv in head_faces for vi in fv)))
    v_remap = {old_i: new_i for new_i, old_i in enumerate(used_v)}
    head_canon_v = sculpted_v[used_v]
    head_posed_v = pose_point_array(head_canon_v)
    head_f_remapped = [[v_remap[vi] for vi in fv] for fv in head_faces]

    me_head = bpy.data.meshes.new("CharacterHeadMesh")
    me_head.from_pydata(head_posed_v.tolist(), [], head_f_remapped)
    me_head.update()

    # Add FaceCylUV (primary UV for seamless procedural skin map) and BaseUV
    uv_cyl = me_head.uv_layers.new(name="FaceCylUV")
    me_head.uv_layers.active = uv_cyl
    me_head.uv_layers.active_index = 0
    uv_base = me_head.uv_layers.new(name="BaseUV")

    u_arr, v_arr = compute_face_cyl_uv(head_canon_v[:, 0], head_canon_v[:, 1], head_canon_v[:, 2])
    for poly_i, poly in enumerate(me_head.polygons):
        f_uv = head_uvs[poly_i]
        for loop_k, loop_idx in enumerate(poly.loop_indices):
            vi = me_head.loops[loop_idx].vertex_index
            uv_cyl.data[loop_idx].uv = (float(u_arr[vi]), float(v_arr[vi]))
            uv_base.data[loop_idx].uv = f_uv[loop_k]

    # Store canonical position attribute for shader nodes
    attr_canon = me_head.attributes.new(name="canon_pos", type="FLOAT_VECTOR", domain="POINT")
    attr_canon.data.foreach_set("vector", head_canon_v.astype(np.float32).ravel())

    for p in me_head.polygons:
        p.use_smooth = True

    obj_head = bpy.data.objects.new("CharacterHead", me_head)
    char_col.objects.link(obj_head)
    obj_head.data.materials.append(mat_skin)

    sub_head = obj_head.modifiers.new(name="Subdivision", type="SUBSURF")
    sub_head.levels = 1
    sub_head.render_levels = 2
    sub_head.uv_smooth = "PRESERVE_BOUNDARIES"

    # --- 2. Build SweaterBody (crew-neck knitted sweater with extruded waist & sleeves) ---
    bm_sw = bmesh.new()
    for co in sculpted_v:
        bm_sw.verts.new(co.tolist())
    bm_sw.verts.ensure_lookup_table()

    for fv in ub_f:
        fc = sculpted_v[fv].mean(axis=0)
        # Keep faces below the crew-neck collar line: z_collar = 1.372 + 0.22*y + 1.8*x^2
        if not (abs(fc[0]) < 0.15 and fc[2] > (1.372 + 0.22 * fc[1] + 1.8 * (fc[0] ** 2))):
            try:
                bm_sw.faces.new([bm_sw.verts[vi] for vi in fv])
            except ValueError:
                pass

    # Delete any loose vertices (e.g. head vertices above the collar line)
    loose_v = [v for v in bm_sw.verts if not v.link_faces]
    if loose_v:
        bmesh.ops.delete(bm_sw, geom=loose_v, context="VERTS")

    bmesh.ops.remove_doubles(bm_sw, verts=bm_sw.verts, dist=1e-5)
    bm_sw.verts.ensure_lookup_table()
    bm_sw.edges.ensure_lookup_table()

    # Find the 4 boundary loops: collar (z~1.376), waist (z~1.11, x~0), left sleeve (x~+0.37), right sleeve (x~-0.37)
    def find_boundary_loops(bm):
        b_edges = [e for e in bm.edges if e.is_boundary]
        visited = set()
        loops = []
        for e in b_edges:
            if e.index in visited:
                continue
            comp_e = []
            comp_v = []
            q = [e]
            visited.add(e.index)
            while q:
                cur = q.pop()
                comp_e.append(cur)
                for v in cur.verts:
                    if v not in comp_v:
                        comp_v.append(v)
                    for ne in v.link_edges:
                        if ne.is_boundary and ne.index not in visited:
                            visited.add(ne.index)
                            q.append(ne)
            vc = np.array([v.co[:] for v in comp_v])
            loops.append((vc.mean(axis=0), comp_e, comp_v))
        return loops

    loops = find_boundary_loops(bm_sw)

    for mean_co, comp_e, comp_v in loops:
        mx, my, mz = mean_co
        if mz > 1.30:
            # Collar loop: smooth stair-stepped polygon cut onto exact collar curve tucked under SweaterCollar
            for v in comp_v:
                z_col = 1.366 + 0.20 * v.co.y + 1.55 * (v.co.x ** 2)
                v.co.z = 0.25 * v.co.z + 0.75 * z_col
        elif abs(mx) < 0.10 and mz < 1.20:
            # Waist loop: first level the stair-stepped polygon cut at z=1.106, then extrude 4 smooth rings
            # downward to z=0.962 plus 1 inward-rolled ribbed hem ring so bottom silhouette is 100% smooth
            for v in comp_v:
                v.co.z = 0.30 * v.co.z + 0.70 * 1.106
            cur_edges = comp_e
            for step in range(5):
                ret = bmesh.ops.extrude_edge_only(bm_sw, edges=cur_edges)
                geom = ret["geom"]
                new_verts = [ele for ele in geom if isinstance(ele, bmesh.types.BMVert)]
                cur_edges = [ele for ele in geom if isinstance(ele, bmesh.types.BMEdge)]
                if step < 4:
                    t_w = (step + 1) / 4.0
                    z_target = 1.106 - (step + 1) * 0.036
                    flare = 1.006 if step < 2 else 0.994
                    for v in new_verts:
                        v.co.z = (v.co.z - 0.036) * (1.0 - t_w) + z_target * t_w
                        v.co.x *= flare
                        v.co.y *= flare
                else:
                    # Subtle inward-rolled bottom hem edge
                    for v in new_verts:
                        v.co.z = 0.958
                        v.co.x *= 0.965
                        v.co.y = -0.035 + (v.co.y - (-0.035)) * 0.965
        elif abs(mx) > 0.25:
            # Left or right forearm sleeve loop: extend downward along natural forearm trajectory
            # to the bust base plane (z~0.955m) and form a crisp, open, double-creased ribbed knit cuff
            # with a deep dark interior sleeve tunnel (completely below the bottom frame of beauty.png)
            side = 1.0 if mx > 0 else -1.0
            elbow = np.array([side * 0.3129, -0.0132, 1.1661])
            hand = np.array([side * 0.3800, -0.1150, 0.9450])
            f_dir = hand - elbow
            f_dir = f_dir / np.linalg.norm(f_dir)
            up_v = np.array([0.0, 0.0, 1.0])
            side_v = np.cross(f_dir, up_v)
            side_v /= np.linalg.norm(side_v)
            up_v = np.cross(side_v, f_dir)

            loop_start_center = np.array([side * 0.365, -0.088, 1.110])
            for v in comp_v:
                p_cur = np.array(v.co[:])
                rel = p_cur - loop_start_center
                d_along = np.dot(rel, f_dir)
                p_proj = p_cur - f_dir * (0.75 * d_along)
                v.co = Vector(p_proj.tolist())

            cur_edges = comp_e
            # Extend sleeve 8.2cm down to z~0.962m (flush with waist hem in all views) with sharp holding loops
            # at the open cuff rim and a deep wide recessed tunnel wall
            cuff_specs = [
                (0.024, 0.042, 0.037, 0.65),
                (0.050, 0.040, 0.035, 0.85),
                (0.070, 0.038, 0.033, 0.95),
                (0.079, 0.036, 0.031, 1.00),  # pre-rim holding loop
                (0.082, 0.036, 0.031, 1.00),  # outer cuff rim edge
                (0.082, 0.028, 0.024, 1.00),  # inner cuff lip edge (8mm thick ribbed cuff wall)
                (0.079, 0.027, 0.023, 1.00),  # inner lip holding loop
                (0.045, 0.026, 0.022, 1.00),  # deep recessed interior sleeve tunnel wall
                (0.036, 0.012, 0.010, 1.00),  # deep shadow termination inside tunnel
            ]
            for step_dist, r_u, r_v, blend in cuff_specs:
                ret = bmesh.ops.extrude_edge_only(bm_sw, edges=cur_edges)
                geom = ret["geom"]
                new_verts = [ele for ele in geom if isinstance(ele, bmesh.types.BMVert)]
                cur_edges = [ele for ele in geom if isinstance(ele, bmesh.types.BMEdge)]
                target_center = loop_start_center + f_dir * step_dist
                for v in new_verts:
                    p_cur = np.array(v.co[:])
                    rel = p_cur - target_center
                    u_coord = np.dot(rel, side_v)
                    v_coord = np.dot(rel, up_v)
                    ang = math.atan2(v_coord, u_coord)
                    p_ideal = target_center + side_v * (r_u * math.cos(ang)) + up_v * (r_v * math.sin(ang))
                    p_final = p_cur * (1.0 - blend) + p_ideal * blend
                    v.co = Vector(p_final.tolist())

    bm_sw.verts.ensure_lookup_table()
    bm_sw.faces.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm_sw, faces=bm_sw.faces)
    # Ensure all sweater face normals point outward from the torso/arm center
    for f in bm_sw.faces:
        fc = f.calc_center_median()
        ref_c = Vector((0.0, -0.03, fc.z)) if abs(fc.x) < 0.20 else Vector((fc.x, -0.08, 1.12))
        if f.normal.dot(fc - ref_c) < 0.0:
            f.normal_flip()
    bm_sw.normal_update()

    # Extract canonical vertices and normals of SweaterBody to apply realistic sweater volume & folds
    sw_canon = np.array([v.co[:] for v in bm_sw.verts], dtype=np.float64)
    sw_norms = np.array([v.normal[:] for v in bm_sw.verts], dtype=np.float64)

    sx = sw_canon[:, 0]
    sy = sw_canon[:, 1]
    sz = sw_canon[:, 2]
    sax = np.abs(sx)

    # Inflate sweater outward along vertex normals so it drapes softly over the body (8mm..15mm loft)
    collar_taper = 1.0 - 0.55 * smoothstep(1.33, 1.375, sz) * (sax < 0.14)
    cuff_taper = 1.0 - 0.45 * smoothstep(0.36, 0.43, sax)
    base_loft = (0.010 + 0.004 * smoothstep(1.32, 1.12, sz)) * collar_taper * cuff_taper

    # Fill out the bust sternum cleavage hollow of the base anatomical mesh so the knitted sweater
    # spans smoothly across the chest like real heavy knitwear
    cleavage_bridge = gauss(sx, 0.0, 0.042) * gauss(sz, 1.21, 0.065) * smoothstep(-0.08, -0.13, sy)
    sw_canon[:, 1] -= 0.018 * cleavage_bridge

    # Add plausible, natural clothing folds (shoulder/armpit drape folds, elbow bend folds, chest drape)
    armpit_fold = (
        gauss(sax, 0.145, 0.030)
        * gauss(sz, 1.26, 0.050)
        * np.sin((sz - 1.20) * 70.0 + sax * 35.0)
    )
    elbow_fold = (
        gauss(sax, 0.29, 0.040)
        * gauss(sz, 1.17, 0.040)
        * np.sin((sax - 0.24) * 85.0 + sy * 45.0)
    )
    torso_drape = (
        smoothstep(1.30, 1.12, sz)
        * (sax < 0.18)
        * 0.0026
        * np.sin(sx * 32.0 + sz * 24.0)
    )
    total_loft = base_loft + 0.0028 * armpit_fold + 0.0024 * elbow_fold + torso_drape
    sw_canon = sw_canon + sw_norms * total_loft[:, None]

    # Pose the sweater vertices with the exact same pose function as the character
    sw_posed = pose_point_array(sw_canon, include_arms=True)
    for i, v in enumerate(bm_sw.verts):
        v.co = Vector(sw_posed[i].tolist())
    bmesh.ops.recalc_face_normals(bm_sw, faces=bm_sw.faces)
    bm_sw.normal_update()
    sw_posed_normals = np.array([v.normal[:] for v in bm_sw.verts], dtype=np.float64)

    me_sw = bpy.data.meshes.new("SweaterBodyMesh")
    bm_sw.to_mesh(me_sw)
    bm_sw.free()
    me_sw.update()

    # Create distortion-free KnitUV on SweaterBody from canonical coordinates
    uv_knit = me_sw.uv_layers.new(name="KnitUV")
    me_sw.uv_layers.active = uv_knit
    u_sw = 0.5 + sw_canon[:, 0] * 1.55 + 0.22 * sw_canon[:, 1]
    v_sw = (sw_canon[:, 2] - 0.94) * 2.15 + 0.35 * np.abs(sw_canon[:, 1])

    for poly in me_sw.polygons:
        poly.use_smooth = True
        for loop_idx in poly.loop_indices:
            vi = me_sw.loops[loop_idx].vertex_index
            uv_knit.data[loop_idx].uv = (float(u_sw[vi]), float(v_sw[vi]))

    attr_sw_canon = me_sw.attributes.new(name="canon_pos", type="FLOAT_VECTOR", domain="POINT")
    attr_sw_canon.data.foreach_set("vector", sw_canon.astype(np.float32).ravel())

    obj_sw = bpy.data.objects.new("SweaterBody", me_sw)
    char_col.objects.link(obj_sw)
    obj_sw.data.materials.append(mat_sweater)

    # Give SweaterBody genuine 3D thickness and smooth subdivision (micro-stitch relief is handled
    # at full 2048x2048 resolution in Mat_Sweater to avoid Nyquist sawtooth stepping on silhouettes)
    mod_sol = obj_sw.modifiers.new(name="Solidify", type="SOLIDIFY")
    mod_sol.thickness = 0.0045
    mod_sol.offset = -1.0  # thicken inward so outer silhouette stays exact

    mod_sub = obj_sw.modifiers.new(name="Subdivision", type="SUBSURF")
    mod_sub.levels = 1
    mod_sub.render_levels = 2
    mod_sub.uv_smooth = "PRESERVE_BOUNDARIES"

    # --- 3. Build SweaterCollar (3D ribbed crew-neck collar ring wrapping the neckline) ---
    collar_posed_v = build_ribbed_collar(char_col, mat_collar)

    return obj_head, obj_sw, head_posed_v, sw_posed, sw_posed_normals, collar_posed_v


def build_ribbed_collar(char_col, mat_collar):
    """
    Creates a genuine 3D torus-band ribbed crew-neck collar around the canonical neckline
    z = 1.370 + 0.22*y + 1.8*x^2, with 64 angular segments and 16 cross-section segments,
    sculpted with vertical rib ridges and posed with pose_point_array.
    """
    n_u = 72
    n_v = 16
    verts_canon = []
    uvs = []
    for iu in range(n_u):
        ang = (iu / n_u) * 2.0 * math.pi  # 0 = front (-Y), pi/2 = +X
        # Neckline ellipse in canonical space: center (0, -0.022), rx = 0.094, ry = 0.068
        cx_base = 0.094 * math.sin(ang)
        cy_base = -0.022 - 0.068 * math.cos(ang)
        cz_base = 1.366 + 0.21 * cy_base + 1.45 * (cx_base ** 2)

        # Outward radial direction in XY and upward direction
        rad_dir = np.array([math.sin(ang), -math.cos(ang), 0.15])
        rad_dir /= np.linalg.norm(rad_dir)
        up_dir = np.array([-0.12 * math.sin(ang), 0.18 * math.cos(ang), 1.0])
        up_dir /= np.linalg.norm(up_dir)

        # 1x1 rib corrugation along angle (56 ribs around collar)
        rib = 0.0011 * math.cos(iu * 56.0 * 2.0 * math.pi / n_u)

        for iv in range(n_v):
            phi = (iv / n_v) * 2.0 * math.pi
            # Cross-section: 19mm tall band, 6.5mm thick rolled knit collar
            d_up = 0.011 * math.sin(phi) + 0.003
            d_rad = (0.0055 + rib * (0.5 + 0.5 * math.cos(phi))) * math.cos(phi) + 0.0035
            pt = np.array([cx_base, cy_base, cz_base]) + up_dir * d_up + rad_dir * d_rad
            verts_canon.append(pt)
            uvs.append((iu / n_u, iv / n_v))

    verts_canon = np.array(verts_canon, dtype=np.float64)
    verts_posed = pose_point_array(verts_canon)

    faces = []
    face_uvs = []
    for iu in range(n_u):
        iu_next = (iu + 1) % n_u
        for iv in range(n_v):
            iv_next = (iv + 1) % n_v
            i0 = iu * n_v + iv
            i1 = iu_next * n_v + iv
            i2 = iu_next * n_v + iv_next
            i3 = iu * n_v + iv_next
            faces.append([i0, i1, i2, i3])
            face_uvs.append([
                (iu / n_u, iv / n_v),
                ((iu + 1) / n_u, iv / n_v),
                ((iu + 1) / n_u, (iv + 1) / n_v),
                (iu / n_u, (iv + 1) / n_v),
            ])

    me_col = bpy.data.meshes.new("SweaterCollarMesh")
    me_col.from_pydata(verts_posed.tolist(), [], faces)
    me_col.update()

    uv_layer = me_col.uv_layers.new(name="KnitUV")
    me_col.uv_layers.active = uv_layer
    for pi, poly in enumerate(me_col.polygons):
        poly.use_smooth = True
        for lk, li in enumerate(poly.loop_indices):
            uv_layer.data[li].uv = face_uvs[pi][lk]

    obj_col = bpy.data.objects.new("SweaterCollar", me_col)
    char_col.objects.link(obj_col)
    obj_col.data.materials.append(mat_collar)

    sub = obj_col.modifiers.new(name="Subdivision", type="SUBSURF")
    sub.levels = 1
    sub.render_levels = 1
    return verts_posed


def build_cuff_hands(char_col, mat_skin):
    """
    Builds clean, smooth anatomical wrist/hand forms emerging slightly from the
    extruded sleeve cuffs so the ends of the sleeves are closed and natural in all views.
    """
    bm = bmesh.new()
    for side in [-1.0, 1.0]:
        elbow = np.array([side * 0.3129, -0.0132, 1.1661])
        hand = np.array([side * 0.4312, -0.1757, 1.0620])
        f_dir = hand - elbow
        f_dir /= np.linalg.norm(f_dir)
        cuff_center = np.array([side * 0.3727, -0.1000, 1.1057]) + f_dir * (3 * 0.022)

        # Create a smooth rounded wrist/palm cap oriented along f_dir
        ret = bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=1.0)
        s_verts = ret["verts"]
        # Build local basis along f_dir
        up_v = np.array([0.0, 0.0, 1.0])
        side_v = np.cross(f_dir, up_v)
        side_v /= np.linalg.norm(side_v)
        up_v = np.cross(side_v, f_dir)

        pts = []
        for v in s_verts:
            lx, ly, lz = v.co.x * 0.028, v.co.y * 0.021, v.co.z * 0.044
            pt = cuff_center + side_v * lx + up_v * ly + f_dir * (lz + 0.010)
            pts.append(pt)
        pts_posed = pose_point_array(np.array(pts, dtype=np.float64), include_arms=True)
        for v, p in zip(s_verts, pts_posed):
            v.co = Vector(p.tolist())

    me_h = bpy.data.meshes.new("CharacterHandsMesh")
    bm.to_mesh(me_h)
    bm.free()
    # Add FaceCylUV on hands mapped to neutral fair skin region (u=0.5, v=0.15)
    uv_h = me_h.uv_layers.new(name="FaceCylUV")
    me_h.uv_layers.active = uv_h
    for p in me_h.polygons:
        p.use_smooth = True
        for li in p.loop_indices:
            uv_h.data[li].uv = (0.5, 0.15)
    obj_h = bpy.data.objects.new("CharacterHands", me_h)
    char_col.objects.link(obj_h)
    obj_h.data.materials.append(mat_skin)
    return obj_h


def build_eyes_lashes_brows(char_col, mat_iris, mat_cornea, mat_tear, mat_lash_brow, cam_pos):
    """
    Constructs realistic 2-layer anatomical eyes (Eyeball + refractive Cornea + Tearline),
    3D tapered Eyelashes, and 3D tapered Eyebrows.
    The eyeball center and canonical gaze vector are calibrated so the warm chestnut iris
    and dark pupil sit naturally in the upper-center of the almond palpebral aperture
    with zero upper or lower scleral show across all camera views.
    """
    eye_radius = 0.0132
    for side, s_label in [(1.0, "L"), (-1.0, "R")]:
        canon_ec = np.array([side * 0.030775, -0.1282, 1.5462])
        posed_ec, head_mat = pose_head_rigid_point(canon_ec)

        # Orient gaze in the canonical head frame toward BeautyCamera (+X in head frame due to -7.5 deg yaw)
        # with gentle convergence and +2.1 deg upward pitch so the upper eyelid covers the top 25% of the iris
        gx = 0.036 if side < 0 else 0.006
        canon_gaze = np.array([gx, -0.9988, 0.036], dtype=np.float64)
        canon_gaze /= np.linalg.norm(canon_gaze)
        gaze_dir = head_mat @ canon_gaze
        gaze_dir /= np.linalg.norm(gaze_dir)

        head_up = head_mat @ np.array([0.0, 0.0, 1.0])
        eye_right = np.cross(gaze_dir, head_up)
        eye_right /= np.linalg.norm(eye_right)
        eye_up = np.cross(eye_right, gaze_dir)
        eye_up /= np.linalg.norm(eye_up)

        # --- A. Eyeball (Sclera + concave Iris + flat/dark Pupil center) ---
        n_lon, n_lat = 48, 32
        ev_list = []
        euv_list = []
        for ilat in range(n_lat + 1):
            theta = (ilat / n_lat) * math.pi  # 0 at front pole (gaze_dir), pi at back pole
            sin_t, cos_t = math.sin(theta), math.cos(theta)
            # Iris spans theta in [0, 0.44 rad] (approx 11.6mm diameter)
            iris_indent = 0.0
            if theta < 0.44:
                t_norm = theta / 0.44
                iris_indent = -0.0006 * (1.0 - t_norm ** 2)
            for ilon in range(n_lon):
                phi = (ilon / n_lon) * 2.0 * math.pi
                local_p = (
                    eye_right * (eye_radius * sin_t * math.cos(phi))
                    + eye_up * (eye_radius * sin_t * math.sin(phi))
                    + gaze_dir * (eye_radius * cos_t + iris_indent)
                )
                ev_list.append((posed_ec + local_p).tolist())
                # Radial UV centered at (0.5, 0.5)
                r_uv = (theta / math.pi) * 1.12
                u = 0.5 + r_uv * math.cos(phi)
                v = 0.5 + r_uv * math.sin(phi)
                euv_list.append((u, v))

        ef_list = []
        ef_uvs = []
        for ilat in range(n_lat):
            for ilon in range(n_lon):
                ilon_next = (ilon + 1) % n_lon
                i0 = ilat * n_lon + ilon
                i1 = ilat * n_lon + ilon_next
                i2 = (ilat + 1) * n_lon + ilon_next
                i3 = (ilat + 1) * n_lon + ilon
                ef_list.append([i0, i1, i2, i3])
                ef_uvs.append([euv_list[i0], euv_list[i1], euv_list[i2], euv_list[i3]])

        me_eye = bpy.data.meshes.new(f"Eyeball_{s_label}_Mesh")
        me_eye.from_pydata(ev_list, [], ef_list)
        me_eye.update()
        uv_e = me_eye.uv_layers.new(name="UVMap")
        me_eye.uv_layers.active = uv_e
        for pi, poly in enumerate(me_eye.polygons):
            poly.use_smooth = True
            for lk, li in enumerate(poly.loop_indices):
                uv_e.data[li].uv = ef_uvs[pi][lk]

        obj_eye = bpy.data.objects.new(f"Eyeball_{s_label}", me_eye)
        char_col.objects.link(obj_eye)
        obj_eye.data.materials.append(mat_iris)

        # --- B. Refractive Cornea shell with realistic anterior corneal bulge ---
        cv_list = []
        for ilat in range(n_lat + 1):
            theta = (ilat / n_lat) * math.pi
            sin_t, cos_t = math.sin(theta), math.cos(theta)
            cornea_bulge = 0.0
            if theta < 0.45:
                t_norm = theta / 0.45
                cornea_bulge = 0.0008 * ((1.0 - t_norm ** 2) ** 1.5)
            r_c = eye_radius + 0.00018
            for ilon in range(n_lon):
                phi = (ilon / n_lon) * 2.0 * math.pi
                local_p = (
                    eye_right * (r_c * sin_t * math.cos(phi))
                    + eye_up * (r_c * sin_t * math.sin(phi))
                    + gaze_dir * (r_c * cos_t + cornea_bulge)
                )
                cv_list.append((posed_ec + local_p).tolist())

        me_cor = bpy.data.meshes.new(f"Cornea_{s_label}_Mesh")
        me_cor.from_pydata(cv_list, [], ef_list)
        me_cor.update()
        for poly in me_cor.polygons:
            poly.use_smooth = True
        obj_cor = bpy.data.objects.new(f"Cornea_{s_label}", me_cor)
        char_col.objects.link(obj_cor)
        obj_cor.data.materials.append(mat_cornea)

        # --- C. Wet Tearline meniscus & inner lacrimal caruncle ---
        build_tearline_and_caruncle(char_col, s_label, side, mat_tear)

        # --- D. 3D Eyelashes (upper and lower curved tapered lashes) ---
        build_eyelashes(char_col, s_label, side, mat_lash_brow)

        # --- E. 3D Eyebrows (soft arched brow hairs along supraorbital ridge) ---
        build_eyebrows(char_col, s_label, side, mat_lash_brow)


def build_tearline_and_caruncle(char_col, s_label, side, mat_tear):
    """Creates a subtle wet tear meniscus strip along the lower eyelid margin and inner canthus."""
    bm = bmesh.new()
    n_seg = 24
    ring_verts = []
    for i in range(n_seg + 1):
        u = i / n_seg
        # Span from inner canthus (|x|=0.0175) to outer canthus (|x|=0.0440)
        cx = side * (0.0175 + u * (0.0440 - 0.0175))
        arch = math.sin(u * math.pi)
        cz = 1.5422 - 0.0024 * arch + 0.0028 * (u ** 1.6)
        cy = -0.1368 - 0.0054 * arch + 0.0012 * u
        # Small triangle/quad cross section for meniscus
        c_pts = np.array([
            [cx, cy - 0.0005, cz + 0.0004],
            [cx, cy + 0.0006, cz + 0.0006],
            [cx, cy - 0.0002, cz - 0.0005],
        ], dtype=np.float64)
        p_pts = pose_point_array(c_pts, include_arms=False)
        rv = [bm.verts.new(p.tolist()) for p in p_pts]
        ring_verts.append(rv)

    for i in range(n_seg):
        r0, r1 = ring_verts[i], ring_verts[i + 1]
        for k in range(3):
            k1 = (k + 1) % 3
            bm.faces.new([r0[k], r1[k], r1[k1], r0[k1]])

    me = bpy.data.meshes.new(f"EyeTearline_{s_label}_Mesh")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(f"EyeTearline_{s_label}", me)
    char_col.objects.link(obj)
    obj.data.materials.append(mat_tear)


def build_eyelashes(char_col, s_label, side, mat_lash_brow):
    """
    Constructs genuine 3D tapered curved eyelash strands along the upper and lower eyelid rims.
    """
    rng = np.random.default_rng(100 if side > 0 else 200)
    verts = []
    faces = []

    def add_lash_strand(root_c, dir_c, curl_c, length, r_base):
        n_steps = 6
        # Build local orthogonal frame around dir_c
        d_norm = dir_c / (np.linalg.norm(dir_c) + 1e-12)
        side_v = np.cross(d_norm, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side_v) < 1e-5:
            side_v = np.array([1.0, 0.0, 0.0])
        side_v /= np.linalg.norm(side_v)
        up_v = np.cross(side_v, d_norm)

        pts_canon = []
        for s in range(n_steps + 1):
            t = s / n_steps
            # Curved trajectory: initial outward direction + upward/backward curl
            center_c = root_c + d_norm * (length * t) + curl_c * (length * (t ** 2.1))
            rad = r_base * (1.0 - 0.86 * (t ** 1.3))
            for k in range(4):
                ang = k * 0.5 * math.pi
                offset = side_v * (rad * math.cos(ang)) + up_v * (rad * math.sin(ang))
                pts_canon.append(center_c + offset)

        pts_posed = pose_point_array(np.array(pts_canon, dtype=np.float64), include_arms=False)
        base_idx = len(verts)
        for p in pts_posed:
            verts.append(p.tolist())

        for s in range(n_steps):
            for k in range(4):
                k1 = (k + 1) % 4
                i0 = base_idx + s * 4 + k
                i1 = base_idx + s * 4 + k1
                i2 = base_idx + (s + 1) * 4 + k1
                i3 = base_idx + (s + 1) * 4 + k
                faces.append([i0, i1, i2, i3])

    # Upper eyelashes (92 strands following the gently lowered almond upper eyelid rim)
    n_upper = 92
    for i in range(n_upper):
        u = (i + 0.5) / n_upper
        u_skew = u ** 0.92
        cx = side * (0.0185 + u_skew * (0.0445 - 0.0185))
        arch = math.sin(u_skew * math.pi)
        cz = 1.5438 + 0.0050 * (arch ** 0.85) + 0.0006 * u_skew
        cy = -0.1372 - 0.0058 * arch + 0.0010 * (u_skew ** 2)

        root_c = np.array([cx, cy, cz])
        # Fan outward laterally and forward (-Y), curling gracefully upward (+Z)
        fan_x = side * (0.15 + 0.55 * (u_skew - 0.25)) + rng.normal(0, 0.04)
        dir_c = np.array([fan_x, -1.0, 0.12 + rng.normal(0, 0.04)])
        curl_c = np.array([side * 0.12 * u_skew, 0.22, 0.72])
        length = (0.0040 + 0.0038 * math.sin(u_skew * math.pi) + 0.0012 * u_skew) * rng.uniform(0.88, 1.12)
        add_lash_strand(root_c, dir_c, curl_c, length, r_base=0.00022)

    # Lower eyelashes (46 delicate shorter strands along lower lid margin)
    n_lower = 46
    for i in range(n_lower):
        u = (i + 0.5) / n_lower
        cx = side * (0.0195 + u * (0.0435 - 0.0195))
        arch = math.sin(u * math.pi)
        cz = 1.5420 - 0.0026 * arch + 0.0025 * (u ** 1.5)
        cy = -0.1365 - 0.0052 * arch + 0.0012 * u

        root_c = np.array([cx, cy, cz])
        fan_x = side * (0.10 + 0.45 * (u - 0.3)) + rng.normal(0, 0.04)
        dir_c = np.array([fan_x, -0.95, -0.45 + rng.normal(0, 0.04)])
        curl_c = np.array([side * 0.05, 0.25, -0.35])
        length = (0.0022 + 0.0016 * arch) * rng.uniform(0.85, 1.12)
        add_lash_strand(root_c, dir_c, curl_c, length, r_base=0.00015)

    me = bpy.data.meshes.new(f"Eyelashes_{s_label}_Mesh")
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(f"Eyelashes_{s_label}", me)
    char_col.objects.link(obj)
    obj.data.materials.append(mat_lash_brow)


def build_eyebrows(char_col, s_label, side, mat_lash_brow):
    """
    Constructs soft, natural 3D eyebrow hair strands following the feminine supraorbital arch.
    """
    rng = np.random.default_rng(300 if side > 0 else 400)
    verts = []
    faces = []

    n_hairs = 210
    for i in range(n_hairs):
        u = rng.beta(1.2, 1.35)  # natural density distribution from medial to lateral brow
        v_off = rng.uniform(-1.0, 1.0)
        # Medial start at |x|=0.0125, arch peak at |x|=0.034, lateral tail at |x|=0.049
        cx = side * (0.0125 + u * (0.0495 - 0.0125))
        arch_z = 1.5585 + 0.0055 * math.sin(u * math.pi * 0.88) - 0.0022 * (u ** 2.2)
        brow_half_h = 0.0030 * (1.0 - 0.50 * (u ** 1.8))
        cz = arch_z + v_off * brow_half_h

        # Project onto forehead/brow surface contour in Y
        cy = -0.1502 + 0.34 * ((abs(cx) - 0.010) ** 1.35) - 0.0006

        root_c = np.array([cx, cy, cz])
        # Hair direction: medial hairs angle upward+outward, mid/lateral hairs sweep laterally and slightly down at tail
        slope_z = 0.42 * (1.0 - 1.35 * u) - 0.15 * v_off
        dir_c = np.array([side * 0.92, 0.38 * u, slope_z])
        dir_c /= np.linalg.norm(dir_c)

        length = (0.0060 + 0.0024 * math.sin(u * math.pi)) * rng.uniform(0.85, 1.15)
        r_base = 0.00021

        n_steps = 4
        pts_canon = []
        up_v = np.array([0.0, 0.0, 1.0])
        out_v = np.array([side * 0.25, -0.96, 0.0])
        for s in range(n_steps + 1):
            t = s / n_steps
            # Hug the brow ridge closely
            c_pt = root_c + dir_c * (length * t) + out_v * (0.0005 * math.sin(t * math.pi))
            c_pt[1] += 0.0008 * ((t * u) ** 1.5)
            rad = r_base * (1.0 - 0.84 * (t ** 1.2))
            for k in range(3):
                ang = k * (2.0 * math.pi / 3.0)
                offset = up_v * (rad * math.cos(ang)) + out_v * (rad * math.sin(ang))
                pts_canon.append(c_pt + offset)

        pts_posed = pose_point_array(np.array(pts_canon, dtype=np.float64), include_arms=False)
        base_idx = len(verts)
        for p in pts_posed:
            verts.append(p.tolist())
        for s in range(n_steps):
            for k in range(3):
                k1 = (k + 1) % 3
                i0 = base_idx + s * 3 + k
                i1 = base_idx + s * 3 + k1
                i2 = base_idx + (s + 1) * 3 + k1
                i3 = base_idx + (s + 1) * 3 + k
                faces.append([i0, i1, i2, i3])

    me = bpy.data.meshes.new(f"Eyebrows_{s_label}_Mesh")
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(f"Eyebrows_{s_label}", me)
    char_col.objects.link(obj)
    obj.data.materials.append(mat_lash_brow)


def enforce_hair_body_collision_canonical(pts, is_bangs=False, cran_scale=1.022):
    """
    Ensures canonical 3D hair points (N, 3) never penetrate the face, cranium, neck,
    or sweater shoulders/back/chest while allowing natural draping over the shoulders.
    """
    p = pts.copy()

    # 1. Cranium ellipsoid repulsion (center (0, -0.052, 1.575), radii (0.079, 0.100, 0.095))
    c_cran = np.array([0.0, -0.052, 1.575])
    r_cran = np.array([0.079, 0.100, 0.095])
    rel = (p - c_cran) / r_cran
    d = np.linalg.norm(rel, axis=1)
    mask = (d < cran_scale) & (p[:, 2] > 1.49)
    if np.any(mask):
        dirs = rel[mask] / (d[mask, None] + 1e-12)
        p[mask] = c_cran + (dirs * cran_scale) * r_cran

    # 2. Face & jaw clearance (for non-bangs strands, keep clear of the front facial mask)
    if not is_bangs:
        c_face = np.array([0.0, -0.088, 1.505])
        r_face = np.array([0.081, 0.084, 0.088])
        rel_f = (p - c_face) / r_face
        df = np.linalg.norm(rel_f, axis=1)
        f_mask = (df < 1.04) & (p[:, 1] < -0.060) & (p[:, 2] > 1.41) & (p[:, 2] < 1.59)
        if np.any(f_mask):
            sx = np.where(p[f_mask, 0] >= 0.0, 1.0, -1.0)
            req_x = c_face[0] + sx * r_face[0] * np.sqrt(
                np.maximum(0.02, 1.04 ** 2 - rel_f[f_mask, 1] ** 2 - rel_f[f_mask, 2] ** 2)
            )
            blend = smoothstep(-0.060, -0.100, p[f_mask, 1])
            p[f_mask, 0] = p[f_mask, 0] * (1.0 - blend) + req_x * blend
    else:
        # Bangs/curtain wisps: smooth C-infinity cranial ellipsoid repulsion so strands never clip into forehead
        c_fore = np.array([0.0, -0.056, 1.575])
        r_fore = np.array([0.080, 0.102, 0.095])
        rel_b = (p - c_fore) / r_fore
        db = np.linalg.norm(rel_b, axis=1)
        b_mask = (db < 1.032) & (p[:, 2] > 1.52)
        if np.any(b_mask):
            dirs_b = rel_b[b_mask] / (db[b_mask, None] + 1e-12)
            p[b_mask] = c_fore + (dirs_b * 1.032) * r_fore

    # 3. Neck & collar clearance (z in 1.33..1.45)
    c_neck = np.array([0.0, -0.020, 1.390])
    r_neck = np.array([0.102, 0.086, 0.090])
    rel_n = (p - c_neck) / r_neck
    dn = np.linalg.norm(rel_n, axis=1)
    n_mask = (dn < 1.05) & (p[:, 2] > 1.33) & (p[:, 2] < 1.45)
    if np.any(n_mask):
        dirs_xy = rel_n[n_mask, :2]
        n_xy = np.linalg.norm(dirs_xy, axis=1, keepdims=True) + 1e-12
        req_r = np.sqrt(np.maximum(0.08, 1.05 ** 2 - rel_n[n_mask, 2:3] ** 2))
        p[n_mask, 0] = c_neck[0] + (dirs_xy[:, 0:1] / n_xy)[:, 0] * req_r[:, 0] * r_neck[0]
        p[n_mask, 1] = c_neck[1] + (dirs_xy[:, 1:2] / n_xy)[:, 0] * req_r[:, 0] * r_neck[1]

    # 4. Continuous cylindrical/anatomical torso & shoulder clearance for all z < 1.385
    # Unlike a sphere, the human torso continues down to the waist (z=0.95), so rx(z) and ry(z) stay full!
    sh_mask = p[:, 2] < 1.385
    if np.any(sh_mask):
        ps = p[sh_mask].copy()
        sz = ps[:, 2]
        # Shoulder width peaks around z=1.28..1.34 (rx=0.225), torso width below z=1.22 is rx=0.185
        shoulder_bell = smoothstep(1.14, 1.28, sz) * (1.0 - 0.35 * smoothstep(1.34, 1.385, sz))
        rx_z = 0.175 + 0.055 * shoulder_bell
        # Back (+Y) of sweater reaches y = +0.108, front chest (-Y) reaches y = -0.175
        y_center = -0.032
        ry_back = 0.145  # so y_center + ry_back = +0.113m on the back!
        ry_front = 0.152  # so y_center - ry_front = -0.184m on the chest!
        dy = ps[:, 1] - y_center
        ry_z = np.where(dy >= 0.0, ry_back, ry_front)

        nx = ps[:, 0] / rx_z
        ny = dy / ry_z
        r_norm = np.sqrt(nx ** 2 + ny ** 2) + 1e-12
        inside = r_norm < 1.04
        if np.any(inside):
            scale_out = 1.04 / r_norm[inside]
            ps[inside, 0] = nx[inside] * scale_out * rx_z[inside]
            ps[inside, 1] = y_center + ny[inside] * scale_out * ry_z[inside]
        p[sh_mask] = ps

    return p


def build_posed_body_radial_envelope(head_posed_v, sw_posed_v, sw_posed_normals, collar_posed_v):
    """
    Builds a fast, exact 2D cylindrical heightfield R_max(z, theta) from the actual posed
    vertices of CharacterHead, SweaterBody (including outward shell margin), and SweaterCollar
    so that every posed hair lock and strand is guaranteed to stay outside the sweater and neck.
    """
    n_z = 96
    n_t = 96
    z_min, z_max = 0.92, 1.67
    r_grid = np.zeros((n_z, n_t), dtype=np.float64)

    sw_outer_v = sw_posed_v + 0.0085 * sw_posed_normals
    all_v = np.vstack([head_posed_v, sw_posed_v, sw_outer_v, collar_posed_v])
    # Focus on torso, neck, head, and shoulders (|x| < 0.25)
    body_v = all_v[np.abs(all_v[:, 0]) < 0.25]
    cx0, cy0 = 0.0, -0.025
    dx = body_v[:, 0] - cx0
    dy = body_v[:, 1] - cy0
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dx, -dy)  # -pi .. +pi (0 at front -Y, +-pi at back +Y)

    iz = np.clip(((body_v[:, 2] - z_min) / (z_max - z_min) * (n_z - 1)).astype(int), 0, n_z - 1)
    it = np.clip(((theta + math.pi) / (2.0 * math.pi) * n_t).astype(int), 0, n_t - 1)
    np.maximum.at(r_grid, (iz, it), r)

    # Dilate (3x3 max filter with periodic wrap in theta) to bridge across polygon faces and step edges
    for _ in range(3):
        padded = np.pad(r_grid, ((1, 1), (0, 0)), mode="edge")
        padded = np.pad(padded, ((0, 0), (1, 1)), mode="wrap")
        nb_max = np.zeros_like(r_grid)
        for dz in range(3):
            for dt in range(3):
                nb_max = np.maximum(nb_max, padded[dz:dz + n_z, dt:dt + n_t])
        r_grid = nb_max

    r_dilated = r_grid.copy()
    # Smooth while strictly taking maximum with r_dilated so step edges (collar/shoulders) NEVER shrink
    for _ in range(2):
        padded = np.pad(r_grid, ((1, 1), (0, 0)), mode="edge")
        padded = np.pad(padded, ((0, 0), (1, 1)), mode="wrap")
        r_smooth = (
            0.40 * r_grid
            + 0.15 * (padded[0:n_z, 1:n_t + 1] + padded[2:n_z + 2, 1:n_t + 1])
            + 0.15 * (padded[1:n_z + 1, 0:n_t] + padded[1:n_z + 1, 2:n_t + 2])
        )
        r_grid = np.maximum(r_dilated, r_smooth)

    def push_outside(pts_posed, margin=0.010, is_bangs=False):
        p = pts_posed.copy()
        cx0, cy0 = 0.0, -0.025
        dx = p[:, 0] - cx0
        dy = p[:, 1] - cy0
        r = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
        theta = np.arctan2(dx, -dy)

        # For bangs, allow sitting just in front of forehead; for all other hair below z=1.48, enforce full sweater/neck clearance
        if is_bangs:
            active = p[:, 2] < 1.50
        else:
            active = p[:, 2] < 1.48

        if not np.any(active):
            return p

        fz = np.clip((p[active, 2] - z_min) / (z_max - z_min) * (n_z - 1), 0.0, n_z - 1.001)
        ft = ((theta[active] + math.pi) / (2.0 * math.pi) * n_t) % n_t

        iz0 = fz.astype(int)
        iz1 = np.minimum(iz0 + 1, n_z - 1)
        wz = fz - iz0

        it0 = ft.astype(int) % n_t
        it1 = (it0 + 1) % n_t
        wt = ft - np.floor(ft)

        r_surf = np.maximum(
            np.maximum(r_grid[iz0, it0], r_grid[iz0, it1]),
            np.maximum(r_grid[iz1, it0], r_grid[iz1, it1]),
        )

        req_r = r_surf + margin
        r_act = r[active]
        need_push = r_act < req_r
        if np.any(need_push):
            scale = np.where(need_push, req_r / r_act, 1.0)
            p_act = p[active]
            p_act[:, 0] = cx0 + dx[active] * scale
            p_act[:, 1] = cy0 + dy[active] * scale
            p[active] = p_act
        return p

    return push_outside


def build_hair_system(
    char_col,
    mat_scalp,
    mat_hair_locks,
    mat_hair_strands,
    head_posed_v,
    sw_posed_v,
    sw_posed_normals,
    collar_posed_v,
):
    """
    Constructs a multi-layered, editable 3D hair system matching input/reference.png:
    1. HairScalpCap: form-fitting dark chestnut-brown scalp & inner hair volume mesh preventing any bald gaps.
    2. HairVolumeLocks: 74 sculpted 3D wavy volumetric locks with broad overlapping tangent ribbons at the crown
       forming lush chestnut waves over the crown, sides, shoulders, and back.
    3. HairStrands_Bangs: wispy see-through bangs (HairCurves + delicate 3D ribbon locks) parting
       at the crown and lightly framing the forehead above the eyebrows with zero eye occlusion.
    4. HairStrands_FaceFraming & HairStrands_Main: ~4,500 individual 3D HairCurves strands
       providing fine strand definition, loose waves, and sunlit chestnut highlights.
    """
    rng = np.random.default_rng(42)
    push_outside_posed = build_posed_body_radial_envelope(
        head_posed_v, sw_posed_v, sw_posed_normals, collar_posed_v
    )

    # --- 1. HairScalpCap (form-fitting cranial cap + plush inner hair volume outside CharacterHead) ---
    n_u, n_v = 48, 24
    scalp_pts = []
    scalp_uvs = []
    for iv in range(n_v + 1):
        v = iv / n_v  # 0 at crown, 1 at lower hairline/nape
        for iu in range(n_u):
            u = iu / n_u
            phi = (u - 0.5) * 2.0 * math.pi  # 0 at front (-Y), +-pi at back (+Y)
            # Keep HairScalpCap tucked 2cm behind the front-crown locks (0.17*pi -> cz~1.658m at front)
            # so HairScalpCap seals the crown interior while NEVER protruding below the front hair locks!
            front_factor = math.exp(-((phi / (0.42 * math.pi)) ** 2))
            max_theta = (0.17 * front_factor + 0.72 * (1.0 - front_factor)) * math.pi
            eff_t = v * max_theta
            st, ct = math.sin(eff_t), math.cos(eff_t)

            cx = 0.0815 * st * math.sin(phi)
            cy = -0.052 - 0.1055 * st * math.cos(phi)
            cz = 1.576 + 0.0952 * ct
            scalp_pts.append([cx, cy, cz])
            scalp_uvs.append((u, v))

    scalp_canon = enforce_hair_body_collision_canonical(
        np.array(scalp_pts, dtype=np.float64), is_bangs=False, cran_scale=1.018
    )
    # On the front half of the cranium (cy < -0.060), tuck HairScalpCap 2mm inside CharacterHead
    # so it NEVER appears as a separate dark scalp-cap layer on the forehead in front/three-quarter/beauty views!
    front_tuck = smoothstep(-0.045, -0.095, scalp_canon[:, 1])
    c_cran_sc = np.array([0.0, -0.052, 1.575])
    scalp_canon = c_cran_sc + (scalp_canon - c_cran_sc) * (1.0 - 0.042 * front_tuck[:, None])
    scalp_posed = pose_hair_point_array(scalp_canon)
    scalp_faces = []
    scalp_f_uvs = []
    for iv in range(n_v):
        for iu in range(n_u):
            iu1 = (iu + 1) % n_u
            i0 = iv * n_u + iu
            i1 = iv * n_u + iu1
            i2 = (iv + 1) * n_u + iu1
            i3 = (iv + 1) * n_u + iu
            scalp_faces.append([i0, i1, i2, i3])
            scalp_f_uvs.append([scalp_uvs[i0], scalp_uvs[i1], scalp_uvs[i2], scalp_uvs[i3]])

    me_sc = bpy.data.meshes.new("HairScalpCapMesh")
    me_sc.from_pydata(scalp_posed.tolist(), [], scalp_faces)
    me_sc.update()
    uv_sc = me_sc.uv_layers.new(name="UVMap")
    for pi, p in enumerate(me_sc.polygons):
        p.use_smooth = True
        for lk, li in enumerate(p.loop_indices):
            uv_sc.data[li].uv = scalp_f_uvs[pi][lk]
    obj_sc = bpy.data.objects.new("HairScalpCap", me_sc)
    char_col.objects.link(obj_sc)
    obj_sc.data.materials.append(mat_hair_locks)

    # --- Helper to generate a 3D trajectory for a hair guide lock ---
    part_x = -0.008  # slightly off-center parting
    part_y = -0.068
    part_z = 1.668
    cran_center_posed = pose_hair_point_array(np.array([[0.0, -0.052, 1.565]], dtype=np.float64))[0]

    def generate_lock_spine(azim, layer_idx, length, wave_amp, wave_freq, wave_phase, forward_drape=0.0):
        """
        azim: angle around head where 0 = front (-Y), +pi/2 = left (+X), -pi/2 = right (-X), +-pi = back (+Y)
        layer_idx: 0 (inner volume), 1 (mid layer), 2 (outer wave layer)
        """
        n_pts = 28
        pts = np.zeros((n_pts, 3), dtype=np.float64)
        sin_a, cos_a = math.sin(azim), math.cos(azim)
        abs_deg = math.degrees(abs(azim))
        # Distribute roots along the full front-to-back cranial parting line:
        # front locks (abs_deg=12..55) root down the front slope (-0.128 -> -0.068) so the hairline sits at a natural height
        front_slide = 1.0 - smoothstep(12.0, 55.0, abs_deg)
        part_y_lock = part_y - 0.060 * front_slide + 0.056 * smoothstep(55.0, 165.0, abs_deg)
        part_z_lock = part_z - 0.024 * (front_slide ** 1.25)

        for i in range(n_pts):
            t = i / (n_pts - 1)  # 0 at root/parting, 1 at tip
            root_drop = 0.008 * (1.0 - layer_idx * 0.30)
            rx = part_x * (1.0 - t) + 0.014 * sin_a * (1.0 - 0.30 * layer_idx)
            ry = part_y_lock * (1.0 - t) - 0.016 * cos_a * (1.0 - 0.30 * layer_idx)
            rz = part_z_lock - root_drop

            # Cranial wrap phase (t in 0..0.24) transitioning to cascading shoulder waves (t in 0.16..1.0)
            wrap = smoothstep(0.0, 0.24, t)
            cascade = smoothstep(0.16, 1.0, t)
            drop_t = smoothstep(0.07, 1.0, t)

            # Gentle crown loft hugging and covering the scalp dome
            crown_loft = (0.009 + 0.003 * layer_idx) * math.sin(min(1.0, t * 2.6) * math.pi)

            # Base radial expansion around head, shoulders, and back
            back_ness = max(0.0, -cos_a)  # 1 at back (+Y), 0 at sides/front
            rad_x = 0.086 + 0.005 * layer_idx + (0.074 - 0.022 * back_ness) * (cascade ** 1.10)
            rad_y = 0.108 + 0.005 * layer_idx + (0.036 + 0.048 * back_ness) * (cascade ** 0.95)

            # For front-temple locks (abs_deg < 46), sweep softly over the upper forehead at t < 0.22
            # and then curve gracefully outward past the temples/cheeks so the eyes and cheeks stay open
            temple_sweep = 0.0
            if abs_deg < 46.0:
                sweep_target = 48.0 if azim > 0 else 43.0
                temple_sweep = math.radians(sweep_target - abs_deg) * smoothstep(0.11, 0.30, t) * (1.0 if azim >= 0 else -1.0)

            eff_azim = azim + temple_sweep - forward_drape * cascade * math.sin(azim) * 0.34
            sa, ca = math.sin(eff_azim), math.cos(eff_azim)

            # Loose S-wave modulation
            w_env = smoothstep(0.16, 0.55, t)
            wave1 = wave_amp * w_env * math.sin(t * wave_freq * 2.0 * math.pi + wave_phase)
            wave2 = 0.45 * wave_amp * w_env * math.cos(t * (wave_freq * 1.6) * 2.0 * math.pi - wave_phase * 0.7)

            cx = rx * (1.0 - wrap) + (rad_x * sa + wave1 * ca + 0.5 * wave1 * sa) * wrap
            cy = ry * (1.0 - wrap) + (-0.048 - rad_y * ca + wave2 * sa - forward_drape * 0.042 * cascade) * wrap
            cz = rz - (length * (drop_t ** 1.12)) + crown_loft

            # Ensure viewer-right (+X) cheek/eye corridor stays open and unoccluded as in reference.png
            if cx > 0.015 and cz > 1.44 and cz < 1.58 and cy < -0.065:
                req_cx = 0.066 + 0.008 * smoothstep(1.56, 1.48, cz)
                if cx < req_cx:
                    cx = req_cx

            pts[i] = [cx, cy, cz]

        for _ in range(2):
            pts[1:-1] = 0.25 * pts[:-2] + 0.50 * pts[1:-1] + 0.25 * pts[2:]
        # Enforce cranium repulsion AFTER smoothing so locks stay 5..11mm outside HairScalpCap everywhere!
        pts = enforce_hair_body_collision_canonical(
            pts, is_bangs=False, cran_scale=1.068 + 0.024 * layer_idx
        )
        return pts

    # --- 2. Build HairVolumeLocks (3D sculpted tube/ribbon locks for rich volume + shadow depth) ---
    lock_verts = []
    lock_faces = []
    lock_uvs = []
    guide_spines = []

    # Distribute 78 overlapping locks around front-crown, temples, sides, shoulders, and back
    azim_list = []
    for side in [-1.0, 1.0]:
        for k, a_deg in enumerate([14, 19, 25, 32, 41, 50, 59, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 166, 174]):
            azim_list.append((side * math.radians(a_deg), 1 if k % 2 == 0 else 2))
            if a_deg >= 19:
                azim_list.append((side * math.radians(a_deg + 4.5), 0))  # broad under-layer volume lock
    # Center-back locks
    azim_list.append((math.pi, 0))
    azim_list.append((math.pi, 1))
    azim_list.append((math.pi, 2))
    azim_list.append((math.radians(177.0), 0))
    azim_list.append((math.radians(-177.0), 0))

    for lock_idx, (azim, layer_idx) in enumerate(azim_list):
        abs_deg = math.degrees(abs(azim))
        f_drape = smoothstep(115.0, 35.0, abs_deg) * 0.85
        length = 0.44 + 0.05 * smoothstep(35.0, 95.0, abs_deg) + rng.uniform(-0.018, 0.022)
        w_amp = 0.013 + 0.005 * layer_idx + rng.uniform(-0.002, 0.003)
        w_freq = 2.1 + rng.uniform(-0.22, 0.25)
        w_phase = rng.uniform(0.0, 2.0 * math.pi)

        spine_canon = generate_lock_spine(azim, layer_idx, length, w_amp, w_freq, w_phase, forward_drape=f_drape)
        spine_posed = pose_hair_point_array(spine_canon)
        spine_posed = push_outside_posed(spine_posed, margin=0.015 + 0.003 * layer_idx, is_bangs=False)
        # Smooth only the lower torso drape (i >= 9) so upper cranial dome clearance is never pulled inward
        for _ in range(2):
            spine_posed[9:-1] = 0.25 * spine_posed[8:-2] + 0.50 * spine_posed[9:-1] + 0.25 * spine_posed[10:]
        spine_posed = push_outside_posed(spine_posed, margin=0.014 + 0.002 * layer_idx, is_bangs=False)

        guide_spines.append((spine_posed, azim, layer_idx))

        # Construct 8-sided rounded-oval 3D lock mesh around posed spine with TRUE 3D skull/body surface tangent
        n_pts = len(spine_posed)
        n_sides = 8
        pts_posed_lock = []
        for i in range(n_pts):
            t = i / (n_pts - 1)
            tang = spine_posed[min(n_pts - 1, i + 1)] - spine_posed[max(0, i - 1)]
            tang /= np.linalg.norm(tang) + 1e-12

            # Compute true 3D outward surface normal: spherical from cranial center on head, cylindrical on torso
            w_cran = smoothstep(1.42, 1.54, spine_posed[i, 2])
            rad_cran = spine_posed[i] - cran_center_posed
            rad_cran /= np.linalg.norm(rad_cran) + 1e-12
            rad_cyl = np.array([spine_posed[i, 0], spine_posed[i, 1] - (-0.035), 0.0])
            rad_cyl /= np.linalg.norm(rad_cyl) + 1e-12
            rad_v = w_cran * rad_cran + (1.0 - w_cran) * rad_cyl
            # Orthogonalize rad_v against tang so binorm is strictly tangent to the skull/body surface!
            rad_v = rad_v - np.dot(rad_v, tang) * tang
            r_norm = np.linalg.norm(rad_v)
            if r_norm < 1e-6:
                rad_v = np.array([0.0, 0.0, 1.0])
            else:
                rad_v /= r_norm
            binorm = np.cross(tang, rad_v)
            binorm /= np.linalg.norm(binorm) + 1e-12
            norm_v = np.cross(binorm, tang)

            # Narrower, deeply fluted multi-bundle lock cross-section veiled beneath 6,400 HairCurves strands
            crown_seal = 1.25 - 0.42 * smoothstep(0.14, 0.34, t)
            root_env = (0.78 + 0.22 * smoothstep(0.0, 0.16, t)) * crown_seal
            tip_env = (1.0 - t) ** 0.58 + 0.08 * (1.0 - t)
            taper = root_env * tip_env
            w_tang = (0.0145 - 0.0022 * layer_idx) * taper
            w_rad = (0.0052 - 0.0007 * layer_idx) * taper

            for k in range(n_sides):
                ang = (k / n_sides) * 2.0 * math.pi
                # Deep 4-lobed fluted profile so each lock splits into fine sub-strands rather than a flat ribbon
                flute = 1.0 - 0.28 * (math.sin(4.0 * ang + t * 6.0) ** 2)
                offset = binorm * (w_tang * flute * math.cos(ang)) + norm_v * (w_rad * flute * math.sin(ang))
                pts_posed_lock.append(spine_posed[i] + offset)

        pts_posed_lock = push_outside_posed(np.array(pts_posed_lock, dtype=np.float64), margin=0.0095, is_bangs=False)
        base_v_idx = len(lock_verts)
        for p in pts_posed_lock:
            lock_verts.append(p.tolist())

        u_seed = (lock_idx * 0.137) % 1.0
        for i in range(n_pts - 1):
            v0 = i / (n_pts - 1)
            v1 = (i + 1) / (n_pts - 1)
            for k in range(n_sides):
                k1 = (k + 1) % n_sides
                idx0 = base_v_idx + i * n_sides + k
                idx1 = base_v_idx + i * n_sides + k1
                idx2 = base_v_idx + (i + 1) * n_sides + k1
                idx3 = base_v_idx + (i + 1) * n_sides + k
                lock_faces.append([idx0, idx1, idx2, idx3])
                lock_uvs.append([
                    (u_seed + k / n_sides * 0.25, v0),
                    (u_seed + (k + 1) / n_sides * 0.25, v0),
                    (u_seed + (k + 1) / n_sides * 0.25, v1),
                    (u_seed + k / n_sides * 0.25, v1),
                ])

    me_locks = bpy.data.meshes.new("HairVolumeLocksMesh")
    me_locks.from_pydata(lock_verts, [], lock_faces)
    me_locks.update()
    uv_l = me_locks.uv_layers.new(name="UVMap")
    me_locks.uv_layers.active = uv_l
    for pi, p in enumerate(me_locks.polygons):
        p.use_smooth = True
        for lk, li in enumerate(p.loop_indices):
            uv_l.data[li].uv = lock_uvs[pi][lk]

    obj_locks = bpy.data.objects.new("HairVolumeLocks", me_locks)
    char_col.objects.link(obj_locks)
    obj_locks.data.materials.append(mat_hair_locks)
    sub_locks = obj_locks.modifiers.new(name="Subdivision", type="SUBSURF")
    sub_locks.levels = 1
    sub_locks.render_levels = 1

    # --- 3. Native Blender 5.1 HairCurves for thousands of individual fine 3D hair strands ---
    def create_hair_curves_object(name, curves_posed_list, r_root, r_tip):
        hc = bpy.data.hair_curves.new(f"{name}_Data")
        counts = [len(c) for c in curves_posed_list]
        hc.add_curves(counts)
        all_pts = np.concatenate(curves_posed_list, axis=0).astype(np.float32)
        radii = []
        for c in curves_posed_list:
            n = len(c)
            t = np.linspace(0.0, 1.0, n, dtype=np.float32)
            r = r_root * ((1.0 - t) ** 0.65) + r_tip * (t ** 0.65)
            radii.append(r)
        all_radii = np.concatenate(radii, axis=0).astype(np.float32)
        hc.points.foreach_set("position", all_pts.ravel())
        hc.points.foreach_set("radius", all_radii)
        hc.materials.append(mat_hair_strands)
        obj_hc = bpy.data.objects.new(name, hc)
        char_col.objects.link(obj_hc)
        return obj_hc

    # 3A. HairStrands_Bangs (380 soft, side-swept curtain bangs + diagonal see-through forehead wisps
    # originating inside the front-crown locks and sweeping in diagonal C-curves across the upper/mid forehead)
    front_crown_guides = [g for g in guide_spines if math.degrees(abs(g[1])) <= 45.0]
    bang_curves_posed = []
    eye_L_posed, _ = pose_head_rigid_point((0.0308, -0.142, 1.545))
    eye_R_posed, _ = pose_head_rigid_point((-0.0308, -0.142, 1.545))
    n_curtain = 240 if not PREVIEW_MODE else 120
    for i in range(n_curtain):
        g_posed, azim, l_idx = front_crown_guides[i % len(front_crown_guides)]
        sub_g = g_posed[:15].copy()
        n_pts = len(sub_g)
        t = np.linspace(0.0, 1.0, n_pts)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        dx = rng.normal(0.0, 0.0032) * (0.15 + 0.85 * t) + 0.0020 * np.sin(t * 6.0 + phase)
        dy = rng.normal(0.0, 0.0028) * (0.15 + 0.85 * t) + 0.0016 * np.cos(t * 5.5 + phase)
        dz = rng.normal(0.0, 0.0025) * t
        sub_g[:, 0] += dx
        sub_g[:, 1] += dy
        sub_g[:, 2] += dz
        sub_g = push_outside_posed(sub_g, margin=0.0105, is_bangs=False)
        bang_curves_posed.append(sub_g)

    # Add 140 diagonal, softly curved see-through forehead wisps sweeping from the front parting (-0.008, -0.125, 1.645)
    # across the forehead in gentle S/C-arcs toward the eyebrows (zero vertical barcode lines)
    n_wisps = 140 if not PREVIEW_MODE else 70
    t_b = np.linspace(0.0, 1.0, 15)
    wisp_targets = [-0.044, -0.029, -0.014, 0.004, 0.021, 0.038]
    for i in range(n_wisps):
        xt_base = wisp_targets[i % len(wisp_targets)]
        xt = xt_base + rng.normal(0.0, 0.0038)
        x_root = -0.008 + 0.35 * xt + rng.normal(0.0, 0.0030)
        z_end = 1.560 + 0.006 * (1.0 - abs(xt) / 0.048) + rng.uniform(-0.004, 0.005)
        sweep_dir = 1.0 if xt >= -0.005 else -1.0
        c_arc = sweep_dir * rng.uniform(0.006, 0.012) * np.sin(t_b * math.pi)

        cx = x_root * (1.0 - t_b) + xt * t_b + c_arc
        cz = 1.644 * (1.0 - (t_b ** 1.25)) + z_end * (t_b ** 1.25) + 0.008 * np.sin(t_b * math.pi)
        # Follow spherical forehead surface smoothly (3mm outside CharacterHead forehead)
        rel_z = np.clip((cz - 1.575) / 0.096, -0.95, 0.98)
        rel_x = np.clip(cx / 0.081, -0.95, 0.95)
        sin_d = np.sqrt(np.maximum(0.05, 1.0 - 0.55 * (rel_z ** 2) - 0.45 * (rel_x ** 2)))
        cy = -0.054 - 0.1035 * sin_d - 0.0025 * np.sin(t_b * math.pi)

        b_spine = enforce_hair_body_collision_canonical(np.column_stack([cx, cy, cz]), is_bangs=True)
        c_p = pose_hair_point_array(b_spine)
        for ec_p in [eye_L_posed, eye_R_posed]:
            dx_e = c_p[:, 0] - ec_p[0]
            dz_e = c_p[:, 2] - ec_p[2]
            in_eye = (np.abs(dx_e) < 0.017) & (dz_e < 0.014) & (dz_e > -0.015)
            if np.any(in_eye):
                c_p[in_eye, 2] = ec_p[2] + 0.0145
        bang_curves_posed.append(c_p)

    create_hair_curves_object("HairStrands_Bangs", bang_curves_posed, r_root=0.00026, r_tip=0.00006)

    # 3B. HairStrands_FaceFraming (~1,120 wavy side-framing strands & loose flyaways around cheeks/shoulders)
    framing_guides = [g for g in guide_spines if math.degrees(abs(g[1])) < 88.0]
    framing_curves_posed = []
    n_framing = 1120 if not PREVIEW_MODE else 560
    for i in range(n_framing):
        g_posed, azim, l_idx = framing_guides[i % len(framing_guides)]
        n_pts = len(g_posed)
        t = np.linspace(0.0, 1.0, n_pts)
        is_flyaway = (i % 12 == 0)
        amp_scale = 1.6 if is_flyaway else 1.0
        phase = rng.uniform(0.0, 2.0 * math.pi)
        dx = rng.normal(0.0, 0.0040 * amp_scale) * (0.2 + 0.8 * t) + 0.0024 * amp_scale * np.sin(t * 9.0 + phase)
        dy = rng.normal(0.0, 0.0038 * amp_scale) * (0.2 + 0.8 * t) + 0.0020 * amp_scale * np.cos(t * 8.0 + phase)
        dz = rng.normal(0.0, 0.0030) * t
        c_p = g_posed.copy()
        c_p[:, 0] += dx
        c_p[:, 1] += dy
        c_p[:, 2] += dz
        c_p = push_outside_posed(c_p, margin=0.0105, is_bangs=False)
        framing_curves_posed.append(c_p)

    create_hair_curves_object("HairStrands_FaceFraming", framing_curves_posed, r_root=0.00038, r_tip=0.00007)

    # 3C. HairStrands_Main (~4,200 long wavy chestnut strands covering crown, sides, and back)
    main_curves_posed = []
    n_main = 4200 if not PREVIEW_MODE else 2100
    for i in range(n_main):
        g_posed, azim, l_idx = guide_spines[i % len(guide_spines)]
        n_pts = len(g_posed)
        t = np.linspace(0.0, 1.0, n_pts)
        is_flyaway = (i % 15 == 0)
        amp_s = 1.5 if is_flyaway else 1.0
        phase = rng.uniform(0.0, 2.0 * math.pi)
        dx = rng.normal(0.0, 0.0052 * amp_s) * (0.18 + 0.82 * t) + 0.0028 * amp_s * np.sin(t * 10.0 + phase)
        dy = rng.normal(0.0, 0.0050 * amp_s) * (0.18 + 0.82 * t) + 0.0024 * amp_s * np.cos(t * 9.5 + phase)
        dz = rng.normal(0.0, 0.0040) * t + (0.0025 * np.sin(t * math.pi) if is_flyaway else 0.0)
        c_p = g_posed.copy()
        c_p[:, 0] += dx
        c_p[:, 1] += dy
        c_p[:, 2] += dz
        c_p = push_outside_posed(c_p, margin=0.0105, is_bangs=False)
        main_curves_posed.append(c_p)

    create_hair_curves_object("HairStrands_Main", main_curves_posed, r_root=0.00042, r_tip=0.00008)


def generate_and_pack_textures():
    """
    Generates <=2048x2048 hybrid photographic + procedural textures from input/reference.png and packs them:
    - SkinAlbedoTex (2048x2048, sRGB): Landmark-registered TPS-warped, de-haired, delighted facial albedo
      extracted from input/reference.png (with eye-aperture exclusion so zero double-eye artifacts occur),
      blended seamlessly into color-matched procedural skin across the temples, ears, neck, and back of head.
    - SkinRoughBumpTex (2048x2048, Non-Color): R=photographic high-pass + procedural micro-pore & lip groove height,
      G=spatially varying roughness, B=subsurface weight mask.
    - EyeIrisTex (1024x1024, sRGB): Warm chestnut-brown iris calibrated to reference.png iris chromaticity with
      radial stroma fibers, collarette ring, dark limbal ring, pupil, and warm sclera.
    - HairStrandTex (1024x1024, sRGB): Multi-strand longitudinal chestnut hair texture sampled from reference.png
      root/mid/highlight tones to give HairVolumeLocks fine fibrous strand definition.
    - SweaterKnitHeightTex (2048x2048, Non-Color): Tileable waffle/seed-stitch knit heightmap blended with
      high-pass yarn micro-relief extracted from the unoccluded sweater patch in reference.png.
    - SweaterKnitAlbedoTex (2048x2048, sRGB): Dusty pink knitted yarn albedo calibrated to reference.png
      sweater chromaticity with seamless photographic yarn grain and stitch-crevice AO.
    """
    rng = np.random.default_rng(2026)

    # Load input/reference.png once into a numpy array (top-to-bottom row order, sRGB [0,1]) and remove temp datablock
    ref_path = os.path.join(WORKDIR, "input", "reference.png")
    img_ref_tmp = bpy.data.images.load(ref_path)
    ref_w, ref_h = img_ref_tmp.size
    ref_srgb = np.array(img_ref_tmp.pixels[:], dtype=np.float32).reshape((ref_h, ref_w, 4))[::-1, :, :3].copy()
    bpy.data.images.remove(img_ref_tmp)
    ref_luma = 0.2126 * ref_srgb[:, :, 0] + 0.7152 * ref_srgb[:, :, 1] + 0.0722 * ref_srgb[:, :, 2]

    # --- 1. SkinAlbedoTex & SkinRoughBumpTex (2048x2048 on FaceCylUV) ---
    res_skin = 2048 if not PREVIEW_MODE else 1024
    v_grid, u_grid = np.linspace(0.0, 1.0, res_skin, endpoint=False), np.linspace(0.0, 1.0, res_skin, endpoint=False)
    U, V = np.meshgrid(u_grid, v_grid)

    # Recover canonical angle theta (-pi..pi, 0 at front) and height cz (1.32..1.68)
    theta = (U - 0.5) * (2.0 * math.pi)
    cz = 1.320 + V * (1.680 - 1.320)
    cx = 0.098 * np.sin(theta)
    ax = np.abs(cx)
    front_mask = smoothstep(1.35, 0.75, np.abs(theta))

    # Calibrate base procedural skin tone to rich warm peachy-fair complexion (avoiding AgX highlight desaturation)
    base_r = np.full_like(U, 0.818, dtype=np.float32)
    base_g = np.full_like(U, 0.612, dtype=np.float32)
    base_b = np.full_like(U, 0.542, dtype=np.float32)

    # Multi-scale organic skin chromophore mottling (periodic in theta so zero seam at back)
    mottle = (
        0.009 * np.sin(theta * 14.0) * np.cos(V * 36.0)
        + 0.006 * np.cos(theta * 29.0 + V * 48.0)
        + 0.005 * np.sin(theta * 63.0 - V * 95.0)
    )
    base_r += mottle
    base_g += mottle * 0.64
    base_b += mottle * 0.44

    # Natural neck & submental (under-jaw) tonal transition for 3D jawline definition
    neck_shade = (1.0 - smoothstep(1.415, 1.455, cz)) * 0.09
    base_r *= 1.0 - neck_shade * 0.82
    base_g *= 1.0 - neck_shade * 0.96
    base_b *= 1.0 - neck_shade * 1.02

    # Warm rosy cheek blush on malar cheeks (|cx|~0.038, cz~1.516)
    blush = gauss(ax, 0.038, 0.018) * gauss(cz, 1.516, 0.019) * front_mask
    base_r = base_r * (1.0 - 0.44 * blush) + 0.875 * (0.44 * blush)
    base_g = base_g * (1.0 - 0.44 * blush) + 0.435 * (0.44 * blush)
    base_b = base_b * (1.0 - 0.44 * blush) + 0.410 * (0.44 * blush)

    # Subtle warmth on nose tip & alae (cz~1.508, |cx|<0.018) + soft alar crease shadow
    nose_warm = gauss(ax, 0.0, 0.015) * gauss(cz, 1.508, 0.008) * front_mask
    base_r = base_r * (1.0 - 0.28 * nose_warm) + 0.865 * (0.28 * nose_warm)
    base_g = base_g * (1.0 - 0.28 * nose_warm) + 0.475 * (0.28 * nose_warm)
    base_b = base_b * (1.0 - 0.28 * nose_warm) + 0.445 * (0.28 * nose_warm)

    alar_crease = gauss(ax, 0.0145, 0.0028) * gauss(cz, 1.5055, 0.0055) * front_mask
    base_r *= 1.0 - 0.20 * alar_crease
    base_g *= 1.0 - 0.26 * alar_crease
    base_b *= 1.0 - 0.28 * alar_crease

    # Subtle upper eyelid shadow / warmth, aegyo-sal lower lid shading & crisp upper lash line
    eye_shadow = gauss(ax, 0.031, 0.013) * gauss(cz, 1.548, 0.0055) * front_mask
    base_r *= 1.0 - 0.14 * eye_shadow
    base_g *= 1.0 - 0.20 * eye_shadow
    base_b *= 1.0 - 0.22 * eye_shadow

    aegyo_line = gauss(ax, 0.031, 0.0105) * gauss(cz, 1.5355, 0.0014) * front_mask
    base_r *= 1.0 - 0.14 * aegyo_line
    base_g *= 1.0 - 0.19 * aegyo_line
    base_b *= 1.0 - 0.21 * aegyo_line

    lash_liner = gauss(ax, 0.0315, 0.0120) * gauss(cz, 1.5478 + 0.0008 * ((ax - 0.031) / 0.012), 0.0015) * front_mask
    base_r = base_r * (1.0 - 0.82 * lash_liner) + 0.09 * (0.82 * lash_liner)
    base_g = base_g * (1.0 - 0.82 * lash_liner) + 0.04 * (0.82 * lash_liner)
    base_b = base_b * (1.0 - 0.82 * lash_liner) + 0.03 * (0.82 * lash_liner)

    # Double eyelid crease line (warm shadow crease above lowered almond upper lid)
    lid_crease = gauss(ax, 0.0315, 0.0115) * gauss(cz, 1.5515, 0.0011) * front_mask
    base_r *= 1.0 - 0.28 * lid_crease
    base_g *= 1.0 - 0.35 * lid_crease
    base_b *= 1.0 - 0.38 * lid_crease

    # Soft eyebrow skin shadow tint under the 3D brow hairs
    brow_arch_z = 1.5588 + 0.0050 * np.sin(np.clip((ax - 0.012) / 0.037, 0.0, 1.0) * math.pi * 0.88)
    brow_mask = (
        smoothstep(0.011, 0.015, ax)
        * (1.0 - smoothstep(0.044, 0.051, ax))
        * gauss(cz, brow_arch_z, 0.0026)
        * front_mask
    )
    base_r = base_r * (1.0 - 0.56 * brow_mask) + 0.24 * (0.56 * brow_mask)
    base_g = base_g * (1.0 - 0.56 * brow_mask) + 0.15 * (0.56 * brow_mask)
    base_b = base_b * (1.0 - 0.56 * brow_mask) + 0.12 * (0.56 * brow_mask)

    # Nostril cavity dark shading (|cx|~0.0075, cz~1.503)
    nostril_dark = gauss(ax, 0.0075, 0.0030) * gauss(cz, 1.5032, 0.0024) * front_mask
    base_r *= 1.0 - 0.72 * nostril_dark
    base_g *= 1.0 - 0.78 * nostril_dark
    base_b *= 1.0 - 0.80 * nostril_dark

    # Natural rosy-peach lips with Cupid's bow and smiling parting line (Note: lower face lifted +5.5mm)
    lip_w = smoothstep(0.0268, 0.0195, ax) * front_mask
    cupid_z = 1.4908 - 0.0014 * gauss(ax, 0.0, 0.0022) + 0.0010 * gauss(ax, 0.0050, 0.0025) - 0.0055 * ((ax / 0.024) ** 2)
    lower_z = 1.4772 + 0.0060 * ((ax / 0.024) ** 2)
    lip_mask = smoothstep(lower_z - 0.0018, lower_z + 0.0016, cz) * (1.0 - smoothstep(cupid_z - 0.0014, cupid_z + 0.0016, cz)) * lip_w

    is_upper_lip = smoothstep(1.4832, 1.4855, cz)
    lip_r = 0.735 * is_upper_lip + 0.815 * (1.0 - is_upper_lip)
    lip_g = 0.365 * is_upper_lip + 0.435 * (1.0 - is_upper_lip)
    lip_b = 0.355 * is_upper_lip + 0.425 * (1.0 - is_upper_lip)

    base_r = base_r * (1.0 - 0.85 * lip_mask) + lip_r * (0.85 * lip_mask)
    base_g = base_g * (1.0 - 0.85 * lip_mask) + lip_g * (0.85 * lip_mask)
    base_b = base_b * (1.0 - 0.85 * lip_mask) + lip_b * (0.85 * lip_mask)

    smile_seam_z = 1.4838 + 2.2 * (ax ** 2)
    lip_seam = gauss(cz, smile_seam_z, 0.0010) * smoothstep(0.0275, 0.020, ax) * front_mask
    base_r = base_r * (1.0 - 0.68 * lip_seam) + 0.28 * (0.68 * lip_seam)
    base_g = base_g * (1.0 - 0.68 * lip_seam) + 0.11 * (0.68 * lip_seam)
    base_b = base_b * (1.0 - 0.68 * lip_seam) + 0.11 * (0.68 * lip_seam)

    # --- 1B. Landmark-Registered Photographic Facial Texture Extraction & De-Hairing / Delighting ---
    # Pre-clean reference.png across the facial region by computing a smooth local 82nd-percentile skin envelope
    # so dark bang strands on the forehead and hair along the cheeks are completely removed from skin pixels!
    grid_ny, grid_nx = 28, 28
    skin_env = np.zeros((grid_ny, grid_nx, 3), dtype=np.float32)
    default_skin = np.array([0.960, 0.785, 0.735], dtype=np.float32)
    for gy in range(grid_ny):
        y0_i = int((gy / grid_ny) * ref_h)
        y1_i = min(ref_h, int(((gy + 1.5) / grid_ny) * ref_h))
        for gx in range(grid_nx):
            x0_i = int((gx / grid_nx) * ref_w)
            x1_i = min(ref_w, int(((gx + 1.5) / grid_nx) * ref_w))
            blk = ref_srgb[y0_i:y1_i, x0_i:x1_i]
            blk_l = ref_luma[y0_i:y1_i, x0_i:x1_i]
            valid = (blk_l > 0.56) & ((blk[:, :, 0] - blk[:, :, 2]) > 0.10)
            if np.count_nonzero(valid) > 12:
                skin_env[gy, gx] = np.percentile(blk[valid], 80, axis=0)
            else:
                skin_env[gy, gx] = default_skin

    # Smooth the coarse skin envelope grid
    for _ in range(3):
        p_env = np.pad(skin_env, ((1, 1), (1, 1), (0, 0)), mode="edge")
        skin_env = (
            0.40 * skin_env
            + 0.15 * (p_env[:-2, 1:-1] + p_env[2:, 1:-1] + p_env[1:-1, :-2] + p_env[1:-1, 2:])
        )

    # Fit 25-landmark Thin-Plate Spline (TPS) warp from canonical (theta, cz) on FaceCylUV to reference.png (u_ref, v_top_ref)
    def canon_theta(x_c, y_c):
        return math.atan2(x_c, -(y_c - (-0.055)))

    lm_pairs = [
        ((-0.0308, -0.1380, 1.5452), (0.4240, 0.2990)),  # Eye_R_Center
        ((-0.0175, -0.1370, 1.5425), (0.4430, 0.2950)),  # Eye_R_Inner
        ((-0.0440, -0.1310, 1.5450), (0.4050, 0.3030)),  # Eye_R_Outer
        ((-0.0308, -0.1415, 1.5510), (0.4210, 0.2890)),  # Eye_R_Upper
        ((-0.0308, -0.1405, 1.5395), (0.4270, 0.3090)),  # Eye_R_Lower
        (( 0.0308, -0.1380, 1.5452), (0.5355, 0.2390)),  # Eye_L_Center
        (( 0.0175, -0.1370, 1.5425), (0.5140, 0.2520)),  # Eye_L_Inner
        (( 0.0440, -0.1310, 1.5450), (0.5560, 0.2280)),  # Eye_L_Outer
        (( 0.0308, -0.1415, 1.5510), (0.5310, 0.2290)),  # Eye_L_Upper
        (( 0.0308, -0.1405, 1.5395), (0.5390, 0.2490)),  # Eye_L_Lower
        ((-0.0310, -0.1475, 1.5635), (0.4100, 0.2680)),  # Brow_R_Mid
        (( 0.0310, -0.1475, 1.5635), (0.5250, 0.1980)),  # Brow_L_Mid
        (( 0.0000, -0.1480, 1.5440), (0.4820, 0.2740)),  # Nasion
        (( 0.0000, -0.1605, 1.5110), (0.5100, 0.3400)),  # Nose_Tip
        ((-0.0095, -0.1530, 1.5060), (0.4980, 0.3520)),  # Nostril_R
        (( 0.0095, -0.1530, 1.5060), (0.5335, 0.3445)),  # Nostril_L
        (( 0.0000, -0.1545, 1.4905), (0.5240, 0.3680)),  # Lip_Upper_Peak
        (( 0.0000, -0.1530, 1.4840), (0.5300, 0.3845)),  # Mouth_Center
        ((-0.0250, -0.1420, 1.4865), (0.4880, 0.3960)),  # Mouth_Corner_R
        (( 0.0250, -0.1420, 1.4865), (0.5680, 0.3650)),  # Mouth_Corner_L
        (( 0.0000, -0.1510, 1.4770), (0.5360, 0.4020)),  # Lip_Lower_Bot
        (( 0.0000, -0.1380, 1.4450), (0.5550, 0.4430)),  # Chin_Bottom
        ((-0.0420, -0.1250, 1.5170), (0.4450, 0.3450)),  # Cheek_R
        (( 0.0420, -0.1250, 1.5170), (0.5750, 0.2850)),  # Cheek_L
        (( 0.0000, -0.1480, 1.5900), (0.4600, 0.2150)),  # Forehead_Mid
    ]
    src_tps = np.array([[canon_theta(c[0], c[1]), (c[2] - 1.50) * 5.0] for c, _ in lm_pairs], dtype=np.float64)
    dst_tps = np.array([r for _, r in lm_pairs], dtype=np.float64)
    n_lm = len(src_tps)
    d_lm = src_tps[:, None, :] - src_tps[None, :, :]
    r2_lm = np.sum(d_lm ** 2, axis=-1)
    K_lm = r2_lm * np.log(r2_lm + 1e-6)
    P_lm = np.hstack([np.ones((n_lm, 1)), src_tps])
    L_lm = np.zeros((n_lm + 3, n_lm + 3), dtype=np.float64)
    L_lm[:n_lm, :n_lm] = K_lm + np.eye(n_lm) * 1.5e-4
    L_lm[:n_lm, n_lm:] = P_lm
    L_lm[n_lm:, :n_lm] = P_lm.T
    W_tps = np.linalg.solve(L_lm, np.vstack([dst_tps, np.zeros((3, 2))]))

    # Evaluate TPS warp on the active front-face region (|theta| < 0.72 rad, cz in [1.425, 1.615])
    active_face = (np.abs(theta) < 0.72) & (cz > 1.425) & (cz < 1.615)
    th_act = theta[active_face]
    cz_act = cz[active_face]
    cx_act = cx[active_face]
    ax_act = np.abs(cx_act)

    q_pts = np.column_stack([th_act, (cz_act - 1.50) * 5.0])
    dq = q_pts[:, None, :] - src_tps[None, :, :]
    rq2 = np.sum(dq ** 2, axis=-1)
    Kq = rq2 * np.log(rq2 + 1e-6)
    Pq = np.column_stack([np.ones(len(q_pts)), q_pts])
    uv_ref = Kq @ W_tps[:n_lm] + Pq @ W_tps[n_lm:]

    u_ref = np.clip(uv_ref[:, 0], 0.34, 0.65)
    v_ref = np.clip(uv_ref[:, 1], 0.15, 0.46)

    # Bilinear sample from ref_srgb and from the clean skin_env grid
    px_f = u_ref * (ref_w - 1)
    py_f = v_ref * (ref_h - 1)
    ix0 = np.clip(px_f.astype(int), 0, ref_w - 2)
    iy0 = np.clip(py_f.astype(int), 0, ref_h - 2)
    wx = (px_f - ix0)[:, None]
    wy = (py_f - iy0)[:, None]
    sampled_photo = (
        (1.0 - wx) * (1.0 - wy) * ref_srgb[iy0, ix0]
        + wx * (1.0 - wy) * ref_srgb[iy0, ix0 + 1]
        + (1.0 - wx) * wy * ref_srgb[iy0 + 1, ix0]
        + wx * wy * ref_srgb[iy0 + 1, ix0 + 1]
    )

    gx_f = u_ref * (grid_nx - 1)
    gy_f = v_ref * (grid_ny - 1)
    gix0 = np.clip(gx_f.astype(int), 0, grid_nx - 2)
    giy0 = np.clip(gy_f.astype(int), 0, grid_ny - 2)
    gwx = (gx_f - gix0)[:, None]
    gwy = (gy_f - giy0)[:, None]
    sampled_env = (
        (1.0 - gwx) * (1.0 - gwy) * skin_env[giy0, gix0]
        + gwx * (1.0 - gwy) * skin_env[giy0, gix0 + 1]
        + (1.0 - gwx) * gwy * skin_env[giy0 + 1, gix0]
        + gwx * gwy * skin_env[giy0 + 1, gix0 + 1]
    )

    # Feature preservation mask in canonical space (brows, outer eyelid creases, nose tip/nostrils, lips)
    # Outside these features (e.g. forehead, temples, cheeks), de-hair any dark bang/hair strands!
    in_brows = gauss(ax_act, 0.030, 0.016) * gauss(cz_act, 1.562, 0.0045)
    in_lids = gauss(ax_act, 0.031, 0.014) * gauss(cz_act, 1.546, 0.0085)
    in_nose = gauss(ax_act, 0.000, 0.016) * gauss(cz_act, 1.507, 0.0075)
    in_lips = gauss(ax_act, 0.000, 0.025) * gauss(cz_act, 1.484, 0.0110)
    feat_keep = np.clip(in_brows + in_lids + in_nose + in_lips, 0.0, 1.0)[:, None]

    s_luma = 0.2126 * sampled_photo[:, 0] + 0.7152 * sampled_photo[:, 1] + 0.0722 * sampled_photo[:, 2]
    e_luma = 0.2126 * sampled_env[:, 0] + 0.7152 * sampled_env[:, 1] + 0.0722 * sampled_env[:, 2]
    # Detect dark hair wisps crossing forehead/cheeks
    hair_contamin = smoothstep(0.025, 0.085, (e_luma - s_luma))[:, None] * (1.0 - feat_keep)
    clean_photo = sampled_photo * (1.0 - hair_contamin) + sampled_env * hair_contamin

    # Delight low-frequency illumination gradient by normalizing skin envelope toward warm peachy target (0.822, 0.616, 0.546)
    target_skin = np.array([0.822, 0.616, 0.546], dtype=np.float32)
    illum_ratio = np.clip(target_skin / np.maximum(sampled_env, 0.35), 0.72, 1.16)
    # Apply 68% delighting on skin while keeping rich contrast on lips, nostrils, brows, and lash lines
    delight_Photo = np.clip(clean_photo * (0.32 + 0.68 * illum_ratio), 0.0, 1.0)

    # Strict Eye-Aperture Exclusion: inside the palpebral opening where 3D Eyeball_L/R sit,
    # set photo blend weight to 0.0 so zero sclera/iris from reference.png is ever painted onto CharacterHead!
    eye_aperture_excl = np.zeros_like(th_act)
    for side_s in [-1.0, 1.0]:
        dx_e = (cx_act - side_s * 0.0308) / 0.0135
        dz_e = (cz_act - 1.5450) / 0.0058
        r_e = np.sqrt(dx_e ** 2 + dz_e ** 2)
        eye_aperture_excl = np.maximum(eye_aperture_excl, 1.0 - smoothstep(0.72, 1.12, r_e))

    # Smooth feather weight across the front face (tapers smoothly to 0 right above the eyebrows at cz=1.555..1.576
    # so the open forehead above the eyebrows uses 100% clean, un-shadowed peachy skin with zero bang residue)
    w_theta = 1.0 - smoothstep(0.38, 0.64, np.abs(th_act))
    w_cz_low = smoothstep(1.432, 1.465, cz_act)
    w_cz_high = 1.0 - smoothstep(1.554, 1.576, cz_act)
    w_photo = (0.78 * w_theta * w_cz_low * w_cz_high * (1.0 - eye_aperture_excl)).astype(np.float32)

    base_r[active_face] = base_r[active_face] * (1.0 - w_photo) + delight_Photo[:, 0] * w_photo
    base_g[active_face] = base_g[active_face] * (1.0 - w_photo) + delight_Photo[:, 1] * w_photo
    base_b[active_face] = base_b[active_face] * (1.0 - w_photo) + delight_Photo[:, 2] * w_photo

    # Extract high-frequency photographic micro-relief (pores, lip creases, eyelid folds) for SkinRoughBumpTex
    photo_micro = np.zeros_like(U, dtype=np.float32)
    clean_luma = 0.2126 * clean_photo[:, 0] + 0.7152 * clean_photo[:, 1] + 0.0722 * clean_photo[:, 2]
    photo_micro[active_face] = np.clip((clean_luma - e_luma) * 2.2, -0.24, 0.24) * w_photo

    # Soft hairline shadow ONLY high on the crown dome (cz > 1.648 on front, cz > 1.622 on sides/back)
    # with zero step function so the entire forehead skin is 100% continuous and un-banded
    hairline_z0 = 1.622 + 0.026 * front_mask
    scalp_dark = smoothstep(hairline_z0, hairline_z0 + 0.024, cz)
    base_r = base_r * (1.0 - 0.50 * scalp_dark) + 0.24 * (0.50 * scalp_dark)
    base_g = base_g * (1.0 - 0.50 * scalp_dark) + 0.14 * (0.50 * scalp_dark)
    base_b = base_b * (1.0 - 0.50 * scalp_dark) + 0.10 * (0.50 * scalp_dark)

    skin_rgba = np.ones((res_skin, res_skin, 4), dtype=np.float32)
    skin_rgba[:, :, 0] = np.clip(base_r, 0.0, 1.0)
    skin_rgba[:, :, 1] = np.clip(base_g, 0.0, 1.0)
    skin_rgba[:, :, 2] = np.clip(base_b, 0.0, 1.0)

    img_skin_albedo = bpy.data.images.new("SkinAlbedoTex", width=res_skin, height=res_skin, alpha=False)
    img_skin_albedo.colorspace_settings.name = "sRGB"
    img_skin_albedo.pixels.foreach_set(skin_rgba.ravel())
    img_skin_albedo.pack()

    # SkinRoughBumpTex: R = photo micro-relief + micro-pore + lip vertical grooves height, G = roughness, B = SSS weight
    pore_noise = rng.normal(0.0, 1.0, (res_skin, res_skin)).astype(np.float32)
    pore_zone = (gauss(cz, 1.515, 0.025) * gauss(ax, 0.022, 0.028) + 0.45) * front_mask
    lip_grooves = np.sin(theta * 420.0) * np.cos(V * 90.0) * lip_mask
    height_map = 0.5 + 0.14 * pore_noise * pore_zone + 0.24 * lip_grooves - 0.25 * lip_seam + 0.48 * photo_micro

    # Roughness: ~0.32 on nose tip & lips (natural healthy sheen), ~0.43 on cheeks/forehead (soft matte skin, not plastic)
    rough_map = (
        0.43
        - 0.11 * lip_mask
        - 0.06 * nose_warm
        - 0.04 * blush
        + 0.05 * (1.0 - smoothstep(1.38, 1.44, cz))
        + 0.025 * pore_noise
    )
    sss_map = 1.0 - 0.25 * lip_mask

    rb_rgba = np.ones((res_skin, res_skin, 4), dtype=np.float32)
    rb_rgba[:, :, 0] = np.clip(height_map, 0.0, 1.0)
    rb_rgba[:, :, 1] = np.clip(rough_map, 0.18, 0.85)
    rb_rgba[:, :, 2] = np.clip(sss_map, 0.0, 1.0)

    img_skin_rb = bpy.data.images.new("SkinRoughBumpTex", width=res_skin, height=res_skin, alpha=False)
    img_skin_rb.colorspace_settings.name = "Non-Color"
    img_skin_rb.pixels.foreach_set(rb_rgba.ravel())
    img_skin_rb.pack()

    # --- 2. EyeIrisTex (1024x1024, sRGB, rich dark chestnut-brown iris with crisp catchlight & zero upper sclera) ---
    res_eye = 1024
    ey_grid = np.linspace(-1.0, 1.0, res_eye, endpoint=False)
    EX, EY = np.meshgrid(ey_grid, ey_grid)
    ER = np.sqrt(EX ** 2 + EY ** 2)
    EA = np.arctan2(EY, EX)

    # Full 11.8mm iris diameter (r_iris=0.365) and expressive dark pupil (r_pupil=0.152)
    r_iris = 0.365
    r_pupil = 0.152

    fibers = (
        0.50 * np.sin(EA * 72.0 + 1.5 * np.sin(ER * 45.0))
        + 0.30 * np.cos(EA * 135.0 - 2.0 * np.cos(ER * 60.0))
        + 0.20 * np.sin(EA * 210.0)
    )
    r_norm = np.clip((ER - r_pupil) / (r_iris - r_pupil + 1e-6), 0.0, 1.0)
    collarette = gauss(r_norm, 0.34, 0.16)
    limbal = smoothstep(0.64, 0.96, r_norm)

    # Deep warm espresso-chestnut brown matching East Asian iris in reference.png
    iris_r = (0.23 + 0.11 * collarette + 0.05 * fibers) * (1.0 - 0.78 * limbal)
    iris_g = (0.11 + 0.06 * collarette + 0.025 * fibers) * (1.0 - 0.82 * limbal)
    iris_b = (0.06 + 0.03 * collarette + 0.015 * fibers) * (1.0 - 0.85 * limbal)

    scl_warm = smoothstep(r_iris, 0.85, ER)
    vein = np.maximum(0.0, np.sin(EA * 19.0 + ER * 18.0) * np.cos(EA * 31.0 - ER * 12.0)) * scl_warm
    scl_r = 0.880 - 0.05 * scl_warm
    scl_g = 0.835 - 0.09 * scl_warm - 0.05 * vein
    scl_b = 0.815 - 0.10 * scl_warm - 0.06 * vein

    is_pupil = 1.0 - smoothstep(r_pupil - 0.012, r_pupil + 0.012, ER)
    is_sclera = smoothstep(r_iris - 0.015, r_iris + 0.018, ER)
    is_iris = (1.0 - is_pupil) * (1.0 - is_sclera)

    # Add a crisp, natural upper-left window catchlight reflection on the upper-left iris/pupil margin
    catch_1 = np.exp(-(((EX + 0.105) / 0.038) ** 2 + ((EY - 0.105) / 0.032) ** 2))
    catch_2 = 0.45 * np.exp(-(((EX - 0.120) / 0.022) ** 2 + ((EY + 0.085) / 0.020) ** 2))
    catchlight = np.clip(catch_1 + catch_2, 0.0, 1.0)

    eye_rgba = np.ones((res_eye, res_eye, 4), dtype=np.float32)
    eye_rgba[:, :, 0] = np.clip(0.012 * is_pupil + iris_r * is_iris + scl_r * is_sclera + 0.92 * catchlight, 0.0, 1.0)
    eye_rgba[:, :, 1] = np.clip(0.010 * is_pupil + iris_g * is_iris + scl_g * is_sclera + 0.92 * catchlight, 0.0, 1.0)
    eye_rgba[:, :, 2] = np.clip(0.010 * is_pupil + iris_b * is_iris + scl_b * is_sclera + 0.92 * catchlight, 0.0, 1.0)

    img_eye = bpy.data.images.new("EyeIrisTex", width=res_eye, height=res_eye, alpha=False)
    img_eye.colorspace_settings.name = "sRGB"
    img_eye.pixels.foreach_set(eye_rgba.ravel())
    img_eye.pack()

    # --- 3. HairStrandTex (1024x1024, sRGB, multi-strand fibrous texture sampled from reference.png hair) ---
    res_hair = 1024
    hu = np.linspace(0.0, 1.0, res_hair, endpoint=False)
    HU, HV = np.meshgrid(hu, hu)
    # Multi-frequency longitudinal hair fibers along V (with slight wave in U)
    u_waved = HU + 0.008 * np.sin(HV * 6.0 * math.pi)
    strand_f = (
        0.45 * np.sin(u_waved * 180.0 * 2.0 * math.pi)
        + 0.35 * np.cos(u_waved * 420.0 * 2.0 * math.pi + HV * 8.0)
        + 0.20 * np.sin(u_waved * 860.0 * 2.0 * math.pi)
    )
    root_to_tip = smoothstep(0.0, 0.85, HV)
    # Warm chestnut root (0.245, 0.142, 0.098) to sunlit caramel-chestnut highlight (0.365, 0.220, 0.150) in sRGB
    # matching Mat_HairStrands so HairVolumeLocks and HairScalpCap blend seamlessly without dark root banding
    hr_r = 0.245 + 0.120 * root_to_tip + 0.055 * strand_f
    hr_g = 0.142 + 0.078 * root_to_tip + 0.032 * strand_f
    hr_b = 0.098 + 0.052 * root_to_tip + 0.022 * strand_f

    hair_rgba = np.ones((res_hair, res_hair, 4), dtype=np.float32)
    hair_rgba[:, :, 0] = np.clip(hr_r, 0.02, 1.0)
    hair_rgba[:, :, 1] = np.clip(hr_g, 0.01, 1.0)
    hair_rgba[:, :, 2] = np.clip(hr_b, 0.01, 1.0)

    img_hair = bpy.data.images.new("HairStrandTex", width=res_hair, height=res_hair, alpha=False)
    img_hair.colorspace_settings.name = "sRGB"
    img_hair.pixels.foreach_set(hair_rgba.ravel())
    img_hair.pack()

    # --- 4. SweaterKnitHeightTex & SweaterKnitAlbedoTex (2048x2048, hybrid photo-sampled + seed stitch) ---
    res_knit = 2048 if not PREVIEW_MODE else 1024
    ku = np.linspace(0.0, 1.0, res_knit, endpoint=False)
    KU, KV = np.meshgrid(ku, ku)

    # Extract clean unoccluded sweater patch from reference.png (x in [0.63, 0.77], y_top in [0.59, 0.73])
    sw_patch = ref_srgb[int(0.59 * ref_h):int(0.73 * ref_h), int(0.63 * ref_w):int(0.77 * ref_w)].copy()
    # Remove low-frequency lighting gradient across patch to get pure photographic knit variation
    sw_mean_rgb = sw_patch.mean(axis=(0, 1))
    sw_row_grad = sw_patch.mean(axis=1, keepdims=True)
    sw_col_grad = sw_patch.mean(axis=0, keepdims=True)
    sw_flat = sw_patch - (sw_row_grad - sw_mean_rgb) * 0.8 - (sw_col_grad - sw_mean_rgb) * 0.8
    # Make 2x2 mirror-tiled patch so it tiles 100% seamlessly across KnitUV
    sw_row0 = np.concatenate([sw_flat, sw_flat[:, ::-1, :]], axis=1)
    sw_row1 = np.concatenate([sw_flat[::-1, :, :], sw_flat[::-1, ::-1, :]], axis=1)
    sw_tile_2x2 = np.concatenate([sw_row0, sw_row1], axis=0)
    th_p, tw_p = sw_tile_2x2.shape[:2]
    # Tile 4x4 times across the 2048x2048 KnitUV space
    py_k = ((KV * 4.0) % 1.0 * (th_p - 1)).astype(int)
    px_k = ((KU * 4.0) % 1.0 * (tw_p - 1)).astype(int)
    photo_knit_rgb = sw_tile_2x2[py_k, px_k]
    photo_knit_luma = (
        0.2126 * photo_knit_rgb[:, :, 0]
        + 0.7152 * photo_knit_rgb[:, :, 1]
        + 0.0722 * photo_knit_rgb[:, :, 2]
    )
    photo_knit_detail = np.clip((photo_knit_luma - photo_knit_luma.mean()) * 2.2, -0.18, 0.18)

    # Procedural seed/waffle stitch structure (64 cols x 80 rows per UV tile)
    n_cols = 64.0
    n_rows = 80.0
    cx_k = KU * n_cols
    cy_k = KV * n_rows

    col_idx = np.floor(cx_k)
    row_idx = np.floor(cy_k)
    fx = cx_k - col_idx - 0.5
    fy = cy_k - row_idx - 0.5

    parity = np.mod(col_idx + row_idx, 2.0)
    v_leg_dist = np.abs(fy - (0.35 - 1.35 * np.abs(fx)))
    knit_loop_h = np.exp(-((v_leg_dist / 0.28) ** 2)) * np.exp(-((fx / 0.46) ** 4))
    purl_bump_h = np.exp(-((fx / 0.42) ** 2 + (fy / 0.30) ** 2))
    stitch_h = parity * purl_bump_h + (1.0 - parity) * knit_loop_h

    yarn_twist = 0.06 * np.sin((cx_k * 6.0 + cy_k * (4.0 * parity - 2.0)) * 2.0 * math.pi)
    yarn_fuzz = rng.normal(0.0, 0.022, (res_knit, res_knit)).astype(np.float32)

    knit_height = np.clip(0.20 + 0.68 * stitch_h + 0.25 * photo_knit_detail + yarn_twist + yarn_fuzz, 0.0, 1.0)

    kh_rgba = np.ones((res_knit, res_knit, 4), dtype=np.float32)
    kh_rgba[:, :, 0] = knit_height
    kh_rgba[:, :, 1] = knit_height
    kh_rgba[:, :, 2] = knit_height

    img_knit_h = bpy.data.images.new("SweaterKnitHeightTex", width=res_knit, height=res_knit, alpha=False)
    img_knit_h.colorspace_settings.name = "Non-Color"
    img_knit_h.pixels.foreach_set(kh_rgba.ravel())
    img_knit_h.pack()

    # Dusty pink sweater albedo calibrated directly to reference.png sweater chromaticity + seamless photo knit grain
    ao = 0.64 + 0.36 * smoothstep(0.12, 0.88, knit_height)
    heather = 0.012 * np.sin(KU * 120.0 * math.pi) * np.cos(KV * 140.0 * math.pi) + 0.010 * yarn_fuzz
    photo_chroma_dev = photo_knit_rgb - sw_mean_rgb[None, None, :]
    sw_r = np.clip((0.840 + 0.35 * photo_chroma_dev[:, :, 0] + heather) * (0.76 + 0.24 * ao), 0.0, 1.0)
    sw_g = np.clip((0.555 + 0.35 * photo_chroma_dev[:, :, 1] + heather * 0.7) * (0.72 + 0.28 * ao), 0.0, 1.0)
    sw_b = np.clip((0.548 + 0.35 * photo_chroma_dev[:, :, 2] + heather * 0.7) * (0.72 + 0.28 * ao), 0.0, 1.0)

    ka_rgba = np.ones((res_knit, res_knit, 4), dtype=np.float32)
    ka_rgba[:, :, 0] = sw_r
    ka_rgba[:, :, 1] = sw_g
    ka_rgba[:, :, 2] = sw_b

    img_knit_a = bpy.data.images.new("SweaterKnitAlbedoTex", width=res_knit, height=res_knit, alpha=False)
    img_knit_a.colorspace_settings.name = "sRGB"
    img_knit_a.pixels.foreach_set(ka_rgba.ravel())
    img_knit_a.pack()

    knit_disp_tex = bpy.data.textures.new("SweaterKnitDispTex", type="IMAGE")
    knit_disp_tex.image = img_knit_h

    return {
        "skin_albedo": img_skin_albedo,
        "skin_rb": img_skin_rb,
        "eye_iris": img_eye,
        "hair_strand": img_hair,
        "knit_height": img_knit_h,
        "knit_albedo": img_knit_a,
        "knit_disp_tex": knit_disp_tex,
    }


def build_materials(tex_dict):
    """
    Creates physically based Cycles node materials for skin, eyes, hair, lashes/brows, and sweater.
    """
    # --- 1. Mat_Skin (Subsurface Scattering + micro-pore bump + roughness variation) ---
    mat_skin = bpy.data.materials.new("Mat_Skin")
    nt = mat_skin.node_tree
    nodes = nt.nodes
    links = nt.links
    bsdf = nodes["Principled BSDF"]

    uv_node = nodes.new("ShaderNodeUVMap")
    uv_node.uv_map = "FaceCylUV"

    tex_alb = nodes.new("ShaderNodeTexImage")
    tex_alb.image = tex_dict["skin_albedo"]
    tex_alb.extension = "REPEAT"

    tex_rb = nodes.new("ShaderNodeTexImage")
    tex_rb.image = tex_dict["skin_rb"]
    tex_rb.extension = "REPEAT"

    sep = nodes.new("ShaderNodeSeparateColor")
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.0012

    links.new(uv_node.outputs["UV"], tex_alb.inputs["Vector"])
    links.new(uv_node.outputs["UV"], tex_rb.inputs["Vector"])
    links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex_rb.outputs["Color"], sep.inputs["Color"])
    links.new(sep.outputs["Red"], bump.inputs["Height"])
    links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Controlled Subsurface Scattering so 3D facial contours remain crisp while skin glows warmly
    bsdf.inputs["Subsurface Weight"].default_value = 0.16
    bsdf.inputs["Subsurface Scale"].default_value = 0.0045
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.36, 0.22)
    bsdf.inputs["Subsurface Anisotropy"].default_value = 0.35
    bsdf.inputs["IOR"].default_value = 1.40
    bsdf.inputs["Specular IOR Level"].default_value = 0.32
    bsdf.inputs["Coat Weight"].default_value = 0.03
    bsdf.inputs["Coat Roughness"].default_value = 0.34

    # --- 2. Mat_EyeIrisSclera, Mat_Cornea, Mat_Tearline ---
    mat_iris = bpy.data.materials.new("Mat_EyeIrisSclera")
    nt_i = mat_iris.node_tree
    bsdf_i = nt_i.nodes["Principled BSDF"]
    tex_eye = nt_i.nodes.new("ShaderNodeTexImage")
    tex_eye.image = tex_dict["eye_iris"]
    nt_i.links.new(tex_eye.outputs["Color"], bsdf_i.inputs["Base Color"])
    # Inner iris/pupil is inside aqueous humor; outer Cornea provides grazing specular reflection
    bsdf_i.inputs["Roughness"].default_value = 0.90
    bsdf_i.inputs["Specular IOR Level"].default_value = 0.02
    bsdf_i.inputs["Subsurface Weight"].default_value = 0.05
    bsdf_i.inputs["Subsurface Scale"].default_value = 0.002

    # Mat_Cornea: low-glare Fresnel (IOR 1.12) so large area lights don't fog the dark pupil
    mat_cornea = bpy.data.materials.new("Mat_Cornea")
    nt_c = mat_cornea.node_tree
    nt_c.nodes.clear()
    out_c = nt_c.nodes.new("ShaderNodeOutputMaterial")
    lp = nt_c.nodes.new("ShaderNodeLightPath")
    fresnel = nt_c.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.12
    transp = nt_c.nodes.new("ShaderNodeBsdfTransparent")
    glossy = nt_c.nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs["Roughness"].default_value = 0.004

    mix_fresnel = nt_c.nodes.new("ShaderNodeMixShader")
    nt_c.links.new(fresnel.outputs["Fac"], mix_fresnel.inputs["Fac"])
    nt_c.links.new(transp.outputs["BSDF"], mix_fresnel.inputs[1])
    nt_c.links.new(glossy.outputs["BSDF"], mix_fresnel.inputs[2])

    mix_non_cam = nt_c.nodes.new("ShaderNodeMixShader")
    nt_c.links.new(lp.outputs["Is Camera Ray"], mix_non_cam.inputs["Fac"])
    nt_c.links.new(transp.outputs["BSDF"], mix_non_cam.inputs[1])
    nt_c.links.new(mix_fresnel.outputs["Shader"], mix_non_cam.inputs[2])
    nt_c.links.new(mix_non_cam.outputs["Shader"], out_c.inputs["Surface"])

    # Mat_Tearline: wet translucent meniscus
    mat_tear = bpy.data.materials.new("Mat_Tearline")
    bsdf_t = mat_tear.node_tree.nodes["Principled BSDF"]
    bsdf_t.inputs["Base Color"].default_value = (0.72, 0.34, 0.30, 1.0)
    bsdf_t.inputs["Roughness"].default_value = 0.03
    bsdf_t.inputs["Alpha"].default_value = 0.22
    bsdf_t.inputs["Coat Weight"].default_value = 0.9
    bsdf_t.inputs["Coat Roughness"].default_value = 0.02

    # --- 3. Mat_Hair_Strands (Principled Hair BSDF calibrated for rich warm chestnut brown) ---
    mat_hair_strands = bpy.data.materials.new("Mat_Hair_Strands")
    nt_hs = mat_hair_strands.node_tree
    nt_hs.nodes.clear()
    out_hs = nt_hs.nodes.new("ShaderNodeOutputMaterial")
    hair_bsdf = nt_hs.nodes.new("ShaderNodeBsdfHairPrincipled")
    hair_bsdf.parametrization = "COLOR"
    hair_bsdf.model = "CHIANG"

    hi = nt_hs.nodes.new("ShaderNodeHairInfo")
    ramp = nt_hs.nodes.new("ShaderNodeValToRGB")
    elems = ramp.color_ramp.elements
    elems[0].position = 0.0
    elems[0].color = (0.014, 0.0045, 0.0020, 1.0)  # Deep espresso-chestnut root
    elems[1].position = 0.55
    elems[1].color = (0.038, 0.0130, 0.0055, 1.0)  # Rich warm chestnut mid-shaft
    e2 = ramp.color_ramp.elements.new(0.92)
    e2.color = (0.076, 0.0270, 0.0115, 1.0)        # Warm copper-chestnut sunlit highlight

    math_add = nt_hs.nodes.new("ShaderNodeMath")
    math_add.operation = "ADD"
    math_mul = nt_hs.nodes.new("ShaderNodeMath")
    math_mul.operation = "MULTIPLY"
    math_mul.inputs[1].default_value = 0.30

    nt_hs.links.new(hi.outputs["Random"], math_mul.inputs[0])
    nt_hs.links.new(hi.outputs["Intercept"], math_add.inputs[0])
    nt_hs.links.new(math_mul.outputs["Value"], math_add.inputs[1])
    nt_hs.links.new(math_add.outputs["Value"], ramp.inputs["Fac"])
    nt_hs.links.new(ramp.outputs["Color"], hair_bsdf.inputs["Color"])

    hair_bsdf.inputs["Roughness"].default_value = 0.27
    hair_bsdf.inputs["Radial Roughness"].default_value = 0.36
    hair_bsdf.inputs["Coat"].default_value = 0.12
    hair_bsdf.inputs["IOR"].default_value = 1.55
    hair_bsdf.inputs["Random Roughness"].default_value = 0.08
    nt_hs.links.new(hair_bsdf.outputs["BSDF"], out_hs.inputs["Surface"])

    # --- 4. Mat_Hair_Locks & Mat_Scalp (Multi-strand textured warm chestnut shader for 3D locks & scalp) ---
    mat_hair_locks = bpy.data.materials.new("Mat_Hair_Locks")
    nt_hl = mat_hair_locks.node_tree
    bsdf_hl = nt_hl.nodes["Principled BSDF"]
    uv_hl = nt_hl.nodes.new("ShaderNodeUVMap")

    tex_hair = nt_hl.nodes.new("ShaderNodeTexImage")
    tex_hair.image = tex_dict["hair_strand"]
    tex_hair.extension = "REPEAT"

    # Fine longitudinal strand bump along UV.x
    wave_hl = nt_hl.nodes.new("ShaderNodeTexWave")
    wave_hl.wave_type = "BANDS"
    wave_hl.bands_direction = "X"
    wave_hl.inputs["Scale"].default_value = 210.0
    wave_hl.inputs["Distortion"].default_value = 3.0
    bump_hl = nt_hl.nodes.new("ShaderNodeBump")
    bump_hl.inputs["Strength"].default_value = 0.52
    bump_hl.inputs["Distance"].default_value = 0.0025

    nt_hl.links.new(uv_hl.outputs["UV"], tex_hair.inputs["Vector"])
    nt_hl.links.new(tex_hair.outputs["Color"], bsdf_hl.inputs["Base Color"])
    nt_hl.links.new(uv_hl.outputs["UV"], wave_hl.inputs["Vector"])
    nt_hl.links.new(wave_hl.outputs["Fac"], bump_hl.inputs["Height"])
    nt_hl.links.new(bump_hl.outputs["Normal"], bsdf_hl.inputs["Normal"])

    bsdf_hl.inputs["Roughness"].default_value = 0.50
    bsdf_hl.inputs["Anisotropic"].default_value = 0.58
    bsdf_hl.inputs["Anisotropic Rotation"].default_value = 0.25
    bsdf_hl.inputs["Specular IOR Level"].default_value = 0.11
    bsdf_hl.inputs["Specular Tint"].default_value = (0.72, 0.38, 0.18, 1.0)
    bsdf_hl.inputs["Sheen Weight"].default_value = 0.14
    bsdf_hl.inputs["Sheen Roughness"].default_value = 0.52
    bsdf_hl.inputs["Sheen Tint"].default_value = (0.30, 0.12, 0.06, 1.0)

    mat_scalp = bpy.data.materials.new("Mat_Scalp")
    bsdf_sc = mat_scalp.node_tree.nodes["Principled BSDF"]
    bsdf_sc.inputs["Base Color"].default_value = (0.016, 0.0055, 0.0025, 1.0)
    bsdf_sc.inputs["Roughness"].default_value = 0.72

    # --- 5. Mat_LashesBrows (Dark warm espresso-brown for 3D eyelashes & eyebrows) ---
    mat_lash_brow = bpy.data.materials.new("Mat_LashesBrows")
    bsdf_lb = mat_lash_brow.node_tree.nodes["Principled BSDF"]
    bsdf_lb.inputs["Base Color"].default_value = (0.022, 0.009, 0.005, 1.0)
    bsdf_lb.inputs["Roughness"].default_value = 0.36

    # --- 6. Mat_Sweater & Mat_SweaterCollar (Dusty pink chunky knit with 3D relief + fuzzy wool sheen) ---
    def make_sweater_material(name, uv_scale_x, uv_scale_y):
        mat = bpy.data.materials.new(name)
        nt_s = mat.node_tree
        bsdf_s = nt_s.nodes["Principled BSDF"]

        uv_s = nt_s.nodes.new("ShaderNodeUVMap")
        uv_s.uv_map = "KnitUV"
        mapping = nt_s.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (uv_scale_x, uv_scale_y, 1.0)

        tex_ka = nt_s.nodes.new("ShaderNodeTexImage")
        tex_ka.image = tex_dict["knit_albedo"]
        tex_ka.extension = "REPEAT"

        tex_kh = nt_s.nodes.new("ShaderNodeTexImage")
        tex_kh.image = tex_dict["knit_height"]
        tex_kh.extension = "REPEAT"

        bump_s = nt_s.nodes.new("ShaderNodeBump")
        bump_s.inputs["Strength"].default_value = 0.42
        bump_s.inputs["Distance"].default_value = 0.0038

        nt_s.links.new(uv_s.outputs["UV"], mapping.inputs["Vector"])
        nt_s.links.new(mapping.outputs["Vector"], tex_ka.inputs["Vector"])
        nt_s.links.new(mapping.outputs["Vector"], tex_kh.inputs["Vector"])
        nt_s.links.new(tex_ka.outputs["Color"], bsdf_s.inputs["Base Color"])
        nt_s.links.new(tex_kh.outputs["Color"], bump_s.inputs["Height"])
        nt_s.links.new(bump_s.outputs["Normal"], bsdf_s.inputs["Normal"])

        bsdf_s.inputs["Roughness"].default_value = 0.82
        bsdf_s.inputs["Diffuse Roughness"].default_value = 0.65
        bsdf_s.inputs["Specular IOR Level"].default_value = 0.14
        bsdf_s.inputs["Sheen Weight"].default_value = 0.28
        bsdf_s.inputs["Sheen Roughness"].default_value = 0.62
        bsdf_s.inputs["Sheen Tint"].default_value = (0.85, 0.46, 0.46, 1.0)
        bsdf_s.inputs["Subsurface Weight"].default_value = 0.04
        bsdf_s.inputs["Subsurface Scale"].default_value = 0.003
        bsdf_s.inputs["Subsurface Radius"].default_value = (0.95, 0.45, 0.45)
        return mat

    mat_sweater = make_sweater_material("Mat_Sweater", 1.0, 1.0)
    mat_collar = make_sweater_material("Mat_SweaterCollar", 2.2, 0.45)

    return {
        "skin": mat_skin,
        "iris": mat_iris,
        "cornea": mat_cornea,
        "tear": mat_tear,
        "hair_strands": mat_hair_strands,
        "hair_locks": mat_hair_locks,
        "scalp": mat_scalp,
        "lash_brow": mat_lash_brow,
        "sweater": mat_sweater,
        "collar": mat_collar,
    }


def build_cafe_environment_and_lighting(env_col, light_col):
    """
    Constructs a genuine 3D softly lit modern cafe background behind the character (y > 0.5m)
    matching input/reference.png:
    - Large bright daylight window on viewer's upper-left (x < -0.15)
    - Lush green potted cafe plants on viewer's upper-left (x in [-0.75, -0.18], z in [1.35, 1.95])
    - Warm wooden cafe table in mid-left background (x ~ -0.45, z ~ 0.92..1.18)
    - Warm cream/wood cafe counter, shelves, coffee equipment & glass jars on viewer's right (x > 0.20)
    Plus soft natural portrait lighting.
    """
    def make_simple_mat(name, color_srgb, rough=0.55, metallic=0.0, emission_strength=0.0, transmission=0.0):
        m = bpy.data.materials.new(name)
        b = m.node_tree.nodes["Principled BSDF"]
        lin = srgb_to_linear(np.array(color_srgb[:3]))
        b.inputs["Base Color"].default_value = (float(lin[0]), float(lin[1]), float(lin[2]), 1.0)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metallic
        if emission_strength > 0.0:
            b.inputs["Emission Color"].default_value = (float(lin[0]), float(lin[1]), float(lin[2]), 1.0)
            b.inputs["Emission Strength"].default_value = emission_strength
        if transmission > 0.0:
            b.inputs["Transmission Weight"].default_value = transmission
            b.inputs["Roughness"].default_value = 0.08
        return m

    mat_wall = make_simple_mat("Mat_CafeWall", (0.86, 0.80, 0.72), rough=0.75)
    mat_window = make_simple_mat("Mat_CafeWindow", (0.95, 0.97, 0.92), rough=0.2, emission_strength=2.2)
    mat_wood_warm = make_simple_mat("Mat_CafeWood", (0.72, 0.49, 0.32), rough=0.45)
    mat_wood_light = make_simple_mat("Mat_CafeWoodLight", (0.82, 0.67, 0.50), rough=0.50)
    mat_leaf_1 = make_simple_mat("Mat_CafePlantLeaf1", (0.56, 0.66, 0.32), rough=0.42)
    mat_leaf_2 = make_simple_mat("Mat_CafePlantLeaf2", (0.68, 0.76, 0.42), rough=0.45)
    mat_pot = make_simple_mat("Mat_CafePot", (0.88, 0.85, 0.80), rough=0.35)
    mat_metal = make_simple_mat("Mat_CafeMetal", (0.78, 0.76, 0.74), rough=0.25, metallic=0.85)
    mat_glass = make_simple_mat("Mat_CafeGlass", (0.92, 0.94, 0.96), rough=0.08, transmission=0.85)

    def add_box(name, loc, scale, mat, rot=(0.0, 0.0, 0.0)):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        me = bpy.data.meshes.new(f"{name}_Mesh")
        bm.to_mesh(me)
        bm.free()
        obj = bpy.data.objects.new(name, me)
        obj.location = loc
        obj.scale = scale
        obj.rotation_euler = rot
        obj.data.materials.append(mat)
        env_col.objects.link(obj)
        return obj

    # Back cafe wall & ceiling soffit
    add_box("CafeWall_Back", (0.0, 2.35, 1.45), (3.6, 0.08, 2.2), mat_wall)
    # Bright cafe window on viewer-left (x < -0.05, z > 1.35)
    add_box("CafeWindow_Left", (-0.65, 2.15, 1.78), (1.35, 0.04, 0.95), mat_window)

    # Potted green plants in upper-left background (creates soft green foliage bokeh in upper-left)
    rng = np.random.default_rng(77)
    bm_leaves = bmesh.new()
    plant_centers = [(-0.55, 1.45, 1.62), (-0.34, 1.55, 1.68), (-0.72, 1.40, 1.52), (-0.22, 1.65, 1.75)]
    for pc in plant_centers:
        for _ in range(22):
            off = rng.normal(0.0, [0.11, 0.08, 0.10])
            ret = bmesh.ops.create_icosphere(bm_leaves, subdivisions=1, radius=1.0)
            s_v = ret["verts"]
            ang = rng.uniform(0, 3.14, 3)
            rm = Euler(ang.tolist(), "XYZ").to_matrix()
            for v in s_v:
                p = Vector((v.co.x * 0.065, v.co.y * 0.040, v.co.z * 0.022))
                v.co = Vector(pc) + Vector(off.tolist()) + (rm @ p)
    me_pl = bpy.data.meshes.new("CafePlantsMesh")
    bm_leaves.to_mesh(me_pl)
    bm_leaves.free()
    for p in me_pl.polygons:
        p.use_smooth = True
    obj_pl = bpy.data.objects.new("CafePlants", me_pl)
    obj_pl.data.materials.append(mat_leaf_1)
    env_col.objects.link(obj_pl)

    # Warm wooden cafe table & chair backs in mid-left background (viewer-left, z ~ 1.0..1.35)
    add_box("CafeTable_Left", (-0.52, 1.25, 1.12), (0.65, 0.45, 0.05), mat_wood_warm, rot=(0.0, 0.0, 0.25))
    add_box("CafeChair_Left", (-0.42, 1.10, 1.24), (0.32, 0.04, 0.28), mat_wood_warm, rot=(0.0, 0.0, -0.35))
    add_box("CafeCup_Left", (-0.55, 1.18, 1.18), (0.09, 0.09, 0.11), mat_pot)

    # Right-side cafe counter, wooden shelves, coffee machine & glass jars (viewer-right, x > 0.20)
    add_box("CafeCounter_Right", (0.58, 1.55, 1.08), (0.95, 0.48, 0.55), mat_wood_light)
    add_box("CafeShelf_UpperRight", (0.55, 1.85, 1.62), (0.90, 0.24, 0.04), mat_wood_warm)
    add_box("CafeShelf_MidRight", (0.55, 1.85, 1.38), (0.90, 0.24, 0.04), mat_wood_warm)
    add_box("CafeEspressoMachine", (0.38, 1.50, 1.44), (0.26, 0.24, 0.24), mat_metal)
    add_box("CafeJar_1", (0.56, 1.78, 1.48), (0.10, 0.10, 0.15), mat_glass)
    add_box("CafeJar_2", (0.72, 1.78, 1.48), (0.11, 0.11, 0.16), mat_wood_warm)
    add_box("CafeJar_3", (0.48, 1.78, 1.72), (0.12, 0.12, 0.16), mat_wood_light)

    # --- Natural Soft Cafe Portrait Lighting ---
    def add_area_light(name, loc, target, energy, size_x, size_y, color):
        ld = bpy.data.lights.new(name=name, type="AREA")
        ld.energy = energy
        ld.shape = "RECTANGLE"
        ld.size = size_x
        ld.size_y = size_y
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        direction = Vector(target) - Vector(loc)
        lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        light_col.objects.link(lo)
        return lo

    # 1. Key Light: Large soft daylight window from viewer's left / character's right-front
    add_area_light(
        "KeyLight_Window",
        loc=(-0.95, -1.05, 1.78),
        target=(0.0, -0.08, 1.42),
        energy=42.0,
        size_x=1.35,
        size_y=1.45,
        color=(1.0, 0.97, 0.93),
    )

    # 2. Fill Light: Gentle warm cafe interior bounce from front-right so facial shading has depth
    add_area_light(
        "FillLight_CafeBounce",
        loc=(0.95, -1.15, 1.45),
        target=(0.0, -0.08, 1.38),
        energy=18.5,
        size_x=1.45,
        size_y=1.45,
        color=(1.0, 0.94, 0.89),
    )

    # 3. Top/Back Hair Rim & Turnaround Fill Light: illuminates crown highlights and back diagnostic view
    add_area_light(
        "RimLight_TopBack",
        loc=(0.15, 1.15, 2.05),
        target=(0.0, 0.0, 1.38),
        energy=25.0,
        size_x=1.35,
        size_y=1.25,
        color=(1.0, 0.95, 0.90),
    )

    # 4. Gentle lower bounce light (simulating light bouncing off the light-wood cafe tabletop)
    add_area_light(
        "BounceLight_Table",
        loc=(0.0, -0.85, 0.72),
        target=(0.0, -0.10, 1.35),
        energy=8.5,
        size_x=1.15,
        size_y=0.85,
        color=(1.0, 0.93, 0.89),
    )

    # World shader: soft warm-neutral studio/cafe ambient environment
    world = bpy.data.worlds.new("CafeWorld")
    bpy.context.scene.world = world
    w_nt = world.node_tree
    bg = w_nt.nodes["Background"]
    bg.inputs["Color"].default_value = (0.50, 0.47, 0.45, 1.0)
    bg.inputs["Strength"].default_value = 0.46


def setup_cameras(cam_col):
    """
    Creates BeautyCamera (matching reference.png selfie framing, head tilt, and shallow DoF)
    and the 4 diagnostic cameras (Cam_Front, Cam_ThreeQuarter, Cam_Side, Cam_Back) at 0, 45, 90, 180 deg.
    """
    # --- 1. BeautyCamera (960x1040, portrait framing matching reference.png landmarks & shallow DoF) ---
    cd_beauty = bpy.data.cameras.new("BeautyCameraData")
    cd_beauty.lens = 42.0
    cd_beauty.sensor_width = 36.0
    cd_beauty.dof.use_dof = True
    cd_beauty.dof.aperture_fstop = 1.8

    cam_beauty = bpy.data.objects.new("BeautyCamera", cd_beauty)
    cam_pos = Vector((-0.056, -0.745, 1.464))
    cam_target = Vector((-0.058, -0.080, 1.403))
    cam_beauty.location = cam_pos

    # Compute focus distance to the character's eyes (~1.545m Z)
    eye_mid_posed, _ = pose_head_rigid_point((0.0, -0.138, 1.545))
    cd_beauty.dof.focus_distance = (Vector(eye_mid_posed.tolist()) - cam_pos).length

    # Aim at cam_target with a subtle -3.5 deg camera roll complementing the 21.5 deg 3D head tilt
    look_quat = (cam_target - cam_pos).to_track_quat("-Z", "Y")
    roll_quat = Euler((0.0, 0.0, math.radians(-3.5)), "XYZ").to_quaternion()
    cam_beauty.rotation_euler = (look_quat @ roll_quat).to_euler()
    cam_col.objects.link(cam_beauty)

    # --- 2. Diagnostic Cameras (768x768 at 0, 45, 90, 180 degrees around the character) ---
    # Center of upper-body asset is at (0.0, -0.03, 1.335), spanning z in [0.94, 1.69] (height ~0.75m)
    diag_target = Vector((0.0, -0.03, 1.335))
    diag_dist = 1.52
    diag_specs = [
        ("Cam_Front", 0.0, "front.png"),
        ("Cam_ThreeQuarter", 45.0, "three-quarter.png"),
        ("Cam_Side", 90.0, "side.png"),
        ("Cam_Back", 180.0, "back.png"),
    ]
    diag_cams = []
    for c_name, deg, fname in diag_specs:
        cd = bpy.data.cameras.new(f"{c_name}Data")
        cd.lens = 58.0
        cd.sensor_width = 36.0
        cd.dof.use_dof = False
        co = bpy.data.objects.new(c_name, cd)
        rad = math.radians(deg)
        # 0 deg = front (0, -dist), 45 deg = (+X/-Y), 90 deg = side (+X, 0), 180 deg = back (0, +dist)
        loc = Vector((
            diag_target.x + diag_dist * math.sin(rad),
            diag_target.y - diag_dist * math.cos(rad),
            diag_target.z + 0.03,
        ))
        co.location = loc
        co.rotation_euler = (diag_target - loc).to_track_quat("-Z", "Y").to_euler()
        cam_col.objects.link(co)
        diag_cams.append((co, fname, deg))

    return cam_beauty, diag_cams, np.array(cam_pos[:], dtype=np.float64)


def configure_render_settings():
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.render.threads_mode = "FIXED"
    scn.render.threads = 3
    scn.cycles.samples = 16 if PREVIEW_MODE else 32
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = "OPENIMAGEDENOISE"
    scn.cycles.max_bounces = 6
    scn.cycles.diffuse_bounces = 3
    scn.cycles.glossy_bounces = 3
    scn.cycles.transmission_bounces = 6
    scn.cycles.transparent_max_bounces = 8
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.color_depth = "8"
    scn.view_settings.view_transform = "AgX" if "AgX" in [
        i.identifier for i in scn.view_settings.bl_rna.properties["view_transform"].enum_items
    ] else "Filmic"
    scn.view_settings.look = "None"
    scn.view_settings.exposure = 0.05


def compute_scene_report(char_col, render_files):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    base_verts = 0
    base_faces = 0
    eval_verts = 0
    eval_faces = 0
    curve_objects = []
    total_curves = 0
    total_curve_points = 0
    bmin = np.array([1e9, 1e9, 1e9], dtype=np.float64)
    bmax = np.array([-1e9, -1e9, -1e9], dtype=np.float64)
    obj_names = []

    for obj in char_col.objects:
        obj_names.append(obj.name)
        if obj.type == "MESH":
            base_verts += len(obj.data.vertices)
            base_faces += len(obj.data.polygons)
            eval_obj = obj.evaluated_get(depsgraph)
            eval_me = eval_obj.to_mesh()
            eval_verts += len(eval_me.vertices)
            eval_faces += len(eval_me.polygons)
            if len(eval_me.vertices) > 0:
                coords = np.zeros(len(eval_me.vertices) * 3, dtype=np.float32)
                eval_me.vertices.foreach_get("co", coords)
                coords = coords.reshape((-1, 3))
                bmin = np.minimum(bmin, coords.min(axis=0))
                bmax = np.maximum(bmax, coords.max(axis=0))
            eval_obj.to_mesh_clear()
        elif obj.type == "CURVES":
            n_splines = len(obj.data.curves)
            n_pts = len(obj.data.points)
            total_curves += n_splines
            total_curve_points += n_pts
            curve_objects.append({
                "name": obj.name,
                "curves": int(n_splines),
                "controlPoints": int(n_pts),
            })
            if n_pts > 0:
                coords = np.zeros(n_pts * 3, dtype=np.float32)
                obj.data.points.foreach_get("position", coords)
                coords = coords.reshape((-1, 3))
                bmin = np.minimum(bmin, coords.min(axis=0))
                bmax = np.maximum(bmax, coords.max(axis=0))

    scn = bpy.context.scene
    report = {
        "blenderVersion": bpy.app.version_string,
        "renderEngine": scn.render.engine,
        "renderDevice": scn.cycles.device,
        "threads": int(scn.render.threads),
        "samples": int(scn.cycles.samples),
        "denoiser": scn.cycles.denoiser if scn.cycles.use_denoising else None,
        "importantObjectNames": sorted(obj_names),
        "characterBounds": {
            "min": [round(float(x), 4) for x in bmin],
            "max": [round(float(x), 4) for x in bmax],
            "dimensions": [round(float(x), 4) for x in (bmax - bmin)],
        },
        "meshCounts": {
            "baseVertices": int(base_verts),
            "baseFaces": int(base_faces),
            "evaluatedVertices": int(eval_verts),
            "evaluatedFaces": int(eval_faces),
            "evaluatedVerticesPlusCurveControlPoints": int(eval_verts + total_curve_points),
        },
        "hairCurveCounts": {
            "totalCurveObjects": len(curve_objects),
            "totalCurves": int(total_curves),
            "totalControlPoints": int(total_curve_points),
            "objects": curve_objects,
        },
        "generatedTextures": [
            {
                "name": "SkinAlbedoTex",
                "resolution": list(bpy.data.images["SkinAlbedoTex"].size),
                "colorspace": "sRGB",
                "packed": bool(bpy.data.images["SkinAlbedoTex"].packed_file is not None),
                "source": "Hybrid: input/reference.png facial photograph + procedural skin synthesis",
                "mappingMethod": "25-landmark Thin-Plate Spline (TPS) warp onto canonical FaceCylUV with 80th-percentile local skin envelope de-hairing (removing forehead/cheek hair wisps), 68% low-frequency delighting, strict eye-aperture exclusion, and cosine-feathered blending into 360-degree color-matched procedural skin on forehead, temples, ears, neck, and back of head.",
                "uvMap": "FaceCylUV",
            },
            {
                "name": "SkinRoughBumpTex",
                "resolution": list(bpy.data.images["SkinRoughBumpTex"].size),
                "colorspace": "Non-Color",
                "packed": bool(bpy.data.images["SkinRoughBumpTex"].packed_file is not None),
                "source": "Hybrid: input/reference.png high-pass facial micro-relief + procedural micro-pores & lip grooves",
                "mappingMethod": "Packed channel map on FaceCylUV: R = TPS-warped photographic high-pass skin/lip relief + procedural pores & lip grooves, G = spatially varying roughness (0.32 lips/nose tip, 0.43 cheeks/forehead, 0.48 neck), B = subsurface scattering weight.",
                "uvMap": "FaceCylUV",
            },
            {
                "name": "EyeIrisTex",
                "resolution": list(bpy.data.images["EyeIrisTex"].size),
                "colorspace": "sRGB",
                "packed": bool(bpy.data.images["EyeIrisTex"].packed_file is not None),
                "source": "Calibrated from input/reference.png iris pixel sampling + procedural radial iris synthesis",
                "mappingMethod": "Planar polar UV mapping on Eyeball_L / Eyeball_R (r_iris=0.365, r_pupil=0.152) with warm chestnut stroma fibers, collarette ring, dark limbal ring, catchlights, and warm off-white sclera.",
                "uvMap": "UVMap",
            },
            {
                "name": "HairStrandTex",
                "resolution": list(bpy.data.images["HairStrandTex"].size),
                "colorspace": "sRGB",
                "packed": bool(bpy.data.images["HairStrandTex"].packed_file is not None),
                "source": "Calibrated from input/reference.png espresso root, warm chestnut mid-shaft, and copper-brown highlight pixels",
                "mappingMethod": "Longitudinal UV mapping along HairVolumeLocks and ScalpCap with multi-frequency fibrous strand striations and root-to-tip tonal gradient.",
                "uvMap": "UVMap",
            },
            {
                "name": "SweaterKnitHeightTex",
                "resolution": list(bpy.data.images["SweaterKnitHeightTex"].size),
                "colorspace": "Non-Color",
                "packed": bool(bpy.data.images["SweaterKnitHeightTex"].packed_file is not None),
                "source": "Hybrid: unoccluded sweater patch (x in [0.63,0.77], y in [0.59,0.73]) from input/reference.png + procedural seed/waffle stitch heightmap",
                "mappingMethod": "2x2 mirror-tiled delighted photographic yarn relief combined with 64x80 interlocking seed/waffle stitch heightmap on KnitUV, driving both Subdivision+Displace modifier and shader Bump.",
                "uvMap": "KnitUV",
            },
            {
                "name": "SweaterKnitAlbedoTex",
                "resolution": list(bpy.data.images["SweaterKnitAlbedoTex"].size),
                "colorspace": "sRGB",
                "packed": bool(bpy.data.images["SweaterKnitAlbedoTex"].packed_file is not None),
                "source": "Hybrid: input/reference.png dusty-pink sweater chromaticity & mirror-tiled yarn grain + procedural stitch crevice AO",
                "mappingMethod": "Seamless UV mapping on KnitUV across torso, shoulders, and sleeves with physically calibrated dusty-rose yarn chromaticity.",
                "uvMap": "KnitUV",
            },
        ],
        "renderedFilenames": render_files,
        "limitations": [
            "Single-image reference means exact side profile, back hair arrangement, back sweater drapery, and unseen lower torso/hands are anatomically inferred.",
            "Facial likeness is sculpted by deforming the CC0 MakeHuman hm08 base topology combined with landmark-registered TPS photographic albedo projection rather than a multi-camera 3D photogrammetry scan.",
            "Hair combines 78 fluted 8-sided 3D wave locks with 5,700 native 3D HairCurves strands (including 380 see-through bang and curtain-hairline curves) rather than >100,000 individual scalp follicles to stay well within the <1M evaluated vertex and CPU render budget.",
            "Cafe background is a stylized 3D geometric representation (window, plants, wooden tables, counter, and jars) rendered with shallow depth of field.",
        ],
    }
    report_path = os.path.join(WORKDIR, "scene-report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        f"Wrote scene report: mesh_eval_verts={eval_verts}, mesh_eval_faces={eval_faces}, "
        f"hair_curves={total_curves}, hair_pts={total_curve_points}"
    )


def main():
    t0 = time.time()
    clear_scene()

    scn = bpy.context.scene
    char_col = bpy.data.collections.new("Character")
    env_col = bpy.data.collections.new("Environment")
    light_col = bpy.data.collections.new("Lighting")
    cam_col = bpy.data.collections.new("Cameras")
    for col in [char_col, env_col, light_col, cam_col]:
        scn.collection.children.link(col)

    tex_dict = generate_and_pack_textures()
    mats = build_materials(tex_dict)
    cam_beauty, diag_cams, cam_pos = setup_cameras(cam_col)

    _obj_head, _obj_sw, head_posed_v, sw_posed_v, sw_posed_normals, collar_posed_v = build_head_and_sweater_meshes(
        char_col,
        mats["skin"],
        mats["sweater"],
        mats["collar"],
        tex_dict["knit_disp_tex"],
    )
    build_eyes_lashes_brows(
        char_col,
        mats["iris"],
        mats["cornea"],
        mats["tear"],
        mats["lash_brow"],
        cam_pos,
    )
    build_hair_system(
        char_col,
        mats["scalp"],
        mats["hair_locks"],
        mats["hair_strands"],
        head_posed_v,
        sw_posed_v,
        sw_posed_normals,
        collar_posed_v,
    )
    build_cafe_environment_and_lighting(env_col, light_col)
    configure_render_settings()

    # Set BeautyCamera active and save character.blend BEFORE rendering
    scn.camera = cam_beauty
    scn.render.resolution_x = 480 if PREVIEW_MODE else 960
    scn.render.resolution_y = 520 if PREVIEW_MODE else 1040

    os.makedirs(os.path.join(WORKDIR, "renders"), exist_ok=True)
    blend_path = os.path.join(WORKDIR, "character.blend")
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved {blend_path} in {time.time() - t0:.1f}s")

    render_files = [
        "renders/beauty.png",
        "renders/front.png",
        "renders/three-quarter.png",
        "renders/side.png",
        "renders/back.png",
    ]
    compute_scene_report(char_col, render_files)

    # Render beauty.png (with 3D cafe environment visible)
    env_col.hide_render = False
    scn.camera = cam_beauty
    scn.render.resolution_x = 480 if PREVIEW_MODE else 960
    scn.render.resolution_y = 520 if PREVIEW_MODE else 1040
    scn.render.filepath = os.path.join(WORKDIR, "renders", "beauty.png")
    bpy.ops.render.render(write_still=True)

    # Render 4 diagnostic views (0, 45, 90, 180 deg) with plain studio background
    # Keep character geometry and materials 100% identical
    env_col.hide_render = True
    diag_res = 384 if PREVIEW_MODE else 768
    scn.render.resolution_x = diag_res
    scn.render.resolution_y = diag_res
    for co, fname, _deg in diag_cams:
        scn.camera = co
        scn.render.filepath = os.path.join(WORKDIR, "renders", fname)
        bpy.ops.render.render(write_still=True)

    # Restore BeautyCamera & Cafe environment and re-save so character.blend opens on beauty view
    env_col.hide_render = False
    scn.camera = cam_beauty
    scn.render.resolution_x = 960
    scn.render.resolution_y = 1040
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"All renders complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

