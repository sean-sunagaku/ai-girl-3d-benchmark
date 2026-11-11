import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from lib_anatomy import pose_point, pose_points_array


def _build_scalp_volume_core(anat_info, mat_core):
    """
    Build a recessed dark-chestnut inner scalp cap strictly over the posterior/crown skull
    (staying smoothly behind y >= -0.105 so zero dark patches ever touch forehead, temples, or ears!).
    """
    bm = bmesh.new()
    c_skull = np.array([0.0, -0.048, 1.560], dtype=np.float64)
    rx, ry, rz = 0.0725, 0.0865, 0.0915

    n_lat = 22
    n_lon = 40
    grid = []
    for i in range(n_lat + 1):
        v_frac = i / n_lat
        phi = v_frac * (math.pi * 0.62)
        row = []
        for j in range(n_lon):
            u_frac = j / n_lon
            theta = -0.25 * math.pi + u_frac * (1.50 * math.pi)
            sp, cp = math.sin(phi), math.cos(phi)
            ct, st = math.cos(theta), math.sin(theta)

            px = c_skull[0] + rx * sp * ct
            py = max(-0.104, c_skull[1] + ry * sp * st)
            pz = c_skull[2] + rz * cp

            # Keep completely behind exposed left ear (+X > 0.056, z < 1.575)
            if px > 0.056 and pz < 1.575:
                py = max(py, -0.018)

            pw = pose_point([px, py, pz], neck_weight=1.0)
            row.append(bm.verts.new(pw))
        grid.append(row)

    for i in range(n_lat):
        for j in range(n_lon - 1):
            f = bm.faces.new((grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]))
            f.smooth = True

    mesh = bpy.data.meshes.new("Hair_Volume_Core_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Hair_Volume_Core", mesh)
    obj.data.materials.append(mat_core)
    sub = obj.modifiers.new(name="CoreSubsurf", type='SUBSURF')
    sub.levels = 1
    sub.render_levels = 1
    return obj


def _push_outside_colliders(pt, head_bvh, sweater_bvhs, head_clear=0.0030, sw_clear=0.0038):
    v = Vector((float(pt[0]), float(pt[1]), float(pt[2])))

    if v.z > 1.335:
        loc, norm, _, dist = head_bvh.find_nearest(v)
        if loc is not None and dist < 0.042:
            delta = v - loc
            if delta.dot(norm) < head_clear:
                v = loc + norm * head_clear

    if v.z < 1.405:
        for sbvh in sweater_bvhs:
            loc, norm, _, dist = sbvh.find_nearest(v)
            if loc is not None and dist < 0.045:
                delta = v - loc
                if delta.dot(norm) < sw_clear:
                    v = loc + norm * sw_clear

    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _smooth_polyline(pts, passes=2, pin_root=2):
    out = pts.copy()
    for _ in range(passes):
        nxt = out.copy()
        for i in range(pin_root, len(out) - 1):
            nxt[i] = 0.25 * out[i - 1] + 0.50 * out[i] + 0.25 * out[i + 1]
        out = nxt
    return out


def _snap_scalp_root_rest(rest_verts, rx, ry, rz):
    c_skull = np.array([0.0, -0.056, 1.558], dtype=np.float64)
    d2 = np.sum((rest_verts - np.array([rx, ry, rz])) ** 2, axis=1)
    idx = np.argsort(d2)[:10]
    w = np.exp(-d2[idx] / (0.008 ** 2)) + 1e-6
    surf = np.sum(rest_verts[idx] * w[:, None], axis=0) / np.sum(w)
    nrm = surf - c_skull
    nrm /= max(1e-6, np.linalg.norm(nrm))
    return surf, nrm


def _create_curves_obj(name, pts_list, rad_list, mat):
    hc = bpy.data.hair_curves.new(name + "_Data")
    hc.add_curves([len(p) for p in pts_list])
    hc.position_data.foreach_set("vector", np.concatenate(pts_list, axis=0).astype(np.float32).ravel())
    hc.points.foreach_set("radius", np.concatenate(rad_list, axis=0).astype(np.float32).ravel())
    hc.set_types(type='CATMULL_ROM')
    hc.materials.append(mat)
    return bpy.data.objects.new(name, hc)


def build_character_hair(char_col, anat_info, sweater_info, mat_hair, mat_core):
    """
    Build seamless, skull-hugging, wavy 3D Chestnut Hair Curves matching input/reference.png:
    1. Delicate airy see-through bangs + soft side-swept wisps (`Character_Hair_Bangs`)
    2. Fluffy right-side (-X, viewer's left) face-framing & bold S-wave chest cascade (`Character_Hair_Cascade_Right`)
    3. Continuous skull-hugging left-side (+X, viewer's right) temple-to-behind-ear drape (`Character_Hair_Drape_Left`)
    4. Full layered wavy back hair (`Character_Hair_Back`)
    5. Sunlit fine crown & silhouette halo flyaways (`Character_Hair_Flyaways`)
    """
    from lib_sweater import _catmull_rom_spline

    deps = bpy.context.evaluated_depsgraph_get()
    head_bvh = BVHTree.FromObject(anat_info["head_obj"], deps)
    sweater_bvhs = [BVHTree.FromObject(ob, deps) for ob in sweater_info["all_sweater_meshes"]]

    rest_verts = anat_info["rest_verts"]
    rng = np.random.default_rng(2026)

    part_x = -0.012

    # =========================================================================
    # 1. DELICATE AIRY SEE-THROUGH BANGS & FEATHERED FOREHEAD WISPS
    # =========================================================================
    bang_pts, bang_rad = [], []
    # 38 airy micro-wisps & soft side-curtain wisps fanning across the forehead & right temple (like reference.png!)
    for ci in range(38):
        u = ci / 37.0  # 0 (left temple -0.048) .. 1 (right temple curtain +0.072)
        rx_tip_bias = -0.048 + u * 0.120
        rx = part_x + (rx_tip_bias - part_x) * 0.48 + rng.uniform(-0.003, 0.003)
        ry = -0.132 - 0.010 * (1.0 - (rx / 0.058) ** 2) + rng.uniform(-0.003, 0.003)
        rz = 1.628 - 0.012 * (abs(rx) / 0.052) ** 1.4
        root_r, _ = _snap_scalp_root_rest(rest_verts, rx, ry, rz)

        n_seg = 14
        # Center wisps end near brow/lash; right-temple curtain wisps (u > 0.65) sweep longer toward upper left ear!
        right_curtain = max(0.0, (u - 0.62) / 0.38) ** 1.3
        left_curtain = max(0.0, (0.28 - u) / 0.28) ** 1.5
        drop_z = (root_r[2] - 1.552) + 0.032 * right_curtain + 0.020 * left_curtain + rng.uniform(-0.004, 0.005)
        sweep_dx = (rx_tip_bias - root_r[0]) + 0.014 * (u - 0.22)

        g_rest = np.zeros((n_seg, 3), dtype=np.float64)
        for k in range(n_seg):
            t = k / (n_seg - 1)
            px = root_r[0] + sweep_dx * (t ** 1.08)
            pz = root_r[2] + 0.003 * math.sin(math.pi * min(1.0, t * 2.0)) - drop_z * (t ** 1.12)
            # Curve smoothly back along temple as |px| approaches 0.072!
            f_y = -0.153 - 0.004 * math.sin(math.pi * t) + 0.022 * ((abs(px) / 0.055) ** 2.1)
            py = root_r[1] * (1.0 - min(1.0, t * 2.0)) + f_y * min(1.0, t * 2.0)
            g_rest[k] = [px, py, pz]

        guide_w = pose_points_array(g_rest, rigid_head=True)
        for k in range(2, n_seg):
            guide_w[k] = _push_outside_colliders(guide_w[k], head_bvh, sweater_bvhs, head_clear=0.0026, sw_clear=0.0035)
        guide_w = _smooth_polyline(guide_w, passes=2, pin_root=2)

        n_children = rng.integers(16, 24) if right_curtain < 0.1 else rng.integers(24, 34)
        t_arr = np.linspace(0.0, 1.0, n_seg)
        for _ in range(n_children):
            jit_r = rng.normal(0.0, 0.0028, size=3)
            jit_m = rng.normal(0.0, 0.0030, size=3)
            pinch = rng.uniform(0.35, 0.78)
            child = np.zeros_like(guide_w)
            for k in range(n_seg):
                t = k / (n_seg - 1)
                child[k] = guide_w[k] + jit_r * (1.0 - t) + jit_m * math.sin(math.pi * t) * (1.0 - t * pinch)
                if k >= 2:
                    child[k] = _push_outside_colliders(child[k], head_bvh, sweater_bvhs, head_clear=0.0022, sw_clear=0.0030)
            bang_pts.append(child)
            r0 = rng.uniform(0.000048, 0.000074)
            bang_rad.append(r0 * np.clip(1.0 - 0.86 * (t_arr ** 1.5), 0.10, 1.0))

    obj_bangs = _create_curves_obj("Character_Hair_Bangs", bang_pts, bang_rad, mat_hair)

    # =========================================================================
    # Helper: Grow skull-hugging upper hair + bold S-wave lower locks!
    # =========================================================================
    def grow_skull_hugging_clumps(clump_specs, children_range, r_range, seed_offset):
        sub_rng = np.random.default_rng(seed_offset)
        all_pts, all_rad = [], []
        n_seg = 26

        for spec in clump_specs:
            head_pts_w = pose_points_array(np.asarray(spec["head_pts_rest"], dtype=np.float64), rigid_head=True)
            world_tail = np.asarray(spec["world_tail"], dtype=np.float64)
            ctrl = np.vstack([head_pts_w, world_tail])

            guide = _catmull_rom_spline(ctrl, n_seg)

            # First resolve collisions & smooth the base trajectory BEFORE superimposing S-waves!
            for _ in range(2):
                for k in range(1, n_seg):
                    guide[k] = _push_outside_colliders(guide[k], head_bvh, sweater_bvhs, head_clear=0.0045, sw_clear=0.0065)
                guide = _smooth_polyline(guide, passes=1, pin_root=1)

            # Now superimpose lush 3D S-wave curls along the hanging shaft (t > 0.24) so waves stay vivid!
            wave_amp = spec.get("wave_amp", 0.026)
            wave_freq = spec.get("wave_freq", 2.3)
            wave_phase = spec.get("wave_phase", 0.0)

            tang = np.gradient(guide, axis=0)
            tang /= np.maximum(1e-6, np.linalg.norm(tang, axis=1, keepdims=True))
            side_v = np.cross(tang, np.array([0.0, 1.0, 0.0]))
            side_n = np.linalg.norm(side_v, axis=1, keepdims=True)
            side_v = np.where(side_n > 1e-4, side_v / np.maximum(1e-6, side_n), np.array([1.0, 0.0, 0.0]))
            perp_v = np.cross(tang, side_v)

            for k in range(5, n_seg):
                t = k / (n_seg - 1)
                env = math.sin(math.pi * min(1.0, (t - 0.18) / 0.78)) ** 0.75 if t > 0.18 else 0.0
                # Keep gentle curl right down to the tapered tips!
                tip_curl = 0.35 * max(0.0, (t - 0.70) / 0.30)
                w1 = wave_amp * (env + tip_curl) * math.sin(2.0 * math.pi * wave_freq * t + wave_phase)
                w2 = (wave_amp * 0.32) * env * math.cos(2.0 * math.pi * wave_freq * t + wave_phase)
                guide[k] += side_v[k] * w1 + perp_v[k] * w2
                guide[k] = _push_outside_colliders(guide[k], head_bvh, sweater_bvhs, head_clear=0.0040, sw_clear=0.0055)

            n_ch = sub_rng.integers(children_range[0], children_range[1])
            t_arr = np.linspace(0.0, 1.0, n_seg)
            for ch_i in range(n_ch):
                is_fine = (ch_i % 4 == 0)
                root_off = sub_rng.normal(0.0, 0.0055, size=3)
                mid_off = sub_rng.normal(0.0, 0.0110 if is_fine else 0.0060, size=3)
                ph_jit = sub_rng.uniform(-0.45, 0.45)
                len_trim = sub_rng.uniform(0.84, 1.0)

                child = np.zeros((n_seg, 3), dtype=np.float64)
                for k in range(n_seg):
                    t = (k / (n_seg - 1)) * len_trim
                    gf = t * (n_seg - 1)
                    gi = min(n_seg - 2, int(math.floor(gf)))
                    gt = gf - gi
                    gp = guide[gi] * (1.0 - gt) + guide[gi + 1] * gt

                    clump_env = (1.0 - 0.58 * (t ** 1.35)) if not is_fine else (0.90 + 0.65 * t)
                    micro_wave = 0.0038 * math.sin(12.0 * t + ph_jit * 6.0) * side_v[gi]
                    cp = gp + root_off * (1.0 - 0.55 * t) + mid_off * math.sin(math.pi * t) * clump_env + micro_wave
                    if k >= 1:
                        cp = _push_outside_colliders(cp, head_bvh, sweater_bvhs, head_clear=0.0028, sw_clear=0.0038)
                    child[k] = cp

                all_pts.append(child)
                r0 = sub_rng.uniform(r_range[0], r_range[1]) * (0.48 if is_fine else 1.0)
                all_rad.append(r0 * np.clip(1.0 - 0.85 * (t_arr ** 1.55), 0.10, 1.0))

        return all_pts, all_rad

    # =========================================================================
    # 2. VOLUMINOUS RIGHT-SIDE (-X, VIEWER'S LEFT) WAVY FACE-FRAMING & CHEST CASCADE
    # =========================================================================
    right_specs = []
    for i in range(54):
        u = i / 53.0  # seamless front hairline (-0.140) -> crown/back (+0.025)
        ry_root = -0.140 + u * 0.165 + rng.uniform(-0.003, 0.003)
        rx_root = part_x - 0.002 - 0.016 * ((i % 4) / 3.0)
        rz_root = 1.644 - 0.028 * ((ry_root + 0.045) / 0.105) ** 2
        r0, n0 = _snap_scalp_root_rest(rest_verts, rx_root, ry_root, rz_root)

        y_side = np.clip(ry_root + 0.006, -0.135, 0.040)
        # Exit tangentially at scalp height (r0[2]+4mm) so crown forms a smooth natural dome without a center V-trench!
        head_pts = [
            r0 - n0 * 0.001,
            [r0[0] - 0.022, y_side, r0[2] + 0.004],
            [-0.062 - 0.008 * (1.0 - u), y_side, r0[2] - 0.008],
            [-0.088 - 0.008 * math.sin(math.pi * u), y_side, 1.576],
            [-0.094 - 0.010 * (1.0 - u), y_side - 0.008, 1.502],
        ]

        chest_x = -0.048 - 0.108 * (u ** 0.88) + rng.uniform(-0.012, 0.012)
        end_z = 0.950 + 0.16 * ((i * 7) % 11) / 10.0
        # Explicit loose S-wave sway across the pink sweater chest (matching reference.png!)
        sway_sign = 1.0 if (i % 3 != 0) else -0.6
        if u < 0.74:
            world_tail = [
                [chest_x - 0.032, -0.128 + 0.055 * u, 1.355],
                [chest_x + 0.028 * sway_sign, -0.162 + 0.028 * u, 1.225],
                [chest_x - 0.024 * sway_sign, -0.158 + 0.024 * u, 1.100],
                [chest_x + 0.016 * sway_sign, -0.148 + 0.022 * u, end_z],
            ]
        else:
            su = (u - 0.74) / 0.26
            world_tail = [
                [-0.162 + 0.02 * su, -0.025 + 0.085 * su, 1.355],
                [-0.172 + 0.03 * su, 0.025 + 0.085 * su, 1.215],
                [-0.158 + 0.03 * su, 0.055 + 0.065 * su, end_z + 0.04],
            ]

        right_specs.append({
            "head_pts_rest": head_pts,
            "world_tail": world_tail,
            "wave_amp": rng.uniform(0.020, 0.034),
            "wave_freq": rng.uniform(2.0, 2.7),
            "wave_phase": (i * 0.55) + rng.uniform(-0.35, 0.35),
        })

    r_pts, r_rad = grow_skull_hugging_clumps(right_specs, (92, 115), (0.000125, 0.000180), seed_offset=1101)
    obj_right = _create_curves_obj("Character_Hair_Cascade_Right", r_pts, r_rad, mat_hair)

    # =========================================================================
    # 3. SEAMLESS LEFT-SIDE (+X, VIEWER'S RIGHT) TEMPLE-TO-BEHIND-EAR DRAPE
    # =========================================================================
    # Continuous progression from anterior hairline (-0.142) to back (+0.030):
    # ZERO gap on upper-right forehead/crown, sweeps diagonally over temple & tucks behind left ear!
    left_specs = []
    for i in range(48):
        u = i / 47.0  # 0 = front temple hairline (-0.142) .. 1 = mid-back crown (+0.030)
        ry_root = -0.142 + u * 0.172 + rng.uniform(-0.003, 0.003)
        rx_root = part_x + 0.003 + 0.016 * ((i % 4) / 3.0)
        rz_root = 1.644 - 0.028 * ((ry_root + 0.045) / 0.108) ** 2
        r0, n0 = _snap_scalp_root_rest(rest_verts, rx_root, ry_root, rz_root)

        # Drape naturally across right temple hairline (eliminating bare temple!) then tuck behind upper ear at z=1.564!
        y_dome = ry_root + 0.008
        y_temple = np.clip(ry_root + 0.032, -0.122, 0.045)
        y_ear_top = np.clip(ry_root + 0.072, -0.066, 0.055)
        y_behind_ear = max(-0.016, ry_root + 0.115)

        head_pts = [
            r0 - n0 * 0.001,
            [r0[0] + 0.024, y_dome, r0[2] + 0.004],
            [0.056 + 0.006 * u, y_temple, r0[2] - 0.012],
            [0.078, y_ear_top, 1.568],      # sweeps across temple right to top-front of ear helix!
            [0.078, y_behind_ear, 1.505],   # tucks cleanly behind exposed left ear pinna!
        ]

        # Flow gracefully down BEHIND her left neck & shoulder (leaving front right collar clean like reference.png!)
        if u < 0.35:
            fu = u / 0.35
            world_tail = [
                [0.096 + 0.022 * fu, -0.012 + 0.025 * fu, 1.385],
                [0.132 + 0.028 * fu, -0.032 + 0.035 * fu, 1.280],
                [0.148 + 0.025 * fu, -0.045 + 0.045 * fu, 1.125 - 0.04 * fu],
            ]
        else:
            bu = (u - 0.35) / 0.65
            world_tail = [
                [0.102 + 0.032 * bu, 0.028 + 0.045 * bu, 1.375],
                [0.130 + 0.032 * bu, 0.078 + 0.035 * bu, 1.230],
                [0.125 + 0.030 * bu, 0.102 + 0.022 * bu, 1.035 + 0.08 * ((i * 3) % 5) / 4.0],
            ]

        left_specs.append({
            "head_pts_rest": head_pts,
            "world_tail": world_tail,
            "wave_amp": rng.uniform(0.015, 0.025),
            "wave_freq": rng.uniform(2.0, 2.6),
            "wave_phase": (i * 0.6) + rng.uniform(-0.3, 0.3),
        })

    l_pts, l_rad = grow_skull_hugging_clumps(left_specs, (88, 110), (0.000125, 0.000175), seed_offset=2202)
    obj_left = _create_curves_obj("Character_Hair_Drape_Left", l_pts, l_rad, mat_hair)

    # =========================================================================
    # 4. FULL LAYERED WAVY BACK HAIR (`Character_Hair_Back`)
    # =========================================================================
    back_specs = []
    for i in range(52):
        col = i % 13
        row = i // 13
        u = (col / 12.0) * 2.0 - 1.0
        rx_root = u * (0.072 - 0.014 * row) + rng.uniform(-0.004, 0.004)
        rz_root = 1.630 - row * 0.042 - 0.016 * (u * u)
        ry_root = 0.015 + 0.028 * (1.0 - 0.35 * row) * (1.0 - 0.45 * u * u)
        r0, n0 = _snap_scalp_root_rest(rest_verts, rx_root, ry_root, rz_root)

        head_pts = [
            r0 - n0 * 0.001,
            [rx_root * 1.05, ry_root + 0.014, rz_root - 0.025],
            [rx_root * 1.08, 0.048 + 0.008 * (3 - row), 1.495 - 0.018 * row],
        ]
        spread_x = u * (0.148 - 0.010 * row) + rng.uniform(-0.010, 0.010)
        end_z = 0.980 + 0.13 * ((i * 5) % 9) / 8.0 - 0.02 * row
        world_tail = [
            [spread_x * 0.92 - 0.010, 0.088 + 0.010 * (3 - row), 1.365],
            [spread_x, 0.112 + 0.008 * (3 - row), 1.215],
            [spread_x * 1.04, 0.112, end_z],
        ]
        back_specs.append({
            "head_pts_rest": head_pts,
            "world_tail": world_tail,
            "wave_amp": rng.uniform(0.018, 0.028),
            "wave_freq": rng.uniform(2.0, 2.7),
            "wave_phase": rng.uniform(0.0, 2.0 * math.pi),
        })

    b_pts, b_rad = grow_skull_hugging_clumps(back_specs, (92, 114), (0.000130, 0.000185), seed_offset=3303)
    obj_back = _create_curves_obj("Character_Hair_Back", b_pts, b_rad, mat_hair)

    # =========================================================================
    # 5. SUNLIT HALO FLYAWAYS & FINE LOOSE WISPS (`Character_Hair_Flyaways`)
    # =========================================================================
    fly_pts, fly_rad = [], []
    all_source = r_pts[::5] + l_pts[::7] + b_pts[::9]
    for base_c in all_source:
        n_s = len(base_c)
        t_arr = np.linspace(0.0, 1.0, n_s)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        amp = rng.uniform(0.008, 0.020)
        out_dir = np.array([math.cos(ang), -0.30 * abs(math.sin(ang)), 0.40 + 0.35 * math.sin(ang)])
        out_dir /= np.linalg.norm(out_dir)
        fc = base_c.copy()
        for k in range(2, n_s):
            t = k / (n_s - 1)
            fc[k] += out_dir * amp * (math.sin(math.pi * (t ** 0.8)) + 0.30 * t)
            fc[k] = _push_outside_colliders(fc[k], head_bvh, sweater_bvhs, head_clear=0.0038, sw_clear=0.0045)
        fly_pts.append(fc)
        r0 = rng.uniform(0.000048, 0.000072)
        fly_rad.append(r0 * np.clip(1.0 - 0.86 * (t_arr ** 1.4), 0.12, 1.0))

    obj_fly = _create_curves_obj("Character_Hair_Flyaways", fly_pts, fly_rad, mat_hair)
    obj_core = _build_scalp_volume_core(anat_info, mat_core)

    hair_objs = [obj_core, obj_bangs, obj_right, obj_left, obj_back, obj_fly]
    for ob in hair_objs:
        char_col.objects.link(ob)

    return hair_objs
