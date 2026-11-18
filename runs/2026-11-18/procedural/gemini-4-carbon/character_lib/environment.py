"""Sunlit cafe 3D scenery, natural portrait lighting, beauty selfie camera, and diagnostic cameras."""

import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix


BEAUTY_CAM_LOC = Vector((-0.014, -0.545, 1.480))
BEAUTY_CAM_TARGET = Vector((-0.008, -0.120, 1.436))


def _make_simple_mat(name: str, color, rough=0.45, metallic=0.0, transmission=0.0, sss=0.0, emission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Transmission Weight"].default_value = transmission
    if sss > 0.0:
        bsdf.inputs["Subsurface Weight"].default_value = sss
        bsdf.inputs["Subsurface Scale"].default_value = 0.008
    if emission > 0.0:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    return m


def _make_wood_mat(name: str, base_col=(0.44, 0.25, 0.12), dark_col=(0.24, 0.12, 0.05), rough=0.36):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.0, 14.0, 2.0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.5
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Distortion"].default_value = 1.6

    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    rgba_in = [s for s in mix.inputs if s.type == "RGBA"]
    rgba_out = [s for s in mix.outputs if s.type == "RGBA"][0]
    rgba_in[0].default_value = (*dark_col, 1.0)
    rgba_in[1].default_value = (*base_col, 1.0)

    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(rgba_out, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    return m


def _add_box(name: str, size_xyz, loc_xyz, rot_euler=(0.0, 0.0, 0.0), mat=None, col=None, bevel=0.006):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size_xyz), verts=bm.verts)
    if bevel > 0.0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=2, profile=0.5, affect="EDGES")
    for f in bm.faces:
        f.smooth = True
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc_xyz)
    obj.rotation_euler = rot_euler
    if mat:
        obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def _add_cylinder(name: str, radius: float, depth: float, loc_xyz, rot_euler=(0, 0, 0), mat=None, col=None, segs=24):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=radius, radius2=radius, depth=depth)
    for f in bm.faces:
        f.smooth = True
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(loc_xyz)
    obj.rotation_euler = rot_euler
    if mat:
        obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def _build_cafe_chair(name: str, loc: Vector, rot_z: float, wood_mat, col):
    """Curved wooden cafe chair matching the bentwood chair backs visible in reference.png."""
    bm = bmesh.new()
    # Curved backrest arc
    segs = 18
    arc_r = 0.23
    h_bot, h_top = 0.30, 0.42
    ring_prev = None
    for i in range(segs + 1):
        t = i / segs
        ang = -0.78 + 1.56 * t
        cx = arc_r * math.sin(ang)
        cy = arc_r * (1.0 - math.cos(ang))
        # Rectangular cross-section of curved wooden rail
        pts = [
            Vector((cx, cy - 0.014, h_bot)),
            Vector((cx, cy + 0.014, h_bot)),
            Vector((cx, cy + 0.014, h_top)),
            Vector((cx, cy - 0.014, h_top)),
        ]
        ring = [bm.verts.new(p) for p in pts]
        if ring_prev is not None:
            for k in range(4):
                k2 = (k + 1) % 4
                f = bm.faces.new([ring_prev[k], ring[k], ring[k2], ring_prev[k2]])
                f.smooth = True
        ring_prev = ring

    # Two angled wooden side uprights
    for sx in (-1.0, 1.0):
        b_sub = bmesh.new()
        bmesh.ops.create_cube(b_sub, size=1.0)
        bmesh.ops.scale(b_sub, vec=Vector((0.028, 0.028, 0.46)), verts=b_sub.verts)
        bmesh.ops.rotate(b_sub, verts=b_sub.verts, matrix=Matrix.Rotation(math.radians(-8.0), 3, "X"))
        bmesh.ops.translate(b_sub, verts=b_sub.verts, vec=Vector((sx * 0.18, 0.05, 0.20)))
        for f in b_sub.faces:
            f.smooth = True
        mesh_tmp = bpy.data.meshes.new("tmp_c")
        b_sub.to_mesh(mesh_tmp)
        b_sub.free()
        bm.from_mesh(mesh_tmp)
        bpy.data.meshes.remove(mesh_tmp)

    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = loc
    obj.rotation_euler = (0.0, 0.0, rot_z)
    obj.data.materials.append(wood_mat)
    col.objects.link(obj)
    return obj


def _build_potted_ficus(loc: Vector, pot_mat, stem_mat, leaf_mat, col):
    """Potted green indoor cafe tree on viewer's left catching bright window backlight."""
    # Cream ceramic pot
    bm_pot = bmesh.new()
    bmesh.ops.create_cone(bm_pot, cap_ends=True, cap_tris=False, segments=28, radius1=0.15, radius2=0.20, depth=0.38)
    for f in bm_pot.faces:
        f.smooth = True
    mpot = bpy.data.meshes.new("Cafe_Planter_Pot_Mesh")
    bm_pot.to_mesh(mpot)
    bm_pot.free()
    opot = bpy.data.objects.new("Cafe_Planter_Pot", mpot)
    opot.location = loc + Vector((0.0, 0.0, 0.19))
    opot.data.materials.append(pot_mat)
    col.objects.link(opot)

    # Woody branches + lush 3D oval leaves in a single clean mesh
    rng = np.random.default_rng(707)
    bm_fol = bmesh.new()
    # Main stems
    for si in range(4):
        ang = si * 1.55 + 0.3
        b_st = bmesh.new()
        bmesh.ops.create_cone(b_st, cap_ends=True, segments=8, radius1=0.014, radius2=0.005, depth=0.88)
        bmesh.ops.rotate(b_st, verts=b_st.verts, matrix=Matrix.Rotation(0.18, 3, "X") @ Matrix.Rotation(ang, 3, "Z"))
        bmesh.ops.translate(b_st, verts=b_st.verts, vec=Vector((0.03 * math.cos(ang), 0.03 * math.sin(ang), 0.72)))
        for f in b_st.faces:
            f.material_index = 0
            f.smooth = True
        mt = bpy.data.meshes.new("tmp_st")
        b_st.to_mesh(mt)
        b_st.free()
        bm_fol.from_mesh(mt)
        bpy.data.meshes.remove(mt)

    # ~135 3D arched ficus leaves catching window sun
    for li in range(135):
        r_rad = rng.uniform(0.05, 0.36)
        az = rng.uniform(0.0, 2.0 * math.pi)
        zh = rng.uniform(0.42, 1.25)
        lx = r_rad * math.cos(az)
        ly = r_rad * math.sin(az)
        lsize = rng.uniform(0.045, 0.082)

        # Diamond-arched quad leaf (6 verts, 2 quads -> curved midrib!)
        v0 = Vector((0.0, 0.0, 0.0))
        v1 = Vector((-0.42 * lsize, 0.45 * lsize, 0.08 * lsize))
        v2 = Vector((0.0, 0.50 * lsize, -0.02 * lsize))
        v3 = Vector((0.42 * lsize, 0.45 * lsize, 0.08 * lsize))
        v4 = Vector((0.0, 1.05 * lsize, -0.12 * lsize))

        rot = (
            Matrix.Rotation(float(rng.uniform(0, 2 * math.pi)), 3, "Z")
            @ Matrix.Rotation(float(rng.uniform(-0.7, 0.7)), 3, "X")
            @ Matrix.Rotation(float(rng.uniform(-0.5, 0.5)), 3, "Y")
        )
        origin = Vector((lx, ly, zh))
        bv = [bm_fol.verts.new(origin + rot @ p) for p in (v0, v1, v2, v3, v4)]
        f1 = bm_fol.faces.new([bv[0], bv[1], bv[4], bv[2]])
        f2 = bm_fol.faces.new([bv[0], bv[2], bv[4], bv[3]])
        f1.material_index = 1
        f2.material_index = 1
        f1.smooth = True
        f2.smooth = True

    mfol = bpy.data.meshes.new("Cafe_Ficus_Plant_Mesh")
    bm_fol.to_mesh(mfol)
    bm_fol.free()
    ofol = bpy.data.objects.new("Cafe_Ficus_Plant", mfol)
    ofol.location = loc
    ofol.data.materials.append(stem_mat)
    ofol.data.materials.append(leaf_mat)
    col.objects.link(ofol)


def _build_flower_vase(loc: Vector, glass_mat, leaf_mat, petal_mat, col):
    """Small glass vase with green stems and white blossoms on right-side cafe table (matching reference.png!)."""
    # Glass bottle/vase
    bm_v = bmesh.new()
    profile = [(0.036, 0.0), (0.042, 0.04), (0.040, 0.09), (0.022, 0.125), (0.026, 0.155)]
    segs = 20
    prev_r = None
    for rad, hz in profile:
        ring = [
            bm_v.verts.new(Vector((rad * math.cos(2 * math.pi * k / segs), rad * math.sin(2 * math.pi * k / segs), hz)))
            for k in range(segs)
        ]
        if prev_r is not None:
            for k in range(segs):
                k2 = (k + 1) % segs
                f = bm_v.faces.new([prev_r[k], ring[k], ring[k2], prev_r[k2]])
                f.smooth = True
        prev_r = ring
    mv = bpy.data.meshes.new("Cafe_Vase_Mesh")
    bm_v.to_mesh(mv)
    bm_v.free()
    ov = bpy.data.objects.new("Cafe_Flower_Vase", mv)
    ov.location = loc
    ov.data.materials.append(glass_mat)
    col.objects.link(ov)

    # Stems & white blossoms
    rng = np.random.default_rng(99)
    bm_fl = bmesh.new()
    for i in range(14):
        ang = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(0.08, 0.42)
        top = Vector((math.sin(tilt) * math.cos(ang) * 0.14, math.sin(tilt) * math.sin(ang) * 0.14, rng.uniform(0.20, 0.30)))
        # Tiny stem
        b_s = bmesh.new()
        bmesh.ops.create_cone(b_s, cap_ends=True, segments=6, radius1=0.0025, radius2=0.0018, depth=top.length)
        up = Vector((0, 0, 1))
        rot_q = up.rotation_difference(top.normalized()).to_matrix()
        bmesh.ops.rotate(b_s, verts=b_s.verts, matrix=rot_q)
        bmesh.ops.translate(b_s, verts=b_s.verts, vec=top * 0.5 + Vector((0, 0, 0.04)))
        for f in b_s.faces:
            f.material_index = 0
        mt = bpy.data.meshes.new("ts")
        b_s.to_mesh(mt)
        b_s.free()
        bm_fl.from_mesh(mt)
        bpy.data.meshes.remove(mt)

        # Tiny white blossom sphere cluster
        b_p = bmesh.new()
        bmesh.ops.create_icosphere(b_p, subdivisions=1, radius=rng.uniform(0.009, 0.015))
        bmesh.ops.translate(b_p, verts=b_p.verts, vec=top + Vector((0, 0, 0.04)))
        for f in b_p.faces:
            f.material_index = 1
            f.smooth = True
        mp = bpy.data.meshes.new("tp")
        b_p.to_mesh(mp)
        b_p.free()
        bm_fl.from_mesh(mp)
        bpy.data.meshes.remove(mp)

    mfl = bpy.data.meshes.new("Cafe_Flowers_Mesh")
    bm_fl.to_mesh(mfl)
    bm_fl.free()
    ofl = bpy.data.objects.new("Cafe_Flowers", mfl)
    ofl.location = loc
    ofl.data.materials.append(leaf_mat)
    ofl.data.materials.append(petal_mat)
    col.objects.link(ofl)


def build_cafe_and_lighting_and_cameras(scene: bpy.types.Scene) -> dict:
    """Create Cafe_Environment, Diagnostic_Backdrop, Lighting, and all 5 required Cameras."""
    col_cafe = bpy.data.collections.new("Cafe_Environment")
    col_diag = bpy.data.collections.new("Diagnostic_Backdrop")
    col_lights = bpy.data.collections.new("Lighting")
    col_cams = bpy.data.collections.new("Cameras")
    for c in (col_cafe, col_diag, col_lights, col_cams):
        scene.collection.children.link(c)

    # Materials for cafe props
    mat_oak = _make_wood_mat("Mat_Cafe_Oak", base_col=(0.50, 0.30, 0.15), dark_col=(0.30, 0.16, 0.07), rough=0.34)
    mat_walnut = _make_wood_mat("Mat_Cafe_Walnut", base_col=(0.32, 0.18, 0.09), dark_col=(0.17, 0.09, 0.04), rough=0.42)
    mat_wall = _make_simple_mat("Mat_Cafe_Wall", (0.86, 0.82, 0.76), rough=0.82)
    mat_pot = _make_simple_mat("Mat_Cafe_Pot", (0.78, 0.73, 0.67), rough=0.55)
    mat_stem = _make_simple_mat("Mat_Plant_Stem", (0.18, 0.11, 0.06), rough=0.75)
    mat_leaf = _make_simple_mat("Mat_Plant_Leaf", (0.22, 0.46, 0.11), rough=0.36, sss=0.65)
    mat_glass = _make_simple_mat("Mat_Vase_Glass", (0.95, 0.98, 0.96), rough=0.04, transmission=0.95)
    mat_petal = _make_simple_mat("Mat_Flower_Petal", (0.96, 0.95, 0.90), rough=0.45, sss=0.4)
    mat_ceramic = _make_simple_mat("Mat_Cafe_Ceramic", (0.88, 0.85, 0.80), rough=0.22)
    mat_canister = _make_simple_mat("Mat_Cafe_Canister", (0.09, 0.08, 0.08), rough=0.32, metallic=0.25)
    mat_steel = _make_simple_mat("Mat_Cafe_Steel", (0.72, 0.73, 0.75), rough=0.24, metallic=0.90)
    mat_win_glow = _make_simple_mat("Mat_Window_Daylight", (1.0, 0.97, 0.91), rough=0.5, emission=3.8)

    # 1. Hero Bentwood Chair right behind her right shoulder (-X, viewer's left in reference.png!)
    _build_cafe_chair("Cafe_Chair_Hero", Vector((-0.24, 0.16, 0.89)), math.radians(16.0), mat_oak, col_cafe)

    # 2. Foreground Wooden Tabletop edge (subtle warm wood glimpse at bottom right of reference.png!)
    _add_box("Cafe_Table_Foreground", (0.68, 0.36, 0.032), (0.24, -0.34, 0.945), (0, 0, 0.08), mat_oak, col_cafe)

    # 3. Middleground Right Wooden Table + Curved Chairs + Flower Vase (viewer's right in reference.png!)
    _add_box("Cafe_Table_Right", (0.82, 0.68, 0.036), (0.68, 0.88, 1.06), (0, 0, -0.06), mat_oak, col_cafe)
    _add_cylinder("Cafe_Table_Leg", 0.032, 0.72, (0.68, 0.88, 0.68), (0, 0, 0), mat_canister, col_cafe)
    _build_cafe_chair("Cafe_Chair_MidRight", Vector((0.52, 1.18, 0.86)), math.radians(172.0), mat_oak, col_cafe)
    _build_flower_vase(Vector((0.62, 0.72, 1.078)), mat_glass, mat_leaf, mat_petal, col_cafe)

    # 4. Left-Side Sunlit Window Mullions + Emissive Exterior Daylight Backdrop + Potted Ficus Tree!
    _add_box("Cafe_Window_Backdrop", (1.65, 0.04, 1.95), (-0.82, 1.85, 1.65), (0, 0, 0.25), mat_win_glow, col_cafe, bevel=0.0)
    _add_box("Cafe_Window_Mullion_V", (0.045, 0.06, 1.95), (-0.98, 1.55, 1.65), (0, 0, 0.25), mat_canister, col_cafe)
    _add_box("Cafe_Window_Mullion_H", (1.60, 0.06, 0.042), (-0.80, 1.80, 1.32), (0, 0, 0.25), mat_canister, col_cafe)
    _build_potted_ficus(Vector((-0.44, 0.85, 0.72)), mat_pot, mat_stem, mat_leaf, col_cafe)
    # Second higher foliage cluster in upper-center-left background matching reference.png!
    _build_potted_ficus(Vector((0.12, 1.65, 1.08)), mat_pot, mat_stem, mat_leaf, col_cafe)

    # 5. Back-Right Cafe Wall, Wooden Counter & 3-Tier Shelves with Mugs/Canisters
    _add_box("Cafe_Back_Wall", (2.60, 0.08, 2.20), (0.55, 2.45, 1.65), (0, 0, 0), mat_wall, col_cafe, bevel=0.0)
    _add_box("Cafe_Counter", (1.55, 0.52, 0.92), (0.72, 1.95, 0.92), (0, 0, 0), mat_walnut, col_cafe)
    _add_box("Cafe_Counter_Top", (1.60, 0.56, 0.045), (0.72, 1.95, 1.39), (0, 0, 0), mat_oak, col_cafe)

    for si, sz in enumerate([1.68, 1.94, 2.20]):
        _add_box(f"Cafe_Shelf_{si+1}", (1.18, 0.22, 0.032), (0.78, 2.32, sz), (0, 0, 0), mat_oak, col_cafe)
        # Mugs, canisters, and jars along shelves
        for k, ox in enumerate([-0.38, -0.18, 0.04, 0.24, 0.42]):
            m_choice = [mat_ceramic, mat_canister, mat_steel, mat_ceramic, mat_canister][(si + k) % 5]
            rad = 0.038 if (k % 2 == 0) else 0.048
            ht = 0.095 if (k % 2 == 0) else 0.135
            _add_cylinder(
                f"Cafe_Jar_{si}_{k}",
                rad,
                ht,
                (0.78 + ox, 2.30, sz + 0.016 + ht * 0.5),
                (0, 0, 0),
                m_choice,
                col_cafe,
                segs=16,
            )

    # Chalkboard menu sign on upper shelf
    _add_box("Cafe_Menu_Sign", (0.26, 0.025, 0.20), (0.48, 2.33, 2.06), (-0.12, 0, 0), mat_canister, col_cafe)

    # ==================================================================
    # Plain Neutral Backdrop for Diagnostic Views (hidden in beauty view!)
    # ==================================================================
    mat_studio = _make_simple_mat("Mat_Diagnostic_Backdrop", (0.24, 0.235, 0.245), rough=0.95)
    # Large enclosing cyclorama sphere hidden for beauty render, enabled for diagnostic turnaround renders
    bm_cyc = bmesh.new()
    bmesh.ops.create_uvsphere(bm_cyc, u_segments=48, v_segments=24, radius=4.5)
    bmesh.ops.reverse_faces(bm_cyc, faces=bm_cyc.faces)
    for f in bm_cyc.faces:
        f.smooth = True
    m_cyc = bpy.data.meshes.new("Diagnostic_Studio_Dome_Mesh")
    bm_cyc.to_mesh(m_cyc)
    bm_cyc.free()
    o_cyc = bpy.data.objects.new("Diagnostic_Studio_Dome", m_cyc)
    o_cyc.location = Vector((0.0, 0.0, 1.35))
    o_cyc.data.materials.append(mat_studio)
    col_diag.objects.link(o_cyc)
    col_diag.hide_viewport = True
    col_diag.hide_render = True

    # ==================================================================
    # Natural Cafe Daylight & Soft Fill Lighting (consistent across views!)
    # ==================================================================
    def add_area_light(name: str, loc, target, energy: float, size: float, color=(1.0, 0.96, 0.90), shape="DISK"):
        ldata = bpy.data.lights.new(name=f"{name}_Data", type="AREA")
        ldata.energy = energy
        ldata.shape = shape
        ldata.size = size
        ldata.color = color
        lobj = bpy.data.objects.new(name, ldata)
        lobj.location = Vector(loc)
        direction = (Vector(target) - Vector(loc)).normalized()
        lobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        col_lights.objects.link(lobj)
        return lobj

    head_focus = Vector((0.0, -0.12, 1.50))

    # 1. Main Front-Upper-Left Soft Cafe Window Key Light (creates luminous warm skin & crisp 10:30 eye catchlights!)
    add_area_light(
        "Light_Window_Key",
        loc=(-0.58, -0.88, 1.82),
        target=head_focus,
        energy=25.0,
        size=0.48,
        color=(1.0, 0.965, 0.925),
        shape="SQUARE",
    )

    # 2. Warm Sunlit Rim / Hair Back-Left Window Light (creates sunlit halo on wavy chestnut hair & left pink sleeve!)
    add_area_light(
        "Light_Window_SunRim",
        loc=(-1.05, 0.32, 1.92),
        target=Vector((-0.08, -0.05, 1.38)),
        energy=44.0,
        size=0.62,
        color=(1.0, 0.93, 0.83),
        shape="SQUARE",
    )

    # 3. Broad Soft Right Cafe Interior Fill (keeps shadow side soft, airy, and flattering like reference.png!)
    add_area_light(
        "Light_Cafe_RightFill",
        loc=(0.88, -0.92, 1.52),
        target=head_focus,
        energy=11.0,
        size=1.15,
        color=(1.0, 0.95, 0.89),
        shape="DISK",
    )

    # 4. Warm Wood Tabletop Upward Bounce (softens under-chin, neck, and lower eyelid shadows!)
    add_area_light(
        "Light_Table_Bounce",
        loc=(0.0, -0.58, 0.92),
        target=Vector((0.0, -0.12, 1.48)),
        energy=5.5,
        size=0.75,
        color=(1.0, 0.90, 0.80),
        shape="DISK",
    )

    # 5. Gentle Back-Right Fill so diagnostic back/side views are also evenly & naturally illuminated!
    add_area_light(
        "Light_Back_SoftFill",
        loc=(0.55, 1.25, 1.72),
        target=Vector((0.0, 0.02, 1.38)),
        energy=15.0,
        size=1.20,
        color=(1.0, 0.94, 0.88),
        shape="DISK",
    )

    # World shader: soft warm sunlit cafe ambient sky
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.50, 0.47, 0.43, 1.0)
    bg.inputs["Strength"].default_value = 0.28

    # ==================================================================
    # Cameras: Beauty Camera (960x1040) + 4 Turnaround Diagnostic Cameras (768x768)
    # ==================================================================
    def make_cam(name: str, loc: Vector, target: Vector, lens_mm: float, use_dof: bool = False, fstop: float = 2.0, dof_focus: Vector = None):
        cdata = bpy.data.cameras.new(f"{name}_Data")
        cdata.lens = lens_mm
        cdata.sensor_width = 36.0
        cdata.clip_start = 0.05
        cdata.clip_end = 25.0
        if use_dof and dof_focus is not None:
            cdata.dof.use_dof = True
            cdata.dof.aperture_fstop = fstop
            cdata.dof.focus_distance = (dof_focus - loc).length
        cobj = bpy.data.objects.new(name, cdata)
        cobj.location = loc
        direction = (target - loc).normalized()
        cobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        col_cams.objects.link(cobj)
        return cobj

    eye_focus_approx = Vector((-0.012, -0.142, 1.542))
    cam_beauty = make_cam(
        "Cam_Beauty",
        loc=BEAUTY_CAM_LOC,
        target=BEAUTY_CAM_TARGET,
        lens_mm=30.0,
        use_dof=True,
        fstop=2.0,
        dof_focus=eye_focus_approx,
    )
    scene.camera = cam_beauty

    diag_center = Vector((0.0, -0.095, 1.355))
    diag_r = 1.32
    diag_cams = {}
    # Azimuth 0 deg = Front (0, -1), +45 deg = Three-Quarter (+X, -Y), +90 deg = Side (+X showing face profile & left ear!), 180 deg = Back (0, +1)
    for key, cname, deg in [
        ("front", "Cam_Front", 0.0),
        ("three_quarter", "Cam_ThreeQuarter", 45.0),
        ("side", "Cam_Side", 90.0),
        ("back", "Cam_Back", 180.0),
    ]:
        rad = math.radians(deg)
        cloc = diag_center + Vector((diag_r * math.sin(rad), -diag_r * math.cos(rad), 0.035))
        diag_cams[key] = make_cam(cname, cloc, diag_center, lens_mm=50.0, use_dof=False)

    return {
        "col_cafe": col_cafe,
        "col_diag": col_diag,
        "col_lights": col_lights,
        "col_cams": col_cams,
        "cam_beauty": cam_beauty,
        "diag_cams": diag_cams,
        "world_bg": bg,
    }
