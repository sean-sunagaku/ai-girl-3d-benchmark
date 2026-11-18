"""Sunlit cafe scenery, depth-of-field beauty camera, studio diagnostic cameras, and lighting setup."""

import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector


def _look_at(obj: bpy.types.Object, loc: Vector, target: Vector):
    obj.location = loc
    direction = Vector(target) - Vector(loc)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _make_principled_mat(name: str, color, roughness: float = 0.5, tex_img=None, emission=None, transmission: float = 0.0):
    mat = bpy.data.materials.new(name)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if transmission > 0.0:
        bsdf.inputs["Transmission Weight"].default_value = transmission
        bsdf.inputs["IOR"].default_value = 1.48
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission[:3], 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission[3]
    if tex_img is not None:
        uv = nodes.new("ShaderNodeTexCoord")
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = tex_img
        links.new(uv.outputs["Generated"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def build_cafe_scenery(wood_img: bpy.types.Image, collection: bpy.types.Collection):
    """Build tasteful sunlit cafe background props (chair, table, sunlit window, potted plant, counter, flower vase)."""
    mat_oak = _make_principled_mat("Cafe_Oak_Wood", (0.55, 0.36, 0.21), roughness=0.45, tex_img=wood_img)
    mat_wall = _make_principled_mat("Cafe_Cream_Wall", (0.86, 0.82, 0.76), roughness=0.85)
    mat_win_glow = _make_principled_mat("Cafe_Window_Light", (1.0, 0.98, 0.94), roughness=0.2, emission=(1.0, 0.98, 0.93, 3.0))
    mat_frame = _make_principled_mat("Cafe_Window_Frame", (0.28, 0.24, 0.21), roughness=0.55)
    mat_pot = _make_principled_mat("Cafe_Ceramic_Pot", (0.85, 0.80, 0.73), roughness=0.50)
    mat_leaf = _make_principled_mat("Cafe_Plant_Leaf", (0.18, 0.34, 0.14), roughness=0.42)
    mat_glass = _make_principled_mat("Cafe_Vase_Glass", (0.94, 0.97, 0.96), roughness=0.06, transmission=0.92)
    mat_flower = _make_principled_mat("Cafe_White_Flower", (0.96, 0.95, 0.92), roughness=0.55)
    mat_mug = _make_principled_mat("Cafe_Ceramic_Mug", (0.90, 0.88, 0.85), roughness=0.28)
    mat_counter = _make_principled_mat("Cafe_Dark_Wood", (0.36, 0.24, 0.14), roughness=0.48, tex_img=wood_img)

    def link_obj(obj):
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        collection.objects.link(obj)
        return obj

    # 1. Curved Oak Wooden Chair Back behind character (peeks softly behind her right shoulder!)
    bm = bmesh.new()
    nu_c, nv_c = 24, 6
    pts_top = []
    for i in range(nu_c):
        t = (i / (nu_c - 1)) - 0.5  # -0.5..+0.5
        ang = t * 1.45
        cx = -0.12 + 0.36 * math.sin(ang)
        cy = 0.26 - 0.12 * (1.0 - math.cos(ang))
        row_t = []
        for k in range(nv_c):
            ang_tube = 2.0 * math.pi * k / nv_c
            r_x = 0.015 * math.cos(ang_tube)
            r_z = 0.022 * math.sin(ang_tube)
            z = 1.325 + 0.025 * math.cos(ang * 1.1) + r_z
            row_t.append(bm.verts.new((cx + r_x * 0.4, cy + r_x, z)))
        pts_top.append(row_t)
    for i in range(nu_c - 1):
        for k in range(nv_c):
            kn = (k + 1) % nv_c
            bm.faces.new((pts_top[i][k], pts_top[i + 1][k], pts_top[i + 1][kn], pts_top[i][kn]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    m_chair = bpy.data.meshes.new("Cafe_Chair_Back_Mesh")
    bm.to_mesh(m_chair)
    bm.free()
    chair_obj = bpy.data.objects.new("Cafe_Chair_Back", m_chair)
    chair_obj.data.materials.append(mat_oak)
    for p in chair_obj.data.polygons:
        p.use_smooth = True
    chair_obj.modifiers.new("Subdivision", type="SUBSURF").levels = 1
    link_obj(chair_obj)

    # Chair uprights
    for sx in (-0.21, 0.21):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.016, depth=0.45, location=(sx, 0.25, 1.05))
        post = link_obj(bpy.context.active_object)
        post.name = f"Cafe_Chair_Post_{'L' if sx > 0 else 'R'}"
        post.data.materials.append(mat_oak)

    # 2. Foreground Oak Table edge (subtle hint at bottom-right of beauty frame)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.25, -0.35, 0.96))
    tbl = link_obj(bpy.context.active_object)
    tbl.name = "Cafe_Foreground_Table"
    tbl.scale = (0.95, 0.65, 0.032)
    tbl.data.materials.append(mat_oak)

    # 3. Back Cafe Wall, Sunlit Left Window, Counter & Shelves
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0.6, 2.75, 1.5))
    wall = link_obj(bpy.context.active_object)
    wall.name = "Cafe_Back_Wall"
    wall.rotation_euler = (math.pi * 0.5, 0.0, 0.0)
    wall.data.materials.append(mat_wall)

    # Left bright sunlit window panel & mullions
    bpy.ops.mesh.primitive_plane_add(size=2.4, location=(-1.45, 2.55, 1.75))
    win = link_obj(bpy.context.active_object)
    win.name = "Cafe_Sunlit_Window"
    win.rotation_euler = (math.pi * 0.5, 0.0, -0.25)
    win.data.materials.append(mat_win_glow)

    for mx in (-1.45, -0.85, -0.35):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(mx, 2.40, 1.75))
        mul = link_obj(bpy.context.active_object)
        mul.name = f"Cafe_Window_Mullion_{mx:.1f}"
        mul.scale = (0.045, 0.05, 2.2)
        mul.data.materials.append(mat_frame)

    # 4. Potted Green Cafe Plant on viewer's left (clearly visible behind left window corner!)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.14, depth=0.22, location=(-0.58, 1.55, 1.38))
    pot = link_obj(bpy.context.active_object)
    pot.name = "Cafe_Plant_Pot"
    pot.data.materials.append(mat_pot)

    rng = np.random.default_rng(77)
    for k in range(34):
        ang = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.03, 0.25)
        lz = rng.uniform(1.48, 2.08)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rng.uniform(0.035, 0.068), location=(-0.58 + r * math.cos(ang), 1.55 + r * math.sin(ang), lz))
        leaf = link_obj(bpy.context.active_object)
        leaf.name = f"Cafe_Plant_Leaf_{k}"
        leaf.scale = (1.3, 0.85, 0.35)
        leaf.rotation_euler = (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), ang)
        leaf.data.materials.append(mat_leaf)

    # 5. Wooden Counter, Shelves & Mugs on viewer's right
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.95, 2.25, 1.15))
    cnt = link_obj(bpy.context.active_object)
    cnt.name = "Cafe_Counter"
    cnt.scale = (2.1, 0.55, 0.65)
    cnt.data.materials.append(mat_counter)

    for sh_z in (1.78, 2.08):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.05, 2.55, sh_z))
        sh = link_obj(bpy.context.active_object)
        sh.name = f"Cafe_Shelf_{sh_z:.2f}"
        sh.scale = (1.8, 0.28, 0.03)
        sh.data.materials.append(mat_counter)
        for m_i in range(5):
            mx = 0.40 + m_i * 0.28
            bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=0.11, location=(mx, 2.52, sh_z + 0.07))
            mug = link_obj(bpy.context.active_object)
            mug.name = f"Cafe_Mug_{sh_z:.2f}_{m_i}"
            mug.data.materials.append(mat_mug)

    # 6. Background Table & Small Glass Vase with White Flowers + Stems on viewer's right
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.68, 1.25, 1.05))
    btbl = link_obj(bpy.context.active_object)
    btbl.name = "Cafe_Back_Table"
    btbl.scale = (0.75, 0.55, 0.035)
    btbl.data.materials.append(mat_oak)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.040, depth=0.14, location=(0.56, 1.22, 1.14))
    vase = link_obj(bpy.context.active_object)
    vase.name = "Cafe_Flower_Vase"
    vase.data.materials.append(mat_glass)

    for f_i in range(14):
        fa = f_i * 2.0 * math.pi / 14.0 + (f_i % 3) * 0.3
        fr = 0.025 + 0.032 * (f_i % 3) / 2.0
        fz = 1.24 + 0.065 * ((f_i % 4) / 3.0)
        fx = 0.56 + fr * math.cos(fa)
        fy = 1.22 + fr * math.sin(fa)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.003, depth=0.12, location=(0.56 + 0.5 * fr * math.cos(fa), 1.22 + 0.5 * fr * math.sin(fa), fz - 0.06))
        stem = link_obj(bpy.context.active_object)
        stem.name = f"Cafe_FlowerStem_{f_i}"
        stem.rotation_euler = (-0.25 * math.sin(fa), 0.25 * math.cos(fa), 0.0)
        stem.data.materials.append(mat_leaf)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.015, location=(fx, fy, fz))
        fl = link_obj(bpy.context.active_object)
        fl.name = f"Cafe_Flower_{f_i}"
        fl.data.materials.append(mat_flower)


def setup_lighting_and_cameras(beauty_cam_pos: Vector, face_focus_pos: Vector, lights_col: bpy.types.Collection, cams_col: bpy.types.Collection):
    """Create beauty camera with shallow DOF, diagnostic turnaround cameras, and soft cafe daylight rig."""
    scene = bpy.context.scene

    # World background: soft warm sunlit cafe atmosphere
    world = bpy.data.worlds.new("Cafe_World")
    scene.world = world
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.86, 0.84, 0.79, 1.0)
    bg.inputs[1].default_value = 0.45

    def add_area_light(name, loc, target, energy, size_x, size_y, color):
        ld = bpy.data.lights.new(name, type="AREA")
        ld.shape = "RECTANGLE"
        ld.size = size_x
        ld.size_y = size_y
        ld.energy = energy
        ld.color = color
        lo = bpy.data.objects.new(name, ld)
        _look_at(lo, Vector(loc), Vector(target))
        lights_col.objects.link(lo)
        return lo

    # 1. Main sunlit cafe window light on viewer's left (bright soft key + eye catchlights!)
    add_area_light("Key_CafeWindow", (-1.35, -0.45, 1.92), (0.0, -0.05, 1.46), 210.0, 1.4, 1.8, (1.0, 0.94, 0.87))
    # 2. Warm cafe interior fill on front-right (subtle shaping shadow on cheek & nose bridge)
    add_area_light("Fill_CafeInterior", (1.25, -0.95, 1.48), (0.0, -0.05, 1.42), 42.0, 1.9, 1.6, (1.0, 0.93, 0.87))
    # 3. Sunlit rim light from window back-left (hair & sleeve halo!)
    add_area_light("Rim_BackWindow", (-1.15, 1.25, 2.05), (0.0, 0.0, 1.48), 185.0, 1.3, 1.5, (1.0, 0.94, 0.84))
    # 4. Soft ceiling bounce fill
    add_area_light("Fill_TopBounce", (0.0, -0.65, 2.35), (0.0, -0.05, 1.42), 48.0, 2.2, 2.2, (1.0, 0.97, 0.94))

    # Beauty Camera (shallow portrait DOF focused strictly on her eyes!)
    bcam_data = bpy.data.cameras.new("Beauty_Camera_Data")
    bcam_data.lens = 33.0
    bcam_data.sensor_width = 36.0
    bcam_data.dof.use_dof = True
    bcam_data.dof.focus_distance = (face_focus_pos - beauty_cam_pos).length
    bcam_data.dof.aperture_fstop = 2.6
    bcam_obj = bpy.data.objects.new("Beauty_Camera", bcam_data)
    _look_at(bcam_obj, beauty_cam_pos, Vector((-0.032, -0.05, 1.400)))
    cams_col.objects.link(bcam_obj)
    scene.camera = bcam_obj

    # Diagnostic Turnaround Cameras (0, 45, 90, 180 deg around character center)
    diag_center = Vector((0.0, -0.15, 1.32))
    diag_radius = 1.48
    diag_cams = {}
    for cname, deg in (("Cam_Front", 0.0), ("Cam_ThreeQuarter", 45.0), ("Cam_Side", 90.0), ("Cam_Back", 180.0)):
        rad = math.radians(deg)
        loc = diag_center + Vector((diag_radius * math.sin(rad), -diag_radius * math.cos(rad), 0.04))
        cd = bpy.data.cameras.new(f"{cname}_Data")
        cd.lens = 50.0
        cd.sensor_width = 36.0
        co = bpy.data.objects.new(cname, cd)
        _look_at(co, loc, diag_center)
        cams_col.objects.link(co)
        diag_cams[cname] = co

    return bcam_obj, diag_cams
