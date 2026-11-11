import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from lib_materials import new_material, rgba_srgb


def _make_simple_mat(name, srgb, rough=0.45, metal=0.0, trans=0.0, emission_srgb=None, emission_str=0.0, tex_img=None):
    mat = new_material(name)
    nt = mat.node_tree
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (600, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (220, 0)
    bsdf.inputs['Base Color'].default_value = rgba_srgb(*srgb)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    bsdf.inputs['Transmission Weight'].default_value = trans
    if emission_srgb is not None and emission_str > 0.0:
        bsdf.inputs['Emission Color'].default_value = rgba_srgb(*emission_srgb)
        bsdf.inputs['Emission Strength'].default_value = emission_str
    if tex_img is not None:
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = tex_img
        tex.location = (-160, 120)
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def _add_box(col, name, center, size, mat, rot_euler=(0.0, 0.0, 0.0), bevel=True):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    uv_lay = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        f.smooth = True
        for i, lp in enumerate(f.loops):
            lp[uv_lay].uv = ((i in (1, 2)) * 1.5, (i in (2, 3)) * 1.5)
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.rotation_euler = rot_euler
    obj.data.materials.append(mat)
    if bevel:
        bev = obj.modifiers.new(name="Bevel", type='BEVEL')
        bev.width = 0.006
        bev.segments = 2
    col.objects.link(obj)
    return obj


def _add_cylinder(col, name, center, radius, depth, mat, segs=24, rot_euler=(0.0, 0.0, 0.0)):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=radius, radius2=radius, depth=depth)
    for f in bm.faces:
        f.smooth = True
    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.rotation_euler = rot_euler
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


def build_cafe_environment(cafe_col, textures):
    """
    Build genuine 3D sunlit cafe interior behind the character matching input/reference.png:
    - Left sunlit cafe window + potted leafy green plant in cream ceramic pot
    - Wooden cafe chair backrest behind character's right shoulder
    - Warm honey-oak cafe tables + clear glass vase with white flowers on right table
    - Warm wooden coffee counter + wall shelves with ceramic mugs & canisters in back-right
    """
    mat_wood = _make_simple_mat("Mat_Cafe_Wood", (0.68, 0.48, 0.30), rough=0.38, tex_img=textures["cafe_wood"])
    mat_wood_dark = _make_simple_mat("Mat_Cafe_Wood_Dark", (0.45, 0.30, 0.18), rough=0.48)
    mat_wall = _make_simple_mat("Mat_Cafe_Wall_Cream", (0.93, 0.90, 0.85), rough=0.82)
    mat_window = _make_simple_mat(
        "Mat_Cafe_Sunlit_Window", (0.99, 0.98, 0.95), rough=0.2, emission_srgb=(1.0, 0.97, 0.92), emission_str=5.2
    )
    mat_mullion = _make_simple_mat("Mat_Cafe_Mullion", (0.28, 0.27, 0.26), rough=0.45, metal=0.4)
    mat_pot = _make_simple_mat("Mat_Cafe_Planter", (0.86, 0.82, 0.76), rough=0.65)
    mat_leaf = _make_simple_mat("Mat_Cafe_Foliage", (0.36, 0.56, 0.22), rough=0.42)
    mat_leaf_bright = _make_simple_mat(
        "Mat_Cafe_Foliage_Sunlit", (0.52, 0.72, 0.28), rough=0.38, emission_srgb=(0.55, 0.75, 0.30), emission_str=0.45
    )
    mat_glass = _make_simple_mat("Mat_Cafe_Glass", (0.98, 0.99, 0.98), rough=0.03, trans=0.95)
    mat_petal = _make_simple_mat("Mat_Cafe_Flower_White", (0.97, 0.96, 0.92), rough=0.50, emission_srgb=(0.98, 0.96, 0.90), emission_str=0.25)
    mat_ceramic = _make_simple_mat("Mat_Cafe_Ceramic", (0.92, 0.89, 0.84), rough=0.25)
    mat_metal = _make_simple_mat("Mat_Cafe_Steel", (0.76, 0.77, 0.78), rough=0.28, metal=0.85)
    mat_chalk = _make_simple_mat("Mat_Cafe_Chalkboard", (0.14, 0.15, 0.15), rough=0.75)

    # 1. Back & Right Cafe Walls + Sunlit Left Window Pane
    _add_box(cafe_col, "Cafe_Back_Wall", (0.45, 2.35, 1.55), (3.2, 0.08, 2.4), mat_wall, bevel=False)
    _add_box(cafe_col, "Cafe_Window_SunPanel", (-0.92, 1.15, 1.58), (1.35, 0.04, 1.95), mat_window, rot_euler=(0, 0, math.radians(28)), bevel=False)
    _add_box(cafe_col, "Cafe_Window_SidePanel", (-1.18, 0.20, 1.58), (0.04, 1.45, 1.95), mat_window, bevel=False)
    # Window frame mullions on left
    _add_box(cafe_col, "Cafe_Window_Mullion_V", (-0.88, 0.72, 1.55), (0.035, 0.035, 1.95), mat_mullion, bevel=False)
    _add_box(cafe_col, "Cafe_Window_Mullion_H", (-0.92, 0.95, 1.32), (0.035, 1.25, 0.035), mat_mullion, bevel=False)

    # 2. Wooden Chair Backrest behind Character's Right Shoulder (-X, viewer's left!)
    _add_box(cafe_col, "Cafe_Chair_Backrest", (-0.28, 0.14, 1.18), (0.34, 0.028, 0.11), mat_wood, rot_euler=(math.radians(-8), 0, math.radians(16)))
    _add_box(cafe_col, "Cafe_Chair_Post_L", (-0.43, 0.18, 1.05), (0.032, 0.032, 0.42), mat_wood, rot_euler=(math.radians(-8), 0, math.radians(16)))

    # 3. Foreground Table Edge (bottom right) + Midground Sunlit Cafe Table & Chair on Right (+X)
    _add_box(cafe_col, "Cafe_Table_Foreground", (0.18, -0.25, 0.875), (0.68, 0.48, 0.034), mat_wood, rot_euler=(0, 0, math.radians(-8)))
    _add_box(cafe_col, "Cafe_Table_Right", (0.62, 0.78, 1.02), (0.78, 0.68, 0.036), mat_wood, rot_euler=(0, 0, math.radians(6)))
    _add_box(cafe_col, "Cafe_Right_Chair_Back", (0.46, 1.15, 1.24), (0.46, 0.03, 0.10), mat_wood, rot_euler=(0, 0, math.radians(-12)))

    # 4. Clear Glass Vase with White & Green Blossoms on Right Table (matches reference.png!)
    vase_pos = (0.56, 0.64, 1.11)
    _add_cylinder(cafe_col, "Cafe_Flower_Vase", vase_pos, 0.045, 0.15, mat_glass, segs=20)
    rng = np.random.default_rng(808)
    for fi in range(14):
        ang = rng.uniform(0, 2 * math.pi)
        sp = rng.uniform(0.015, 0.065)
        ht = rng.uniform(0.11, 0.22)
        sx = vase_pos[0] + sp * math.cos(ang)
        sy = vase_pos[1] + sp * math.sin(ang)
        sz = vase_pos[2] + ht
        _add_cylinder(
            cafe_col, f"Cafe_Stem_{fi}", (0.5 * (vase_pos[0] + sx), 0.5 * (vase_pos[1] + sy), vase_pos[2] + 0.5 * ht),
            0.003, ht, mat_leaf, segs=6, rot_euler=(sp * math.sin(ang) * 2.2, -sp * math.cos(ang) * 2.2, 0)
        )
        _add_cylinder(cafe_col, f"Cafe_Blossom_{fi}", (sx, sy, sz), rng.uniform(0.012, 0.020), 0.014, mat_petal, segs=8)

    # 5. Potted Green Leafy Plant by Left Sunlit Window (-X, viewer's left!)
    pot_pos = (-0.58, 0.78, 1.12)
    _add_cylinder(cafe_col, "Cafe_Plant_Pot", pot_pos, 0.135, 0.26, mat_pot, segs=20)
    # Leafy branches rising into sunlit window glow (Z ~ 1.25 .. 1.92)
    for li in range(46):
        la = rng.uniform(0, 2 * math.pi)
        lr = rng.uniform(0.03, 0.26)
        lz = rng.uniform(1.26, 1.92)
        lx = pot_pos[0] - 0.05 + lr * math.cos(la)
        ly = pot_pos[1] + 0.08 + lr * math.sin(la)
        lmat = mat_leaf_bright if (lx < -0.52 or lz > 1.62) else mat_leaf
        _add_box(
            cafe_col, f"Cafe_Plant_Leaf_{li}", (lx, ly, lz),
            (rng.uniform(0.035, 0.068), rng.uniform(0.022, 0.042), 0.005),
            lmat, rot_euler=(rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(0, 3.14)), bevel=False
        )

    # 6. Back-Right Cafe Counter, Wooden Wall Shelves, Mugs, Jars & Chalkboard
    _add_box(cafe_col, "Cafe_Counter_Top", (0.72, 1.88, 1.42), (1.35, 0.42, 0.045), mat_wood)
    _add_box(cafe_col, "Cafe_Counter_Front", (0.72, 1.69, 1.12), (1.32, 0.04, 0.58), mat_wood_dark)
    _add_box(cafe_col, "Cafe_Shelf_Lower", (0.76, 2.22, 1.72), (1.15, 0.22, 0.030), mat_wood)
    _add_box(cafe_col, "Cafe_Shelf_Upper", (0.76, 2.22, 1.98), (1.15, 0.22, 0.030), mat_wood)
    _add_box(cafe_col, "Cafe_Menu_Board", (0.48, 2.28, 2.18), (0.36, 0.02, 0.24), mat_chalk)

    # Mugs & canisters on shelves and counter
    for mi, mx in enumerate([0.36, 0.52, 0.68, 0.86, 1.02]):
        _add_cylinder(cafe_col, f"Cafe_Shelf_Jar_{mi}", (mx, 2.18, 1.78), 0.035, 0.09, mat_ceramic if mi % 2 == 0 else mat_wood_dark, segs=14)
        _add_cylinder(cafe_col, f"Cafe_Counter_Cup_{mi}", (0.30 + mi * 0.16, 1.82, 1.48), 0.032, 0.08, mat_metal if mi < 2 else mat_ceramic, segs=14)
    # Upper crown greenery hint near center-back ceiling
    for gi in range(12):
        _add_box(
            cafe_col, f"Cafe_Top_Plant_{gi}",
            (0.08 + rng.uniform(-0.14, 0.14), 1.65 + rng.uniform(-0.12, 0.12), 1.92 + rng.uniform(-0.08, 0.10)),
            (0.05, 0.03, 0.006), mat_leaf, rot_euler=(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0, 3.14)), bevel=False
        )


def _point_obj_at(obj, target, roll_deg=0.0):
    direction = Vector(target) - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()
    if abs(roll_deg) > 1e-4:
        # Roll around local viewing axis (-Z)
        mat = rot_quat.to_matrix() @ Matrix.Rotation(math.radians(roll_deg), 3, 'Z')
        obj.rotation_euler = mat.to_euler()


def build_lighting_and_world(light_col):
    """
    Build natural, flattering sunlit cafe portrait lighting:
    - Warm daylight window wrap key/rim from upper-left (-X)
    - Luminous soft front cafe fill producing natural iris catchlights
    - Warm honey-oak table bounce under chin & sweater
    - Gentle back hair fill so turnaround views have rich detail on all sides
    """
    # World shader: soft warm daylight cafe ambient
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("Cafe_World")
        bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    wnt = world.node_tree
    wnt.nodes.clear()
    w_out = wnt.nodes.new('ShaderNodeOutputWorld')
    w_bg = wnt.nodes.new('ShaderNodeBackground')
    w_bg.inputs['Color'].default_value = rgba_srgb(0.84, 0.82, 0.80)
    w_bg.inputs['Strength'].default_value = 0.55
    wnt.links.new(w_bg.outputs['Background'], w_out.inputs['Surface'])

    def make_area(name, loc, target, energy, size_x, size_y, srgb):
        ldata = bpy.data.lights.new(name=name + "_Data", type='AREA')
        ldata.shape = 'RECTANGLE'
        ldata.size = size_x
        ldata.size_y = size_y
        ldata.energy = energy
        ldata.color = rgba_srgb(*srgb)[:3]
        lobj = bpy.data.objects.new(name, ldata)
        lobj.location = loc
        _point_obj_at(lobj, target)
        light_col.objects.link(lobj)
        return lobj

    # 1. Sunlit Cafe Window Key Light (viewer's upper-left: wraps softly across forehead, cheeks & sweater!)
    l_key = make_area(
        "Light_Window_Key", (-0.95, -0.58, 1.78), (0.0, -0.14, 1.45),
        energy=46.0, size_x=0.90, size_y=1.15, srgb=(1.0, 0.965, 0.925)
    )
    # 2. Sunlit Window Rim / Backlight (viewer's back-left: creates luminous halo on hair wisps & left pink sleeve!)
    l_rim = make_area(
        "Light_Window_Rim", (-0.82, 0.42, 1.76), (-0.08, -0.14, 1.38),
        energy=42.0, size_x=0.85, size_y=1.10, srgb=(1.0, 0.96, 0.90)
    )
    # 3. Soft Front-Right Cafe Interior Fill + Eye Catchlight
    l_fill = make_area(
        "Light_Cafe_Front_Fill", (0.58, -1.15, 1.56), (0.0, -0.14, 1.48),
        energy=15.0, size_x=0.95, size_y=0.85, srgb=(0.99, 0.94, 0.89)
    )
    # 4. Warm Wood Table Bounce (lifts under-chin smile & sweater knit texture softly)
    l_bounce = make_area(
        "Light_Table_Warm_Bounce", (0.05, -0.65, 0.82), (0.0, -0.14, 1.38),
        energy=7.5, size_x=0.85, size_y=0.65, srgb=(0.98, 0.86, 0.74)
    )
    # 5. Gentle Back Hair Fill (for rich detail in 90° side and 180° back diagnostic views without over-brightening beauty)
    l_back = make_area(
        "Light_Back_Soft_Fill", (0.35, 1.45, 1.68), (0.0, 0.04, 1.32),
        energy=14.0, size_x=1.10, size_y=1.00, srgb=(0.98, 0.95, 0.92)
    )
    # 6. Dedicated Crisp Eye Catchlight (simulates bright cafe window reflection at 10:30 o'clock on corneas!)
    l_catch = make_area(
        "Light_Eye_Catchlight", (-0.24, -0.82, 1.66), (0.0, -0.14, 1.54),
        energy=4.2, size_x=0.16, size_y=0.16, srgb=(1.0, 0.98, 0.95)
    )

    return [l_key, l_rim, l_fill, l_bounce, l_back, l_catch], w_bg


def build_cameras(cam_col, focus_point):
    """
    Create:
    - `Beauty_Camera`: 32mm selfie portrait lens at (960x1040) matching input/reference.png's
      upper-body composition, raised-angle selfie perspective, gentle tilt, and f/2.2 optical bokeh.
    - Diagnostic turnaround cameras (`Cam_Front`, `Cam_ThreeQuarter`, `Cam_Side`, `Cam_Back`) at
      0, 45, 90, and 180 degrees around the character on a 65mm distortion-free lens (768x768).
    """
    # 1. Beauty Camera
    b_data = bpy.data.cameras.new("Beauty_Camera_Data")
    b_data.lens = 32.5
    b_data.sensor_width = 36.0
    b_data.dof.use_dof = True
    b_data.dof.aperture_fstop = 2.2
    b_cam = bpy.data.objects.new("Beauty_Camera", b_data)
    b_cam.location = (-0.012, -0.615, 1.535)
    _point_obj_at(b_cam, (-0.008, -0.115, 1.392), roll_deg=-4.2)
    b_data.dof.focus_distance = (Vector(focus_point) - b_cam.location).length
    cam_col.objects.link(b_cam)

    # 2. Diagnostic Turnaround Cameras (0°, 45°, 90°, 180° around character center)
    # Orbit center chosen so ENTIRE upper-body asset (Z=0.90..1.70, Y=-0.55..+0.16) fits cleanly in 768x768!
    orbit_center = Vector((0.0, -0.155, 1.305))
    radius = 1.88

    diag_specs = [
        ("Cam_Front", 0.0, "front.png"),
        ("Cam_ThreeQuarter", 45.0, "three-quarter.png"),
        ("Cam_Side", 90.0, "side.png"),
        ("Cam_Back", 180.0, "back.png"),
    ]
    diag_cams = {}
    for cname, deg, fname in diag_specs:
        cdata = bpy.data.cameras.new(cname + "_Data")
        cdata.lens = 65.0
        cdata.sensor_width = 36.0
        cdata.dof.use_dof = False
        cobj = bpy.data.objects.new(cname, cdata)
        rad = math.radians(deg)
        # 0° = (0, -R) front; 45° = (+X, -Y) three-quarter; 90° = (+X, 0) exposed left ear & face profile; 180° = (0, +R) back
        cobj.location = Vector((
            orbit_center.x + radius * math.sin(rad),
            orbit_center.y - radius * math.cos(rad),
            orbit_center.z + 0.03,
        ))
        _point_obj_at(cobj, orbit_center)
        cam_col.objects.link(cobj)
        diag_cams[fname] = cobj

    return b_cam, diag_cams
