"""
cafe_env.py - Sunlit 3D cafe background props (potted plant, oak chair back, cafe table
with white-flower glass vase, rear coffee bar counter & shelves, sunlit left window),
natural portrait lighting rig, and beauty + 4 diagnostic turnaround cameras.
"""
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler


def _make_box(name, size, loc, rot=(0.0, 0.0, 0.0), mat=None, col=None, bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, verts=bm.verts, vec=size)
    if bevel > 0.0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=2, affect='EDGES', clamp_overlap=True)
    me = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = (bevel > 0.0)
    obj = bpy.data.objects.new(name, me)
    obj.location = loc
    obj.rotation_euler = rot
    if mat:
        obj.data.materials.append(mat)
    if col:
        col.objects.link(obj)
    return obj


def _make_cyl(name, radius, depth, loc, rot=(0.0, 0.0, 0.0), segs=24, mat=None, col=None):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=radius, radius2=radius, depth=depth)
    me = bpy.data.meshes.new(f"{name}_Mesh")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    obj.location = loc
    obj.rotation_euler = rot
    if mat:
        obj.data.materials.append(mat)
    if col:
        col.objects.link(obj)
    return obj


def _build_potted_plant(base_loc, mats, col):
    """Build a sunlit leafy potted cafe ficus plant (viewer's left background bokeh)."""
    bx, by, bz = base_loc
    # Ceramic pot
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=28, radius1=0.14, radius2=0.19, depth=0.34)
    me_pot = bpy.data.meshes.new("Cafe_PlantPot_Mesh")
    bm.to_mesh(me_pot)
    bm.free()
    for p in me_pot.polygons:
        p.use_smooth = True
    pot = bpy.data.objects.new("Cafe_PlantPot", me_pot)
    pot.location = (bx, by, bz + 0.17)
    pot.data.materials.append(mats["pot_ceramic"])
    col.objects.link(pot)

    # Branch stems + clustered 3D leaves
    bm_l = bmesh.new()
    rng = np.random.default_rng(909)
    n_leaves = 130
    for i in range(n_leaves):
        h = rng.uniform(0.32, 1.15)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = rng.uniform(0.04, 0.34) * (0.55 + 0.45 * math.sin((h - 0.3) / 0.9 * math.pi))
        lx = bx + rad * math.cos(ang)
        ly = by + rad * math.sin(ang)
        lz = bz + h
        leaf_len = rng.uniform(0.065, 0.115)
        leaf_w = leaf_len * rng.uniform(0.45, 0.65)
        rot = Euler((rng.uniform(-0.8, 0.8), rng.uniform(-0.7, 0.7), ang + rng.uniform(-0.6, 0.6))).to_matrix()

        # Simple 4-vert diamond leaf
        pts = [
            Vector((0.0, 0.0, 0.0)),
            Vector((leaf_len * 0.45, leaf_w * 0.5, 0.008)),
            Vector((leaf_len, 0.0, -0.006)),
            Vector((leaf_len * 0.45, -leaf_w * 0.5, 0.008)),
        ]
        vs = [bm_l.verts.new(Vector((lx, ly, lz)) + rot @ p) for p in pts]
        f = bm_l.faces.new(vs)
        f.smooth = True

    me_l = bpy.data.meshes.new("Cafe_PlantLeaves_Mesh")
    bm_l.to_mesh(me_l)
    bm_l.free()
    leaves = bpy.data.objects.new("Cafe_PlantLeaves", me_l)
    leaves.data.materials.append(mats["leaf_green"])
    col.objects.link(leaves)
    return [pot, leaves]


def build_cafe_environment(cafe_mats, collection):
    """Build lightweight 3D sunlit cafe interior in its own collection for beauty portrait bokeh."""
    objs = []
    # Cream cafe back wall & right wall
    objs.append(_make_box("Cafe_Wall_Back", (4.5, 0.12, 3.4), (0.3, 2.15, 1.55), mat=cafe_mats["wall_cream"], col=collection))
    objs.append(_make_box("Cafe_Wall_Right", (0.12, 3.6, 3.4), (1.85, 0.8, 1.55), mat=cafe_mats["wall_cream"], col=collection))

    # Bright sunlit tall window panel on the back-left (X < -0.55) matching reference.png's bright left window!
    objs.append(_make_box("Cafe_Window_LightPanel", (1.6, 0.05, 2.5), (-1.15, 1.85, 1.55), rot=(0.0, 0.0, math.radians(-22)), mat=cafe_mats["window_glow"], col=collection))
    # Window mullion frames
    objs.append(_make_box("Cafe_Window_FrameV", (0.06, 0.08, 2.5), (-0.75, 1.68, 1.55), rot=(0.0, 0.0, math.radians(-22)), mat=cafe_mats["wood_dark"], col=collection))
    objs.append(_make_box("Cafe_Window_FrameH", (1.6, 0.08, 0.06), (-1.15, 1.82, 1.32), rot=(0.0, 0.0, math.radians(-22)), mat=cafe_mats["wood_dark"], col=collection))

    # Potted lush green plant in front of sunlit window (viewer's left behind right shoulder)
    objs.extend(_build_potted_plant((-0.68, 1.05, 0.88), cafe_mats, collection))
    # High hanging greenery hint near top center-right
    objs.extend(_build_potted_plant((0.18, 1.55, 1.55), cafe_mats, collection))

    # Wooden cafe chair back behind her right side
    objs.append(_make_box("Cafe_Chair_BackTop", (0.48, 0.04, 0.11), (-0.28, 0.28, 1.24), rot=(0.08, -0.05, 0.25), mat=cafe_mats["wood_oak"], col=collection, bevel=0.008))
    objs.append(_make_cyl("Cafe_Chair_PostL", 0.018, 0.48, (-0.47, 0.24, 1.04), rot=(0.08, 0.0, 0.25), mat=cafe_mats["wood_oak"], col=collection))

    # Warm oak foreground table edge at very bottom & mid-right cafe table with white flower glass vase
    objs.append(_make_box("Cafe_Table_Right", (0.85, 0.72, 0.04), (0.72, 0.72, 1.20), rot=(0.0, 0.0, 0.08), mat=cafe_mats["wood_oak"], col=collection, bevel=0.006))
    objs.append(_make_box("Cafe_Chair_Right", (0.46, 0.04, 0.09), (0.68, 1.08, 1.34), rot=(0.0, 0.0, -0.10), mat=cafe_mats["wood_oak"], col=collection, bevel=0.006))

    # Glass vase + small white flowers on right table (visible soft bokeh right of head in reference.png!)
    objs.append(_make_cyl("Cafe_Vase", 0.042, 0.14, (0.58, 0.72, 1.29), mat=cafe_mats["glass_vase"], col=collection))
    rng = np.random.default_rng(1212)
    for k in range(12):
        fx = 0.58 + rng.uniform(-0.06, 0.06)
        fy = 0.72 + rng.uniform(-0.05, 0.05)
        fz = 1.40 + rng.uniform(0.0, 0.12)
        bm_f = bmesh.new()
        bmesh.ops.create_icosphere(bm_f, subdivisions=1, radius=rng.uniform(0.014, 0.022))
        me_f = bpy.data.meshes.new(f"Cafe_Flower_{k}_Mesh")
        bm_f.to_mesh(me_f)
        bm_f.free()
        fo = bpy.data.objects.new(f"Cafe_Flower_{k}", me_f)
        fo.location = (fx, fy, fz)
        fo.data.materials.append(cafe_mats["flower_white"] if k % 2 == 0 else cafe_mats["leaf_green"])
        collection.objects.link(fo)
        objs.append(fo)

    # Rear oak coffee bar counter & wall shelves on back-right (matching reference.png top-right!)
    objs.append(_make_box("Cafe_Counter_Base", (1.6, 0.45, 0.95), (0.82, 1.75, 1.00), mat=cafe_mats["wood_dark"], col=collection))
    objs.append(_make_box("Cafe_Counter_Top", (1.65, 0.50, 0.05), (0.82, 1.74, 1.48), mat=cafe_mats["wood_oak"], col=collection))
    objs.append(_make_box("Cafe_Shelf_1", (1.35, 0.22, 0.035), (0.85, 2.02, 1.82), mat=cafe_mats["wood_oak"], col=collection))
    objs.append(_make_box("Cafe_Shelf_2", (1.35, 0.22, 0.035), (0.85, 2.02, 2.12), mat=cafe_mats["wood_oak"], col=collection))
    # Metallic espresso machine / canisters / mugs silhouettes for authentic cafe shelf bokeh
    objs.append(_make_box("Cafe_EspressoMachine", (0.32, 0.28, 0.28), (0.52, 1.72, 1.64), mat=cafe_mats["metal_cafe"], col=collection, bevel=0.01))
    for j, sx in enumerate([0.45, 0.62, 0.78, 0.95, 1.12]):
        objs.append(_make_cyl(f"Cafe_Mug_{j}", 0.038, 0.095, (sx, 1.98, 1.89), mat=cafe_mats["pot_ceramic" if j % 2 == 0 else "wood_dark"], col=collection))
        objs.append(_make_cyl(f"Cafe_Glass_{j}", 0.032, 0.11, (sx + 0.05, 1.98, 2.19), mat=cafe_mats["metal_cafe" if j % 2 == 0 else "pot_ceramic"], col=collection))

    return objs


def aim_object_at(obj, target_pt):
    direction = Vector(target_pt) - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


def build_lighting_rig(collection):
    """
    Create natural sunlit cafe portrait lighting:
      - Bright soft daylight window key from viewer's front-left (-X, -Y, +Z)
      - Warm cafe fill bounce from front-right (+X, -Y)
      - Sunlit rim / hair backlight from back-left (-X, +Y, +Z) creating translucent chestnut hair glow
      - Subtle soft frontal eye catchlight / bounce card so both warm brown eyes sparkle vividly
      - Soft warm-neutral world environment (adjustable between cafe beauty & plain diagnostic backdrop)
    """
    lights = {}

    def add_area(name, loc, target, energy, color, size_x, size_y=None, spread=math.radians(115)):
        ld = bpy.data.lights.new(name=name, type='AREA')
        ld.energy = energy
        ld.color = color
        ld.shape = 'RECTANGLE' if size_y is not None else 'DISK'
        ld.size = size_x
        if size_y is not None:
            ld.size_y = size_y
        if hasattr(ld, "spread"):
            ld.spread = spread
        obj = bpy.data.objects.new(name, ld)
        obj.location = loc
        aim_object_at(obj, target)
        collection.objects.link(obj)
        return obj

    face_target = (0.0, -0.10, 1.48)

    # 1. Primary Sunlit Window Key Light (viewer's left-front, soft large window)
    lights["key"] = add_area(
        "Light_Window_Key",
        loc=(-1.42, -0.92, 1.96),
        target=face_target,
        energy=43.0,
        color=(1.0, 0.96, 0.91),
        size_x=1.40,
        size_y=1.90,
        spread=math.radians(102),
    )

    # 2. Warm Soft Cafe Fill (viewer's right-front, gentle shadow fill preserving nose & cheek modeling)
    lights["fill"] = add_area(
        "Light_Cafe_Fill",
        loc=(1.48, -1.15, 1.50),
        target=face_target,
        energy=11.0,
        color=(1.0, 0.92, 0.84),
        size_x=1.8,
        size_y=1.6,
        spread=math.radians(125),
    )

    # 3. Sunlit Back-Left Rim Light (creates warm backlit chestnut hair highlight & shoulder rim)
    lights["rim"] = add_area(
        "Light_Sun_Rim",
        loc=(-1.15, 0.85, 2.02),
        target=(0.0, 0.0, 1.48),
        energy=50.0,
        color=(1.0, 0.93, 0.82),
        size_x=1.2,
        size_y=1.5,
        spread=math.radians(88),
    )

    # 4. Soft Back-Right Fill (clean modeling on back/side hair and knit in diagnostic views)
    lights["back_fill"] = add_area(
        "Light_Back_Soft",
        loc=(0.95, 1.40, 1.85),
        target=(0.0, 0.0, 1.40),
        energy=22.0,
        color=(0.96, 0.95, 1.0),
        size_x=1.6,
        size_y=1.4,
        spread=math.radians(125),
    )

    # 5. Front-Upper Window Catchlight Reflector (gives bright lifelike sparkle in corneas)
    lights["catchlight"] = add_area(
        "Light_Eye_Catch",
        loc=(-0.24, -0.95, 1.82),
        target=(0.0, -0.14, 1.54),
        energy=9.5,
        color=(1.0, 0.99, 0.97),
        size_x=0.30,
        size_y=0.34,
        spread=math.radians(78),
    )

    # World Shader (warm soft cafe ambient by default)
    world = bpy.data.worlds.new("World_Portrait")
    bpy.context.scene.world = world
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.name = "World_BG"
    bg.inputs["Color"].default_value = (0.85, 0.81, 0.76, 1.0)
    bg.inputs["Strength"].default_value = 0.20
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    return lights, world


def build_cameras(collection, eye_world_center):
    """
    Create the 5 named cameras:
      - Camera_Beauty: selfie portrait matching input/reference.png framing (960x1040, shallow DoF on eyes)
      - Camera_Front (0 deg), Camera_ThreeQuarter (45 deg), Camera_Side (90 deg), Camera_Back (180 deg)
        for clear full upper-body diagnostic inspection at 768x768.
    """
    cams = {}

    # 1. Beauty Camera (flattering close selfie framing matching reference.png!)
    cd_b = bpy.data.cameras.new("Camera_Beauty_Data")
    cd_b.lens = 38.0
    cd_b.sensor_width = 36.0
    cd_b.clip_start = 0.05
    cd_b.dof.use_dof = True
    cd_b.dof.aperture_fstop = 2.6
    cam_b = bpy.data.objects.new("Camera_Beauty", cd_b)
    cam_b.location = Vector((0.010, -0.66, 1.48))
    aim_object_at(cam_b, (0.000, -0.04, 1.385))
    cd_b.dof.focus_distance = (Vector(eye_world_center) - cam_b.location).length
    collection.objects.link(cam_b)
    cams["beauty"] = cam_b

    # 2. Diagnostic Turnaround Cameras (85mm portrait lens at 1.48m radius around upper-body center)
    # Center of upper-body asset: (0.0, -0.05, 1.355) -> captures entire head, hair, sweater, and reaching sleeves cleanly!
    diag_center = Vector((0.0, -0.07, 1.355))
    diag_radius = 1.62
    diag_specs = [
        ("front", "Camera_Front", 0.0),
        ("three-quarter", "Camera_ThreeQuarter", 45.0),
        ("side", "Camera_Side", 90.0),
        ("back", "Camera_Back", 180.0),
    ]
    for key, name, deg in diag_specs:
        rad = math.radians(deg)
        # 0 deg = -Y (front), 45 deg = +X/-Y (3/4 showing visible left ear + face), 90 deg = +X (side profile), 180 deg = +Y (back)
        cx = diag_center.x + diag_radius * math.sin(rad)
        cy = diag_center.y - diag_radius * math.cos(rad)
        cz = diag_center.z + 0.04
        cd = bpy.data.cameras.new(f"{name}_Data")
        cd.lens = 78.0
        cd.sensor_width = 36.0
        cd.clip_start = 0.05
        cd.dof.use_dof = False
        co = bpy.data.objects.new(name, cd)
        co.location = Vector((cx, cy, cz))
        aim_object_at(co, diag_center)
        collection.objects.link(co)
        cams[key] = co

    return cams
