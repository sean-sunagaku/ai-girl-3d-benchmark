"""A deliberately simple, fully 3D cafe corner (window light, plants, wooden furniture, vase, shelves).

It only has to read as a softly lit cafe behind a shallow depth of field; the character has priority.
"""
import math
import numpy as np
import bpy
from mathutils import Vector, Euler
from . import util

FLOOR_Z = 0.415          # seated: the supplied upper body keeps its standing height
TABLE_Z = 1.135


def _mat(name, color, rough=0.6, spec=0.3, emission=None, strength=1.0, transmission=0.0, ior=1.45):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*color, 1); p.inputs['Roughness'].default_value = rough
    p.inputs['Specular IOR Level'].default_value = spec
    if emission is not None:
        p.inputs['Emission Color'].default_value = (*emission, 1); p.inputs['Emission Strength'].default_value = strength
    if transmission:
        p.inputs['Transmission Weight'].default_value = transmission; p.inputs['IOR'].default_value = ior
    m.diffuse_color = (*color, 1)
    return m


def _wood(name, light, dark, scale=1.0, rough=0.45):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord')
    mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (2.0 * scale, 24.0 * scale, 24.0 * scale)
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    n = nt.nodes.new('ShaderNodeTexNoise'); n.inputs['Scale'].default_value = 3.0; n.inputs['Detail'].default_value = 6.0; n.inputs['Distortion'].default_value = 1.2
    nt.links.new(mp.outputs['Vector'], n.inputs['Vector'])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (*dark, 1); mix.inputs['B'].default_value = (*light, 1)
    nt.links.new(n.outputs['Fac'], mix.inputs['Factor'])
    nt.links.new(mix.outputs['Result'], p.inputs['Base Color'])
    p.inputs['Roughness'].default_value = rough
    m.diffuse_color = (*light, 1)
    return m


def _box(name, lo, hi, mat, col, bevel=0.0):
    lo = Vector(lo); hi = Vector(hi)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(lo + hi) / 2)
    ob = bpy.context.object; ob.name = name; ob.scale = hi - lo
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        b = ob.modifiers.new('Bevel', 'BEVEL'); b.width = bevel; b.segments = 2
    ob.data.materials.append(mat); util.link(ob, col)
    return ob


def _cyl(name, base, radius, height, mat, col, verts=24, taper=1.0):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=radius, radius2=radius * taper, depth=height,
                                    location=(base[0], base[1], base[2] + height / 2))
    ob = bpy.context.object; ob.name = name; bpy.ops.object.shade_smooth()
    ob.data.materials.append(mat); util.link(ob, col)
    return ob


def _leaves(name, centre, spread, n, size, mat, col, rng, droop=0.5):
    V = []; F = []
    for k in range(n):
        d = rng.normal(0, 1, 3); d /= np.linalg.norm(d)
        d[2] = abs(d[2]) * 0.6 + 0.1
        base = np.array(centre) + rng.normal(0, 1, 3) * np.array(spread)
        tip_dir = d - np.array([0, 0, droop * rng.random()]); tip_dir /= np.linalg.norm(tip_dir)
        side = np.cross(tip_dir, [0, 0, 1.0]); side /= np.linalg.norm(side) + 1e-9
        L = size * rng.uniform(0.7, 1.3); Wd = L * 0.30
        nrm = np.cross(side, tip_dir)
        pts = [base, base + tip_dir * L * 0.28 + side * Wd * 0.85, base + tip_dir * L * 0.62 + side * Wd + nrm * L * 0.05,
               base + tip_dir * L + nrm * L * 0.10, base + tip_dir * L * 0.62 - side * Wd + nrm * L * 0.05, base + tip_dir * L * 0.28 - side * Wd * 0.85]
        i0 = len(V); V += pts; F.append(tuple(range(i0, i0 + 6)))
    ob = util.mesh_object(name, V, F, col, smooth=False); ob.data.materials.append(mat)
    return ob


def _chair(name, pos, yaw, wood, col):
    """Simple wooden cafe chair; pos = seat centre on the floor plane."""
    parts = []
    seat_z = FLOOR_Z + 0.45
    def add(lo, hi, bevel=0.006):
        parts.append(_box(name + '_p%d' % len(parts), lo, hi, wood, col, bevel))
    add((-0.21, -0.21, seat_z - 0.02), (0.21, 0.21, seat_z + 0.02))
    for sx in (-1, 1):
        for sy in (-1, 1):
            add((sx * 0.19 - 0.018, sy * 0.19 - 0.018, FLOOR_Z), (sx * 0.19 + 0.018, sy * 0.19 + 0.018, seat_z))
        add((sx * 0.19 - 0.018, 0.19 - 0.018, seat_z), (sx * 0.19 + 0.018, 0.19 + 0.018, seat_z + 0.44))
    add((-0.21, 0.175, seat_z + 0.36), (0.21, 0.205, seat_z + 0.45), 0.01)
    add((-0.20, 0.180, seat_z + 0.20), (0.20, 0.200, seat_z + 0.25), 0.008)
    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    ob = bpy.context.object; ob.name = name
    ob.location = (pos[0], pos[1], 0.0); ob.rotation_euler = (0, 0, math.radians(yaw))
    return ob


def build(col, rng):
    wood_l = _wood('Cafe_Wood_Light', (0.60, 0.40, 0.22), (0.40, 0.24, 0.12))
    wood_d = _wood('Cafe_Wood_Warm', (0.50, 0.31, 0.16), (0.30, 0.17, 0.08), 0.7)
    wall = _mat('Cafe_Wall', (0.72, 0.62, 0.50), 0.9)
    floor = _mat('Cafe_Floor', (0.36, 0.33, 0.30), 0.7)
    white = _mat('Cafe_Ceramic', (0.86, 0.85, 0.82), 0.25)
    metal = _mat('Cafe_Metal', (0.45, 0.45, 0.46), 0.3); metal.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.9
    glass = _mat('Cafe_Glass', (1, 1, 1), 0.02, transmission=1.0)
    leaf = _mat('Cafe_Leaf', (0.16, 0.36, 0.10), 0.5)
    leaf2 = _mat('Cafe_Leaf_Light', (0.30, 0.50, 0.16), 0.5)
    for lm in (leaf, leaf2):
        lm.node_tree.nodes['Principled BSDF'].inputs['Subsurface Weight'].default_value = 0.4
        lm.node_tree.nodes['Principled BSDF'].inputs['Subsurface Radius'].default_value = (0.3, 0.8, 0.2)
        lm.node_tree.nodes['Principled BSDF'].inputs['Subsurface Scale'].default_value = 0.02
    petal = _mat('Cafe_Petal', (0.95, 0.94, 0.88), 0.6)
    window = _mat('Cafe_Window_Daylight', (1, 1, 1), 0.5, emission=(1.0, 0.97, 0.90), strength=4.2)
    dark = _mat('Cafe_Frame_Dark', (0.06, 0.06, 0.07), 0.5)

    # room shell
    _box('Cafe_Floor', (-3.2, -2.5, FLOOR_Z - 0.05), (3.2, 3.0, FLOOR_Z), floor, col)
    _box('Cafe_Wall_Back', (-0.55, 2.55, FLOOR_Z), (3.2, 2.65, 3.2), wall, col)
    _box('Cafe_Wall_Right', (3.1, -2.5, FLOOR_Z), (3.2, 2.6, 3.2), wall, col)
    _box('Cafe_Ceiling', (-3.2, -2.5, 3.15), (3.2, 3.0, 3.2), wall, col)
    # glazed corner on her right (image left): bright panes with dark mullions
    _box('Cafe_Window_Back', (-3.1, 2.62, FLOOR_Z + 0.25), (-0.55, 2.64, 3.0), window, col)
    _box('Cafe_Window_Side', (-3.12, -1.2, FLOOR_Z + 0.25), (-3.10, 2.62, 3.0), window, col)
    for x in (-2.45, -1.75, -1.05, -0.57):
        _box('Cafe_Mullion', (x - 0.022, 2.55, FLOOR_Z), (x + 0.022, 2.62, 3.1), dark, col)
    for y in (-0.6, 0.5, 1.6):
        _box('Cafe_Mullion', (-3.10, y - 0.022, FLOOR_Z), (-3.03, y + 0.022, 3.1), dark, col)
    _box('Cafe_Sill', (-3.1, 2.50, FLOOR_Z + 0.18), (-0.55, 2.62, FLOOR_Z + 0.25), dark, col)
    # tall plants near the glass
    for i, (px, py, h) in enumerate(((-1.25, 1.75, 1.75), (-2.1, 1.2, 1.45), (-0.25, 2.25, 1.9))):
        _cyl('Cafe_Planter', (px, py, FLOOR_Z), 0.19, 0.42, white, col, taper=1.12)
        _cyl('Cafe_Trunk', (px, py, FLOOR_Z + 0.4), 0.016, h - 0.5, wood_d, col, 10)
        _leaves('Cafe_Plant_Leaves', (px, py, FLOOR_Z + h * 0.72), (0.24, 0.24, h * 0.20), 320, 0.105, leaf if i % 2 == 0 else leaf2, col, rng)
        _leaves('Cafe_Plant_Leaves', (px + 0.05, py, FLOOR_Z + h * 0.80), (0.20, 0.20, h * 0.16), 200, 0.09, leaf2, col, rng)
    # chairs: the one she sits on (back rail behind her) and neighbours
    _chair('Cafe_Chair_Own', (0.0, 0.10), 4.0, wood_l, col)
    _chair('Cafe_Chair_Left', (-0.80, 0.55), -18.0, wood_l, col)
    _chair('Cafe_Chair_Back', (1.05, 1.75), 160.0, wood_l, col)
    # her own table (front right) and the neighbouring table with a vase
    _box('Cafe_Table_Front_Top', (-0.62, -1.10, TABLE_Z - 0.035), (1.10, -0.215, TABLE_Z), wood_l, col, 0.006)
    _cyl('Cafe_Table_Front_Leg', (0.25, -0.68, FLOOR_Z), 0.035, TABLE_Z - FLOOR_Z - 0.03, dark, col)
    _box('Cafe_Table_Side_Top', (0.62, 0.55, TABLE_Z - 0.035), (1.55, 1.45, TABLE_Z), wood_l, col, 0.006)
    _cyl('Cafe_Table_Side_Leg', (1.08, 1.0, FLOOR_Z), 0.035, TABLE_Z - FLOOR_Z - 0.03, dark, col)
    vx, vy = 0.92, 0.88
    _cyl('Cafe_Vase', (vx, vy, TABLE_Z), 0.034, 0.13, glass, col, taper=0.8)
    _cyl('Cafe_Vase_Water', (vx, vy, TABLE_Z + 0.004), 0.030, 0.06, _mat('Cafe_Water', (0.9, 0.95, 0.95), 0.05, transmission=1.0, ior=1.33), col, taper=0.9)
    stems = []; flowers = []
    for k in range(11):
        a = rng.uniform(0, 2 * math.pi); lean = rng.uniform(0.05, 0.22); hgt = rng.uniform(0.20, 0.34)
        top = (vx + math.cos(a) * lean * hgt, vy + math.sin(a) * lean * hgt, TABLE_Z + 0.05 + hgt)
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.0022, depth=hgt)
        st = bpy.context.object; st.name = 'Cafe_Stem'
        st.location = ((vx + top[0]) / 2, (vy + top[1]) / 2, TABLE_Z + 0.05 + hgt / 2)
        st.rotation_euler = Vector((top[0] - vx, top[1] - vy, hgt)).to_track_quat('Z', 'Y').to_euler()
        st.data.materials.append(leaf2); util.link(st, col)
        for j in range(rng.integers(3, 7)):
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=rng.uniform(0.008, 0.014),
                                                  location=(top[0] + rng.normal(0, 0.016), top[1] + rng.normal(0, 0.016), top[2] + rng.normal(0, 0.014)))
            fl = bpy.context.object; fl.name = 'Cafe_Flower'; fl.data.materials.append(petal); util.link(fl, col)
    _leaves('Cafe_Vase_Leaves', (vx, vy, TABLE_Z + 0.20), (0.035, 0.035, 0.05), 26, 0.06, leaf2, col, rng, 0.2)
    # counter and shelves on the right/back wall
    _box('Cafe_Counter', (1.35, 1.95, FLOOR_Z), (3.1, 2.55, FLOOR_Z + 1.02), wood_d, col, 0.01)
    for k in range(14):
        x = 1.38 + k * 0.125
        _box('Cafe_Counter_Slat', (x, 1.93, FLOOR_Z + 0.05), (x + 0.06, 1.95, FLOOR_Z + 0.98), wood_l, col)
    _box('Cafe_Counter_Top', (1.30, 1.90, FLOOR_Z + 1.02), (3.1, 2.55, FLOOR_Z + 1.06), wood_l, col, 0.006)
    for z in (1.95, 2.30, 2.65):
        _box('Cafe_Shelf', (0.55, 2.30, z), (3.1, 2.55, z + 0.03), wood_l, col, 0.004)
    for x in (0.57, 1.80, 3.05):
        _box('Cafe_Shelf_Post', (x - 0.02, 2.30, 1.55), (x + 0.02, 2.55, 3.0), wood_l, col)
    for z in (1.98, 2.33, 2.68):
        x = 0.68
        while x < 3.0:
            r = rng.uniform(0.028, 0.05); hgt = rng.uniform(0.07, 0.17)
            m = (white, metal, glass, _mat('Cafe_Jar', tuple(rng.uniform(0.25, 0.8, 3)), 0.4))[int(rng.integers(0, 4))]
            _cyl('Cafe_Shelf_Item', (x, 2.43 + rng.normal(0, 0.02), z), r, hgt, m, col, 14, rng.uniform(0.8, 1.0))
            x += r * 2 + rng.uniform(0.03, 0.12)
    for (x, hgt) in ((1.55, 0.30), (1.80, 0.34), (2.3, 0.26), (2.55, 0.22)):
        _cyl('Cafe_Counter_Item', (x, 2.25, FLOOR_Z + 1.06), 0.06, hgt, metal if hgt > 0.28 else white, col, 16, 0.92)
    _box('Cafe_Menu_Board', (1.2, 2.53, 2.72), (2.1, 2.55, 3.05), dark, col)
