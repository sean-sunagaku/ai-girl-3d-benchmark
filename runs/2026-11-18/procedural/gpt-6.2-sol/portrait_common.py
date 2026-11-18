"""Geometry and material helpers for the Blender portrait."""
from math import exp
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent


def mix(a, b, t):
    return a * (1 - t) + b * t


def ease(a, b, value):
    t = min(1., max(0., (value - a) / (b - a)))
    return t * t * (3 - 2 * t)


def gauss(value, center, radius):
    return exp(-((value - center) / radius) ** 2)


def linear(color):
    return tuple(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in color)


def collection(name, parent=None):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def mesh_object(name, col, vertices, faces, material=None, smooth=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    if material:
        mesh.materials.append(material)
    if smooth:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    return obj


def subdivide(obj, level, name="Smooth surface"):
    mod = obj.modifiers.new(name, 'SUBSURF')
    mod.levels = mod.render_levels = level
    return mod


def point_color(mesh, name, colors):
    attr = mesh.color_attributes.new(name=name, type='FLOAT_COLOR', domain='POINT')
    attr.data.foreach_set('color', np.asarray(colors, dtype=np.float32).reshape(-1))


def surface(name, color, roughness=.5, **properties):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    color = (*linear(color), 1)
    mat.diffuse_color = color
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = roughness
    for key, value in properties.items():
        bsdf.inputs[key].default_value = value
    return mat, mat.node_tree, bsdf


def node(tree, type_code, title, xy):
    n = tree.nodes.new(type_code)
    n.label = title
    n.location = xy
    return n


def noise_bump(tree, bsdf, scale, strength, distance, coordinates=None, label="Microstructure"):
    noise = node(tree, 'ShaderNodeTexNoise', label, (-415, -185))
    noise.inputs['Scale'].default_value = scale
    noise.inputs['Detail'].default_value = 2.2
    if coordinates:
        tree.links.new(coordinates, noise.inputs['Vector'])
    bump = node(tree, 'ShaderNodeBump', label, (-140, -170))
    bump.inputs['Strength'].default_value = strength
    bump.inputs['Distance'].default_value = distance
    tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return bump


def native_strands(name, col, paths, radii, material):
    """Blender Curves, with true per-point radii for Cycles hair."""
    data = bpy.data.hair_curves.new(name)
    data.add_curves([len(p) for p in paths])
    data.position_data.foreach_set('vector', np.concatenate(paths).astype(np.float32).reshape(-1))
    radius = data.attributes.new('radius', 'FLOAT', 'POINT')
    radius.data.foreach_set('value', np.concatenate(radii).astype(np.float32))
    data.materials.append(material)
    obj = bpy.data.objects.new(name, data)
    col.objects.link(obj)
    return obj


def read_obj(path, transform=lambda p: p, group=None):
    vertices, faces = [], []
    group_name = None
    for line in path.read_text().splitlines():
        if line.startswith('v '):
            vertices.append(transform(tuple(map(float, line.split()[1:4]))))
        elif line.startswith('g '):
            group_name = line.split()[1]
        elif line.startswith('f ') and (group is None or group == group_name):
            faces.append(tuple(int(index.split('/')[0]) - 1 for index in line.split()[1:]))
    return vertices, faces


def subset(vertices, faces):
    kept = sorted({i for face in faces for i in face})
    index = {old: new for new, old in enumerate(kept)}
    return [vertices[i] for i in kept], [tuple(index[i] for i in face) for face in faces]


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def catmull(knots, count):
    """Interpolate a small list of 3D control points using a centripetal spline."""
    p = np.asarray(knots, dtype=float)
    params = np.concatenate(([0.], np.cumsum(np.maximum(1e-7, np.linalg.norm(np.diff(p, axis=0), axis=1)) ** .5)))
    delta = np.diff(params)
    tangent = np.empty_like(p)
    tangent[0] = (p[1] - p[0]) / delta[0]
    tangent[-1] = (p[-1] - p[-2]) / delta[-1]
    tangent[1:-1] = ((p[1:-1] - p[:-2]) / delta[:-1, None] * delta[1:, None]
                     + (p[2:] - p[1:-1]) / delta[1:, None] * delta[:-1, None]) / (delta[:-1, None] + delta[1:, None])
    s = np.linspace(0, params[-1], count)
    seg = np.clip(np.searchsorted(params, s, side='right') - 1, 0, len(delta)-1)
    t = ((s - params[seg]) / delta[seg])[:, None]
    t2, t3 = t*t, t*t*t
    return ((2*t3-3*t2+1)*p[seg] + (t3-2*t2+t)*delta[seg,None]*tangent[seg]
            + (-2*t3+3*t2)*p[seg+1] + (t3-t2)*delta[seg,None]*tangent[seg+1])
