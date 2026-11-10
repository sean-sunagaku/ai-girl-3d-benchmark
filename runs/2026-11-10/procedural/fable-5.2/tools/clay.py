"""Clay preview of the sculpted, un-posed head with placeholder eyeballs. Usage: Blender -b --factory-startup --python tools/clay.py -- tag"""
import sys, os, math
sys.path.insert(0, os.getcwd())
import bpy, numpy as np
from mathutils import Vector
from fable.base import Body
from fable import sculpt
tag = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else 'clay'
raw = '--raw' in sys.argv
bpy.ops.wm.read_factory_settings(use_empty=True)
b = Body('input')
if not raw:
    sculpt.sculpt_face(b)
ob = b.make_object('Body')
m = ob.modifiers.new('sub', 'SUBSURF'); m.levels = 2; m.render_levels = 2
def mat(name, col, rough=0.55, spec=0.3):
    mt = bpy.data.materials.new(name); mt.use_nodes = True
    p = mt.node_tree.nodes['Principled BSDF']; p.inputs['Base Color'].default_value = (*col, 1); p.inputs['Roughness'].default_value = rough
    return mt
ob.data.materials.append(mat('clay', (0.55, 0.50, 0.47)))
for side in (1, -1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sculpt.EYE_R, location=tuple(sculpt.EYE_C * np.array([side, 1, 1])), segments=48, ring_count=24)
    e = bpy.context.object; e.data.materials.append(mat('eye', (0.85, 0.85, 0.85), 0.2)); bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0066, location=tuple(sculpt.EYE_C * np.array([side, 1, 1]) + np.array([0, -0.0138, 0])), segments=24, ring_count=12)
    i = bpy.context.object; i.scale = (1, 0.25, 1); i.data.materials.append(mat('iris', (0.12, 0.07, 0.04), 0.3))
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 24; sc.cycles.use_denoising = True
sc.render.threads_mode = 'FIXED'; sc.render.threads = 3; sc.render.use_stamp_filename = False
sc.view_settings.view_transform = 'Standard'
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.5, 0.52, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.25
def light(name, loc, energy, size):
    l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size
    o = bpy.data.objects.new(name, l); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (Vector((0, -0.05, 1.52)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
light('key', (-0.7, -1.1, 2.0), 45, 0.9); light('fill', (1.0, -0.9, 1.4), 10, 1.2); light('rim', (0.6, 1.0, 1.9), 25, 0.6)
cd = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cd); sc.collection.objects.link(cam); sc.camera = cam
def shot(name, loc, target, ortho, res=(640, 640)):
    cam.location = loc; cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    cd.type = 'ORTHO'; cd.ortho_scale = ortho
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.filepath = '//previews/%s_%s.png' % (tag, name)
    bpy.ops.render.render(write_still=True)
T = (0, -0.06, 1.535)
shot('front', (0, -2, 1.535), (0, 0, 1.535), 0.27)
shot('q3', (-1.35, -1.45, 1.535), T, 0.27)
shot('side', (-2, -0.06, 1.535), T, 0.27)
shot('high', (-0.25, -1.75, 2.35), (0, -0.08, 1.52), 0.27)
shot('eye', (0.031, -2, 1.546), (0.031, 0, 1.546), 0.055, (640, 420))
shot('mouth', (-0.4, -1.9, 1.49), (0, -0.14, 1.485), 0.085, (640, 420))
