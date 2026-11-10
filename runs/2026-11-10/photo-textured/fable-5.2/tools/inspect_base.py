"""Import the supplied upper-body mesh and render quick clay previews for inspection.
Run: Blender --background --factory-startup --threads 3 --python tools/inspect_base.py
"""
import bpy, math, json
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath='input/upper-body.obj', forward_axis='Y', up_axis='Z')
ob = bpy.context.selected_objects[0]
print('IMPORTED', ob.name, len(ob.data.vertices), len(ob.data.polygons), [round(x,4) for x in ob.dimensions])
vs = [ob.matrix_world @ v.co for v in ob.data.vertices]
print('BOUNDS', [round(min(v[i] for v in vs),4) for i in range(3)], [round(max(v[i] for v in vs),4) for i in range(3)])
mod = ob.modifiers.new('sub', 'SUBSURF'); mod.levels = 1; mod.render_levels = 2
for p in ob.data.polygons: p.use_smooth = True
mat = bpy.data.materials.new('clay'); mat.use_nodes = True
b = mat.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (0.62,0.6,0.58,1); b.inputs['Roughness'].default_value = 0.6
ob.data.materials.append(mat)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 16; sc.cycles.use_denoising = True
sc.render.threads_mode = 'FIXED'; sc.render.threads = 3
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.75,0.75,0.78,1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.6
def light(name, loc, energy, size):
    l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size
    o = bpy.data.objects.new(name, l); sc.collection.objects.link(o); o.location = loc
    d = Vector((0,0,1.5)) - Vector(loc); o.rotation_euler = d.to_track_quat('-Z','Y').to_euler()
light('key', (-0.8,-1.2,2.1), 120, 0.8); light('fill', (1.0,-0.8,1.5), 40, 1.0); light('rim', (0.5,1.2,2.0), 60, 0.6)
cam_d = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cam_d); sc.collection.objects.link(cam); sc.camera = cam
def shot(name, loc, target, ortho, res=(700,700)):
    cam.location = loc; d = Vector(target) - Vector(loc); cam.rotation_euler = d.to_track_quat('-Z','Y').to_euler()
    cam_d.type = 'ORTHO'; cam_d.ortho_scale = ortho
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.filepath = '//previews/base_%s.png' % name
    sc.render.use_stamp_filename = False
    bpy.ops.render.render(write_still=True)
shot('face_front', (0,-2,1.53), (0,0,1.53), 0.30)
shot('face_side', (-2,-0.05,1.53), (0,-0.05,1.53), 0.30)
shot('face_34', (-1.4,-1.4,1.53), (0,-0.05,1.53), 0.30)
shot('body_front', (0,-3,1.38), (0,0,1.38), 0.95)
shot('body_side', (-3,0,1.38), (0,0,1.38), 0.95)
shot('eye_close', (0.031,-2,1.546), (0.031,0,1.546), 0.06)
shot('mouth_close', (0,-2,1.48), (0,0,1.48), 0.09)
