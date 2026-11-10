"""Posed clay preview (body only). Blender -b --factory-startup --python tools/pose_preview.py -- tag"""
import sys, os, math, json
sys.path.insert(0, os.getcwd())
import bpy, numpy as np
from mathutils import Vector, Matrix
from fable.base import Body
from fable import sculpt, rig
tag = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else 'pose'
bpy.ops.wm.read_factory_settings(use_empty=True)
b = Body('input'); sculpt.sculpt_face(b)
tmp = b.make_object('tmp')
sk = rig.Skeleton(b.landmarks); W = sk.bone_heat_weights(tmp)
bpy.data.objects.remove(tmp, do_unlink=True)
rot, P = rig.build_pose(sk)
rig.arm_ik(sk, rot, 'r', P['r_wrist'], P['r_elbow'], hand_fwd=(0.25, -0.75, 0.6), hand_palm=(0.7, 0.3, -0.6))
rig.arm_ik(sk, rot, 'l', P['l_wrist'], P['l_elbow'], hand_fwd=(-0.2, -0.9, 0.35), hand_palm=(-0.6, 0.2, -0.75))
for s in ('l', 'r'):
    rig.finger_curl(sk, rot, s, {2: (18, 26, 14), 3: (24, 32, 18), 4: (30, 36, 20), 5: (36, 38, 22), 1: (8, 14, 10)})
M = sk.solve(rot)
Vp = sk.skin(b.V, W, M)
ob = b.make_object('Body', Vp)
m = ob.modifiers.new('sub', 'SUBSURF'); m.levels = 2; m.render_levels = 2
mt = bpy.data.materials.new('clay'); mt.use_nodes = True
mt.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.5, 0.45, 0.42, 1)
ob.data.materials.append(mt)
Mh = M['head']
for side in (1, -1):
    c = sk.apply(Mh, sculpt.EYE_C * np.array([side, 1, 1]))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sculpt.EYE_R, location=tuple(c), segments=32, ring_count=16)
    bpy.ops.object.shade_smooth()
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 16; sc.cycles.use_denoising = True
sc.render.threads_mode = 'FIXED'; sc.render.threads = 3; sc.render.use_stamp_filename = False
sc.view_settings.view_transform = 'Standard'
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.5, 0.52, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.35
def light(name, loc, energy, size):
    l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.size = size
    o = bpy.data.objects.new(name, l); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (Vector((0, -0.05, 1.45)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
light('key', (-0.9, -1.4, 2.2), 30, 1.0); light('fill', (1.2, -1.0, 1.5), 8, 1.4); light('rim', (0.6, 1.2, 2.0), 18, 0.8)
cd = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cd); sc.collection.objects.link(cam); sc.camera = cam
def ortho(name, az, scale=0.98, target=(0, -0.16, 1.40), elev=0.0, res=(560, 560)):
    a = math.radians(az); e = math.radians(elev)
    d = Vector((-math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    cam.location = Vector(target) + d * 3; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    cd.type = 'ORTHO'; cd.ortho_scale = scale
    sc.render.resolution_x, sc.render.resolution_y = res; sc.render.filepath = '//previews/%s_%s.png' % (tag, name)
    bpy.ops.render.render(write_still=True)
ortho('front', 0); ortho('q3', 45); ortho('side', 90); ortho('back', 180); ortho('top', 0, elev=75)
# beauty-like perspective camera
head_c = sk.apply(Mh, np.array([0, -0.10, 1.53]))
cd.type = 'PERSP'; cd.sensor_fit = 'VERTICAL'; cd.sensor_height = 24; cd.lens = 24 / (2 * math.tan(math.radians(74) / 2))
cpos = Vector((-0.085, -0.455, 1.705)); look = Vector((-0.02, -0.12, 1.40))
cam.location = cpos; cam.rotation_euler = (look - cpos).to_track_quat('-Z', 'Y').to_euler()
sc.render.resolution_x, sc.render.resolution_y = 480, 520; sc.render.filepath = '//previews/%s_beauty.png' % tag
bpy.ops.render.render(write_still=True)
# report projected landmarks for comparison with the photograph (pixel coordinates in the 1205x1306 frame)
from bpy_extras.object_utils import world_to_camera_view
bpy.context.view_layer.update()
def proj(p):
    v = world_to_camera_view(sc, cam, Vector(p)); return [round(v.x * 1205, 1), round((1 - v.y) * 1306, 1)]
pts = {'eye_R': sculpt.EYE_C * np.array([-1, 1, 1]) + np.array([0, -0.0149, 0]), 'eye_L': sculpt.EYE_C + np.array([0, -0.0149, 0])}
out = {k: proj(sk.apply(Mh, v)) for k, v in pts.items()}
for name, idx in (('nose_tip', 297), ('stomion', 474), ('chin', 732)):
    out[name] = proj(Vp[idx])
print('PROJ', json.dumps(out))
print('HEADC', head_c, 'dist cam-face', (Vector(head_c) - cpos).length)
