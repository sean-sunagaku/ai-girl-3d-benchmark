"""Preview renders from a saved scene.
Blender -b character.blend --threads 3 --python tools/shot.py -- --tag t1 --shots beauty,face --scale 0.5 --samples 16
shots: beauty front three-quarter side back face (tight on the face from the beauty direction) face_front face_side face_q3 eye
"""
import sys, os, math, argparse
import bpy
from mathutils import Vector
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='t'); ap.add_argument('--shots', default='beauty')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--samples', type=int, default=16)
ap.add_argument('--exposure', type=float, default=None); ap.add_argument('--studio', action='store_true')
a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
sc = bpy.context.scene
def set_mode(mode):
    beauty = mode == 'beauty'
    for name, vis in (('Cafe_Set', beauty), ('Lights_Portrait', beauty), ('Lights_Diagnostic_Studio', not beauty)):
        c = bpy.data.collections.get(name)
        if c is not None: c.hide_render = not vis
    sc.world = bpy.data.worlds['World_Cafe' if beauty else 'World_Studio']
sc.cycles.samples = a.samples; sc.render.threads_mode = 'FIXED'; sc.render.threads = 3
if a.exposure is not None: sc.view_settings.exposure = a.exposure
os.makedirs('previews', exist_ok=True)
beauty = bpy.data.objects['Cam_Beauty']
head = Vector((-0.045, -0.115, 1.525))
def temp_cam(name, pos, target, lens=None, ortho=None):
    d = bpy.data.cameras.new(name); o = bpy.data.objects.new(name, d); sc.collection.objects.link(o)
    o.location = pos; o.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y').to_euler()
    if ortho: d.type = 'ORTHO'; d.ortho_scale = ortho
    else: d.lens = lens
    d.clip_start = 0.01
    return o
for s in a.shots.split(','):
    if s == 'beauty': cam = beauty; res = (960, 1040)
    elif s in ('front', 'three-quarter', 'side', 'back'): cam = bpy.data.objects['Cam_diagnostic_' + s]; res = (768, 768)
    elif s == 'face':
        p = Vector(beauty.location); cam = temp_cam('c_face', p, head, lens=62); res = (900, 900)
    elif s == 'face_front': cam = temp_cam('c_ff', head + Vector((0, -3, 0)), head, ortho=0.30); res = (900, 900)
    elif s == 'face_q3': cam = temp_cam('c_fq', head + Vector((2.1, -2.1, 0)), head, ortho=0.30); res = (900, 900)
    elif s == 'face_q3r': cam = temp_cam('c_fqr', head + Vector((-2.1, -2.1, 0)), head, ortho=0.30); res = (900, 900)
    elif s == 'face_side': cam = temp_cam('c_fs', head + Vector((3, 0, 0)), head, ortho=0.30); res = (900, 900)
    elif s == 'head':
        p = Vector(beauty.location); cam = temp_cam('c_head', p, head + Vector((0, 0, 0.03)), lens=34); res = (900, 900)
    elif s == 'top':
        cam = temp_cam('c_top', head + Vector((-0.25, -0.9, 1.6)), head + Vector((0, 0.02, 0.1)), ortho=0.36); res = (900, 900)
    elif s == 'crown':
        p = Vector(beauty.location); cam = temp_cam('c_crown', p, head + Vector((-0.01, 0.0, 0.12)), lens=70); res = (900, 700)
    elif s == 'eye':
        p = Vector(beauty.location); cam = temp_cam('c_eye', p, head + Vector((0.0, -0.02, 0.012)), lens=170); res = (1000, 620)
    else: continue
    sc.camera = cam
    set_mode('beauty' if (s in ('beauty', 'face', 'head', 'eye', 'crown') and not a.studio) else 'diagnostic')
    sc.render.resolution_x = int(res[0] * a.scale); sc.render.resolution_y = int(res[1] * a.scale); sc.render.resolution_percentage = 100
    sc.render.filepath = os.path.join(os.getcwd(), 'previews', '%s_%s.png' % (a.tag, s))
    bpy.ops.render.render(write_still=True)
