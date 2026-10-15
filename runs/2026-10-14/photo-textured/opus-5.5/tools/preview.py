# usage: Blender -b file.blend --python tools/preview.py -- prefix engine(W|C) samples res view1 view2 ...
# view = name:cx,cy,cz:az,el:ortho_or_dist[:lens]
import bpy, sys, math
from mathutils import Vector
argv = sys.argv[sys.argv.index("--")+1:]
prefix, engine, samples, res = argv[0], argv[1], int(argv[2]), argv[3]
rx, ry = map(int, res.split("x"))
sc = bpy.context.scene
if engine == "W":
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.show_cavity = True
    sc.display.shading.color_type = __import__('os').environ.get('WBCOL','MATERIAL')
else:
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = samples
    sc.cycles.use_denoising = True; sc.render.threads_mode='FIXED'; sc.render.threads = 3
sc.render.resolution_x, sc.render.resolution_y = rx, ry
sc.render.resolution_percentage = 100
for v in argv[4:]:
    p = v.split(":")
    name = p[0]
    if name.startswith("cam="):
        sc.camera = bpy.data.objects[name[4:]]
        sc.render.filepath = "previews/%s_%s.png" % (prefix, name[4:])
        bpy.ops.render.render(write_still=True); continue
    c = Vector(map(float, p[1].split(","))); az, el = map(float, p[2].split(","))
    cam = bpy.data.objects.new("pv", bpy.data.cameras.new("pv")); sc.collection.objects.link(cam)
    d = Vector((math.sin(math.radians(az))*math.cos(math.radians(el)), -math.cos(math.radians(az))*math.cos(math.radians(el)), math.sin(math.radians(el))))
    if p[3].startswith("o"):
        cam.data.type = 'ORTHO'; cam.data.ortho_scale = float(p[3][1:]); dist = 2.0
    else:
        dist = float(p[3]); cam.data.lens = float(p[4]) if len(p) > 4 else 85
    cam.data.clip_start = 0.01
    cam.location = c + d*dist
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = cam
    sc.render.filepath = "previews/%s_%s.png" % (prefix, name)
    bpy.ops.render.render(write_still=True)
