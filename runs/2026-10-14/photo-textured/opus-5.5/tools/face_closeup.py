"""Development helper: render the face region of Cam_Beauty at full resolution (border crop).

    Blender -b scene.blend --python tools/face_closeup.py -- out.png [samples] [view]
view = beauty (default) or any Cam_* name suffix.
"""
import bpy, sys, os
a = sys.argv[sys.argv.index("--") + 1:]
out = a[0]; spp = int(a[1]) if len(a) > 1 else 16; view = a[2] if len(a) > 2 else "beauty"
sc = bpy.context.scene
sc.cycles.samples = spp
sc.cycles.use_denoising = True
sc.render.threads_mode = 'FIXED'; sc.render.threads = 3
if view == "beauty":
    sc.camera = bpy.data.objects["Cam_Beauty"]
    sc.render.resolution_x, sc.render.resolution_y = 960, 1040
    sc.render.use_border = True; sc.render.use_crop_to_border = True
    sc.render.border_min_x, sc.render.border_max_x = 0.28, 0.75
    sc.render.border_min_y, sc.render.border_max_y = 0.45, 0.88
else:
    sc.camera = bpy.data.objects["Cam_" + view]
    sc.render.resolution_x, sc.render.resolution_y = 768, 768
    if "Cafe_Set" in bpy.data.collections: bpy.data.collections["Cafe_Set"].hide_render = True
    if "Diag_Fill_Back" in bpy.data.objects: bpy.data.objects["Diag_Fill_Back"].hide_render = False
sc.render.resolution_percentage = 100
sc.render.filepath = os.path.abspath(out)
bpy.ops.render.render(write_still=True)
