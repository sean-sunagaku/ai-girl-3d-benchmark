"""Render the saved scene (character.blend) to renders/*.png.

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 \
        character.blend --python render.py [-- beauty front three-quarter side back] [--preview]

The beauty shot uses the cafe set exactly as saved.  The four diagnostic views use the
same character objects/materials; only the cafe set collection is excluded and a plain
grey world plus one extra back-fill light are enabled so the profile/back are readable.
"""
import bpy, sys, os, time

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
preview = "--preview" in argv
views = [a for a in argv if not a.startswith("--")] or ["beauty", "front", "three-quarter", "side", "back"]
ROOT = os.path.dirname(bpy.data.filepath)
OUT = os.path.join(ROOT, "previews" if preview else "renders")
os.makedirs(OUT, exist_ok=True)

sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.render.threads_mode = 'FIXED'
sc.render.threads = 3
sc.cycles.samples = 12 if preview else 32
sc.cycles.use_denoising = True
cafe = bpy.data.collections["Cafe_Set"]
diag_light = bpy.data.objects["Diag_Fill_Back"]
world = sc.world
bg = world.node_tree.nodes["Background"]
saved_bg = (tuple(bg.inputs[0].default_value), bg.inputs[1].default_value)

for v in views:
    t0 = time.time()
    if v == "beauty":
        cam = bpy.data.objects["Cam_Beauty"]
        cafe.hide_render = False
        diag_light.hide_render = True
        bg.inputs[0].default_value, bg.inputs[1].default_value = saved_bg
        sc.render.resolution_x, sc.render.resolution_y = 960, 1040
    else:
        cam = bpy.data.objects["Cam_" + v]
        cafe.hide_render = True
        diag_light.hide_render = False
        bg.inputs[0].default_value = (0.42, 0.42, 0.43, 1.0)
        bg.inputs[1].default_value = 0.55
        sc.render.resolution_x, sc.render.resolution_y = 768, 768
    sc.render.resolution_percentage = 50 if preview else 100
    sc.camera = cam
    sc.render.filepath = os.path.join(OUT, ("pv_" if preview else "") + v + ".png")
    bpy.ops.render.render(write_still=True)
    print("[render] %s -> %s (%.1fs)" % (v, sc.render.filepath, time.time() - t0), flush=True)
