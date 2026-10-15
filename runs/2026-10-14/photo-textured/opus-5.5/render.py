"""Render the saved scene (character.blend) to renders/*.png.

Called by build.py after it has saved and reopened character.blend.  It can also be run on
its own to re-render from the saved scene:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 \
        character.blend --python render.py [-- beauty front three-quarter side back] [--preview]

The beauty shot uses the cafe set exactly as saved.  The four turntable views use the same
character objects/materials/textures; only the Cafe_Set collection is excluded and a plain
grey world plus one extra back-fill light are enabled so profile and back read clearly.
"""
import bpy, sys, os, time

VIEWS = ["beauty", "front", "three-quarter", "side", "back"]


def render_views(views=None, preview=False, root=None):
    views = views or VIEWS
    root = root or os.path.dirname(bpy.data.filepath)
    out_dir = os.path.join(root, "previews" if preview else "renders")
    os.makedirs(out_dir, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.render.threads_mode = 'FIXED'
    sc.render.threads = 3
    sc.cycles.samples = 12 if preview else 32
    sc.cycles.use_denoising = True
    sc.render.use_border = False
    cafe = bpy.data.collections["Cafe_Set"]
    diag_light = bpy.data.objects["Diag_Fill_Back"]
    bg = sc.world.node_tree.nodes["Background"]
    saved_bg = (tuple(bg.inputs[0].default_value), bg.inputs[1].default_value)
    times = {}
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
        sc.render.filepath = os.path.join(out_dir, ("pv_" if preview else "") + v + ".png")
        bpy.ops.render.render(write_still=True)
        times[v] = round(time.time() - t0, 1)
        print("[render] %s -> %s (%.1fs)" % (v, sc.render.filepath, times[v]), flush=True)
    # leave the scene as saved: beauty camera, cafe visible
    cafe.hide_render = False
    diag_light.hide_render = True
    bg.inputs[0].default_value, bg.inputs[1].default_value = saved_bg
    sc.camera = bpy.data.objects["Cam_Beauty"]
    sc.render.resolution_x, sc.render.resolution_y = 960, 1040
    sc.render.resolution_percentage = 100
    sc.cycles.samples = 32
    return times


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    render_views([a for a in argv if not a.startswith("--")] or None, "--preview" in argv)
