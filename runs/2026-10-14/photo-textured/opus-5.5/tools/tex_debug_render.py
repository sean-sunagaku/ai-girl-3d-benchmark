"""Development helper: show the photo skin layer as pure emission (alpha 0 -> red) from the
beauty camera, face crop, to check projection alignment against the photo.
    Blender -b character.blend --python tools/tex_debug_render.py -- out.png [view]
view = beauty (face crop, default) or front / three-quarter / side / back (full 768 frame).
"""
import bpy, sys, os
args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
view = args[1] if len(args) > 1 else "beauty"
m = bpy.data.materials["Skin"]; nt = m.node_tree
tex = nt.nodes["Skin_Photo_RGBA"]
o = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'
nt.links.new(tex.outputs["Alpha"], mix.inputs[0]); mix.inputs[6].default_value = (1, 0, 0, 1)
nt.links.new(tex.outputs["Color"], mix.inputs[7])
em = nt.nodes.new("ShaderNodeEmission"); nt.links.new(mix.outputs[2], em.inputs[0])
nt.links.new(em.outputs[0], o.inputs["Surface"])
for n in ("Hair_Strands", "Hair_Bangs", "Eyelashes", "Eyebrows"):
    if n in bpy.data.objects: bpy.data.objects[n].hide_render = True
sc = bpy.context.scene
sc.view_settings.view_transform = 'Standard'; sc.view_settings.look = 'None'; sc.view_settings.exposure = 0
sc.cycles.samples = 8
if view == "beauty":
    sc.camera = bpy.data.objects["Cam_Beauty"]; bpy.data.cameras["Cam_Beauty"].dof.use_dof = False
    sc.render.resolution_x, sc.render.resolution_y = 1205, 1306; sc.render.resolution_percentage = 100
    sc.render.use_border = True; sc.render.use_crop_to_border = True
    sc.render.border_min_x, sc.render.border_max_x = 380/1205, 880/1205
    sc.render.border_min_y, sc.render.border_max_y = 1-680/1306, 1-180/1306
else:
    sc.camera = bpy.data.objects["Cam_" + view]
    bpy.data.collections["Cafe_Set"].hide_render = True
    sc.render.resolution_x = sc.render.resolution_y = 768; sc.render.resolution_percentage = 100
sc.render.filepath = os.path.abspath(out)
bpy.ops.render.render(write_still=True)
