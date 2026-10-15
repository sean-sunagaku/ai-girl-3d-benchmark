"""Reopen character.blend in Blender, verify contents, and write scene-report.json."""
import bpy, json, os, numpy as np
ROOT = os.path.dirname(bpy.data.filepath)
br = json.load(open(os.path.join(ROOT, "previews", "build-report.json")))
sc = bpy.context.scene
dg = bpy.context.evaluated_depsgraph_get()
char = bpy.data.collections["Character"]
mesh_v = 0; mesh_f = 0; curves = 0; cpts = 0
for ob in char.all_objects:
    ev = ob.evaluated_get(dg)
    if ob.type == 'MESH':
        me = ev.to_mesh(); mesh_v += len(me.vertices); mesh_f += len(me.polygons); ev.to_mesh_clear()
    elif ob.type == 'CURVES':
        curves += len(ob.data.curves); cpts += len(ob.data.points)
scene_v = 0
for ob in sc.objects:
    if ob.type == 'MESH':
        me = ob.evaluated_get(dg).to_mesh(); scene_v += len(me.vertices); ob.evaluated_get(dg).to_mesh_clear()
missing = [i.name for i in bpy.data.images if i.source == 'FILE' and not i.packed_file]
renders = {}
for n in ("beauty", "front", "three-quarter", "side", "back"):
    p = os.path.join(ROOT, "renders", n + ".png")
    if os.path.exists(p):
        im = bpy.data.images.load(p); renders["renders/%s.png" % n] = list(im.size); bpy.data.images.remove(im)
rep = {
    "blender_version": bpy.app.version_string,
    "reopened_from": "character.blend",
    "reopen_check": {"objects": len(bpy.data.objects), "unpacked_file_images": missing,
                     "scene_camera": sc.camera.name if sc.camera else None},
    "render_engine": sc.render.engine,
    "render_device": "CPU (" + sc.cycles.device + ")",
    "samples": sc.cycles.samples, "denoising": sc.cycles.use_denoising,
    "denoiser": sc.cycles.denoiser, "threads": sc.render.threads,
    "color_management": br["color_management"],
    "important_objects": br["important_objects"],
    "collections": br["collections"],
    "materials": br["materials"],
    "character_bounds_m": br["character_bounds_m"],
    "base_mesh": {"source": "input/upper-body.obj (MakeHuman hm08, CC0)", **br["base_mesh"]},
    "evaluated_counts": {"character_mesh_vertices": mesh_v, "character_mesh_faces": mesh_f,
                         "whole_scene_mesh_vertices_incl_cafe": scene_v,
                         "hair_curves_total": curves, "hair_control_points_total": cpts},
    "per_object_counts": br["objects"],
    "hair": br.get("hair"), "eyelashes": br.get("lashes"), "brow_hairs": br.get("brow_hairs"),
    "generated_textures": br["textures"] + [
        {"name": "(procedural shader nodes, no image)", "description":
         "skin: per-vertex colour/roughness attributes painted by build.py + noise mottling, voronoi pores, fine bump; "
         "iris: radial/angular noise fibres + collarette/limbal ring; sclera gradient; hair: Principled Hair melanin model; "
         "wood: wave texture; all editable in the node editor"}],
    "packed_images": br["packed_images"],
    "rendered_files": renders,
    "render_times_s": None,
    "limitations": [
        "Likeness is approximate: face sculpted by hand-tuned procedural deformations of the MakeHuman base, not reconstructed from the photo.",
        "Back/side of head, hair behind the shoulders, torso and sweater back are inferred (not visible in the reference).",
        "Hands/fingers are simple tube + ellipsoid geometry with nails; not anatomically detailed and not rigged; hidden in the beauty frame as in the photo.",
        "Hair is a procedural guide/child Curves groom; waves are sinusoidal and some strands clump near the collar on her raised-arm side (visible as a darker band in side/back views).",
        "Bangs are wispy but more uniform/straight than the photo; part/crown shows a slightly lighter sheen.",
        "Sweater is a displaced shell derived from the posed body (no cloth simulation); folds are procedural, knit is a bump-mapped generated tile; front of the torso is smoothed and slightly flat.",
        "Skin shading is smoother and slightly pinker/doll-like than the photo; eyes are a little small and less luminous than the reference.",
        "Diagnostic views exclude the cafe set and use a plain grey world plus one back-fill light; character geometry/materials are identical in all views.",
        "No production rig; the temporary pose armature is baked and removed.",
    ],
}
with open(os.path.join(ROOT, "scene-report.json"), "w") as f:
    json.dump(rep, f, indent=2)
print("REPORT OK", mesh_v, curves, cpts, missing, renders)
