"""Write scene-report.json from the *reopened* character.blend (imported by build.py)."""
import bpy, json, os
import numpy as np

LIMITATIONS = [
    "Not a scan and not an exact likeness: a CC0 MakeHuman base sculpted by hand-tuned deformations and then "
    "warped to 19 hand-picked photo landmarks + 5 cheek-contour points + the visible ear, as seen from one camera. "
    "Depth (nose/cheek/jaw/skull profile) is inferred, so the profile and back are plausible guesses.",
    "The face texture comes from one bright, beauty-filtered selfie: it is smooth, partly de-lit (low-frequency "
    "division only), and some photo shading/makeup remains baked in. Areas the photo never saw (sides/back of head, "
    "ears, under the chin, scalp) use a procedural skin layer tinted to the photo's median complexion.",
    "Eyes: the right (far) eye still sits a few pixels off the photo; iris colour comes from the photo but the "
    "sclera/cornea are procedural; the eyes read slightly larger and more 'doll-like' than in the photo.",
    "Hair is a procedural guide/child Curves groom (not photo-textured): bangs, ear tuck and waves are approximations, "
    "the crown reads a little flat and dark, strand clumping is more regular than real hair, and a white-ish "
    "backlit tuft over the left ear plus a few stray strands show in the turntable views.",
    "The 26 mm selfie perspective makes the torso smaller relative to the head than in the reference (hem visible). "
    "The torso/collar sits about 60 px (in the 1205 px photo) away from the photographed collar; solving an extra "
    "neck rotation for it broke the hair at the nape, so it was not applied.",
    "Sweater is a displaced shell from the posed body (no cloth simulation); knit is a generated height tile with a "
    "photo-sampled base colour, not the photographed knit.",
    "Hands/fingers are simple tube + ellipsoid geometry; they are outside the selfie frame (as in the photo) but "
    "visible and crude in the turntable views. No production rig.",
    "Turntable views hide the cafe set and use a grey world + one back fill light; the character is identical.",
    "The result is still visibly CG at close inspection (skin micro-detail, eye wetness, hair).",
]


def mesh_stats(ob, dg):
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    n_v, n_f = len(me.vertices), len(me.polygons)
    co = np.empty(n_v * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    mw = np.array(ob.matrix_world)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    return n_v, n_f, co


def write_report(root, render_times=None, reopen_info=None):
    br = json.load(open(os.path.join(root, "previews", "build-report.json")))
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    char = bpy.data.collections["Character"]
    per_obj, mins, maxs = {}, np.full(3, 1e9), np.full(3, -1e9)
    mesh_v = mesh_f = curves = cpts = 0
    for ob in sorted(char.all_objects, key=lambda o: o.name):
        if ob.type == 'MESH':
            n_v, n_f, co = mesh_stats(ob, dg)
            per_obj[ob.name] = {"type": "MESH", "base_vertices": len(ob.data.vertices),
                                "evaluated_vertices": n_v, "evaluated_faces": n_f,
                                "uv_maps": [u.name for u in ob.data.uv_layers],
                                "materials": [m.name for m in ob.data.materials if m]}
            mesh_v += n_v
            mesh_f += n_f
        elif ob.type == 'CURVES':
            n = len(ob.data.points)
            co = np.empty(n * 3, np.float32)
            ob.data.position_data.foreach_get("vector", co)
            co = co.reshape(-1, 3)
            per_obj[ob.name] = {"type": "CURVES (native hair)", "curves": len(ob.data.curves), "control_points": n,
                                "materials": [m.name for m in ob.data.materials if m]}
            curves += len(ob.data.curves)
            cpts += n
        else:
            continue
        mins, maxs = np.minimum(mins, co.min(0)), np.maximum(maxs, co.max(0))
    scene_v = 0
    for ob in sc.objects:
        if ob.type == 'MESH':
            scene_v += mesh_stats(ob, dg)[0]
    renders = {}
    for n in ("beauty", "front", "three-quarter", "side", "back"):
        p = os.path.join(root, "renders", n + ".png")
        if os.path.exists(p):
            im = bpy.data.images.load(p)
            renders["renders/%s.png" % n] = {"size": list(im.size),
                                             "render_seconds": (render_times or {}).get(n)}
            bpy.data.images.remove(im)
    images = []
    for im in bpy.data.images:
        if im.type != 'IMAGE':
            continue
        images.append({"name": im.name, "size": list(im.size), "packed": bool(im.packed_file),
                       "colorspace": im.colorspace_settings.name})
    rep = {
        "blender_version": bpy.app.version_string,
        "files": {"scene": "character.blend", "build_script": "build.py",
                  "helpers": ["photo_fit.py", "photo_texture.py", "render.py", "scene_report.py"],
                  "renders": list(renders.keys()),
                  "previous_first_pass_scene_kept_unmodified": "source-character.blend (not read by build.py)"},
        "reopen_check": reopen_info,
        "render": {"engine": sc.render.engine, "device": sc.cycles.device, "threads": sc.render.threads,
                   "threads_mode": sc.render.threads_mode, "samples": sc.cycles.samples,
                   "denoising": sc.cycles.use_denoising, "denoiser": sc.cycles.denoiser,
                   "view_transform": sc.view_settings.view_transform, "look": sc.view_settings.look,
                   "exposure": round(sc.view_settings.exposure, 3),
                   "beauty_resolution": [960, 1040], "turntable_resolution": [768, 768]},
        "cameras": {o.name: {"lens_mm": round(o.data.lens, 2), "location": [round(x, 4) for x in o.location]}
                    for o in bpy.data.objects if o.type == 'CAMERA'},
        "collections": {c.name: sorted(o.name for o in c.objects) for c in bpy.data.collections},
        "character_bounds_m": {"min": mins.round(4).tolist(), "max": maxs.round(4).tolist(),
                               "size": (maxs - mins).round(4).tolist()},
        "geometry_counts": {
            "character_evaluated_mesh_vertices": mesh_v, "character_evaluated_mesh_faces": mesh_f,
            "whole_scene_evaluated_mesh_vertices_incl_cafe": scene_v,
            "native_hair_curves_total": curves, "native_hair_control_points_total": cpts,
            "per_object": per_obj},
        "base_mesh": {"source": "input/upper-body.obj (MakeHuman hm08 base mesh, CC0)", **br.get("base_mesh", {})},
        "photo_fit": br.get("photo_fit"),
        "hair": br.get("hair"), "eyelashes": br.get("lashes"), "brow_hairs": br.get("brow_hairs"),
        "textures": br.get("textures"),
        "procedural_layers": "Skin: per-vertex colour/roughness attributes (procedural layer, tinted to the photo "
                             "median) + noise mottling, Voronoi pores and fine bump, lash tight-line attribute; "
                             "Iris/sclera: procedural fibres/limbal ring/sclera gradient + AO contact shadow, photo "
                             "iris mixed on top; Hair: Principled Hair (Chiang) melanin model; Knit: generated height "
                             "tiles + photo-sampled colour ramp; Cafe: simple procedural materials.",
        "images_in_blend": images,
        "materials": sorted(m.name for m in bpy.data.materials),
        "notes": br.get("notes"),
        "remaining_limitations": LIMITATIONS,
    }
    with open(os.path.join(root, "scene-report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep
