"""
build.py - Reproducible Blender 5.1.1 entry script for the 3D upper-body portrait.

Usage (full build, save, reopen verification, and final renders):
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py

Usage (fast visual preview pass to previews/):
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --preview
"""
import os
import sys
import json
import time
import math

# Ensure local modules in working directory are importable under --factory-startup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import bpy
from mathutils import Vector

import sculpt
import textures
import materials
import sweater
import hair
import cafe_env


def parse_args():
    argv = sys.argv
    if "--" in argv:
        extra = argv[argv.index("--") + 1:]
    else:
        extra = []
    return {
        "preview": "--preview" in extra,
        "skip_render": "--skip-render" in extra,
        "only_beauty": "--only-beauty" in extra,
    }


def make_collection(name, parent=None):
    col = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(col)
    else:
        parent.children.link(col)
    return col


def configure_cycles_render(scene, samples=32):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    if hasattr(scene.cycles, "denoising_input_passes"):
        scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'

    # Clean light bounce budget (enough for SSS + glossy eyes + hair transmission without excess CPU cost)
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.volume_bounces = 0
    scene.cycles.sample_clamp_indirect = 8.0

    # Smooth cylindrical 3D hair curves in Cycles
    scene.cycles_curves.shape = 'THICK'
    scene.cycles_curves.subdivisions = 2

    # Natural portrait color management (AgX)
    scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'AgX - Base Contrast'
    except Exception:
        scene.view_settings.look = 'None'
    scene.view_settings.exposure = -0.18
    scene.view_settings.gamma = 1.0

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'


def measure_character_metrics(char_collections):
    """Measure character world bounding box and base/evaluated mesh & curve vertex counts."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    min_co = [1e9, 1e9, 1e9]
    max_co = [-1e9, -1e9, -1e9]
    base_verts = 0
    base_polys = 0
    eval_verts = 0
    eval_polys = 0
    curve_strands = 0
    curve_points = 0
    important_objects = []

    seen = set()
    for col in char_collections:
        for obj in col.objects:
            if obj.name in seen:
                continue
            seen.add(obj.name)
            important_objects.append(obj.name)
            if obj.type == 'MESH':
                base_verts += len(obj.data.vertices)
                base_polys += len(obj.data.polygons)
                eo = obj.evaluated_get(depsgraph)
                me = eo.to_mesh()
                eval_verts += len(me.vertices)
                eval_polys += len(me.polygons)
                mat = eo.matrix_world
                for v in me.vertices:
                    w = mat @ v.co
                    for k in range(3):
                        if w[k] < min_co[k]:
                            min_co[k] = float(w[k])
                        if w[k] > max_co[k]:
                            max_co[k] = float(w[k])
                eo.to_mesh_clear()
            elif obj.type == 'CURVES':
                hd = obj.data
                curve_strands += len(hd.curves)
                pos_attr = hd.attributes.get("position")
                if pos_attr:
                    curve_points += len(pos_attr.data)
                    # Sample bounding box across points
                    step = max(1, len(pos_attr.data) // 4000)
                    mat = obj.matrix_world
                    for i in range(0, len(pos_attr.data), step):
                        w = mat @ pos_attr.data[i].vector
                        for k in range(3):
                            if w[k] < min_co[k]:
                                min_co[k] = float(w[k])
                            if w[k] > max_co[k]:
                                max_co[k] = float(w[k])

    return {
        "important_objects": important_objects,
        "bounds_meters": {
            "min": [round(x, 4) for x in min_co],
            "max": [round(x, 4) for x in max_co],
            "size": [round(max_co[k] - min_co[k], 4) for k in range(3)],
        },
        "mesh_counts": {
            "base_vertices": base_verts,
            "base_polygons": base_polys,
            "evaluated_vertices": eval_verts,
            "evaluated_polygons": eval_polys,
            "hair_curve_strands": curve_strands,
            "hair_curve_control_points": curve_points,
        },
    }


def build_scene():
    t_start = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Clean, well-organized scene collections
    col_char = make_collection("Character")
    col_body = make_collection("Character_Body_And_Eyes", parent=col_char)
    col_clothing = make_collection("Character_Clothing", parent=col_char)
    col_hair = make_collection("Character_Hair", parent=col_char)
    col_env = make_collection("Cafe_Environment")
    col_lights = make_collection("Lighting")
    col_cams = make_collection("Cameras")

    # 1. Build base procedural textures that don't depend on body mesh yet (eyes + knit)
    print("[build] Generating eye iris and wool knit textures...")
    iris_img = textures.generate_eye_iris_texture(size=1024)
    knit_imgs = textures.generate_knit_textures(size=1024)

    # 2. Build materials (skin material will be attached right after body sculpt + UV bake)
    eye_mats = materials.create_eye_materials(iris_img)
    hair_mats = materials.create_hair_materials()
    sweater_mats = materials.create_sweater_materials(knit_imgs)
    cafe_mats = materials.create_cafe_materials()

    all_mats = {}
    all_mats.update(eye_mats)
    all_mats.update(hair_mats)
    all_mats.update(sweater_mats)

    # 3. Sculpt character body + eyes (aiming eyes right at eye-level camera gaze)
    eye_gaze_target = (0.010, -0.66, 1.545)
    print("[build] Importing MakeHuman upper-body.obj, sculpting likeness & 3D eyes...")
    body_info = sculpt.build_character_body(col_body, all_mats, beauty_cam_pos=eye_gaze_target)

    # 4. Bake high-res 3D-to-UV procedural facial skin maps (1536x1536) & assign Mat_Skin
    print("[build] Generating 3D-to-UV facial skin maps (albedo, roughness, bump, lip mask)...")
    skin_imgs = textures.generate_skin_textures(
        body_info["body_obj"],
        body_info["canonical_verts"],
        body_info["eye_c_l"],
        body_info["eye_c_r"],
        size=1536,
    )
    skin_mat = materials.create_skin_material(skin_imgs)
    body_info["body_obj"].data.materials.clear()
    body_info["body_obj"].data.materials.append(skin_mat)
    all_mats["skin"] = skin_mat

    # 5. Build chunky dusty-pink knitted sweater with forward reaching sleeves & wool fuzz
    print("[build] Building 3D dusty-pink knitted crew-neck sweater...")
    sweater_info = sweater.build_sweater(col_clothing, all_mats, knit_imgs)

    # 6. Build multi-layer 3D hair, wispy bangs, front cascade, brows & eyelashes with BVH collision
    print("[build] Generating 3D hair strands, wispy bangs, brows, and eyelashes...")
    hair_info = hair.build_all_hair(body_info, sweater_info, all_mats, col_hair)

    # 7. Build sunlit cafe background props, lighting rig, and all 5 cameras
    print("[build] Building cafe environment, lighting rig, and cameras...")
    cafe_env.build_cafe_environment(cafe_mats, col_env)
    cafe_env.build_lighting_rig(col_lights)

    eye_mid_can = 0.5 * (body_info["eye_c_l"] + body_info["eye_c_r"])
    eye_mid_world = body_info["head_mat"] @ Vector(eye_mid_can)
    cams = cafe_env.build_cameras(col_cams, eye_mid_world)

    # Default active camera & resolution set to beauty portrait ready to open & inspect
    scene.camera = cams["beauty"]
    scene.render.resolution_x = 960
    scene.render.resolution_y = 1040
    scene.render.resolution_percentage = 100
    configure_cycles_render(scene, samples=32)

    # Pack all generated images into .blend for complete portability
    bpy.ops.file.pack_all()

    metrics = measure_character_metrics([col_body, col_clothing, col_hair])
    print(f"[build] Scene built in {time.time() - t_start:.2f}s. Metrics: {metrics['mesh_counts']}")
    return metrics


def render_all_views(blend_path, out_dir="renders", preview=False, only_beauty=False):
    """
    Reopen the saved character.blend cleanly to verify portability, then render:
      - beauty.png (960x1040, or 480x520 in preview mode) with Cafe_Environment visible
      - front.png, three-quarter.png, side.png, back.png (768x768, or 384x384 in preview mode)
        with Cafe_Environment hidden and a clean neutral studio backdrop, keeping exact same character geometry/materials.
    """
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene
    configure_cycles_render(scene, samples=12 if preview else 32)

    env_col = bpy.data.collections.get("Cafe_Environment")
    world_bg = scene.world.node_tree.nodes.get("World_BG") if scene.world and scene.world.node_tree else None

    rendered_files = []

    # 1. Beauty Portrait (with cafe environment)
    if env_col:
        env_col.hide_render = False
        env_col.hide_viewport = False
    if world_bg:
        world_bg.inputs["Color"].default_value = (0.85, 0.81, 0.76, 1.0)
        world_bg.inputs["Strength"].default_value = 0.20

    cam_beauty = bpy.data.objects.get("Camera_Beauty")
    scene.camera = cam_beauty
    scene.render.resolution_x = 480 if preview else 960
    scene.render.resolution_y = 520 if preview else 1040
    beauty_out = os.path.join(out_dir, "beauty.png")
    scene.render.filepath = beauty_out
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[render] Rendered {beauty_out} in {time.time() - t0:.2f}s")
    rendered_files.append(beauty_out)

    if only_beauty:
        return rendered_files

    # 2. Diagnostic Turnaround Views (plain neutral studio background, identical character)
    if env_col:
        env_col.hide_render = True
        env_col.hide_viewport = True
    if world_bg:
        world_bg.inputs["Color"].default_value = (0.28, 0.29, 0.31, 1.0)
        world_bg.inputs["Strength"].default_value = 0.45

    diag_views = [
        ("front", "Camera_Front"),
        ("three-quarter", "Camera_ThreeQuarter"),
        ("side", "Camera_Side"),
        ("back", "Camera_Back"),
    ]
    diag_res = 384 if preview else 768
    scene.render.resolution_x = diag_res
    scene.render.resolution_y = diag_res

    for fname, cam_name in diag_views:
        cam = bpy.data.objects.get(cam_name)
        scene.camera = cam
        out_path = os.path.join(out_dir, f"{fname}.png")
        scene.render.filepath = out_path
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[render] Rendered {out_path} in {time.time() - t0:.2f}s")
        rendered_files.append(out_path)

    return rendered_files


def write_scene_report(metrics, rendered_files, report_path="scene-report.json"):
    scene = bpy.context.scene
    tex_descs = []
    for img in bpy.data.images:
        if img.size[0] == 0:
            continue
        tex_descs.append({
            "name": img.name,
            "resolution": [int(img.size[0]), int(img.size[1])],
            "colorspace": img.colorspace_settings.name,
            "packed": img.packed_file is not None,
        })

    report = {
        "blender_version": bpy.app.version_string,
        "render_engine": scene.render.engine,
        "render_device": scene.cycles.device,
        "samples": int(scene.cycles.samples),
        "denoiser": scene.cycles.denoiser if scene.cycles.use_denoising else None,
        "important_objects": metrics["important_objects"],
        "character_bounds_meters": metrics["bounds_meters"],
        "mesh_counts": metrics["mesh_counts"],
        "generated_textures": tex_descs,
        "rendered_files": rendered_files,
        "remaining_limitations": [
            "Procedural sculpting approximates the reference likeness from a single 2D selfie photograph; unseen side/back skull, ear, neck, hair, and sweater shapes are inferred.",
            "Lower arms end in closed chunky ribbed cuffs without modeled hands or unseen lower torso/legs.",
            "Hair strands are generated via procedural guide curves and BVH collision rather than cloth/strand physics simulation, so close inspection of dense curl intersections shows simplified wave regularity.",
            "Skin pore and micro-color detail are procedurally synthesized in 3D-to-UV space rather than photogrammetry scans.",
            "Cafe background props are intentionally simplified 3D bokeh silhouettes so character quality remains the priority."
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] Wrote {report_path}")


def main():
    args = parse_args()
    blend_path = os.path.abspath("character.blend")

    metrics = build_scene()

    # Save character.blend BEFORE rendering so geometry always survives any render issue
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[build] Saved {blend_path}")

    if args["skip_render"]:
        return

    out_dir = "previews" if args["preview"] else "renders"
    rendered_files = render_all_views(
        blend_path,
        out_dir=out_dir,
        preview=args["preview"],
        only_beauty=args["only_beauty"],
    )

    if not args["preview"]:
        write_scene_report(metrics, rendered_files, report_path="scene-report.json")


if __name__ == "__main__":
    main()
