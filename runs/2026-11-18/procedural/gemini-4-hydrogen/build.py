"""Reproducible Blender 5.1.1 build script for the upper-body portrait scene.

Usage (from this working directory):
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
Optional fast preview:
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --preview
"""

import os
import sys
import json
import math
import numpy as np
import bpy
from mathutils import Vector

# Ensure local modules resolve cleanly regardless of Blender's python cwd handling
WORKDIR = os.path.dirname(os.path.abspath(__file__))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

import textures
import character_mesh
import rig_and_pose
import sweater_builder
import hair_builder
import cafe_scene


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
    else:
        extra = []
    return {
        "preview": "--preview" in extra,
    }


def _configure_render_settings(scene: bpy.types.Scene, samples: int = 32):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.sample_clamp_indirect = 8.0

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.20


def _measure_character_metrics(char_collections=("Character", "Sweater", "Hair")) -> dict:
    """Measure exact character bounding box, mesh vertex/face counts, and hair curve counts."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    min_pt = np.array([1e9, 1e9, 1e9], dtype=np.float64)
    max_pt = np.array([-1e9, -1e9, -1e9], dtype=np.float64)

    base_verts = 0
    base_faces = 0
    eval_verts = 0
    eval_faces = 0
    hair_curves = 0
    hair_points = 0
    important_objects = []

    for col_name in char_collections:
        col = bpy.data.collections.get(col_name)
        if not col:
            continue
        for obj in col.objects:
            important_objects.append(obj.name)
            if obj.type == "MESH":
                base_verts += len(obj.data.vertices)
                base_faces += len(obj.data.polygons)
                eval_obj = obj.evaluated_get(depsgraph)
                emesh = eval_obj.to_mesh()
                if len(emesh.vertices) > 0:
                    eval_verts += len(emesh.vertices)
                    eval_faces += len(emesh.polygons)
                    mat = np.array(eval_obj.matrix_world, dtype=np.float64)
                    co = np.array([v.co[:] for v in emesh.vertices], dtype=np.float64)
                    co_h = np.hstack([co, np.ones((len(co), 1), dtype=np.float64)])
                    w_co = (co_h @ mat.T)[:, :3]
                    min_pt = np.minimum(min_pt, w_co.min(axis=0))
                    max_pt = np.maximum(max_pt, w_co.max(axis=0))
                eval_obj.to_mesh_clear()
            elif obj.type == "CURVES":
                hc = obj.data
                hair_curves += len(hc.curves)
                hair_points += len(hc.points)
                if len(hc.points) > 0:
                    pos = np.empty(len(hc.points) * 3, dtype=np.float32)
                    hc.points.foreach_get("position", pos)
                    pos = pos.reshape(-1, 3).astype(np.float64)
                    min_pt = np.minimum(min_pt, pos.min(axis=0))
                    max_pt = np.maximum(max_pt, pos.max(axis=0))

    return {
        "important_objects": important_objects,
        "bounds_meters": {
            "min": [round(float(x), 4) for x in min_pt],
            "max": [round(float(x), 4) for x in max_pt],
            "size": [round(float(x), 4) for x in (max_pt - min_pt)],
        },
        "mesh_counts": {
            "base_vertices": int(base_verts),
            "base_faces": int(base_faces),
            "evaluated_vertices": int(eval_verts),
            "evaluated_faces": int(eval_faces),
        },
        "hair_curve_counts": {
            "total_curves": int(hair_curves),
            "total_control_points": int(hair_points),
        },
    }


def _switch_to_diagnostic_studio_lighting():
    """Hide cafe scenery and set uniform neutral studio backdrop & balanced 4-way lighting without modifying character."""
    cafe_col = bpy.data.collections.get("CafeScenery")
    if cafe_col is not None:
        cafe_col.hide_render = True
        cafe_col.hide_viewport = True

    scene = bpy.context.scene
    # Plain neutral studio background
    if scene.world and scene.world.node_tree:
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.40, 0.39, 0.38, 1.0)
            bg.inputs[1].default_value = 0.55

    # Soften strong directional window key slightly and add balanced back/side fill for diagnostic readability
    lights_col = bpy.data.collections.get("Lights")
    if lights_col:
        for obj in lights_col.objects:
            if obj.type == "LIGHT":
                if obj.name == "Key_CafeWindow":
                    obj.data.energy = 145.0
                elif obj.name == "Fill_CafeInterior":
                    obj.data.energy = 95.0
                elif obj.name == "Rim_BackWindow":
                    obj.data.energy = 110.0
        # Add clean back-right fill for diagnostic back & side profile views
        ld = bpy.data.lights.new("Diag_BackFill", type="AREA")
        ld.shape = "RECTANGLE"
        ld.size = 1.8
        ld.size_y = 1.6
        ld.energy = 115.0
        ld.color = (1.0, 0.96, 0.93)
        lo = bpy.data.objects.new("Diag_BackFill", ld)
        cafe_scene._look_at(lo, Vector((0.85, 1.35, 1.65)), Vector((0.0, -0.05, 1.35)))
        lights_col.objects.link(lo)


def build_and_render():
    os.chdir(WORKDIR)
    args = _parse_args()
    is_preview = args["preview"]

    bpy.ops.wm.read_factory_settings(use_empty=True)

    with open("input/environment.json", "r", encoding="utf-8") as f:
        env = json.load(f)

    scene = bpy.context.scene
    root_col = scene.collection

    col_char = bpy.data.collections.new("Character")
    col_sw = bpy.data.collections.new("Sweater")
    col_hair = bpy.data.collections.new("Hair")
    col_cafe = bpy.data.collections.new("CafeScenery")
    col_lights = bpy.data.collections.new("Lights")
    col_cams = bpy.data.collections.new("Cameras")
    for c in (col_char, col_sw, col_hair, col_cafe, col_lights, col_cams):
        root_col.children.link(c)

    # 1. Build sculpted base body, UVs, and multi-layer eyes
    body_obj, lm_uv, eye_centers = character_mesh.build_sculpted_body(env, col_char)

    skin_size = 1024 if is_preview else 2048
    iris_size = 512 if is_preview else 1024
    knit_size = 1024

    skin_imgs = textures.generate_skin_textures(lm_uv, size=skin_size)
    iris_img = textures.generate_iris_texture(size=iris_size)
    knit_imgs = textures.generate_knit_textures(size=knit_size)
    wood_img = textures.generate_oak_wood_texture(size=512)

    mat_face, mat_hand = character_mesh.create_skin_materials(skin_imgs)
    body_obj.data.materials.append(mat_face)
    body_obj.data.materials.append(mat_hand)

    eye_mats = character_mesh.create_eye_materials(iris_img)
    eye_groups = character_mesh.build_eye_pair(eye_centers, eye_mats, col_char)

    # 2. Rig and pose into selfie portrait composition
    arm_obj = rig_and_pose.create_armature(env["landmarks"], eye_centers, col_char)
    rig_and_pose.assign_analytical_weights(body_obj, arm_obj)
    rig_and_pose.parent_eyes_to_rig(eye_groups, arm_obj)

    beauty_cam_pos = Vector((-0.080, -0.64, 1.500))
    rig_and_pose.pose_for_selfie(arm_obj, beauty_cam_pos)

    # Compute exact posed head eye-midpoint for camera DOF focus!
    m_head = hair_builder._get_head_pose_matrix(arm_obj)
    eye_mid_rest = 0.5 * (Vector(eye_centers["l"]) + Vector(eye_centers["r"]))
    face_focus_pos = m_head @ Vector((eye_mid_rest.x, eye_mid_rest.y - 0.014, eye_mid_rest.z))

    # 3. Build tailored dusty-pink knitted sweater & subtle wool fuzz
    sw_obj = sweater_builder.build_sweater(body_obj, arm_obj, env["landmarks"], knit_imgs, col_sw)
    sweater_builder.build_sweater_fuzz(sw_obj, col_sw, count=3600)

    # 4. Build multi-layer chestnut wavy hair, sheer bangs, flyaways, 3D brows & lashes
    hair_builder.build_hair_and_facial_details(body_obj, sw_obj, arm_obj, eye_centers, col_hair)

    # 5. Build sunlit cafe background & lighting & cameras
    cafe_scene.build_cafe_scenery(wood_img, col_cafe)
    bcam_obj, diag_cams = cafe_scene.setup_lighting_and_cameras(beauty_cam_pos, face_focus_pos, col_lights, col_cams)

    # 6. Configure Cycles CPU (32 samples + denoising)
    samples = 16 if is_preview else 32
    _configure_render_settings(scene, samples=samples)
    scene.render.resolution_x = 480 if is_preview else 960
    scene.render.resolution_y = 520 if is_preview else 1040

    # 7. Pack all resources and SAVE character.blend BEFORE rendering!
    bpy.ops.file.pack_all()
    if not is_preview:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(WORKDIR, "character.blend"))
        # Reopen character.blend to verify packed portability before rendering!
        bpy.ops.wm.open_mainfile(filepath=os.path.join(WORKDIR, "character.blend"))
        scene = bpy.context.scene
        _configure_render_settings(scene, samples=samples)

    os.makedirs("renders", exist_ok=True)
    if is_preview:
        os.makedirs("scratch_previews", exist_ok=True)

    # 8. Measure exact character metrics from the reopened scene
    metrics = _measure_character_metrics()

    # 9. Render Beauty Portrait (960x1040)
    scene.camera = bpy.data.objects["Beauty_Camera"]
    scene.render.resolution_x = 480 if is_preview else 960
    scene.render.resolution_y = 520 if is_preview else 1040
    beauty_out = "scratch_previews/preview_beauty.png" if is_preview else "renders/beauty.png"
    scene.render.filepath = os.path.join(WORKDIR, beauty_out)
    bpy.ops.render.render(write_still=True)

    # 10. Switch to diagnostic plain background (without altering character or saving over character.blend!)
    _switch_to_diagnostic_studio_lighting()
    diag_res = 512 if is_preview else 768
    scene.render.resolution_x = diag_res
    scene.render.resolution_y = diag_res

    diag_map = [
        ("Cam_Front", "scratch_previews/preview_front.png" if is_preview else "renders/front.png"),
        ("Cam_ThreeQuarter", "scratch_previews/preview_34.png" if is_preview else "renders/three-quarter.png"),
        ("Cam_Side", "scratch_previews/preview_side.png" if is_preview else "renders/side.png"),
        ("Cam_Back", "scratch_previews/preview_back.png" if is_preview else "renders/back.png"),
    ]
    for cam_name, out_rel in diag_map:
        scene.camera = bpy.data.objects[cam_name]
        scene.render.filepath = os.path.join(WORKDIR, out_rel)
        bpy.ops.render.render(write_still=True)

    if not is_preview:
        report = {
            "blender_version": bpy.app.version_string,
            "render_engine": scene.render.engine,
            "render_device": scene.cycles.device,
            "samples": scene.cycles.samples,
            "denoising": bool(scene.cycles.use_denoising),
            "view_transform": scene.view_settings.view_transform,
            "important_objects": metrics["important_objects"] + [
                "Character_Rig",
                "Beauty_Camera",
                "Cam_Front",
                "Cam_ThreeQuarter",
                "Cam_Side",
                "Cam_Back",
            ],
            "character_bounds": metrics["bounds_meters"],
            "mesh_counts": metrics["mesh_counts"],
            "hair_curve_counts": metrics["hair_curve_counts"],
            "generated_textures": [
                {
                    "name": "skin_albedo.png",
                    "resolution": [skin_size, skin_size],
                    "colorspace": "sRGB",
                    "description": "Procedural warm fair porcelain-peach face/neck albedo with landmark-registered cheek blush, coral-rose lip gradient, subtle lash-line warmth, and fine mottled melanin variation.",
                },
                {
                    "name": "skin_roughness.png",
                    "resolution": [skin_size, skin_size],
                    "colorspace": "Non-Color",
                    "description": "Facial roughness map with dewy T-zone/cheekbone highlights, moist lower-lip sheen, and micro-pore roughness variation.",
                },
                {
                    "name": "skin_bump.png",
                    "resolution": [skin_size, skin_size],
                    "colorspace": "Non-Color",
                    "description": "Restrained multi-scale skin pore heightmap and vertical lip micro-creases.",
                },
                {
                    "name": "iris_eye.png",
                    "resolution": [iris_size, iris_size],
                    "colorspace": "sRGB",
                    "description": "Warm chestnut-brown iris with radial collagen striations, amber peripupillary collarette, dark limbal ring, and soft off-white sclera.",
                },
                {
                    "name": "knit_albedo.png",
                    "resolution": [knit_size, knit_size],
                    "colorspace": "sRGB",
                    "description": "Tileable dusty rose-pink melange stockinette V-stitch yarn color map.",
                },
                {
                    "name": "knit_height.png",
                    "resolution": [knit_size, knit_size],
                    "colorspace": "Non-Color",
                    "description": "Tileable stockinette V-stitch heightmap for yarn relief bump shading.",
                },
                {
                    "name": "knit_roughness.png",
                    "resolution": [knit_size, knit_size],
                    "colorspace": "Non-Color",
                    "description": "Tileable matte-wool knit yarn roughness map with fiber noise.",
                },
                {
                    "name": "knit_rib_albedo.png",
                    "resolution": [knit_size, knit_size],
                    "colorspace": "sRGB",
                    "description": "Tileable dusty rose-pink 2x2 ribbed knit albedo for crew-neck collar, cuffs, and waist hem.",
                },
                {
                    "name": "knit_rib_height.png",
                    "resolution": [knit_size, knit_size],
                    "colorspace": "Non-Color",
                    "description": "Tileable 2x2 ribbed knit heightmap.",
                },
                {
                    "name": "cafe_oak_wood.png",
                    "resolution": [512, 512],
                    "colorspace": "sRGB",
                    "description": "Procedural oak wood grain texture for cafe chair, foreground table, and counter.",
                },
            ],
            "rendered_files": [
                "renders/beauty.png",
                "renders/front.png",
                "renders/three-quarter.png",
                "renders/side.png",
                "renders/back.png",
            ],
            "limitations": [
                "Face likeness and soft closed-mouth smile are sculpted procedurally from the CC0 MakeHuman hm08 upper-body base topology rather than photogrammetry or manual digital sculpting.",
                "Unseen anatomy (lower torso, lower back hair length, back of sweater, and relaxed hands extending past the ribbed cuffs) is inferred consistently for turnaround inspection.",
                "Hair waves and wispy bangs are constructed from procedural guide curves with BVH surface sliding rather than baked physics cloth/hair simulation.",
                "Cafe background furniture, window, plant, and flower vase are low-detail background props designed to sit softly out of focus behind the portrait.",
            ],
        }
        with open(os.path.join(WORKDIR, "scene-report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    build_and_render()
