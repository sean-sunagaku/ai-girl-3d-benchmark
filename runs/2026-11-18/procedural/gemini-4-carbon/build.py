"""
Reproducible Blender 5.1.1 upper-body portrait scene builder and renderer.

Run from working directory:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
"""

import json
import os
import sys
import bpy
from mathutils import Vector

# Ensure current working directory is on sys.path for character_lib imports
WORKDIR = os.path.dirname(os.path.abspath(__file__))
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

from character_lib.textures import generate_all_textures
from character_lib.anatomy import (
    load_upper_body_with_hands_and_uvs,
    sculpt_feminine_likeness,
    build_realistic_eyes,
    build_and_pose_armature,
)
from character_lib.clothing import build_knitted_sweater
from character_lib.hair import build_all_hair_brows_lashes
from character_lib.materials import populate_skin_color_attributes, build_all_materials
from character_lib.environment import BEAUTY_CAM_LOC, build_cafe_and_lighting_and_cameras


def clear_factory_scene():
    """Remove default factory startup objects, meshes, lights, cameras, and materials."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for datablock_col in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.armatures,
        bpy.data.hair_curves,
        bpy.data.textures,
        bpy.data.images,
    ):
        for block in list(datablock_col):
            datablock_col.remove(block)


def configure_render_settings(scene: bpy.types.Scene, samples: int = 32):
    """Configure Cycles CPU strictly within benchmark resource rules (--threads 3, CPU, <=32 samples)."""
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 3

    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"

    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.sample_clamp_indirect = 6.0

    # Smooth 3D cylindrical hair curves
    scene.cycles_curves.shape = "THICK"
    scene.cycles_curves.subdivisions = 2

    # Natural AgX photographic tone mapping
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.08
    scene.view_settings.gamma = 1.0

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"


def measure_character_metrics(scene: bpy.types.Scene) -> dict:
    """Measure character world bounds and base/evaluated geometry counts on reopened scene."""
    deps = bpy.context.evaluated_depsgraph_get()
    char_cols = ["Character", "Eyes", "Hair"]

    min_b = [1e9, 1e9, 1e9]
    max_b = [-1e9, -1e9, -1e9]

    base_mesh_verts = 0
    base_mesh_faces = 0
    eval_mesh_verts = 0
    eval_mesh_faces = 0
    hair_curve_strands = 0
    hair_curve_points = 0
    per_object = {}

    for cname in char_cols:
        col = bpy.data.collections.get(cname)
        if not col:
            continue
        for obj in col.objects:
            if obj.type == "MESH":
                bv = len(obj.data.vertices)
                bf = len(obj.data.polygons)
                ev_obj = obj.evaluated_get(deps)
                ev_mesh = ev_obj.to_mesh()
                ev_v = len(ev_mesh.vertices)
                ev_f = len(ev_mesh.polygons)

                mat_w = ev_obj.matrix_world
                for v in ev_mesh.vertices:
                    wp = mat_w @ v.co
                    for k in range(3):
                        if wp[k] < min_b[k]:
                            min_b[k] = wp[k]
                        if wp[k] > max_b[k]:
                            max_b[k] = wp[k]
                ev_obj.to_mesh_clear()

                base_mesh_verts += bv
                base_mesh_faces += bf
                eval_mesh_verts += ev_v
                eval_mesh_faces += ev_f
                per_object[obj.name] = {
                    "type": "MESH",
                    "baseVertices": bv,
                    "baseFaces": bf,
                    "evaluatedVertices": ev_v,
                    "evaluatedFaces": ev_f,
                }
            elif obj.type == "CURVES":
                nc = len(obj.data.curves)
                np_pts = len(obj.data.points)
                hair_curve_strands += nc
                hair_curve_points += np_pts
                mat_w = obj.matrix_world
                for pt in obj.data.points:
                    wp = mat_w @ pt.position
                    for k in range(3):
                        if wp[k] < min_b[k]:
                            min_b[k] = wp[k]
                        if wp[k] > max_b[k]:
                            max_b[k] = wp[k]
                per_object[obj.name] = {
                    "type": "CURVES",
                    "strands": nc,
                    "controlPoints": np_pts,
                }
            elif obj.type == "ARMATURE":
                per_object[obj.name] = {
                    "type": "ARMATURE",
                    "bones": len(obj.data.bones),
                }

    # Also measure Cafe_Environment evaluated mesh vertices so total scene stays well below 1M
    cafe_eval_verts = 0
    col_cafe = bpy.data.collections.get("Cafe_Environment")
    if col_cafe:
        for obj in col_cafe.objects:
            if obj.type == "MESH":
                ev_obj = obj.evaluated_get(deps)
                em = ev_obj.to_mesh()
                cafe_eval_verts += len(em.vertices)
                ev_obj.to_mesh_clear()

    return {
        "boundsMinMeters": [round(x, 5) for x in min_b],
        "boundsMaxMeters": [round(x, 5) for x in max_b],
        "dimensionsMeters": [round(max_b[k] - min_b[k], 5) for k in range(3)],
        "characterBaseMeshVertices": base_mesh_verts,
        "characterBaseMeshFaces": base_mesh_faces,
        "characterEvaluatedMeshVertices": eval_mesh_verts,
        "characterEvaluatedMeshFaces": eval_mesh_faces,
        "characterHairStrands": hair_curve_strands,
        "characterHairControlPoints": hair_curve_points,
        "cafeEnvironmentEvaluatedVertices": cafe_eval_verts,
        "totalSceneEvaluatedVerticesPlusCurvePoints": eval_mesh_verts + cafe_eval_verts + hair_curve_points,
        "objects": per_object,
    }


def main():
    preview_mode = os.environ.get("PREVIEW", "0") == "1"
    render_scale = float(os.environ.get("RENDER_SCALE", "0.5" if preview_mode else "1.0"))
    samples = int(os.environ.get("SAMPLES", "12" if preview_mode else "32"))
    only_views = os.environ.get("ONLY_VIEWS", "").strip()
    selected_views = [v.strip() for v in only_views.split(",") if v.strip()] if only_views else None

    print(f"[build.py] Starting portrait build (preview={preview_mode}, samples={samples}, scale={render_scale})...")
    clear_factory_scene()
    scene = bpy.context.scene
    configure_render_settings(scene, samples=samples)

    # Create organized top-level character collections
    col_char = bpy.data.collections.new("Character")
    col_eyes = bpy.data.collections.new("Eyes")
    col_hair = bpy.data.collections.new("Hair")
    for c in (col_char, col_eyes, col_hair):
        scene.collection.children.link(c)

    # 1. Generate & pack procedural <=2048px textures
    textures = generate_all_textures(os.path.join(WORKDIR, "textures"))

    # 2. Load CC0 MakeHuman upper-body + welded wrist/hand & UVs, and sculpt feminine likeness
    upper_path = os.path.join(WORKDIR, "input", "upper-body.obj")
    base_path = os.path.join(WORKDIR, "input", "base.obj")
    body_obj = load_upper_body_with_hands_and_uvs(upper_path, base_path, col_char)
    sculpt_feminine_likeness(body_obj)

    # 3. Build realistic two-layer eyeballs, caruncles, and tear-meniscus strips
    eye_objs = build_realistic_eyes(col_eyes)

    # 4. Build editable Character_Armature and apply cafe selfie pose
    arm_obj = build_and_pose_armature(body_obj, eye_objs, col_char, BEAUTY_CAM_LOC)

    # 5. Build genuine 3D relaxed knitted crew-neck sweater with rolled ribbed trim & stitch relief
    sw_obj = build_knitted_sweater(body_obj, arm_obj, textures["knit_height"][0], col_char)

    # 6. Populate seamless 3D anatomical skin color attributes and build all materials
    populate_skin_color_attributes(body_obj)
    mats = build_all_materials(textures)

    body_obj.data.materials.append(mats["skin"])
    sw_obj.data.materials.append(mats["sweater_knit"])
    sw_obj.data.materials.append(mats["sweater_rib"])
    sw_obj.data.materials.append(mats["sweater_inner"])

    for side_label in ("L", "R"):
        eye_objs[side_label]["inner"].data.materials.append(mats["eye_inner"])
        eye_objs[side_label]["cornea"].data.materials.append(mats["cornea"])
        eye_objs[side_label]["caruncle"].data.materials.append(mats["caruncle"])
        eye_objs[side_label]["tear"].data.materials.append(mats["cornea"])

    # 7. Build collision-aware multi-layered 3D hair curves, wispy bangs, eyebrows, and eyelashes
    hair_objs = build_all_hair_brows_lashes(body_obj, sw_obj, arm_obj, col_hair)
    for hkey in ("underlayer", "main_waves", "bangs", "tendrils", "flyaways"):
        hair_objs[hkey].data.materials.append(mats["hair"])
    hair_objs["eyebrows"].data.materials.append(mats["brows"])
    hair_objs["eyelashes"].data.materials.append(mats["lashes"])

    # 8. Build Cafe_Environment, Diagnostic_Backdrop, Lighting, and Cameras
    env = build_cafe_and_lighting_and_cameras(scene)

    # Set exact DOF focus distance on Cam_Beauty to posed midpoint between left & right corneas!
    bpy.context.view_layer.update()
    eye_l_w = arm_obj.matrix_world @ arm_obj.pose.bones["eye.L"].head
    eye_r_w = arm_obj.matrix_world @ arm_obj.pose.bones["eye.R"].head
    eye_mid_w = 0.5 * (eye_l_w + eye_r_w) + Vector((0.0, -0.014, 0.0))
    cam_b = env["cam_beauty"]
    cam_b.data.dof.focus_distance = (eye_mid_w - cam_b.location).length

    # Ensure beauty camera and Cafe_Environment are active before saving character.blend
    scene.camera = cam_b
    env["col_cafe"].hide_render = False
    env["col_cafe"].hide_viewport = False
    env["col_diag"].hide_render = True
    env["col_diag"].hide_viewport = True
    scene.render.resolution_x = int(round(960 * render_scale))
    scene.render.resolution_y = int(round(1040 * render_scale))
    scene.render.resolution_percentage = 100

    # Pack all generated images for 100% portable reopening and save character.blend BEFORE rendering!
    bpy.ops.file.pack_all()
    blend_path = os.path.join(WORKDIR, "character.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    print(f"[build.py] Saved packed editable scene to {blend_path}")

    # 9. Reopen character.blend from disk to verify portability and render all views from the saved file!
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene
    # Re-enforce 3 CPU threads after file load
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 3
    scene.cycles.samples = samples

    metrics = measure_character_metrics(scene)
    print(
        f"[build.py] Reopened scene verified: eval mesh verts={metrics['characterEvaluatedMeshVertices']}, "
        f"hair strands={metrics['characterHairStrands']} ({metrics['characterHairControlPoints']} pts)"
    )

    renders_dir = os.path.join(WORKDIR, "renders_preview" if preview_mode else "renders")
    os.makedirs(renders_dir, exist_ok=True)

    col_cafe = bpy.data.collections["Cafe_Environment"]
    col_diag = bpy.data.collections["Diagnostic_Backdrop"]
    world_bg = scene.world.node_tree.nodes["Background"]

    render_manifest = [
        ("beauty", "Cam_Beauty", 960, 1040, True, "beauty.png"),
        ("front", "Cam_Front", 768, 768, False, "front.png"),
        ("three-quarter", "Cam_ThreeQuarter", 768, 768, False, "three-quarter.png"),
        ("side", "Cam_Side", 768, 768, False, "side.png"),
        ("back", "Cam_Back", 768, 768, False, "back.png"),
    ]

    rendered_files = []
    for view_id, cam_name, rx, ry, use_cafe, filename in render_manifest:
        if selected_views is not None and view_id not in selected_views:
            continue
        scene.camera = bpy.data.objects[cam_name]
        scene.render.resolution_x = int(round(rx * render_scale))
        scene.render.resolution_y = int(round(ry * render_scale))
        scene.render.resolution_percentage = 100

        # Toggle ONLY background scenery vs plain diagnostic backdrop; character geometry & materials NEVER change!
        col_cafe.hide_render = not use_cafe
        col_diag.hide_render = use_cafe
        if use_cafe:
            world_bg.inputs["Color"].default_value = (0.50, 0.47, 0.43, 1.0)
            world_bg.inputs["Strength"].default_value = 0.28
        else:
            world_bg.inputs["Color"].default_value = (0.28, 0.27, 0.28, 1.0)
            world_bg.inputs["Strength"].default_value = 0.34

        out_file = os.path.join(renders_dir, filename)
        scene.render.filepath = out_file
        print(f"[build.py] Rendering {view_id} ({scene.render.resolution_x}x{scene.render.resolution_y}) -> {out_file}")
        bpy.ops.render.render(write_still=True)
        rendered_files.append(os.path.relpath(out_file, WORKDIR))

    if not preview_mode:
        report = {
            "blenderVersion": bpy.app.version_string,
            "renderEngine": scene.render.engine,
            "renderDevice": scene.cycles.device,
            "threads": scene.render.threads,
            "samples": scene.cycles.samples,
            "denoiser": scene.cycles.denoiser if scene.cycles.use_denoising else None,
            "colorManagement": {
                "viewTransform": scene.view_settings.view_transform,
                "look": scene.view_settings.look,
                "exposure": round(scene.view_settings.exposure, 4),
            },
            "importantObjects": {
                "characterBody": "Character_Body",
                "armature": "Character_Armature",
                "sweater": "Character_Sweater",
                "eyes": [
                    "Eyeball_Inner_L",
                    "Eyeball_Cornea_L",
                    "Eye_Caruncle_L",
                    "Eye_TearLine_L",
                    "Eyeball_Inner_R",
                    "Eyeball_Cornea_R",
                    "Eye_Caruncle_R",
                    "Eye_TearLine_R",
                ],
                "hairAndFacialHair": [
                    "Hair_Underlayer",
                    "Hair_Main_Waves",
                    "Hair_Wispy_Bangs",
                    "Hair_Framing_Tendrils",
                    "Hair_Flyaways",
                    "Character_Eyebrows",
                    "Character_Eyelashes",
                ],
                "cameras": ["Cam_Beauty", "Cam_Front", "Cam_ThreeQuarter", "Cam_Side", "Cam_Back"],
            },
            "characterBounds": {
                "minMeters": metrics["boundsMinMeters"],
                "maxMeters": metrics["boundsMaxMeters"],
                "dimensionsMeters": metrics["dimensionsMeters"],
            },
            "geometryCounts": {
                "characterBaseMeshVertices": metrics["characterBaseMeshVertices"],
                "characterBaseMeshFaces": metrics["characterBaseMeshFaces"],
                "characterEvaluatedMeshVertices": metrics["characterEvaluatedMeshVertices"],
                "characterEvaluatedMeshFaces": metrics["characterEvaluatedMeshFaces"],
                "characterHairStrands": metrics["characterHairStrands"],
                "characterHairControlPoints": metrics["characterHairControlPoints"],
                "cafeEnvironmentEvaluatedVertices": metrics["cafeEnvironmentEvaluatedVertices"],
                "totalEvaluatedVerticesAndCurvePoints": metrics["totalSceneEvaluatedVerticesPlusCurvePoints"],
                "perObject": metrics["objects"],
            },
            "generatedTextures": [
                {
                    "name": "iris_albedo_1k",
                    "file": "textures/iris_albedo_1k.png",
                    "resolution": [1024, 1024],
                    "packed": True,
                    "description": "Procedural warm chestnut-amber radial iris stroma, crypts, collarette, dark limbal ring, pupil, and warm ivory sclera transition.",
                },
                {
                    "name": "iris_bump_1k",
                    "file": "textures/iris_bump_1k.png",
                    "resolution": [1024, 1024],
                    "packed": True,
                    "description": "Procedural Non-Color radial iris fiber and collarette relief height map.",
                },
                {
                    "name": "skin_micro_2k",
                    "file": "textures/skin_micro_2k.png",
                    "resolution": [2048, 2048],
                    "packed": True,
                    "description": "Procedural 2048x2048 tileable skin cellular micro-pore height, Langer sulcus micro-roughness, and subsurface variation channels.",
                },
                {
                    "name": "knit_albedo_2k",
                    "file": "textures/knit_albedo_2k.png",
                    "resolution": [2048, 2048],
                    "packed": True,
                    "description": "Procedural 2048x2048 dusty-pink interlocking V-stitch knitwear albedo with heathered rose yarn plies and stitch-valley ambient occlusion.",
                },
                {
                    "name": "knit_height_2k",
                    "file": "textures/knit_height_2k.png",
                    "resolution": [2048, 2048],
                    "packed": True,
                    "description": "Procedural 2048x2048 Non-Color stockinette V-loop and 2-ply yarn twist height map used for both 3D Displace modifier and shader bump.",
                },
                {
                    "name": "knit_roughness_1k",
                    "file": "textures/knit_roughness_1k.png",
                    "resolution": [1024, 1024],
                    "packed": True,
                    "description": "Procedural 1024x1024 yarn micro-roughness map.",
                },
                {
                    "name": "rib_knit_height_1k",
                    "file": "textures/rib_knit_height_1k.png",
                    "resolution": [1024, 1024],
                    "packed": True,
                    "description": "Procedural 1024x1024 chunky 1x1 vertical rib-knit height map for crew-neck collar, sleeve cuffs, and waistband hem.",
                },
            ],
            "renderedFiles": rendered_files,
            "remainingLimitations": [
                "Lower body below the ribbed sweater waistband (Z < ~1.00 m) is omitted because this is an upper-body portrait asset; the hem interior is cleanly capped.",
                "Side profile, posterior skull/hair styling, sweater back drape, and hand curl near the camera edges are anatomically inferred from a single frontal selfie photograph.",
                "Facial sculpting is performed procedurally over subdivided MakeHuman hm08 quad topology rather than high-resolution photogrammetric scan data, so micro-asymmetries and individual pore placement are stylized approximations.",
                "Cycles CPU rendering is capped at 32 samples with OpenImageDenoise per shared benchmark constraints, which slightly softens ultra-fine hair flyaways and micro-pore grain compared to high-sample offline renders.",
                "Cafe background props (chairs, tables, ficus plant, counter, shelves, flower vase) are simplified 3D architectural/prop approximations designed to reproduce the reference bokeh and warm bounce lighting.",
            ],
        }
        report_path = os.path.join(WORKDIR, "scene-report.json")
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)
        print(f"[build.py] Wrote {report_path}")


if __name__ == "__main__":
    main()
