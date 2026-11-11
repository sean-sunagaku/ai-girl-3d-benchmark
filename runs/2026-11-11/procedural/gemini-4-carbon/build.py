import os
import sys
import json
import time
import numpy as np
import bpy
from mathutils import Vector

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

from lib_textures import build_all_textures
from lib_materials import (
    build_skin_material,
    build_eye_materials,
    build_hair_materials,
    build_sweater_fuzz_material,
    rgba_srgb,
)
from lib_anatomy import build_character_anatomy, pose_point
from lib_sweater import build_character_sweater
from lib_hair import build_character_hair
from lib_scene import build_cafe_environment, build_lighting_and_world, build_cameras


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def configure_cycles_cpu(scene, samples=32):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 8
    scene.cycles_curves.shape = 'THICK'

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = 0.15


def measure_scene_stats(char_col, all_cols):
    deps = bpy.context.evaluated_depsgraph_get()
    base_verts = 0
    base_faces = 0
    eval_verts = 0
    eval_faces = 0
    hair_curves_count = 0
    hair_points_count = 0

    bmin = np.array([1e9, 1e9, 1e9], dtype=np.float64)
    bmax = np.array([-1e9, -1e9, -1e9], dtype=np.float64)

    for col in all_cols:
        for obj in col.objects:
            ev = obj.evaluated_get(deps)
            if obj.type == 'MESH':
                base_verts += len(obj.data.vertices)
                base_faces += len(obj.data.polygons)
                em = ev.to_mesh()
                eval_verts += len(em.vertices)
                eval_faces += len(em.polygons)
                if col == char_col and len(em.vertices) > 0:
                    co = np.empty((len(em.vertices), 3), dtype=np.float64)
                    em.vertices.foreach_get("co", co.ravel())
                    # Transform by object matrix_world
                    mw = np.array(ev.matrix_world, dtype=np.float64)
                    co_w = (co @ mw[:3, :3].T) + mw[:3, 3]
                    bmin = np.minimum(bmin, co_w.min(axis=0))
                    bmax = np.maximum(bmax, co_w.max(axis=0))
                ev.to_mesh_clear()
            elif obj.type == 'CURVES':
                cdata = ev.data
                nc = len(cdata.curves)
                np_pts = len(cdata.points)
                hair_curves_count += nc
                hair_points_count += np_pts
                if col == char_col and np_pts > 0:
                    co = np.empty((np_pts, 3), dtype=np.float32)
                    cdata.position_data.foreach_get("vector", co.ravel())
                    bmin = np.minimum(bmin, co.min(axis=0))
                    bmax = np.maximum(bmax, co.max(axis=0))

    return {
        "characterBoundsMeters": {
            "min": [round(float(x), 4) for x in bmin],
            "max": [round(float(x), 4) for x in bmax],
            "dimensions": [round(float(bmax[i] - bmin[i]), 4) for i in range(3)],
        },
        "counts": {
            "baseMeshVertices": int(base_verts),
            "baseMeshFaces": int(base_faces),
            "evaluatedMeshVertices": int(eval_verts),
            "evaluatedMeshFaces": int(eval_faces),
            "hairAndFiberCurves": int(hair_curves_count),
            "hairAndFiberControlPoints": int(hair_points_count),
            "totalEvaluatedVerticesAndPoints": int(eval_verts + hair_points_count),
        },
    }


def main():
    t_start = time.time()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    is_preview = "--preview" in argv
    skip_render = "--no-render" in argv

    clear_scene()
    scene = bpy.context.scene
    root_col = scene.collection

    char_col = bpy.data.collections.new("Character")
    cafe_col = bpy.data.collections.new("Cafe_Environment")
    light_col = bpy.data.collections.new("Lighting")
    cam_col = bpy.data.collections.new("Cameras")
    for c in (char_col, cafe_col, light_col, cam_col):
        root_col.children.link(c)

    print("[1/6] Generating procedural <=2048px textures...")
    textures = build_all_textures(WORK_DIR)

    print("[2/6] Building physically-based skin, eye, hair & knitwear materials...")
    mat_skin = build_skin_material(textures)
    mat_eye_out, mat_eye_iris, mat_tear = build_eye_materials(textures)
    mat_hair, mat_core, mat_brow, mat_lash = build_hair_materials()
    mat_fuzz = build_sweater_fuzz_material()

    # Beauty camera lens target for eye convergence
    beauty_cam_loc = (-0.012, -0.605, 1.532)

    print("[3/6] Sculpting head, face, eyelids, smile, eyeballs, brows & lashes...")
    anat_info = build_character_anatomy(
        WORK_DIR, char_col, mat_skin, mat_eye_out, mat_eye_iris, mat_tear, mat_brow, mat_lash,
        gaze_target=beauty_cam_loc,
    )

    print("[4/6] Constructing dusty-pink fisherman-rib knitted sweater & forward arms...")
    sweater_info = build_character_sweater(char_col, textures, mat_skin, mat_fuzz)

    print("[5/6] Grooming collision-resolved chestnut wavy hair curves & wispy bangs...")
    hair_objs = build_character_hair(char_col, anat_info, sweater_info, mat_hair, mat_core)

    print("[6/6] Building 3D cafe interior, natural window lighting & cameras...")
    build_cafe_environment(cafe_col, textures)
    lights, world_bg = build_lighting_and_world(light_col)

    # Focus optical f/2.2 DOF directly on the front cornea of her closer eye!
    focus_eye = pose_point(anat_info["eye_L_rest"] + np.array([0.0, -0.014, 0.0]), neck_weight=1.0)
    beauty_cam, diag_cams = build_cameras(cam_col, focus_point=focus_eye)

    # Set Beauty_Camera active and configure Cycles CPU
    scene.camera = beauty_cam
    samples = 14 if is_preview else 32
    configure_cycles_cpu(scene, samples=samples)
    scene.render.resolution_x = 480 if is_preview else 960
    scene.render.resolution_y = 520 if is_preview else 1040
    scene.render.resolution_percentage = 100

    # Measure geometry counts & bounds before saving
    stats = measure_scene_stats(char_col, [char_col, cafe_col])
    print("Scene Stats:", json.dumps(stats, indent=2))

    # Pack all generated textures and save character.blend BEFORE rendering!
    bpy.ops.file.pack_all()
    blend_path = os.path.join(WORK_DIR, "character.blend")
    if not is_preview:
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
        print(f"Saved packed editable scene to {blend_path}")

    if skip_render:
        return

    out_folder = os.path.join(WORK_DIR, "previews" if is_preview else "renders")
    os.makedirs(out_folder, exist_ok=True)

    # 1. Render Beauty Portrait (with 3D Cafe Environment visible!)
    cafe_col.hide_render = False
    world_bg.inputs['Color'].default_value = rgba_srgb(0.84, 0.82, 0.80)
    world_bg.inputs['Strength'].default_value = 0.55
    scene.camera = beauty_cam
    scene.render.resolution_x = 480 if is_preview else 960
    scene.render.resolution_y = 520 if is_preview else 1040
    beauty_out = os.path.join(out_folder, "beauty.png")
    scene.render.filepath = beauty_out
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {beauty_out} in {time.time() - t0:.1f}s")

    # 2. Render 4 Diagnostic Turnaround Views (0°, 45°, 90°, 180° on plain studio background;
    #    Character geometry and materials remain 100% identical!)
    cafe_col.hide_render = True
    world_bg.inputs['Color'].default_value = rgba_srgb(0.76, 0.76, 0.78)
    world_bg.inputs['Strength'].default_value = 0.65
    diag_res = 384 if is_preview else 768
    scene.render.resolution_x = diag_res
    scene.render.resolution_y = diag_res

    rendered_files = ["renders/beauty.png"]
    for fname, dcam in diag_cams.items():
        scene.camera = dcam
        out_p = os.path.join(out_folder, fname)
        scene.render.filepath = out_p
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {out_p} in {time.time() - t0:.1f}s")
        rendered_files.append(f"renders/{fname}")

    # Restore Beauty state in memory as well
    cafe_col.hide_render = False
    world_bg.inputs['Color'].default_value = rgba_srgb(0.84, 0.82, 0.80)
    world_bg.inputs['Strength'].default_value = 0.55
    scene.camera = beauty_cam
    scene.render.resolution_x = 960
    scene.render.resolution_y = 1040

    if not is_preview:
        report = {
            "blenderVersion": bpy.app.version_string,
            "renderEngine": scene.render.engine,
            "renderDevice": scene.cycles.device,
            "samples": scene.cycles.samples,
            "denoiser": scene.cycles.denoiser,
            "threads": 3,
            "importantObjects": {
                "characterCollection": [ob.name for ob in char_col.objects],
                "cafeCollection": [ob.name for ob in cafe_col.objects[:12]] + [f"... ({len(cafe_col.objects)} total objects)"],
                "lightingCollection": [ob.name for ob in light_col.objects],
                "camerasCollection": [ob.name for ob in cam_col.objects],
            },
            "characterBounds": stats["characterBoundsMeters"],
            "meshAndCurveCounts": stats["counts"],
            "generatedTextures": [
                {
                    "name": "iris_albedo.png",
                    "resolution": [1024, 1024],
                    "colorSpace": "sRGB",
                    "description": "Procedural warm chestnut-brown and golden-amber radial iris stroma fibers, pupil, collarette, crypts, and dark limbal ring.",
                },
                {
                    "name": "iris_bump.png",
                    "resolution": [1024, 1024],
                    "colorSpace": "Non-Color",
                    "description": "Procedural height map of radial iris stroma fibers and crypt relief.",
                },
                {
                    "name": "skin_micro_detail.png",
                    "resolution": [1024, 1024],
                    "colorSpace": "Non-Color",
                    "description": "Seamless tileable Voronoi polygonal skin-cell relief, follicular micro-pores, and roughness modulation.",
                },
                {
                    "name": "sweater_knit_height.png",
                    "resolution": [1024, 1024],
                    "colorSpace": "Non-Color",
                    "description": "Seamless tileable fisherman-rib V-stitch wales, horizontal purl tuck bars, and 2-ply yarn twist height map driving true 3D Displace modifiers.",
                },
                {
                    "name": "sweater_knit_normal.png",
                    "resolution": [1024, 1024],
                    "colorSpace": "Non-Color",
                    "description": "Tangent-space normal map derived from procedural fisherman-rib yarn loop geometry.",
                },
                {
                    "name": "sweater_knit_albedo.png",
                    "resolution": [1024, 1024],
                    "colorSpace": "sRGB",
                    "description": "Dusty-pink yarn albedo with stitch-crevice ambient occlusion and subtle wool fiber mottling.",
                },
                {
                    "name": "cafe_wood.png",
                    "resolution": [512, 512],
                    "colorSpace": "sRGB",
                    "description": "Procedural warm honey-oak wood grain for cafe tables, chairs, counter, and shelves.",
                },
            ],
            "renderedFiles": rendered_files,
            "buildDurationSeconds": round(time.time() - t_start, 2),
            "limitations": [
                "Unseen anatomy below the sweater, back of the head/hair, back of the sweater, and hands/cuffs outside the reference frame are inferred procedurally.",
                "Facial likeness is sculpted via smooth anatomical deformation fields on subdivided MakeHuman CC0 topology rather than 3D scan data, so subtle individual micro-asymmetries remain an approximation.",
                "Sweater drape and hair waves use static procedural splines and BVH collision projection rather than full dynamic cloth/hair physics simulation.",
                "Cafe background props are simplified 3D architectural and tabletop forms designed for realistic out-of-focus portrait depth of field.",
            ],
        }
        with open(os.path.join(WORK_DIR, "scene-report.json"), "w") as rf:
            json.dump(report, rf, indent=2)
        print("Wrote scene-report.json")


if __name__ == "__main__":
    main()
