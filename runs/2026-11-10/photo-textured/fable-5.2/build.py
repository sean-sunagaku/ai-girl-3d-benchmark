"""Claude Fable 5.2 - cafe-selfie-v1 (second pass, photo textures allowed): builds the whole portrait scene from the
supplied inputs (CC0 base mesh, landmark file and the reference photograph as texture source).

Run from this directory:
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py

Development switches may follow a "--": --quick, --skip a,b, --no-render, --views beauty,front
"""
import os
import sys
import time
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import bpy
import numpy as np

from fable import base, sculpt, rig, eyes, skin, util


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='lower densities for fast previews')
    ap.add_argument('--skip', default='', help='comma list: sweater,hair,cafe')
    ap.add_argument('--no-render', action='store_true')
    ap.add_argument('--views', default='beauty,front,three-quarter,side,back')
    ap.add_argument('--out', default='character.blend')
    ap.add_argument('--debug-maps', action='store_true', help='write the generated skin maps to previews/ (development)')
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return ap.parse_args(argv)


def main():
    args = parse()
    skip = set(filter(None, args.skip.split(',')))
    q = 0.35 if args.quick else 1.0
    t0 = time.time()
    log = lambda *a: print('[build %6.1fs]' % (time.time() - t0), *a, flush=True)
    rng = np.random.default_rng(20261110)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0        # no .blend1 backups in the delivery folder
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    col_char = util.collection('Character')
    col_body = util.collection('Body', col_char)
    col_face = util.collection('Eyes_Brows_Lashes', col_char)

    # ---- 1. supplied base, sculpt, pose --------------------------------------------------------------
    body = base.Body(os.path.join(HERE, 'input'))
    sculpt_params = sculpt.sculpt_face(body)
    tmp = body.make_object('tmp_weights')
    sk = rig.Skeleton(body.landmarks)
    W = sk.bone_heat_weights(tmp)
    bpy.data.objects.remove(tmp, do_unlink=True)
    rot, pose = rig.build_pose(sk)
    rig.arm_ik(sk, rot, 'r', pose['r_wrist'], pose['r_elbow'], hand_fwd=pose['r_hand_fwd'], hand_palm=pose['r_hand_palm'])
    rig.arm_ik(sk, rot, 'l', pose['l_wrist'], pose['l_elbow'], hand_fwd=pose['l_hand_fwd'], hand_palm=pose['l_hand_palm'])
    for s in ('l', 'r'):
        rig.finger_curl(sk, rot, s, pose['fingers'])
    M = sk.solve(rot)
    V_posed = sk.skin(body.V, W, M)
    Mh = M['head']
    log('base sculpted and posed: %d verts, %d faces' % (len(body.V), len(body.F)))

    ob_body = body.make_object('Body', V_posed, col_body)
    sub = ob_body.modifiers.new('Subdivision', 'SUBSURF')
    sub.levels = 1; sub.render_levels = 2; sub.uv_smooth = 'NONE'
    rest_attr = ob_body.data.attributes.new('rest_position_base', 'FLOAT_VECTOR', 'POINT')
    rest_attr.data.foreach_set('vector', body.V0.astype(np.float32).ravel())

    # ---- 2. skin: generated maps + the registered reference photograph projected onto the real head ----------
    from fable import scene as scene_mod, photo as photo_mod
    from fable.sculpt import EYE_C, EYE_R
    cam_spec = scene_mod.beauty_camera_spec(sk, M)
    ref = photo_mod.load_photo(os.path.join(HERE, 'input', 'reference.png'))
    cam = photo_mod.Camera(cam_spec)
    eyes_posed = [sk.apply(Mh, EYE_C), sk.apply(Mh, EYE_C * np.array([-1.0, 1.0, 1.0]))]
    model_px, photo_px = photo_mod.landmark_pairs(body, V_posed, eyes_posed, EYE_R, cam)
    resid = np.linalg.norm(model_px - photo_px, axis=1)
    warp = photo_mod.tps(model_px, photo_px)
    sampled = photo_mod.sample_colours(ref)
    pv, pf, _ = util.subdivided(V_posed, body.F, 1)
    maps = skin.paint_skin(body, 1024 if args.quick else 2048,
                           photo=dict(img=ref, cam=cam, warp=warp, Mh=Mh, bvh=util.bvh_from(pv, pf)))
    mat_skin = skin.skin_material(maps)
    ob_body.data.materials.append(mat_skin)
    if args.debug_maps:
        os.makedirs(os.path.join(HERE, 'previews'), exist_ok=True)
        for name in ('Skin_Color_Generated',):
            im = bpy.data.images[name]; im.filepath_raw = os.path.join(HERE, 'previews', 'dbg_' + name + '.png'); im.file_format = 'PNG'; im.save()
            im.filepath_raw = ''; im.pack()
        w = maps.get('photo_weight')
        if w is not None:
            wi = bpy.data.images.new('dbg_w', w.shape[1], w.shape[0]); wi.pixels.foreach_set(np.repeat(w[..., None], 4, 2).astype(np.float32).ravel() * 0 + np.dstack([w, w, w, np.ones_like(w)]).astype(np.float32).ravel())
            wi.filepath_raw = os.path.join(HERE, 'previews', 'dbg_photo_weight.png'); wi.file_format = 'PNG'; wi.save(); bpy.data.images.remove(wi)
    photo_facts = dict(landmarks=int(len(resid)), landmark_residual_px_before_warp=dict(mean=round(float(resid.mean()), 2), max=round(float(resid.max()), 2)),
                       sampled_srgb={k: [round(float(c), 3) for c in v] for k, v in sampled.items()}, **maps['info'])
    log('skin maps painted; photo projection: %s' % json.dumps(photo_facts))

    # ---- 3. eyes, lashes, brows -------------------------------------------------------------------------
    eye_mats = eyes.eye_materials(sampled['iris'])
    eyes.build_eyes(col_face, Mh, cam_spec['position'], eye_mats, q)
    surf_rest, surf_F, _ = util.subdivided(body.V, body.F, 2)
    surf_bvh = util.bvh_from(surf_rest, surf_F)
    mat_lash, _ = eyes.hair_material('Hair_Lashes', (0.018, 0.012, 0.010), roughness=0.35, radial=0.5)
    mat_brow, _ = eyes.hair_material('Hair_Brows', (0.080, 0.056, 0.044), roughness=0.6, radial=0.7)
    eyes.build_lashes(body, surf_rest, Mh, col_face, mat_lash, rng, q)
    eyes.build_brows(surf_bvh, Mh, col_face, mat_brow, rng, q)
    log('eyes, lashes and brows built')

    ctx = dict(body=body, sk=sk, M=M, W=W, V_posed=V_posed, ob_body=ob_body, col_char=col_char, rng=rng, q=q,
               cam_spec=cam_spec, pose=pose, log=log, args=args, sampled=sampled)

    # ---- 4. sweater, hair ---------------------------------------------------------------------------------
    if 'sweater' not in skip:
        from fable import sweater
        sweater.build(ctx)
        log('sweater built')
    if 'hair' not in skip:
        from fable import hair
        hair.build(ctx)
        log('hair built')

    # ---- 5. set, lights, cameras ---------------------------------------------------------------------------
    scene_mod.build(ctx, cafe=('cafe' not in skip))
    log('scene, lights and cameras ready')

    # ---- 6. save before rendering, then re-open the saved file and work only from that ------------------------
    bpy.ops.file.pack_all()
    out_path = os.path.join(HERE, args.out)
    scrub_ui_paths()
    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
    scrub_ui_paths()                                           # the save operator re-fills the file browser state
    bpy.ops.wm.save_mainfile(compress=True)
    log('saved', args.out)
    facts = dict(photo=photo_facts, hair=ctx.get('hair_info'), pose=dict((k, v) for k, v in pose.items() if isinstance(v, (int, float))),
                 camera=dict(distance_to_eyes_m=round(cam_spec['distance'], 4), vertical_fov_deg=cam_spec['fov_v']))
    bpy.ops.wm.open_mainfile(filepath=out_path)
    expect = ['Body', 'Eye_L', 'Eye_R', 'Eyelashes', 'Eyebrows', 'Cam_Beauty'] + [n for n, k in (('Sweater', 'sweater'), ('Hair', 'hair')) if k not in skip]
    missing = [n for n in expect if n not in bpy.data.objects]
    unpacked = [i.name for i in bpy.data.images if i.type != 'RENDER_RESULT' and i.size[0] > 0 and not i.packed_file]
    assert not missing and not unpacked, ('re-open check failed', missing, unpacked)
    log('re-opened %s: %d objects, %d packed images' % (args.out, len(bpy.data.objects), len([i for i in bpy.data.images if i.packed_file])))

    from fable import render, report
    views = [v for v in args.views.split(',') if v]
    if not args.no_render:
        render.render_views(None, views)
        log('renders done')
    report.write(os.path.join(HERE, 'scene-report.json'), dict(
        stage='photo-textured (second pass; the supplied photograph is a texture source, geometry stays fully 3D)',
        textures=TEXTURES,
        build=dict(script='build.py', quick=bool(args.quick), skipped=sorted(skip), seconds=round(time.time() - t0, 1),
                   reopened_saved_file_before_rendering=True),
        rendered_files=['renders/%s.png' % v for v in views] if not args.no_render else [],
        details=facts, notes=NOTES))
    log('scene-report.json written')


def scrub_ui_paths():
    """Blender keeps the last file-browser directory (the workstation home) in the UI data: keep it out of the file."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == 'FILE_BROWSER' and space.params is not None:
                    space.params.directory = b'_' * 700          # overwrite the whole fixed-size buffer first
                    space.params.directory = b'//'
                    space.params.filename = '_' * 200
                    space.params.filename = ''
    bpy.context.scene.render.filepath = '//renders/'


TEXTURES = [
    dict(image='Skin_Color_Generated', size=2048, uv='MakeHuman hm08 UV layout of the Body mesh (UVMap)',
         source='generated complexion everywhere + supplied reference photograph on the central face',
         projection='perspective projection through the fitted selfie camera (Cam_Beauty placement) onto the posed, '
                    'sculpted head, registered with a 14-landmark thin-plate-spline warp and baked per texel into UV space',
         photo_coverage='cheeks, nose, lips, chin below the eye line; weight = visibility x facing x clean-skin mask x region',
         excluded='forehead and brows (covered by the fringe in the photo), lid margins and eye openings, ears, scalp, '
                  'neck, everything turned away from or hidden from the camera, all hair/background pixels',
         levels='photo values rescaled to the generated albedo level (not used at photographed exposure)'),
    dict(image='Skin_Data_Generated', size=2048, uv='UVMap', source='generated (roughness / coat / lip-line height)', projection=None),
    dict(image='Iris_Generated_PhotoTone', size=512, uv='planar iris UV', projection=None,
         source='authored radial-fibre iris; palette mixed 50/50 with the median iris colour sampled from the photograph'),
    dict(image='Knit_Stockinette_Height_AO_Twist / Knit_Rib_Height_AO_Twist', size=512, uv='stitch-unit UVs per knitted panel',
         source='generated tiles; yarn colours are scalars sampled from the photographed sweater (lit and shaded medians)', projection=None),
    dict(image=None, material='Hair_Chestnut', source='Principled Hair melanin / redness derived from the median photographed hair colour', projection=None),
]

NOTES = {
    'reference_use': 'Second pass: the supplied photograph is a texture source. It is projected onto the real 3D head and baked '
                     'into the skin UV map on the central face, and scalar colours (iris, hair, yarn) are sampled from it. '
                     'It is never used as a plane, background or composite; the image datablock is removed after sampling.',
    'previous_pass': 'Same geometry pipeline as my procedural-only first pass (kept unchanged in source-character.blend, which '
                     'build.py does not read). Repairs: hair tucked behind the left ear now falls behind the shoulder so the '
                     'neck and collar are visible.',
    'inferred': 'Side and back of the head, the ears, forehead under the fringe, the hair behind the shoulders, the back and lower '
                'part of the sweater, the hands and everything outside the photo frame are inferred / generated.',
    'limitations': [
        'Still visibly CG. The head is an analytic re-shaping of the CC0 MakeHuman base, so the photographed features sit on proportions that only approximate the person.',
        'Photo texture covers only the camera-facing central face below the eye line; forehead, lids, ears, neck and all side/back surfaces are generated and tone-matched.',
        'The photo is not de-lit beyond a level/hue match: its soft shading and lip highlights are baked into the albedo and add to the scene lighting.',
        'Registration uses 14 hand-measured landmarks (a few pixels of error each); fine features can sit up to about a millimetre off the geometry.',
        'Hair is a combed/settled procedural groom, the knit is a bump/colour tile, hands are loosely posed, the cafe is a minimal out-of-focus set.',
        'Hair_Export_Proxy is a ribbon mesh that is invisible to every Cycles ray; it exists only for mesh-only exchange formats.',
        'No production rig: the pose is baked into the meshes.',
    ],
}


if __name__ == '__main__':
    main()
