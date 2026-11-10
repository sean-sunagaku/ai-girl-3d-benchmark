"""Claude Fable 5.2 - cafe-selfie-v1 (procedural stage): builds the whole portrait scene from the supplied inputs.

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

    # ---- 2. skin ---------------------------------------------------------------------------------------
    maps = skin.paint_skin(body, 1024 if args.quick else 2048)
    mat_skin = skin.skin_material(maps)
    ob_body.data.materials.append(mat_skin)
    log('skin maps painted')

    # ---- 3. eyes, lashes, brows -------------------------------------------------------------------------
    from fable import scene as scene_mod
    cam_spec = scene_mod.beauty_camera_spec(sk, M)
    eye_mats = eyes.eye_materials()
    eyes.build_eyes(col_face, Mh, cam_spec['position'], eye_mats, q)
    surf_rest, surf_F, _ = util.subdivided(body.V, body.F, 2)
    surf_bvh = util.bvh_from(surf_rest, surf_F)
    mat_lash, _ = eyes.hair_material('Hair_Lashes', (0.018, 0.012, 0.010), roughness=0.35, radial=0.5)
    mat_brow, _ = eyes.hair_material('Hair_Brows', (0.080, 0.056, 0.044), roughness=0.6, radial=0.7)
    eyes.build_lashes(body, surf_rest, Mh, col_face, mat_lash, rng, q)
    eyes.build_brows(surf_bvh, Mh, col_face, mat_brow, rng, q)
    log('eyes, lashes and brows built')

    ctx = dict(body=body, sk=sk, M=M, W=W, V_posed=V_posed, ob_body=ob_body, col_char=col_char, rng=rng, q=q,
               cam_spec=cam_spec, pose=pose, log=log, args=args)

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
    facts = dict(hair=ctx.get('hair_info'), pose=dict((k, v) for k, v in pose.items() if isinstance(v, (int, float))),
                 camera=dict(distance_to_eyes_m=round(cam_spec['distance'], 4), vertical_fov_deg=cam_spec['fov_v']))
    bpy.ops.wm.open_mainfile(filepath=out_path)
    missing = [n for n in ('Body', 'Sweater', 'Hair', 'Eye_L', 'Eye_R', 'Eyelashes', 'Eyebrows', 'Cam_Beauty') if n not in bpy.data.objects]
    unpacked = [i.name for i in bpy.data.images if i.type != 'RENDER_RESULT' and i.size[0] > 0 and not i.packed_file]
    assert not missing and not unpacked, ('re-open check failed', missing, unpacked)
    log('re-opened %s: %d objects, %d packed images' % (args.out, len(bpy.data.objects), len([i for i in bpy.data.images if i.packed_file])))

    from fable import render, report
    views = [v for v in args.views.split(',') if v]
    if not args.no_render:
        render.render_views(None, views)
        log('renders done')
    report.write(os.path.join(HERE, 'scene-report.json'), dict(
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


NOTES = {
    'reference_use': 'The photograph was only looked at and measured (2D landmark positions, coarse region colour averages). '
                     'No photo pixels are used in any texture, material, background or render.',
    'inferred': 'Side and back of the head, the ears, the hair behind the shoulders, the back and lower part of the sweater, '
                'the hands and everything outside the photo frame are inferred.',
    'limitations': [
        'The face is a smooth analytic re-shaping of the CC0 MakeHuman base: proportions follow the reference, but it reads as a stylised CG face, not a likeness.',
        'Skin detail is painted procedurally at 2K; there are no real pores, fine wrinkles or asymmetries beyond a slightly lop-sided smile.',
        'The fringe and long hair are combed/settled analytically (no physical simulation); the strand count is far below real hair and ends look thin in places.',
        'The sweater is one structured raglan mesh posed with skin weights and settled against the body; stitches are a bump/colour tile, not yarn geometry, and folds are simplified.',
        'Hands come from the same CC0 base and are only loosely posed; they are outside the beauty frame.',
        'The cafe is a minimal set of boxes, cylinders and leaf cards meant to be seen out of focus.',
        'Hair_Export_Proxy is a ribbon mesh that is invisible to every Cycles ray; it exists only for mesh-only exchange formats.',
        'No production rig: the pose is baked into the meshes.',
    ],
}


if __name__ == '__main__':
    main()
