"""Build the portable Blender portrait from the local, supplied inputs."""
import json
import sys
from pathlib import Path

import bpy

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

from portrait_common import collection
from portrait_face import build_face
from portrait_hair import build_hair
from portrait_set import BEAUTY_LOC, build_staging
from portrait_wardrobe import build_wardrobe


def main():
    env=json.loads((ROOT/'input/environment.json').read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene
    scene.name="Chestnut and rose • café portrait"
    scene.unit_settings.system='METRIC'
    scene.render.engine='CYCLES'
    scene.cycles.device='CPU'
    scene.cycles.samples=32
    scene.cycles.use_adaptive_sampling=True
    scene.cycles.adaptive_threshold=.036
    scene.cycles.use_denoising=True
    scene.cycles.max_bounces=8
    scene.cycles.diffuse_bounces=4
    scene.cycles.glossy_bounces=4
    scene.cycles.transmission_bounces=5
    scene.cycles.transparent_max_bounces=8
    scene.cycles.sample_clamp_indirect=5.0
    scene.render.threads_mode='FIXED'
    scene.render.threads=3
    scene.render.resolution_x=960
    scene.render.resolution_y=1040
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.image_settings.color_depth='8'
    scene.render.filepath='//renders/beauty.png'
    scene.view_settings.view_transform='AgX'
    scene.view_settings.look='AgX - Medium High Contrast'
    scene.view_settings.exposure=-.37
    scene.render.film_transparent=False
    character=collection('Character')
    skin=collection('Face and eyes',character)
    wardrobe=collection('Knitwear and hands',character)
    groom=collection('Chestnut groom',character)
    head,tree=build_face(skin,env,BEAUTY_LOC)
    print('Face and ocular geometry complete',flush=True)
    build_wardrobe(wardrobe,env,head)
    print('Tailored knitwear and hands complete',flush=True)
    build_hair(groom,tree)
    print('Curves groom complete',flush=True)
    build_staging(scene)
    scene['Reference use']='input/reference.png was viewed as a design reference. All visible surfaces are modeled.'
    scene['Base mesh']='MakeHuman hm08 body topology, CC0. See README.md and input/ASSET-LICENSE.md.'
    for ob in bpy.context.selected_objects:
        ob.select_set(False)
    bpy.context.view_layer.objects.active=None
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_perspective='CAMERA'
                area.spaces.active.shading.color_type='MATERIAL'
                area.spaces.active.overlay.show_overlays=False
    bpy.ops.file.pack_all()
    (ROOT/'renders').mkdir(exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'character.blend'),compress=True)


if __name__=='__main__':
    main()
