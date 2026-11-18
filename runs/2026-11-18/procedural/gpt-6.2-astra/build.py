"""Build and save the editable portrait from the common inputs."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from atelier import *
from face import make_skin, build_eyes, build_features
from knit import make_sweater
from hands import make_hands
from groom import make_hair
from room import make_room,make_rig,make_cameras

bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene;s.name='Window seat • portrait';s.unit_settings.system='METRIC'
s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=32;s.cycles.use_denoising=True;s.cycles.denoiser='OPENIMAGEDENOISE';s.cycles.denoising_use_gpu=False;s.cycles.preview_denoising_use_gpu=False
s.cycles.use_adaptive_sampling=True;s.cycles.adaptive_min_samples=6;s.cycles.adaptive_threshold=.045
s.cycles.max_bounces=7;s.cycles.diffuse_bounces=3;s.cycles.glossy_bounces=3;s.cycles.transmission_bounces=4;s.cycles.transparent_max_bounces=7
s.render.use_stamp_filename=False;s.render.threads_mode='FIXED';s.render.threads=3;s.render.resolution_x=960;s.render.resolution_y=1040;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.image_settings.color_depth='8';s.render.filepath='//renders/beauty.png';s.render.film_transparent=False;s.view_settings.view_transform='AgX'
character=col('Character');portrait=col('01 • Skin and eyes',character);groom=col('02 • Chestnut groom',character);knit=col('03 • Dusty rose knit',character);cafe=col('Café set');lights=col('Window lighting');cameras=col('Cameras')
print('Building face',flush=True);face=make_skin(portrait);build_eyes(portrait);build_features(portrait)
print('Knitting garment',flush=True);make_sweater(knit);make_hands(portrait)
print('Grooming hair',flush=True);make_hair(groom)
print('Dressing set',flush=True);make_room(cafe);make_rig(lights);s.camera=make_cameras(cameras)
s['project']='Window seat';s['appearance_source']='Supplied photograph viewed as reference, no photographic assets in scene.'
bpy.context.view_layer.objects.active=face;face.select_set(True)
for screen in bpy.data.screens:
    for area in screen.areas:
        for space in area.spaces:
            if space.type=='FILE_BROWSER' and space.params:
                # Scrub the factory file picker's remembered local directory, including its previous buffer.
                space.params.directory=b'.'*max(1,len(space.params.directory))
                space.params.directory=b'//'
        if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.overlay.show_overlays=False;area.spaces.active.shading.color_type='MATERIAL'
bpy.context.preferences.filepaths.save_version=0
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'character.blend'),compress=True)
print('Saved editable portrait',flush=True)
