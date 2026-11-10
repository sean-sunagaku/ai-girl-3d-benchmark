"""Render the five required views from the saved scene state."""
import os
import bpy
from .scene import BEAUTY_RES, DIAG_RES, set_mode


def render_views(ctx, views, scale=1.0, samples=None, out_dir='renders', prefix=''):
    sc = bpy.context.scene
    os.makedirs(out_dir, exist_ok=True)
    if samples:
        sc.cycles.samples = samples
    for v in views:
        if v == 'beauty':
            sc.camera = bpy.data.objects['Cam_Beauty']; res = BEAUTY_RES; set_mode('beauty')
        else:
            sc.camera = bpy.data.objects['Cam_diagnostic_' + v]; res = DIAG_RES; set_mode('diagnostic')
        sc.render.resolution_x = int(res[0] * scale); sc.render.resolution_y = int(res[1] * scale)
        sc.render.resolution_percentage = 100
        sc.render.filepath = os.path.join(os.getcwd(), out_dir, prefix + v + '.png')
        bpy.ops.render.render(write_still=True)
    set_mode('beauty')
    sc.camera = bpy.data.objects['Cam_Beauty']
    sc.render.resolution_x, sc.render.resolution_y = BEAUTY_RES
