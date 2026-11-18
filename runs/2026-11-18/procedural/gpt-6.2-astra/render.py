"""Render the saved scene; command arguments after -- choose views and preview mode."""
import sys
from pathlib import Path
import bpy
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
preview='--preview' in args
views=[v for v in args if not v.startswith('--')] or ['beauty','front','three-quarter','side','back']
root=Path(bpy.data.filepath).parent
s=bpy.context.scene;s.cycles.device='CPU';s.cycles.denoising_use_gpu=False;s.render.threads_mode='FIXED';s.render.threads=3
for view in views:
    beauty=view=='beauty';s.camera=bpy.data.objects['Portrait • window seat' if beauty else 'diagnostic • '+view];bpy.data.collections['Café set'].hide_render=not beauty
    s.render.resolution_x=960 if beauty else 768;s.render.resolution_y=1040 if beauty else 768
    s.cycles.samples=6 if preview else 32;s.cycles.adaptive_min_samples=3 if preview else 6;s.render.resolution_percentage=40 if preview else 100
    output=root/('previews' if preview else 'renders')/(view+'.png');output.parent.mkdir(exist_ok=True);s.render.filepath=str(output)
    print('Rendering',view,s.cycles.samples,'CPU samples',flush=True);bpy.ops.render.render(write_still=True)
