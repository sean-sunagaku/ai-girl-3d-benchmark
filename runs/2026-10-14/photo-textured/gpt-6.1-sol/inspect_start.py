import bpy
from pathlib import Path
R=Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(R/'source-character.blend'))
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=12;s.cycles.use_denoising=True;s.render.threads_mode='FIXED';s.render.threads=3
(R/'previews').mkdir(exist_ok=True)
for name,cam,size in [('previous-beauty','Beauty | selfie portrait',(480,520)),('previous-side','Inspection | side',(480,480)),('previous-face','Preview | close facial inspection',(640,640))]:
 s.camera=bpy.data.objects[cam];s.render.resolution_x,s.render.resolution_y=size
 bpy.data.collections['06 Cafe'].hide_render=name!='previous-beauty'
 s.render.filepath=str(R/'previews'/f'{name}.png');bpy.ops.render.render(write_still=True)
