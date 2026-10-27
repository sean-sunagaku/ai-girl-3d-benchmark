import bpy, os
root=os.path.dirname(os.path.abspath(__file__)); os.chdir(root)
s=bpy.context.scene
s.cycles.device='CPU'; s.cycles.samples=8; s.cycles.use_denoising=True
s.render.resolution_percentage=50; s.render.filepath='//previews/portrait.png'
bpy.ops.render.render(write_still=True)
