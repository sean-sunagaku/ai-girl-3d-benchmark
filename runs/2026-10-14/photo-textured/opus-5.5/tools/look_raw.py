import bpy, math
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis='Y', up_axis='Z')
ob = bpy.context.selected_objects[0]
sc = bpy.context.scene
sc.render.engine='BLENDER_WORKBENCH'
sc.display.shading.light='STUDIO'
sc.display.shading.show_cavity=True
sc.render.resolution_x=600; sc.render.resolution_y=600
cam = bpy.data.objects.new("c", bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera=cam
cam.data.lens=85
for name,loc,tgt,ortho in [("face",(0,-0.9,1.54),(0,0,1.54),0.2),("faceside",(0.9,-0.08,1.54),(0,-0.08,1.54),0.2),("eye",(0.03,-0.5,1.546),(0.03,0,1.546),0.05),("body",(0,-3,1.38),(0,0,1.38),0.9)]:
    cam.location=loc
    d=Vector(tgt)-Vector(loc); cam.rotation_euler=d.to_track_quat('-Z','Y').to_euler()
    cam.data.type='ORTHO'; cam.data.ortho_scale=ortho
    sc.render.filepath="previews/raw_%s.png"%name
    bpy.ops.render.render(write_still=True)
