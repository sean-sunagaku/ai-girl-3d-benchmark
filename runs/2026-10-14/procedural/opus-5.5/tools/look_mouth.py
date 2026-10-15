import bpy, numpy as np
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis='Y', up_axis='Z')
ob = bpy.context.selected_objects[0]; me=ob.data
co=np.array([v.co for v in me.vertices])
# lip seam: vertices around z 1.47-1.49 front, print extreme lateral cavity-ish verts
m=(np.abs(co[:,0])<0.04)&(co[:,2]>1.465)&(co[:,2]<1.495)&(co[:,1]<-0.12)
idx=np.where(m)[0]
# find verts whose neighbors go deep (y>-0.125) 
nb={i:set() for i in idx}
for e in me.edges:
    a,b=e.vertices
    if a in nb: nb[a].add(b)
    if b in nb: nb[b].add(a)
seam=[i for i in idx if any(co[j,1]>-0.125 for j in nb[i])]
for i in sorted(seam,key=lambda i:co[i,0]): print(i, co[i].round(4))
sc=bpy.context.scene; sc.render.engine='BLENDER_WORKBENCH'; sc.display.shading.show_cavity=True
sc.render.resolution_x=800; sc.render.resolution_y=800
cam=bpy.data.objects.new("c",bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera=cam
cam.data.type='ORTHO'; cam.data.ortho_scale=0.08
cam.location=(0,-1,1.49); cam.rotation_euler=(Vector((0,1,0))).to_track_quat('-Z','Y').to_euler()
sc.render.filepath="previews/raw_mouth.png"; bpy.ops.render.render(write_still=True)
