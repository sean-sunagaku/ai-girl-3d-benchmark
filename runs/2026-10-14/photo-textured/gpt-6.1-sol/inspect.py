import bpy, math, json
from mathutils import Vector
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.obj_import(filepath='input/upper-body.obj')
o=bpy.context.object;o.rotation_euler=(0,0,0)
for p in o.data.polygons:p.use_smooth=True
s=o.modifiers.new('Smooth','SUBSURF');s.levels=2
m=bpy.data.materials.new('clay');m.diffuse_color=(.55,.35,.25,1);o.data.materials.append(m)
def aim(o,p):o.rotation_euler=(Vector(p)-o.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(0,-.75,1.55));c=bpy.context.object;c.data.type='ORTHO';c.data.ortho_scale=.30;aim(c,(0,-.07,1.54));bpy.context.scene.camera=c
for pos,power,size in [((-.3,-.4,1.8),45,.35),((.25,-.2,1.6),15,.25)]:
 bpy.ops.object.light_add(type='AREA',location=pos);l=bpy.context.object;l.data.energy=power;l.data.shape='DISK';l.data.size=size;aim(l,(0,-.05,1.54))
sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=12;sc.cycles.use_denoising=True;sc.render.resolution_x=640;sc.render.resolution_y=640;sc.render.resolution_percentage=100;sc.world.color=(.2,.2,.2);sc.render.filepath='previews/base-face.png';bpy.ops.render.render(write_still=True)
c.location=(.6,-.6,1.55);aim(c,(0,-.04,1.54));sc.render.filepath='previews/base-quarter.png';bpy.ops.render.render(write_still=True)
