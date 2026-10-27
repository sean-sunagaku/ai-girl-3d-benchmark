"""Render the saved scene from its beauty camera and four consistent studio angles.
Use -- --preview for smaller, quick studio views. Does not overwrite character.blend.
"""
import bpy, os, math, sys
from mathutils import Vector
root=os.path.dirname(os.path.abspath(__file__)); os.chdir(root)
s=bpy.context.scene; preview='--preview' in sys.argv
s.render.engine='CYCLES'; s.cycles.device='CPU'; s.cycles.samples=8 if preview else 32; s.cycles.use_denoising=True
s.render.threads_mode='FIXED'; s.render.threads=3; s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG'; s.render.image_settings.color_mode='RGBA'
if not preview:
    s.camera=bpy.data.objects['Beauty camera']; s.render.resolution_x=960; s.render.resolution_y=1040
    s.render.filepath='//renders/beauty.png'; bpy.ops.render.render(write_still=True)
# Remove only the cafe scenery from this temporary in-memory rendering session.
# The editable .blend on disk is never saved with these changes.
for ob in list(bpy.data.collections['Cafe suggestion'].all_objects): bpy.data.objects.remove(ob,do_unlink=True)
world=s.world; world.use_nodes=True; bg=world.node_tree.nodes.get('Background')
bg.inputs['Color'].default_value=(.23,.23,.23,1); bg.inputs['Strength'].default_value=.6

def track(ob,p): ob.rotation_euler=(Vector(p)-ob.location).to_track_quat('-Z','Y').to_euler()
def light(name,loc,power):
    data=bpy.data.lights.new(name,'AREA'); data.energy=power; data.shape='DISK'; data.size=1.7
    ob=bpy.data.objects.new(name,data); s.collection.objects.link(ob); ob.location=loc; track(ob,(0,-.08,1.32))
for name,loc,power in [('Studio front',(-1,-2,2.1),14),('Studio right',(2,0,1.9),17),('Studio rear',(0,2,2.0),17),('Studio left',(-2,0,1.7),14)]: light(name,loc,power)
target=Vector((0,-.14,1.33)); distance=2.6
for name,angle in [('front',0),('three-quarter',45),('side',90),('back',180)]:
    a=math.radians(angle); data=bpy.data.cameras.new('Studio camera '+name); data.type='ORTHO'
    ob=bpy.data.objects.new('Studio camera '+name,data); s.collection.objects.link(ob)
    ob.location=target+Vector((distance*math.sin(a),-distance*math.cos(a),0)); track(ob,target); s.camera=ob
    bpy.context.view_layer.update()
    # Frame all evaluated character bounds in each direction, including the
    # forward-reaching sleeves; move the camera in its image plane only.
    inv=ob.matrix_world.inverted(); coords=[]; graph=bpy.context.evaluated_depsgraph_get()
    for item in bpy.data.collections['Character'].all_objects:
        if item.type not in {'MESH','CURVE'}: continue
        ev=item.evaluated_get(graph)
        if item.type=='CURVE':
            # Curve bound_box is a generic +/-1 in Blender 5.1; use control points.
            transform=inv @ ev.matrix_world
            coords.extend(transform @ Vector(p.co[:3]) for spline in item.data.splines for p in spline.points)
        else:
            coords.extend(inv @ ev.matrix_world @ Vector(corner) for corner in ev.bound_box)
    lo_x=min(v.x for v in coords); hi_x=max(v.x for v in coords)
    lo_y=min(v.y for v in coords); hi_y=max(v.y for v in coords)
    data.ortho_scale=max(hi_x-lo_x,hi_y-lo_y)*1.12
    ob.location+=ob.rotation_euler.to_matrix() @ Vector(((lo_x+hi_x)/2,(lo_y+hi_y)/2,0))
    s.render.resolution_x=s.render.resolution_y=384 if preview else 768
    s.render.filepath=('//previews/' if preview else '//renders/')+name+'.png'; bpy.ops.render.render(write_still=True)
