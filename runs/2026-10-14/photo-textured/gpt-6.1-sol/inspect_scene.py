import bpy
from mathutils import Vector
bpy.ops.wm.open_mainfile(filepath='character.blend');dg=bpy.context.evaluated_depsgraph_get()
for o in bpy.data.objects:
 if o.type in ('MESH','CURVE') and any(c.name.startswith(('01','02','03','04')) for c in o.users_collection):
  ev=o.evaluated_get(dg);me=ev.to_mesh();pts=[o.matrix_world@v.co for v in me.vertices];print(o.name,len(me.vertices), [[round(min(p[k] for p in pts),4),round(max(p[k] for p in pts),4)] for k in range(3)]);ev.to_mesh_clear()
