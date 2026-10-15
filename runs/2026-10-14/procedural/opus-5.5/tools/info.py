import bpy, numpy as np
for ob in bpy.data.objects:
    if ob.type=='MESH':
        co=np.array([v.co for v in ob.data.vertices]); 
        print(ob.name, len(co), co.min(0).round(3), co.max(0).round(3), ob.hide_render, ob.hide_get())
    else: print(ob.name, ob.type)
