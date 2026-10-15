# Blender -b file --python tools/section.py -- out.png axis(x|y|z) value halfwidth  h0,h1 v0,v1 obj1 obj2...
import bpy, sys, numpy as np
a = sys.argv[sys.argv.index("--")+1:]
out, axis, val, hw = a[0], "xyz".index(a[1]), float(a[2]), float(a[3])
h0,h1 = map(float,a[4].split(",")); v0,v1 = map(float,a[5].split(","))
objs = a[6:]
other = [i for i in range(3) if i != axis]
W=700; img=np.ones((W,W,4),np.float32)
cols=[(0,0,0),(1,0,0),(0,0,1),(0,0.6,0)]
dg=bpy.context.evaluated_depsgraph_get()
for k,name in enumerate(objs):
    ob=bpy.data.objects[name]; me=ob.evaluated_get(dg).to_mesh()
    co=np.array([ob.matrix_world @ v.co for v in me.vertices]); ed=np.array([e.vertices[:] for e in me.edges])
    m=np.abs(co[:,axis]-val)<hw
    for i,j in ed:
        if m[i] and m[j]:
            for t in np.linspace(0,1,12):
                p=co[i]*(1-t)+co[j]*t
                u=int((p[other[0]]-h0)/(h1-h0)*(W-1)); v=int((p[other[1]]-v0)/(v1-v0)*(W-1))
                if 0<=u<W and 0<=v<W: img[v,u,:3]=cols[k%4]
for g in np.arange(np.ceil(h0*50)/50,h1,0.02):
    u=int((g-h0)/(h1-h0)*(W-1)); img[:,u,:3]=np.minimum(img[:,u,:3],0.85)
for g in np.arange(np.ceil(v0*50)/50,v1,0.02):
    v=int((g-v0)/(v1-v0)*(W-1)); img[v,:,:3]=np.minimum(img[v,:,:3],0.85)
im=bpy.data.images.new("s",W,W); im.pixels=img.ravel(); im.filepath_raw=out; im.file_format='PNG'; im.save()
