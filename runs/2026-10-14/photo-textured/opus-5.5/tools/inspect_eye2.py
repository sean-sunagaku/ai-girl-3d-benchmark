import bpy, bmesh, numpy as np
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis='Y', up_axis='Z')
ob = bpy.context.selected_objects[0]; me=ob.data
co = np.array([v.co for v in me.vertices])
bm=bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
# components
import collections
lab=-np.ones(len(co),int); c=0
for s in range(len(co)):
    if lab[s]>=0: continue
    st=[s]
    while st:
        v=st.pop()
        if lab[v]>=0: continue
        lab[v]=c
        st+=[e.other_vert(bm.verts[v]).index for e in bm.verts[v].link_edges]
    c+=1
print("components",c,collections.Counter(lab))
for k in range(c):
    p=co[lab==k]; print(k,p.min(0).round(4),p.max(0).round(4))
seed=4348
seen={seed}; ring=[seed]
for i in range(18):
    p=co[ring]
    print(i,len(ring),"meanY %.4f minY %.4f maxY %.4f x %.4f..%.4f z %.4f..%.4f"%(p[:,1].mean(),p[:,1].min(),p[:,1].max(),p[:,0].min(),p[:,0].max(),p[:,2].min(),p[:,2].max()))
    nxt=[]
    for vi in ring:
        for e in bm.verts[vi].link_edges:
            o=e.other_vert(bm.verts[vi]).index
            if o not in seen: seen.add(o); nxt.append(o)
    ring=nxt
