import bpy, bmesh, numpy as np
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis='Y', up_axis='Z')
ob = bpy.context.selected_objects[0]; me=ob.data
co = np.array([v.co for v in me.vertices])
bm=bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
def rings(seed, n=8):
    seen={seed}; ring=[seed]; out=[]
    for i in range(n):
        out.append(ring)
        nxt=[]
        for vi in ring:
            for e in bm.verts[vi].link_edges:
                o=e.other_vert(bm.verts[vi]).index
                if o not in seen: seen.add(o); nxt.append(o)
        ring=nxt
    return out
for label,c in [("eye",(0.031,-0.13,1.546)),("mouth",(0.0,-0.13,1.49))]:
    # deepest vertex near center line in x,z
    d=np.hypot(co[:,0]-c[0], co[:,2]-c[2])
    cand=np.where((d<0.006)&(co[:,1]>-0.17)&(co[:,1]<-0.09))[0]
    print(label, "cands", [(i, co[i].round(4).tolist()) for i in cand])
    seed=cand[np.argmax(co[cand,1])]
    for k,r in enumerate(rings(seed,9)):
        p=co[r]
        print(k,len(r),"meanY %.4f minY %.4f maxY %.4f xspan %.4f..%.4f zspan %.4f..%.4f"%(p[:,1].mean(),p[:,1].min(),p[:,1].max(),p[:,0].min(),p[:,0].max(),p[:,2].min(),p[:,2].max()))
