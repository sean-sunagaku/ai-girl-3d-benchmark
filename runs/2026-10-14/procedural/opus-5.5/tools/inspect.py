import bpy, bmesh, numpy as np
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis='Y', up_axis='Z')
ob = bpy.context.selected_objects[0]
me = ob.data
print("verts", len(me.vertices), "faces", len(me.polygons))
co = np.array([v.co for v in me.vertices])
print("bounds", co.min(0), co.max(0))
bm = bmesh.new(); bm.from_mesh(me)
bm.edges.ensure_lookup_table()
bnd = [e for e in bm.edges if e.is_boundary]
# loops
import collections
adj = collections.defaultdict(list)
for e in bnd:
    a,b = e.verts; adj[a.index].append(b.index); adj[b.index].append(a.index)
seen=set(); loops=[]
for s in adj:
    if s in seen: continue
    stack=[s]; comp=[]
    while stack:
        v=stack.pop()
        if v in seen: continue
        seen.add(v); comp.append(v); stack+=adj[v]
    loops.append(comp)
for l in loops:
    c = co[l]
    print("loop n=%d center=%s min=%s max=%s"%(len(l), c.mean(0).round(4), c.min(0).round(4), c.max(0).round(4)))
nf = collections.Counter(len(f.verts) for f in bm.faces); print(nf)
