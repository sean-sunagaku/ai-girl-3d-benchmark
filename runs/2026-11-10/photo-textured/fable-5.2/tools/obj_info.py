"""Topology inspection of the supplied OBJ files (plain Python, no Blender)."""
import numpy as np, collections, sys

def load(path):
    V=[];VT=[];F=[];FT=[];G=[];cur=None
    for line in open(path):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('vt '):
            VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith('g '):
            cur=line.split()[1]
        elif line.startswith('f '):
            idx=[];tidx=[]
            for tok in line.split()[1:]:
                p=tok.split('/')
                idx.append(int(p[0])-1)
                tidx.append(int(p[1])-1 if len(p)>1 and p[1] else -1)
            F.append(idx);FT.append(tidx);G.append(cur)
    return np.array(V),np.array(VT),F,FT,G

V,VT,F,FT,G=load('input/upper-body.obj')
print('upper verts',V.shape,'faces',len(F),'face sizes',collections.Counter(len(f) for f in F))
print('bounds',V.min(0),V.max(0))
edges=collections.Counter()
for f in F:
    for i in range(len(f)):
        a,b=f[i],f[(i+1)%len(f)]
        edges[(min(a,b),max(a,b))]+=1
print('edge use counts',collections.Counter(edges.values()))
bnd=[e for e,c in edges.items() if c==1]
# boundary loops
adj=collections.defaultdict(list)
for a,b in bnd:
    adj[a].append(b);adj[b].append(a)
seen=set();loops=[]
for s in adj:
    if s in seen: continue
    loop=[s];seen.add(s);stack=[s]
    while stack:
        x=stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y);loop.append(y);stack.append(y)
    loops.append(loop)
print('boundary loops',len(loops))
for l in loops:
    P=V[l]
    print('  n=%d center=%s min=%s max=%s'%(len(l),np.round(P.mean(0),4),np.round(P.min(0),4),np.round(P.max(0),4)))
# connected components
parent=list(range(len(V)))
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]];x=parent[x]
    return x
for a,b in edges:
    ra,rb=find(a),find(b)
    if ra!=rb: parent[ra]=rb
comp=collections.Counter(find(i) for i in range(len(V)))
print('components',len(comp),sorted(comp.values(),reverse=True)[:10])

B,BT,BF,BFT,BG=load('input/base.obj')
print('base verts',B.shape,'vt',BT.shape,'faces',len(BF))
gc=collections.Counter(BG)
print('non-joint groups',{k:v for k,v in gc.items() if not k.startswith('joint')})
# transform base to prepared coordinates
T=np.stack([B[:,0]*0.1,-B[:,2]*0.1,(B[:,1]+8.168)*0.1],1)
# map upper verts to base verts by position
from scipy.spatial import cKDTree
