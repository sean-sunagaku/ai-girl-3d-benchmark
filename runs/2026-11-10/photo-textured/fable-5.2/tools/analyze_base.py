"""Numeric landmark study of the supplied MakeHuman base (rest pose). Blender-side (uses BVH)."""
import bpy, bmesh, json, math, collections
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def load_obj(path):
    V=[];VT=[];F=[];FT=[];G=[];cur=None
    for line in open(path):
        if line.startswith('v '): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('vt '): VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith('g '): cur=line.split()[1]
        elif line.startswith('f '):
            idx=[];tidx=[]
            for tok in line.split()[1:]:
                p=tok.split('/'); idx.append(int(p[0])-1); tidx.append(int(p[1])-1 if len(p)>1 and p[1] else -1)
            F.append(idx);FT.append(tidx);G.append(cur)
    return np.array(V),np.array(VT),F,FT,G
B,BT,BF,BFT,BG=load_obj('input/base.obj')
T=np.stack([B[:,0]*0.1,-B[:,2]*0.1,(B[:,1]+8.168)*0.1],1)
body=[(f,ft) for f,ft,g in zip(BF,BFT,BG) if g=='body']
zcut=1.0968
upper=[(f,ft) for f,ft in body if min(T[i,2] for i in f)>=zcut-1e-9]
rest=[(f,ft) for f,ft in body if not min(T[i,2] for i in f)>=zcut-1e-9]
arm=[(f,ft) for f,ft in rest if min(abs(T[i,0]) for i in f)>0.26 and min(T[i,2] for i in f)>0.85]
print('upper faces',len(upper),'arm/hand faces',len(arm))
zs=[T[i,2] for f,ft in arm for i in f]; xs=[abs(T[i,0]) for f,ft in arm for i in f]
print('arm z range',min(zs),max(zs),'|x| range',min(xs),max(xs))
# other faces between z 0.85..1.0968 and their |x| max
oth=[(f,ft) for f,ft in rest if not (min(abs(T[i,0]) for i in f)>0.26 and min(T[i,2] for i in f)>0.85)]
ox=[abs(T[i,0]) for f,ft in oth for i in f if T[i,2]>0.85]; print('non-arm max |x| above z=0.85:',max(ox))
faces=upper+arm
used=sorted({i for f,ft in faces for i in f}); remap={v:k for k,v in enumerate(used)}
V=T[used]
F=[[remap[i] for i in f] for f,ft in faces]
print('assembled verts',len(V),'faces',len(F))
bm=bmesh.new()
bv=[bm.verts.new(p) for p in V]
for f in F:
    try: bm.faces.new([bv[i] for i in f])
    except ValueError: print('dup face')
bm.normal_update(); bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
tree=BVHTree.FromBMesh(bm)
N=np.array([v.normal[:] for v in bm.verts])
enc=np.zeros(len(V),bool)
for i,v in enumerate(bm.verts):
    o=v.co+v.normal*1e-4
    hit=tree.ray_cast(o,v.normal,0.06)
    if hit[0] is not None: enc[i]=True
print('enclosed verts',enc.sum())
out={}
def near(c,r): return np.where(np.linalg.norm(V-np.array(c),axis=1)<r)[0]
EYE=np.array([0.03078,-0.12454,1.54521])
idx=near(EYE,0.03)
print('near eye',len(idx),'enclosed',enc[idx].sum())
# margin: non-enclosed verts adjacent to enclosed verts, near eye
adj=collections.defaultdict(set)
for e in bm.edges:
    a,b=e.verts[0].index,e.verts[1].index; adj[a].add(b); adj[b].add(a)
margin=[i for i in idx if not enc[i] and any(enc[j] for j in adj[i])]
P=V[margin]; ang=np.degrees(np.arctan2(P[:,2]-EYE[2],P[:,0]-EYE[0]))
order=np.argsort(ang)
print('L eye margin loop (non-enclosed adjacent to enclosed): n=',len(margin))
for k in order:
    i=margin[k]; print('  v%5d ang=%7.1f pos=(%.4f,%.4f,%.4f) dist=%.4f'%(i,ang[k],V[i,0],V[i,1],V[i,2],np.linalg.norm(V[i]-EYE)))
out['eye_margin_L']=[int(margin[k]) for k in order]
# enclosed near eye: distance stats
de=np.linalg.norm(V[idx][enc[idx]]-EYE,axis=1); print('socket bag dist from eye center: min %.4f max %.4f mean %.4f'%(de.min(),de.max(),de.mean()))
# midline profile (x~0, front)
mid=[i for i in range(len(V)) if abs(V[i,0])<1e-5 and V[i,1]<0.0 and not enc[i] and V[i,2]>1.38]
mid.sort(key=lambda i:-V[i,2])
print('MIDLINE profile (z, y):')
for i in mid: print('  v%5d z=%.4f y=%.4f'%(i,V[i,2],V[i,1]))
out['midline']=[int(i) for i in mid]
# face half-width by z slices (non-enclosed, y<0.0)
print('half-width by z (front half y<-0.02):')
for z in np.arange(1.40,1.67,0.01):
    s=[i for i in range(len(V)) if abs(V[i,2]-z)<0.006 and V[i,1]<-0.02 and not enc[i] and abs(V[i,0])<0.12]
    if s:
        j=max(s,key=lambda i:V[i,0]); print('  z=%.2f xmax=%.4f at y=%.4f (v%d)'%(z,V[j,0],V[j,1],j))
# mouth: enclosed verts near mouth; slit line
M=np.array([0,-0.13,1.478]); im=near(M,0.045)
lipm=[i for i in im if not enc[i] and any(enc[j] for j in adj[i])]
print('mouth margin n=',len(lipm))
for i in sorted(lipm,key=lambda i:(V[i,0],V[i,2])):
    if V[i,0]>=-1e-6: print('  v%5d pos=(%.4f,%.4f,%.4f)'%(i,V[i,0],V[i,1],V[i,2]))
out['mouth_margin']=[int(i) for i in lipm]
# nostril enclosed
NO=np.array([0.012,-0.145,1.502]); ino=near(NO,0.02)
print('nostril enclosed verts:',[(int(i),tuple(np.round(V[i],4))) for i in ino if enc[i] and V[i,0]>0][:30])
# head extremes
hd=[i for i in range(len(V)) if V[i,2]>1.42]
print('head top',V[hd][:,2].max(),'back y max',V[hd][:,1].max(),'front y min',V[hd][:,1].min(),'x max',V[hd][:,0].max())
ear=[i for i in hd if V[i,0]>0.068]; print('ear-ish verts n',len(ear),'z range',V[ear][:,2].min(),V[ear][:,2].max(),'y range',V[ear][:,1].min(),V[ear][:,1].max(),'x max',V[ear][:,0].max())
out['enclosed']=[int(i) for i in np.where(enc)[0]]
json.dump(out,open('previews/analysis.json','w'))
