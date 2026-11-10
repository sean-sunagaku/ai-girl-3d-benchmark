"""Inspect helper groups of the supplied base.obj in the prepared (meters, Z-up) frame."""
import numpy as np, collections, json
def load(path):
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
B,BT,BF,BFT,BG=load('input/base.obj')
T=np.stack([B[:,0]*0.1,-B[:,2]*0.1,(B[:,1]+8.168)*0.1],1)
gv=collections.defaultdict(set)
for f,g in zip(BF,BG):
    gv[g].update(f)
for g in sorted(gv):
    if g.startswith('joint'): continue
    idx=np.array(sorted(gv[g])); P=T[idx]
    print('%-24s nv=%5d idx[%d..%d] min=%s max=%s mean=%s'%(g,len(idx),idx.min(),idx.max(),np.round(P.min(0),4),np.round(P.max(0),4),np.round(P.mean(0),4)))
# eye sphere fit
for g in ('helper-l-eye','helper-r-eye'):
    idx=np.array(sorted(gv[g])); P=T[idx]; c=P.mean(0)
    A=np.c_[2*P,np.ones(len(P))]; b=(P**2).sum(1); sol=np.linalg.lstsq(A,b,rcond=None)[0]; cen=sol[:3]; r=np.sqrt(sol[3]+(cen**2).sum())
    d=np.linalg.norm(P-cen,axis=1)
    print(g,'fit center',np.round(cen,5),'r',round(float(r),5),'dist range',round(float(d.min()),5),round(float(d.max()),5),'front-most y',round(float(P[:,1].min()),5))
U,UT,UF,UFT,UG=load('input/upper-body.obj')
# map upper-body verts -> base body verts (exact position match)
body=np.array(sorted(gv['body']))
key=lambda p:(round(p[0],5),round(p[1],5),round(p[2],5))
d={key(T[i]):i for i in body}
m=[d.get(key(p),-1) for p in U]
m=np.array(m); print('upper->base matched',(m>=0).sum(),'of',len(U),'base body verts',len(body),'body idx range',body.min(),body.max())
print('is prefix order?', np.all(np.diff(m[m>=0])>0), m[:10])
# UV coverage for upper-body faces
uvs=[]
fb={tuple(sorted(f)):ft for f,ft,g in zip(BF,BFT,BG) if g=='body'}
cnt=0
for f in UF:
    k=tuple(sorted(m[f]))
    if k in fb: cnt+=1
print('upper faces found in base (uv available):',cnt,'of',len(UF))
allft=np.array([t for f,ft,g in zip(BF,BFT,BG) if g=='body' for t in ft]); print('body uv range',BT[allft].min(0),BT[allft].max(0))
