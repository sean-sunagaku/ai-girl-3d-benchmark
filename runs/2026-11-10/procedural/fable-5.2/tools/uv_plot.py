"""Plot the MakeHuman UV layout for the assembled upper body + arms (study only)."""
import numpy as np
from PIL import Image, ImageDraw
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
S=1400
im=Image.new('RGB',(S,S),(255,255,255)); d=ImageDraw.Draw(im)
for f,ft,g in zip(BF,BFT,BG):
    if g!='body': continue
    z=min(T[i,2] for i in f); ax=min(abs(T[i,0]) for i in f)
    if z>=1.0968-1e-9: 
        zc=np.mean([T[i,2] for i in f]); yc=np.mean([T[i,1] for i in f])
        col=(200,0,0) if zc>1.42 and yc<-0.06 else ((0,120,0) if zc>1.42 else (0,0,200))
    elif ax>0.26 and z>0.85: col=(200,120,0)
    else: continue
    pts=[(BT[t,0]*S,(1-BT[t,1])*S) for t in ft]
    d.polygon(pts,outline=col)
im.save('previews/uv_layout.png'); print('ok')
