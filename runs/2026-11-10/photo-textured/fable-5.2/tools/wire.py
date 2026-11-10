"""Draw orthographic wireframes of the supplied base (prepared frame) with PIL for topology study."""
import numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
def load(path):
    V=[];F=[];G=[];cur=None
    for line in open(path):
        if line.startswith('v '): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith('g '): cur=line.split()[1]
        elif line.startswith('f '): F.append([int(t.split('/')[0])-1 for t in line.split()[1:]]); G.append(cur)
    return np.array(V),F,G
V,F,G=load('input/upper-body.obj')
def wire(out, axis_u, axis_v, lo, hi, size=1500, label=False, cull=None, flip_u=False):
    # lo, hi: (u0,v0),(u1,v1) window in meters
    s=size/(hi[0]-lo[0]); H=int((hi[1]-lo[1])*s)
    im=Image.new('RGB',(size,H),(255,255,255)); d=ImageDraw.Draw(im)
    try: font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',9)
    except Exception: font=ImageFont.load_default()
    def P(i):
        u=(V[i,axis_u]-lo[0])*s; 
        if flip_u: u=size-u
        return (u,H-(V[i,axis_v]-lo[1])*s)
    for f in F:
        if cull is not None and not cull(V[f]): continue
        pts=[P(i) for i in f]
        d.polygon(pts,outline=(60,60,60))
    if label:
        seen=set()
        for f in F:
            if cull is not None and not cull(V[f]): continue
            for i in f:
                if i in seen: continue
                seen.add(i); p=P(i)
                if 0<=p[0]<size and 0<=p[1]<H: d.text((p[0]+1,p[1]-4),str(i),fill=(200,0,0),font=font)
    im.save(out)
    print(out, im.size)
front=lambda P:(P[:,1].mean()<-0.06)
wire('previews/wire_face_front.png',0,2,(-0.085,1.43),(0.085,1.62),size=1600,cull=front)
wire('previews/wire_face_side.png',1,2,(-0.17,1.40),(0.07,1.67),size=1400,cull=lambda P:(P[:,0].mean()>-0.001),flip_u=False)
wire('previews/wire_eye.png',0,2,(0.008,1.527),(0.056,1.565),size=1600,cull=lambda P:(P[:,1].mean()<-0.10 and P[:,0].mean()>0),label=True)
wire('previews/wire_mouth.png',0,2,(-0.002,1.455),(0.040,1.505),size=1500,cull=lambda P:(P[:,1].mean()<-0.105 and P[:,0].mean()>-0.003),label=True)
wire('previews/wire_nose.png',0,2,(-0.002,1.49),(0.035,1.56),size=1300,cull=lambda P:(P[:,1].mean()<-0.12 and P[:,0].mean()>-0.003),label=True)
wire('previews/wire_torso_front.png',0,2,(-0.45,1.05),(0.45,1.45),size=1800,cull=lambda P:(P[:,1].mean()<-0.02 and P[:,2].mean()<1.46))
wire('previews/wire_torso_side.png',1,2,(-0.22,1.05),(0.12,1.45),size=1100,cull=lambda P:(P[:,0].mean()>0.0 and P[:,2].mean()<1.46 and P[:,0].mean()<0.2))
wire('previews/wire_torso_top.png',0,1,(-0.45,-0.22),(0.45,0.12),size=1800,cull=lambda P:(P[:,2].mean()<1.46 and P[:,2].mean()>1.2))
