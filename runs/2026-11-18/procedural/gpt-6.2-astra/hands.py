"""Inferred hands beyond the photographic crop, modeled as a continuous surface."""
from atelier import *
from knit import sleeve_path


def make_hands(C):
    skin=bpy.data.materials['Skin | apricot, rose and vermilion']
    nail,_=principle('Nails | translucent natural rose',(.60,.325,.294),.32,**{'Coat Weight':.28,'Coat Roughness':.21,'Subsurface Weight':.12,'Subsurface Scale':.0006})
    for side,label in [(-1,'R'),(1,'L')]:
        controls=sleeve_path(side,(0,0,0));wrist=np.array(controls[-1]);forward=wrist-np.asarray(controls[-2]);forward=forward.astype(float);forward/=np.linalg.norm(forward)
        top=np.array((0.,0.,1.));top-=forward*np.dot(forward,top);top/=np.linalg.norm(top)
        inward=np.cross(top,forward)*(-side);angle=math.radians(42 if side<0 else 48)
        U=inward*cos(angle)+top*sin(angle);H=top*cos(angle)-inward*sin(angle)
        def world(q):return wrist+q[0]*U+q[1]*forward+q[2]*H
        verts=[];colors=[];fs=[];nails=[];nailfaces=[];lookup={}
        def vertex(q,palmar=0):
            key=tuple(round(float(x),4) for x in q)
            if key in lookup:return lookup[key]
            idx=len(verts);lookup[key]=idx;verts.append(world(q));red=.23*gauss(q[1]-91,28)+.22*smooth(117,153,q[1]);c=lerp(np.array((.625,.339,.263)),np.array((.60,.256,.224)),red);c=lerp(c,np.array((.70,.388,.318)),palmar*.35);colors.append((*c,1));return idx
        # The digit webs close at the distal row. The thumb grows from a cutout on its radial side.
        n=24;sub=6;rows=[(-30,25,16),(-18,25.6,16),(-5,26.2,16.5),(11,28.7,17),(30,33.0,18),(48,36.8,16.5),(64,37.3,14.7),(74,36.7,12.5),(82,35.5,11),(88,35.,9.3)]
        grids={};local={};skipped=[]
        roots_t=np.array([71.,79.,87.,90.,85.])
        for j,(t,w,depth) in enumerate(rows):
            for face in (-1,1):
                ids=[]
                for i in range(n+1):
                    u=i/n;unit=(2*u-1);x=w*unit+2.8*sin(pi*j/(len(rows)-1));rounded=max(0,1-unit**2)**.47
                    distal=smooth(44,88,t);end_t=np.interp(i/sub,np.arange(5),roots_t);tg=t+distal*(end_t-88)
                    wave=abs(sin(pi*(i%sub)/sub))**.60
                    thick=depth*lerp(rounded,wave,smooth(71,88,t));palmarch=(2.3*(1-unit*unit)+1.7*unit)*distal
                    z=palmarch+face*thick*(1.07 if face<0 else .88)
                    if face<0:z-=2.0*gauss(x-16,13)*gauss(t-34,22)*rounded
                    if face>0:z+=1.4*gauss(t-70,11)*wave*rounded
                    q=np.array((x,tg,z));k=vertex(q,face<0);local[k]=q;ids.append(k)
                grids[j,face]=ids
        for j in range(len(rows)-1):
            for face in (-1,1):
                lo=grids[j,face];hi=grids[j+1,face]
                for i in range(n):
                    f=(lo[i],lo[i+1],hi[i+1],hi[i]);f=f if face>0 else tuple(reversed(f))
                    if 2<=j<5 and i>=n-3:skipped.append(f)
                    else:fs.append(f)
        def make_nail(center,width,length,normal,axis,across):
            start=len(nails);segments=20;rings=5
            for j in range(rings):
                r=.005+.995*j/(rings-1)
                for k in range(segments):
                    a=2*pi*k/segments;dx=r*cos(a);dy=r*sin(a);rise=.51*(1-dx*dx)-.30*dy*dy
                    q=center+across*width*dx+axis*length*dy+normal*rise;nails.append(world(q))
            for j in range(rings-1):
                for k in range(segments):
                    a=start+j*segments+k;b=start+j*segments+(k+1)%segments;nailfaces.append((a,a+segments,b+segments,b))
        for finger,(length,bend,spread) in enumerate([(59,46,-6.4),(72,49,-3.6),(80,39,-.6),(76,29,3.2)]):
            sel=slice(finger*sub,(finger+1)*sub+1);d=grids[len(rows)-1,1][sel];v=grids[len(rows)-1,-1][sel];root=d+list(reversed(v[1:-1]));rp=np.array([local[k] for k in root]);center=rp.mean(0);center[1]+=1
            theta=np.arange(len(root))*2*pi/len(root);rx=(rp[:,0].max()-rp[:,0].min())*.50;prev=root
            def finger_axis(t):
                a=math.radians(bend)*(smooth(.17,.53,t)*.70+smooth(.56,.86,t)*.70)
                return np.array((0.,cos(a),-sin(a))),np.array((0.,sin(a),cos(a)))
            def finger_centre(t):
                # Integrate the flexion at the two interphalangeal joints.
                ts=np.linspace(0,t,18);direction=np.mean([finger_axis(v)[0] for v in ts],axis=0)
                return center+direction*(t*length)+np.array((spread*t*t,0,0))
            for j,t in enumerate([.035,.095,.20,.33,.42,.48,.54,.65,.72,.78,.85,.90,.94,.976,.995,1.]):
                axis,normal=finger_axis(t);cc=finger_centre(t); taper=np.interp(t,[0,.20,.39,.46,.59,.72,.86,.94,.98,1],[1,.94,.93,.99,.84,.85,.76,.70,.46,.07]);cur=[]
                for i,ang in enumerate(theta):
                    target=cc+np.array((-cos(ang)*rx*taper,0,0))+normal*sin(ang)*rx*taper*.91
                    q=lerp(rp[i]+(cc-center),target,smooth(0,.20,t));cur.append(vertex(q,sin(ang)<0))
                for i in range(len(cur)):
                    k=(i+1)%len(cur);fs.append((prev[k],prev[i],cur[i],cur[k]))
                prev=cur
            fs.append(tuple(reversed(prev)));ax,no=finger_axis(.86);pos=finger_centre(.86)+no*(rx*.76*.91-.10);make_nail(pos,rx*.56,length*.076,no,ax,np.array((1,0,0)))
        # Recover the boundary of the omitted patch, in contour order.
        from collections import Counter
        counts=Counter(tuple(sorted((a,b))) for f in skipped for a,b in zip(f,f[1:]+f[:1]));edge={a:b for f in skipped for a,b in zip(f,f[1:]+f[:1]) if counts[tuple(sorted((a,b)))]==1};start=next(iter(edge));root=[start];k=edge[start]
        while k!=start:
            root.append(k);k=edge[k]
        rp=np.array([local[i] for i in root]);rc=rp.mean(0);path=[rc,(38,38,-.9),(47,52,-3.2),(52,66,-6.1),(53,78,-11),(50.7,89,-16),(48.8,93,-17.5)];ts=np.linspace(0,1,17);pc=catmull(path,ts);tang=np.gradient(pc,axis=0);tang/=np.linalg.norm(tang,axis=1)[:,None];topl=np.array((0.,0.,1.));xv=np.cross(tang[0],topl);xv/=np.linalg.norm(xv);yv=np.cross(xv,tang[0]);angles=np.arctan2((rp-rc)@yv,(rp-rc)@xv);prev=root
        for j,t in enumerate(ts[1:],1):
            xv=np.cross(tang[j],topl);xv/=np.linalg.norm(xv);yv=np.cross(xv,tang[j]);radius=np.interp(t,[0,.23,.44,.64,.76,.88,.96,1],[15.3,11,9.6,9.7,8.2,7.7,6,.8]);cur=[]
            for i,ang in enumerate(angles):
                q=pc[j]+(xv*cos(ang)+yv*sin(ang)*.88)*radius;q=lerp(rp[i]+(pc[j]-rc),q,smooth(0,.30,t));cur.append(vertex(q,sin(ang)<0))
            for i in range(len(cur)):
                k=(i+1)%len(cur);fs.append((prev[i],prev[k],cur[k],cur[i]))
            prev=cur
        fs.append(tuple(prev));j=13;ax=tang[j];ac=np.cross(ax,topl);ac/=np.linalg.norm(ac);no=np.cross(ac,ax);make_nail(pc[j]+no*6.95,4.5,5.7,no,ax,ac)
        ob=mesh('12 • Reaching hand.'+label,verts,fs,skin,C);attr_color(ob,'Complexion',colors);attr_color(ob,'Hydration',[(.15,.15,.15,1)]*len(colors));subdiv(ob,1)
        # Recalculate the consistently outward-facing shell, including the digit web saddles.
        import bmesh
        bm=bmesh.new();bm.from_mesh(ob.data);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(ob.data);bm.free()
        na=mesh('13 • Nail plates.'+label,nails,nailfaces,nail,C);subdiv(na,1)
