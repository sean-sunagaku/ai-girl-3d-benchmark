"""A sewn quad sweater with UV-oriented, procedurally synthesized yarn relief."""
from atelier import *
from face import collar_height

TILE_X=15.0
TILE_Y=12.0

def stitch_tile():
    # Metric 4-wale by 4-course repeat. Yarn legs are swept in 2.5D so that
    # overlapping loops occlude their crowns and twist highlights follow each leg.
    n=384; u=(np.arange(n)+.5)*4/n
    X,Y=np.meshgrid(u,u); height=np.zeros((n,n)); twist=np.zeros((n,n)); variation=np.zeros((n,n))
    rng=np.random.default_rng(343); offset=rng.uniform(-.040,.040,(4,4,3))
    for row in range(-1,5):
        for column in range(-1,5):
            dx,dy,phase=offset[row%4,column%4];cx=column+.5+dx;cy=row+dy
            ix=np.where(abs(u-cx)<.64)[0];iy=np.where((u>cy-.29)&(u<cy+1.30))[0]
            if not len(ix) or not len(iy):continue
            sel=np.ix_(iy,ix); xx=X[sel];yy=Y[sel]
            h=height[sel].copy();fiber=twist[sel].copy();var=variation[sel].copy()
            for side in (-1,1):
                # From lower valley, up the visible leg and around the back of the head.
                t=np.linspace(0,1,22); px=cx+side*(.025+.335*np.sin(t*pi*.83)**1.12); py=cy-.135+1.255*t
                px[-5:]-=side*np.array([.008,.028,.050,.084,.121])
                depth=.18+.09*np.sin(pi*t)-.24*np.maximum(0,(t-.71)/.29)**.72
                for j in range(len(t)-1):
                    vx=px[j+1]-px[j];vy=py[j+1]-py[j]; length=vx*vx+vy*vy
                    along=np.clip(((xx-px[j])*vx+(yy-py[j])*vy)/length,0,1)
                    ex=xx-px[j]-along*vx; ey=yy-py[j]-along*vy
                    width=.143+.025*sin(pi*t[j]); dist=np.sqrt(ex*ex+(ey*.88)**2)/width
                    cross=np.clip((ex*vy-ey*vx)/(sqrt(length)*width),-1,1)
                    arc=(j+along)/(len(t)-1)
                    fib=np.sin(2*pi*(6.8*arc+phase)+2.8*np.arcsin(cross))
                    new=(depth[j]+.49*np.sqrt(np.maximum(0,1-dist*dist))+.027*fib)*(dist<1.)
                    front=new>h;fiber=np.where(front,fib,fiber);var=np.where(front,.4+dx*2.3+.13*side,var);h=np.maximum(h,new)
            height[sel]=h;twist[sel]=fiber;variation[sel]=var
    heights=np.clip(height*1.14,0,1); value=np.clip(.562+.40*height+.048*twist+.054*variation+.031*np.cos(2*pi*(X-.5)+.23*np.sin(2*pi*Y)),0,1)
    arr=np.stack((heights,value,np.clip(.50+.37*twist,0,1),np.ones_like(value)),axis=-1).astype(np.float32)
    im=bpy.data.images.new('Stockinette • four wales × four courses',width=n,height=n,alpha=False)
    im.colorspace_settings.name='Non-Color';im.pixels.foreach_set(arr.ravel());im.update();im.pack()
    return im

def wool_material(image,rib=False):
    color=(.56,.245,.271)
    m,b=principle('Knit | '+('ribbed neck, hem and cuffs' if rib else 'dusty rose lambswool'),color,.78,**{'Sheen Weight':.40,'Sheen Roughness':.54,'Specular IOR Level':.27})
    tc=node(m,'ShaderNodeTexCoord','Sewn fabric UV',-940,100);tex=node(m,'ShaderNodeTexImage','Stockinette relief and ply',-735,250);tex.image=image;tex.interpolation='Linear';tex.extension='REPEAT'
    link(m,tc.outputs['UV'],tex.inputs['Vector']);split=node(m,'ShaderNodeSeparateColor','Height, yarn value, ply',-500,250);link(m,tex.outputs['Color'],split.inputs[0])
    tone=node(m,'ShaderNodeValToRGB','Rose marl',-270,330);tone.color_ramp.elements[0].position=.36;tone.color_ramp.elements[0].color=(.29,.105,.107,1);tone.color_ramp.elements[1].position=.97;tone.color_ramp.elements[1].color=(.730,.313,.326,1)
    link(m,split.outputs['Green'],tone.inputs['Fac'])
    noise=node(m,'ShaderNodeTexNoise','Soft dye variation',-545,-160);noise.inputs['Scale'].default_value=140;noise.inputs['Detail'].default_value=3.3;link(m,tc.outputs['Object'],noise.inputs[0]);mix=node(m,'ShaderNodeMixRGB','Heathered wool',-40,260);mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.19;link(m,tone.outputs[0],mix.inputs[1]);link(m,noise.outputs['Color'],mix.inputs[2]);link(m,mix.outputs[0],b.inputs['Base Color'])
    bump=node(m,'ShaderNodeBump','Interlocking yarn loops',-53,-60);bump.inputs['Distance'].default_value=.00080;bump.inputs['Strength'].default_value=.62;link(m,split.outputs['Red'],bump.inputs['Height']);link(m,bump.outputs[0],b.inputs['Normal'])
    if rib:
        axis=node(m,'ShaderNodeSeparateXYZ','Wale direction',-730,-405);link(m,tc.outputs['UV'],axis.inputs[0]);freq=node(m,'ShaderNodeMath','Twinned stitches per rib',-500,-404);freq.operation='MULTIPLY';freq.inputs[1].default_value=2*pi*104/42;link(m,axis.outputs['X'],freq.inputs[0]);wave=node(m,'ShaderNodeMath','Raised and recessed wales',-282,-403);wave.operation='COSINE';link(m,freq.outputs[0],wave.inputs[0]);value=node(m,'ShaderNodeMapRange','Rib trough shading',-68,-391);value.inputs['From Min'].default_value=-1;value.inputs['To Min'].default_value=.67;value.inputs['To Max'].default_value=1.0;link(m,wave.outputs[0],value.inputs[0]);tint=node(m,'ShaderNodeMixRGB','Folded rib colour',171,220);tint.blend_type='MULTIPLY';tint.inputs[0].default_value=.82;link(m,mix.outputs[0],tint.inputs[1]);link(m,value.outputs[0],tint.inputs[2]);link(m,tint.outputs[0],b.inputs['Base Color'])
    return m

class Pattern:
    def __init__(self):
        self.v=[]; self.f=[]; self.uv=[]; self.index={};self.mats=[]
    def point(self,p):
        key=tuple(round(float(i),4) for i in p)
        if key not in self.index:self.index[key]=len(self.v);self.v.append(tuple(p))
        return self.index[key]
    def quad(self,ids,uv,material=0,reverse=False):
        if len(set(ids))<3:return
        self.f.append(tuple(reversed(ids)) if reverse else tuple(ids));self.uv.extend(reversed(uv) if reverse else uv);self.mats.append(material)
    def finish(self,name,materials,C,sub=1):
        obj=mesh(name,self.v,self.f,materials[0],C,self.uv)
        for mat in materials[1:]:obj.data.materials.append(mat)
        obj.data.polygons.foreach_set('material_index',np.asarray(self.mats,dtype=np.int32));subdiv(obj,sub);return obj

# The collar follows the skin at its entire inside edge. The outer edge is the yoke seam.
def collar(angle,t,backside=False,ribs=True):
    si,co=sin(angle),cos(angle)
    xi=60.8*si;yi=-21-(65.0 if co>0 else 62.3)*co
    zi=collar_height(angle)
    xo=(80.2 if co>0 else 78.8)*si;yo=-21-(83.6 if co>0 else 79)*co
    zo=zi-(12.5+2.8*co)
    # A soft roll, with real small vertical ribs on the band.
    bulge=1.95*sin(pi*t);rib=(1.12*cos(104*angle)+.22*cos(208*angle))*sin(pi*t)**.42 if ribs else 0
    x=lerp(xi,xo,t)+(bulge+rib)*si*.77;y=lerp(yi,yo,t)-(bulge+rib)*co*.77
    z=lerp(zi,zo,t)+(bulge+rib)*.65
    if backside:z-=2.3
    return np.array((x,y,z))

def body_panel(u,d,stage,t):
    # d=-1 is front. u runs from left to right across the knitting pattern.
    ab=abs(u);angle=u*pi/2;cs=max(0,cos(angle));si=sin(angle);side=1 if u>=0 else -1
    def lower(z):
        width=np.interp(z,[1000,1031,1100,1170,1225,1250],[139,145,155,153,158,158]);cy=-37+4*(z-1000)/250
        radius=(99+7*gauss(z-1132,88)) if d<0 else (89+5*gauss(z-1200,77))
        x=width*si;y=cy+d*radius*cs
        y-=19*cs*gauss(abs(x)-77,50)*gauss(z-1229,67) if d<0 else 0
        return np.array((x,y,z))
    if stage=='lower':
        z=1000+250*t;p=lower(z);p[2]+=8.0*(side<0)*ab**3*t**2
    else:
        a=.40
        if ab<=a:
            ang=side*(ab/a)*pi/2
            if d>0:ang=pi-ang
            top=collar(ang,1,False,False)
        else:
            rr=(ab-a)/(1-a);neck=collar(side*pi/2,1,False,False)
            top=np.array([side*lerp(abs(neck[0]),181.8,rr),lerp(neck[1],-23-7*(side<0),rr),lerp(neck[2],1360+32*(side<0),rr)+8.3*sin(pi*rr)+15.5*sin(pi*rr)*(side<0)])
        start=lower(1250);start[2]+=8.0*(side<0)*ab**3;p=lerp(start,top,t)
        p[1]+=d*(41*sin(pi*t)**.87*ab**3+9.5*sin(pi*t)*cs)
        p[0]+=side*4.1*sin(pi*t)*ab
        z=p[2]
    # Broad gravity folds, deeper under arms and near the waistband.
    x=p[0];along=(x/147);env=sin(pi*sat((z-998)/391))**.7
    dr=(3.0*sin(.040*x+.009*z+1.3*sin(.010*z)) + 1.7*sin(.082*x-.013*z))*env
    dr+=4.7*gauss(z-1060,59)*sin(.083*x+.020*z)
    dr+=4.1*gauss(z-(1208-46*(1-abs(along))),28)*sin(.091*z+.045*abs(x))*abs(along)**1.7
    if stage=='upper':dr*=max(0,1-t**3)
    p[1]+=d*cs*dr;p[0]+=si*dr*.56
    return p

def sleeve_path(side,root):
    if side<0:
        return [root,(-189,-66,1322),(-202,-120,1292),(-197,-185,1261),(-180,-257,1241),(-160,-335,1234),(-139,-422,1231)]
    return [root,(179,-64,1299),(178,-117,1268),(160,-181,1239),(132,-249,1226),(101,-327,1219),(70,-405,1217)]

def make_sweater(C):
    img=stitch_tile();fabric=wool_material(img);rib=wool_material(img,True);mats=[fabric,rib]
    b=Pattern(); cols=80;nl=38;nt=26;us=np.linspace(-1,1,cols+1);edges={};bottom={}
    for d in (-1,1):
        rows=[];row_uv=[]
        for j in range(nl+nt+1):
            stage='lower' if j<=nl else 'upper';t=j/nl if j<=nl else (j-nl)/nt
            coords=[body_panel(u,d,stage,t) for u in us];rows.append([b.point(p) for p in coords]);row_uv.append([(u*195/TILE_X,(p[2]-1000)/TILE_Y) for u,p in zip(us,coords)])
        for j in range(len(rows)-1):
            for i in range(cols):
                ids=(rows[j][i],rows[j][i+1],rows[j+1][i+1],rows[j+1][i]);uv=(row_uv[j][i],row_uv[j][i+1],row_uv[j+1][i+1],row_uv[j+1][i]);b.quad(ids,uv,reverse=d>0)
        for side,k in [(-1,0),(1,cols)]:edges[(side,d)]=[r[k] for r in rows[nl:]]
        bottom[d]=rows[0]
    # Sleeves grow directly out of the body armholes; their seam vertices are shared.
    for side in (-1,1):
        root=edges[(side,-1)]+list(reversed(edges[(side,1)][1:-1]));n=len(root);rootp=np.array([b.v[i] for i in root]);c0=rootp.mean(axis=0)
        path=sleeve_path(side,c0)
        trav=np.linspace(0,1,85);centers=catmull(path,trav);tang=np.gradient(centers,axis=0);tang/=np.linalg.norm(tang,axis=1)[:,None]
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))];prev=root
        # Surface phase follows the upper seam from the underarm around the sleeve.
        angles=np.arange(n)*2*pi/n-pi/2
        for j,t in enumerate(trav[1:],1):
            normal=np.array((0.,0.,1.));normal-=tang[j]*np.dot(normal,tang[j]);normal/=np.linalg.norm(normal)
            across=np.cross(tang[j],normal)*side
            radius=np.interp(t,[0,.15,.32,.50,.65,.82,.89,.92,.985,1],[52,55,54,51,50,46,44.5,35.8,33,32.0]);curr=[]
            for i,ang in enumerate(angles):
                wave=2.4*sin(4*ang+7*t)+1.4*sin(9*ang-10*t+.5*sin(12*t));elbow=gauss(t-.56,.13)
                wave+=elbow*(3.8*sin(38*t+2.3*sin(ang+.9))+1.6*sin(72*t+1.8*cos(ang)))
                wave+=2.5*gauss(t-.835,.08)*sin(75*t+2*sin(ang))
                wave*=smooth(.015,.2,t)*(1-.86*smooth(.89,.93,t))
                pt=centers[j]+(radius+wave)*cos(ang)*across+(radius*(1.04-.23*smooth(.88,1,t))+wave)*sin(ang)*normal
                w=smooth(0,.20,t);pt=lerp(rootp[i]+(centers[j]-centers[0]),pt,w)
                curr.append(b.point(pt))
            for i in range(n):
                k=(i+1)%n;u0=i/n*302/TILE_X;u1=(i+1)/n*302/TILE_X
                uv=((u0,length[j-1]/TILE_Y),(u1,length[j-1]/TILE_Y),(u1,length[j]/TILE_Y),(u0,length[j]/TILE_Y))
                b.quad((prev[i],prev[k],curr[k],curr[i]),uv,material=int(t>.905),reverse=side>0)
            prev=curr
        # Rolled inner return at the cropped forearm; the garment has real wall thickness.
        terminal=centers[-1];rimpts=np.array([b.v[i] for i in prev]);p0=prev
        for shrink,dist in [(.97,.8),(.92,-1.5),(.91,-7),(.92,-17)]:
            new=[b.point(terminal+(p-terminal)*shrink+dist*tang[-1]) for p in rimpts]
            for i in range(n):
                k=(i+1)%n;b.quad((p0[i],p0[k],new[k],new[i]),[(i/n*21,0),((i+1)/n*21,0),((i+1)/n*21,.4),(i/n*21,.4)],1,side>0)
            p0=new
    # Hem band, narrowing and turning under at the lower edge.
    prev=bottom[-1]+list(reversed(bottom[1][1:-1]));n=len(prev);base=np.array([b.v[i] for i in prev]);ctr=np.array((0.,-37.,1000.))
    for j,(z,scale) in enumerate([(999,.999),(995,.995),(987,.992),(977,.988),(970,.984),(968,.980),(968,.96),(971,.957),(978,.957)]):
        cur=[]
        for i,p in enumerate(base):
            q=ctr+(p-ctr)*scale;q[2]=z+.9*cos(2*pi*i/n*3);cur.append(b.point(q))
        for i in range(n):
            k=(i+1)%n;b.quad((prev[i],prev[k],cur[k],cur[i]),[(i/n*55,(j-1)*.56),((i+1)/n*55,(j-1)*.56),((i+1)/n*55,j*.56),(i/n*55,j*.56)],1,True)
        prev=cur
    sweater=b.finish('10 • Sweater | joined shoulder and forward sleeves',mats,C)
    # Saddle and shoulder seam thread is stitched along the smoothed garment.
    from mathutils.bvhtree import BVHTree
    graph=bpy.context.evaluated_depsgraph_get();tree=BVHTree.FromObject(sweater,graph)
    yarn,_=principle('Knit | mattress-stitch yarn',(.496,.203,.215),.75,**{'Sheen Weight':.35});paths=[];rads=[]
    for side in [-1,1]:
        root=edges[(side,-1)]+list(reversed(edges[(side,1)][1:-1]));rim=np.array([b.v[k] for k in root+[root[0]]]);shoulder=np.array([body_panel(side*u,-1,'upper',1) for u in np.linspace(.4,1,35)])
        for guide in [rim,shoulder]:
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(guide,axis=0),axis=1))];n=int(distance[-1]/3.45)
            for k in range(n):
                pts=[]
                for t in np.linspace((k+.10)/n,(k+.82)/n,4):
                    p=np.array([np.interp(t*distance[-1],distance,guide[:,a]) for a in range(3)])*.001;hit,normal,_,_=tree.find_nearest(Vector(p))
                    if hit is not None:pts.append(np.asarray(hit+normal*(.00020+.00012*sin(pi*(len(pts)/3))))*1000)
                if len(pts)==4:paths.append(pts);rads.append([.14,.26,.26,.13])
    strands('15 • Mattress stitch | shoulder and sleeve caps',paths,rads,yarn,C)
    # A higher resolution ribbed collar is a separately editable sewn-on band.
    band=Pattern();n=512;previous=None
    for j,(t,under) in enumerate([(1,True),(.96,True),(.70,True),(.10,True),(0,True),(0,False),(.045,False),(.14,False),(.32,False),(.55,False),(.78,False),(.94,False),(1,False)]):
        row=[band.point(collar(2*pi*i/n,t,under)) for i in range(n)]
        if previous:
            for i in range(n):
                k=(i+1)%n;u0=i/n*42;u1=(i+1)/n*42;band.quad((previous[i],previous[k],row[k],row[i]),[(u0,pt*2),(u1,pt*2),(u1,t*2),(u0,t*2)],reverse=True)
        previous=row;pt=t
    band.finish('11 • Crew neck | 104 knitted ribs',[rib],C,1)
    # Sparse physical wool fibres interrupt the silhouette over the sewn garment.
    fuzz,fb=principle('Knit | loose rose wool fibres',(.61,.305,.306),.77,**{'Sheen Weight':.55})
    graph=bpy.context.evaluated_depsgraph_get();ev=sweater.evaluated_get(graph);me=ev.to_mesh();me.calc_loop_triangles()
    xyz=np.empty(len(me.vertices)*3,dtype=np.float32);me.vertices.foreach_get('co',xyz);xyz=xyz.reshape(-1,3)*1000
    ids=np.empty(len(me.loop_triangles)*3,dtype=np.int32);me.loop_triangles.foreach_get('vertices',ids);tris=xyz[ids.reshape(-1,3)];ev.to_mesh_clear()
    norm=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);weights=np.linalg.norm(norm,axis=1);norm/=np.maximum(weights[:,None],1e-10);rng=np.random.default_rng(5408);chosen=rng.choice(len(tris),6800,p=weights.astype(float)/weights.astype(float).sum());uv=rng.random((len(chosen),2));uv=np.where(uv.sum(1)[:,None]>1,1-uv,uv);tri=tris[chosen];root=tri[:,0]+uv[:,0,None]*(tri[:,1]-tri[:,0])+uv[:,1,None]*(tri[:,2]-tri[:,0]);nrm=norm[chosen];dr=rng.normal(size=nrm.shape);dr-=np.sum(nrm*dr,axis=1)[:,None]*nrm;dr/=np.maximum(np.linalg.norm(dr,axis=1)[:,None],1e-7);ln=rng.uniform(.65,2.6,(len(root),1));tip=root+ln*(nrm*.62+dr*.82);middle=root+ln*(nrm*.53+dr*.34)
    paths=np.stack((root-.07*nrm,middle,tip),axis=1);thick=rng.uniform(.020,.044,(len(root),1))*np.array([1,.76,.10]);strands('14 • Sweater | loose wool fibres',paths,thick,fuzz,C)
    return sweater
