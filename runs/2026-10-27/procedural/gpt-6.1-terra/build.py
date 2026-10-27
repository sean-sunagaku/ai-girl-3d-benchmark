"""Editable upper-body portrait. Run with Blender 5.1.1, --threads 3.
All appearance and fabric imagery is procedural; the reference photograph is not loaded.
"""
import bpy, math, random, os, json
from mathutils import Vector, Matrix
from math import sin, cos, pi, sqrt, exp
import numpy as np

ROOT=os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
random.seed(48271)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
for c in list(bpy.data.collections):
    if c.name != 'Collection': bpy.data.collections.remove(c)
basecol=bpy.data.collections.get('Collection'); basecol.name='Character'
scene=bpy.context.scene

def collection(name):
    c=bpy.data.collections.new(name); scene.collection.children.link(c); return c
setcol=collection('Cafe suggestion'); lightcol=collection('Lights'); camcol=collection('Cameras')
def link(obj,col=basecol): col.objects.link(obj); return obj
def relocate(obj,col):
    for c in list(obj.users_collection): c.objects.unlink(obj)
    col.objects.link(obj)
def mat(name,color,rough=.5,metal=0.0):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF'); p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough; p.inputs['Metallic'].default_value=metal
    return m,p

def smooth(obj):
    if obj.type=='MESH':
        for p in obj.data.polygons:p.use_smooth=True

def mesh(name,verts,faces,material=None,uvs=None,col=basecol):
    me=bpy.data.meshes.new(name+' mesh'); me.from_pydata(verts,[],faces); me.update()
    ob=link(bpy.data.objects.new(name,me),col)
    if material: ob.data.materials.append(material)
    if uvs:
        layer=me.uv_layers.new(name='Knitted surface UV')
        for poly in me.polygons:
            for li in poly.loop_indices: layer.data[li].uv=uvs[li]
    smooth(ob); return ob

def curve_paths(name,paths,radius,material,col=basecol):
    cu=bpy.data.curves.new(name+' curves','CURVE'); cu.dimensions='3D'; cu.resolution_u=1
    cu.bevel_depth=radius; cu.bevel_resolution=1; cu.resolution_u=2
    for path in paths:
        sp=cu.splines.new('POLY'); sp.points.add(len(path)-1)
        for i,(co,thick) in enumerate(path): sp.points[i].co=(*co,1); sp.points[i].radius=thick
    ob=link(bpy.data.objects.new(name,cu),col); ob.data.materials.append(material); return ob

def sphere(name,loc,scale,material,segments=48,rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,location=loc)
    ob=bpy.context.object; ob.name=name; ob.scale=scale; relocate(ob,basecol); smooth(ob); ob.data.materials.append(material); return ob

def track(ob,at): ob.rotation_euler=(Vector(at)-ob.location).to_track_quat('-Z','Y').to_euler()
def area(name,loc,at,energy,size,color=(1,1,1)):
    data=bpy.data.lights.new(name,'AREA'); data.energy=energy; data.shape='DISK'; data.size=size; data.color=color
    ob=link(bpy.data.objects.new(name,data),lightcol); ob.location=loc; track(ob,at); return ob

def camera(name,loc,target,lens,ortho=None):
    data=bpy.data.cameras.new(name); ob=link(bpy.data.objects.new(name,data),camcol); ob.location=loc; track(ob,target)
    if ortho: data.type='ORTHO'; data.ortho_scale=ortho
    else: data.lens=lens
    return ob

# Skin: hand-shaped blush and lip colour on the actual continuous mesh; subtle pore relief.
skin,p=mat('Skin | warm porcelain, soft subsurface',(0.67,.36,.29),.48)
p.inputs['Subsurface Weight'].default_value=.12
p.inputs['Subsurface Radius'].default_value=(1.0,.48,.28)
p.inputs['Subsurface Scale'].default_value=.012
p.inputs['Specular IOR Level'].default_value=.32
n=skin.node_tree.nodes; l=skin.node_tree.links
attr=n.new('ShaderNodeVertexColor'); attr.layer_name='complexion'; attr.label='Painted cheek and lip colour'
l.new(attr.outputs['Color'],p.inputs['Base Color'])
noise=n.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value=1550; noise.inputs['Detail'].default_value=2; noise.inputs['Roughness'].default_value=.7
coord=n.new('ShaderNodeTexCoord'); l.new(coord.outputs['Object'],noise.inputs['Vector'])
bump=n.new('ShaderNodeBump'); bump.inputs['Strength'].default_value=.15; bump.inputs['Distance'].default_value=.00008
l.new(noise.outputs['Fac'],bump.inputs['Height']); l.new(bump.outputs['Normal'],p.inputs['Normal'])
white,p=mat('Eyeball | warm sclera',(0.65,.60,.56),.30); p.inputs['Subsurface Weight'].default_value=.04
iris_mats=[]
for i,c in enumerate([(0.065,.022,.009),(.12,.041,.013),(.19,.075,.024),(.095,.029,.010),(.23,.100,.031),(.045,.015,.006)]):
    m,p=mat('Iris | radial brown %02d'%i,c,.26); p.inputs['Coat Weight'].default_value=.28; iris_mats.append(m)
pupil,p=mat('Pupil | deep brown black',(.008,.005,.004),.16); p.inputs['Coat Weight'].default_value=.3
cornea=bpy.data.materials.new('Cornea | clear reflective shell'); cornea.use_nodes=True
nodes=cornea.node_tree.nodes; links=cornea.node_tree.links; nodes.clear()
out=nodes.new('ShaderNodeOutputMaterial'); transparent=nodes.new('ShaderNodeBsdfTransparent')
gloss=nodes.new('ShaderNodeBsdfGlossy'); gloss.inputs['Color'].default_value=(.96,.96,.96,1); gloss.inputs['Roughness'].default_value=.12
mix=nodes.new('ShaderNodeMixShader'); mix.inputs[0].default_value=.025
links.new(transparent.outputs[0],mix.inputs[1]); links.new(gloss.outputs[0],mix.inputs[2]); links.new(mix.outputs[0],out.inputs['Surface'])
lashmat,_=mat('Brows and lashes | dark chestnut',(.065,.026,.015),.58)
lidmat,_=mat('Lower lid | warm skin',(.48,.26,.22),.5)
crease,_=mat('Nostril and lip crease | muted rose brown',(.25,.08,.063),.55)
hairmats=[]
for i,c in enumerate([(.043,.014,.008),(.072,.026,.014),(.100,.038,.019),(.14,.055,.027),(.17,.071,.034)]):
    m,p=mat('Hair | chestnut %02d'%i,c,.34); p.inputs['Coat Weight'].default_value=.18; p.inputs['Coat Roughness'].default_value=.36
    hairmats.append(m)
capmat,p=mat('Hair | underlying chestnut volume',(.046,.016,.009),.40)
p.inputs['Coat Weight'].default_value=.2
nn=capmat.node_tree.nodes; ll=capmat.node_tree.links
tex=nn.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value=110; tex.inputs['Detail'].default_value=3
tex.inputs['Roughness'].default_value=.7
b=nn.new('ShaderNodeBump'); b.inputs['Strength'].default_value=.12; b.inputs['Distance'].default_value=.0002
ll.new(tex.outputs['Fac'],b.inputs['Height']); ll.new(b.outputs['Normal'],p.inputs['Normal'])

# A repeatable 512px original knit swatch: overlapping bent yarn ridges, used only as fabric.
N=512
Y,X=np.mgrid[0:N,0:N].astype(float)
u=(X/N*8)%1; v=(Y/N*8)%1
# Two bent sides of each tiny stitch, with a row offset like knitted loops.
shift=np.where((np.floor(Y/N*8).astype(int)%2)==0,0.0,.5)
u=(X/N*8+shift)%1
x=u-.5; yy=v-.5
w=.11+.20*(v**.72)
d=np.minimum(np.abs(x-w),np.abs(x+w))
yarn=np.exp(-((d/.075)**2))*np.exp(-((yy/.59)**8))
# Small fibre grain and thread roundness.
gr=(np.sin(X*1.37+Y*.83)+np.sin(X*.77-Y*1.63))*.018
h=np.clip(.30+.47*yarn+gr,0,1)
rgba=np.empty((N,N,4),np.float32); rgba[:,:,0]=h; rgba[:,:,1]=h; rgba[:,:,2]=h; rgba[:,:,3]=1
image=bpy.data.images.new('Generated knit relief 512',width=N,height=N,alpha=True)
image.pixels.foreach_set(rgba.ravel()); image.pack()
sweater,p=mat('Sweater | rose wool knit',(0.58,.29,.30),.73)
p.inputs['Sheen Weight'].default_value=.32; p.inputs['Sheen Roughness'].default_value=.65
nodes=sweater.node_tree.nodes; links=sweater.node_tree.links
uv=nodes.new('ShaderNodeUVMap'); uv.uv_map='Knitted surface UV'
it=nodes.new('ShaderNodeTexImage'); it.image=image; it.extension='REPEAT'; it.interpolation='Linear'; links.new(uv.outputs['UV'],it.inputs['Vector'])
ramp=nodes.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].position=.20; ramp.color_ramp.elements[0].color=(.43,.19,.21,1)
ramp.color_ramp.elements[1].position=.8; ramp.color_ramp.elements[1].color=(.59,.30,.31,1)
links.new(it.outputs['Color'],ramp.inputs[0]); links.new(ramp.outputs['Color'],p.inputs['Base Color'])
b=nodes.new('ShaderNodeBump'); b.inputs['Strength'].default_value=.23; b.inputs['Distance'].default_value=.0003
links.new(it.outputs['Color'],b.inputs['Height']); links.new(b.outputs['Normal'],p.inputs['Normal'])

# Import the supplied CC0 face, retain the neck, and soften / articulate its features.
bpy.ops.wm.obj_import(filepath=os.path.join(ROOT,'input/upper-body.obj'),forward_axis='NEGATIVE_Y',up_axis='Z')
original=bpy.context.selected_objects[0]
verts=[tuple(v.co) for v in original.data.vertices]
faces=[]
for f in original.data.polygons:
    ids=list(f.vertices); vv=[verts[i] for i in ids]
    if all(co[2]>1.423 for co in vv): faces.append(ids)
bpy.data.objects.remove(original,do_unlink=True)
used=sorted({i for face in faces for i in face}); remap={v:i for i,v in enumerate(used)}
newverts=[]
for idx in used:
    x,y,z=verts[idx]
    # Eye openings are enlarged modestly on the continuous source surface.
    if y < -.103:
        cx=.0312 if x>=0 else -.0312
        dx=x-cx; dz=z-1.5452
        g=exp(-((dx/.023)**2+(dz/.020)**2)*1.25)
        x+=dx*.12*g; z+=dz*.23*g
        # Lift the corners and soften the mouth, leaving its topology intact.
        g=exp(-(((abs(x)-.021)/.012)**2+((z-1.482)/.012)**2))
        z+=.005*g
        # Slightly narrow the lower jaw.
        f=exp(-((z-1.475)/.050)**2); x*=1+.045*f
    # A small exposed ear on the camera-right side, partly tucked into the hair.
    if x>.068 and y>-.095:
        ear=exp(-(((x-.086)/.015)**2+((z-1.535)/.033)**2+((y+.046)/.055)**2))
        x+=.007*ear
    newverts.append((x,y,z))
face=mesh('Face and neck | sculpted CC0 topology',newverts,[[remap[i] for i in f] for f in faces],skin)
colors=face.data.color_attributes.new(name='complexion',type='FLOAT_COLOR',domain='POINT')
for i,v in enumerate(face.data.vertices):
    x,y,z=v.co; rgb=np.array([.67,.395,.315])
    front=max(0,min(1,(-y-.08)/.05))
    cheeks=exp(-(((abs(x)-.046)/.022)**2+((z-1.517)/.018)**2))*front
    nose=exp(-((x/.014)**2+((z-1.515)/.013)**2))*front
    lips=exp(-((x/.025)**4+((z-1.484)/.013)**4))*front
    tint=min(.39,cheeks*.25+nose*.10); rgb=rgb*(1-tint)+np.array([.80,.275,.26])*tint
    rgb=rgb*(1-lips*.76)+np.array([.55,.15,.145])*lips*.76
    colors.data[i].color=(*rgb,1)
sub=face.modifiers.new('Sculpt surface smoothing','SUBSURF'); sub.levels=2; sub.render_levels=2

# All head components share a single non-destructive tilted parent.
head=link(bpy.data.objects.new('Head tilt | editable pivot',None)); head.location=(0,-.045,1.375)
bpy.context.view_layer.update(); head_inverse=head.matrix_world.inverted().copy()
def head_parent(ob): ob.parent=head; ob.matrix_parent_inverse=head_inverse
head_parent(face)

# A fitted continuous neck replaces the cropped-off shoulders of the source mesh.
neck_profiles=[(1.363,.047,.038,-.028),(1.382,.046,.039,-.028),
               (1.401,.044,.040,-.027),(1.423,.043,.041,-.027),
               (1.442,.043,.043,-.029),(1.451,.043,.045,-.030)]
nv=[]; nf=[]; ns=64
for z,rx,ry,cy in neck_profiles:
    for i in range(ns):
        a=2*pi*i/ns; nv.append((rx*cos(a),cy+ry*sin(a),z))
for k in range(len(neck_profiles)-1):
    for i in range(ns): nf.append((k*ns+i,k*ns+(i+1)%ns,(k+1)*ns+(i+1)%ns,(k+1)*ns+i))
neck=mesh('Neck | tapered anatomical bridge',nv,nf,skin)
nc=neck.data.color_attributes.new(name='complexion',type='FLOAT_COLOR',domain='POINT')
for c in nc.data: c.color=(.67,.395,.315,1)
sub=neck.modifiers.new('Neck smoothing','SUBSURF'); sub.levels=2
head_parent(neck)

# Eyeballs and three-dimensional iris caps, integrated under the eyelid openings.
def iris_geometry(side):
    cx=side*.0312; cy=-.1257; cz=1.5452; R=.0151
    eye=sphere(('Right' if side<0 else 'Left')+' eye | sclera',(cx,cy,cz),(R,R,R),white); head_parent(eye)
    segments=96; radii=[0,.00245,.0031,.0044,.0056,.00625]; vv=[]; ff=[]; mid=[]
    for ri,r in enumerate(radii):
        for j in range(segments):
            a=2*pi*j/segments; xx=cx+r*cos(a); zz=cz+r*sin(a); yy=cy-sqrt(max(0,R*R-r*r))-.00018
            vv.append((xx,yy,zz))
    for k in range(len(radii)-1):
        for j in range(segments):
            ff.append((k*segments+j,k*segments+(j+1)%segments,(k+1)*segments+(j+1)%segments,(k+1)*segments+j))
            mid.append(6 if k==0 else (5 if k==len(radii)-2 else random.randrange(5)))
    ob=mesh(('Right' if side<0 else 'Left')+' eye | iris and pupil',vv,ff,None)
    for m in iris_mats+[pupil]:ob.data.materials.append(m)
    for f,i in zip(ob.data.polygons,mid):f.material_index=i
    head_parent(ob)
    # A genuinely curved transparent/glossy corneal surface over each iris.
    vv=[]; ff=[]; rings=6; seg=64; rmax=.007
    for k in range(rings+1):
        r=rmax*k/rings
        for j in range(seg):
            a=2*pi*j/seg; y=cy-sqrt(max(0,R*R-r*r))-.00025-.00045*(1-(r/rmax)**2)
            vv.append((cx+r*cos(a),y,cz+r*sin(a)))
    for k in range(rings):
        for j in range(seg): ff.append((k*seg+j,k*seg+(j+1)%seg,(k+1)*seg+(j+1)%seg,(k+1)*seg+j))
    ob=mesh(('R' if side<0 else 'L')+' eye | clear corneal shell',vv,ff,cornea); head_parent(ob)
    # Narrow strips follow the curvature of the sphere and blend back into the socket.
    for top in [True,False]:
        vv=[]; ff=[]; steps=40
        for row in range(4):
            q=row/3
            for j in range(steps+1):
                t=-1+2*j/steps; dx=.0141*t; arc=sqrt(max(0,1-t*t))
                dz=(1 if top else -1)*(.0048+.0028*q)*arc
                y=cy-sqrt(max(.000001,R*R-dx*dx-min(dz*dz,.000035)))+.0011*q
                vv.append((cx+dx,y,cz+dz))
        for row in range(3):
            for j in range(steps):
                k=row*(steps+1)+j; ff.append((k,k+1,k+steps+2,k+steps+1))
        ob=mesh(('R' if side<0 else 'L')+(' upper' if top else ' lower')+' | sculpted lid',vv,ff,skin)
        col=ob.data.color_attributes.new(name='complexion',type='FLOAT_COLOR',domain='POINT')
        for c in col.data: c.color=(.67,.395,.315,1)
        sub=ob.modifiers.new('Lid smoothing','SUBSURF'); sub.levels=1
        head_parent(ob)
    # The upper lid detail and individual lashes follow the eyeball, not planes.
    upper=[]; lower=[]; lashes=[]
    for j in range(35):
        t=-1+2*j/34; dx=.0142*t; lift=sqrt(max(0,1-t*t))
        for sign,collection0 in [(1,upper),(-1,lower)]:
            dz=(.0048 if sign==1 else -.0048)*lift
            y=cy-sqrt(max(.00001,R*R-dx*dx-dz*dz))-.0003
            collection0.append(((cx+dx,y,cz+dz),.72+lift*.28))
    ob=curve_paths(('R' if side<0 else 'L')+' | upper lid edge',[upper],.00021,lashmat); head_parent(ob)
    ob=curve_paths(('R' if side<0 else 'L')+' | lower lid edge',[lower],.00025,lidmat); head_parent(ob)
    for j in range(29):
        t=-.85+1.7*j/28; dx=.0142*t; lift=sqrt(max(0,1-t*t)); dz=.0048*lift
        yy=cy-sqrt(max(.00001,R*R-dx*dx-dz*dz))-.0004
        length=.002+.0025*(.5+.5*abs(t))+random.random()*.0012
        path=[((cx+dx,yy,cz+dz),.8),((cx+dx+t*.0012,yy-.0006,cz+dz+length*.6),.45),((cx+dx+t*.002,yy-.0001,cz+dz+length),.04)]
        lashes.append(path)
    ob=curve_paths(('R' if side<0 else 'L')+' | upper lashes',lashes,.00020,lashmat); head_parent(ob)
for s in [-1,1]: iris_geometry(s)

# Subtle inset nostril edges and the soft, closed-mouth crease are actual curves.
nostrils=[]
for side in [-1,1]:
    path=[]
    for j in range(9):
        t=j/8; x=side*(.004+.008*t); z=1.504-.0015*sin(pi*t)
        y=-.162+.009*t
        path.append(((x,y,z),.15+.55*sin(pi*t)))
    nostrils.append(path)
ob=curve_paths('Nose | nostril creases',nostrils,.00028,crease); head_parent(ob)
path=[]
for j in range(33):
    t=-1+2*j/32; x=.024*t; z=1.484+.0032*abs(t)**1.5; y=-.160+.010*abs(t)
    path.append(((x,y,z),.18+.57*(1-abs(t))))
ob=curve_paths('Mouth | gentle closed smile crease',[path],.0003,crease); head_parent(ob)

# Fine, individual brow hairs resting on the face.
for side in [-1,1]:
    paths=[]
    for j in range(145):
        t=random.random(); x=side*(.017+.036*t); z=1.568+.0045*sin(pi*t)-.001*t+random.gauss(0,.0011)
        y=-.151+.30*(abs(x)-.017)
        paths.append([((x,y,z),.5),((x+side*.0018,y-.0006,z+.0012),.75),((x+side*.0033,y,z+.0017),.08)])
    ob=curve_paths(('R' if side<0 else 'L')+' | eyebrow hairs',paths,.00017,lashmat); head_parent(ob)

# Scalp cap and a continuous underlayer for long hair, then thousands of editable strands.
def phi_end(theta):
    c=cos(theta)
    # A high central hairline curves down at the temples instead of a straight fringe.
    return .86+.84*(1-c)**.65 if c>0 else 1.70+.32*(-c)
def cap(theta,phi,extra=0):
    c=cos(theta)
    # The skull is fuller at the back than an ellipse centred at the part.
    depth=.100+.028*max(0,-c)
    tuck=.024*exp(-(((theta-1.70)/.39)**2+((phi-1.65)/.36)**2))
    return ((.088+extra-tuck)*sin(phi)*sin(theta), -.060-(depth+extra)*sin(phi)*c,1.551+(.133+extra)*cos(phi))
nt=100; nr=24; v=[]; f=[]
for j in range(nr+1):
    for i in range(nt):
        a=2*pi*i/nt; v.append(cap(a,.01+(phi_end(a)-.01)*j/nr,.002))
for j in range(nr):
    for i in range(nt): f.append((j*nt+i,j*nt+(i+1)%nt,(j+1)*nt+(i+1)%nt,(j+1)*nt+i))
ob=mesh('Hair | scalp cap',v,f,capmat); head_parent(ob)
sub=ob.modifiers.new('Scalp softness','SUBSURF'); sub.levels=1

def strand_tail(theta,t,length,phase,off=0):
    start=cap(theta,phi_end(theta),.003+off)
    si=sin(theta); co=cos(theta)
    # Locks fall over the left-front shoulder and behind the other shoulder.
    endz=length
    x0=si*(.091+.052*t) + .017*sin(8*t+phase)*sin(pi*t)
    y0=-.060-co*(.104+.043*t) + .009*sin(7*t+phase+1)*sin(pi*t)
    x=(1-t)*start[0]+t*x0; y=(1-t)*start[1]+t*y0
    z=(1-t)*start[2]+t*endz
    # Follow the elliptical sweater's back surface after the hair reaches it.
    # Account for the head tilt before looking up the body cross-section.
    tilt=math.radians(-10); dz=z-1.375
    xworld=cos(tilt)*x+sin(tilt)*dz; zworld=1.375-sin(tilt)*x+cos(tilt)*dz
    body=[(.965,.195,.119,-.005),(1.00,.207,.128,-.007),(1.07,.218,.133,-.010),
          (1.14,.226,.137,-.012),(1.22,.232,.137,-.012),(1.28,.232,.130,-.009),
          (1.315,.218,.116,-.008),(1.34,.184,.098,-.009),(1.355,.139,.081,-.013),
          (1.369,.079,.057,-.025)]
    if zworld<body[-1][0]:
        for k in range(len(body)-1):
            if body[k][0]<=zworld<=body[k+1][0]:
                b0,b1=body[k],body[k+1]; q=(zworld-b0[0])/(b1[0]-b0[0])
                rx=b0[1]*(1-q)+b1[1]*q; ry=b0[2]*(1-q)+b1[2]*q; cy=b0[3]*(1-q)+b1[3]*q
                if abs(xworld)<rx:
                    outside=cy+ry*sqrt(max(0,1-(xworld/rx)**2))+.012
                    fade=max(0,min(1,(1.369-zworld)/.006))*max(0,min(1,-co*5))
                    y+=max(0,outside-y)*fade
                break
    return (x,y,z)
# Underlayer follows the same geometry as the strands and is open only at the face.
na=90; nrow=28; vv=[]; ff=[]
for j in range(nrow+1):
 t=j/nrow
 for i in range(na+1):
    a=.88+(2*pi-1.76)*i/na
    length=1.13+.035*cos(a)+.035*sin(a)+.005*sin(13*a)+.003*sin(29*a)
    vv.append(strand_tail(a,t,length,1.0,0))
for j in range(nrow):
 for i in range(na): ff.append((j*(na+1)+i,j*(na+1)+i+1,(j+1)*(na+1)+i+1,(j+1)*(na+1)+i))
ob=mesh('Hair | flowing back and shoulder underlayer',vv,ff,capmat); head_parent(ob)
sub=ob.modifiers.new('Long hair flow smoothing','SUBSURF'); sub.levels=1
strands=[[] for _ in hairmats]
for i in range(1700):
    a=random.uniform(.88,2*pi-.88); phase=random.uniform(0,2*pi); surf=random.uniform(.001,.007)
    startphi=random.uniform(.07,.55); stop=phi_end(a)
    path=[]
    for j in range(9):
        t=j/8; angle=a + .09*sin(pi*t+phase)*sin(pi*t)
        pos=cap(angle,startphi+(stop-startphi)*t,surf)
        path.append((pos,.5+.5*min(1,t*5)))
    length=1.095+.035*cos(a)+.035*sin(a)+random.uniform(-.04,.045)
    for j in range(1,16):
        t=j/15; p0=strand_tail(a,t,length,phase,surf)
        # Tiny uncorrelated offset, not a random tangle.
        p0=(p0[0]+surf*sin(a),p0[1]-surf*cos(a),p0[2])
        path.append((p0,max(.08,1-t**3*.92)))
    strands[random.choices(range(len(hairmats)),weights=[25,33,25,13,4])[0]].append(path)
for k,paths in enumerate(strands):
    ob=curve_paths('Hair | long strands %02d'%k,paths,.00022,hairmats[k]); head_parent(ob)
# Fine combed strands on the front scalp soften the transition into the fringe.
paths=[[] for _ in range(3)]
for i in range(800):
    a=random.uniform(-1.05,1.05); start=random.uniform(.10,.42); end=phi_end(a)
    phase=random.uniform(0,2*pi); extra=random.uniform(.0025,.004)
    path=[]
    for j in range(14):
        t=j/13; angle=a+.10*sin(pi*t)*sin(phase)
        path.append((cap(angle,start+(end-start)*t,extra),max(.16,1-t*.70)))
    paths[random.randrange(3)].append(path)
for i,pp in enumerate(paths):
    ob=curve_paths('Hair | front scalp fibres %02d'%i,pp,.00015,hairmats[i+1]); head_parent(ob)
# Separated wispy fringe strands sweep off the part across the forehead.
paths=[[] for _ in hairmats]
for i in range(580):
    startx=random.uniform(-.004,.025); endx=random.uniform(-.055,.046)
    zstart=random.uniform(1.645,1.678); zend=random.uniform(1.575,1.627)
    phase=random.uniform(0,2*pi); path=[]
    for j in range(15):
        t=j/14; x=startx+(endx-startx)*t + .002*sin(pi*t)*sin(phase)
        z=zstart+(zend-zstart)*t
        y=-.105-.052*sin(t*pi*.5)+.003*(x/.05)**2-random.random()*.00035
        path.append(((x,y,z),max(.05,1-t*.9)))
    paths[random.choice([1,1,2,2,3])].append(path)
for i,pp in enumerate(paths):
    if pp:
        ob=curve_paths('Hair | wispy fringe %02d'%i,pp,.00022,hairmats[i]); head_parent(ob)

# Asymmetrical hair fan sweeps from the left temple over the sweater. Its
# contiguous under-surface avoids gaps; thin paths give individually editable fibres.
def front_lock(u,t,offset=0):
    phase=.3+u*1.5
    x=-.081-.022*u-.075*u*t + .021*sin(t*6.3+phase)*sin(pi*t)
    y=-.108-.047*t-.004*cos(t*6+phase)*sin(pi*t)-offset
    endz=1.085+.018*sin(u*8)
    z=(1-t)*(1.57-.04*u)+t*endz
    return (x,y,z)
v=[]; f=[]; columns=40; rows=24
for j in range(rows+1):
    for i in range(columns+1):v.append(front_lock(i/columns,j/rows))
for j in range(rows):
    for i in range(columns):
        k=j*(columns+1)+i; f.append((k,k+1,k+columns+2,k+columns+1))
ob=mesh('Hair | left swept volume',v,f,capmat); head_parent(ob)
paths=[[] for _ in range(3)]
for i in range(850):
    u=random.random(); off=random.uniform(.00025,.002)
    paths[random.randrange(3)].append([(front_lock(u,j/24,off),max(.06,1-(j/24)**4*.94)) for j in range(25)])
for i,pp in enumerate(paths):
    ob=curve_paths('Hair | swept fibres %02d'%i,pp,.00016,hairmats[i+1]); head_parent(ob)
head.rotation_euler[1]=math.radians(-10)

# Garment torso is a smooth continuous ring loft; sleeves have individual round cloth lofts.
def ring_loft(name,profiles,mat):
    ns=96; vs=[]; fs=[]; uvs=[]
    for z,rx,ry,cy in profiles:
        for j in range(ns):
            a=2*pi*j/ns; fold=.0018*sin(7*a+z*32)+.0011*sin(15*a-z*18)
            vs.append(((rx+fold)*cos(a),cy+(ry+fold)*sin(a),z+.0015*sin(5*a+z*30)))
    for i in range(len(profiles)-1):
        for j in range(ns):
            fs.append((i*ns+j,i*ns+(j+1)%ns,(i+1)*ns+(j+1)%ns,(i+1)*ns+j))
            u0=j/ns*10; u1=(j+1)/ns*10; v0=i/(len(profiles)-1)*9; v1=(i+1)/(len(profiles)-1)*9
            uvs.extend([(u0,v0),(u1,v0),(u1,v1),(u0,v1)])
    ob=mesh(name,vs,fs,mat,uvs); sub=ob.modifiers.new('Tailored smoothing','SUBSURF'); sub.levels=2; return ob
profiles=[(.965,.195,.119,-.005),(1.00,.207,.128,-.007),(1.07,.218,.133,-.010),(1.14,.226,.137,-.012),(1.22,.232,.137,-.012),(1.28,.232,.130,-.009),(1.315,.218,.116,-.008),(1.34,.184,.098,-.009),(1.355,.139,.081,-.013),(1.369,.079,.057,-.025),(1.374,.061,.048,-.032)]
torso=ring_loft('Sweater | shaped body',profiles,sweater)
# Ribbed crew neck: oval band with narrow vertical ridges.
def collar():
    n=144; vs=[]; fs=[]; uvs=[]
    for k in range(5):
        z=1.369+k*.0035; rx=.061-(k/4)*.003; ry=.048-(k/4)*.003
        for j in range(n):
            a=2*pi*j/n; rib=.00065*cos(56*a); vs.append(((rx+rib)*cos(a),-.032+(ry+rib)*sin(a),z))
    for k in range(4):
        for j in range(n):
            fs.append((k*n+j,k*n+(j+1)%n,(k+1)*n+(j+1)%n,(k+1)*n+j))
            uvs.extend([(j/n*3,k/4),((j+1)/n*3,k/4),((j+1)/n*3,(k+1)/4),(j/n*3,(k+1)/4)])
    return mesh('Sweater | ribbed crew neck',vs,fs,sweater,uvs)
collar()

def sleeve(side):
    ns=72; nr=30; vs=[]; fs=[]; uvs=[]
    p0=Vector((side*.145,-.006,1.275)); p1=Vector((side*.275,-.235,1.16)); p2=Vector((side*.29,-.50,1.055))
    for i in range(nr+1):
        t=i/nr; p=(1-t)**2*p0+2*(1-t)*t*p1+t*t*p2
        tangent=((1-t)*(p1-p0)+t*(p2-p1)).normalized(); axis1=Vector((1,0,0)); axis1=(axis1-tangent*tangent.dot(axis1)).normalized(); axis2=tangent.cross(axis1).normalized()
        radius=.076+.007*sin(pi*t)-.009*t + .0012*sin(t*7*pi)
        if t>.88: radius*=1-.10*(t-.88)/.12
        for j in range(ns):
            a=2*pi*j/ns; fold=.002*sin(7*a+t*14)+.0015*sin(13*a-t*22)
            co=p+(radius+fold)*(cos(a)*axis1+sin(a)*axis2); vs.append(tuple(co))
    for i in range(nr):
        for j in range(ns):
            fs.append((i*ns+j,i*ns+(j+1)%ns,(i+1)*ns+(j+1)%ns,(i+1)*ns+j))
            uvs.extend([(j/ns*6,i/nr*10),((j+1)/ns*6,i/nr*10),((j+1)/ns*6,(i+1)/nr*10),(j/ns*6,(i+1)/nr*10)])
    # Cloth closes both ends, with the shoulder end hidden inside the body.
    rootidx=len(vs); vs.append(tuple(p0))
    for j in range(ns): fs.append(((j+1)%ns,j,rootidx)); uvs.extend([((j+1)/ns*3,0),(j/ns*3,0),(1.5,-.5)])
    cidx=len(vs); vs.append(tuple(p2+Vector((0,-.001,0))))
    for j in range(ns): fs.append((nr*ns+j,nr*ns+(j+1)%ns,cidx)); uvs.extend([(j/ns*3,5),((j+1)/ns*3,5),(1.5,5.5)])
    ob=mesh(('Right' if side<0 else 'Left')+' sleeve | reaching forward',vs,fs,sweater,uvs)
    sub=ob.modifiers.new('Soft fabric smoothing','SUBSURF'); sub.levels=2
    return ob
sleeve(-1); sleeve(1)

# Simple out-of-focus cafe geometry, completely original and photographic-free.
wall,_=mat('Cafe | warm plaster',(.52,.43,.33),.85)
wood,_=mat('Cafe | honey wood',(.31,.15,.063),.43)
darkwood,_=mat('Cafe | shelf shadow',(.12,.064,.031),.55)
glass,_=mat('Cafe | pale ceramic',(.67,.64,.54),.34)
leaf,_=mat('Cafe | foliage',(.11,.21,.07),.7)
def cube(name,loc,scale,material):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc); ob=bpy.context.object; ob.name=name; ob.scale=scale; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); relocate(ob,setcol); ob.data.materials.append(material); return ob
cube('Cafe wall',(0,1.5,1.5),(5,.08,3.2),wall)
for z in [1.24,1.62,1.99]:
    cube('Cafe shelf',(.48,1.25,z),(2.0,.30,.035),wood)
for i in range(14):
    x=.05+(i%7)*.15; z=1.28+(i//7)*.38
    cube('Cafe jars %02d'%i,(x,1.23,z),(.055,.06,.08+(i%3)*.015),glass if i%3 else darkwood)
cube('Cafe table',(.62,.72,1.05),(1.25,.65,.055),wood)
# A few soft leafy silhouettes, physically behind the subject.
for i in range(34):
    random.seed(i+920); x=-.9+random.uniform(-.28,.24); y=.95+random.uniform(-.1,.2); z=1.30+random.uniform(0,.85)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=8,location=(x,y,z)); o=bpy.context.object; o.name='Cafe greenery'; o.scale=(.055,.025,.11); o.rotation_euler[1]=random.uniform(-1.2,1.2); relocate(o,setcol); o.data.materials.append(leaf); smooth(o)

# Portrait light is broad, warm and natural.
scene.world.color=(.10,.10,.10)
area('Key | large window',(-.55,-.72,1.98),(0,-.05,1.47),13,1.0,(1.0,.91,.84))
area('Fill | camera side',(.55,-.53,1.60),(0,-.06,1.50),4,.85,(1.0,.90,.85))
area('Hair rim',(.25,.55,1.85),(0,0,1.48),11,.7,(1.0,.78,.60))
area('Cafe | reflected window',(-.2,.48,2.3),(0,1.5,1.5),30,1.5,(1.0,.87,.72))
beauty=camera('Beauty camera',(0,-1.12,1.58),(0,-.065,1.41),62)
beauty.data.dof.use_dof=True; beauty.data.dof.focus_distance=(Vector((0,-.15,1.53))-beauty.location).length; beauty.data.dof.aperture_fstop=2.8
scene.camera=beauty
scene.render.engine='CYCLES'; scene.cycles.device='CPU'; scene.cycles.samples=32; scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED'; scene.render.threads=3
scene.render.resolution_x=960; scene.render.resolution_y=1040; scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGBA'; scene.render.film_transparent=False
scene.view_settings.view_transform='AgX'
scene.render.filepath='//renders/beauty.png'
# Save the editable asset BEFORE any render.
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT,'character.blend'))
print('BUILD COMPLETE: saved character.blend')
