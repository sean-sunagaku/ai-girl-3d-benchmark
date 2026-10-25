import bpy, math, random, os, sys, json
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from collections import Counter
ROOT=os.path.abspath(os.path.dirname(__file__))
os.chdir(ROOT)
random.seed(27)
PREVIEW='--preview' in sys.argv
os.makedirs('renders',exist_ok=True); os.makedirs('previews',exist_ok=True)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
for d in list(bpy.data.collections):
 if d.name!='Collection' and d.users==0: bpy.data.collections.remove(d)
def collection(name):
 c=bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c); return c
SKIN=collection('01 • continuous skin and hands'); EYES=collection('02 • eyes lids brows lashes'); HAIR=collection('03 • chestnut scalp locks and strands'); CLOTH=collection('04 • dusty pink knitted sweater'); SET=collection('05 • cafe geometry'); LIGHTS=collection('06 • soft photographic lighting'); CAMS=collection('07 • cameras')
def move(o,c):
 for x in list(o.users_collection): x.objects.unlink(o)
 c.objects.link(o)
def rgb(h):
 h=h.lstrip('#'); a=[int(h[i:i+2],16)/255 for i in (0,2,4)]; return tuple(v/12.92 if v<.04045 else ((v+.055)/1.055)**2.4 for v in a)+(1,)
def material(name,color,rough=.5):
 m=bpy.data.materials.new(name); m.use_nodes=True; b=m.node_tree.nodes.get('Principled BSDF'); b.inputs['Base Color'].default_value=rgb(color) if isinstance(color,str) else color; b.inputs['Roughness'].default_value=rough; m.diffuse_color=b.inputs['Base Color'].default_value; return m,b
def node(m,kind,name):
 n=m.node_tree.nodes.new(kind); n.label=name; return n
def mesh_obj(name,verts,faces,mat,col,smooth=True):
 me=bpy.data.meshes.new(name+' • editable mesh'); me.from_pydata(verts,[],faces); me.update(); o=bpy.data.objects.new(name,me); col.objects.link(o)
 if mat: me.materials.append(mat)
 if smooth:
  for p in me.polygons: p.use_smooth=True
 return o
def subdiv(o,n=2):
 m=o.modifiers.new('Smooth continuous surface','SUBSURF'); m.levels=n; m.render_levels=n; return m
def solid(o,t=.002):
 m=o.modifiers.new('Editable thickness','SOLIDIFY'); m.thickness=t; m.offset=-1; return m
def colors(o,vals,name='SkinTint'):
 a=o.data.color_attributes.new(name=name,type='FLOAT_COLOR',domain='POINT')
 for d,c in zip(a.data,vals): d.color=c
def curve_obj(name,paths,radius,mats,col):
 cu=bpy.data.curves.new(name+' • editable splines','CURVE'); cu.dimensions='3D'; cu.resolution_u=2; cu.bevel_depth=radius; cu.bevel_resolution=0; cu.resolution_u=2
 for m in mats: cu.materials.append(m)
 for points in paths:
  s=cu.splines.new('POLY'); s.points.add(len(points)-1)
  for i,(p,co) in enumerate(zip(s.points,points)):
   p.co=(*co[:3],1); p.radius=co[3] if len(co)>3 else 1
  s.material_index=random.randrange(len(mats))
 o=bpy.data.objects.new(name,cu); col.objects.link(o); return o
def smooth(x): x=max(0,min(1,x)); return x*x*(3-2*x)
def gauss(x,s): return math.exp(-(x/s)**2)
def catmull(points,n=48):
 ps=[Vector(p) for p in points]; out=[]
 for i in range(n):
  t=i/(n-1)*(len(ps)-1); j=min(int(t),len(ps)-2); u=t-j; p0=ps[max(0,j-1)]; p1=ps[j]; p2=ps[j+1]; p3=ps[min(len(ps)-1,j+2)]
  out.append(.5*((2*p1)+(-p0+p2)*u+(2*p0-5*p1+4*p2-p3)*u*u+(-p0+3*p1-3*p2+p3)*u*u*u))
 return out
skin,bs=material('Skin • warm fair SSS with vertex blush','#EFC2AE',.46)
bs.inputs['Subsurface Weight'].default_value=.11; bs.inputs['Subsurface Radius'].default_value=(1,.46,.26); bs.inputs['Subsurface Scale'].default_value=.012; bs.inputs['IOR'].default_value=1.43; bs.inputs['Specular IOR Level'].default_value=.28
vc=node(skin,'ShaderNodeVertexColor','Painted anatomical color, no image'); vc.layer_name='SkinTint'; skin.node_tree.links.new(vc.outputs['Color'],bs.inputs['Base Color'])
tex=node(skin,'ShaderNodeTexNoise','Fine pores at submillimeter scale'); tex.inputs['Scale'].default_value=1450; tex.inputs['Detail'].default_value=2.2; bump=node(skin,'ShaderNodeBump','Restrained pore relief'); bump.inputs['Strength'].default_value=.19; bump.inputs['Distance'].default_value=.00013; skin.node_tree.links.new(tex.outputs['Fac'],bump.inputs['Height']); skin.node_tree.links.new(bump.outputs['Normal'],bs.inputs['Normal'])
hair_m=[]
for i,h in enumerate(['#281711','#382117','#482B1E','#5A3827','#6B4632']):
 m,b=material('Chestnut hair • shade %02d'%i,h,.31+(i%2)*.035); b.inputs['Anisotropic'].default_value=.62; b.inputs['Coat Weight'].default_value=.16; b.inputs['Coat Roughness'].default_value=.3; hair_m.append(m)
knit,kb=material('Sweater • dusty rose knitted wool','#C88E91',.83); kb.inputs['Sheen Weight'].default_value=.38; kb.inputs['Sheen Roughness'].default_value=.72
knitcolor=node(knit,'ShaderNodeTexNoise','Soft wool color variation'); knitcolor.inputs['Scale'].default_value=95; knitcolor.inputs['Detail'].default_value=3
ramp=node(knit,'ShaderNodeValToRGB','Dusty pink yarn palette'); ramp.color_ramp.elements[0].position=.15; ramp.color_ramp.elements[0].color=rgb('#B97C82'); ramp.color_ramp.elements[1].position=.85; ramp.color_ramp.elements[1].color=rgb('#D9A3A4'); knit.node_tree.links.new(knitcolor.outputs['Fac'],ramp.inputs[0]); knit.node_tree.links.new(ramp.outputs[0],kb.inputs['Base Color'])
w1=node(knit,'ShaderNodeTexWave','Knit courses'); w1.bands_direction='X'; w1.inputs['Scale'].default_value=115; w1.inputs['Distortion'].default_value=2.1; w1.inputs['Detail Scale'].default_value=1.5
w2=node(knit,'ShaderNodeTexWave','Knit wales'); w2.bands_direction='Z'; w2.inputs['Scale'].default_value=145; w2.inputs['Distortion'].default_value=1.2
mix=node(knit,'ShaderNodeMath','Interlaced yarn'); mix.operation='MULTIPLY'; knit.node_tree.links.new(w1.outputs['Color'],mix.inputs[0]); knit.node_tree.links.new(w2.outputs['Color'],mix.inputs[1]); bump=node(knit,'ShaderNodeBump','Raised knitted loops'); bump.inputs['Strength'].default_value=.48; bump.inputs['Distance'].default_value=.00075; knit.node_tree.links.new(mix.outputs[0],bump.inputs['Height']); knit.node_tree.links.new(bump.outputs[0],kb.inputs['Normal'])
rib,rb=material('Ribbed collar cuffs and hem','#BC8388',.8); rb.inputs['Sheen Weight'].default_value=.32
rw=node(rib,'ShaderNodeTexWave','Rib knit relief'); rw.bands_direction='X'; rw.inputs['Scale'].default_value=170; rbu=node(rib,'ShaderNodeBump','Rib depth'); rbu.inputs['Strength'].default_value=.5; rbu.inputs['Distance'].default_value=.0008; rib.node_tree.links.new(rw.outputs['Color'],rbu.inputs['Height']); rib.node_tree.links.new(rbu.outputs[0],rb.inputs['Normal'])
lashmat,_=material('Brows and lashes • soft dark umber','#35231D',.52)
mouthmat,_=material('Mouth seam • muted rose shadow','#7E4244',.6)
vs=[]; fs=[]; group=''
for line in open('input/base.obj'):
 s=line.split()
 if not s: continue
 if s[0]=='g': group=s[1]
 elif s[0]=='v':
  x,y,z=map(float,s[1:4]); vs.append(Vector((x*.1,-z*.1,(y+8.168)*.1)))
 elif s[0]=='f' and group=='body': fs.append(tuple(int(t.split('/')[0])-1 for t in s[1:]))
keep=[]
for f in fs:
 c=sum((vs[i] for i in f),Vector())/len(f)
 if c.z>=1.096 or (abs(c.x)>.275 and .90<c.z<1.22): keep.append(f)
used=sorted(set(i for f in keep for i in f)); remap={v:i for i,v in enumerate(used)}; orig=[vs[i].copy() for i in used]; faces=[tuple(remap[i] for i in f) for f in keep]
old_dirs={}; new_dirs={}; rotations={}
for s in (-1,1):
 elbow=Vector((s*.312936,-.013201,1.166119)); wrist=Vector((s*.431185,-.175655,1.061981)); old=(wrist-elbow).normalized(); new=Vector((s*.11,-.25,-.035)).normalized(); old_dirs[s]=old; new_dirs[s]=new; rotations[s]=old.rotation_difference(new)
def arm_pose(p):
 s=1 if p.x>0 else -1
 if abs(p.x)<.235 or p.z>1.30: return p.copy()
 w=smooth((1.19-p.z)/.065); e=Vector((s*.312936,-.013201,1.166119)); return p.lerp(e+rotations[s]@(p-e),w)
posed=[]
for p in orig:
 q=p.copy(); x,y,z=q
 if z>1.37:
  front=smooth((-y-.075)/.07)
  if z<1.525: q.x*=1-.095*smooth((1.525-z)/.085)
  q.y-=.0035*gauss(abs(x)-.045,.023)*gauss(z-1.505,.027)*front
  q.y-=.002*gauss(x,.026)*gauss(z-1.449,.019)*front
  q.z+=.0031*gauss(abs(x)-.026,.014)*gauss(z-1.480,.017)*front
  q.x*=1-.045*gauss(z-1.518,.023)*gauss(x,.025)
  for s in (-1,1):
   dx=x-s*.030775; dz=z-1.545215; r=(dx/.021)**2+(dz/.014)**2
   if r<1.8 and y<-.10:
    w=1-r/1.8; q.x=s*.030775+dx*(1+.075*w); q.z=1.545215+dz*(1+.08*w)
   if r<1 and y<-.105: q.y+=.012*(1-r)**2
 posed.append(arm_pose(q))
body=mesh_obj('Woman • continuous sculpted head neck torso and posed hands',posed,faces,skin,SKIN)
body.data.update(); normals=[v.normal.copy() for v in body.data.vertices]
bvh=BVHTree.FromPolygons(posed,faces)
def front_y(x,z):
 p=bvh.ray_cast(Vector((x,-.5,z)),Vector((0,1,0)))[0]; return p.y if p else -.14
def skin_color(p):
 x,y,z=p; base=Vector(rgb('#EFC2AE')[:3]); front=smooth((-y-.07)/.07)
 cheek=gauss(abs(x)-.044,.025)*gauss(z-1.502,.024)*front*.24; base=base.lerp(Vector(rgb('#E79B96')[:3]),cheek)
 lip=(1-smooth((abs(x)-.024)/.009))*(1-smooth((abs(z-1.481)-.007)/.007))*smooth((-y-.137)/.015)
 base=base.lerp(Vector(rgb('#CF7F7E')[:3]),lip*.82)
 nose=gauss(x,.012)*gauss(z-1.51,.018)*front*.06; base=base.lerp(Vector(rgb('#F4C6B2')[:3]),nose)
 return (*base,1)
colorvals=[skin_color(p) for p in posed]; colors(body,colorvals)
pivot=Vector((0,-.006805,1.405825)); HM=Matrix.Translation(pivot)@Matrix.Rotation(math.radians(-9),4,'Y')@Matrix.Rotation(math.radians(-3),4,'Z')@Matrix.Translation(-pivot)
def H(p): return HM@Vector(p)
for v,p,c in zip(body.data.vertices,posed,colorvals):
 w=smooth((p.z-1.36)/.078); v.co=p.lerp(H(p),w)
subdiv(body,2)
capfaces=[]
for f in faces:
 c=sum((posed[i] for i in f),Vector())/len(f); front=smooth((-c.y-.025)/.11); threshold=1.485+.104*front
 if c.z>threshold and not (abs(c.x)>.068 and c.z<1.565 and c.y<-.015): capfaces.append(f)
cap=mesh_obj('Hair • fitted scalp volume', [H(p+normals[i]*.0055) for i,p in enumerate(posed)],capfaces,hair_m[1],HAIR); subdiv(cap,2); solid(cap,.0015)
white,wb=material('Eyes • warm sclera','#E9E5DC',.19); wb.inputs['Subsurface Weight'].default_value=.035; wb.inputs['Coat Weight'].default_value=.32; wb.inputs['Coat Roughness'].default_value=.08
iris,bi=material('Eyes • radial amber brown iris','#6B432B',.28); iv=node(iris,'ShaderNodeVertexColor','Individual radial iris fibers'); iv.layer_name='IrisColor'; iris.node_tree.links.new(iv.outputs['Color'],bi.inputs['Base Color']); bi.inputs['Coat Weight'].default_value=.3
pupil,_=material('Eyes • deep brown pupil','#090706',.16)
cornea,cb=material('Eyes • clear corneal surface',(1,1,1,1),.025); cb.inputs['Transmission Weight'].default_value=1; cb.inputs['IOR'].default_value=1.376
lidpaths=[]; lashpaths=[]
def sphere(name,center,r,mat,col):
 bpy.ops.mesh.primitive_uv_sphere_add(segments=64,ring_count=32,radius=r,location=center); o=bpy.context.object; o.name=name; move(o,col); o.data.materials.append(mat)
 for p in o.data.polygons: p.use_smooth=True
 return o
def iris_disk(name,cx,cy,cz,r,mat,col,cor=False):
 n=72; rings=7; vv=[(cx,cy-r-.00015,cz)]; ff=[]; cc=[rgb('#25170F') if not cor else (1,1,1,1)]
 for j in range(1,rings+1):
  rr=r*j/rings
  for k in range(n):
   a=2*math.pi*k/n; x=rr*math.cos(a); z=rr*math.sin(a); y=cy-math.sqrt(max(.000001,.0195*.0195-x*x-z*z))-(.00045 if cor else .00012)
   vv.append((x and cx+x or cx,y,cz+z))
   fiber=(.5+.5*math.sin(a*19+math.sin(a*7)*2))*(.5+.5*math.sin(a*43+j)); edge=smooth((j/rings-.78)/.22); c=Vector(rgb('#4A2B19')[:3]).lerp(Vector(rgb('#A16B35')[:3]),.25+.6*fiber).lerp(Vector(rgb('#241711')[:3]),edge*.85); cc.append((*c,1) if not cor else (1,1,1,1))
 for k in range(n): ff.append((0,1+k,1+(k+1)%n))
 for j in range(rings-1):
  a=1+j*n; b=a+n
  for k in range(n): ff.append((a+k,b+k,b+(k+1)%n,a+(k+1)%n))
 o=mesh_obj(name,[H(p) for p in vv],ff,mat,col)
 if not cor: colors(o,cc,'IrisColor')
 return o
for s in (-1,1):
 cx=s*.030775; cy=-.132; cz=1.5454; r=.0195
 sphere(('Left' if s>0 else 'Right')+' eye • complete scleral globe',H((cx,cy,cz)),r,white,EYES)
 iris_disk(('Left' if s>0 else 'Right')+' iris • radial chestnut fibers',cx,cy,cz,.0059,iris,EYES)
 iris_disk(('Left' if s>0 else 'Right')+' pupil',cx,cy,cz,.00255,pupil,EYES)
 iris_disk(('Left' if s>0 else 'Right')+' cornea • transparent cap',cx,cy,cz,.00635,cornea,EYES,True)
 vv=[]; ff=[]; N=72; R=7
 for j in range(R):
  t=j/(R-1)
  for k in range(N):
   a=2*math.pi*k/N; dx=.0148*math.cos(a); dz=(.0058 if math.sin(a)>0 else .0048)*math.sin(a); dz+=.0011*s*math.cos(a)*abs(math.cos(a)); ix=cx+dx; iz=cz+dz
   iy=cy-math.sqrt(max(.000001,r*r-dx*dx-dz*dz))-.00035
   ox=cx+dx*1.78; oz=cz+dz*3.25; oy=front_y(ox,oz)+.0007
   x=ix+(ox-ix)*t; z=iz+(oz-iz)*t; base=front_y(x,z); y=iy+(oy-iy)*t-.0012*math.sin(math.pi*t)
   if j==R-1: y=base+.0007
   vv.append(H((x,y,z)))
 for j in range(R-1):
  for k in range(N): ff.append((j*N+k,j*N+(k+1)%N,(j+1)*N+(k+1)%N,(j+1)*N+k))
 o=mesh_obj(('Left' if s>0 else 'Right')+' eyelids • sphere-wrapped almond aperture',vv,ff,skin,EYES); colors(o,[skin_color((cx, -.155,cz))]*len(vv)); subdiv(o,1)
 for k in range(37):
  a=.12+(math.pi-.24)*k/36; dx=.0148*math.cos(a); dz=.0058*math.sin(a)+.0011*s*math.cos(a)*abs(math.cos(a)); x=cx+dx; z=cz+dz; y=cy-math.sqrt(max(.000001,r*r-dx*dx-dz*dz))-.0007; length=random.uniform(.0015,.0032)*( .65+.35*math.sin(a)); lashpaths.append([(*H((x,y,z)),.8),(*H((x+s*.0004,y-.0007,z+length*.55)),.55),(*H((x+s*.0008,y-.001,z+length)),.08)])
curve_obj('Eyelashes • individually tapered upper lashes',lashpaths,.000115,[lashmat],EYES)
browpaths=[]
for s in (-1,1):
 for k in range(115):
  u=random.random(); x=s*(.011+.047*u); z=1.5665+.0038*math.sin(u*math.pi*.9)-.003*u+random.uniform(-.0012,.0012); y=front_y(x,z)-.0005; length=random.uniform(.0012,.0026); browpaths.append([(*H((x,y,z)),random.uniform(.55,1)),(*H((x+s*length*.65,y-.00015,z+length*.45)),.12)])
curve_obj('Eyebrows • fine combed individual hairs',browpaths,.00015,[hair_m[1],hair_m[2],lashmat],EYES)
smile=[]
for i in range(35):
 x=-.027+.054*i/34; z=1.4802+.0017*(abs(x)/.027)**1.5; y=front_y(x,z)-.00025; smile.append((*H((x,y,z)),.25+.55*math.sin(math.pi*i/34)))
curve_obj('Smile • delicate closed lip seam',[smile],.00018,[mouthmat],EYES)
center=Vector((0,-.02,1.55))
def scalp(theta,phi,offset=.0065):
 d=Vector((math.sin(phi)*math.sin(theta),-math.sin(phi)*math.cos(theta),math.cos(phi))); hit=bvh.ray_cast(center,d)[0]
 return H(hit+d*offset) if hit else H(center+Vector((d.x*.079,d.y*.087,d.z*.114)))
scalp_paths=[]
for i in range(560):
 theta=random.uniform(-math.pi,math.pi); end=1.15+1.2*(1-math.cos(theta))/2+random.uniform(-.06,.08); start=random.uniform(.03,.35); path=[]
 for j in range(28):
  t=j/27; phi=start+(end-start)*t; th=theta-.24*(1-t)*math.sin(theta)+.035*math.sin(t*math.pi); p=scalp(th,phi,random.uniform(.0055,.008)); path.append((*p,.65+.3*math.sin(math.pi*t)))
 scalp_paths.append(path)
curve_obj('Hair • 560 scalp-combed fine strands',scalp_paths,.00024,hair_m,HAIR)
bangpaths=[]
for i in range(190):
 u=(i+random.random())/190; ex=-.055+.114*u; ez=1.583-.018*u+random.uniform(-.006,.006); root=Vector((-.026+random.uniform(-.012,.012),-.035+random.uniform(-.008,.008),1.646+random.uniform(-.006,.004))); end=Vector((ex,front_y(ex,ez)-random.uniform(.003,.007),ez)); p1=Vector((root.x+.008,-.095,1.624)); p2=Vector((ex-.014,-.153+random.uniform(-.006,.005),ez+.025)); path=[]
 for j in range(32):
  t=j/31; p=(1-t)**3*root+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*end; path.append((*H(p),.35+.65*math.sin(math.pi*(t*.75+.1))))
 bangpaths.append(path)
curve_obj('Bangs • light diagonal wispy strands',bangpaths,.000235,hair_m,HAIR)
lockverts=[]; lockfaces=[]; lockmi=[]; strandpaths=[]
for s in (-1,1):
 for back in (False,True):
  count=9 if back else 15
  for k in range(count):
   u=(k+.5)/count; phase=random.uniform(0,6.28); width=random.uniform(.009,.016)*(1.12 if s<0 and not back else 1); thick=random.uniform(.0032,.0055)
   if back:
    controls=[(s*.025,.045,1.64),(s*.065,.065,1.57),(s*(.083+.012*u),.073,1.45),(s*(.095+.02*u),.065,1.30),(s*(.08+.035*u),.045,1.13+.045*u)]
   else:
    controls=[(s*(.018+.025*u),-.005,1.645-.015*u),(s*(.062+.014*u),-.015,1.59),(s*(.083+.012*u),-.035-.018*u,1.49),(s*(.092+.018*u),-.072-.025*u,1.35),(s*(.084+.033*u),-.095-.02*u,1.19+.035*u),(s*(.065+.04*u),-.09-.018*u,1.095+.065*u)]
   path=catmull(controls,52); frames=[]
   for j,p in enumerate(path):
    t=j/(len(path)-1); p=p+Vector((s*.0055*math.sin(t*math.pi*4+phase)*math.sin(math.pi*t),.004*math.cos(t*math.pi*5+phase)*math.sin(math.pi*t),0)); path[j]=p
    tangent=(path[min(j+1,len(path)-1)]-path[max(0,j-1)]).normalized(); side=tangent.cross(Vector((0,1,0))).normalized(); depth=side.cross(tangent).normalized(); frames.append((side,depth))
   base=len(lockverts); sides=10; mi=random.choices(range(5),[2,4,4,2,1])[0]
   for j,p in enumerate(path):
    t=j/(len(path)-1); profile=(.38+.62*math.sin(math.pi*(.1+.8*t)))*(1-t)**.28; w=width*profile+.00015; th=thick*profile+.0001; side,depth=frames[j]
    for a in range(sides):
     ang=2*math.pi*a/sides; lockverts.append(H(p+side*math.cos(ang)*w+depth*math.sin(ang)*th))
   for j in range(len(path)-1):
    for a in range(sides): lockfaces.append((base+j*sides+a,base+j*sides+(a+1)%sides,base+(j+1)*sides+(a+1)%sides,base+(j+1)*sides+a)); lockmi.append(mi)
   for q in range(20):
    ang=2*math.pi*(q+.25*random.random())/20; strand=[]
    for j in range(0,len(path),2):
     t=j/(len(path)-1); profile=(.38+.62*math.sin(math.pi*(.1+.8*t)))*(1-t)**.28; side,depth=frames[j]; p=path[j]+side*math.cos(ang)*(width*profile+.0003)+depth*math.sin(ang)*(thick*profile+.00025); strand.append((*H(p),.25+.75*(1-t)**.3))
    strandpaths.append(strand)
locks=mesh_obj('Hair • sculpted long wavy lock volumes',lockverts,lockfaces,hair_m[0],HAIR)
for m in hair_m[1:]: locks.data.materials.append(m)
for p,i in zip(locks.data.polygons,lockmi): p.material_index=i
subdiv(locks,1)
curve_obj('Hair • 960 surface-following chestnut fibers',strandpaths,.00026,hair_m,HAIR)
flypaths=[]
for i in range(95):
 s=random.choice((-1,1)); z=random.uniform(1.42,1.62); x=s*random.uniform(.078,.105); y=random.uniform(-.035,.035); path=catmull([(x,y,z),(x+s*.009,y-.006,z-.025),(x+s*.013,y-.01,z-.052)],15); flypaths.append([(*H(v),.55*(1-j/18)) for j,v in enumerate(path)])
curve_obj('Hair • restrained fine flyaways',flypaths,.00013,hair_m[1:],HAIR)
clothes=[]
for f in faces:
 c=sum((orig[i] for i in f),Vector())/len(f)
 if (1.096<c.z<1.374 and (abs(c.x)<.24 or c.z>1.24)) or (abs(c.x)>.22 and 1.075<c.z<1.34): clothes.append(f)
sv=[]
for i,p in enumerate(posed):
 o=orig[i]; q=p.copy(); n=normals[i]
 if abs(o.x)<.24 and o.z<1.35: q.x*=1.045; q.y=-.03+(q.y+.03)*1.075
 fold=.0032*math.sin(58*o.z+17*o.x+4*o.y)*math.sin(25*o.y+9*o.x)+.0025*math.sin(38*o.z-21*abs(o.x)); q+=n*(.014+fold); sv.append(q)
sweater=mesh_obj('Sweater • continuous loose torso and bent sleeves',sv,clothes,knit,CLOTH); subdiv(sweater,2); solid(sweater,.003)
def ellipse_band(name,center,rx,ry,minor,mat,axis=None):
 center=Vector(center); vv=[]; ff=[]; N=96; M=10; axis=Vector(axis).normalized() if axis else Vector((0,0,1)); u=axis.cross(Vector((0,1,0))).normalized()
 if u.length<.1: u=axis.cross(Vector((1,0,0))).normalized()
 v=axis.cross(u).normalized()
 for i in range(N):
  a=2*math.pi*i/N; radial=u*math.cos(a)+v*math.sin(a); p=center+u*(rx*math.cos(a))+v*(ry*math.sin(a))
  for j in range(M):
   b=2*math.pi*j/M; vv.append(p+radial*(minor*math.cos(b))+axis*(minor*math.sin(b)))
 for i in range(N):
  for j in range(M): ff.append((i*M+j,((i+1)%N)*M+j,((i+1)%N)*M+(j+1)%M,i*M+(j+1)%M))
 return mesh_obj(name,vv,ff,mat,CLOTH)
ellipse_band('Sweater • ribbed crew-neck collar',(0,-.008,1.377),.050,.041,.0065,rib)
ellipse_band('Sweater • soft lower rib hem',(0,-.048,1.108),.139,.094,.007,rib)
for s in (-1,1):
 c=arm_pose(Vector((s*.427,-.169,1.079))); ellipse_band(('Left' if s>0 else 'Right')+' sleeve • ribbed wrist cuff',c,.036,.032,.006,rib,new_dirs[s])
wood,_=material('Cafe • warm oak','#9A683C',.62); wall,_=material('Cafe • warm plaster','#C7B9A2',.9); ceramic,_=material('Cafe • cream ceramics','#D7D0BE',.32); green,_=material('Cafe • soft foliage','#4F6738',.75); dark,_=material('Cafe • dark window frame','#313735',.45)
def box(name,loc,dims,mat):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc); o=bpy.context.object; o.name=name; o.scale=dims; move(o,SET); o.data.materials.append(mat); bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); bevel=o.modifiers.new('Soft edges','BEVEL'); bevel.width=.012; bevel.segments=3; return o
box('Cafe • plaster wall',(0,.92,1.48),(2.4,.08,1.5),wall); box('Cafe • oak table behind sitter',(0,.48,1.035),(1.8,.72,.045),wood)
for z in (1.39,1.65): box('Cafe • floating oak shelf',(.34,.79,z),(.88,.20,.028),wood)
for i in range(7):
 x=.02+.115*(i%4); z=1.425+(i//4)*.26; bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=.027+random.random()*.01,depth=.07+random.random()*.04,location=(x,.72,z)); o=bpy.context.object; o.name='Cafe • blurred cup'; move(o,SET); o.data.materials.append(ceramic)
for x in (-.62,-.37): box('Cafe • window mullion',(x,.84,1.5),(.018,.03,1.1),dark)
for i in range(9): sphere('Cafe • defocused plant leaf',(-.55+random.uniform(-.12,.12),.65+random.uniform(-.08,.08),1.42+random.uniform(-.1,.3)),random.uniform(.025,.055),green,SET)
def point_at(o,p): o.rotation_euler=(Vector(p)-o.location).to_track_quat('-Z','Y').to_euler()
def area(name,loc,power,size,color):
 d=bpy.data.lights.new(name,'AREA'); d.energy=power; d.shape='DISK'; d.size=size; d.color=color; o=bpy.data.objects.new(name,d); LIGHTS.objects.link(o); o.location=loc; point_at(o,(0,-.04,1.47)); return o
area('Key • large soft window',(-.62,-.82,1.96),48,.72,(1,.88,.76)); area('Fill • cafe bounce',(.62,-.52,1.56),19,.65,(.83,.90,1)); area('Rim • warm hair separation',(.28,.40,1.91),38,.48,(1,.80,.61))
world=bpy.data.worlds.new('World • soft neutral ambient'); bpy.context.scene.world=world; world.use_nodes=True; bg=world.node_tree.nodes.get('Background'); bg.inputs['Color'].default_value=(.32,.38,.43,1); bg.inputs['Strength'].default_value=.28
focus=bpy.data.objects.new('Focus • eyes',None); CAMS.objects.link(focus); focus.location=H((0,-.145,1.535)); focus.empty_display_size=.015
def camera(name,loc,target,lens=65,ortho=None):
 d=bpy.data.cameras.new(name); o=bpy.data.objects.new(name,d); CAMS.objects.link(o); o.location=loc; point_at(o,target); d.lens=lens; d.clip_start=.01; d.clip_end=100
 if ortho: d.type='ORTHO'; d.ortho_scale=ortho
 return o
beauty=camera('Camera • beauty selfie',(0,-1.12,1.505),(0,-.025,1.397),70); beauty.data.dof.use_dof=True; beauty.data.dof.focus_object=focus; beauty.data.dof.aperture_fstop=5.6
frontcam=camera('Camera • diagnostic 0 front',(0,-2,1.36),(0,-.025,1.34),65,.76); threecam=camera('Camera • diagnostic 45 three-quarter',(1.45,-1.45,1.36),(0,-.025,1.34),65,.76); sidecam=camera('Camera • diagnostic 90 side',(2,0,1.36),(0,-.025,1.34),65,.76); backcam=camera('Camera • diagnostic 180 back',(0,2,1.36),(0,-.025,1.34),65,.76)
scene=bpy.context.scene; scene.name='Portrait • editable procedural character'; scene.camera=beauty; scene.render.engine='CYCLES'; scene.cycles.device='CPU'; scene.cycles.samples=32; scene.cycles.use_denoising=True; scene.cycles.max_bounces=7; scene.cycles.diffuse_bounces=3; scene.cycles.glossy_bounces=4; scene.cycles.transmission_bounces=6
scene.render.threads_mode='FIXED'; scene.render.threads=3; scene.render.resolution_x=960; scene.render.resolution_y=1040; scene.render.resolution_percentage=100; scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'; scene.render.film_transparent=False; scene.view_settings.view_transform='AgX'
try: scene.view_settings.look='AgX - Medium High Contrast'
except: pass
scene.render.filepath=os.path.join(ROOT,'renders','beauty.png'); scene['Reference use']='Visually inspected decoded reference; no photographic pixels or image planes used in scene.'; scene['Unseen geometry']='Back hair, side silhouette, sweater construction and posed hands are inferred from supplied CC0 topology and procedural design.'; scene['Build command']='Blender --background --factory-startup --threads 3 --python build.py'
def char_objects(): return [o for o in bpy.data.objects if any(c.name.startswith(('01 •','02 •','03 •','04 •')) for c in o.users_collection)]
def frame_diagnostic_cameras():
 # Coordinator technical framing fix: retain every authored vertex/material
 # and camera direction; use one shared orthographic scale that fits the full
 # evaluated upper-body asset in all four diagnostic views, with 8% margin.
 bpy.context.view_layer.update(); deps=bpy.context.evaluated_depsgraph_get(); points=[]
 for o in char_objects():
  ev=o.evaluated_get(deps); me=ev.to_mesh()
  if me:
   points.extend(ev.matrix_world@v.co for v in me.vertices); ev.to_mesh_clear()
 scale=0.0
 for cam in (frontcam,threecam,sidecam,backcam):
  inv=cam.matrix_world.inverted()
  for point in points:
   p=inv@point; scale=max(scale,abs(p.x)*2,abs(p.y)*2)
 for cam in (frontcam,threecam,sidecam,backcam): cam.data.ortho_scale=scale*1.08
def write_report():
 deps=bpy.context.evaluated_depsgraph_get(); counts={}; totalv=totalf=0; points=[]
 for o in char_objects():
  ev=o.evaluated_get(deps); me=ev.to_mesh()
  if me:
   counts[o.name]={'base_vertices':len(o.data.vertices) if o.type=='MESH' else None,'base_polygons':len(o.data.polygons) if o.type=='MESH' else None,'evaluated_vertices':len(me.vertices),'evaluated_polygons':len(me.polygons)}; totalv+=len(me.vertices); totalf+=len(me.polygons); points.extend([ev.matrix_world@Vector(p) for p in ev.bound_box]); ev.to_mesh_clear()
 bounds=None
 if points: bounds={'min':[min(p[i] for p in points) for i in range(3)],'max':[max(p[i] for p in points) for i in range(3)],'units':'meters'}
 files={}
 for n in ('beauty','front','three-quarter','side','back'):
  p=os.path.join(ROOT,'renders',n+'.png'); files['renders/'+n+'.png']={'exists':os.path.exists(p),'bytes':os.path.getsize(p) if os.path.exists(p) else None}
 report={'blender_version':bpy.app.version_string,'render_engine':scene.render.engine,'render_device':scene.cycles.device,'cpu_threads':scene.render.threads,'samples':32,'denoising':True,'important_objects':[o.name for o in char_objects()],'character_bounds':bounds,'base_mesh_source_vertices':len(orig),'base_mesh_source_polygons':len(faces),'evaluated_character_vertices':totalv,'evaluated_character_polygons':totalf,'object_mesh_counts':counts,'generated_texture_descriptions':['No photographic or external textures. Skin uses editable point colors for warm skin, cheek blush and lip tint; procedural Noise drives restrained pore bump.','Chestnut locks and fine hair are editable meshes and beveled 3D splines with five procedural material shades.','Dusty-pink wool uses procedural Noise color variation and crossed Wave textures for knit bump; collar, cuffs and hem are ribbed geometry.','Brown irises use editable radial vertex colors, separate pupils, scleral globes and clear corneal caps.'],'rendered_files':files,'limitations':['Likeness is a procedural interpretation, not a scan or exact reconstruction.','Facial expression, eye/lid transition, hair locks and knit relief remain approximate at close inspection.','Unseen side/back anatomy, back hairstyle and sweater construction are inferred.','Hands use reposed CC0 MakeHuman topology and are not a production rig; no legs or animation rig are supplied.','The cafe is deliberately simple 3D set dressing with optical depth of field.']}
 with open('scene-report.json','w') as f: json.dump(report,f,indent=2,ensure_ascii=False)
readme='''# Editable upper-body portrait

## Build, open, render

Run from this directory with installed Blender 5.1.1:

`/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py`

This builds the character, writes `character.blend`, packs resources, reopens the saved file, and renders all five required PNGs with Cycles CPU, 32 samples and denoising. For a faster two-view check, append `-- --preview`. Open `character.blend` in Blender; the active camera is **Camera • beauty selfie**. Character geometry is organized in collections 01–04, with editable meshes, materials, curve splines and modifiers.

## Reference access and design

The decoded `input/reference.png` **was actually seen and visually inspected with an image-reading tool** during production. `input/environment.json` and `input/ASSET-LICENSE.md` were also read. The reference guided the warm fair skin, brown eyes, slight closed-mouth smile, diagonal wispy bangs, long chestnut waves, dusty-pink crew neck, forward arm pose and gentle head tilt. No photograph, crop, generated portrait, billboard or photographic texture is used by the scene or renders.

The continuous MakeHuman facial surface is conservatively sculpted at the jaw, cheeks, nose, eye region and mouth corners. Real scleral globes sit behind custom almond eyelid surfaces; radial brown irises, pupils, clear corneas, lashes and individual brow hairs are separate editable geometry. The fitted scalp carries hundreds of combed strands, with additional bangs, wavy solid lock volumes, surface fibers and flyaways. The sweater expands and reposes anatomical topology into a loose garment with modeled folds, ribbed collar, cuffs and hem; procedural wool color and crossed-wave bump provide knit detail. Lighting uses large soft areas and a simple genuinely 3D cafe background. Diagnostic views hide only the cafe collection and use identical character geometry and materials.

## Inference and self-review

The unseen side/back silhouette, rear hair, garment construction and arm/hand posing are inferred. This is a procedural likeness study, not a scan or exact reconstruction. The face, expression and eye/lid blending remain approximations; lock-based hair and procedural knit cannot match every fine feature of the photograph. The simple cafe is secondary to the portrait. The hands are reposed base-mesh anatomy, not a production hand rig, and there is no full-body or animation rig. All final PNGs are direct Blender renders without photographic retouching or compositing.

## Base-mesh attribution

MakeHuman **hm08**, CC0, pinned commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`. Source supplied in `input/base.obj`; prepared topology in `input/upper-body.obj`. Source: https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj . The original header names Data Collection AB, Joel Palmius and Jonas Hauquier at the CC0 release. Geometry was cropped to the clothed upper body/hands, sculpted, reposed and supplemented here with original procedural hair, eyes, clothing, materials and set geometry. No additional models or textures were downloaded.
'''
with open('README.md','w') as f: f.write(readme)
frame_diagnostic_cameras(); write_report(); bpy.ops.file.pack_all(); bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT,'character.blend')); bpy.ops.wm.open_mainfile(filepath=os.path.join(ROOT,'character.blend'))
assert bpy.data.objects.get('Camera • beauty selfie') and bpy.data.objects.get('Woman • continuous sculpted head neck torso and posed hands')
scene=bpy.context.scene; scene.render.engine='CYCLES'; scene.cycles.device='CPU'; scene.render.threads_mode='FIXED'; scene.render.threads=3; scene.cycles.use_denoising=True; cafe=bpy.data.collections.get('05 • cafe geometry'); bg=scene.world.node_tree.nodes.get('Background')
def render(camera_name,path,w,h,samples):
 scene.camera=bpy.data.objects[camera_name]; scene.render.resolution_x=w; scene.render.resolution_y=h; scene.render.resolution_percentage=100; scene.cycles.samples=samples; scene.render.filepath=os.path.join(ROOT,path); bpy.ops.render.render(write_still=True)
if PREVIEW:
 render('Camera • beauty selfie','previews/beauty.png',600,650,12); cafe.hide_render=True; bg.inputs['Color'].default_value=(.20,.24,.28,1); bg.inputs['Strength'].default_value=.5; render('Camera • diagnostic 45 three-quarter','previews/three-quarter.png',600,600,12)
else:
 cafe.hide_render=False; render('Camera • beauty selfie','renders/beauty.png',960,1040,32); cafe.hide_render=True; bg.inputs['Color'].default_value=(.20,.24,.28,1); bg.inputs['Strength'].default_value=.5
 for name,cam in [('front','Camera • diagnostic 0 front'),('three-quarter','Camera • diagnostic 45 three-quarter'),('side','Camera • diagnostic 90 side'),('back','Camera • diagnostic 180 back')]: render(cam,'renders/'+name+'.png',768,768,32)
 cafe.hide_render=False; bg.inputs['Color'].default_value=(.32,.38,.43,1); bg.inputs['Strength'].default_value=.28; scene.camera=bpy.data.objects['Camera • beauty selfie']; scene.render.resolution_x=960; scene.render.resolution_y=1040; scene.render.filepath=os.path.join(ROOT,'renders','beauty.png'); scene.cycles.samples=32; write_report(); bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT,'character.blend'))
print('PORTRAIT_BUILD_COMPLETE', 'PREVIEW' if PREVIEW else 'FINAL')
