"""Editable portrait. Run using Blender 5.1.1 --background --factory-startup --threads 3 --python build.py.
PORTRAIT_PREVIEW=1 produces small inspection renders; default produces all deliverables.
No downloaded or photographic character textures. All geometry/textures generated here.
"""
import bpy, bmesh, math, random, os, json
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
random.seed(28107)
PREVIEW=os.environ.get('PORTRAIT_PREVIEW')=='1'
for d in ('renders','previews'): (ROOT/d).mkdir(exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
for d in (bpy.data.materials, bpy.data.curves, bpy.data.meshes):
 for x in list(d):
  if x.users==0:d.remove(x)
sc=bpy.context.scene
sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=12 if PREVIEW else 32
sc.cycles.use_denoising=True;sc.render.threads_mode='FIXED';sc.render.threads=3
sc.cycles.max_bounces=7;sc.cycles.diffuse_bounces=3;sc.cycles.glossy_bounces=4;sc.cycles.transmission_bounces=4
sc.render.image_settings.file_format='PNG';sc.render.image_settings.color_mode='RGBA';sc.render.film_transparent=False
sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast';sc.view_settings.exposure=-.35
sc.world.use_nodes=True;sc.world.node_tree.nodes.get('Background').inputs[0].default_value=(.31,.35,.42,1);sc.world.node_tree.nodes.get('Background').inputs[1].default_value=.28
collections={}
for name in ('01 Skin and face','02 Eyes','03 Hair','04 Knit sweater','05 Portrait studio','06 Cafe','07 Cameras'):
 c=bpy.data.collections.new(name);sc.collection.children.link(c);collections[name]=c
CHAR=[]
def place(o,col,char=True):
 for c in list(o.users_collection):c.objects.unlink(o)
 collections[col].objects.link(o)
 if char:CHAR.append(o)
 return o
def mat(name,color,rough=.45):
 m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough
 return m,p
skin,p=mat('Skin | warm fair complexion, SSS and microscopic pores',(.59,.36,.28),.42)
p.inputs['Subsurface Weight'].default_value=.16;p.inputs['Subsurface Radius'].default_value=(1,.46,.25);p.inputs['Subsurface Scale'].default_value=.0028
p.inputs['Specular IOR Level'].default_value=.32
n=skin.node_tree.nodes;l=skin.node_tree.links
col=n.new('ShaderNodeVertexColor');col.layer_name='Complexion';l.new(col.outputs['Color'],p.inputs['Base Color'])
tex=n.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=1700;tex.inputs['Detail'].default_value=3;tex.inputs['Roughness'].default_value=.7
bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.18;bump.inputs['Distance'].default_value=.00009;l.new(tex.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],p.inputs['Normal'])
ramp=n.new('ShaderNodeMapRange');ramp.inputs['From Min'].default_value=0;ramp.inputs['From Max'].default_value=1;ramp.inputs['To Min'].default_value=.36;ramp.inputs['To Max'].default_value=.48;l.new(tex.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs[0],p.inputs['Roughness'])
# Head pivot. Every eye/hair/brow follows this exact rigid head tilt.
bpy.ops.object.empty_add();head=bpy.context.object;head.name='Head pose | 20 degree gentle tilt';head.location=(0,-.02,1.395);place(head,'01 Skin and face')
def head_parent(o):
 o.parent=head;o.matrix_parent_inverse=head.matrix_world.inverted();return o
# Common continuous anatomical surface, sculpted analytically, never used as a photograph plane.
bpy.ops.wm.obj_import(filepath=str(ROOT/'input/upper-body.obj'))
body=bpy.context.object;body.rotation_euler=(0,0,0);body.name='Continuous sculpt | face ears neck';place(body,'01 Skin and face')
bm=bmesh.new();bm.from_mesh(body.data)
bmesh.ops.delete(bm,geom=[v for v in bm.verts if v.co.z<1.352 or (v.co.z<1.402 and abs(v.co.x)>.065)],context='VERTS');bm.to_mesh(body.data);bm.free()
original=[v.co.copy() for v in body.data.vertices]
def gauss(x,z,cx,cz,sx,sz):return math.exp(-((x-cx)/sx)**2-((z-cz)/sz)**2)
def sculpt(pt):
 x,y,z=pt
 if z>1.425:
  # Slight taper of lower jaw, softer cheek fullness, finer nasal bridge/tip.
  x*=1+.045*math.exp(-((z-1.505)/.043)**2)+.018*math.exp(-((z-1.465)/.025)**2)
  z+=.0045*math.exp(-((z-1.438)/.021)**2)
  if y<-.09:
   y-=.0034*(gauss(x,z,.041,1.52,.022,.026)+gauss(x,z,-.041,1.52,.022,.026))
   y+=.0037*gauss(x,z,0,1.523,.014,.035)
   x*=1-.075*gauss(x,z,0,1.512,.019,.023)
   # Eyes: enlarge the orbital opening and wrap its skin on the real eye spheres.
   for s in (-1,1):
    cx=s*.031
    w=gauss(x,z,cx,1.545,.021,.015)
    x+=(x-cx)*.07*w;z+=(z-1.545)*.24*w
   # Close the tiny native mouth slit without detaching or adding lip spheres.
   z+=(1.4782-z)*.58*gauss(x,z,0,1.4782,.027,.00155)
   # Lift lip corners and adjacent smile muscles, maintain closed mouth.
   z+=.0037*(gauss(x,z,.024,1.478,.009,.018)+gauss(x,z,-.024,1.478,.009,.018))
   y-=.0006*gauss(x,z,0,1.475,.023,.014)
 return Vector((x,y,z))
for v,pt in zip(body.data.vertices,original):v.co=sculpt(pt)
for poly in body.data.polygons:poly.use_smooth=True
body.data.materials.append(skin)
colors=body.data.color_attributes.new(name='Complexion',type='FLOAT_COLOR',domain='POINT')
for i,v in enumerate(body.data.vertices):
 x,y,z=v.co;c=np.array([.55,.335,.265])
 if y<-.10 and z>1.43:
  blush=(gauss(x,z,.044,1.516,.027,.019)+gauss(x,z,-.044,1.516,.027,.019))*.34
  c=c*(1-blush)+np.array([.66,.285,.26])*blush
  redness=.14*gauss(x,z,0,1.505,.014,.012)
  c=c*(1-redness)+np.array([.61,.255,.215])*redness
  # Color the actual integrated lip surface with a softened contour.
  ax=abs(x);u=min(1,ax/.0255);sm=.0034*math.exp(-((ax-.024)/.009)**2)
  top=1.4883-.009*u**1.8+sm-.0018*math.exp(-(x/.003)**2)
  bottom=1.4667+.011*u*u+sm
  def smooth(a,b,t):
   q=max(0,min(1,(t-a)/(b-a)));return q*q*(3-2*q)
  lip=smooth(bottom-.0008,bottom+.0012,z)*(1-smooth(top-.0012,top+.0006,z))*(1-smooth(.023,.026,ax))
  if y>-.147:lip*=.15
  c=c*(1-lip*.83)+np.array([.53,.16,.18])*lip*.83
  # Subtle warm lid/tissue tint.
  lid=sum(gauss(x,z,s*.031,1.55,.019,.010) for s in (-1,1))*.10
  c=c*(1-lid)+np.array([.52,.275,.23])*lid
 colors.data[i].color=(*c,1)
sub=body.modifiers.new('Continuous facial subdivision','SUBSURF');sub.levels=2;sub.render_levels=2
# BVH of subdivided face for brows and lip details before tilting.
bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get();ev=body.evaluated_get(deps);me=ev.to_mesh();face_bvh=BVHTree.FromPolygons([v.co for v in me.vertices],[p.vertices for p in me.polygons]);ev.to_mesh_clear()
def skin_front(x,z,offset=.0002):
 hit=face_bvh.ray_cast(Vector((x,-.4,z)),Vector((0,1,0)))
 return Vector((x,hit[0].y-offset if hit[0] else -.142,z))
def mesh_obj(name,verts,faces,material,col,parent=False):
 me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update();o=bpy.data.objects.new(name,me);collections[col].objects.link(o);CHAR.append(o)
 if material:o.data.materials.append(material)
 for p in me.polygons:p.use_smooth=True
 if parent:head_parent(o)
 return o
def sphere(name,loc,scale,material,col='02 Eyes',parent=True,segments=64,rings=32):
 bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,location=loc);o=bpy.context.object;o.name=name;o.scale=scale;o.data.materials.append(material)
 for p in o.data.polygons:p.use_smooth=True
 place(o,col)
 if parent:head_parent(o)
 return o
# Efficient editable 3D polyline strands with radius profile.
def curves(name,paths,radius,material,col='03 Hair',parent=True):
 cu=bpy.data.curves.new(name,'CURVE');cu.dimensions='3D';cu.resolution_u=1;cu.bevel_depth=radius;cu.bevel_resolution=0;cu.resolution_u=1;cu.use_fill_caps=True
 for path in paths:
  s=cu.splines.new('POLY');s.points.add(len(path)-1)
  for i,(p,co) in enumerate(zip(s.points,path)):
   p.co=(*co[:3],1);p.radius=co[3] if len(co)>3 else max(.08,1-(i/max(1,len(path)-1))**5)
 o=bpy.data.objects.new(name,cu);collections[col].objects.link(o);CHAR.append(o);cu.materials.append(material)
 if parent:head_parent(o)
 return o
sclera,p=mat('Sclera | warm ivory with restrained wet sheen',(.64,.64,.58),.24);p.inputs['Subsurface Weight'].default_value=.035;p.inputs['Specular IOR Level'].default_value=.38
pupil,p=mat('Pupil | deep brown black',(.0015,.0008,.0004),.18)
limbal,p=mat('Iris limbal transition',(.023,.010,.004),.28)
iris,p=mat('Iris | amber brown radial fibers',(.16,.066,.025),.28)
n=iris.node_tree.nodes;l=iris.node_tree.links;uv=n.new('ShaderNodeTexCoord');sep=n.new('ShaderNodeSeparateXYZ');l.new(uv.outputs['UV'],sep.inputs[0]);mult=n.new('ShaderNodeMath');mult.operation='MULTIPLY';mult.inputs[1].default_value=310;l.new(sep.outputs['X'],mult.inputs[0]);sin=n.new('ShaderNodeMath');sin.operation='SINE';l.new(mult.outputs[0],sin.inputs[0]);noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=85;noise.inputs['Detail'].default_value=2;l.new(uv.outputs['UV'],noise.inputs[0]);mix=n.new('ShaderNodeMath');mix.operation='MULTIPLY_ADD';mix.inputs[2].default_value=.5;l.new(sin.outputs[0],mix.inputs[0]);l.new(noise.outputs['Fac'],mix.inputs[1]);r=n.new('ShaderNodeValToRGB');r.color_ramp.elements[0].position=.15;r.color_ramp.elements[0].color=(.025,.009,.003,1);r.color_ramp.elements[1].position=.85;r.color_ramp.elements[1].color=(.145,.058,.019,1);l.new(mix.outputs[0],r.inputs[0]);l.new(r.outputs[0],p.inputs['Base Color'])
cornea,p=mat('Cornea | clear layered refractive surface',(.95,.98,1),.055);p.inputs['Transmission Weight'].default_value=1;p.inputs['IOR'].default_value=1.376;p.inputs['Specular IOR Level'].default_value=.5
wet,p=mat('Tearline | muted rose wet tissue',(.42,.18,.15),.19)
brow,p=mat('Brows and eyelashes | soft dark chestnut',(.047,.017,.009),.5)
# Eye sphere fitted behind the lids; iris/cornea are curved 3D anterior surfaces.
for s,label in [(-1,'R'),(1,'L')]:
 center=(s*.0312,-.1224,1.5452);rad=.0185
 sphere('Eyeball '+label,center,(rad,rad,rad),sclera)
 def iris_surface(name,r0,r1,material,depth=0):
  verts=[];faces=[];N=96;R=8
  for j in range(R+1):
   rr=r0+(r1-r0)*j/R
   for k in range(N):
    t=2*math.pi*k/N;x=rr*math.cos(t);z=rr*math.sin(t)
    bulge=.00105*(1-(rr/.0068)**2) if material==cornea else 0
    y=center[1]-math.sqrt(rad*rad-rr*rr)-depth-bulge
    verts.append((center[0]+x,y,center[2]+z))
  for j in range(R):
   for k in range(N):faces.append((j*N+k,j*N+(k+1)%N,(j+1)*N+(k+1)%N,(j+1)*N+k))
  o=mesh_obj(name,verts,faces,material,'02 Eyes',True);uv=o.data.uv_layers.new(name='Radial iris UV')
  for poly in o.data.polygons:
   for li in poly.loop_indices:
    vi=o.data.loops[li].vertex_index;uv.data[li].uv=((vi%N)/N,(vi//N)/R)
  return o
 iris_surface('Limbal ring '+label,.0060,.0067,limbal,.00012)
 iris_surface('Radial brown iris '+label,.00305,.0063,iris,.00018)
 iris_surface('Deep pupil '+label,0,.0031,pupil,.0002)
 iris_surface('Corneal curved cap '+label,0,.0068,cornea,.00039)
 # Hand-laid tapered upper lashes; follows sculpted native lid rim.
 paths=[]
 for i in range(42):
  t=i/41;x=s*(.0165+.030*t);z=1.5448+.0082*math.sin(math.pi*t)+.0008*t
  p0=skin_front(x,z,.00030);length=.002+.0028*math.sin(math.pi*t)**.7
  path=[]
  for j in range(6):
   q=j/5;path.append((p0.x+s*.0013*q,p0.y-.0025*q,p0.z+length*q-.0015*q*q,1-q*.9))
  paths.append(path)
 curves('Upper eyelashes '+label,paths,.000085,brow,'01 Skin and face')
 paths=[]
 for i in range(18):
  t=i/17;x=s*(.018+.026*t);z=1.545-.0067*math.sin(math.pi*t)
  p0=skin_front(x,z,.00018);paths.append([(p0.x,p0.y,p0.z,1),(p0.x+s*.0005,p0.y-.001,p0.z-.0017,.1)])
 curves('Lower fine lashes '+label,paths,.000045,brow,'01 Skin and face')
 # Individually planted eyebrows rather than an opaque painted bar.
 paths=[]
 for i in range(200):
  t=random.random();x=s*(.016+.038*t);z=1.563+.0052*math.sin(t*math.pi)-.004*t+random.uniform(-.0013,.0013)
  a=skin_front(x,z,.00017);b=skin_front(x+s*.0024,z+.002*math.sin(math.pi*t),.0003)
  paths.append([(*a,.8),(*((a+b)/2+Vector((0,-.0002,.0003))),.65),(*b,.05)])
 curves('Natural eyebrow hairs '+label,paths,.00013,brow,'01 Skin and face')
foldmat,p=mat('Eyelid crease | soft warm tissue shadow',(.34,.19,.15),.65)
for sign,label in [(-1,'R'),(1,'L')]:
 path=[]
 for j in range(30):
  t=j/29;x=sign*(.0175+.0305*t);z=1.555+.0050*math.sin(math.pi*t)-.0010*t
  v=skin_front(x,z,.0001);path.append((*v,.05+.65*math.sin(math.pi*t)))
 curves('Upper eyelid sculpt fold '+label,[path],.00010,foldmat,'01 Skin and face')
# A very fine lip seam rides the continuous skin, so the smile remains closed.
lipseam,p=mat('Closed smile | soft maroon occlusion',(.115,.029,.028),.58)
path=[]
for i in range(61):
 x=-.0245+.049*i/60;ax=abs(x);z=1.4775+.0035*math.exp(-((ax-.024)/.009)**2)-.0006*math.cos(x/.0245*math.pi/2)
 a=skin_front(x,z,.00017);path.append((*a,.25+.65*math.sin(math.pi*i/60)))
curves('Closed mouth fine smile crease',[path],.00007,lipseam,'01 Skin and face')
# Sweater: lofted torso and continuous rounded sleeve union, sculpted folds.
pink,p=mat('Dusty pink | procedural knitted yarn',(.52,.235,.25),.76);p.inputs['Sheen Weight'].default_value=.3;p.inputs['Sheen Roughness'].default_value=.65
# Portable 256px mathematical tile: interlocking yarn loop, twisted fine fibers.
S=256; yy,xx=np.mgrid[0:S,0:S]/S
height=np.zeros((S,S),dtype=np.float32)
for side in (-1,1):
 # yarn slants downward into a rounded knit V, periodic in both directions
 for shiftx in (-1,0,1):
  for shifty in (-1,0,1):
   t=np.clip((yy-shifty-.08)/.86,0,1)
   cx=.5+side*(.33*(1-t)+.035*np.sin(t*math.pi))+shiftx
   dist=np.sqrt((xx-cx)**2+np.maximum(0,.08+shifty-yy)**2+np.maximum(0,yy-(.94+shifty))**2)
   yarn=np.exp(-(dist/.067)**2)
   twist=.85+.15*np.sin(t*115+side*dist*330)
   height=np.maximum(height,yarn*twist)
height=np.clip(.12+.8*height+.022*np.sin(xx*540+yy*161)*np.sin(yy*700),0,1)
img=bpy.data.images.new('Procedural knit stitch | 256px loop and twisted yarn',width=S,height=S)
rgba=np.ones((S,S,4),dtype=np.float32);rgba[:,:,:3]=height[:,:,None];img.pixels.foreach_set(rgba.ravel());img.pack()
n=pink.node_tree.nodes;l=pink.node_tree.links;coord=n.new('ShaderNodeTexCoord');se=n.new('ShaderNodeSeparateXYZ');l.new(coord.outputs['Object'],se.inputs[0]);comb=n.new('ShaderNodeCombineXYZ')
for out,target,scale in [('X','X',290),('Z','Y',240)]:
 m=n.new('ShaderNodeMath');m.operation='MULTIPLY';m.inputs[1].default_value=scale;l.new(se.outputs[out],m.inputs[0]);l.new(m.outputs[0],comb.inputs[target])
im=n.new('ShaderNodeTexImage');im.image=img;im.extension='REPEAT';l.new(comb.outputs[0],im.inputs[0]);r=n.new('ShaderNodeValToRGB');r.color_ramp.elements[0].color=(.32,.128,.143,1);r.color_ramp.elements[1].color=(.59,.295,.312,1);l.new(im.outputs['Color'],r.inputs[0]);l.new(r.outputs[0],p.inputs['Base Color']);b=n.new('ShaderNodeBump');b.inputs['Strength'].default_value=.65;b.inputs['Distance'].default_value=.00075;l.new(im.outputs[0],b.inputs['Height']);l.new(b.outputs[0],p.inputs['Normal'])
yarn,p=mat('Raised stitches and rib | dusty rose yarn',(.53,.253,.27),.8);p.inputs['Sheen Weight'].default_value=.35

def loft_torso():
 rows=[(1.018,.178,.112),(1.052,.19,.12),(1.13,.186,.13),(1.21,.198,.135),(1.279,.222,.126),(1.331,.231,.116),(1.354,.198,.105),(1.379,.112,.089),(1.382,.055,.070)]
 verts=[];faces=[];N=96
 for j,(z,rx,ry) in enumerate(rows):
  for k in range(N):
   t=2*math.pi*k/N;rr=1+.018*math.sin(9*t+z*70)+.009*math.cos(17*t-z*42)
   zz=z+.0025*math.sin(4*t+j*.7);verts.append((rx*math.cos(t)*rr,ry*math.sin(t)-.032,zz))
 for j in range(len(rows)-1):
  for k in range(N):faces.append((j*N+k,j*N+(k+1)%N,(j+1)*N+(k+1)%N,(j+1)*N+k))
 faces.append(tuple(reversed(range(N))));faces.append(tuple((len(rows)-1)*N+k for k in range(N)))
 return mesh_obj('Sweater torso construction',verts,faces,pink,'04 Knit sweater')
def sleeve(s):
 centers=[(s*.180,-.012,1.314,.073),(s*.232,-.058,1.284,.073),(s*.262,-.14,1.21,.074),(s*.283,-.24,1.14,.072),(s*.308,-.355,1.085,.072),(s*.343,-.465,1.025,.075),(s*.36,-.53,.99,.071)]
 verts=[];faces=[];N=40
 for j,(x,y,z,rad) in enumerate(centers):
  c=Vector((x,y,z));prev=Vector(centers[max(0,j-1)][:3]);nxt=Vector(centers[min(len(centers)-1,j+1)][:3]);axis=(nxt-prev).normalized();a=axis.cross(Vector((0,0,1))).normalized();b=axis.cross(a).normalized()
  for k in range(N):
   t=2*math.pi*k/N;r=rad*(1+.038*math.sin(t*5+j*1.3));pt=c+r*(a*math.cos(t)+b*math.sin(t));verts.append(tuple(pt))
 for j in range(len(centers)-1):
  for k in range(N):faces.append((j*N+k,j*N+(k+1)%N,(j+1)*N+(k+1)%N,(j+1)*N+k))
 faces.append(tuple(reversed(range(N))));faces.append(tuple((len(centers)-1)*N+k for k in range(N)))
 return mesh_obj('Sleeve construction '+str(s),verts,faces,pink,'04 Knit sweater')
torso=loft_torso();sleeves=[sleeve(-1),sleeve(1)]
bpy.ops.object.select_all(action='DESELECT')
for o in [torso]+sleeves:o.select_set(True)
for o in sleeves:CHAR.remove(o)
bpy.context.view_layer.objects.active=torso;bpy.ops.object.join();torso.name='Sweater | unified shoulders and selfie sleeves'
rem=torso.modifiers.new('Joined cloth volume','REMESH');rem.mode='VOXEL';rem.voxel_size=.006;rem.use_smooth_shade=True;bpy.ops.object.modifier_apply(modifier=rem.name)
smo=torso.modifiers.new('Relaxed cloth surface','SMOOTH');smo.factor=1.4;smo.iterations=5;bpy.ops.object.modifier_apply(modifier=smo.name)
# Subtract real crew neck cavity; it is never a black disk.
bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=1,depth=1,location=(0,-.032,1.493));cut=bpy.context.object;cut.scale=(.046,.061,.25);bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
bpy.context.view_layer.objects.active=torso;mod=torso.modifiers.new('Open crew neck','BOOLEAN');mod.operation='DIFFERENCE';mod.object=cut;bpy.ops.object.modifier_apply(modifier=mod.name);bpy.data.objects.remove(cut,do_unlink=True)
for v in torso.data.vertices:
 x,y,z=v.co
 # fabric weight and folds over lower chest, elbows
 front=max(0,min(1,(-y-.025)/.10));fold=.0022*math.sin(92*z+9*math.sin(x*18))+.0011*math.sin(x*155+z*17)
 if z<1.33:v.co.y-=fold*front;v.co.x+=.0012*math.sin(z*100+x*17)
for p in torso.data.polygons:p.use_smooth=True
sub=torso.modifiers.new('Soft fabric subdivision','SUBSURF');sub.levels=1
# Raised crew neck band, true rib stitches.
verts=[];faces=[];N=160;R=12
for j in range(R+1):
 q=j/R;z=1.375+.017*q;rx=.125-.082*q;ry=.094-.030*q
 for k in range(N):
  t=2*math.pi*k/N;rib=.0007*math.cos(t*80);verts.append((.023*q+(rx+rib)*math.cos(t),-.032+(ry+rib)*math.sin(t),z+.0008*math.sin(3*t)))
for j in range(R):
 for k in range(N):faces.append((j*N+k,j*N+(k+1)%N,(j+1)*N+(k+1)%N,(j+1)*N+k))
collar=mesh_obj('Crew neck | ribbed knitted band',verts,faces,pink,'04 Knit sweater');sol=collar.modifiers.new('Collar thickness','SOLIDIFY');sol.thickness=.003
paths=[]
for k in range(120):
 t=2*math.pi*k/120;path=[]
 for j in range(10):
  q=j/9;rx=.125-.082*q;ry=.094-.030*q;path.append((.023*q+(rx+.0007)*math.cos(t),-.032+(ry+.0007)*math.sin(t),1.375+.017*q+.0008*math.sin(3*t),.7))
 paths.append(path)
curves('Crew neck raised rib yarn',paths,.00042,yarn,'04 Knit sweater',False)
# A selection of actual 3D knit loops on front complements the fine procedural tile.
bpy.context.view_layer.update();ev=torso.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.to_mesh();cloth_bvh=BVHTree.FromPolygons([v.co for v in me.vertices],[p.vertices for p in me.polygons]);ev.to_mesh_clear();paths=[]
for j in range(40):
 z=1.045+j*.0068
 for k in range(55):
  x=-.18+k*.0067;hit=cloth_bvh.ray_cast(Vector((x,-.36,z)),Vector((0,1,0)))
  if not hit[0] or hit[0].y>-.068:continue
  y=hit[0].y-.00035
  path=[(x-.0021,y,z+.0024,.7),(x-.0017,y-.0001,z+.0012,1),(x-.0009,y-.00025,z-.0004,1),(x,y-.0003,z-.0017,1),(x+.0009,y-.00025,z-.0004,1),(x+.0017,y-.0001,z+.0012,1),(x+.0021,y,z+.0024,.7)]
  paths.append(path)
curves('Front fabric | individually formed knit V loops',paths,.00027,yarn,'04 Knit sweater',False)
# Hair shaders, opaque depth foundation and many individually tapered fine strands.
hairm=[]
for i,c in enumerate([(.038,.013,.0065),(.057,.020,.009),(.072,.027,.014),(.026,.008,.0035),(.094,.038,.019)]):
 m,p=mat('Chestnut hair | strand tone '+str(i),c,.48+i*.014);p.inputs['Anisotropic'].default_value=.48;p.inputs['Coat Weight'].default_value=.045;p.inputs['Coat Roughness'].default_value=.4;hairm.append(m)
# Fitted scalp, with a natural frontal/temporal boundary.
verts=[];faces=[];N=100;R=32
for j in range(R+1):
 for k in range(N):
  th=2*math.pi*k/N;front=max(0,-math.sin(th));phmax=1.83-.75*front**2
  ph=.012+(phmax-.012)*j/R
  verts.append((.091*math.sin(ph)*math.cos(th),-.043+.123*math.sin(ph)*math.sin(th),1.55+.137*math.cos(ph)))
for j in range(R):
 for k in range(N):faces.append((j*N+k,(j+1)*N+k,(j+1)*N+(k+1)%N,j*N+(k+1)%N))
scalp=mesh_obj('Hair | fitted continuous scalp foundation',verts,faces,hairm[3],'03 Hair',True)
# Smooth interpolation used only to construct tangible curves/lock volume.
def catmull(points,num):
 out=[]
 for i in range(num):
  t=i/(num-1)*(len(points)-1);k=min(len(points)-2,int(t));u=t-k
  a=Vector(points[max(0,k-1)]);b=Vector(points[k]);c=Vector(points[k+1]);d=Vector(points[min(len(points)-1,k+2)])
  out.append(.5*((2*b)+(-a+c)*u+(2*a-5*b+4*c-d)*u*u+(-a+3*b-3*c+d)*u*u*u))
 return out
locks=[]
# Side curtain: reference's fuller long fall on viewer-left, right tucked behind ear.
for s in (-1,1):
 for k in range(34):
  a=k/33;front=a<.56;endz=random.uniform(1.055,1.19) if s==-1 else random.uniform(1.10,1.25)
  th=(-math.pi+.25+a*.75) if s==-1 else (-.25-a*.75)
  # crown roots run across the cranial surface into side lengths
  root=(s*random.uniform(.001,.025),random.uniform(-.065,.003),1.689+random.uniform(-.003,.003))
  yside=-.083-.045*a if s==-1 else -.010-.05*a
  points=[root,(s*.064,-.082-.023*a,1.655),(s*.084,yside,1.594),(s*(.078+.046*a),yside+.008,1.505),(s*(.070+.072*a),yside-.025,1.40),(s*(.025+.122*a),-.151-.020*a,1.28),(s*(.026+.115*a),-.160-.022*a,endz+.065),(s*(.031+.098*a),-.169-.015*a,endz)]
  path=catmull(points,30)
  for v in path:
   if v.z<1.41:
    hit=cloth_bvh.ray_cast(Vector((v.x,-.45,v.z)),Vector((0,1,0)))
    if hit[0]:v.y=min(v.y,hit[0].y-.018)
  locks.append((path,random.uniform(.005,.010),s,k))
# Full back fall, all visible in the unchanged 180 degree view.
for k in range(42):
 th=.07+(math.pi-.14)*k/41;endz=random.uniform(1.04,1.16)
 points=[(.035*math.cos(th),-.033+.036*math.sin(th),1.681),(.086*math.cos(th),-.028+.099*math.sin(th),1.625),(.089*math.cos(th),-.018+.11*math.sin(th),1.52),(.101*math.cos(th),.015+.105*math.sin(th),1.39),(.145*math.cos(th),.025+.112*math.sin(th),1.26),(.152*math.cos(th),.016+.113*math.sin(th),endz)]
 locks.append((catmull(points,30),random.uniform(.007,.013),0,k))
# Underlying flattened locks carry depth; their surface is hidden largely by fine fibers.
verts=[];faces=[]
for idx,(path,width,s,k) in enumerate(locks):
 path=[p0 for p0 in path if p0.z<1.605]
 start=len(verts)
 for j,p0 in enumerate(path):
  q=j/(len(path)-1);wave=max(0,(1.62-p0.z)/.55)
  center=p0+Vector((.014*math.sin((1.61-p0.z)*29+idx*.47)*wave,.010*math.sin((1.61-p0.z)*25+idx*.27)*wave,0))
  tangent=(path[min(j+1,len(path)-1)]-path[max(0,j-1)]).normalized();side=tangent.cross(Vector((0,1,0))).normalized();normal=tangent.cross(side).normalized();w=width*.43*(.6+.4*math.sin(q*math.pi))*(1-.87*q**9)
  for u in range(8):
   t=2*math.pi*u/8;verts.append(tuple(center+side*(math.cos(t)*w)+normal*(math.sin(t)*w*.33)))
 for j in range(len(path)-1):
  for u in range(8):faces.append((start+j*8+u,start+j*8+(u+1)%8,start+(j+1)*8+(u+1)%8,start+(j+1)*8+u))
 mesh_dummy=0
volume=mesh_obj('Hair | layered wave lock volumes',verts,faces,None,'03 Hair',True)
for m in hairm:volume.data.materials.append(m)
for p in volume.data.polygons:p.material_index=3
# Fine strand paths follow lock shape with controlled irregularity and tapered tips.
groups=[[] for _ in hairm]
for idx,(path,width,s,k) in enumerate(locks):
 for f in range(28):
  spread=random.uniform(-.95,.95)*width;depth=random.uniform(-.9,.9)*width*.23;phase=random.uniform(0,math.pi*2);arr=[]
  for j,p0 in enumerate(path):
   q=j/(len(path)-1);tangent=(path[min(j+1,len(path)-1)]-path[max(0,j-1)]).normalized();side=tangent.cross(Vector((0,1,0))).normalized();normal=tangent.cross(side).normalized();wave=max(0,(1.62-p0.z)/.55)
   p1=p0+side*(spread*(.55+.45*math.sin(q*math.pi)))+normal*depth+Vector((.014*math.sin((1.61-p0.z)*29+idx*.47)*wave,.010*math.sin((1.61-p0.z)*25+idx*.27)*wave,0))
   p1+=Vector((.0007*math.sin(q*32+phase),-.0006*math.sin(q*23+phase),0));arr.append((*p1,.85*(1-q**5)+.06))
  groups[random.choices(range(5),[34,30,14,17,5])[0]].append(arr)
# Thousands of short crown-combing strands cover the full cap and make the part legible.
for f in range(650):
 th=random.uniform(-math.pi,math.pi);fr=max(0,-math.sin(th));end=1.80-.75*fr**2;start=random.uniform(.07,.7);arr=[]
 for j in range(25):
  q=j/24;ph=start+(end-start)*q;theta=th+.10*math.sin(math.pi*q)
  if math.sin(th)<-.2:
   endth=-.28 if math.cos(th)>0 else -math.pi+.28
   theta=th+(endth-th)*q*q;p0=Vector((.0922*math.sin(ph)*math.cos(theta),-.043+.1244*math.sin(ph)*math.sin(theta),1.55+.1384*math.cos(ph)))
  arr.append((*p0,.9*(1-q**6)+.06))
 groups[random.randrange(4)].append(arr)
for i,paths in enumerate(groups):curves('Hair | fine chestnut strands '+str(i),paths,.00016 if i!=4 else .00012,hairm[i])
# Additional individual frontal crown fibers prevent a smooth helmet surface.
paths=[]
for k in range(360):
 th=random.uniform(-2.75,-.39);fr=max(0,-math.sin(th));end=1.83-.75*fr**2+random.uniform(-.015,.035);start=random.uniform(.08,.40);arr=[]
 for j in range(24):
  q=j/23;ph=start+(end-start)*q;theta=th+.10*math.sin(q*math.pi)
  v=Vector((.0922*math.sin(ph)*math.cos(theta),-.043+.1244*math.sin(ph)*math.sin(theta),1.55+.1384*math.cos(ph)))
  arr.append((*v,.72*(1-q**8)+.03))
 paths.append(arr)
curves('Hair | fine frontal crown fibers',paths,.000125,hairm[1])
# Wispy curtain bangs: swept outside the facial surface, distinct from long side locks.
groups=[[] for _ in hairm]
for k in range(16):
 s=-1 if k<8 else 1;a=(k%8)/7
 endx=s*(.028+.038*a);endz=1.579-.023*a
 points=[(-.012+s*.007,-.093,1.659),(-.020+s*.014,-.137,1.633),(s*(.022+.023*a),-.158,1.608),(s*(.029+.026*a),-.160,1.590),(endx,-.153+.014*a,endz)]
 base=catmull(points,30)
 for f in range(25):
  dx=random.uniform(-.0027,.0027);dy=random.uniform(-.001,.001);arr=[]
  for j,v in enumerate(base):
   q=j/(len(base)-1);arr.append((v.x+dx*math.sin(q*math.pi/2),v.y+dy,v.z+random.uniform(-.0003,.0003),.8*(1-q**3)+.035))
  groups[random.randrange(4)].append(arr)
for i,paths in enumerate(groups):
 if paths:curves('Bangs | separated wisps '+str(i),paths,.00012,hairm[i])
# A restrained number of hair flyaways, rather than a uniformly perfect wig.
paths=[]
for k in range(95):
 s=random.choice([-1,1]);z0=random.uniform(1.46,1.65);points=[(s*.076,-.044,z0),(s*.105,-.055,z0+.012),(s*.122,-.048,z0-.005),(s*.117,-.08,z0-.036)];path=catmull(points,15);paths.append([(*v,.55*(1-j/14)+.04) for j,v in enumerate(path)])
curves('Hair | fine loose flyaways',paths,.000075,hairm[2])
# Apply the head pose consistently to face/neck and all head components.
angle=math.radians(-20);rot=Matrix.Rotation(angle,4,'Y');pivot=head.location.copy()
# Long hair follows gravity: root tilt fades into the unrotated hanging lengths.
for o in list(collections['03 Hair'].objects):
 if o.type not in ('MESH','CURVE'):continue
 points=o.data.vertices if o.type=='MESH' else [p for sp in o.data.splines for p in sp.points]
 for p in points:
  v=Vector(p.co[:3]);w=max(0,min(1,(v.z-1.28)/.18));v=pivot+Matrix.Rotation(-angle*(1-w),4,'Y')@(v-pivot)-Matrix.Rotation(-angle,4,'Y')@Vector((.025*(1-w),0,0))
  if o.type=='MESH':p.co=v
  else:p.co=(*v,1)
for v in body.data.vertices:
 w=max(0,min(1,(v.co.z-1.36)/.066));r=Matrix.Rotation(angle*w,4,'Y');v.co=pivot+r@(v.co-pivot)+Vector((.025*w,0,0))
head.rotation_euler[1]=angle;head.location.x+=.025
# Fully geometric, modest cafe environment with daylight and shelves.
wood,p=mat('Cafe | honey oak',(.31,.16,.067),.5);n=wood.node_tree.nodes;l=wood.node_tree.links;noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=7;noise.inputs['Detail'].default_value=2;r=n.new('ShaderNodeValToRGB');r.color_ramp.elements[0].color=(.17,.071,.025,1);r.color_ramp.elements[1].color=(.40,.23,.11,1);l.new(noise.outputs['Fac'],r.inputs[0]);l.new(r.outputs[0],p.inputs['Base Color'])
wall,p=mat('Cafe | warm plaster',(.62,.55,.43),.85);ceramic,p=mat('Cafe | ivory cups',(.66,.66,.58),.32);green,p=mat('Cafe | leafy greens',(.035,.11,.024),.65);dark,p=mat('Cafe | charcoal accents',(.026,.031,.032),.6)
def cube(name,loc,scale,material,bevel=.02):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.object;o.name=name;o.scale=scale;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(material);place(o,'06 Cafe',False)
 if bevel:b=o.modifiers.new('Soft furniture edges','BEVEL');b.width=bevel;b.segments=3;o.modifiers.new('Weighted furniture normals','WEIGHTED_NORMAL')
 return o
cube('Cafe back plaster',(0,1.65,1.3),(4,.1,3),wall)
for z in (1.16,1.62,2.07):cube('Oak cafe shelf',( .48,1.49,z),(1.7,.21,.045),wood)
for z in (1.2,1.66,2.11):
 for k in range(7):
  x=-.1+k*.21;bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=random.uniform(.032,.045),depth=random.uniform(.08,.12),location=(x,1.39,z+.05));o=bpy.context.object;o.name='Cafe ceramic cup';o.data.materials.append(ceramic if k%3 else dark);place(o,'06 Cafe',False)
cube('Window upright',(-.78,.8,1.5),(.035,.13,2.5),wood)
cube('Window daylight surround',(-1.35,1.0,1.6),(1.05,.07,2.4),ceramic)
cube('Cafe table behind',(.55,.69,.98),(1.15,.65,.05),wood)
cube('Selfie table foreground',(0,-.2,.975),(1.12,.75,.035),wood)
# Rounded geometric leaves on stems; sufficiently distant to read as blurred greenery.
leafstems=[]
for k in range(40):
 x=-.54+random.uniform(-.19,.19);y=.57+random.uniform(-.12,.12);z=1.2+random.uniform(0,.95)
 o=sphere('Cafe leaf',(x,y,z),(.038,.009,.018),green,'06 Cafe',False,16,8);o.rotation_euler=(random.random()*2,random.random()*2,random.random()*3)
 if o in CHAR:CHAR.remove(o)
 leafstems.append([(-.54,.59,1.13,.8),(x*.5-.27,(y+.59)*.5,z-.13,.6),(x,y,z,.15)])
stems=curves('Cafe | plant stems',leafstems,.0012,green,'06 Cafe',False);CHAR.remove(stems)
bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=.14,depth=.23,location=(-.54,.59,1.02));o=bpy.context.object;o.name='Cafe | ceramic planter';o.data.materials.append(ceramic);place(o,'06 Cafe',False)
# Cameras and lights. Portrait and diagnostic views always use identical character geometry.
def aim(o,pt):o.rotation_euler=(Vector(pt)-o.location).to_track_quat('-Z','Y').to_euler()
def camera(name,loc,target,lens=50,ortho=None):
 bpy.ops.object.camera_add(location=loc);o=bpy.context.object;o.name=name;o.data.lens=lens;aim(o,target);place(o,'07 Cameras',False)
 if ortho:o.data.type='ORTHO';o.data.ortho_scale=ortho
 return o
beauty=camera('Beauty | selfie portrait',(0,-1.015,1.55),(-.005,-.055,1.356),50)
beauty.data.dof.use_dof=True;beauty.data.dof.focus_object=head;beauty.data.dof.aperture_fstop=7.1;beauty.data.dof.aperture_blades=8
# Actual eye focus distance, independent of the neck pivot.
bpy.ops.object.empty_add(location=(-.028,-.142,1.54));focus=bpy.context.object;focus.name='Portrait focus | eyes';place(focus,'07 Cameras',False);beauty.data.dof.focus_object=focus
cams={}
target=(0,-.21,1.315)
for name,deg in [('front',0),('three-quarter',45),('side',90),('back',180)]:
 a=math.radians(deg);center=(0,-.30,1.315) if name=='three-quarter' else target
 cams[name]=camera('Inspection | '+name,(1.55*math.sin(a),-1.55*math.cos(a)+center[1],1.40),center,50,1.12 if name=='three-quarter' else 1.04)
facecam=camera('Preview | close facial inspection',(0,-.68,1.55),(-.025,-.08,1.55),65,.285)
def area(name,loc,target,power,size,color):
 bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.name=name;o.data.energy=power;o.data.shape='DISK';o.data.size=size;o.data.color=color;aim(o,target);place(o,'05 Portrait studio',False)
area('Window | large soft daylight',(-.55,-.50,1.96),(0,-.05,1.48),32,.80,(1,.88,.78))
area('Front | broad reflected fill',(.43,-.65,1.56),(0,-.03,1.45),12,.75,(.78,.86,1))
area('Cafe | hair separation',(.20,.50,1.88),(0,-.02,1.46),24,.7,(1,.77,.55))
area('Low reflected fill | sleeve ends',(0,-.88,.92),(0,-.20,1.15),3.5,.8,(1,.9,.82))
area('Back view | soft studio fill',(-.50,.64,1.40),(0,.01,1.38),8,.7,(.85,.90,1))
# Helpful default viewport and final portrait settings.
sc.camera=beauty;sc.render.resolution_x=960;sc.render.resolution_y=1040;sc.render.resolution_percentage=100;sc.render.filepath=str(ROOT/'renders/beauty.png')
for screen in bpy.data.screens:
 for a in screen.areas:
  if a.type=='VIEW_3D':a.spaces.active.region_3d.view_perspective='CAMERA'
bpy.ops.object.select_all(action='DESELECT');body.select_set(True);bpy.context.view_layer.objects.active=body
# Remove deleted construction entries, and measure evaluated render geometry.
CHAR=[o for o in CHAR if o.name in bpy.data.objects and o.type in ('MESH','CURVE') and any(c.name.startswith(('01','02','03','04')) for c in o.users_collection)]
bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();base_v=0;eval_v=0;eval_faces=0;allpts=[]
for o in CHAR:
 if o.type=='MESH':base_v+=len(o.data.vertices)
 ev=o.evaluated_get(dg);me=ev.to_mesh();eval_v+=len(me.vertices);eval_faces+=len(me.polygons)
 for v in me.vertices:allpts.append(ev.matrix_world@v.co)
 ev.to_mesh_clear()
bounds={'min':[min(p[i] for p in allpts) for i in range(3)],'max':[max(p[i] for p in allpts) for i in range(3)]}
# Saved BEFORE rendering. Embedded knit texture is portable, and no reference photo is rendered.
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'character.blend'))
# Genuine reopening in Blender checks the file rather than assuming the save succeeded.
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'character.blend'))
sc=bpy.context.scene;beauty=bpy.data.objects['Beauty | selfie portrait'];cafe=bpy.data.collections['06 Cafe']
if PREVIEW:
 sc.render.resolution_x=480;sc.render.resolution_y=520;sc.render.filepath=str(ROOT/'previews/assembled.png');bpy.ops.render.render(write_still=True)
 cafe.hide_render=True;sc.camera=bpy.data.objects['Preview | close facial inspection'];sc.camera.data.dof.use_dof=False;sc.render.resolution_x=640;sc.render.resolution_y=640;sc.render.filepath=str(ROOT/'previews/face.png');bpy.ops.render.render(write_still=True)
 sc.camera=bpy.data.objects['Inspection | side'];sc.render.resolution_x=480;sc.render.resolution_y=480;sc.render.filepath=str(ROOT/'previews/side.png');bpy.ops.render.render(write_still=True)
else:
 sc.cycles.samples=32;sc.render.filepath=str(ROOT/'renders/beauty.png');bpy.ops.render.render(write_still=True)
 cafe.hide_render=True;sc.render.resolution_x=768;sc.render.resolution_y=768
 sc.world.node_tree.nodes.get('Background').inputs[0].default_value=(.19,.21,.25,1)
 for name in ('front','three-quarter','side','back'):
  sc.camera=bpy.data.objects['Inspection | '+name];sc.render.filepath=str(ROOT/'renders'/f'{name}.png');bpy.ops.render.render(write_still=True)
 # Restore editable scene to beauty composition and cafe for portable opening.
 cafe.hide_render=False;sc.camera=beauty;sc.world.node_tree.nodes.get('Background').inputs[0].default_value=(.31,.35,.42,1)
 sc.render.resolution_x=960;sc.render.resolution_y=1040;sc.render.filepath=str(ROOT/'renders/beauty.png');bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'character.blend'))
report={'blender_version':bpy.app.version_string,'render_engine':sc.render.engine,'render_device':'CPU','threads':3,'samples':sc.cycles.samples,'denoising':True,'reference_image_actually_decoded_and_seen':True,'reference_used_as_render_texture':False,'important_objects':[o.name for o in bpy.data.objects if o.type in ('MESH','CURVE') and any(c.name.startswith(('01','02','03','04')) for c in o.users_collection)],'character_bounds_m':bounds,'base_mesh_vertices':base_v,'evaluated_vertices':eval_v,'evaluated_polygons':eval_faces,'generated_textures':[{'name':'Procedural knit stitch','size':[256,256],'description':'Mathematical interlocking loop/yarn height tile; generated from numpy, packed into blend. No photograph data.'}, {'name':'Skin','description':'Per-vertex complexion plus shader noise pores and roughness; no raster skin photograph.'}, {'name':'Iris','description':'Radial UV sine/noise brown fibers on curved iris geometry.'}], 'rendered_filenames':['renders/beauty.png','renders/front.png','renders/three-quarter.png','renders/side.png','renders/back.png'] if not PREVIEW else ['previews/assembled.png','previews/face.png','previews/side.png'],'limitations':['Single photograph interpretation, not a scan or exact facial reconstruction.','Side/back anatomy and garment/hair arrangement inferred.','Procedural hair can read as a groomed wig, with simplified lock topology.','No production rig; face/neck pose is baked, head pivot groups eyes/hair. Sleeve ends are cropped closed cloth sections without hands.','Knit is procedural relief supplemented by selected physical yarn loops, not an entire simulated woven garment.','Cornea/eye alignment and facial likeness are handcrafted approximations.', 'Skin remains too uniform and the face visibly CG rather than photographic.', 'Hair has coarse tips, organized strand/lock patterns and simplified fringe.', 'Cloth drape and cafe furnishings are simplified.']}
(ROOT/'scene-report.json').write_text(json.dumps(report,indent=2))
print('PORTRAIT COMPLETE',eval_v,'evaluated vertices',bounds)
