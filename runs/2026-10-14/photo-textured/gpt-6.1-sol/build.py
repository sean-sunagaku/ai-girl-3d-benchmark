"""Second finishing pass. Uses only immutable source-character.blend and input/reference.png.
Run Blender --background --factory-startup --threads 3 --python build.py.
PORTRAIT_PREVIEW=1 renders low-resolution inspection views; default builds all deliverables.
"""
import bpy, math, random, json, os
import numpy as np
from pathlib import Path
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
PREVIEW=os.environ.get('PORTRAIT_PREVIEW')=='1'
random.seed(90351)
rng=np.random.default_rng(90351)
for folder in ('renders','previews','textures'):(ROOT/folder).mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'source-character.blend'))
s=bpy.context.scene
s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=12 if PREVIEW else 32
s.cycles.use_denoising=True;s.render.threads_mode='FIXED';s.render.threads=3
s.cycles.max_bounces=8;s.cycles.transmission_bounces=5
s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGBA'
s.render.use_compositing=False;s.render.use_sequencer=False;s.render.film_transparent=False
s.view_settings.view_transform='AgX';s.view_settings.look='AgX - Medium High Contrast';s.view_settings.exposure=-.4
skin_col=bpy.data.collections['01 Skin and face'];hair_col=bpy.data.collections['03 Hair'];eyes_col=bpy.data.collections['02 Eyes']
# Preserve only the first pass's usable hanging tubular hair fibers and depth cores.
# Its faulty crown curves are excluded by strand extent; the scalp and crown are rebuilt below.
bpy.context.view_layer.update()
source_long=[]
for o in hair_col.objects:
 if o.type=='CURVE' and o.name.startswith('Hair | fine chestnut strands'):
  rows=[]
  for sp in o.data.splines:
   pts=[o.matrix_world@Vector(p.co[:3]) for p in sp.points]
   if max(p.z for p in pts)-min(p.z for p in pts)>.22:
    rows.append([(p.copy(),sp.points[j].radius) for j,p in enumerate(pts)])
  source_long.append((o.name,rows,o.data.materials[0]))
o=bpy.data.objects['Hair | layered wave lock volumes']
source_core=([o.matrix_world@v.co for v in o.data.vertices],[tuple(p.vertices) for p in o.data.polygons],o.data.materials[3])

head=bpy.data.objects['Head pose | 20 degree gentle tilt']
pivot=Vector((0,-.02,1.395));oldpivot=head.location.copy();oldrot=Matrix.Rotation(math.radians(-20),3,'Y')
newpivot=Vector((.025,-.02,1.395));newrot=Matrix.Rotation(math.radians(-26),3,'Y')
body=bpy.data.objects['Continuous sculpt | face ears neck']
# Undo the baked first-pass tilt, reshape in anatomical coordinates, then reapply.
local=[]
for v in body.data.vertices:
 # neck blend is practically rigid above 1.426. Fixed-point inversion below it.
 p=oldrot.inverted()@(v.co-oldpivot)+pivot
 for _ in range(8):
  w=max(0,min(1,(p.z-1.36)/.066));r=Matrix.Rotation(math.radians(-20)*w,3,'Y')
  p=r.inverted()@(v.co-pivot-Vector((.025*w,0,0)))+pivot
 x,y,z=p
 if y<-.10:
  # Retain continuous anatomical mouth topology while reducing the native opening.
  w=math.exp(-(x/.028)**4-((z-1.4782)/.0045)**2)
  z+=(1.4782-z)*.35*w
  # Soft integrated tip and slight fullness at the cheekbone.
  y+=.0014*math.exp(-(x/.013)**2-((z-1.509)/.018)**2)
 p=Vector((x,y,z));local.append(p.copy())
 w=max(0,min(1,(z-1.36)/.066));v.co=pivot+Matrix.Rotation(math.radians(-26)*w,3,'Y')@(p-pivot)+Vector((.025*w,0,0))
head.rotation_euler[1]=math.radians(-26);head.name='Head pose | gentle 26 degree tilt'
# First-pass eye centers were recessed behind the lids. Move each complete optical assembly.
for o in eyes_col.objects:
 if o.type=='MESH':
  if o.name.startswith('Eyeball'):o.location.y-=.003
  else:
   for v in o.data.vertices:v.co.y-=.003
# Rebuild curved iris optics, keeping pupil and cornea fitted to the same eyeball.
for side in ('L','R'):
 cx=.0312 if side=='L' else -.0312
 for prefix in ('Limbal ring','Radial brown iris','Deep pupil','Corneal curved cap'):
  o=bpy.data.objects[prefix+' '+side]
  for v in o.data.vertices:
   x=(v.co.x-cx)*1.10;z=(v.co.z-1.5452)*1.10;rr=(x*x+z*z)**.5
   depth={'Limbal ring':.00008,'Radial brown iris':.00013,'Deep pupil':.00018,'Corneal curved cap':.00031}[prefix]
   bulge=.00050*max(0,1-(rr/.00748)**2) if prefix=='Corneal curved cap' else 0
   v.co=(cx+x,-.1254-math.sqrt(max(.000001,.0185**2-rr**2))-depth-bulge,1.5452+z)
cor=bpy.data.materials['Cornea | clear layered refractive surface'].node_tree.nodes.get('Principled BSDF')
cor.inputs['Roughness'].default_value=.035
# Decode reference through Blender, generate genuine packed UV textures from its pixels.
ref=bpy.data.images.load(str(ROOT/'input/reference.png'),check_existing=True)
W,H=ref.size
photo=np.array(ref.pixels[:],dtype=np.float32).reshape(H,W,4)
# Blender pixel buffer is bottom-up; all manually inspected landmarks below are top-down.
photo=photo[::-1,:,:3]
def sample(px,py):
 px=np.clip(px,0,W-1.001);py=np.clip(py,0,H-1.001)
 x0=px.astype(np.int32);y0=py.astype(np.int32);fx=(px-x0)[...,None];fy=(py-y0)[...,None]
 return photo[y0,x0]*(1-fx)*(1-fy)+photo[y0,x0+1]*fx*(1-fy)+photo[y0+1,x0]*(1-fx)*fy+photo[y0+1,x0+1]*fx*fy
# Thin-plate spline face projection. Source image landmarks chosen by direct visual inspection.
anchors=[
 (-.0312,1.5452,507,394),(.0312,1.5452,648,319),
 (-.047,1.545,475,401),(-.016,1.545,528,382),(.016,1.545,620,334),(.047,1.545,674,303),
 (-.031,1.554,501,377),(.031,1.554,640,299),(-.031,1.537,513,416),(.031,1.537,652,335),
 (0,1.510,610,451),(-.011,1.502,587,466),(.011,1.502,646,449),
 (-.024,1.481,590,515),(.024,1.481,713,447),(0,1.478,650,490),
 (0,1.487,641,477),(0,1.469,659,512),(0,1.442,674,576),
 (-.055,1.510,477,477),(.056,1.510,778,410),
 (-.042,1.469,568,550),(.042,1.469,741,514),
 (0,1.610,501,221),(-.055,1.581,414,326),(.055,1.581,699,249)]
a=np.array([[x*100,(z-1.54)*100] for x,z,_,_ in anchors]);b=np.array([[u,v] for _,_,u,v in anchors])
def kernel(d):return d*np.log(np.maximum(d,1e-12))
d=((a[:,None,:]-a[None,:,:])**2).sum(2);K=kernel(d)+np.eye(len(a))*.006
P=np.column_stack((np.ones(len(a)),a));L=np.block([[K,P],[P.T,np.zeros((3,3))]])
coeff=np.linalg.solve(L,np.vstack((b,np.zeros((3,2)))))
def warp(x,z):
 q=np.column_stack((x.ravel()*100,(z.ravel()-1.54)*100));dd=((q[:,None,:]-a[None,:,:])**2).sum(2)
 out=kernel(dd)@coeff[:len(a)]+np.column_stack((np.ones(len(q)),q))@coeff[len(a):]
 return out[:,0].reshape(x.shape),out[:,1].reshape(x.shape)
def smooth(lo,hi,v):
 t=np.clip((v-lo)/(hi-lo),0,1);return t*t*(3-2*t)
def image_asset(name,rgba,filename,noncolor=False):
 h,w=rgba.shape[:2];im=bpy.data.images.new(name,width=w,height=h,alpha=True)
 im.colorspace_settings.name='Non-Color' if noncolor else 'sRGB'
 im.pixels.foreach_set(rgba.astype(np.float32).ravel());im.filepath_raw=str(ROOT/'textures'/filename);im.file_format='PNG';im.save();im.pack()
 return im
# 1024 UV atlas, frontal anatomical (x,z), with baked feathered exclusion masks.
N=1024
zz,xx=np.mgrid[0:N,0:N];xx=-.105+.210*xx/(N-1);zz=1.40+.28*zz/(N-1)
# Batch TPS avoids a large temporary array.
uvp=[]
for j in range(0,N,64):uvp.append(warp(xx[j:j+64],zz[j:j+64]))
px=np.concatenate([q[0] for q in uvp]);py=np.concatenate([q[1] for q in uvp]);rgb=sample(px,py)
# Smooth photometric normalization: reduce the source's broad light/shadow variation.
# Keep photographic chroma, pores, freckles and lip/brow color. No source eye catchlights used on skin.
base=np.array([.81,.665,.61]);lum=rgb.mean(2)
# Frequency filtering removes broad lighting rather than adding a photograph to the final render.
f=np.fft.rfft2(lum);fy=np.fft.fftfreq(N)[:,None];fx=np.fft.rfftfreq(N)[None,:]
low=np.fft.irfft2(f*np.exp(-2*math.pi**2*22**2*(fx*fx+fy*fy)),s=(N,N)).real
correct=np.clip((.70/np.maximum(low,.18))**.32,.78,1.23)
rgb=np.clip(rgb*correct[:,:,None]*.86,0,1)
# Projection support: clear only the photographed central face. Background/hair excluded by chroma/luminance.
mask=(1-smooth(.061,.078,np.abs(xx)))*smooth(1.432,1.448,zz)*(1-smooth(1.600,1.621,zz))
# Eye bulbs are entirely 3D: suppress source irises/sclerae on adjacent skin.
for cx in (-.0312,.0312):
 dist=((xx-cx)/.018)**2+((zz-1.5452)/.0085)**2
 mask*=smooth(.75,1.35,dist)
# Reject dark hair pixels, except deliberately remapped eyebrow and lip regions.
brows=np.maximum(np.exp(-((xx-.031)/.024)**4-((zz-1.565)/.006)**4),np.exp(-((xx+.031)/.024)**4-((zz-1.565)/.006)**4))
lips=np.exp(-(xx/.026)**6-((zz-1.479)/.012)**6)
hair_reject=smooth(.30,.46,rgb.mean(2))*smooth(.53,.73,rgb[:,:,2]/np.maximum(rgb[:,:,0],.01))
mask*=np.maximum(hair_reject,np.maximum(brows,lips))
rgba=np.ones((N,N,4),np.float32);rgba[:,:,:3]=rgb;rgba[:,:,3]=mask
face_img=image_asset('Reference-derived face | masked de-lit frontal UV atlas',rgba,'face-albedo.png')
uv=body.data.uv_layers.new(name='Face reference projection | editable anatomical x-z')
for loop in body.data.loops:
 p=local[loop.vertex_index];uv.data[loop.index].uv=((p.x+.105)/.210,(p.z-1.40)/.28)
# Geometry-dependent support excludes ear, side and back surfaces even if they share frontal UVs.
weight=body.data.color_attributes.new(name='PhotoSupport',type='FLOAT_COLOR',domain='POINT')
for i,p in enumerate(local):
 w=(1-smooth(-.120,-.084,p.y))*smooth(1.430,1.450,p.z)
 weight.data[i].color=(w,w,w,1)
skin=bpy.data.materials['Skin | warm fair complexion, SSS and microscopic pores']
n=skin.node_tree.nodes;l=skin.node_tree.links;p=n.get('Principled BSDF')
# Reduce skin's overly waxy first-pass scattering and retain fine procedural normal relief.
p.inputs['Subsurface Weight'].default_value=.10;p.inputs['Subsurface Scale'].default_value=.0018
p.inputs['Specular IOR Level'].default_value=.28
old=n.get('Color Attribute')
base_node=n.new('ShaderNodeRGB');base_node.label='Fair warm skin on unseen side / ear / neck';base_node.outputs[0].default_value=(.57,.395,.33,1)
t=n.new('ShaderNodeTexImage');t.image=face_img;t.label='Supplied photograph, masked and lighting attenuated';t.interpolation='Linear'
u=n.new('ShaderNodeUVMap');u.uv_map=uv.name;l.new(u.outputs[0],t.inputs[0])
vc=n.new('ShaderNodeVertexColor');vc.layer_name='PhotoSupport'
m=n.new('ShaderNodeMath');m.operation='MULTIPLY';l.new(t.outputs['Alpha'],m.inputs[0]);l.new(vc.outputs['Color'],m.inputs[1])
mix=n.new('ShaderNodeMixRGB');mix.blend_type='MIX';l.new(m.outputs[0],mix.inputs[0]);l.new(base_node.outputs[0],mix.inputs[1]);l.new(t.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],p.inputs['Base Color'])
# Iris photo texture: isolate one actual brown iris, no lids/catchlights retained.
I=256;iy,ix=np.mgrid[0:I,0:I];x=(ix/(I-1)*2-1);y=(iy/(I-1)*2-1)
# Iris center in source is (507,394), radius approximately 15px; replace baked glints by neighboring brown pixels.
irgb=sample(507+x*15,394-y*15);bright=(irgb.max(2)>.50)|(irgb[:,:,2]/np.maximum(irgb[:,:,0],.01)>.78)|((x*x+y*y)<.60**2)
irgb[bright]=np.array([.30,.17,.095]);irgb=np.clip(irgb*.88,0,1)
irgba=np.ones((I,I,4));irgba[:,:,:3]=irgb
iris_img=image_asset('Reference brown iris | isolated and source glint removed',irgba,'iris-albedo.png')
for side in ('L','R'):
 o=bpy.data.objects['Radial brown iris '+side];u=o.data.uv_layers.new(name='Photographic iris planar UV')
 cx=.0312 if side=='L' else -.0312
 for lp in o.data.loops:
  v=o.data.vertices[lp.vertex_index].co;u.data[lp.index].uv=((v.x-cx)/.01386+.5,(v.z-1.5452)/.01386+.5)
m=bpy.data.materials['Iris | amber brown radial fibers'];n=m.node_tree.nodes;l=m.node_tree.links;p=n.get('Principled BSDF')
t=n.new('ShaderNodeTexImage');t.image=iris_img;u=n.new('ShaderNodeUVMap');u.uv_map='Photographic iris planar UV';l.new(u.outputs[0],t.inputs[0]);l.new(t.outputs[0],p.inputs['Base Color']);p.inputs['Roughness'].default_value=.25
# Source knit patch contains only sweater; make a mirror-repeat tile and remove broad shading.
K=256;yy,xx=np.mgrid[0:K,0:K];knit=sample(815+xx*150/(K-1),820+yy*150/(K-1))
lum=knit.mean(2);F=np.fft.rfft2(lum);fy=np.fft.fftfreq(K)[:,None];fx=np.fft.rfftfreq(K)[None,:]
low=np.fft.irfft2(F*np.exp(-2*math.pi**2*12**2*(fx*fx+fy*fy)),s=(K,K)).real
rel=np.clip(lum/np.maximum(low,.1),.62,1.34)
rel=np.block([[rel,rel[:,::-1]],[rel[::-1,:],rel[::-1,::-1]]])
knrgba=np.ones((512,512,4));knrgba[:,:,:3]=rel[:,:,None]*np.array([.78,.53,.555])
knit_img=image_asset('Reference sweater | normalized seamless mirror knit tile',knrgba,'knit-albedo.png')
htrgba=np.ones((512,512,4));htrgba[:,:,:3]=np.clip((rel-.65)/.70,0,1)[:,:,None]
knit_height=image_asset('Reference knit | yarn relief from normalized luminance',htrgba,'knit-height.png',True)
cloth=bpy.data.materials['Dusty pink | procedural knitted yarn'];n=cloth.node_tree.nodes;l=cloth.node_tree.links;p=n.get('Principled BSDF')
coord=n.new('ShaderNodeTexCoord');mapping=n.new('ShaderNodeVectorMath');mapping.operation='SCALE';mapping.inputs['Scale'].default_value=17;l.new(coord.outputs['Object'],mapping.inputs[0])
al=n.new('ShaderNodeTexImage');al.image=knit_img;al.projection='BOX';al.projection_blend=.28;l.new(mapping.outputs[0],al.inputs[0]);l.new(al.outputs[0],p.inputs['Base Color'])
ht=n.new('ShaderNodeTexImage');ht.image=knit_height;ht.projection='BOX';ht.projection_blend=.28;l.new(mapping.outputs[0],ht.inputs[0])
bu=n.new('ShaderNodeBump');bu.inputs['Strength'].default_value=.48;bu.inputs['Distance'].default_value=.00048;l.new(ht.outputs[0],bu.inputs['Height']);l.new(bu.outputs[0],p.inputs['Normal'])
p.inputs['Roughness'].default_value=.82;p.inputs['Sheen Weight'].default_value=.28
# Remove selected huge repeated front yarn Vs, replacing with photographic knit relief over solid cloth.
bpy.data.objects.remove(bpy.data.objects['Front fabric | individually formed knit V loops'],do_unlink=True)
# Give cloth editable UVs as well as its seamless triplanar material.
for o in bpy.data.collections['04 Knit sweater'].objects:
 if o.type=='MESH' and not o.data.uv_layers:
  u=o.data.uv_layers.new(name='Cloth editable cylindrical UV')
  for lp in o.data.loops:
   v=o.data.vertices[lp.vertex_index].co;u.data[lp.index].uv=((math.atan2(v.y+.032,v.x)/(2*math.pi))%1,(v.z-.93)/.50)
# Entire old groom is replaced: no polygon strips across the forehead and no reused crown bug.
for o in list(hair_col.objects):bpy.data.objects.remove(o,do_unlink=True)
def material(name,color,rough):
 m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=rough
 return m,p
rootmat,p=material('Hair root foundation | dark chestnut scalp',(.021,.007,.0035),.65)
# Fine ellipsoidal root cap only, hidden beneath tens of thousands of real native hairs.
V=[];F=[];NN=128;RR=40
RX=.097;RY=.128;RZ=.142;CY=-.043;CZ=1.55
for j in range(RR+1):
 for k in range(NN):
  th=2*math.pi*k/NN;fr=max(0,-math.sin(th));end=1.94-.80*fr**3;ph=.003+(end-.003)*j/RR
  V.append((RX*math.sin(ph)*math.cos(th),CY+RY*math.sin(ph)*math.sin(th),CZ+RZ*math.cos(ph)))
for j in range(RR):
 for k in range(NN):F.append((j*NN+k,(j+1)*NN+k,(j+1)*NN+(k+1)%NN,j*NN+(k+1)%NN))
me=bpy.data.meshes.new('Fitted scalp root surface');me.from_pydata(V,[],F);me.update();cap=bpy.data.objects.new('Hair | fitted scalp beneath native fibers',me);hair_col.objects.link(cap);me.materials.append(rootmat)
for f in me.polygons:f.use_smooth=True
for v in me.vertices:v.co=newpivot+newrot@(v.co-pivot)
# Genuine photographic hair texture on the fitted, volumetric scalp; never a card or plane.
# Native strands remain on and beyond this cranial surface. Unobserved hair uses a source-hair tile.
HN=1024;hz,hx=np.mgrid[0:HN,0:HN];hx=-.105+.210*hx/(HN-1);hz=1.49+.22*hz/(HN-1)
hp=[]
for j in range(0,HN,64):hp.append(warp(hx[j:j+64],hz[j:j+64]))
hpx=np.concatenate([q[0] for q in hp]);hpy=np.concatenate([q[1] for q in hp]);hrgb=sample(hpx,hpy)
hmask=(1-smooth(.67,.79,hrgb.mean(2)))*(1-smooth(.58,.76,hrgb[:,:,2]/np.maximum(hrgb[:,:,0],.01)))
hmask*=smooth(2,18,hpy)*smooth(100,145,hpx)*(1-smooth(870,930,hpx))
hairrgba=np.ones((HN,HN,4));hairrgba[:,:,:3]=np.where(hmask[:,:,None]>.05,hrgb*.68,np.array([.23,.13,.08]));hairrgba[:,:,3]=hmask
hair_image=image_asset('Reference crown hair | masked anatomical projection',hairrgba,'hair-crown-albedo.png')
T=256;ty,tx=np.mgrid[0:T,0:T];hairtile=sample(433+tx*76/(T-1),810+ty*135/(T-1))
hairtile=np.concatenate((np.concatenate((hairtile,hairtile[:,::-1]),1),np.concatenate((hairtile[::-1],hairtile[::-1,::-1]),1)),0)
trgba=np.ones((512,512,4));trgba[:,:,:3]=hairtile*.78
hair_tile=image_asset('Reference hair | inferred-side source strand tile',trgba,'hair-strand-tile.png')
uvh=me.uv_layers.new(name='Crown source projection x-z');uvt=me.uv_layers.new(name='Cranial hair cylindrical UV')
maskh=me.color_attributes.new(name='CrownFrontSupport',type='FLOAT_COLOR',domain='POINT')
for i,v in enumerate(V):
 x,y,z=v;maskh.data[i].color=(1-smooth(-.12,-.02,y),)*3+(1,)
for lp in me.loops:
 x,y,z=V[lp.vertex_index];uvh.data[lp.index].uv=((x+.105)/.210,(z-1.49)/.22);uvt.data[lp.index].uv=((math.atan2(y-CY,x)/(2*math.pi))%1*4,(z-1.49)/.19*2)
n=rootmat.node_tree.nodes;l=rootmat.node_tree.links;p=n.get('Principled BSDF')
u=n.new('ShaderNodeUVMap');u.uv_map=uvh.name;h=n.new('ShaderNodeTexImage');h.image=hair_image;l.new(u.outputs[0],h.inputs[0])
u=n.new('ShaderNodeUVMap');u.uv_map=uvt.name;t=n.new('ShaderNodeTexImage');t.image=hair_tile;t.extension='REPEAT';l.new(u.outputs[0],t.inputs[0])
v=n.new('ShaderNodeVertexColor');v.layer_name='CrownFrontSupport';mm=n.new('ShaderNodeMath');mm.operation='MULTIPLY';l.new(h.outputs['Alpha'],mm.inputs[0]);l.new(v.outputs['Color'],mm.inputs[1])
x=n.new('ShaderNodeMixRGB');l.new(mm.outputs[0],x.inputs[0]);l.new(t.outputs['Color'],x.inputs[1]);l.new(h.outputs['Color'],x.inputs[2]);l.new(x.outputs[0],p.inputs['Base Color']);p.inputs['Roughness'].default_value=.85;p.inputs['Specular IOR Level'].default_value=0
b=n.new('ShaderNodeBump');b.inputs['Strength'].default_value=.15;b.inputs['Distance'].default_value=.00017;l.new(t.outputs['Color'],b.inputs['Height']);l.new(b.outputs[0],p.inputs['Normal'])
# True hair curves are measured separately from evaluated polygon geometry.
hairm=[]
for i,c in enumerate([(.070,.027,.013),(.094,.040,.021),(.046,.015,.006),(.126,.059,.031)]):
 m=bpy.data.materials.new('Native chestnut hair | tone '+str(i));m.use_nodes=True;n=m.node_tree.nodes;n.clear();out=n.new('ShaderNodeOutputMaterial');p=n.new('ShaderNodeBsdfHairPrincipled');p.parametrization='MELANIN';p.inputs['Melanin'].default_value=[.79,.72,.86,.65][i];p.inputs['Melanin Redness'].default_value=.55;p.inputs['Roughness'].default_value=.43;p.inputs['Radial Roughness'].default_value=.48;p.inputs['Random Color'].default_value=.18;m.node_tree.links.new(p.outputs[0],out.inputs[0]);hairm.append(m)
def hair_object(name,paths,radii,mat):
 cu=bpy.data.hair_curves.new(name);cu.add_curves([len(p) for p in paths]);o=bpy.data.objects.new(name,cu);hair_col.objects.link(o)
 cu.attributes['position'].data.foreach_set('vector',np.array([p for a in paths for p in a],np.float32).ravel())
 rad=cu.attributes.new('radius','FLOAT','POINT');rad.data.foreach_set('value',np.array([r for a in radii for r in a],np.float32));cu.materials.append(mat)
 return o
def posed(v):
 # Crown follows skull; long hanging fibers blend smoothly into gravity.
 w=smooth(1.25,1.49,v[2]);return np.array(pivot+Matrix.Rotation(math.radians(-26)*float(w),3,'Y')@(Vector(v)-pivot)+Vector((.025*float(w),0,0)))
def catmull(points,num):
 out=[]
 for j in range(num):
  t=j/(num-1)*(len(points)-1);k=min(len(points)-2,int(t));u=t-k
  a,b,c,d=[np.array(points[i]) for i in [max(0,k-1),k,k+1,min(len(points)-1,k+2)]]
  out.append(.5*((2*b)+(-a+c)*u+(2*a-5*b+4*c-d)*u*u+(-a+3*b-3*c+d)*u**3))
 return out
paths=[[] for _ in hairm];rads=[[] for _ in hairm]
def add(path,r=.000065,tone=None):
 i=random.choices(range(4),[43,27,23,7])[0] if tone is None else tone
 paths[i].append([posed(v) for v in path]);rads[i].append([r*(.88+.12*math.sin(j/(len(path)-1)*math.pi))*(.15+.85*min(1,j/3))*(1-.97*(j/(len(path)-1))**8) for j in range(len(path))])
# Dense combed crown: every point independently evaluated on its own ellipsoid.
for k in range(2800):
 th=random.uniform(-math.pi,math.pi);fr=max(0,-math.sin(th));phend=1.92-.78*fr**3+random.uniform(-.05,.025)
 phstart=random.uniform(.025,.85);startth=-math.pi/2 if math.sin(th)<0 else math.pi/2
 # root part runs crown to front, two sides fan softly toward temples
 if math.cos(th)<0:startth=-math.pi/2-.14 if math.sin(th)<0 else math.pi/2+.14
 else:startth=-math.pi/2+.14 if math.sin(th)<0 else math.pi/2-.14
 depth=random.uniform(.0010,.0048);jitter=random.uniform(0,6.28);arr=[]
 for j in range(30):
  q=j/29;ph=phstart+(phend-phstart)*q;theta=th+.045*math.sin(q*math.pi)+.008*math.sin(q*12+jitter)
  arr.append(((RX+depth)*math.sin(ph)*math.cos(theta)-.0015*(1-q),(CY+(RY+depth)*math.sin(ph)*math.sin(theta)),CZ+(RZ+depth)*math.cos(ph)))
 add(arr,random.uniform(.000045,.000071))
# Retain the first pass's useful hanging volume and tubular strand normals.
# The photographic scalp and newly laid crown have replaced its faulty crown construction.
def retilt_world(v):
 w=float(smooth(1.28,1.49,v.z));return newpivot+Matrix.Rotation(math.radians(-6)*w,3,'Y')@(v-newpivot)
for name,rows,material in source_long:
 cu=bpy.data.curves.new('Repaired hanging fibers | '+name[-1],'CURVE');cu.dimensions='3D';cu.resolution_u=1;cu.bevel_depth=.00015;cu.bevel_resolution=0;cu.use_fill_caps=True
 for row in rows:
  # Leave the crown to the fitted scalp. Roots tuck beneath its side surface.
  row=[(v,r) for v,r in row if v.z<1.60]
  if len(row)<5:continue
  sp=cu.splines.new('POLY');sp.points.add(len(row)-1)
  for j,((v,r),p) in enumerate(zip(row,sp.points)):
   vv=retilt_world(v)
   if j<4:
    lv=newrot.inverted()@(vv-newpivot)+pivot;d=lv-Vector((0,CY,CZ));q=math.sqrt((d.x/RX)**2+(d.y/RY)**2+(d.z/RZ)**2)
    if q>.965:
     tucked=Vector((0,CY,CZ))+d*(.965/q);tucked=newpivot+newrot@(tucked-pivot);vv=vv.lerp(tucked,(1-j/4)**2)
   p.co=(*vv,1);p.radius=r*(.25+.75*min(1,j/2))
 o=bpy.data.objects.new('Hair | editable tubular hanging fibers '+name[-1],cu);hair_col.objects.link(o);cu.materials.append(material)
verts,faces,material=source_core
me2=bpy.data.meshes.new('Inferred wave cores');me2.from_pydata([retilt_world(v) for v in verts],[],faces);me2.update();o=bpy.data.objects.new('Hair | overlapping sculpted wave depth',me2);hair_col.objects.link(o);me2.materials.append(material)
for f in me2.polygons:f.use_smooth=True
# Light, separate wispy bangs, slim fibers over the forehead rather than a solid fringe patch.
for k in range(800):
 side=random.choice([-1,1]);t=random.random();ex=side*(.018+.041*t);ez=1.578-.026*t
 points=[(-.014+random.uniform(-.007,.006),-.099,1.656),(-.019+side*.009,-.142,1.626),(side*(.013+.018*t),-.162,1.604),(ex,-.165+.008*t,ez)]
 arr=catmull(points,28);dx=random.uniform(-.003,.003);dy=random.uniform(-.002,.002)
 for j,v in enumerate(arr):v[0]+=dx*j/27;v[1]+=dy
 add(arr,random.uniform(.000033,.000052))
# Sparse long flyaways and neck wisps.
for k in range(0):
 side=random.choice([-1,1]);z=random.uniform(1.45,1.65);arr=catmull([(side*.076,-.040,z),(side*.109,-.072,z+.022),(side*.129,-.100,z-.020),(side*.118,-.132,z-.068)],24);add(arr,.000035)
for i in range(4):hair_object('Hair | native chestnut fibers '+str(i),paths[i],rads[i],hairm[i])
# Face crease and brows soften with the photograph's fine detail.
for o in skin_col.objects:
 if o.type=='CURVE' and o.name.startswith('Natural eyebrow'):o.data.bevel_depth*=.62
 if o.name.startswith('Upper eyelid sculpt'):o.data.bevel_depth*=.6
 if o.name.startswith('Closed mouth'):o.data.bevel_depth*=.65
# Small natural shoulder slope, while keeping the garment a single solid upper-body mesh.
clothpivot=Vector((0,-.032,1.30));clothrot=Matrix.Rotation(math.radians(3.5),3,'Y')
for o in bpy.data.collections['04 Knit sweater'].objects:
 if o.type=='MESH':
  for v in o.data.vertices:v.co=clothpivot+clothrot@(v.co-clothpivot)
 elif o.type=='CURVE':
  for sp in o.data.splines:
   for p in sp.points:p.co=(*(clothpivot+clothrot@(Vector(p.co[:3])-clothpivot)),1)
# Resolve photographed front hair against the actual tilted collar and cloth surface.
bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();clothverts=[];clothfaces=[]
for o in bpy.data.collections['04 Knit sweater'].objects:
 if o.type!='MESH':continue
 ev=o.evaluated_get(dg);mm=ev.to_mesh();offset=len(clothverts);clothverts.extend([ev.matrix_world@v.co for v in mm.vertices]);clothfaces.extend([tuple(offset+i for i in f.vertices) for f in mm.polygons]);ev.to_mesh_clear()
clothbvh=BVHTree.FromPolygons(clothverts,clothfaces)
for o in hair_col.objects:
 if o.type=='CURVE':pts=[p for sp in o.data.splines for p in sp.points]
 elif o.type=='MESH' and o.name=='Hair | overlapping sculpted wave depth':pts=o.data.vertices
 else:continue
 for p in pts:
  v=Vector(p.co[:3])
  if v.y<-.085 and v.z<1.415:
   hit=clothbvh.ray_cast(Vector((v.x,-.6,v.z)),Vector((0,1,0)))
   if hit[0]:v.y=min(v.y,hit[0].y-.009)
  if o.type=='CURVE':p.co=(*v,1)
  else:p.co=v
# Camera framing gives the face more space and the selfie sleeves a foreground presence.
def aim(o,pt):o.rotation_euler=(Vector(pt)-o.location).to_track_quat('-Z','Y').to_euler()
beauty=bpy.data.objects['Beauty | selfie portrait'];beauty.location=(.012,-.89,1.535);beauty.data.lens=48;aim(beauty,(-.017,-.09,1.365))
focus=bpy.data.objects['Portrait focus | eyes'];focus.location=newpivot+newrot@(Vector((0,-.15,1.545))-pivot)
beauty.data.dof.focus_object=focus;beauty.data.dof.aperture_fstop=7.1
# Plain warm geometric studio: no photographed reference background anywhere.
cafe=bpy.data.collections['06 Cafe'];cafe.hide_render=True
world=s.world.node_tree.nodes.get('Background');world.inputs[0].default_value=(.30,.275,.245,1);world.inputs[1].default_value=.32
# Bright soft portrait light, avoids printing hard shadows onto the UV skin.
bpy.data.objects['Window | large soft daylight'].data.energy=26
bpy.data.objects['Front | broad reflected fill'].data.energy=13
bpy.data.objects['Cafe | hair separation'].data.energy=9
bpy.data.objects['Cafe | hair separation'].data.color=(1,.90,.80)
for name,deg in [('front',0),('three-quarter',45),('side',90),('back',180)]:
 o=bpy.data.objects['Inspection | '+name];a=math.radians(deg);pt=Vector((-.025,-.22,1.33));o.location=pt+Vector((1.55*math.sin(a),-1.55*math.cos(a),.045));aim(o,pt);o.data.ortho_scale=1.10 if name=='three-quarter' else 1.02
facecam=bpy.data.objects['Preview | close facial inspection'];facecam.location=(0,-.7,1.54);aim(facecam,(-.035,-.10,1.535));facecam.data.ortho_scale=.32
# Scene provenance stored with actual editable resources.
s['method']='CC0 continuous anatomical mesh + masked photograph-derived frontal skin UV; photo-textured volumetric scalp with real native hair; solid cloth; isolated source iris and knit.'
s['source_scene']='source-character.blend (immutable first-pass asset)';s['reference_texture_permission']='Second pass explicitly permits genuine photographic textures.'
# Align render/viewport subdivision so measured evaluated counts also describe the final renders.
for o in bpy.data.objects:
 for mod in o.modifiers:
  if mod.type=='SUBSURF':mod.render_levels=mod.levels
# Remove unused first-pass images/materials; pack all used textures, never pack original full photo.
bpy.data.images.remove(ref)
for im in list(bpy.data.images):
 if im.users==0 and im.name!='Render Result':bpy.data.images.remove(im)
for im in bpy.data.images:
 if im.type not in ('RENDER_RESULT','COMPOSITING') and im.name not in ('Render Result','Viewer Node'):im.pack()
s.camera=beauty;s.render.resolution_x=960;s.render.resolution_y=1040;s.render.filepath=str(ROOT/'renders/beauty.png')
# Measurement from actual dependency graph. Native hair is excluded from polygon count and reported separately.
bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();counts={};mins=np.array([np.inf]*3);maxs=-mins.copy();totv=totf=control=0;native_points=native_curves=0
for o in bpy.data.objects:
 if not any(c.name.startswith(('01','02','03','04')) for c in o.users_collection):continue
 if o.type=='CURVES':
  pts=np.empty(len(o.data.points)*3,np.float32);o.data.attributes['position'].data.foreach_get('vector',pts);pts=pts.reshape(-1,3)
  counts[o.name]={'type':'native hair curves','curves':len(o.data.curves),'points':len(o.data.points),'evaluated_mesh_vertices':None}
  native_points+=len(o.data.points);native_curves+=len(o.data.curves)
 elif o.type in ('MESH','CURVE'):
  ev=o.evaluated_get(dg);me=ev.to_mesh();pts=np.array([ev.matrix_world@v.co for v in me.vertices]);nv=len(me.vertices);nf=len(me.polygons)
  counts[o.name]={'type':o.type,'control_vertices':len(o.data.vertices) if o.type=='MESH' else None,'evaluated_vertices':nv,'evaluated_polygons':nf,'curve_points':sum(len(sp.points) for sp in o.data.splines) if o.type=='CURVE' else None}
  totv+=nv;totf+=nf;control+=len(o.data.vertices) if o.type=='MESH' else 0;ev.to_mesh_clear()
 else:continue
 if len(pts):mins=np.minimum(mins,pts.min(0));maxs=np.maximum(maxs,pts.max(0))
assert totv<1000000,(totv,'geometry budget exceeded')
# Save before rendering, genuinely reopen. Every PNG then renders this same saved scene.
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'character.blend'))
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'character.blend'))
s=bpy.context.scene
views=[('beauty','Beauty | selfie portrait',960,1040),('front','Inspection | front',768,768),('three-quarter','Inspection | three-quarter',768,768),('side','Inspection | side',768,768),('back','Inspection | back',768,768)]
if PREVIEW:views=[('beauty','Beauty | selfie portrait',480,520),('face','Preview | close facial inspection',640,640)]+[(n,c,400,400) for n,c,_,_ in views[1:]]
for name,cam,w,h in views:
 s.camera=bpy.data.objects[cam];s.render.resolution_x=w;s.render.resolution_y=h;s.render.filepath=str(ROOT/('previews' if PREVIEW else 'renders')/f'{name}.png');bpy.ops.render.render(write_still=True)
s.camera=bpy.data.objects['Beauty | selfie portrait'];s.render.resolution_x=960;s.render.resolution_y=1040;s.render.filepath=str(ROOT/'renders/beauty.png')
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'character.blend'))
limitations=['Visibly CG: handcrafted likeness, not a scan or perfect reconstruction.','Single photographed face cannot determine exact skull, ear, side/back anatomy or hairstyle.','Residual broad photographic illumination remains in the frontal skin albedo despite attenuation.','Face UV projection is hand-landmark warped and softly masked; profile blends into inferred warm procedural skin.','The tubular hanging fibers, sculpted wave cores and native crown/fringe are actual geometry, but remain more organized than natural hair.','Cloth volume and selfie pose are approximate; mirrored source knit tile repeats and relief is shader bump.','No production facial rig. Neck/face pose is baked; eyes and brows remain grouped to the head pose.','Sleeve ends are closed cropped cloth sections; hands and lower body are omitted.']
report={'blender_version':bpy.app.version_string,'render_engine':s.render.engine,'render_device':s.cycles.device,'threads':s.render.threads,'samples':s.cycles.samples,'denoising':s.cycles.use_denoising,'compositing_enabled':s.render.use_compositing,'sequencer_enabled':s.render.use_sequencer,'final_saved_scene_reopened_before_rendering':True,'image_access':{'reference':'Direct visual decoded image through view_image; reference pixels also decoded by Blender.','first_pass':'Missing copied render files; actually rendered and directly viewed source beauty, side, and close face before changes.','final_views_visually_inspected':False},'source_scene':'source-character.blend','source_base':{'file':'input/upper-body.obj','license':'CC0 MakeHuman hm08','commit':'a8bc2d54ff0ac92e78ff71431b1023eda42bf482'},'character_bounds_m':{'min':mins.tolist(),'max':maxs.tolist()},'geometry':{'mesh_control_vertices':control,'evaluated_vertices_excluding_native_hair':totv,'evaluated_polygons_excluding_native_hair':totf,'native_hair_curves':native_curves,'native_hair_points':native_points},'objects':counts,'textures':[{'name':im.name,'size':list(im.size),'packed':bool(im.packed_file),'file':str(Path(im.filepath).relative_to(ROOT)) if im.filepath.startswith(str(ROOT)) else im.filepath} for im in bpy.data.images if im.type not in ('RENDER_RESULT','COMPOSITING') and im.name not in ('Render Result','Viewer Node')],'texture_methods':{'face':'1024x1024 atlas from supplied photograph; TPS anatomical x/z UV landmark remap; hair and eye pixel exclusion; alpha feather; per-vertex front-depth support; attenuated low-frequency illumination; unseen surfaces constant warm skin plus procedural pores.','iris':'256x256 isolated reference left iris, source white glint suppressed; planar UV on curved brown iris under real refractive cornea.','sweater':'512x512 mirrored source sweater crop, luminance-normalized; seamless triplanar Object mapping at 17 repeats/m; derived non-color yarn relief. Editable cylindrical cloth UV also provided.','hair':'1024x1024 masked photographic crown projection onto a fitted cranial 3D scalp using anatomical x/z UV, with front-only support; inferred sides/back use a 512x512 supplied-hair crop tile on cylindrical UV. The hanging lengths retain repaired editable tubular curve fibers and sculpted 3D wave cores from the first pass. Surface-following crown and fringe use native hair curves with melanin Principled Hair BSDF.'},'reference_landmarks_pixels':[{'model_xz_m':[x,z],'photo_topdown_xy':[u,v]} for x,z,u,v in anchors],'reference_reprojection_error_pixels':None,'likeness_metric':None,'rendered_filenames':[str(Path(suffix)/f'{n}.png') for n,_,_,_ in views for suffix in ['previews' if PREVIEW else 'renders']],'limitations':limitations}
(ROOT/('preview-report.json' if PREVIEW else 'scene-report.json')).write_text(json.dumps(report,indent=2))
print('FINISHED',totv,'evaluated polygon vertices;',native_curves,'native hair curves;',native_points,'hair points')
