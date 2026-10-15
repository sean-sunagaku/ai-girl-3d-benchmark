"""Independently reopen the final scene and measure/check deliverables. No scene edits.
Run after build.py with Blender --background --threads 3 --python verify.py.
PORTRAIT_VISUAL_REVIEWED=1 records a completed human/assistant direct image review.
"""
import bpy, json, os, hashlib
from pathlib import Path
from mathutils import Vector
R=Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(R/'character.blend'))
s=bpy.context.scene;dg=bpy.context.evaluated_depsgraph_get()
assert bpy.app.version_string=='5.1.1'
assert s.render.engine=='CYCLES' and s.cycles.device=='CPU'
assert s.render.threads_mode=='FIXED' and s.render.threads==3
assert s.cycles.samples==32 and s.cycles.use_denoising
assert s.camera.name=='Beauty | selfie portrait'
assert (s.render.resolution_x,s.render.resolution_y)==(960,1040)
assert not s.render.use_compositing and not s.render.use_sequencer and not s.render.film_transparent
assert 'Continuous sculpt | face ears neck' in bpy.data.objects
per={};nvertices=0;nfaces=0;haircurves=0;hairpoints=0;points=[]
for o in bpy.data.objects:
 if not any(c.name.startswith(('01','02','03','04')) for c in o.users_collection):continue
 if o.type=='CURVES':
  haircurves+=len(o.data.curves);hairpoints+=len(o.data.points)
  for p in o.data.points:points.append(o.matrix_world@p.position)
  per[o.name]={'type':'native hair','curves':len(o.data.curves),'points':len(o.data.points)}
 elif o.type in ('MESH','CURVE'):
  for mod in o.modifiers:
   if mod.type=='SUBSURF':assert mod.levels==mod.render_levels
  ev=o.evaluated_get(dg);me=ev.to_mesh();nv=len(me.vertices);nf=len(me.polygons);nvertices+=nv;nfaces+=nf
  points.extend(ev.matrix_world@v.co for v in me.vertices)
  per[o.name]={'type':o.type,'evaluated_vertices':nv,'evaluated_polygons':nf,'control_vertices':len(o.data.vertices) if o.type=='MESH' else None,'curve_splines':len(o.data.splines) if o.type=='CURVE' else None,'curve_points':sum(len(sp.points) for sp in o.data.splines) if o.type=='CURVE' else None}
  ev.to_mesh_clear()
assert nvertices<1000000,nvertices
sizes={}
for name in ('beauty','front','three-quarter','side','back'):
 path=R/'renders'/f'{name}.png';assert path.exists()
 im=bpy.data.images.load(str(path),check_existing=False)
 size=list(im.size);expected=[960,1040] if name=='beauty' else [768,768];assert size==expected,(name,size)
 sizes[f'renders/{name}.png']={'size':size,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 bpy.data.images.remove(im)
images=[]
for im in bpy.data.images:
 if im.type in ('RENDER_RESULT','COMPOSITING') or im.name in ('Render Result','Viewer Node'):continue
 assert im.packed_file,im.name
 assert max(im.size)<=2048,(im.name,list(im.size))
 images.append({'name':im.name,'size':list(im.size),'packed':True,'colorspace':im.colorspace_settings.name})
assert not any('reference.png' in im.filepath for im in bpy.data.images)
folderbytes=sum(p.stat().st_size for p in R.rglob('*') if p.is_file())
assert folderbytes<1500000000
result={'saved_scene_independently_reopened':True,'engine':s.render.engine,'device':s.cycles.device,'threads':s.render.threads,'samples':s.cycles.samples,'denoising':s.cycles.use_denoising,'saved_camera':s.camera.name,'compositor_enabled':s.render.use_compositing,'geometry':{'evaluated_vertices_excluding_native_hair':nvertices,'evaluated_polygons_excluding_native_hair':nfaces,'native_hair_curves':haircurves,'native_hair_points':hairpoints},'bounds_m':{'min':[min(p[k] for p in points) for k in range(3)],'max':[max(p[k] for p in points) for k in range(3)]},'per_object_counts':per,'packed_images':images,'render_sizes_and_hashes':sizes,'folder_bytes':folderbytes,'stable_source_sha256':hashlib.sha256((R/'source-character.blend').read_bytes()).hexdigest(),'final_views_directly_visually_inspected':os.environ.get('PORTRAIT_VISUAL_REVIEWED')=='1'}
(R/'verification.json').write_text(json.dumps(result,indent=2))
report=json.loads((R/'scene-report.json').read_text());report['independent_verification']=result
report['image_access']['final_views_visually_inspected']=result['final_views_directly_visually_inspected']
(R/'scene-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:result[k] for k in ('saved_scene_independently_reopened','geometry','folder_bytes','saved_camera','final_views_directly_visually_inspected')},indent=2))
