"""Read-only scene and delivered-render verification in installed Blender."""
import bpy, json, struct, os
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parent
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'character.blend'))
sc=bpy.context.scene
assert bpy.app.version_string=='5.1.1'
assert sc.render.engine=='CYCLES' and sc.cycles.device=='CPU'
assert sc.cycles.samples==32 and sc.render.threads==3
assert sc.camera.name=='Beauty | selfie portrait'
assert not bpy.data.collections['06 Cafe'].hide_render
assert (sc.render.resolution_x,sc.render.resolution_y)==(960,1040)
images=[]
for im in bpy.data.images:
 if im.name.startswith('Procedural knit'):
  assert im.packed_file is not None
  assert max(im.size)<=2048
  images.append({'name':im.name,'size':list(im.size),'packed':True})
assert images
files={}
for name in ('beauty','front','three-quarter','side','back'):
 path=ROOT/'renders'/f'{name}.png'
 with path.open('rb') as f:
  header=f.read(24)
 assert header[:8]==b'\x89PNG\r\n\x1a\n'
 dims=struct.unpack('>II',header[16:24]);expect=(960,1040) if name=='beauty' else (768,768)
 assert dims==expect,(name,dims)
 files[str(path.relative_to(ROOT))]={'size':list(dims),'bytes':path.stat().st_size}
char=[o for o in sc.objects if o.type in ('MESH','CURVE') and any(c.name.startswith(('01','02','03','04')) for c in o.users_collection)]
dg=bpy.context.evaluated_depsgraph_get();counts={};nv=0;nf=0;bounds_min=[float('inf')]*3;bounds_max=[float('-inf')]*3
for o in char:
 ev=o.evaluated_get(dg);me=ev.to_mesh();nv+=len(me.vertices);nf+=len(me.polygons)
 counts[o.name]={'evaluated_vertices':len(me.vertices),'evaluated_polygons':len(me.polygons),'mesh_control_vertices':len(o.data.vertices) if o.type=='MESH' else None,'curve_control_points':sum(len(sp.points)+len(sp.bezier_points) for sp in o.data.splines) if o.type=='CURVE' else None,'curve_splines':len(o.data.splines) if o.type=='CURVE' else None}
 for v in me.vertices:
  p=ev.matrix_world@v.co
  for i in range(3):bounds_min[i]=min(bounds_min[i],p[i]);bounds_max[i]=max(bounds_max[i],p[i])
 ev.to_mesh_clear()
assert nv<1000000,nv
folder_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
assert folder_bytes<1500000000
result={'final_saved_scene_reopened':True,'blender_version':bpy.app.version_string,'engine':sc.render.engine,'device':sc.cycles.device,'samples':sc.cycles.samples,'threads':sc.render.threads,'beauty_camera_restored':True,'cafe_restored':True,'packed_images':images,'render_files':files,'evaluated_vertices':nv,'evaluated_polygons':nf,'character_bounds_m':{'min':bounds_min,'max':bounds_max},'per_object_counts':counts,'folder_bytes':folder_bytes}
(ROOT/'verification.json').write_text(json.dumps(result,indent=2))
# Append actual verification measurements, not estimates, to the requested scene report.
p=ROOT/'scene-report.json';report=json.loads(p.read_text());report['final_saved_scene_reopened_and_verified']=True;report['packed_images']=images;report['render_sizes_and_bytes']=files;report['per_object_counts']=counts;report['evaluated_vertices']=nv;report['evaluated_polygons']=nf;report['character_bounds_m']=result['character_bounds_m'];report['folder_bytes_at_verification']=folder_bytes;report['reference_reprojection_error_pixels']=None;report['likeness_metric']=None;report['source_base_mesh']={'file':'input/upper-body.obj','vertices':6093,'faces':6044,'license':'CC0 MakeHuman hm08'};p.write_text(json.dumps(report,indent=2))
print('VERIFIED',nv,'vertices,',nf,'polygons;',folder_bytes,'bytes; packed procedural images;',files)
