"""Measure the saved scene and report reproducible deliverables (run after rendering)."""
import bpy, os, json, struct
import numpy as np
from mathutils import Vector
root=os.path.dirname(os.path.abspath(__file__)); os.chdir(root)
s=bpy.context.scene; bpy.context.view_layer.update(); dg=bpy.context.evaluated_depsgraph_get()
baseverts=0; basefaces=0; curvepoints=0; splines=0; evalverts=0; evalfaces=0; objects=[]
lo=np.array([np.inf]*3); hi=-lo
for o in bpy.data.collections['Character'].all_objects:
    if o.type not in ('MESH','CURVE'): continue
    objects.append(o.name)
    if o.type=='MESH': baseverts+=len(o.data.vertices); basefaces+=len(o.data.polygons)
    else:
        splines+=len(o.data.splines); curvepoints+=sum(len(sp.points)+len(sp.bezier_points) for sp in o.data.splines)
    e=o.evaluated_get(dg); me=e.to_mesh()
    if me is not None:
        n=len(me.vertices); evalverts+=n; evalfaces+=len(me.polygons)
        if n:
            coords=np.empty(n*3,dtype=np.float32); me.vertices.foreach_get('co',coords); coords=coords.reshape((-1,3))
            mat=np.array(e.matrix_world,dtype=float); coords=coords@mat[:3,:3].T+mat[:3,3]
            lo=np.minimum(lo,coords.min(axis=0)); hi=np.maximum(hi,coords.max(axis=0))
        e.to_mesh_clear()

def png_size(path):
    if not os.path.exists(path): return None
    with open(path,'rb') as f: data=f.read(24)
    return list(struct.unpack('>II',data[16:24])) if data[:8]==b'\x89PNG\r\n\x1a\n' else None
files=['renders/beauty.png','renders/front.png','renders/three-quarter.png','renders/side.png','renders/back.png']
result={
 'blender_version':bpy.app.version_string,
 'blender_build_hash':bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash,bytes) else str(bpy.app.build_hash),
 'render_engine':s.render.engine,'render_device':s.cycles.device,'samples':s.cycles.samples,'denoising':s.cycles.use_denoising,'threads':s.render.threads,
 'beauty_camera':s.camera.name,'important_objects':objects,
 'character_bounds_meters':{'min':list(np.round(lo,6)),'max':list(np.round(hi,6))},
 'character_geometry':{'base_mesh_vertices':baseverts,'base_mesh_faces':basefaces,'curve_splines':splines,'curve_control_points':curvepoints,'evaluated_vertices':evalverts,'evaluated_faces':evalfaces},
 'packed_images':[{'name':im.name,'size':list(im.size),'packed':bool(im.packed_file)} for im in bpy.data.images if im.name=='Generated knit relief 512'],
 'generated_texture_descriptions':['512 x 512 original repeatable procedural knitted yarn relief packed in the blend; used for sweater colour and bump','procedural noise bump for skin micro-roughness and hair volume; no photographic character texture'],
 'renders':[{'file':f,'dimensions':png_size(f)} for f in files],
 'remaining_limitations':[
  'The face, expression, skin, lashes and eye surfaces are procedural approximations and remain visibly stylized; a close photorealistic likeness was not achieved.',
  'Hair has thousands of editable strands but its broad underlying locks are simplified and straighter than the photograph; local overlap and styling artifacts remain.',
  'The ears, neck, profile, back and hidden clothing are inferred from the supplied base and a single front photograph.',
  'Sweater folds and knit are procedural rather than a cloth simulation; the forward sleeves terminate in covered caps with no hands.',
  'The cafe is a deliberately simplified 3D suggestion. No identity, hair dynamics, production rig, or unseen lower body is reconstructed.',
  'The cornea is an approximate curved transparent/glossy shell rather than a volumetric refractive anatomical model.'
 ]}
with open('scene-report.json','w') as f: json.dump(result,f,indent=2,ensure_ascii=False); f.write('\n')
print(json.dumps({'geometry':result['character_geometry'],'bounds':result['character_bounds_meters'],'packed':result['packed_images']},indent=2))
