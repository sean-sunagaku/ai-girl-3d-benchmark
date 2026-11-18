"""Measure the reopened scene and identify its five actual Blender PNG files."""
import bpy
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

root=Path(bpy.data.filepath).parent
scene=bpy.context.scene
graph=bpy.context.evaluated_depsgraph_get()
character=bpy.data.collections['Character']
world_points=[];mesh_counts=[];curve_counts=[]
for ob in sorted(character.all_objects,key=lambda o:o.name):
    ev=ob.evaluated_get(graph);transform=np.asarray(ev.matrix_world,dtype=float)
    if ob.type=='MESH':
        me=ev.to_mesh();pts=np.empty(len(me.vertices)*3,dtype=np.float32);me.vertices.foreach_get('co',pts)
        mesh_counts.append({'name':ob.name,'base_vertices':len(ob.data.vertices),'evaluated_vertices':len(me.vertices),'evaluated_polygons':len(me.polygons)})
        ev.to_mesh_clear()
    elif ob.type=='CURVES':
        cv=ev.data;pts=np.empty(len(cv.points)*3,dtype=np.float32);cv.position_data.foreach_get('vector',pts)
        curve_counts.append({'name':ob.name,'strands':len(cv.curves),'control_points':len(cv.points)})
    else: continue
    world_points.append(pts.reshape(-1,3)@transform[:3,:3].T+transform[:3,3])
points=np.vstack(world_points)
projections=[]
for view in ('front','three-quarter','side','back'):
    cam=bpy.data.objects['diagnostic • '+view];m=cam.calc_matrix_camera(graph,x=768,y=768)@cam.matrix_world.inverted();m=np.asarray(m)
    clip=points@m[:,:3].T+m[:,3];uv=.5+.5*clip[:,:2]/clip[:,3,None];a=uv.min(0);b=uv.max(0)
    projections.append({'view':view,'lower_left_uv':a.round(6).tolist(),'upper_right_uv':b.round(6).tolist(),'fully_in_frame':bool(np.all(a>0) and np.all(b<1))})

def png_record(view):
    p=root/'renders'/(view+'.png');data=p.read_bytes();at=8;props={};w=h=None
    assert data.startswith(b'\x89PNG\r\n\x1a\n')
    while at+12<=len(data):
        size=struct.unpack_from('>I',data,at)[0];kind=data[at+4:at+8];chunk=data[at+8:at+8+size];at+=12+size
        if kind==b'IHDR':w,h=struct.unpack_from('>II',chunk)
        if kind==b'tEXt':
            k,v=chunk.split(b'\0',1);props[k.decode('utf8','replace')]=v.decode('utf8','replace')
        if kind==b'IEND':break
    expected=(960,1040) if view=='beauty' else (768,768)
    assert (w,h)==expected,(view,w,h)
    samples=props.get('cycles.ViewLayer.samples');assert samples=='32',(view,samples)
    assert 'File' not in props
    return {'file':'renders/'+p.name,'width':w,'height':h,'camera':props.get('Camera'),'cycles_samples_metadata':int(samples),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

check=json.loads((root/'validation.json').read_text())
assert check['source_unchanged'] and check['reopened_successfully'] and all(check['limits'].values())
assert hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest()==check['source_sha256_after']
assert all(p['fully_in_frame'] for p in projections), projections
scalar=lambda key:sum(o[key] for o in mesh_counts)
records=[png_record(v) for v in ('beauty','front','three-quarter','side','back')]
report={'title':'Window seat','scene':'character.blend','source_sha256':check['source_sha256_after'],'blender_version':bpy.app.version_string,'render_engine':scene.render.engine,'device':scene.cycles.device,'threads':scene.render.threads,'samples':scene.cycles.samples,'adaptive_sampling':scene.cycles.use_adaptive_sampling,'denoiser':scene.cycles.denoiser,'denoiser_gpu':scene.cycles.denoising_use_gpu,'saved_beauty_camera':scene.camera.name,'units':'metres, Z up; unposed face toward -Y','character_bounds':{'min':points.min(0).round(7).tolist(),'max':points.max(0).round(7).tolist(),'method':'Evaluated mesh vertices and native curve centreline control points in world space; microscopic fibre radii are excluded.'},'character_mesh_totals':{'base_vertices':scalar('base_vertices'),'evaluated_vertices':scalar('evaluated_vertices'),'evaluated_polygons':scalar('evaluated_polygons')},'character_native_curves':{'strands':sum(c['strands'] for c in curve_counts),'control_points':sum(c['control_points'] for c in curve_counts)},'whole_scene_evaluated_mesh_vertices':check['total_evaluated_mesh_vertices'],'character_meshes':mesh_counts,'character_curve_objects':curve_counts,'diagnostic_camera_coverage':projections,'generated_textures':[{'name':'Stockinette • four wales × four courses','dimensions':[384,384],'packed':bool(bpy.data.images['Stockinette • four wales × four courses'].packed_file),'channels':{'R':'Overlapping yarn relief','G':'Marl and yarn values','B':'Ply phase'},'source':'Deterministic distance fields around overlapping curved yarn legs in a four-stitch by four-row repeat.'},{'name':'Complexion, hydration and pigment region','dimensions':None,'source':'Sculpt-aligned paintable attributes control circulation, hydration and vermilion. The shader evaluates pores, uneven pigment, sparse melanin flecks and directional lip furrows.'},{'name':'Stroma','dimensions':None,'source':'Radial mesh pigment attributes with collarette, limbal falloff and azimuthal variation.'}],'external_image_textures':[],'rendered_files':records,'independent_technical_inspection':'validation.json','artist_observations':{'reference_access':'The decoded input photograph was viewed directly. It is not loaded into Blender.','limitations':['The single photograph gives no depth measurements. Facial identity, ear angles, profile, back, hands and garment construction are interpretations.','The fringe and long waves form deliberately grouped clumps; individual placement and translucency differ from the photograph. The direction change at the nape is visible from behind.','The skin, irises and brows are procedural. Pores and very fine hair detail soften at portrait scale under denoising.','The garment is shaped for this pose. The knit relief repeats, the bust terminates at the sweater hem and the scene has no animation rig or cloth simulation.','The café furniture, plant and ceramics are inferred from the visible setting.']}}
groom_names={o.name for o in bpy.data.collections['02 • Chestnut groom'].all_objects}
report['head_hair_strands']=sum(c['strands'] for c in curve_counts if c['name'] in groom_names)
report['character_mesh_vertices_plus_curve_control_points']=report['character_mesh_totals']['evaluated_vertices']+report['character_native_curves']['control_points']
report['scene_image_count_excluding_render_result']=len(check['images'])
compositor=getattr(scene,'compositing_node_group',None)
report['compositor_group']=compositor.name if compositor else None
(root/'scene-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('Character geometry',report['character_mesh_totals'],report['character_native_curves']);print('All four inspection cameras contain the character; five PNGs verified.')
