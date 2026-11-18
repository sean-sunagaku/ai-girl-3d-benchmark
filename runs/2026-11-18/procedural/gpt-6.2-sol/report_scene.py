"""Measure the reopened portrait and record the completed Blender render set."""
import hashlib
import json
from pathlib import Path
import struct

import bpy
import numpy as np

ROOT=Path(__file__).resolve().parent
VIEWS={'beauty':(960,1040),'front':(768,768),'three-quarter':(768,768),
       'side':(768,768),'back':(768,768)}


def xyz(data, key):
    values=np.empty(len(data)*3,dtype=np.float32)
    data.foreach_get(key,values)
    return values.reshape(-1,3)


def world_points(points,matrix):
    m=np.asarray(matrix)
    return points@np.transpose(m[:3,:3])+m[:3,3]


def render_file(label,expected,angle):
    path=ROOT/'renders'/(label+'.png')
    with path.open('rb') as f:
        magic=f.read(16)
        dimensions=struct.unpack('>II',f.read(8))
    if magic!=b'\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR' or dimensions!=expected:
        raise ValueError(f'{path.name} does not match its required PNG format and size')
    return {'file':str(path.relative_to(ROOT)),'dimensions_px':dimensions,
            'angle_degrees':angle,'source':'Raw Blender render from reopened character.blend',
            'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    source=Path(bpy.data.filepath)
    if source.name!='character.blend':
        raise ValueError('Open character.blend to measure the final scene')
    dg=bpy.context.evaluated_depsgraph_get()
    scene=bpy.context.scene
    char=bpy.data.collections['Character']
    meshes,curves,point_cloud,radii=[],[],[],[]
    all_vertices=0
    for obj in scene.objects:
        if obj.type=='MESH':
            evaluated=obj.evaluated_get(dg)
            mesh=evaluated.to_mesh()
            all_vertices+=len(mesh.vertices)
            if obj.name in char.all_objects:
                meshes.append({'name':obj.name,'base_vertices':len(obj.data.vertices),
                               'base_faces':len(obj.data.polygons),
                               'evaluated_vertices':len(mesh.vertices),
                               'evaluated_faces':len(mesh.polygons)})
                a=world_points(xyz(mesh.vertices,'co'),evaluated.matrix_world)
                point_cloud.append(a)
                radii.append(np.zeros(len(a),dtype=np.float32))
            evaluated.to_mesh_clear()
        elif obj.type=='CURVES' and obj.name in char.all_objects:
            data=obj.data
            positions=world_points(xyz(data.position_data,'vector'),obj.matrix_world)
            radius=np.zeros(len(positions),dtype=np.float32)
            data.attributes['radius'].data.foreach_get('value',radius)
            point_cloud.append(positions)
            radii.append(radius)
            curves.append({'name':obj.name,'strands':len(data.curves),'control_points':len(positions)})
    p=np.concatenate(point_cloud)
    r=np.concatenate(radii)
    bounds={'min':np.min(p-r[:,None],axis=0).tolist(),
            'max':np.max(p+r[:,None],axis=0).tolist(),
            'method':'Actual world-space evaluated mesh positions and native strand control points with strand radii.'}
    framed={}
    for label in VIEWS.keys()-{'beauty'}:
        camera=bpy.data.objects['diagnostic • '+label]
        proj=camera.calc_matrix_camera(dg,x=768,y=768,scale_x=1,scale_y=1)
        coords=np.asarray(proj @ camera.matrix_world.inverted())
        uv=world_points(p,coords)[:,:2]*.5+.5
        margin=(r/camera.data.ortho_scale)[:,None]
        low=np.min(uv-margin,axis=0)
        high=np.max(uv+margin,axis=0)
        framed[label]={'normalized_min':low.tolist(),'normalized_max':high.tolist(),
                       'full_asset_in_frame':bool(np.all(low>=0) and np.all(high<=1))}
    result={
        'scene':source.name,'scene_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'reopened_successfully':True,'blender_version':bpy.app.version_string,
        'render_engine':scene.render.engine,'cycles_device':scene.cycles.device,
        'threads':scene.render.threads,'samples':scene.cycles.samples,
        'denoising':scene.cycles.use_denoising,'saved_camera':scene.camera.name,
        'reference':{'decoded_and_viewed':True,'size_px':[1205,1306],
                     'render_scene_usage':'None. The photograph was used for visual comparison only.',
                     'file':'input/reference.png'},
        'character_bounds_m':bounds,
        'diagnostic_framing':dict(sorted(framed.items())),
        'character_meshes':sorted(meshes,key=lambda item:item['name']),
        'character_curves':sorted(curves,key=lambda item:item['name']),
        'totals':{
            'character_base_mesh_vertices':sum(m['base_vertices'] for m in meshes),
            'character_evaluated_mesh_vertices':sum(m['evaluated_vertices'] for m in meshes),
            'scene_evaluated_mesh_vertices_including_cafe':all_vertices,
            'character_native_strands':sum(a['strands'] for a in curves),
            'character_native_strand_control_points':sum(a['control_points'] for a in curves),
            'hair_tessellated_vertices':None,
        },
        'packed_images':[{'name':img.name,'size_px':list(img.size),'packed':bool(img.packed_file)}
                         for img in bpy.data.images if img.type!='RENDER_RESULT'],
        'generated_textures_and_attributes':[
            'Periodic yarn relief tile traced from looping stitch paths. Knit color and fine fiber noise are Blender nodes.',
            'Skin and lip masks, eye color and corneal gloss are authored vertex attributes.',
            'Skin pores and pigment flecks, wood grain, and hair grooves are procedural Blender shaders.',
        ],
        'renders':[render_file(label,size,angle) for (label,size),angle in zip(VIEWS.items(),[None,0,45,90,180])],
        'source_asset':{'name':'MakeHuman hm08','license':'CC0','prepared_vertices':6093,
                        'prepared_faces':6044,'source':'input/ASSET-LICENSE.md'},
        'limitations':[
            'The face is an artistic likeness from a single image. Soft tissue shape, head profile, ears, and expression are approximate. The skin and hairline read as CG in the close-up.',
            'The back of the head, rear and sides of the sweater, sleeve and hand pose, and hair arrangement away from the camera are inferred.',
            'The sweater uses an authored drape and shader-based knit relief. Its presentation cut at the lower torso ends at the welt.',
            'The portrait is a fixed pose. The editable strand groups use shaped support meshes. A narrow area of scalp is visible between swept temple locks in profile.',
            'The café is a reconstructed suggestion using geometric props and simulated window lighting.',
        ],
    }
    (ROOT/'scene-report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'totals':result['totals'],'camera':result['saved_camera'],
                      'full_diagnostic_framing':all(v['full_asset_in_frame'] for v in framed.values()),
                      'packed_images':result['packed_images']},ensure_ascii=False))


if __name__=='__main__':
    main()
