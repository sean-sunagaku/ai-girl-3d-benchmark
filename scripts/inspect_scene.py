"""Inspect a newly authorized Blender submission; optionally export its GLB.

Run on an already saved scene, without saving changes back to that scene:
Blender --background character.blend --threads 3 --python inspect_scene.py -- \
  --output validation.json [--glb character.glb --include collection-name]

No existing published submission is inspected automatically. This command is
for explicitly requested new runs or technical inspections only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--glb', type=Path)
parser.add_argument('--include', action='append', default=[], help='Exact object or collection name; repeat as needed')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
scene_path = Path(bpy.data.filepath)
source_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
objects = []
for obj in scene.objects:
    item = {'name': obj.name, 'type': obj.type, 'hidden_render': obj.hide_render,
            'collections': [c.name for c in obj.users_collection]}
    if obj.type == 'MESH':
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        item.update(base_vertices=len(obj.data.vertices), evaluated_vertices=len(mesh.vertices),
                    evaluated_polygons=len(mesh.polygons))
        evaluated.to_mesh_clear()
    if obj.type in ('MESH','CURVE','CURVES'):
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        item['bounds'] = [[min(v[i] for v in corners) for i in range(3)],
                          [max(v[i] for v in corners) for i in range(3)]]
    objects.append(item)
images = [{'name': img.name, 'size': list(img.size), 'packed': bool(img.packed_file),
           'source': img.source, 'file_exists': bool(img.packed_file) or not img.filepath or Path(bpy.path.abspath(img.filepath)).is_file()}
          for img in bpy.data.images if img.type != 'RENDER_RESULT']
report = {
    'scene': scene_path.name, 'source_sha256_before': source_hash,
    'reopened_successfully': True, 'blender_version': bpy.app.version_string,
    'render_engine': scene.render.engine,
    'cycles_device': scene.cycles.device if scene.render.engine == 'CYCLES' else None,
    'threads_mode': scene.render.threads_mode, 'threads': scene.render.threads,
    'samples': scene.cycles.samples if scene.render.engine == 'CYCLES' else None,
    'denoising': scene.cycles.use_denoising if scene.render.engine == 'CYCLES' else None,
    'camera': scene.camera.name if scene.camera else None,
    'cameras': [{'name':o.name,'type':o.data.type,'ortho_scale':o.data.ortho_scale if o.data.type=='ORTHO' else None} for o in scene.objects if o.type=='CAMERA'],
    'objects': objects, 'images': images,
    'total_evaluated_mesh_vertices': sum(o.get('evaluated_vertices',0) for o in objects),
    'native_hair_objects': [o.name for o in scene.objects if o.type == 'CURVES'],
    'limits': {'all_images_at_most_2048': all(max(img['size'],default=0) <= 2048 for img in images),
               'resources_resolve': all(img['file_exists'] for img in images)},
}
if args.glb:
    if not args.include:
        raise ValueError('GLB export requires explicit character objects or collections; do not export background accidentally')
    selected=set()
    for name in args.include:
        if name in bpy.data.collections:
            selected.update(bpy.data.collections[name].all_objects)
        elif name in bpy.data.objects:
            selected.add(bpy.data.objects[name])
        else:
            raise ValueError('Requested object/collection absent: '+name)
    selected={o for o in selected if not o.hide_render and o.type in ('MESH','CURVE','CURVES','SURFACE','FONT','ARMATURE','EMPTY')}
    if not selected:
        raise ValueError('No visible character geometry selected')
    actual_points=[]
    for obj in selected:
        if obj.type in ('MESH','CURVE','CURVES','SURFACE','FONT'):
            evaluated=obj.evaluated_get(depsgraph)
            mesh=evaluated.to_mesh()
            if mesh:
                actual_points.extend(evaluated.matrix_world@v.co for v in mesh.vertices)
                evaluated.to_mesh_clear()
    report['actual_evaluated_character_bounds']={'min':[min(v[i] for v in actual_points) for i in range(3)],'max':[max(v[i] for v in actual_points) for i in range(3)],'method':'World coordinates of evaluated mesh vertices, including converted curve geometry.'}
    projected=[]
    for cam in scene.objects:
        if cam.type!='CAMERA' or 'diagnostic' not in cam.name:
            continue
        matrix=cam.calc_matrix_camera(depsgraph,x=768,y=768)@cam.matrix_world.inverted()
        minimum=[float('inf'),float('inf')]; maximum=[float('-inf'),float('-inf')]
        for point in actual_points:
            clip=matrix@point.to_4d()
            uv=[(clip.x/clip.w+1)*.5,(clip.y/clip.w+1)*.5]
            for i in range(2):minimum[i]=min(minimum[i],uv[i]);maximum[i]=max(maximum[i],uv[i])
        projected.append({'camera':cam.name,'normalized_min':minimum,'normalized_max':maximum,
                          'all_character_vertices_in_frame':all(v>=-1e-6 for v in minimum) and all(v<=1+1e-6 for v in maximum)})
    report['diagnostic_projection_bounds']=projected
    bpy.ops.object.select_all(action='DESELECT')
    names=[o.name for o in selected]
    converted=[]
    for obj in list(selected):
        if obj.type in ('CURVE','CURVES','SURFACE','FONT'):
            bpy.ops.object.select_all(action='DESELECT')
            obj.hide_set(False)
            obj.select_set(True)
            bpy.context.view_layer.objects.active=obj
            bpy.ops.object.convert(target='MESH')
            converted.append(obj.name)
    bpy.ops.object.select_all(action='DESELECT')
    for name in names:
        obj=bpy.data.objects.get(name)
        if obj:
            obj.hide_set(False)
            obj.select_set(True)
    # glTF cannot encode a procedural color ramp driven by Blender Noise.
    # In this export-only scene, preserve the source material's authored base
    # swatch instead of letting the exporter silently substitute white. Keep
    # supported vertex-color/image links intact. Never save this back to .blend.
    material_fallbacks=[]
    material_copies={}
    for name in names:
        obj=bpy.data.objects.get(name)
        if not obj or not hasattr(obj.data,'materials'):
            continue
        for slot in obj.material_slots:
            original=slot.material
            if not original or not original.use_nodes:
                continue
            bsdf=original.node_tree.nodes.get('Principled BSDF')
            if not bsdf or not bsdf.inputs['Base Color'].is_linked:
                continue
            links=list(bsdf.inputs['Base Color'].links)
            if all(link.from_node.bl_idname in ('ShaderNodeVertexColor','ShaderNodeTexImage','ShaderNodeAttribute') for link in links):
                continue
            if original.name not in material_copies:
                copy=original.copy()
                copy.name=original.name+' [GLB base swatch]'
                base=copy.node_tree.nodes.get('Principled BSDF').inputs['Base Color']
                for link in list(base.links):copy.node_tree.links.remove(link)
                base.default_value=original.diffuse_color
                material_copies[original.name]=copy
                material_fallbacks.append({'source_material':original.name,'base_color_linear':list(original.diffuse_color),
                    'reason':'Unsupported procedural base-color graph; retained the material author\'s own swatch for GLB only.'})
            slot.material=material_copies[original.name]
    args.glb.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.glb.resolve()),export_format='GLB',
                              use_selection=True,export_apply=True,export_animations=False,
                              export_cameras=False,export_lights=False,export_yup=True)
    report['glb_export']={'path':args.glb.name,'selected_objects':names,
                          'curves_converted_for_export_only':converted,
                          'procedural_material_fallbacks':material_fallbacks,
                          'bytes':args.glb.stat().st_size,
                          'sha256':hashlib.sha256(args.glb.read_bytes()).hexdigest(),
                          'note':'Procedural Blender shaders and lighting can differ in GLB. The source .blend was not saved or modified.'}
report['source_sha256_after']=hashlib.sha256(scene_path.read_bytes()).hexdigest()
report['source_unchanged']=report['source_sha256_after']==source_hash
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2))
print('INSPECTION',json.dumps({k:report[k] for k in ('reopened_successfully','render_engine','total_evaluated_mesh_vertices','source_unchanged','limits')}))
