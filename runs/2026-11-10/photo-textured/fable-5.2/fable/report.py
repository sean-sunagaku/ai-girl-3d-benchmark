"""scene-report.json: measured facts about the saved scene (written by build.py after re-opening the file)."""
import json
import os
import bpy
from mathutils import Vector


def _bounds(objs, dg):
    """World-space bounds from real evaluated geometry (mesh vertices and authored curve points)."""
    import numpy as np
    lo = np.full(3, 1e9); hi = np.full(3, -1e9)
    for ob in objs:
        ev = ob.evaluated_get(dg)
        pts = None
        if ob.type == 'MESH':
            me = ev.to_mesh()
            if len(me.vertices):
                pts = np.empty(len(me.vertices) * 3, dtype=np.float32); me.vertices.foreach_get('co', pts)
            ev.to_mesh_clear()
        elif ob.type == 'CURVES' and len(ob.data.points):
            pts = np.empty(len(ob.data.points) * 3, dtype=np.float32); ob.data.position_data.foreach_get('vector', pts)
        if pts is None:
            continue
        pts = pts.reshape(-1, 3)
        M = np.array(ev.matrix_world)
        w = pts @ M[:3, :3].T + M[:3, 3]
        lo = np.minimum(lo, w.min(0)); hi = np.maximum(hi, w.max(0))
    return [round(float(v), 4) for v in lo], [round(float(v), 4) for v in hi]


def write(path, extra):
    sc = bpy.context.scene
    # count geometry at render subdivision levels (temporarily, nothing is saved)
    saved = {}
    for ob in sc.objects:
        for m in ob.modifiers:
            if m.type == 'SUBSURF':
                saved[(ob.name, m.name)] = m.levels; m.levels = m.render_levels
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    char = bpy.data.collections['Character']
    objects = []
    total_base = 0; total_eval = 0; curves = []
    for ob in char.all_objects:
        item = dict(name=ob.name, type=ob.type, collection=ob.users_collection[0].name if ob.users_collection else None,
                    materials=[s.material.name for s in ob.material_slots if s.material])
        ev = ob.evaluated_get(dg)
        if ob.type == 'MESH':
            me = ev.to_mesh()
            item['base_vertices'] = len(ob.data.vertices); item['evaluated_vertices'] = len(me.vertices); item['evaluated_faces'] = len(me.polygons)
            total_base += len(ob.data.vertices); total_eval += len(me.vertices)
            ev.to_mesh_clear()
            item['cycles_ray_visible'] = bool(ob.visible_camera)
        if ob.type == 'CURVES' or any(m.type == 'NODES' for m in ob.modifiers):
            n_c = n_p = 0
            for inst in dg.object_instances:
                if inst.object.original == ob and inst.object.type in ('CURVES',):
                    n_c += len(inst.object.data.curves); n_p += len(inst.object.data.points)
            if ob.type == 'CURVES':
                item['authored_curves'] = len(ob.data.curves); item['authored_points'] = len(ob.data.points)
            geo = ev.evaluated_geometry() if hasattr(ev, 'evaluated_geometry') else None
            try:
                cv = geo.curves if geo is not None else None
                if cv is not None:
                    item['evaluated_curves'] = len(cv.curves); item['evaluated_curve_points'] = len(cv.points)
            except Exception:
                pass
            curves.append(ob.name)
        objects.append(item)
    lo, hi = _bounds([o for o in char.all_objects if o.type in ('MESH', 'CURVES')], dg)
    for (on, mn), lv in saved.items():
        bpy.data.objects[on].modifiers[mn].levels = lv
    images = [dict(name=i.name, size=list(i.size), packed=bool(i.packed_file), colorspace=i.colorspace_settings.name)
              for i in bpy.data.images if i.type != 'RENDER_RESULT' and i.size[0] > 0]
    rep = dict(
        entrant='Claude Fable 5.2 (claude-fable-5-2)', stage='procedural (no photographic character textures)',
        blender_version=bpy.app.version_string, blender_build_hash=bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
        render=dict(engine=sc.render.engine, device=sc.cycles.device, threads_mode=sc.render.threads_mode, threads=sc.render.threads,
                    samples=sc.cycles.samples, denoising=bool(sc.cycles.use_denoising), denoiser=getattr(sc.cycles, 'denoiser', None),
                    view_transform=sc.view_settings.view_transform, look=sc.view_settings.look, gpu_used=False),
        scene_file=os.path.basename(bpy.data.filepath), active_camera=sc.camera.name if sc.camera else None,
        cameras=[dict(name=o.name, type=o.data.type, lens_mm=round(o.data.lens, 3), ortho_scale=(round(o.data.ortho_scale, 3) if o.data.type == 'ORTHO' else None),
                      dof=bool(o.data.dof.use_dof), fstop=(round(o.data.dof.aperture_fstop, 2) if o.data.dof.use_dof else None)) for o in sc.objects if o.type == 'CAMERA'],
        collections=[c.name for c in bpy.data.collections],
        character_objects=objects,
        character_bounds_m=dict(min=lo, max=hi, note='world space; evaluated mesh vertices and authored curve points of every Character object'),
        geometry=dict(character_base_mesh_vertices=total_base, character_evaluated_mesh_vertices_at_render_levels=total_eval,
                      native_curve_objects=curves, note='hair, lashes, brows and knit fuzz are curves and are counted separately from mesh vertices'),
        generated_images=images,
    )
    rep.update(extra)
    with open(path, 'w') as fh:
        json.dump(rep, fh, indent=2)
    return rep
