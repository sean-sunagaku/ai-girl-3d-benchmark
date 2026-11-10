"""Project facial landmarks of the saved scene through Cam_Beauty and compare with hand-measured photo landmarks.
Blender -b character.blend --python tools/landmarks.py
"""
import bpy, math, json, sys
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
sc = bpy.context.scene; cam = bpy.data.objects['Cam_Beauty']
sc.render.resolution_x, sc.render.resolution_y = 960, 1040
bpy.context.view_layer.update()
body = bpy.data.objects['Body']; me = body.data
W, H = 1205, 1306
def proj(p):
    v = world_to_camera_view(sc, cam, Vector(p)); return np.array([v.x * W, (1 - v.y) * H])
def vert(i): return body.matrix_world @ me.vertices[i].co
ids = dict(nose_tip=297, subnasale=343, stomion=474, lip_top=455, lip_bot=492, chin=732, corner_L=5298)
P = {k: proj(vert(i)) for k, i in ids.items()}
# mirrored corner: find vertex mirrored in the rest attribute
rest = np.array([d.vector[:] for d in me.attributes['rest_position_base'].data])
def mirror_id(i):
    t = rest[i] * np.array([-1, 1, 1]); return int(np.argmin(np.linalg.norm(rest - t, axis=1)))
P['corner_R'] = proj(vert(mirror_id(5298)))
for tag in ('L', 'R'):
    e = bpy.data.objects['Eye_' + tag]
    c = e.matrix_world.translation; axis = (e.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    P['pupil_' + tag] = proj(c + axis * 0.0149)
# lid corners: margin extreme vertices (rest x extremes within lid loop) -- approximate via rest positions
def nearest_rest(p): return int(np.argmin(np.linalg.norm(rest - np.array(p), axis=1)))
for tag, sx in (('L', 1), ('R', -1)):
    P['inner_' + tag] = proj(vert(nearest_rest((sx * 0.0193, -0.1375, 1.5448))))
    P['outer_' + tag] = proj(vert(nearest_rest((sx * 0.0437, -0.1355, 1.5450))))
    P['lid_up_' + tag] = proj(vert(nearest_rest((sx * 0.0311, -0.1411, 1.5499))))
    P['lid_lo_' + tag] = proj(vert(nearest_rest((sx * 0.0311, -0.1405, 1.5408))))
photo = dict(pupil_R=(508, 388), pupil_L=(643, 313), nose_tip=(613, 445), subnasale=(625, 467), stomion=(658, 493),
             lip_top=(647, 478), lip_bot=(667, 520), chin=(686, 574), corner_R=(598, 515), corner_L=(715, 445),
             inner_R=(537, 387), outer_R=(467, 397), inner_L=(615, 337), outer_L=(675, 297))
def frame(D):
    a = np.array(D['pupil_R'], float); b = np.array(D['pupil_L'], float)
    ipd = np.linalg.norm(b - a); e1 = (b - a) / ipd; e2 = np.array([-e1[1], e1[0]])
    if e2[1] < 0: e2 = -e2
    return (a + b) / 2, e1, e2, ipd
def norm(D):
    m, e1, e2, ipd = frame(D)
    return {k: ((np.array(v, float) - m) @ e1 / ipd, (np.array(v, float) - m) @ e2 / ipd) for k, v in D.items()}, m, ipd, math.degrees(math.atan2(-e1[1], e1[0]))
nr, mr, ir, ar = norm(P); nph, mp, ip, ap = norm(photo)
print('RENDER eye-mid (%.0f,%.0f) IPD %.1fpx angle %.1f | PHOTO eye-mid (%.0f,%.0f) IPD %.1fpx angle %.1f' % (mr[0], mr[1], ir, ar, mp[0], mp[1], ip, ap))
for k in ('nose_tip', 'subnasale', 'lip_top', 'stomion', 'lip_bot', 'chin', 'corner_R', 'corner_L', 'inner_R', 'outer_R', 'inner_L', 'outer_L'):
    print('  %-10s render (%+.3f, %+.3f)  photo (%+.3f, %+.3f)  d=(%+.3f, %+.3f)' % (k, nr[k][0], nr[k][1], nph[k][0], nph[k][1], nr[k][0] - nph[k][0], nr[k][1] - nph[k][1]))
for tag in ('L', 'R'):
    w = np.linalg.norm(P['outer_' + tag] - P['inner_' + tag]) / ir; h = np.linalg.norm(P['lid_up_' + tag] - P['lid_lo_' + tag]) / ir
    print('  eye %s width %.3f height %.3f (photo ~0.46 / 0.18)' % (tag, w, h))
print('  mouth width render %.3f photo %.3f' % (np.linalg.norm(P['corner_L'] - P['corner_R']) / ir, np.linalg.norm(np.array(photo['corner_L']) - np.array(photo['corner_R'])) / ip))
