"""Cameras, lighting, the simple cafe set and render settings."""
import math
import numpy as np
import bpy
from mathutils import Vector, Matrix
from . import util

BEAUTY_RES = (960, 1040)
DIAG_RES = (768, 768)


REF_W, REF_H = 1205.0, 1306.0
REF_EYE_MID = (576.0, 351.0)       # where the mid-point between the pupils sits in the reference frame (pixels)
REF_IPD_PX = 154.6                  # pupil distance in the reference frame (pixels)


def beauty_camera_spec(sk, M, fov_v=74.0, elevation=6.0, azimuth=-4.0):
    """Selfie camera placed from the posed eyes so the face lands where it does in the reference framing.

    The camera sits `elevation` degrees above / `azimuth` degrees beside the line of sight of the face, at the
    distance that reproduces the reference pupil spacing; its axis is then tipped down so the eyes appear in the
    upper third, as in the photograph. Only 2D layout numbers (pixel positions) come from the photo."""
    from .sculpt import EYE_C
    Mh = M['head']
    from .sculpt import EYE_R
    eL = sk.apply(Mh, EYE_C); eR = sk.apply(Mh, EYE_C * np.array([-1.0, 1.0, 1.0]))
    el = math.radians(elevation); az = math.radians(azimuth)
    u = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    m = 0.5 * (eL + eR) + u * EYE_R                     # mid-point between the pupils (on the corneas)
    v = eL - eR
    ipd_perp = np.linalg.norm(v - u * np.dot(u, v))
    f_px = (REF_H / 2) / math.tan(math.radians(fov_v) / 2)
    tx = (REF_EYE_MID[0] - REF_W / 2) / f_px            # tangent of the horizontal angle of the eyes from the axis
    ty = (REF_H / 2 - REF_EYE_MID[1]) / f_px            # ... and vertical (positive = above the axis)
    Zw = np.array([0.0, 0.0, 1.0])
    d = f_px * ipd_perp / REF_IPD_PX
    for _ in range(8):
        pos = m + u * d
        f = -u
        for _ in range(6):                              # aim: tip the axis so the eyes sit at (tx, ty)
            r = np.cross(f, Zw); r /= np.linalg.norm(r); up = np.cross(r, f)
            want = r * tx + up * ty + f; want /= np.linalg.norm(want)
            dirn = (m - pos) / np.linalg.norm(m - pos)
            f = f + (dirn - want); f /= np.linalg.norm(f)
        r = np.cross(f, Zw); r /= np.linalg.norm(r); up = np.cross(r, f)
        def px(p):
            q = p - pos
            return np.array([np.dot(q, r), np.dot(q, up)]) / np.dot(q, f) * f_px
        pl = eL + (pos - eL) / np.linalg.norm(pos - eL) * EYE_R; pr = eR + (pos - eR) / np.linalg.norm(pos - eR) * EYE_R
        d *= np.linalg.norm(px(pl) - px(pr)) / REF_IPD_PX   # eyes converge on the lens and sit off-axis: iterate
    return dict(position=pos, target=pos + f, fov_v=fov_v, roll=0.0, fstop=2.6, focus=m, distance=float(d))


def look_at(ob, pos, target, roll=0.0):
    pos = Vector(pos); d = (Vector(target) - pos).normalized()
    q = d.to_track_quat('-Z', 'Y')
    ob.location = pos
    ob.rotation_euler = (q @ Matrix.Rotation(math.radians(roll), 4, 'Z').to_quaternion()).to_euler()


def setup_render(sc, samples=32, quick=False):
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.cycles.max_bounces = 8; sc.cycles.diffuse_bounces = 3; sc.cycles.glossy_bounces = 4
    sc.cycles.transmission_bounces = 8; sc.cycles.transparent_max_bounces = 24
    sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
    sc.render.threads_mode = 'FIXED'; sc.render.threads = 3
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.use_stamp_filename = False        # keep absolute workstation paths out of PNG metadata
    for attr in ('use_stamp_date', 'use_stamp_time', 'use_stamp_render_time', 'use_stamp_memory', 'use_stamp_hostname'):
        if hasattr(sc.render, attr):
            setattr(sc.render, attr, False)
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Punchy'
    sc.view_settings.exposure = 0.0
    sc.cycles_curves.shape = 'RIBBONS' if hasattr(sc, 'cycles_curves') else None


def area(name, loc, target, energy, size, color=(1, 1, 1), col=None, size_y=None):
    l = bpy.data.lights.new(name, 'AREA'); l.energy = energy; l.color = color
    if size_y:
        l.shape = 'RECTANGLE'; l.size = size; l.size_y = size_y
    else:
        l.size = size
    o = bpy.data.objects.new(name, l)
    (col or bpy.context.scene.collection).objects.link(o)
    look_at(o, loc, target)
    return o


def build(ctx, cafe=True):
    sc = bpy.context.scene
    setup_render(sc, 32)
    spec = ctx['cam_spec']
    col_cam = util.collection('Cameras')
    col_light = util.collection('Lights_Portrait')
    col_studio = util.collection('Lights_Diagnostic_Studio')
    # beauty camera
    cd = bpy.data.cameras.new('Cam_Beauty'); cam = bpy.data.objects.new('Cam_Beauty', cd); col_cam.objects.link(cam)
    cd.sensor_fit = 'VERTICAL'; cd.sensor_height = 24.0
    cd.lens = 24.0 / (2 * math.tan(math.radians(spec['fov_v']) / 2))
    cd.clip_start = 0.02; cd.clip_end = 60
    look_at(cam, spec['position'], spec['target'], spec['roll'])
    cd.dof.use_dof = True; cd.dof.aperture_fstop = spec['fstop']
    cd.dof.focus_distance = float(np.linalg.norm(spec['focus'] - spec['position']))
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = BEAUTY_RES
    # diagnostic orthographic cameras orbiting the character (0 = front, positive = toward her left side)
    pts = [np.asarray(ctx[k]).reshape(-1, 3) for k in ('V_posed', 'sweater_P', 'hair_points') if ctx.get(k) is not None]
    pts = np.concatenate(pts) if pts else np.array([[-0.4, -0.6, 0.9], [0.4, 0.2, 1.75]])
    for name, az in (('front', 0), ('three-quarter', 45), ('side', 90), ('back', 180)):
        d = bpy.data.cameras.new('Cam_diagnostic_' + name); o = bpy.data.objects.new('Cam_diagnostic_' + name, d)
        col_cam.objects.link(o)
        a = math.radians(az)
        view = np.array([math.sin(a), -math.cos(a), 0.0])            # from the character towards the camera
        right = np.array([math.cos(a), math.sin(a), 0.0])
        u = pts @ right; v = pts[:, 2]
        cu = 0.5 * (u.min() + u.max()); cv = 0.5 * (v.min() + v.max())
        centre = Vector(right * cu + np.array([0, 0, cv]) + view * float((pts @ view).mean()))
        look_at(o, centre + Vector(view) * 4.0, centre)
        d.type = 'ORTHO'; d.ortho_scale = float(max(u.max() - u.min(), v.max() - v.min()) * 1.10)   # whole asset, 5 % margin each side
        d.clip_start = 0.1; d.clip_end = 20
    # world: soft warm ambient (beauty) -- the diagnostic renders switch to a neutral studio grey
    w = bpy.data.worlds.new('World_Cafe'); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (1.0, 0.95, 0.88, 1); bg.inputs['Strength'].default_value = spec.get('world', 0.22)
    ws = bpy.data.worlds.new('World_Studio'); ws.use_nodes = True; ws.use_fake_user = True
    bgs = ws.node_tree.nodes['Background']
    bgs.inputs['Color'].default_value = (0.62, 0.62, 0.64, 1); bgs.inputs['Strength'].default_value = 0.55
    # portrait lights for the cafe shot: window daylight from her right/back, broad soft fill from the camera side
    tgt = (-0.03, -0.12, 1.50)
    L = spec.get('lights', {})
    area('Light_Window_Key', (-1.55, 0.35, 2.05), tgt, L.get('key', 46), 1.5, (1.0, 0.96, 0.90), col_light)
    fl = area('Light_Front_Fill', (-0.10, -1.70, 1.80), tgt, L.get('fill', 44), 2.0, (1.0, 0.95, 0.91), col_light)
    area('Light_Back_Rim', (-0.95, 1.25, 2.35), (-0.05, 0, 1.58), L.get('rim', 22), 0.9, (1.0, 0.93, 0.82), col_light)
    kl = area('Light_Right_Kicker', (1.15, -0.95, 1.75), tgt, L.get('kick', 22), 1.2, (1.0, 0.95, 0.9), col_light)
    area('Light_Eye_Catch', (-0.55, -1.05, 1.95), tgt, L.get('catch', 34), 0.26, (1.0, 0.98, 0.95), col_light)
    # the two broad fills would mirror as a milky veil in the corneas: exclude just the cornea shells from them
    try:
        veil = bpy.data.collections.new('LightLink_No_Fill_On_Corneas')
        for name in ('Eye_Cornea_L', 'Eye_Cornea_R'):
            veil.objects.link(bpy.data.objects[name])
        for co in veil.collection_objects:
            co.light_linking.link_state = 'EXCLUDE'
        for lo in (fl, kl):
            lo.light_linking.receiver_collection = veil
    except Exception as exc:
        fl.visible_glossy = False; kl.visible_glossy = False
        ctx['log']('light linking unavailable (%s); fills made diffuse-only instead' % exc)
    # neutral three-point studio rig used only for the four plain-background diagnostic views
    c = (-0.02, -0.12, 1.40)
    area('Studio_Key', (-1.6, -2.2, 2.6), c, 150, 2.2, (1, 1, 1), col_studio)
    area('Studio_Fill', (2.2, -1.6, 1.8), c, 70, 2.5, (1, 1, 1), col_studio)
    area('Studio_Back', (1.4, 2.4, 2.6), c, 150, 2.0, (1, 1, 1), col_studio)
    area('Studio_Back_2', (-2.2, 2.0, 2.2), c, 110, 2.0, (1, 1, 1), col_studio)
    area('Studio_Top', (0.0, -0.2, 3.4), c, 60, 2.5, (1, 1, 1), col_studio)
    col_studio.hide_render = True
    if cafe:
        from . import cafe as cafe_mod
        col_cafe = util.collection('Cafe_Set')
        cafe_mod.build(col_cafe, ctx['rng'])


def set_mode(mode):
    """'beauty' = cafe set + portrait lights; 'diagnostic' = plain studio (same character, same materials)."""
    sc = bpy.context.scene
    beauty = mode == 'beauty'
    for name, vis in (('Cafe_Set', beauty), ('Lights_Portrait', beauty), ('Lights_Diagnostic_Studio', not beauty)):
        c = bpy.data.collections.get(name)
        if c is not None:
            c.hide_render = not vis
    sc.world = bpy.data.worlds['World_Cafe' if beauty else 'World_Studio']
