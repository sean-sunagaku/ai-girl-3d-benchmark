"""Build the editable cafe-selfie portrait scene from the supplied inputs, then render it.

Run from this directory:
    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py

Inputs (only): input/upper-body.obj (CC0 MakeHuman), input/environment.json, input/reference.png.
Steps: sculpt + photo-landmark camera fit and face warp (photo_fit.py) -> pose -> eyes, sweater,
hair curves, hands/phone -> photo skin / iris textures projected through the fitted camera
(photo_texture.py) -> save packed character.blend -> reopen and check -> render the five
final views (render.py) -> scene-report.json (scene_report.py).
Set SKIP_RENDER=1 to stop after saving/reopening the .blend.
"""
import bpy, bmesh, math, json, os, sys, time
import numpy as np
from mathutils import Vector, Matrix, Quaternion, Euler
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

sys.path.insert(0, os.getcwd())
import photo_fit as pf
import photo_texture as ptex

ROOT = os.getcwd()
INPUT = os.path.join(ROOT, "input")
ENV = json.load(open(os.path.join(INPUT, "environment.json")))
LM = {k: Vector(v) for k, v in ENV["landmarks"].items()}
STOP_AFTER = os.environ.get("STOP_AFTER", "")
T0 = time.time()
REPORT = {"textures": [], "notes": []}


def log(*a):
    print("[build %6.1fs]" % (time.time() - T0), *a, flush=True)


# ----------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bump(co, c, r):
    """Compact smooth radial falloff (1 at c, 0 beyond r). r may be a 3-vector (ellipsoid)."""
    d = (co - np.asarray(c, float)) / np.asarray(r, float)
    q = np.clip(1.0 - (d * d).sum(1), 0.0, 1.0)
    return q * q


def sbump(co, c, r):
    """Mirror-symmetric bump (feature exists on both sides of the face)."""
    c = np.asarray(c, float)
    c2 = c.copy()
    c2[0] = -c2[0]
    return np.maximum(bump(co, c, r), bump(co, c2, r))


def get_co(me):
    a = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def set_co(me, co):
    me.vertices.foreach_set("co", np.asarray(co, np.float64).ravel())
    me.update()


def link(ob, coll):
    coll.objects.link(ob)
    return ob


def new_coll(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def evaluated_mesh_arrays(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    co = get_co(me)
    mw = np.array(ob.matrix_world)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    ev.to_mesh_clear()
    return co, tris.reshape(-1, 3)


# ----------------------------------------------------------------------------
# scene setup
# ----------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.render.threads_mode = 'FIXED'
    sc.render.threads = 3
    return sc


# ----------------------------------------------------------------------------
# 1. sculpt the neutral MakeHuman cage toward the reference proportions
# ----------------------------------------------------------------------------
def mouth_line(x):
    """z of the lip seam on the source mesh (curves up toward the centre)."""
    return 1.4776 + 0.0005 * np.clip(np.abs(x) / 0.0265, 0, 1.3) ** 2


def sculpt(co):
    co = co.copy()
    X, Y, Z = co[:, 0], co[:, 1], co[:, 2]
    ax, sx = np.abs(X), np.sign(X)
    d = np.zeros_like(co)

    # --- neck: slimmer, slightly longer feminine neck
    nk = smoothstep(1.355, 1.39, Z) * smoothstep(1.47, 1.43, Z)
    cy = -0.012
    d[:, 0] += -0.085 * nk * X
    d[:, 1] += -0.06 * nk * (Y - cy)

    # --- shoulders a little narrower / softer
    sh = smoothstep(0.06, 0.16, ax) * smoothstep(1.05, 1.2, Z) * smoothstep(1.47, 1.40, Z)
    d[:, 0] += -0.022 * sh * sx

    # --- bust: inferred/hidden under the loose sweater; keep it modest so the knit drapes
    bp = sbump(co, (0.08, -0.115, 1.215), (0.085, 0.07, 0.075))
    d[:, 1] += bp * np.maximum(0.0, -0.090 - Y) * 0.75
    # flatten the nipples of the nude base mesh (they would print through the knit)
    d[:, 1] += 0.0045 * sbump(co, (0.070, -0.159, 1.194), (0.016, 0.012, 0.016))

    # --- lower face: narrower jaw, softer V chin, shorter chin
    jaw = smoothstep(1.515, 1.455, Z) * smoothstep(1.405, 1.445, Z) * smoothstep(0.02, -0.06, Y)
    d[:, 0] += -0.10 * jaw * X
    chin = bump(co, (0, -0.145, 1.447), (0.034, 0.03, 0.022))
    d[:, 0] += -0.13 * chin * X
    d[:, 2] += 0.0025 * chin
    d[:, 1] += -0.0012 * chin
    # jaw angle softer (pull in & up a little)
    ja = sbump(co, (0.058, -0.06, 1.462), (0.025, 0.03, 0.025))
    d[:, 0] += -0.004 * ja * sx
    d[:, 2] += 0.002 * ja

    # --- cheeks: fuller, higher (youthful + smiling)
    ck = sbump(co, (0.040, -0.128, 1.512), (0.022, 0.02, 0.02))
    d[:, 1] += -0.0028 * ck
    d[:, 2] += 0.0018 * ck
    d[:, 0] += 0.0012 * ck * sx

    # --- brow ridge softened, forehead rounder
    br = sbump(co, (0.033, -0.142, 1.574), (0.024, 0.02, 0.012))
    d[:, 1] += 0.0022 * br
    fh = bump(co, (0, -0.128, 1.607), (0.045, 0.03, 0.03))
    d[:, 1] += -0.0016 * fh

    # --- eyes: slightly larger opening (uniform scale about socket centre)
    for s in (-1, 1):
        ec = np.array([s * 0.0307, -0.1258, 1.5447])
        w = bump(co, ec, (0.025, 0.027, 0.021))
        d += (co - ec) * (0.11 * w)[:, None]
    # angular warp about each socket centre: opens the lids (mostly vertically) while
    # keeping every lid vertex at its radius, so the lids still wrap the eyeball sphere
    for s in (-1, 1):
        ec = np.array([s * 0.0307, -0.1258, 1.5447])
        v = co - ec
        dist = np.linalg.norm(v, axis=1)
        u = v / dist[:, None]
        theta = np.degrees(np.arccos(np.clip(-u[:, 1], -1, 1)))
        w = smoothstep(0.031, 0.0185, dist) * np.clip(1 - theta / 78.0, 0, 1) * (u[:, 1] < -0.15)
        u2 = u.copy()
        u2[:, 2] *= 1 + np.where(u[:, 2] > 0, 0.30, -0.04) * w
        u2[:, 0] = u[:, 0] * (1 + 0.13 * w)
        u2 /= np.linalg.norm(u2, axis=1)[:, None]
        d += (ec + u2 * dist[:, None] - co) * (w > 0)[:, None]
    # under-eye fullness (soft aegyo-sal)
    ue = sbump(co, (0.031, -0.1445, 1.5325), (0.012, 0.007, 0.0045))
    d[:, 1] += -0.0011 * ue
    d[:, 2] += 0.0006 * ue

    # --- nose: smaller, narrower alae, softer bridge, slightly lifted tip
    nz = bump(co, (0, -0.152, 1.515), (0.026, 0.03, 0.028)) * smoothstep(-0.128, -0.15, Y)
    anchor = np.array([0, -0.138, 1.52])
    d[:, 0] += -0.17 * nz * X
    d[:, 1] += -0.14 * nz * (Y - anchor[1])
    d[:, 2] += 0.0012 * nz
    bridge = bump(co, (0, -0.15, 1.553), (0.012, 0.02, 0.022))
    d[:, 1] += 0.0012 * bridge

    # --- lips: fuller, slight closed-mouth smile
    up = bump(co, (0, -0.158, 1.486), (0.024, 0.012, 0.007))
    lo = bump(co, (0, -0.156, 1.470), (0.022, 0.012, 0.007))
    d[:, 1] += 0.0004 * up + 0.0002 * lo
    d[:, 2] += 0.0006 * up - 0.0005 * lo
    corner = sbump(co, (0.0262, -0.137, 1.478), (0.012, 0.014, 0.009))
    d[:, 2] += 0.0034 * corner
    d[:, 1] += 0.0004 * corner
    d[:, 0] += 0.0009 * corner * sx
    # nasolabial / risorius region lifts with the smile
    sm = sbump(co, (0.033, -0.135, 1.488), (0.014, 0.02, 0.016))
    d[:, 2] += 0.0012 * sm
    d[:, 1] += -0.0006 * sm
    # mouth slightly narrower overall
    mw = bump(co, (0, -0.15, 1.478), (0.035, 0.03, 0.02))
    d[:, 0] += -0.10 * mw * X

    co += d
    # youthful proportions: shorten the face below the eyes (front half of the head)
    X, Y, Z = co[:, 0], co[:, 1], co[:, 2]
    k = 0.86
    w = smoothstep(-0.005, -0.07, Y) * smoothstep(1.385, 1.432, Z) * smoothstep(1.545, 1.53, Z)
    w = w * smoothstep(0.075, 0.055, np.abs(X)) + 0.0 * w
    znew = 1.535 - (1.535 - Z) * k
    co[:, 2] = Z + (znew - Z) * w
    # shorter chin (V-line) and smaller, less protruding lips
    X, Y, Z = co[:, 0], co[:, 1], co[:, 2]
    w = smoothstep(-0.01, -0.07, Y) * smoothstep(1.475, 1.465, Z) * smoothstep(1.385, 1.425, Z) * smoothstep(0.07, 0.04, np.abs(X))
    co[:, 2] += (1.474 - Z) * 0.13 * w
    lip = bump(co, (0, -0.152, 1.486), (0.032, 0.022, 0.017))
    co[:, 2] = co[:, 2] + (mouth_line(co[:, 0]) - co[:, 2]) * 0.15 * lip
    co[:, 1] += 0.0016 * lip
    # close the lip seam (relaxed closed-mouth smile)
    zm = mouth_line(co[:, 0])
    seam = np.exp(-((co[:, 2] - zm) / 0.0032) ** 2) * smoothstep(0.03, 0.024, np.abs(co[:, 0]))
    seam *= smoothstep(-0.128, -0.142, co[:, 1]) * smoothstep(-0.175, -0.165, co[:, 1])
    co[:, 2] += (zm - co[:, 2]) * 0.6 * seam
    # rounder, fuller cheeks (soft youthful face) and a blunter chin
    X, Y, Z = co[:, 0], co[:, 1], co[:, 2]
    rc = sbump(co, (0.050, -0.105, 1.50), (0.03, 0.045, 0.04))
    co[:, 0] += 0.0042 * rc * np.sign(X)
    co[:, 1] -= 0.0012 * sbump(co, (0.042, -0.125, 1.505), (0.02, 0.02, 0.02))
    ch = bump(co, (0, -0.14, 1.445), (0.03, 0.03, 0.02))
    co[:, 0] += 0.06 * ch * X
    return co


# ----------------------------------------------------------------------------
# 2. pose (selfie arms, tilted head) with a temporary DQS armature
# ----------------------------------------------------------------------------
CAM_BEAUTY = Vector((0.02, -0.50, 1.70))
BEAUTY_TARGET = Vector((-0.03, -0.10, 1.46))
# initial guess; replaced by fit_beauty_camera() (photo landmark fit) before posing
CAM_ROT = (Quaternion((BEAUTY_TARGET - CAM_BEAUTY).normalized(), math.radians(-3))
           @ (BEAUTY_TARGET - CAM_BEAUTY).to_track_quat('-Z', 'Y')).to_matrix()
CAM_LENS = 26.0                        # phone selfie camera (~26 mm equivalent)
FIT_UV = {}                            # final projected mesh landmarks (photo pixels)
PHONE = (0.0715, 0.147, 0.0079)        # width, height, thickness (m)


def phone_frame():
    # the phone *is* the selfie camera: its long axis follows the camera's up vector
    v = -CAM_ROT.col[2]                                    # camera view direction
    ul = CAM_ROT.col[1].copy()                             # phone long axis
    ux = CAM_ROT.col[0].copy()                             # toward her left (+X)
    if ux.x < 0:
        ux = -ux
    # lens sits at the top-centre of the screen; the body extends behind the camera
    pc = CAM_BEAUTY - ul * (PHONE[1] / 2 - 0.011) - v * (PHONE[2] / 2 + 0.011)
    return pc, ux, ul, v


def grip_point(side):
    pc, ux, ul, v = phone_frame()
    s = 1 if side == "L" else -1
    # her right hand grips the lower corner of the phone so the hand and cuff stay outside
    # the selfie frame (as in the photo, where both arms leave the frame)
    h = -0.040 if side == "L" else -0.056
    lat = 0.010 if side == "L" else 0.012
    return pc + ux * s * (PHONE[0] / 2 + lat) - v * 0.006 + ul * h


def rot_between(a, b):
    return a.normalized().rotation_difference(b.normalized())


def affine(pivot, q, new_pivot=None):
    """Matrix for v -> new_pivot + q (v - pivot)."""
    new_pivot = pivot if new_pivot is None else new_pivot
    return Matrix.Translation(new_pivot) @ q.to_matrix().to_4x4() @ Matrix.Translation(-pivot)


def two_bone(S, W, L1, L2, pole):
    d = W - S
    D = min(d.length, (L1 + L2) * 0.985)
    dn = d.normalized()
    a = (L1 * L1 - L2 * L2 + D * D) / (2 * D)
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))
    pn = (pole - dn * pole.dot(dn)).normalized()
    return S + dn * a + pn * h, S + dn * D


def pose_definition():
    T = {}
    # head: roll toward her right shoulder, chin a little up toward the raised phone,
    # tiny turn toward her left.  Split between neck and head joints.
    roll, pitch, yaw = math.radians(-26), math.radians(-14), math.radians(5)
    qn = Euler((pitch * 0.45, roll * 0.55, yaw * 0.5), 'XYZ').to_quaternion()
    qh = Euler((pitch * 0.55, roll * 0.45, yaw * 0.5), 'XYZ').to_quaternion()
    Pn = Vector((0, -0.006, 1.392))
    An = affine(Pn, qn)
    Ph = LM["joint-head"].copy()
    Ah = An @ affine(Ph, qh)
    T["neck"], T["head"] = An, Ah
    for side, s in (("L", 1), ("R", -1)):
        lr = side.lower()
        S = LM["joint-%s-shoulder" % lr]
        E = LM["joint-%s-elbow" % lr]
        W = LM["joint-%s-hand" % lr]
        Cl = LM["joint-%s-clavicle" % lr]
        # clavicle: protract + elevate the shoulder for reaching
        dS = Vector((-0.014 * s, -0.050, 0.030))
        qc = rot_between(S - Cl, (S + dS) - Cl)
        Ac = affine(Cl, qc)
        S2 = Ac @ S
        G = grip_point(side)
        f0 = (G - S2).normalized()
        Wt = G - f0 * 0.088
        L1, L2 = (E - S).length, (W - E).length
        E2, W2 = two_bone(S2, Wt, L1, L2, Vector((0.6 * s, 0.1, -1.0)))
        log("arm %s: reach %.3f of %.3f m (%s)" % (side, (Wt - S2).length, L1 + L2,
                                                 "ok" if (Wt - S2).length < (L1 + L2) * 0.985 else "SHORT"))
        qu = rot_between(E - S, E2 - S2)
        Au = affine(S, qu, S2)
        qf = rot_between(qu @ (W - E), W2 - E2) @ qu
        Af = affine(E, qf, E2)
        T["clav." + side], T["upper." + side], T["fore." + side] = Ac, Au, Af
        T["_W2." + side], T["_E2." + side], T["_S2." + side] = W2, E2, S2
        T["_dirf." + side] = (W2 - E2).normalized()
    return T


def pose_weights(rest):
    X, Y, Z = rest[:, 0], rest[:, 1], rest[:, 2]
    ax = np.abs(X)
    W = {}
    # head = everything above the plane from under the chin to the skull base, so the whole
    # face incl. jaw/lips moves rigidly (a horizontal blend band would shear the mouth)
    n_ = np.array([0.0, -0.327, 0.945])
    dist_ = (rest - np.array([0.0, -0.10, 1.425])) @ n_
    W["head"] = smoothstep(-0.010, 0.014, dist_) * smoothstep(1.37, 1.40, Z)
    W["neck"] = np.clip(smoothstep(1.365, 1.43, Z) * smoothstep(0.105, 0.07, ax) - W["head"], 0, 1)
    W["neck"] = np.maximum(W["neck"], 0) + 0 * Z
    for side, s in (("L", 1), ("R", -1)):
        lr = side.lower()
        S = np.array(LM["joint-%s-shoulder" % lr])
        E = np.array(LM["joint-%s-elbow" % lr])
        n = np.array([s * 1.0, 0.0, -0.32])
        n /= np.linalg.norm(n)
        p0 = np.array([s * 0.158, -0.015, 1.30])
        dist = (rest - p0) @ n
        gate = np.maximum(smoothstep(1.235, 1.29, Z), smoothstep(0.155, 0.18, ax))
        warm = smoothstep(-0.03, 0.03, dist) * (np.sign(X) == s) * gate
        u = (E - S) / np.linalg.norm(E - S)
        wf = smoothstep(-0.025, 0.035, (rest - E) @ u)
        wc = smoothstep(0.035, 0.13, ax) * smoothstep(1.20, 1.31, Z) * (np.sign(X) == s) * smoothstep(1.47, 1.40, Z)
        W["fore." + side] = warm * wf
        W["upper." + side] = warm * (1 - wf)
        W["clav." + side] = wc * (1 - warm)
    tot = sum(W.values())
    over = np.maximum(tot, 1.0)
    for k in W:
        W[k] = W[k] / over
    W["spine"] = np.clip(1.0 - sum(W.values()), 0, 1)
    return W


def pose_mesh(ob, rest, T, coll):
    """Pose ob (whose current coords == rest) with dual-quaternion skinning, then bake."""
    W = pose_weights(rest)
    arm = bpy.data.armatures.new("PoseRig")
    rig = link(bpy.data.objects.new("PoseRig", arm), coll)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name in W:
        eb = arm.edit_bones.new(name)
        eb.head = (0, 0, 1.0)
        eb.tail = (0, 0, 1.1)
    bpy.ops.object.mode_set(mode='OBJECT')
    for name in W:
        if name in T:
            rig.pose.bones[name].matrix = T[name] @ arm.bones[name].matrix_local
    bpy.context.view_layer.update()
    for name, w in W.items():
        vg = ob.vertex_groups.new(name=name)
        idx = np.nonzero(w > 1e-4)[0]
        for i in idx:
            vg.add([int(i)], float(w[i]), 'REPLACE')
    mod = ob.modifiers.new("PoseDQS", 'ARMATURE')
    mod.object = rig
    mod.use_deform_preserve_volume = True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    posed = get_co(ev.data).copy()
    ob.modifiers.remove(mod)
    for vg in list(ob.vertex_groups):
        ob.vertex_groups.remove(vg)
    bpy.data.objects.remove(rig)
    bpy.data.armatures.remove(arm)
    set_co(ob.data, posed)
    return posed


def apply_subdiv(ob, levels=1):
    m = ob.modifiers.new("tmp_subd", 'SUBSURF')
    m.levels = levels
    m.render_levels = levels
    dg = bpy.context.evaluated_depsgraph_get()
    newme = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.remove(m)
    ob.data = newme
    bpy.data.meshes.remove(old)
    newme.name = ob.name + "_mesh"


def attr_vec(me, name):
    a = me.attributes[name]
    out = np.empty(len(a.data) * 3, np.float32)
    a.data.foreach_get("vector", out)
    return out.reshape(-1, 3).astype(np.float64)


def attr_float(me, name):
    a = me.attributes[name]
    out = np.empty(len(a.data), np.float32)
    a.data.foreach_get("value", out)
    return out.astype(np.float64)


def set_attr_vec(me, name, arr, domain='POINT'):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name, 'FLOAT_VECTOR', domain)
    a.data.foreach_set("vector", np.asarray(arr, np.float32).ravel())


def set_attr_float(me, name, arr, domain='POINT'):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name, 'FLOAT', domain)
    a.data.foreach_set("value", np.asarray(arr, np.float32).ravel())


def set_color_attr(me, name, rgb):
    if name in me.color_attributes:
        me.color_attributes.remove(me.color_attributes[name])
    a = me.color_attributes.new(name, 'FLOAT_COLOR', 'POINT')
    rgba = np.ones((len(rgb), 4), np.float32)
    rgba[:, :3] = rgb
    a.data.foreach_set("color", rgba.ravel())
    return a


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P * P).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(sol[3] + c @ c)
    return c, r


# ----------------------------------------------------------------------------
# photo match: fit the beauty camera to hand-picked photo landmarks, then warp the face
# ----------------------------------------------------------------------------
def rest_eye_centres(rest):
    out = {}
    for side, sgn in (("L", 1), ("R", -1)):
        ec0 = np.array([sgn * 0.0307, -0.1258, 1.5447])
        dist = np.linalg.norm(rest - ec0, axis=1)
        P = rest[(dist > 0.011) & (dist < 0.0195) & (rest[:, 1] > ec0[1] - 0.0115)]
        for _ in range(4):
            c, r = fit_sphere(P)
            res = np.abs(np.linalg.norm(P - c, axis=1) - r)
            P = P[res < max(0.0008, 2.5 * res.std())]
        out[side] = fit_sphere(P)
    return out


def head_landmarks(me, rest):
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    ecs = rest_eye_centres(rest)
    r_eye = np.mean([r for _, r in ecs.values()]) - 0.00045
    lm, _ = pf.mesh_landmarks(attr_vec(me, "orig_co"), rest, tris, {k: c for k, (c, _) in ecs.items()}, r_eye)
    return lm, tris


def axis_rotation(axis, ang):
    return np.array(Matrix.Rotation(ang, 3, Vector(axis.tolist())))


def eye_open_warp(rest, ec, up_l, f_up, f_lo):
    """Open/close the lids about the eyeball centre ec along the eye's own up axis, keeping
    each vertex at its radius so the lids keep wrapping the eyeball."""
    v = rest - ec
    dist = np.linalg.norm(v, axis=1)
    u = v / dist[:, None]
    theta = np.degrees(np.arccos(np.clip(-u[:, 1], -1, 1)))
    w = smoothstep(0.031, 0.0185, dist) * np.clip(1 - theta / 78.0, 0, 1) * (u[:, 1] < -0.15)
    c = u @ up_l
    u2 = u + up_l[None] * (c * np.where(c > 0, f_up, f_lo) * w)[:, None]
    u2 /= np.linalg.norm(u2, axis=1)[:, None]
    return np.where((w > 0)[:, None], ec + u2 * dist[:, None], rest)


def lid_rotate(rest, ec, axis_l, ang):
    """Rotate the lid region about the eyeball centre (radius preserved), shifting the eye
    opening over the eyeball without detaching the lids."""
    v = rest - ec
    dist = np.linalg.norm(v, axis=1)
    u = v / np.maximum(dist, 1e-9)[:, None]
    theta = np.degrees(np.arccos(np.clip(-u[:, 1], -1, 1)))
    w = smoothstep(0.032, 0.019, dist) * np.clip(1 - theta / 85.0, 0, 1) * (u[:, 1] < -0.1)
    out = rest.copy()
    idx = np.nonzero(w > 0)[0]
    for i in idx:
        Rm_ = np.array(Matrix.Rotation(ang * w[i], 3, Vector(axis_l.tolist())))
        out[i] = ec + Rm_ @ v[i]
    return out


def fit_to_photo(me, rest):
    """Match the face to the photograph as seen from the beauty camera.

    1. rigid 6-dof camera fit (lens fixed at CAM_LENS) to irises, nose, mouth, eye corners
       and the visible ear (photo landmarks picked by eye, see photo_fit.py);
    2. per eye: 3-D similarity of the whole eye region (rotation about the view axis,
       uniform scale about the eyeball centre, image-plane shift) + radius-preserving
       lid-opening / lid-rotation warps, so lids keep wrapping the eyeball;
    3. compact RBF warp (image-plane displacements) for nose, lips, chin, cheek contour, ears.
    Eyes are re-fitted once more after the last RBF pass.
    """
    global CAM_BEAUTY, CAM_ROT, BEAUTY_TARGET
    Ah = np.array(pose_definition()["head"])
    Rh = Ah[:3, :3]
    to_w = lambda P: np.asarray(P) @ Rh.T + Ah[:3, 3]
    res = pf.PHOTO_SIZE
    k = CAM_LENS / 36.0 * max(res)
    R = np.array(CAM_ROT).T
    t = np.array(CAM_BEAUTY)
    rep = REPORT.setdefault("photo_fit", {})
    tmp_me = me.copy()
    tmp_ob = bpy.data.objects.new("_fit_subd", tmp_me)
    bpy.context.scene.collection.objects.link(tmp_ob)
    sm = tmp_ob.modifiers.new("s", 'SUBSURF')
    sm.levels = sm.render_levels = 1

    def subdiv_eval(rest_):
        set_co(tmp_me, rest_)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = tmp_ob.evaluated_get(dg)
        m_ = ev.to_mesh()
        co_ = get_co(m_)
        m_.calc_loop_triangles()
        tr_ = np.empty(len(m_.loop_triangles) * 3, np.int32)
        m_.loop_triangles.foreach_get("vertices", tr_)
        ev.to_mesh_clear()
        return co_, tr_.reshape(-1, 3)

    def measure(rest, tag):
        lm, tris = head_landmarks(me, rest)
        names = [n for n in pf.PHOTO_LM if n != "chin"]
        # eye corner/lid entries are filled from the projected lid ring below
        P = np.array([lm.get(n, lm["eye_%s_iris" % n[4]] if n.startswith("eye_") else None) for n in names])
        uv, z = pf.project(to_w(P), R, t, k, res)
        # eye shape = the opening actually visible from the camera (eyeball sphere vs skin),
        # measured on the subdivided surface that is actually rendered
        sco, stris = subdiv_eval(rest)
        bvh = BVHTree.FromPolygons(sco.tolist(), stris.tolist())
        cam_l = Rh.T @ (t - Ah[:3, 3])
        Rl = Rh.T @ R.T
        ecs = rest_eye_centres(sco)
        for side, sg_img in (("L", 1), ("R", -1)):
            c_, r_ = ecs[side]
            # iris = projected centre of the eyeball that will actually be built (subdivided fit)
            uv[names.index("eye_%s_iris" % side)] = pf.project(to_w(c_[None]), R, t, k, res)[0][0]
            ir = uv[names.index("eye_%s_iris" % side)]
            px = pf.visible_eye_opening(bvh, cam_l, Rl, k, res, c_, r_ - 0.00045, ir)
            for q, p2 in zip(("outer", "inner", "upper", "lower"), pf.opening_points(px, ir, sg_img)):
                uv[names.index("eye_%s_%s" % (side, q))] = p2
        tgt = np.array([pf.PHOTO_LM[n] for n in names], float)
        err = np.linalg.norm(uv - tgt, axis=1)
        log("landmarks %s: mean %.1f px max %.1f" % (tag, err.mean(), err.max()))
        if os.environ.get("FIT_DEBUG"):
            for n, e_, p_ in zip(names, err, uv):
                log("   %-12s err %5.1f  at %6.1f %6.1f" % (n, e_, p_[0], p_[1]))
        rep["landmark_error_px_" + tag] = round(float(err.mean()), 2)
        return lm, tris, names, P, uv, z, tgt

    lm, tris, names, P, uv, z, tgt = measure(rest, "start")
    rig = [i for i, n in enumerate(names) if not n.startswith("eye_") or n.endswith("_iris")]
    wts = np.array([pf.PHOTO_W[names[i]] for i in rig])
    # the visible ear pins the head's yaw, which the near-planar inner-face points leave loose
    X_, Y_, Z_ = rest[:, 0], rest[:, 1], rest[:, 2]
    ear_c = rest[(X_ > 0.066) & (Z_ > 1.49) & (Z_ < 1.585) & (Y_ > -0.06) & (Y_ < 0.02)].mean(0)
    # eye corners (3-D lid-ring extremes) also constrain yaw: where the iris sits inside the
    # opening is a strong head-turn cue in the photo
    corners, ctg = [], []
    for side in ("L", "R"):
        ring = lm["eye_%s_ring" % side]
        lat = ring[:, 0] * (1 if side == "L" else -1)
        corners += [ring[np.argmax(lat)], ring[np.argmin(lat)]]
        ctg += [pf.PHOTO_LM["eye_%s_outer" % side], pf.PHOTO_LM["eye_%s_inner" % side]]
    Pr = np.vstack([P[rig], np.array(corners), ear_c])
    Tr = np.vstack([tgt[rig], np.array(ctg, float), pf.PHOTO_EAR["ear_L"]])
    R, t, e_r = pf.fit_camera(to_w(Pr), Tr, np.r_[wts, 0.7, 0.7, 0.7, 0.7, 0.3], R, t, k, res)
    log("rigid fit incl. ear: mean %.1f px, ear %.1f px" % (e_r.mean(), e_r[-1]))
    # torso check: where the front of the knit collar lands in the photo frame.  (Solving an
    # extra neck pitch for it was tried; even ~7 deg pushed the back of the head into the
    # collar and broke the hair curtain at the nape, so the residual is only reported.)
    collar_pt = np.array([0.0, -0.092, 1.368])
    uvc = pf.project(collar_pt[None], R, t, k, res)[0][0]
    res_c = np.linalg.norm(uvc - np.array(pf.PHOTO_COLLAR, float))
    log("collar front: model %s photo %s (%.1f px)" % (np.round(uvc, 1), pf.PHOTO_COLLAR, res_c))
    rep["collar_residual_px"] = round(float(res_c), 1)
    view_l = Rh.T @ (-R[2])                                       # camera forward, head-local
    for it in range(4):                       # eyes -> lids -> nose/mouth/chin; last pass eyes only
        lm, tris, names, P, uv, z, tgt = measure(rest, "rigid" if it == 0 else "iter%d" % it)
        idx = {n: i for i, n in enumerate(names)}
        ecs = rest_eye_centres(rest)
        for side in ("L", "R"):
            ec = ecs[side][0]
            keys = ["eye_%s_%s" % (side, q) for q in ("outer", "inner", "iris", "upper", "lower")]
            src = np.array([uv[idx[q]] for q in keys])
            dst = np.array([pf.PHOTO_LM[q] for q in keys], float)
            # similarity from corners, iris and lid mid-point
            s_src = np.vstack([src[:3], (src[3] + src[4]) / 2])
            s_dst = np.vstack([dst[:3], (dst[3] + dst[4]) / 2])
            sc, th, tr = pf.similarity_2d(s_src, s_dst)
            sc = float(np.clip(sc, 0.85, 1.2))
            ecw = to_w(ec[None])[0]
            ec_uv, ec_z = pf.project(ecw[None], R, t, k, res)
            new_uv = sc * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) @ ec_uv[0] + tr
            shift_w = pf.unproject(new_uv[None], ec_z, R, t, k, res)[0] - ecw
            shift_l = Rh.T @ shift_w
            # image y is down, so a positive image angle is a clockwise turn about the view axis
            Rot = axis_rotation(view_l, th)
            dist = np.linalg.norm(rest - ec, axis=1)
            w = smoothstep(0.030, 0.016, dist)
            moved = (rest - ec) @ Rot.T * sc + ec + shift_l
            rest = rest + (moved - rest) * w[:, None]
            log("eye %s: scale %.3f rot %.1f deg shift %.1f mm" % (side, sc, math.degrees(th), 1000 * np.linalg.norm(shift_l)))
        # lid opening, measured after the similarity step
        lm, tris, names, P, uv, z, tgt = measure(rest, "eyes%d" % it)
        idx = {n: i for i, n in enumerate(names)}
        ecs = rest_eye_centres(rest)
        for side in ("L", "R"):
            up_m, lo_m = uv[idx["eye_%s_upper" % side]], uv[idx["eye_%s_lower" % side]]
            ir_m = uv[idx["eye_%s_iris" % side]]
            up_p, lo_p = (np.array(pf.PHOTO_LM["eye_%s_%s" % (side, q)], float) for q in ("upper", "lower"))
            ir_p = np.array(pf.PHOTO_LM["eye_%s_iris" % side], float)
            f_up = np.clip(np.linalg.norm(up_p - ir_p) / max(np.linalg.norm(up_m - ir_m), 1e-6) - 1, -0.3, 0.6)
            f_lo = np.clip(np.linalg.norm(lo_p - ir_p) / max(np.linalg.norm(lo_m - ir_m), 1e-6) - 1, -0.3, 0.15)
            # the eye's own up axis (image-plane perpendicular to the photo's corner line), head-local
            ax2 = np.array(pf.PHOTO_LM["eye_%s_outer" % side], float) - np.array(pf.PHOTO_LM["eye_%s_inner" % side], float)
            n2 = np.array([ax2[1], -ax2[0]]) / np.linalg.norm(ax2)
            if n2[1] > 0:
                n2 = -n2
            up_w = R.T @ np.array([n2[0], -n2[1], 0.0])
            up_l = Rh.T @ up_w
            up_l[1] = 0.0
            up_l /= np.linalg.norm(up_l)
            rest = eye_open_warp(rest, ecs[side][0], up_l, 0.8 * f_up, 0.8 * f_lo)
            log("eye %s: lid opening upper %+.2f lower %+.2f" % (side, f_up, f_lo))
            # where the iris sits along the eye axis (photo vs model), corrected by a lid rotation
            o_m, i_m = uv[idx["eye_%s_outer" % side]], uv[idx["eye_%s_inner" % side]]
            o_p = np.array(pf.PHOTO_LM["eye_%s_outer" % side], float)
            i_p = np.array(pf.PHOTO_LM["eye_%s_inner" % side], float)
            axm = (o_m - i_m) / np.linalg.norm(o_m - i_m)
            axp = (o_p - i_p) / np.linalg.norm(o_p - i_p)
            off_m = (ir_m - (o_m + i_m) / 2) @ axm
            off_p = (ir_p - (o_p + i_p) / 2) @ axp
            need_px = off_m - off_p                    # >0: move the opening toward the outer corner
            ec_w = to_w(ecs[side][0][None])[0]
            px_per_m = k / np.linalg.norm(ec_w - t)
            ang = 1.6 * need_px / px_per_m / ecs[side][1]      # over-relaxed: the other steps pull back
            # sign: a positive angle must move the lids (and so the opening) along +axm in the image
            test = ecs[side][0] + np.array([0.0, -0.012, 0.0])     # a point on the lids (front)
            Rt = np.array(Matrix.Rotation(0.05, 3, Vector(up_l.tolist())))
            d_img = pf.project(to_w((ecs[side][0] + Rt @ (test - ecs[side][0]))[None]), R, t, k, res)[0][0] - \
                pf.project(to_w(test[None]), R, t, k, res)[0][0]
            if d_img @ axm < 0:
                ang = -ang
            ang = float(np.clip(ang, -0.25, 0.25))
            if abs(need_px) > 1.0:
                rest = lid_rotate(rest, ecs[side][0], up_l, ang)
            log("eye %s: iris offset model %.1f photo %.1f px -> lid rotation %.1f deg" % (side, off_m, off_p, math.degrees(ang)))
        if it == 3:
            break
        # nose / lips / chin
        lm, tris, names, P, uv, z, tgt = measure(rest, "nm%d" % it)
        co_w = to_w(rest)
        fn = np.cross(rest[tris[:, 1]] - rest[tris[:, 0]], rest[tris[:, 2]] - rest[tris[:, 0]])
        vn = np.zeros_like(rest)
        for j in range(3):
            np.add.at(vn, tris[:, j], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]
        down = np.array([0.40, 0.92])
        ci = pf.chin_point(co_w, vn @ Rh.T, rest, t, R, t, k, res, down)
        if os.environ.get("FIT_DEBUG"):
            log("   chin vertex rest %s at %s" % (np.round(rest[ci], 4), np.round(pf.project(co_w[ci][None], R, t, k, res)[0][0], 1)))
        sel = [i for i, n in enumerate(names) if not n.startswith("eye_")]
        Pw2 = np.vstack([to_w(P[sel]), co_w[ci]])
        tgt2 = np.vstack([tgt[sel], pf.PHOTO_LM["chin"]])
        cl_list = [P[sel], rest[ci][None]]
        # cheek/jaw silhouette on the image-left side
        uv_all, z_all = pf.project(co_w, R, t, k, res)
        tocam = t - co_w
        tocam /= np.linalg.norm(tocam, axis=1)[:, None]
        facing = ((vn @ Rh.T) * tocam).sum(1)
        X_, Y_, Z_ = rest[:, 0], rest[:, 1], rest[:, 2]
        not_ear = ~((np.abs(X_) > 0.062) & (Y_ > -0.065) & (Z_ > 1.48))
        cand = np.nonzero((X_ < -0.02) & (Y_ < -0.01) & (Z_ > 1.40) & (Z_ < 1.56) & not_ear)[0]
        centre = np.array(pf.PHOTO_LM["nose_tip"], float)
        extra_P, extra_T = [], []
        for q in pf.PHOTO_CONTOUR:
            q = np.array(q, float)
            vi, gap = pf.silhouette_vertex(uv_all, facing, cand, centre, q)
            if vi is None:
                continue
            dvec = (q - centre) / np.linalg.norm(q - centre)
            extra_P.append(rest[vi])
            extra_T.append(uv_all[vi] + dvec * gap)
        # ears: centroid / lowest point of the outer ear (subject's left ear is visible)
        ear = (X_ > 0.066) & (Z_ > 1.49) & (Z_ < 1.585) & (Y_ > -0.06) & (Y_ < 0.02)
        if ear.sum() > 20:
            ec_ = rest[ear].mean(0)
            lobe = rest[ear][np.argmin(rest[ear][:, 2])]
            for p3, nm_ in ((ec_, "ear_L"), (lobe, "earlobe_L")):
                extra_P.append(p3)
                extra_T.append(np.array(pf.PHOTO_EAR[nm_], float))
        n_extra = len(extra_P)
        if n_extra:
            Pw2 = np.vstack([Pw2, to_w(np.array(extra_P))])
            tgt2 = np.vstack([tgt2, np.array(extra_T)])
            cl_list.append(np.array(extra_P))
        uv2, z2 = pf.project(Pw2, R, t, k, res)
        dl = (pf.unproject(tgt2, z2, R, t, k, res) - Pw2) @ Rh
        cl = np.vstack(cl_list)
        if n_extra and ear.sum() > 20:
            # the hidden right ear goes to the mirror image of the corrected left ear
            earR = (X_ < -0.066) & (Z_ > 1.49) & (Z_ < 1.585) & (Y_ > -0.06) & (Y_ < 0.02)
            if earR.sum() > 20:
                pr_c = rest[earR].mean(0)
                pr_l = rest[earR][np.argmin(rest[earR][:, 2])]
                mir = np.array([-1, 1, 1])
                for pR, pL, dL in ((pr_c, cl[-2], dl[-2]), (pr_l, cl[-1], dl[-1])):
                    cl = np.vstack([cl, pR])
                    dl = np.vstack([dl, (pL + dL) * mir - pR])
        err_c = np.linalg.norm(uv2 - tgt2, axis=1)[-n_extra:] if n_extra else []
        if os.environ.get("FIT_DEBUG"):
            for a_, b_, c_ in zip(uv2[-n_extra:], tgt2[-n_extra:], cl_list[-1]):
                log("   extra at %s -> %s  rest %s" % (np.round(a_, 1), np.round(b_, 1), np.round(c_, 4)))
        log("contour/ear residuals px: %s" % np.round(err_c, 1).tolist())
        ecs = rest_eye_centres(rest)
        anchors = np.array([(0, -0.12, 1.64), (0.06, -0.08, 1.62), (-0.06, -0.08, 1.62), (0.045, 0.03, 1.55),
                            (-0.045, 0.03, 1.55), (0.04, -0.02, 1.40), (-0.04, -0.02, 1.40), (0, -0.07, 1.39),
                            (0, 0.04, 1.55), (0.0, 0.0, 1.66)] + [ecs[s_][0] + o for s_ in "LR" for o in
                                                ((0, -0.016, 0.006), (0, -0.016, -0.006), (0.009, -0.013, 0), (-0.009, -0.013, 0))])
        cen = np.vstack([cl, anchors])
        disp = np.vstack([dl, np.zeros_like(anchors)])
        face = smoothstep(0.035, -0.01, rest[:, 1]) * smoothstep(1.395, 1.425, rest[:, 2])
        d = pf.rbf_displacement(cen, disp, rest, h=0.045, lam=1e-3) * face[:, None]
        mv = 1000 * np.linalg.norm(dl, axis=1)
        log("warp: max landmark move %.2f mm (constraint #%d of %d)" % (mv.max(), int(np.argmax(mv)), len(mv)))
        rest = rest + d
    lm, tris, names, P, uv, z, tgt = measure(rest, "final")
    bpy.data.objects.remove(tmp_ob)
    bpy.data.meshes.remove(tmp_me)
    FIT_UV.clear()
    FIT_UV.update({n: uv[i] for i, n in enumerate(names)})
    M = Matrix(R.T.tolist())
    CAM_ROT = M
    CAM_BEAUTY = Vector(t.tolist())
    BEAUTY_TARGET = CAM_BEAUTY - M.col[2] * 0.45
    rep.update({"camera_location": [round(float(x), 4) for x in t], "lens_mm": CAM_LENS,
                "landmarks": list(pf.PHOTO_LM.keys()),
                "silhouette_points": {"cheek_contour": pf.PHOTO_CONTOUR, **pf.PHOTO_EAR},
                "collar_check_point": pf.PHOTO_COLLAR,
                "landmark_source": "picked by eye on magnified, pixel-gridded crops of input/reference.png"})
    return rest


# ----------------------------------------------------------------------------
# main build (continued in later sections)
# ----------------------------------------------------------------------------
def build_body(C):
    bpy.ops.wm.obj_import(filepath=os.path.join(INPUT, "upper-body.obj"), forward_axis='Y', up_axis='Z')
    ob = bpy.context.selected_objects[0]
    for c in ob.users_collection:
        c.objects.unlink(ob)
    C["char"].objects.link(ob)
    ob.name = "Body_Skin"
    me = ob.data
    base_counts = (len(me.vertices), len(me.polygons))
    REPORT["base_mesh"] = {"vertices": base_counts[0], "faces": base_counts[1]}
    orig = get_co(me)
    set_attr_vec(me, "orig_co", orig)
    rest = sculpt(orig)
    set_co(me, rest)
    apply_subdiv(ob, 1)
    me = ob.data
    rest = get_co(me)                       # smoothed rest positions after one CC level
    rest = fit_to_photo(me, rest)            # camera fit + landmark warp toward the photo
    set_co(me, rest)
    set_attr_vec(me, "rest_co", rest)
    T = pose_definition()
    if os.environ.get("NOPOSE"):
        for k in list(T):
            if not k.startswith("_"):
                T[k] = Matrix.Identity(4)
    posed = pose_mesh(ob, rest, T, C["char"])
    for p in me.polygons:
        p.use_smooth = True
    log("body posed", len(me.vertices), "verts")
    return ob, T, rest, posed


# ----------------------------------------------------------------------------
# node helpers
# ----------------------------------------------------------------------------
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    return m, m.node_tree


def N(nt, kind, loc=(0, 0), **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def L(nt, a, b):
    nt.links.new(a, b)


def math_node(nt, op, a=None, b=None, loc=(0, 0), clamp=False):
    n = N(nt, "ShaderNodeMath", loc, operation=op, use_clamp=clamp)
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            L(nt, v, n.inputs[i])
    return n.outputs[0]


def map_range(nt, val, f0, f1, t0=0.0, t1=1.0, loc=(0, 0), interp='SMOOTHSTEP'):
    n = N(nt, "ShaderNodeMapRange", loc, interpolation_type=interp, clamp=True)
    L(nt, val, n.inputs["Value"])
    n.inputs["From Min"].default_value = f0
    n.inputs["From Max"].default_value = f1
    n.inputs["To Min"].default_value = t0
    n.inputs["To Max"].default_value = t1
    return n.outputs[0]


def mix_rgb(nt, fac, a, b, loc=(0, 0), blend='MIX'):
    n = N(nt, "ShaderNodeMix", loc, data_type='RGBA', blend_type=blend, clamp_result=True)
    for sock, v in ((n.inputs[0], fac), (n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, tuple):
            sock.default_value = v if len(v) == 4 else (*v, 1.0)
        else:
            L(nt, v, sock)
    return n.outputs[2]


def srgb(*c):
    def f(x):
        x = x / 255.0
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    return np.array([f(x) for x in c])


def principled(nt, loc=(0, 0)):
    return N(nt, "ShaderNodeBsdfPrincipled", loc)


def output(nt, shader, loc=(400, 0)):
    o = N(nt, "ShaderNodeOutputMaterial", loc)
    L(nt, shader, o.inputs["Surface"])
    return o


# ----------------------------------------------------------------------------
# 3. skin colour attributes + skin shader
# ----------------------------------------------------------------------------
def lip_mask(o):
    X, Y, Z = o[:, 0], o[:, 1], o[:, 2]
    ax = np.abs(X)
    q = np.clip(1 - (ax / 0.0262) ** 2, 0, 1)
    zm = mouth_line(X)
    ztop = zm + 0.0088 * q ** 0.6 - 0.0010 * np.exp(-(X / 0.0032) ** 2)
    zbot = zm - 0.0098 * q ** 0.7
    inside = smoothstep(-0.0010, 0.0010, ztop - Z) * smoothstep(-0.0012, 0.0012, Z - zbot)
    inside *= smoothstep(0.0285, 0.0255, ax) * smoothstep(-0.128, -0.138, Y)
    return inside


def skin_attributes(me):
    o = attr_vec(me, "orig_co")
    X, Y, Z = o[:, 0], o[:, 1], o[:, 2]
    ax = np.abs(X)
    n = len(o)
    col = np.tile(srgb(238, 186, 160), (n, 1))
    rough = np.full(n, 0.47)

    def mix(mask, c):
        m = np.clip(mask, 0, 1)[:, None]
        col[:] = col * (1 - m) + np.asarray(c) * m

    # subtle warmth gradient: slightly rosier mid-face, a touch paler neck/chest
    mix(0.30 * smoothstep(1.47, 1.40, Z), srgb(236, 200, 182))
    mix(0.62 * sbump(o, (0.043, -0.128, 1.51), (0.03, 0.03, 0.025)), srgb(238, 146, 140))
    mix(0.30 * bump(o, (0, -0.162, 1.514), (0.016, 0.02, 0.014)), srgb(234, 172, 162))
    mix(0.20 * bump(o, (0, -0.145, 1.452), (0.02, 0.02, 0.012)), srgb(234, 178, 168))
    # soft brown-pink lid tint (natural makeup look in the reference)
    mix(0.22 * sbump(o, (0.031, -0.1415, 1.5575), (0.017, 0.012, 0.006)), srgb(214, 160, 148))
    # ears a little redder (thin tissue)
    mix(0.30 * smoothstep(0.062, 0.072, ax) * smoothstep(1.48, 1.50, Z) * smoothstep(1.58, 1.56, Z), srgb(232, 164, 152))
    # lips
    lm = lip_mask(o)
    mix(0.85 * lm, srgb(222, 118, 124))
    rough = rough * (1 - lm) + 0.27 * lm
    # mouth interior / lip seam
    mi = smoothstep(-0.140, -0.132, Y) * (ax < 0.028) * (Z > 1.455) * (Z < 1.50)
    mix(mi, srgb(150, 70, 72))
    # eye socket interior (conjunctiva / lid margin), wet
    for sgn in (-1, 1):
        ec = np.array([sgn * 0.0307, -0.1258, 1.5447])
        dist = np.linalg.norm(o - ec, axis=1)
        sm = smoothstep(0.0178, 0.0170, dist) * smoothstep(-0.1365, -0.1335, Y)
        mix(sm, srgb(196, 112, 106))
        rough = rough * (1 - sm) + 0.12 * sm
    # scalp painted hair-brown so no pale skin shows between strands
    sd_ = scalp_density(o)
    mix(np.clip(scalp_density(o + np.array([0, 0, 0.006])) * 1.6, 0, 1), srgb(84, 50, 36))
    # T-zone slightly glossier
    rough -= 0.06 * bump(o, (0, -0.15, 1.56), (0.02, 0.03, 0.06))
    set_color_attr(me, "skin_col", col)
    set_attr_float(me, "skin_rough", rough)


def skin_material(name="Skin", use_attr=True):
    m, nt = new_mat(name)
    bsdf = principled(nt, (200, 0))
    bsdf.subsurface_method = 'RANDOM_WALK_SKIN'
    if use_attr:
        a = N(nt, "ShaderNodeAttribute", (-900, 200), attribute_name="skin_col", attribute_type='GEOMETRY')
        r = N(nt, "ShaderNodeAttribute", (-900, -100), attribute_name="skin_rough", attribute_type='GEOMETRY')
        acol, rfac = a.outputs["Color"], r.outputs["Fac"]
    else:
        a = N(nt, "ShaderNodeRGB", (-900, 200))
        a.outputs[0].default_value = (*srgb(236, 196, 182), 1)
        r = N(nt, "ShaderNodeValue", (-900, -100))
        r.outputs[0].default_value = 0.45
        acol, rfac = a.outputs[0], r.outputs[0]
    tc = N(nt, "ShaderNodeTexCoord", (-1300, -300))
    # low-frequency mottling / faint freckles
    mot = N(nt, "ShaderNodeTexNoise", (-900, 450), noise_dimensions='3D')
    mot.inputs["Scale"].default_value = 180.0
    mot.inputs["Detail"].default_value = 4.0
    L(nt, tc.outputs["Object"], mot.inputs["Vector"])
    mot_f = map_range(nt, mot.outputs["Fac"], 0.35, 0.75, -0.035, 0.035, (-700, 450), 'LINEAR')
    frk = N(nt, "ShaderNodeTexVoronoi", (-900, 700))
    frk.inputs["Scale"].default_value = 520.0
    L(nt, tc.outputs["Object"], frk.inputs["Vector"])
    frk_m = map_range(nt, frk.outputs["Distance"], 0.0, 0.18, 1.0, 0.0, (-700, 700))
    frk_n = N(nt, "ShaderNodeTexNoise", (-900, 950))
    frk_n.inputs["Scale"].default_value = 60.0
    L(nt, tc.outputs["Object"], frk_n.inputs["Vector"])
    frk_g = map_range(nt, frk_n.outputs["Fac"], 0.55, 0.7, 0.0, 0.10, (-700, 950))
    frk_amt = math_node(nt, 'MULTIPLY', frk_m, frk_g, (-500, 800))
    col1 = mix_rgb(nt, frk_amt, acol, (0.42, 0.22, 0.15, 1), (-450, 300))
    hsv = N(nt, "ShaderNodeHueSaturation", (-250, 300))
    L(nt, col1, hsv.inputs["Color"])
    vadd = math_node(nt, 'ADD', mot_f, 1.0, (-450, 500))
    L(nt, vadd, hsv.inputs["Value"])
    L(nt, hsv.outputs["Color"], bsdf.inputs["Base Color"])
    # roughness micro variation
    rn = N(nt, "ShaderNodeTexNoise", (-900, -350))
    rn.inputs["Scale"].default_value = 900.0
    L(nt, tc.outputs["Object"], rn.inputs["Vector"])
    rv = map_range(nt, rn.outputs["Fac"], 0.3, 0.7, -0.05, 0.05, (-700, -350), 'LINEAR')
    rsum = math_node(nt, 'ADD', rfac, rv, (-450, -200))
    L(nt, rsum, bsdf.inputs["Roughness"])
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.42, 0.24)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0032
    bsdf.inputs["Subsurface IOR"].default_value = 1.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.42
    # pores + fine micro-relief (restrained)
    vor = N(nt, "ShaderNodeTexVoronoi", (-900, -650), feature='SMOOTH_F1')
    vor.inputs["Scale"].default_value = 1700.0
    L(nt, tc.outputs["Object"], vor.inputs["Vector"])
    pore = map_range(nt, vor.outputs["Distance"], 0.0, 0.35, 0.0, 1.0, (-700, -650))
    fine = N(nt, "ShaderNodeTexNoise", (-900, -900))
    fine.inputs["Scale"].default_value = 4200.0
    fine.inputs["Detail"].default_value = 2.0
    L(nt, tc.outputs["Object"], fine.inputs["Vector"])
    h = math_node(nt, 'ADD', pore, math_node(nt, 'MULTIPLY', fine.outputs["Fac"], 0.6, (-600, -900)), (-450, -700))
    bmp = N(nt, "ShaderNodeBump", (-200, -600))
    bmp.inputs["Strength"].default_value = 0.10
    bmp.inputs["Distance"].default_value = 0.00012
    L(nt, h, bmp.inputs["Height"])
    L(nt, bmp.outputs["Normal"], bsdf.inputs["Normal"])
    output(nt, bsdf.outputs[0], (500, 0))
    return m


# ----------------------------------------------------------------------------
# 3b. photographic skin layer: reference photo -> SkinUV texture via the fitted camera
# ----------------------------------------------------------------------------
TEX_SIZE = 2048


def srgb_to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def skin_uv_unwrap(body):
    me = body.data
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    uv = me.uv_layers.new(name="SkinUV")
    me.uv_layers.active = uv
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(62), island_margin=0.004, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    return uv


SKIN_GAIN = 0.74                     # photo skin -> albedo scale (the photo is bright/over-exposed)


def photo_skin_texture(body, cam, T):
    """Project the cleaned reference photo into the SkinUV layout of Body_Skin."""
    t0 = time.time()
    skin_uv_unwrap(body)
    photo = ptex.load_rgb(os.path.join(INPUT, "reference.png"))
    clean, skinmask, lips = ptex.photo_skin_layers(photo, pf.PHOTO_LM)
    log("photo cleaned (%.1fs)" % (time.time() - t0))
    P, Nw, cov = ptex.bake_attribute_maps(body, "SkinUV", TEX_SIZE)
    log("baked position/normal maps (%.1fs)" % (time.time() - t0))
    res = pf.PHOTO_SIZE
    R, tcam, k = pf.cam_params(cam, res)
    Pf = P.reshape(-1, 3)
    Nf = Nw.reshape(-1, 3)
    Nf = Nf / np.maximum(np.linalg.norm(Nf, axis=1), 1e-6)[:, None]
    uv, z = pf.project(Pf, R, tcam, k, res)
    tocam = tcam - Pf
    tocam /= np.maximum(np.linalg.norm(tocam, axis=1), 1e-9)[:, None]
    facing = (Nf * tocam).sum(1)
    # bake margins extend P/N past the island borders: treat them like interior texels so the
    # filtered texture has no seams (only texels the bake never touched are skipped)
    covf = cov.ravel() | (np.linalg.norm(Pf, axis=1) > 0.1)
    inframe = (uv[:, 0] > 2) & (uv[:, 0] < res[0] - 3) & (uv[:, 1] > 2) & (uv[:, 1] < res[1] - 3) & (z > 0)
    cand = covf & inframe & (facing > 0.08)
    # self-occlusion from the camera (half-resolution ray casts, nearest upsampling)
    S = TEX_SIZE
    sub = np.zeros((S, S), bool)
    sub[::2, ::2] = cand.reshape(S, S)[::2, ::2]
    vis_s = ptex.visibility(body, P, tcam, sub)
    vis = np.repeat(np.repeat(vis_s[::2, ::2], 2, 0), 2, 1).ravel()
    log("visibility (%.1fs)" % (time.time() - t0))
    # landmark morph: sample the photo where *its* feature is, for the texel's mesh feature
    names = [n for n in FIT_UV if n in pf.PHOTO_LM]
    src = np.array([FIT_UV[n] for n in names], float)
    dst = np.array([pf.PHOTO_LM[n] for n in names], float)
    ring = []
    for a in np.radians(np.arange(0, 360, 30)):
        ring.append([640 + 300 * math.cos(a), 430 + 330 * math.sin(a)])
    ring = np.array(ring)
    morph = ptex.tps_morph(np.vstack([src, ring]), np.vstack([dst, ring]), lam=30.0)
    idx = np.nonzero(cand & vis)[0]
    q = uv[idx]
    q = q + morph(q)
    col = ptex.bilinear(clean, q[:, 0], q[:, 1])
    # confidence: facing the camera, inside the face/neck region of the photo, not scalp
    roi = (((q[:, 0] - 640) / 250.0) ** 2 + ((q[:, 1] - 420) / 285.0) ** 2)
    w = ptex.smoothstep(0.25, 0.55, facing[idx]) * ptex.smoothstep(1.0, 0.8, roi)
    # stop well above the photographed collar: its shadow line would otherwise be painted onto
    # the neck above the (slightly lower) 3-D collar as a second neckline
    w *= ptex.smoothstep(622, 598, q[:, 1])
    Ah = np.array(T["head"])
    Hi = np.linalg.inv(Ah)
    loc = Pf[idx] @ Hi[:3, :3].T + Hi[:3, 3]
    w *= 1.0 - np.clip(scalp_density(loc + np.array([0, 0, 0.004])) * 1.4, 0, 1)
    rgba = np.zeros((S * S, 4), np.float32)
    lin = srgb_to_lin(col) * SKIN_GAIN
    rgba[idx, :3] = lin_to_srgb(lin)
    rgba[idx, 3] = w
    # close hairline gaps in the confidence (nostril rims, lid creases, half-res visibility
    # steps) so the procedural layer does not show through as thin lines
    a2 = rgba[:, 3].reshape(S, S)
    a2 = np.maximum(a2, np.minimum(ptex.blur(a2, 2.5) * 1.6, 1.0) * (a2 > 0.0))
    rgba[:, 3] = np.maximum(a2.ravel(), np.minimum(ptex.blur(a2, 1.2).ravel() * 1.3, 1.0))
    # dilate colour into the unconfident texels (alpha stays 0 there) for clean filtering
    rgb_full = ptex.pull_push(rgba[:, :3].reshape(S, S, 3), (rgba[:, 3] > 0.02).reshape(S, S).astype(np.float32), levels=6)
    rgba[:, :3] = rgb_full.reshape(-1, 3)
    img = bpy.data.images.new("Skin_Photo_RGBA", S, S, alpha=True)
    img.colorspace_settings.name = 'sRGB'
    img.alpha_mode = 'CHANNEL_PACKED'
    img.pixels.foreach_set(rgba.reshape(S, S, 4).ravel())          # Blender rows: bottom-up = UV v
    img.pack()
    # median photo skin colour (linear albedo) -> used to tint the procedural layer
    sk = (w > 0.8)
    med = np.median(lin[sk], axis=0) if sk.any() else srgb(236, 196, 182)
    REPORT["textures"].append({
        "name": "Skin_Photo_RGBA", "size": [S, S], "uv_map": "SkinUV (Smart UV Project, 62 deg, packed in .blend)",
        "source": "input/reference.png projected through Cam_Beauty (camera fitted to 19 photo landmarks)",
        "method": "Cycles-baked world position/normal per texel -> perspective projection into the photo -> "
                  "thin-plate landmark morph -> bilinear sample of the cleaned photo (hair/eyes/sweater masked "
                  "and pull-push filled, low-frequency lighting partly divided out). Alpha = facing x visibility "
                  "x face-region confidence; alpha-0 texels show the procedural skin layer.",
        "texels_from_photo": int((w > 0.5).sum())})
    log("photo skin texture done (%.1fs), %d confident texels, median albedo %s" % (
        time.time() - t0, int((w > 0.5).sum()), np.round(med, 3)))
    return img, med


def photo_iris_layers(cam, photo=None):
    """Per-eye photo iris texture (planar along the gaze axis) mixed over the procedural iris."""
    res = pf.PHOTO_SIZE
    R, tcam, k = pf.cam_params(cam, res)
    photo = ptex.load_rgb(os.path.join(INPUT, "reference.png")) if photo is None else photo
    names = [n for n in FIT_UV if n in pf.PHOTO_LM]
    src = np.array([FIT_UV[n] for n in names], float)
    dst = np.array([pf.PHOTO_LM[n] for n in names], float)
    morph = ptex.tps_morph(src, dst, lam=30.0)
    base = bpy.data.materials["Eye_Sclera_Iris"]
    for side in ("L", "R"):
        eb = bpy.data.objects["Eyeball." + side]
        r_eye = max(v.co.length for v in eb.data.vertices)
        pts = [pf.PHOTO_LM["eye_%s_%s" % (side, q)] for q in ("outer", "inner", "upper", "lower")]
        img, S = ptex.photo_iris_image("Iris_Photo." + side, photo, eb, r_eye, IRIS_R,
                                       lambda P: pf.project(P, R, tcam, k, res)[0], morph, pts, gain=SKIN_GAIN * 0.62)
        m = base.copy()
        m.name = "Eye_Sclera_Iris." + side
        nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        proc = bsdf.inputs["Base Color"].links[0].from_socket
        tc = N(nt, "ShaderNodeTexCoord", (-900, 900))
        mp = N(nt, "ShaderNodeMapping", (-700, 900))
        mp.inputs["Location"].default_value = (0.5, 0.5, 0.0)
        mp.inputs["Scale"].default_value = (0.5 / S, 0.5 / S, 1.0)
        L(nt, tc.outputs["Object"], mp.inputs["Vector"])
        tex = N(nt, "ShaderNodeTexImage", (-500, 900), image=img, extension='CLIP', interpolation='Cubic')
        tex.name = tex.label = "Iris_Photo"
        L(nt, mp.outputs[0], tex.inputs[0])
        out = mix_rgb(nt, tex.outputs["Alpha"], proc, tex.outputs["Color"], (300, 700))
        L(nt, out, bsdf.inputs["Base Color"])
        eb.data.materials.clear()
        eb.data.materials.append(m)
        REPORT["textures"].append({
            "name": img.name, "size": list(img.size),
            "uv_map": "none - object-space planar mapping along the eyeball's gaze axis (Mapping node)",
            "source": "input/reference.png iris pixels projected through Cam_Beauty with the landmark morph",
            "method": "catchlight pixels removed and pull-push filled; alpha = visible iris inside the photo "
                      "eye opening; the procedural iris/sclera shows elsewhere"})
    bpy.data.materials.remove(base)
    log("photo iris layers done")


def tint_procedural_skin(body, hand_mat, med):
    """Match the procedural (unobserved-area) skin layer to the photo's median complexion."""
    me = body.data
    a = me.color_attributes["skin_col"]
    rgba = np.empty(len(a.data) * 4, np.float32)
    a.data.foreach_get("color", rgba)
    rgba = rgba.reshape(-1, 4)
    base = srgb(238, 186, 160)
    ratio = np.clip(np.asarray(med) / base, 0.5, 2.0)
    rgba[:, :3] *= ratio
    a.data.foreach_set("color", rgba.ravel())
    rgb = next(n for n in hand_mat.node_tree.nodes if n.type == 'RGB')
    rgb.outputs[0].default_value = (*(srgb(236, 196, 182) * ratio), 1)
    REPORT["notes"].append("procedural skin layer tinted by %s to match the photo complexion" % np.round(ratio, 3).tolist())


def add_photo_layer(mat, img, med):
    """Insert the photo layer into the skin material: photo RGB over the procedural colour."""
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    link_in = bsdf.inputs["Base Color"].links[0]
    proc = link_in.from_socket
    uvn = N(nt, "ShaderNodeUVMap", (-1300, 1300), uv_map="SkinUV")
    tex = N(nt, "ShaderNodeTexImage", (-1050, 1300), image=img, interpolation='Cubic')
    tex.name = tex.label = "Skin_Photo_RGBA"
    L(nt, uvn.outputs[0], tex.inputs[0])
    out = mix_rgb(nt, tex.outputs["Alpha"], proc, tex.outputs["Color"], (0, 400))
    # lash tight-line over both layers (the 3-D lashes root on this line)
    la = N(nt, "ShaderNodeAttribute", (-300, 650), attribute_name="lash_line", attribute_type='GEOMETRY')
    out = mix_rgb(nt, math_node(nt, 'MULTIPLY', la.outputs["Fac"], 0.8, (-100, 650)), out,
                  (0.030, 0.014, 0.010, 1), (150, 500))
    L(nt, out, bsdf.inputs["Base Color"])


# ----------------------------------------------------------------------------
# 4. eyes: real eyeballs fitted to the lid sockets, separate glossy corneas
# ----------------------------------------------------------------------------
def uv_sphere(name, r, seg=64, rings=48):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me)
    bm.free()
    return me


IRIS_R = 0.0069
PUPIL_R = 0.0020


def eye_material(r_eye):
    m, nt = new_mat("Eye_Sclera_Iris")
    bsdf = principled(nt, (500, 0))
    tc = N(nt, "ShaderNodeTexCoord", (-1600, 0))
    sep = N(nt, "ShaderNodeSeparateXYZ", (-1400, 0))
    L(nt, tc.outputs["Object"], sep.inputs[0])
    ln = N(nt, "ShaderNodeVectorMath", (-1400, -200), operation='LENGTH')
    L(nt, tc.outputs["Object"], ln.inputs[0])
    cz = math_node(nt, 'DIVIDE', sep.outputs["Z"], ln.outputs["Value"], (-1200, 0))
    alpha = math_node(nt, 'ARCCOSINE', cz, None, (-1050, 0))
    a_i = math.asin(IRIS_R / r_eye)
    t = math_node(nt, 'DIVIDE', alpha, a_i, (-900, 0))               # 0 centre .. 1 limbus
    az = math_node(nt, 'ARCTAN2', sep.outputs["Y"], sep.outputs["X"], (-1200, -300))
    comb = N(nt, "ShaderNodeCombineXYZ", (-750, -250))
    L(nt, math_node(nt, 'MULTIPLY', az, 6.0, (-950, -300)), comb.inputs[0])
    L(nt, t, comb.inputs[1])
    fib = N(nt, "ShaderNodeTexNoise", (-550, -250))
    fib.inputs["Scale"].default_value = 2.2
    fib.inputs["Detail"].default_value = 9.0
    fib.inputs["Roughness"].default_value = 0.65
    L(nt, comb.outputs[0], fib.inputs["Vector"])
    fib2 = N(nt, "ShaderNodeTexNoise", (-550, -500))
    fib2.inputs["Scale"].default_value = 7.0
    fib2.inputs["Detail"].default_value = 4.0
    L(nt, comb.outputs[0], fib2.inputs["Vector"])
    ramp = N(nt, "ShaderNodeValToRGB", (-350, -250))
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.32, (0.030, 0.012, 0.005, 1)
    cr.elements[1].position, cr.elements[1].color = 0.78, (0.15, 0.062, 0.022, 1)
    e = cr.elements.new(0.52)
    e.color = (0.075, 0.03, 0.011, 1)
    L(nt, fib.outputs["Fac"], ramp.inputs[0])
    # inner amber collarette, darker limbal ring
    inner = map_range(nt, t, 0.62, 0.36, 0, 1, (-350, 100))
    iris = mix_rgb(nt, math_node(nt, 'MULTIPLY', inner, 0.55, (-200, 150)), ramp.outputs[0], (0.25, 0.11, 0.035, 1), (-100, 0))
    blot = map_range(nt, fib2.outputs["Fac"], 0.45, 0.75, 0.0, 0.35, (-350, -450))
    iris = mix_rgb(nt, blot, iris, (0.05, 0.02, 0.008, 1), (50, -100))
    limb = map_range(nt, t, 0.80, 1.0, 0, 1, (-350, 300))
    iris = mix_rgb(nt, limb, iris, (0.022, 0.010, 0.006, 1), (150, 0))
    pupil = map_range(nt, t, PUPIL_R / IRIS_R + 0.02, PUPIL_R / IRIS_R - 0.02, 0, 1, (-350, 450))
    iris = mix_rgb(nt, pupil, iris, (0.003, 0.002, 0.002, 1), (250, 100))
    # sclera with faint warm/red periphery
    sc_n = map_range(nt, alpha, 0.9, 1.6, 0, 1, (-350, 600))
    sclera = mix_rgb(nt, sc_n, (0.38, 0.335, 0.315, 1), (0.34, 0.25, 0.22, 1), (50, 600))
    # contact shadow where the lids and lashes hug the eyeball
    ao = N(nt, "ShaderNodeAmbientOcclusion", (-100, 800), samples=8, only_local=False)
    ao.inputs["Distance"].default_value = 0.004
    sclera = mix_rgb(nt, map_range(nt, ao.outputs["AO"], 0.2, 1.0, 0.65, 0.0, (100, 850)), sclera, (0.16, 0.10, 0.09, 1), (250, 650))
    is_sclera = map_range(nt, t, 0.985, 1.06, 0, 1, (-350, 750))
    col = mix_rgb(nt, is_sclera, iris, sclera, (350, 300))
    L(nt, col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Subsurface Weight"].default_value = 0.25
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.5, 0.35)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0015
    output(nt, bsdf.outputs[0], (800, 0))
    return m


def cornea_material():
    m, nt = new_mat("Eye_Cornea")
    tr = N(nt, "ShaderNodeBsdfTransparent", (0, 100))
    gl = N(nt, "ShaderNodeBsdfGlossy", (0, -100))
    gl.inputs["Roughness"].default_value = 0.015
    fr = N(nt, "ShaderNodeFresnel", (0, 250))
    fr.inputs["IOR"].default_value = 1.376
    mx = N(nt, "ShaderNodeMixShader", (200, 0))
    L(nt, fr.outputs[0], mx.inputs[0])
    L(nt, tr.outputs[0], mx.inputs[1])
    L(nt, gl.outputs[0], mx.inputs[2])
    output(nt, mx.outputs[0], (400, 0))
    return m


def build_eyes(C, body, T):
    Ah = T["head"]
    Ahi = Ah.inverted()
    co, _ = evaluated_mesh_arrays(body)
    R = np.array(Ahi)
    loc = co @ R[:3, :3].T + R[:3, 3]
    target = CAM_BEAUTY + Vector((0.0, 0.0, -0.012))
    eyes = {}
    m_eye = None
    m_cor = cornea_material()
    for side, sgn in (("L", 1), ("R", -1)):
        ec0 = np.array([sgn * 0.0307, -0.1258, 1.5447])
        dist = np.linalg.norm(loc - ec0, axis=1)
        sel = (dist > 0.011) & (dist < 0.0195) & (loc[:, 1] > ec0[1] - 0.0115)
        P = loc[sel]
        for _ in range(4):
            c, r = fit_sphere(P)
            res = np.abs(np.linalg.norm(P - c, axis=1) - r)
            P = P[res < max(0.0008, 2.5 * res.std())]
        c, r = fit_sphere(P)
        log("eye", side, "socket fit centre", c.round(4), "r=%.4f" % r, "n=%d" % len(P))
        r_eye = r - 0.00045
        if m_eye is None:
            m_eye = eye_material(r_eye)
        cw = Ah @ Vector(c)
        gaze = (target - cw).normalized()
        q = gaze.to_track_quat('Z', 'Y')
        eb = link(bpy.data.objects.new("Eyeball." + side, uv_sphere("Eyeball." + side, r_eye)), C["eyes"])
        eb.location = cw
        eb.rotation_euler = q.to_euler()
        eb.data.materials.append(m_eye)
        # cornea: slightly larger sphere with a smooth dome over the iris
        cm = uv_sphere("Cornea." + side, r_eye + 0.00022, 96, 64)
        cco = get_co(cm)
        u = cco / np.linalg.norm(cco, axis=1)[:, None]
        al = np.arccos(np.clip(u[:, 2], -1, 1))
        ac = math.asin(IRIS_R / r_eye) * 1.18
        h = 0.0017 * np.clip(1 - (al / ac) ** 2, 0, 1) ** 1.6
        set_co(cm, cco + u * h[:, None])
        cob = link(bpy.data.objects.new("Cornea." + side, cm), C["eyes"])
        cob.location = cw
        cob.rotation_euler = q.to_euler()
        cm.materials.append(m_cor)
        cob.visible_shadow = False
        eyes[side] = (cw, r_eye, gaze)
    return eyes


def preview_lights(C):
    def area(name, loc, target, size, power, color):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.shape = 'RECTANGLE'
        ld.size, ld.size_y = size
        ld.energy = power
        ld.color = color
        ob = link(bpy.data.objects.new(name, ld), C["light"])
        ob.location = loc
        ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return ob
    area("Key_Window", (-1.35, -1.05, 2.05), (0, -0.1, 1.5), (1.4, 1.7), 95, (1.0, 0.95, 0.88))
    area("Fill_Front", (0.55, -1.3, 1.75), (0, -0.1, 1.5), (1.0, 1.0), 22, (1.0, 0.92, 0.86))
    area("Rim_Back", (-0.7, 1.1, 2.1), (0, 0, 1.55), (1.2, 1.2), 120, (1.0, 0.88, 0.74))
    area("Rim_Back_R", (0.9, 0.9, 1.8), (0, 0, 1.45), (1.0, 1.0), 50, (1.0, 0.9, 0.8))
    w = bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.62, 0.55, 0.48, 1)
    bg.inputs[1].default_value = 0.25


# ----------------------------------------------------------------------------
# 5. sweater: derived from the posed torso/arms, extended, inflated, draped
# ----------------------------------------------------------------------------
def neck_cover(rest, margin=0.0):
    """True where the crew-neck sweater covers the body (computed in rest space)."""
    X, Y, Z = rest[:, 0], rest[:, 1], rest[:, 2]
    # a wide, low crew neck as in the photo (front dips to the base of the neck)
    zN = 1.380 + 0.022 * np.clip((Y + 0.02) / 0.05, -1, 1) - margin
    inside = (X / 0.080) ** 2 + ((Y + 0.012) / 0.073) ** 2 < 1.0
    face_side = (Y < -0.045) & (np.abs(X) < 0.09)
    outside_lim = np.where(face_side, 1.372 - margin, 1.47 - margin)
    return np.where(inside, Z < zN, Z < outside_lim)


def boundary_loops(F):
    from collections import defaultdict
    cnt = defaultdict(int)
    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            cnt[(min(a, b), max(a, b))] += 1
    adj = defaultdict(list)
    for (a, b), c in cnt.items():
        if c == 1:
            adj[a].append(b)
            adj[b].append(a)
    seen, loops = set(), []
    for s0 in list(adj):
        if s0 in seen:
            continue
        loop, prev, cur = [s0], None, s0
        seen.add(s0)
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt or nxt[0] == s0:
                break
            prev, cur = cur, nxt[0]
            if cur in seen:
                break
            seen.add(cur)
            loop.append(cur)
        loops.append(loop)
    return loops


def mesh_adjacency(nv, F):
    E = set()
    for f in F:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            E.add((min(a, b), max(a, b)))
    E = np.array(sorted(E))
    return E


def laplacian(V, E, pin, iters=10, lam=0.5, mu=-0.53, wts=None):
    V = V.copy()
    nv = len(V)
    deg = np.zeros(nv)
    np.add.at(deg, E[:, 0], 1)
    np.add.at(deg, E[:, 1], 1)
    deg = np.maximum(deg, 1)
    w = np.ones(nv) if wts is None else wts
    free = (~pin).astype(float) * w
    for it in range(iters):
        for f in (lam, mu):
            acc = np.zeros_like(V)
            np.add.at(acc, E[:, 0], V[E[:, 1]])
            np.add.at(acc, E[:, 1], V[E[:, 0]])
            V += (f * free)[:, None] * (acc / deg[:, None] - V)
    return V


def vertex_normals(V, F):
    Nn = np.zeros_like(V)
    for f in F:
        if len(f) == 4:
            a, b, c, d = f
            n = np.cross(V[c] - V[a], V[d] - V[b])
        else:
            a, b, c = f[:3]
            n = np.cross(V[b] - V[a], V[c] - V[a])
        for i in f:
            Nn[i] += n
    return Nn / np.maximum(np.linalg.norm(Nn, axis=1), 1e-12)[:, None]


def vnormals_fast(V, Fq, Ft):
    Nn = np.zeros_like(V)
    if len(Fq):
        n = np.cross(V[Fq[:, 2]] - V[Fq[:, 0]], V[Fq[:, 3]] - V[Fq[:, 1]])
        for k in range(4):
            np.add.at(Nn, Fq[:, k], n)
    if len(Ft):
        n = np.cross(V[Ft[:, 1]] - V[Ft[:, 0]], V[Ft[:, 2]] - V[Ft[:, 0]])
        for k in range(3):
            np.add.at(Nn, Ft[:, k], n)
    return Nn / np.maximum(np.linalg.norm(Nn, axis=1), 1e-12)[:, None]


def frame_for(axis):
    a = Vector(axis).normalized()
    up = Vector((0, 0, 1))
    b1 = (up - a * up.dot(a))
    if b1.length < 1e-4:
        b1 = Vector((0, 1, 0))
    b1.normalize()
    b2 = a.cross(b1)
    return np.array(a), np.array(b1), np.array(b2)


def arm_coords(P, T, side):
    """arc length s from the shoulder and angle phi around the sleeve polyline."""
    S2, E2, W2 = (np.array(T["_S2." + side]), np.array(T["_E2." + side]), np.array(T["_W2." + side]))
    dirf = np.array(T["_dirf." + side])
    pts = [S2, E2, W2, W2 + dirf * 0.25]
    best_d = np.full(len(P), 1e9)
    s_out = np.zeros(len(P))
    phi_out = np.zeros(len(P))
    acc = 0.0
    for k in range(3):
        A, B = pts[k], pts[k + 1]
        seg = B - A
        Lk = np.linalg.norm(seg)
        a, b1, b2 = frame_for(seg)
        t = np.clip((P - A) @ a, 0 if k > 0 else -0.2, Lk)
        foot = A + t[:, None] * a
        perp = P - foot
        dd = np.linalg.norm(perp, axis=1)
        m = dd < best_d
        best_d[m] = dd[m]
        s_out[m] = acc + t[m]
        phi_out[m] = np.arctan2(perp[m] @ b2, perp[m] @ b1)
        acc += Lk
    return s_out, phi_out, best_d


def build_sweater(C, body, T, rest_body):
    me = body.data
    V0 = get_co(me)
    cover = neck_cover(rest_body)
    W = pose_weights(rest_body)
    armness = {sd: W["upper." + sd] + W["fore." + sd] for sd in ("L", "R")}
    polys = [list(p.vertices) for p in me.polygons]
    keep = [f for f in polys if cover[f].all()]
    # trim the obliquely-cut forearms with a plane perpendicular to the posed forearm
    trimmed = []
    cutinfo = {}
    for sd in ("L", "R"):
        E2, W2 = np.array(T["_E2." + sd]), np.array(T["_W2." + sd])
        a = np.array(T["_dirf." + sd])
        sgn = 1 if sd == "L" else -1
        armv = (armness[sd] > 0.5)
        tv = (V0 - E2) @ a
        # boundary verts of the original arm cut
        cutv = np.nonzero(armv & (rest_body[:, 2] < 1.10))[0]
        t_cut = tv[cutv].min() - 0.004
        cutinfo[sd] = (t_cut, E2, a, armv, tv)
    for f in keep:
        bad = False
        for sd in ("L", "R"):
            t_cut, E2, a, armv, tv = cutinfo[sd]
            if armv[f].all() and (tv[f] > t_cut).any():
                bad = True
        if not bad:
            trimmed.append(f)
    used = sorted({i for f in trimmed for i in f})
    remap = -np.ones(len(V0), int)
    remap[used] = np.arange(len(used))
    V = V0[used].copy()
    rest = rest_body[used].copy()
    F = [[int(remap[i]) for i in f] for f in trimmed]
    region = np.zeros(len(V))                      # 0 torso, +1 left sleeve, -1 right sleeve
    for sd, sg in (("L", 1), ("R", -1)):
        region[armness[sd][used] > 0.5] = sg
    loops = boundary_loops(F)
    for lp in loops:
        log("sweater boundary loop n=%d centre=%s" % (len(lp), V[lp].mean(0).round(3)))
    extra_V, extra_region = [], []
    newF = []
    nV = len(V)

    def add_vert(p, reg):
        nonlocal nV
        extra_V.append(p)
        extra_region.append(reg)
        nV += 1
        return nV - 1

    neck_loop = None
    for lp in loops:
        P = V[lp]
        c = P.mean(0)
        if c[2] > 1.3 and abs(c[0]) < 0.05:
            neck_loop = lp
            continue
        if c[2] < 1.16 and abs(c[0]) < 0.1:
            # --- hem: extend the torso down with a soft gathered rib hem
            ctr = np.array([0.0, -0.012])
            ang = np.arctan2(P[:, 1] - ctr[1], P[:, 0] - ctr[0])
            rad = np.linalg.norm(P[:, :2] - ctr, axis=1)
            zb = P[:, 2]
            prev = list(lp)
            rings = [(-0.02, 1.02), (-0.04, 1.03), (-0.06, 1.02), (-0.075, 0.99), (-0.085, 0.975)]
            for dz, sc in rings:
                cur = []
                for i in range(len(lp)):
                    r = rad[i] * sc
                    cur.append(add_vert(np.array([ctr[0] + r * math.cos(ang[i]), ctr[1] + r * math.sin(ang[i]), 1.098 + dz]), 0))
                for i in range(len(lp)):
                    j = (i + 1) % len(lp)
                    newF.append([prev[i], prev[j], cur[j], cur[i]])
                prev = cur
            continue
        # --- sleeve: planarise the cut, extend to the wrist, gather into a closed paw
        sd = "L" if c[0] > 0 else "R"
        sg = 1 if sd == "L" else -1
        t_cut, E2, a, armv, tv = cutinfo[sd]
        _, b1, b2 = frame_for(a)
        tl = (P - E2) @ a
        P = P - (tl - tl.max())[:, None] * a[None, :] * 0.0
        tplane = tl.max()
        P = P + (tplane - tl)[:, None] * a[None, :]
        V[lp] = P
        c0 = P.mean(0)
        rel = P - c0
        ang = np.arctan2(rel @ b2, rel @ b1)
        rad = np.linalg.norm(rel - (rel @ a)[:, None] * a, axis=1)
        rmean = rad.mean()
        W2 = np.array(T["_W2." + sd])
        s_w = max((W2 - c0) @ a, 0.03)
        prev = list(lp)
        # hand frame: knuckles lie along the phone's long axis, palm faces the phone edge
        _, ux, ul, _v = phone_frame()
        ux = np.array(ux) - a * (np.array(ux) @ a)
        ux /= np.linalg.norm(ux)
        ulp = np.cross(a, ux)
        prof = []
        for k in range(1, 7):                                # forearm: slightly baggy knit
            u = k / 6.0
            prof.append((s_w * u, 1.0 + 0.16 * math.sin(math.pi * u * 0.85) - 0.10 * u ** 3, 0.0))
        cuff = 0.068
        for j in range(1, 6):                                # gathered rib cuff over the palm
            u = j / 5.0
            prof.append((s_w + cuff * u, 0.0, u))
        prev = list(lp)
        r_cuff_w, r_cuff_t = 0.044, 0.0205
        for (ss, sc, cu) in prof:
            cur = []
            cc = c0 + a * ss
            for i in range(len(lp)):
                dirr = b1 * math.cos(ang[i]) + b2 * math.sin(ang[i])
                if cu == 0.0:
                    r = (rad[i] * (1 - ss / s_w * 0.5) + rmean * 0.5 * ss / s_w) * sc
                    pnt = cc + dirr * r
                else:
                    # blend from the sleeve section to an elliptical cuff opening
                    e = (dirr @ ux) * r_cuff_t * ux + (dirr @ ulp) * r_cuff_w * ulp
                    rw = rad[i] * (1.0 + 0.06 - 0.10)
                    base = dirr * rw
                    w = smoothstep(0.0, 0.7, cu)
                    pnt = cc + base * (1 - w) + (e * (1.0 + 0.04 * math.sin(math.pi * cu))) * w
                cur.append(add_vert(pnt, sg))
            for i in range(len(lp)):
                j = (i + 1) % len(lp)
                newF.append([prev[i], prev[j], cur[j], cur[i]])
            prev = cur
        # rolled lip: turn the cuff edge inward so the opening has visible thickness
        cc = c0 + a * (s_w + cuff - 0.006)
        cur = []
        for i in range(len(lp)):
            dirr = b1 * math.cos(ang[i]) + b2 * math.sin(ang[i])
            e = (dirr @ ux) * (r_cuff_t - 0.005) * ux + (dirr @ ulp) * (r_cuff_w - 0.005) * ulp
            cur.append(add_vert(cc + e, sg))
        for i in range(len(lp)):
            j = (i + 1) % len(lp)
            newF.append([prev[i], prev[j], cur[j], cur[i]])
    V = np.vstack([V, np.array(extra_V)])
    region = np.concatenate([region, np.array(extra_region)])
    F = F + newF
    # --- consistent winding, then one Catmull-Clark level so folds have resolution
    tmp = bpy.data.meshes.new("tmp_sw")
    tmp.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
    set_attr_float(tmp, "region", region)
    bm = bmesh.new()
    bm.from_mesh(tmp)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(tmp)
    bm.free()
    tob = link(bpy.data.objects.new("tmp_sw", tmp), C["cloth"])
    apply_subdiv(tob, 1)
    sm = tob.data
    V = get_co(sm)
    region = attr_float(sm, "region")
    region = np.where(region > 0.5, 1.0, np.where(region < -0.5, -1.0, 0.0))
    F = [list(p.vertices) for p in sm.polygons]
    bpy.data.objects.remove(tob)
    bpy.data.meshes.remove(sm)
    Fq = np.array([f for f in F if len(f) == 4])
    Ft = np.array([f for f in F if len(f) == 3]) if any(len(f) == 3 for f in F) else np.zeros((0, 3), int)
    loops = boundary_loops(F)
    neck_loop = min(loops, key=lambda lp: np.linalg.norm(V[lp].mean(0) - np.array([0.0, -0.02, 1.38])))
    E = mesh_adjacency(len(V), F)
    pin = np.zeros(len(V), bool)
    for lp in loops:
        pin[lp] = True
    torso = region == 0
    Nn = vnormals_fast(V, Fq, Ft)
    cen = np.array([0, -0.01, 1.25])
    if np.mean(((V[torso] - cen) * Nn[torso]).sum(1)) < 0:
        Fq = Fq[:, ::-1]
        Ft = Ft[:, ::-1]
        F = [list(reversed(f)) for f in F]
    log("sweater subdivided verts", len(V))
    # (a) record the body envelope, then smooth away anatomy (nipples, ribs, muscles)
    ctr = np.array([0.0, -0.012])
    th = np.arctan2(V[:, 1] - ctr[1], V[:, 0] - ctr[0])
    rr0 = np.linalg.norm(V[:, :2] - ctr, axis=1)
    nb_t, nb_z = 90, 140
    ti = np.clip(((th + math.pi) / (2 * math.pi) * nb_t).astype(int), 0, nb_t - 1)
    zi = np.clip(((V[:, 2] - 0.98) / 0.42 * nb_z).astype(int), 0, nb_z - 1)
    grid = np.zeros((nb_t, nb_z))
    np.maximum.at(grid, (ti[torso], zi[torso]), rr0[torso])
    msk = (grid > 0).astype(float)

    def blur(g, st, sz):
        kt = np.exp(-0.5 * (np.arange(-3 * st, 3 * st + 1) / st) ** 2)
        kz = np.exp(-0.5 * (np.arange(-3 * sz, 3 * sz + 1) / sz) ** 2)
        kt, kz = kt / kt.sum(), kz / kz.sum()
        out = np.zeros_like(g)
        for i, w in enumerate(kt):
            out += w * np.roll(g, i - 3 * st, 0)
        g2 = np.zeros_like(out)
        for i, w in enumerate(kz):
            g2 += w * np.roll(out, i - 3 * sz, 1)
        return g2

    env = blur(grid * msk, 5, 15) / np.maximum(blur(msk, 5, 15), 1e-6)
    dz = 0.42 / nb_z
    zc = 0.98 + (np.arange(nb_z) + 0.5) * dz
    for k in range(nb_z - 2, -1, -1):           # hang from the bust
        if zc[k] < 1.30:
            env[:, k] = np.maximum(env[:, k], env[:, k + 1] - 0.16 * dz)
    env = blur(env, 1, 5)
    wts = np.where(torso, 1.0, 0.7)
    V = laplacian(V, E, pin, iters=25, lam=0.6, mu=-0.0, wts=wts)
    V = laplacian(V, E, pin, iters=10, wts=wts)
    th = np.arctan2(V[:, 1] - ctr[1], V[:, 0] - ctr[0])
    rr = np.linalg.norm(V[:, :2] - ctr, axis=1)
    ti = np.clip(((th + math.pi) / (2 * math.pi) * nb_t).astype(int), 0, nb_t - 1)
    zi = np.clip(((V[:, 2] - 0.98) / 0.42 * nb_z).astype(int), 0, nb_z - 1)
    tgt = env[ti, zi]
    frontness = 0.35 + 0.65 * smoothstep(-0.3, 0.5, -np.sin(th))
    base_r = np.maximum(rr, rr0 - 0.004)
    want = np.where(torso, base_r + (tgt - base_r) * frontness, rr)
    # anatomy may only push the knit out where it is broad (shoulder blades/back), not lumps
    floor_ok = smoothstep(0.35, 0.1, -np.sin(th))               # sides/back only
    want = np.where(torso, np.maximum(want, (rr0 - 0.012) * floor_ok), rr)
    neckblend = smoothstep(1.37, 1.33, V[:, 2])
    want = rr + (want - rr) * neckblend
    V[:, 0] = ctr[0] + np.cos(th) * want
    V[:, 1] = ctr[1] + np.sin(th) * want
    # (b) ease of a relaxed sweater
    Nn = vnormals_fast(V, Fq, Ft)
    zz = V[:, 2]
    off = np.full(len(V), 0.008)
    off[torso] += 0.005 * smoothstep(1.30, 1.15, zz[torso])
    off[torso] -= 0.003 * smoothstep(1.33, 1.40, zz[torso])
    off[~torso] = 0.010
    V = V + Nn * off[:, None]
    V = laplacian(V, E, pin, iters=8)
    # (c) gather the hem rib slightly
    hem = torso & (V[:, 2] < 1.06)
    g = smoothstep(1.06, 1.035, V[hem, 2])
    V[hem, 0] = ctr[0] + (V[hem, 0] - ctr[0]) * (1 - 0.05 * g)
    V[hem, 1] = ctr[1] + (V[hem, 1] - ctr[1]) * (1 - 0.05 * g)
    # (d) gravity sag of the knit under the near-horizontal arms
    for sd, sg in (("L", 1), ("R", -1)):
        m = np.nonzero(region == sg)[0]
        sa, ph, dd = arm_coords(V[m], T, sd)
        Lu = (T["_E2." + sd] - T["_S2." + sd]).length
        Lw = Lu + (T["_W2." + sd] - T["_E2." + sd]).length
        axis_z = np.interp(sa, [0, Lu, Lw], [T["_S2." + sd].z, T["_E2." + sd].z, T["_W2." + sd].z])
        rel_z = (V[m, 2] - axis_z) / np.maximum(dd, 1e-3)
        lowside = np.clip(-rel_z, 0, 1) ** 1.3
        mid = smoothstep(0.05, 0.15, sa) * smoothstep(Lw + 0.01, Lw - 0.07, sa)
        V[m, 2] -= 0.009 * lowside * mid
    V = laplacian(V, E, pin, iters=3)
    # ---------------- folds ----------------
    Nn = vnormals_fast(V, Fq, Ft)
    rng = np.random.default_rng(11)

    def pnoise(a, b, n=6, fa=(8, 40), mb=(0, 3)):
        out = np.zeros_like(a)
        for _ in range(n):
            w = rng.uniform(*fa)
            m_ = rng.integers(mb[0], mb[1] + 1)
            out += np.sin(a * w + b * m_ + rng.uniform(0, 6.283)) / n ** 0.5
        return out

    disp = np.zeros(len(V))
    for sd, sg in (("L", 1), ("R", -1)):
        m = region == sg
        sa, ph, _ = arm_coords(V[m], T, sd)
        Lup = (T["_E2." + sd] - T["_S2." + sd]).length
        warp = 1.8 * pnoise(sa, ph, 5, (6, 25), (0, 2))
        lam = 0.040
        f = np.sin(2 * np.pi * sa / lam + warp + 0.9 * np.sin(ph + rng.uniform(0, 6)))
        f = np.sign(f) * np.abs(f) ** 0.8
        mask = np.clip(0.45 + 0.7 * pnoise(sa, ph, 5, (10, 30), (1, 3)), 0, 1)
        inner = 0.5 + 0.5 * np.cos(ph - math.pi)
        amp = 0.0016 * mask * (0.4 + 0.6 * inner)
        amp *= 1 + 0.7 * smoothstep(Lup - 0.08, Lup, sa) * smoothstep(Lup + 0.18, Lup + 0.04, sa)
        amp *= smoothstep(0.05, 0.11, sa)
        disp[m] = f * amp
    tm = region == 0
    th = np.arctan2(V[tm, 1] + 0.012, V[tm, 0])
    zt = V[tm, 2]
    gq = 0.0014 * np.sin(th * 7.0 + 1.8 * pnoise(zt, th, 4, (10, 30), (0, 2))) * smoothstep(1.24, 1.08, zt)
    gq += 0.0008 * np.sin(zt * 150.0 + th * 2.0 + pnoise(th, zt, 3, (2, 6), (0, 0))) * smoothstep(1.10, 1.05, zt)
    for sx in (1, -1):
        cpt = np.array([sx * 0.14, -0.07, 1.27])
        rel = V[tm] - cpt
        dist = np.linalg.norm(rel, axis=1)
        ang = np.arctan2(rel[:, 2], rel[:, 0] * sx)
        gq += 0.0010 * np.sin(ang * 5.0 + 2.0) * smoothstep(0.12, 0.05, dist) * smoothstep(0.0, 0.03, dist) * (rel[:, 1] < 0.03)
    disp[tm] = gq
    V = V + Nn * disp[:, None]
    # ---------------- mesh ----------------
    sm = bpy.data.meshes.new("Sweater_mesh")
    sm.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
    sm.validate()
    ob = link(bpy.data.objects.new("Sweater", sm), C["cloth"])
    for p in sm.polygons:
        p.use_smooth = True
    sm.update()
    # per-corner knit UVs: torso = around the body, sleeves = around/along the arm
    TILE = 8
    uvl = sm.uv_layers.new(name="KnitUV")
    loop_v = np.empty(len(sm.loops), int)
    sm.loops.foreach_get("vertex_index", loop_v)
    Vx = get_co(sm)
    uv = np.zeros((len(Vx), 2))
    rib = np.zeros(len(Vx))
    m = region == 0
    th = np.arctan2(-(Vx[m, 1] + 0.012), Vx[m, 0])     # 0 at her left side, seam at the back centre
    uv[m, 0] = th * 0.155 / (STW * TILE)
    uv[m, 1] = Vx[m, 2] / (STH * TILE)
    rib[m] = smoothstep(1.045, 1.035, Vx[m, 2])
    for sd, sg in (("L", 1), ("R", -1)):
        mm = region == sg
        sa, ph, _ = arm_coords(Vx[mm], T, sd)
        uv[mm, 0] = ph * 0.05 / (STW * TILE)
        uv[mm, 1] = -sa / (STH * TILE)
        W2 = np.array(T["_W2." + sd])
        Lw = (T["_E2." + sd] - T["_S2." + sd]).length + (T["_W2." + sd] - T["_E2." + sd]).length
        rib[mm] = smoothstep(Lw - 0.004, Lw + 0.006, sa)
    luv = uv[loop_v].copy()
    # fix wrap-around seams per face
    for p in sm.polygons:
        idx = list(range(p.loop_start, p.loop_start + p.loop_total))
        u = luv[idx, 0]
        per = (2 * math.pi * (0.155 if region[loop_v[idx[0]]] == 0 else 0.05)) / (STW * TILE)
        if u.max() - u.min() > per * 0.5:
            luv[idx, 0] = np.where(u < u.mean(), u + per, u)
    uvl.data.foreach_set("uv", luv.astype(np.float32).ravel())
    set_attr_float(sm, "rib", rib)
    set_attr_float(sm, "region", region)
    mod = ob.modifiers.new("Subdivision", 'SUBSURF')
    mod.levels, mod.render_levels = 1, 1
    return ob, V, neck_loop, region


def build_collar(C, sweater_V, neck_loop, body):
    """Folded 1x1 rib crew-neck band swept around the sweater neckline."""
    P = sweater_V[neck_loop]
    # resample the loop evenly
    P2 = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P2, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    M = 520
    ts = np.linspace(0, cum[-1], M, endpoint=False)
    Q = np.stack([np.interp(ts, cum, P2[:, k]) for k in range(3)], 1)
    Fq_ = np.fft.rfft(Q, axis=0)                # low-pass: keep the broad neckline shape only
    Fq_[7:] = 0
    Q = np.fft.irfft(Fq_, n=M, axis=0)
    cen = Q.mean(0)
    co_b, tris = evaluated_mesh_arrays(body)
    bvh = BVHTree.FromPolygons([tuple(v) for v in co_b], [tuple(t) for t in tris])
    up = np.array([0.0, 0.0, 1.0])
    nrib = 150
    K = 16
    H, TK = 0.018, 0.0068                       # rib band height / thickness
    prof = []
    for k in range(K):                          # rounded-rectangle (superellipse) section
        a_ = 2 * math.pi * k / K
        c_, s_ = math.cos(a_), math.sin(a_)
        prof.append((math.copysign(abs(c_) ** 0.45, c_), math.copysign(abs(s_) ** 0.45, s_)))
    frames = []
    for i in range(M):
        p = Q[i]
        tng = Q[(i + 1) % M] - Q[i - 1]
        tng /= np.linalg.norm(tng)
        out = p - cen
        out -= tng * (out @ tng)
        out -= up * (out @ up)
        out /= np.linalg.norm(out)
        upv = up - tng * (up @ tng) - out * 0.22   # band leans in to hug the neck
        upv /= np.linalg.norm(upv)
        frames.append((p, out, upv))
    # how far each ring must move outward so the band clears the skin (smoothed)
    need = np.zeros(M)
    for i, (p, out, upv) in enumerate(frames):
        for frac in (0.0, 0.5, 1.0):
            q = p + upv * (H * frac) + out * 0.0005
            loc, nrm, _, dist = bvh.find_nearest(Vector(q))
            if loc is None:
                continue
            gap = (Vector(q) - loc).dot(nrm)
            need[i] = max(need[i], 0.0022 - gap)
    for _ in range(40):
        need = np.maximum(need, 0.5 * need + 0.25 * (np.roll(need, 1) + np.roll(need, -1)))
    for _ in range(20):
        need = 0.5 * need + 0.25 * (np.roll(need, 1) + np.roll(need, -1))
    verts = []
    for i, (p, out, upv) in enumerate(frames):
        ribo = 0.00035 * math.cos(2 * math.pi * nrib * i / M)
        for (cx, cy) in prof:
            h = (cy * 0.5 + 0.5) * H - 0.004
            t = (cx * 0.5 + 0.5) * (TK + ribo * (cx > 0)) + need[i] + 0.0005
            verts.append(p + upv * h + out * t)
    verts = np.array(verts)
    faces = []
    for i in range(M):
        j = (i + 1) % M
        for k in range(K):
            l = (k + 1) % K
            faces.append((i * K + k, i * K + l, j * K + l, j * K + k))
    me = bpy.data.meshes.new("Collar_mesh")
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    uvl = me.uv_layers.new(name="KnitUV")
    loop_v = np.empty(len(me.loops), int)
    me.loops.foreach_get("vertex_index", loop_v)
    ring = loop_v // K
    kk = loop_v % K
    luv = np.stack([ring / M * (cum[-1] / (0.0052 * 8)), kk / K * 0.05 / (STH * 8)], 1)
    for p in me.polygons:
        idx = list(range(p.loop_start, p.loop_start + p.loop_total))
        u = luv[idx, 0]
        if u.max() - u.min() > 1:
            luv[idx, 0] = np.where(u < u.mean(), u + cum[-1] / (0.0052 * 8), u)
        v = luv[idx, 1]
        if v.max() - v.min() > 0.05 / (STH * 8) * 0.5:
            luv[idx, 1] = np.where(v < v.mean(), v + 0.05 / (STH * 8), v)
    uvl.data.foreach_set("uv", luv.astype(np.float32).ravel())
    set_attr_float(me, "rib", np.ones(len(verts)))
    ob = link(bpy.data.objects.new("Sweater_Collar", me), C["cloth"])
    mod = ob.modifiers.new("Subdivision", 'SUBSURF')
    mod.levels, mod.render_levels = 1, 1
    me.update()
    # normals outward check
    return ob


STW, STH = 0.0092, 0.0078             # chunky-knit stitch width/height (m)


def make_knit_images():
    """Tileable stockinette + rib height maps (8x8 stitches per tile), packed."""
    R = 1024
    y, x = np.mgrid[0:R, 0:R].astype(np.float64) / (R / 8.0)   # stitch units
    h = np.zeros((R, R))
    fx, fy = x % 1.0, y % 1.0
    for dy in (-1, 0, 1):
        for side in (-1, 1):
            cx = 0.5 + side * 0.215
            cy = 0.5 + dy
            px, py = fx - cx, fy - cy
            ang = math.radians(24) * side
            u = px * math.cos(ang) - py * math.sin(ang)
            v = px * math.sin(ang) + py * math.cos(ang)
            d = (u / 0.215) ** 2 + (v / 0.56) ** 2
            leg = np.sqrt(np.clip(1 - d, 0, 1))
            # ply twist: diagonal stripes along each leg
            ply = 0.82 + 0.18 * np.sin((v * 11.0 + u * 14.0 * side) * math.pi * 2)
            h = np.maximum(h, leg * ply)
    rng = np.random.default_rng(3)
    # fibre fuzz
    nz = rng.random((R // 4, R // 4))
    nz = np.kron(nz, np.ones((4, 4)))
    for _ in range(2):
        nz = 0.25 * (np.roll(nz, 1, 0) + np.roll(nz, -1, 0) + np.roll(nz, 1, 1) + np.roll(nz, -1, 1))
    h = np.clip(h * 0.9 + 0.1 * nz, 0, 1)
    # per-stitch irregularity
    jit = rng.random((8, 8)) * 0.12 + 0.88
    h *= np.kron(jit, np.ones((R // 8, R // 8)))
    # 1x1 rib: raised knit columns separated by deep purl valleys
    rib = 0.5 + 0.5 * np.cos(2 * math.pi * x)
    rib = rib ** 0.7 * (0.75 + 0.25 * h)
    imgs = {}
    os.makedirs(os.path.join(ROOT, "textures"), exist_ok=True)
    for name, arr in (("Knit_Stockinette_Height", h), ("Knit_Rib_Height", rib)):
        tmp = bpy.data.images.new(name + "_tmp", R, R, alpha=False, float_buffer=False)
        px = np.ones((R, R, 4), np.float32)
        px[..., 0] = px[..., 1] = px[..., 2] = arr
        tmp.pixels.foreach_set(px.ravel())
        path = os.path.join(ROOT, "textures", name + ".png")
        tmp.filepath_raw = path
        tmp.file_format = 'PNG'
        tmp.save()
        bpy.data.images.remove(tmp)
        img = bpy.data.images.load(path)
        img.name = name
        img.colorspace_settings.name = 'Non-Color'
        img.pack()
        imgs[name] = img
        REPORT["textures"].append({"name": name, "size": [R, R], "kind": "procedurally generated in numpy by build.py (packed)",
                                   "source": "procedural (no photo pixels)",
                                   "uv_map": "KnitUV on Sweater / Sweater_Collar (analytic around/along-body and arm UVs)",
                                   "description": "tileable %s height map, 8x8 stitches; drives the colour ramp (photo-sampled "
                                                  "dusty pink) and bump" % ("stockinette V-stitch" if "Stock" in name else "1x1 rib")})
    return imgs


def photo_knit_colour():
    """Median sRGB of the photographed sweater front (a plain knit area below the hair)."""
    ph = ptex.load_rgb(os.path.join(INPUT, "reference.png"))
    patch = ph[760:1000, 760:1000].reshape(-1, 3)
    lum = patch @ np.array([0.3, 0.59, 0.11])
    lit = patch[(lum > np.percentile(lum, 35)) & (lum < np.percentile(lum, 90))]
    med = np.median(lit, axis=0)
    REPORT["notes"].append("knit base colour sampled from photo region x760-1000,y760-1000: sRGB %s" %
                           np.round(med * 255).astype(int).tolist())
    return med


def sweater_material(imgs):
    m, nt = new_mat("Knit_DustyPink")
    kc = photo_knit_colour() * 255.0                       # sampled dusty pink (sRGB 0..255)
    # the photo is brightly exposed: the sampled (lit) colour is scaled down to an albedo
    k_lo, k_mid, k_hi, k_var = kc * 0.52, kc * 0.78, kc * 0.92, kc * 0.74
    bsdf = principled(nt, (600, 0))
    uv = N(nt, "ShaderNodeUVMap", (-1300, 0), uv_map="KnitUV")
    t1 = N(nt, "ShaderNodeTexImage", (-1050, 150), image=imgs["Knit_Stockinette_Height"], interpolation='Cubic')
    t2 = N(nt, "ShaderNodeTexImage", (-1050, -150), image=imgs["Knit_Rib_Height"], interpolation='Cubic')
    L(nt, uv.outputs[0], t1.inputs[0])
    L(nt, uv.outputs[0], t2.inputs[0])
    rib = N(nt, "ShaderNodeAttribute", (-1050, 400), attribute_name="rib", attribute_type='GEOMETRY')
    h = mix_rgb(nt, rib.outputs["Fac"], t1.outputs[0], t2.outputs[0], (-750, 0))
    hv = N(nt, "ShaderNodeRGBToBW", (-600, 0))
    L(nt, h, hv.inputs[0])
    tc = N(nt, "ShaderNodeTexCoord", (-1300, -500))
    fz = N(nt, "ShaderNodeTexNoise", (-1050, -500))
    fz.inputs["Scale"].default_value = 900.0
    fz.inputs["Detail"].default_value = 6.0
    L(nt, tc.outputs["Object"], fz.inputs["Vector"])
    lo = N(nt, "ShaderNodeTexNoise", (-1050, -750))
    lo.inputs["Scale"].default_value = 25.0
    L(nt, tc.outputs["Object"], lo.inputs["Vector"])
    ramp = N(nt, "ShaderNodeValToRGB", (-400, 250))
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.05, (*srgb(*k_lo), 1)
    cr.elements[1].position, cr.elements[1].color = 0.85, (*srgb(*k_hi), 1)
    e = cr.elements.new(0.45)
    e.color = (*srgb(*k_mid), 1)
    L(nt, hv.outputs[0], ramp.inputs[0])
    col = mix_rgb(nt, map_range(nt, lo.outputs["Fac"], 0.3, 0.7, 0.0, 0.12, (-400, -700), 'LINEAR'), ramp.outputs[0], (*srgb(*k_var), 1), (-100, 200))
    L(nt, col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Sheen Weight"].default_value = 0.85
    bsdf.inputs["Sheen Roughness"].default_value = 0.45
    bsdf.inputs["Sheen Tint"].default_value = (*srgb(255, 228, 228), 1)
    bsdf.inputs["Subsurface Weight"].default_value = 0.15
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.5, 0.5)
    bsdf.inputs["Subsurface Scale"].default_value = 0.002
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    hsum = math_node(nt, 'ADD', hv.outputs[0], math_node(nt, 'MULTIPLY', fz.outputs["Fac"], 0.25, (-800, -500)), (-500, -300))
    bmp = N(nt, "ShaderNodeBump", (300, -300))
    bmp.inputs["Strength"].default_value = 0.9
    bmp.inputs["Distance"].default_value = 0.0026
    L(nt, hsum, bmp.inputs["Height"])
    L(nt, bmp.outputs["Normal"], bsdf.inputs["Normal"])
    output(nt, bsdf.outputs[0], (900, 0))
    return m


def trim_skin_under_sweater(body):
    me = body.data
    rest = attr_vec(me, "rest_co")
    # keep a generous band of skin under the neckline: from the high selfie camera one looks
    # down into the collar, and a cut edge there showed as a dark line
    hidden = neck_cover(rest, margin=0.045)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    kill = [f for f in bm.faces if all(hidden[v.index] for v in f.verts)]
    bmesh.ops.delete(bm, geom=kill, context='FACES')
    bm.to_mesh(me)
    bm.free()
    me.update()


# ----------------------------------------------------------------------------
# 6. hair: guide strands grown over the posed head/body with collision,
#    then interpolated/clumped child strands (Blender hair Curves)
# ----------------------------------------------------------------------------
PART_X = 0.004
CROWN = np.array([0.0, 0.032, 1.655])


def scalp_density(q):
    """0..1 hair density on the scalp in rest/head-local coordinates."""
    X, Y, Z = q[:, 0], q[:, 1], q[:, 2]
    az = np.degrees(np.arctan2(np.abs(X), -(Y + 0.02)))
    zh = np.interp(az, [0, 20, 38, 50, 56, 62, 90, 130, 180],
                   [1.629, 1.626, 1.613, 1.588, 1.548, 1.540, 1.505, 1.475, 1.463])
    ear = (((Y + 0.047) / 0.027) ** 2 + ((Z - 1.530) / 0.041) ** 2 < 1) & (np.abs(X) > 0.05)
    return smoothstep(zh - 0.001, zh + 0.012, Z) * (~ear)


def collider_bvh(objs):
    allv, allt, off = [], [], 0
    for ob in objs:
        co, tris = evaluated_mesh_arrays(ob)
        allv.append(co)
        allt.append(tris + off)
        off += len(co)
    V = np.vstack(allv)
    Tt = np.vstack(allt)
    return BVHTree.FromPolygons(V.tolist(), Tt.tolist()), V, Tt


def push_out(bvh, p, off):
    loc, nrm, idx, dist = bvh.find_nearest(Vector(p), 0.05)
    if loc is None:
        return p, False
    d = Vector(p) - loc
    if d.dot(nrm) < 0 or dist < off:
        return np.array(loc + nrm * off), True
    return p, False


def sample_surface(co, tris, weight_fn, n, rng):
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    w = area * weight_fn((a + b + c) / 3)
    w = w / w.sum()
    ti = rng.choice(len(tris), n, p=w)
    r1, r2 = rng.random(n), rng.random(n)
    sq = np.sqrt(r1)
    u, v = 1 - sq, sq * (1 - r2)
    pts = a[ti] * u[:, None] + b[ti] * v[:, None] + c[ti] * (1 - u - v)[:, None]
    nrm = np.cross(b[ti] - a[ti], c[ti] - a[ti])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    return pts, nrm


def poisson_subset(pts, rmin, rng):
    """Greedy dart-throwing on a voxel hash: well-spaced guide roots."""
    cell = rmin
    grid = {}
    keep = []
    for i in rng.permutation(len(pts)):
        p = pts[i]
        k = tuple((p // cell).astype(int))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if np.sum((pts[j] - p) ** 2) < rmin * rmin:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            keep.append(i)
            grid.setdefault(k, []).append(i)
    return np.array(keep)


def resample_polyline(P, n):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, P[:, k]) for k in range(3)], 1)


def strand_kind(q):
    """classify roots (rest/head-local coords)."""
    X, Y, Z = q[:, 0], q[:, 1], q[:, 2]
    bang = (Y < -0.084) & (np.abs(X - PART_X) < 0.042 + 0.3 * (-0.084 - Y))
    frame = (Y < -0.045) & ~bang & (np.abs(X) > 0.035)
    back = (Y > CROWN[1] - 0.012) & ~bang & ~frame
    kind = np.where(bang, 1, np.where(frame, 2, np.where(back, 3, 0)))
    return kind


def grow_guides(bvh, roots_w, nrm_w, q, Rh, rng, Ah=None):
    """Integrate guide strands under gravity with scalp/body collision."""
    kind = strand_kind(q)
    n = len(roots_w)
    Rm = np.array(Rh)
    Ahn = np.array(Ah) if Ah is not None else np.eye(4)
    to_local = lambda p_: Rm.T @ (p_ - Ahn[:3, 3])
    down = np.array([0, 0, -1.0])
    guides = []
    meta = []
    for i in range(n):
        p0, nr, ql, kd = roots_w[i], nrm_w[i], q[i], kind[i]
        side = 1.0 if ql[0] >= PART_X else -1.0
        # initial comb direction in head-local space
        if kd == 1:          # bangs: styled down over the forehead (head-local), little gravity
            a = np.array([0.50 * (ql[0] - PART_X) / 0.04, -0.55, -1.0])
            L = 0.100 + 0.8 * (ql[1] + 0.115) + rng.uniform(-0.010, 0.010)
            g = 0.07
            layer = 0.5 + 0.5 * rng.random()
        elif kd == 2:        # face-framing strands
            a = np.array([0.35 * side, -0.55, -1.0])
            L = rng.choice([rng.uniform(0.15, 0.22), rng.uniform(0.32, 0.46)], p=[0.35, 0.65] if side < 0 else [0.7, 0.3])
            g = 0.12
            layer = 0.3 + 0.4 * rng.random()
        else:
            if kd == 3:
                a = ql - CROWN
                a[2] = min(a[2], -0.02)
                a = a / np.linalg.norm(a) + np.array([0.0, 0.0, -1.1])   # down the back of the head
            else:
                a = np.array([side * 0.45, 0.05, -1.0])
            L = (ql[2] - 1.08) + 0.08 + rng.uniform(-0.04, 0.04)
            g = 0.16
            layer = np.clip((ql[2] - 1.48) / 0.16, 0, 1) * 0.95 + 0.25 * rng.random()
        # her left side (image right in the photo): hair from the temple is tucked behind the ear
        tuck = (side > 0 and kd in (0, 2) and ql[1] < 0.012 and ql[2] > 1.52 and rng.random() > 0.06)
        if tuck:
            a = np.array([0.35, 0.30, -1.0])                # down over the temple first
            L = max(L, rng.uniform(0.30, 0.42))
        if kd in (0, 3) and ql[2] > 1.60:
            layer = min(1.0, layer + 0.25)                 # a little more lift at the crown
        a = Rm @ a
        a = a - nr * (a @ nr)
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(nr, [1, 0, 0])
        d = a / np.linalg.norm(a)
        if kd == 1:
            d = d + nr * 0.30
            d /= np.linalg.norm(d)
        off_t = 0.0015 + 0.0115 * layer if kd != 1 else 0.003 + 0.006 * layer
        if side > 0:      # her left: arm is raised in front of the shoulder -> hair goes behind
            front = ((kd == 1) or (kd == 2 and rng.random() < 0.5) or (ql[1] < -0.06 and rng.random() < 0.25)) and not tuck
        else:
            front = (ql[1] < -0.035) or (kd in (1, 2)) or (ql[1] < -0.01 and rng.random() < 0.6)
        ds = 0.006
        P = [p0.copy()]
        p = p0 + nr * 0.0004
        s = 0.0
        while s < L:
            g_now = g * (1.0 if s > 0.02 else 0.4)
            bias = np.zeros(3)
            if 1.36 < p[2] < 1.50 and abs(p[0]) > 0.045:
                if front:
                    bias[1] = -0.05 if p[1] > -0.09 else 0.0
                elif p[1] < 0.03:
                    bias[1] = 0.05
            if not front and 1.40 < p[2] < 1.58 and side > 0 and p[1] < 0.0:
                bias[1] += 0.05
                bias[0] -= 0.02
            if tuck:
                pl = to_local(p)
                if 1.46 < pl[2] < 1.585 and pl[1] < 0.03 and pl[0] > 0.035:
                    bias += Rm @ np.array([0.0, 0.17, 0.03])     # then over and behind the ear
            d = d + down * g_now + bias
            d /= np.linalg.norm(d)
            pn = p + d * ds
            off = off_t * min(1.0, 0.25 + s / 0.03)
            pn, hit = push_out(bvh, pn, off)
            hug = kd == 1 and s < 0.85 * L
            if hug:
                # styled bangs hug the forehead at their layer offset instead of sailing off it
                loc_, nrm_, _, dist_ = bvh.find_nearest(Vector(pn), 0.05)
                if loc_ is not None and dist_ > off:
                    pn = np.array(loc_ + nrm_ * off)
            st = pn - p
            sl = np.linalg.norm(st)
            if sl < 0.25 * ds:
                # wedged (e.g. between the tilted-back head and the collar): slide out and down
                # instead of ending the strand early, which left a hard 'shelf' in the back view
                loc_, nrm_, _, _ = bvh.find_nearest(Vector(p), 0.05)
                esc = down + (np.array(nrm_) * 0.8 if loc_ is not None else 0.0)
                d = esc / np.linalg.norm(esc)
                pn = p + d * ds
                st = pn - p
                sl = np.linalg.norm(st)
            d_new = st / sl
            d = 0.6 * d + 0.4 * d_new
            d /= np.linalg.norm(d)
            p = p + st * (ds / sl) if sl > ds else pn
            P.append(p.copy())
            s += ds
        guides.append(np.array(P))
        meta.append((kd, side, front, layer, tuck))
    return guides, meta


def wave_guides(guides, meta, q, rng):
    out = []
    for i, (P, (kd, side, front, layer, tuck)) in enumerate(zip(guides, meta)):
        if len(P) < 4:
            out.append(P)
            continue
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        tang = np.gradient(P, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-9)[:, None]
        up = np.array([0, 0, 1.0])
        b1 = np.cross(tang, up)
        b1 /= np.maximum(np.linalg.norm(b1, axis=1), 1e-9)[:, None]
        b2 = np.cross(tang, b1)
        ql = q[i]
        ph = 6.0 * ql[0] / 0.05 + 4.0 * ql[2] / 0.05 + rng.uniform(-0.7, 0.7)
        lam = 0.135 + rng.uniform(-0.015, 0.02)             # loose waves, as in the photo
        if kd == 1:
            A = 0.0016 * smoothstep(0.02, 0.07, s)
            lam = 0.06
        else:
            A = (0.0100 + 0.005 * rng.random()) * smoothstep(0.10, 0.26, s)
        # slowly drifting wavelength so the waves are not a regular crimp
        phase = 2 * np.pi * s / lam * (1 + 0.18 * np.sin(s / 0.21 + ph)) + ph
        w1 = np.sin(phase)
        w2 = np.cos(phase)
        P2 = P + (A * w1)[:, None] * b1 + (0.55 * A * w2)[:, None] * b2
        out.append(P2)
    return out


def hair_material():
    m, nt = new_mat("Hair_Chestnut")
    hb = N(nt, "ShaderNodeBsdfHairPrincipled", (300, 0))
    try:
        hb.model = 'CHIANG'
    except Exception:
        pass
    hb.parametrization = 'MELANIN'
    info = N(nt, "ShaderNodeHairInfo", (-400, 0))
    mel = map_range(nt, info.outputs["Intercept"], 0.0, 1.0, 0.93, 0.87, (-150, 150), 'LINEAR')
    L(nt, mel, hb.inputs["Melanin"])
    hb.inputs["Melanin Redness"].default_value = 0.66
    hb.inputs["Tint"].default_value = (1.0, 0.86, 0.80, 1.0)
    hb.inputs["Roughness"].default_value = 0.33
    hb.inputs["Radial Roughness"].default_value = 0.55
    hb.inputs["Coat"].default_value = 0.0
    hb.inputs["Random Color"].default_value = 0.12
    hb.inputs["Random Roughness"].default_value = 0.25
    L(nt, info.outputs["Random"], hb.inputs["Random"])
    output(nt, hb.outputs[0], (600, 0))
    return m


def make_curves_object(name, strands, radius_root, radius_tip, coll, mat):
    cv = bpy.data.hair_curves.new(name)
    sizes = [len(sp) for sp in strands]
    cv.add_curves(sizes)
    pos = np.vstack(strands).astype(np.float32)
    cv.position_data.foreach_set("vector", pos.ravel())
    rad = []
    for sp in strands:
        t = np.linspace(0, 1, len(sp))
        rad.append(radius_root + (radius_tip - radius_root) * t ** 1.5)
    rad = np.concatenate(rad).astype(np.float32)
    if "radius" in cv.attributes:
        ra = cv.attributes["radius"]
    else:
        ra = cv.attributes.new("radius", 'FLOAT', 'POINT')
    ra.data.foreach_set("value", rad)
    cv.materials.append(mat)
    ob = link(bpy.data.objects.new(name, cv), coll)
    return ob


class CollarClearance:
    """Polar map of the collar band: keeps hair outside the knit collar instead of
    letting it slip between collar and neck (nearest-surface pushes can pick the neck)."""

    def __init__(self, ob, nb=180):
        co, _ = evaluated_mesh_arrays(ob)
        self.c = co[:, :2].mean(0)
        rel = co[:, :2] - self.c
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        rad = np.linalg.norm(rel, axis=1)
        self.nb = nb
        b = ((ang + math.pi) / (2 * math.pi) * nb).astype(int) % nb
        self.rmax = np.zeros(nb)
        self.zlo = np.full(nb, 9.0)
        self.zhi = np.full(nb, -9.0)
        np.maximum.at(self.rmax, b, rad)
        np.minimum.at(self.zlo, b, co[:, 2])
        np.maximum.at(self.zhi, b, co[:, 2])

    def _push(self, P, off):
        rel = P[:, :2] - self.c
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        rad = np.linalg.norm(rel, axis=1)
        b = ((ang + math.pi) / (2 * math.pi) * self.nb).astype(int) % self.nb
        # smooth vertical falloff so strands bend gradually over the collar (no "hair tie")
        zc = 0.5 * (self.zlo[b] + self.zhi[b])
        hz = 0.5 * (self.zhi[b] - self.zlo[b])
        fall = smoothstep(hz + 0.030, hz + 0.002, np.abs(P[:, 2] - zc))
        back = rel[:, 1] > -0.03
        want = self.rmax[b] + off * fall
        need = back & (fall > 0) & (rad < want)
        if need.any():
            sc_ = want[need] / np.maximum(rad[need], 1e-6)
            P[need, 0] = self.c[0] + rel[need, 0] * sc_
            P[need, 1] = self.c[1] + rel[need, 1] * sc_
        return P

    def apply(self, P, off=0.0025):
        P0 = P.copy()
        D = self._push(P.copy(), off) - P0
        if not D.any():
            return P
        for _ in range(6):                       # spread the push along the strand (keeps waves)
            Dn = D.copy()
            Dn[1:-1] = 0.5 * D[1:-1] + 0.25 * (D[:-2] + D[2:])
            Dn[-1] = 0.5 * (D[-1] + D[-2])
            D = np.where(np.linalg.norm(Dn, axis=1)[:, None] > np.linalg.norm(D, axis=1)[:, None], Dn, D)
        D[0] = 0
        return self._push(P0 + D, off)


def build_hair(C, body, colliders, T, rng):
    Ah = T["head"]
    Rh = Ah.to_3x3()
    Ahi = np.array(Ah.inverted())
    bvh, _, _ = collider_bvh(colliders)
    co, tris = evaluated_mesh_arrays(body)
    loc = co @ Ahi[:3, :3].T + Ahi[:3, 3]
    dens = lambda P: scalp_density(P)
    # map sample positions to head-local for the density test
    def wfn(P):
        Ql = P @ Ahi[:3, :3].T + Ahi[:3, 3]
        return scalp_density(Ql)
    cand, cn = sample_surface(co, tris, wfn, 26000, rng)
    gi = poisson_subset(cand, 0.0042, rng)
    groots, gnrm = cand[gi], cn[gi]
    gq = groots @ Ahi[:3, :3].T + Ahi[:3, 3]
    log("hair guides", len(groots))
    guides, meta = grow_guides(bvh, groots, gnrm, gq, Rh, rng, Ah)
    guides = wave_guides(guides, meta, gq, rng)
    # re-collide waved guides
    clr = CollarClearance(bpy.data.objects["Sweater_Collar"])
    for gi_, g in enumerate(guides):
        for k in range(2, len(g)):
            g[k], _ = push_out(bvh, g[k], 0.0018)
        guides[gi_] = clr.apply(g, 0.003 + 0.004 * meta[gi_][3])
    log("guides grown")
    # children
    NCH = int(os.environ.get("HAIR_CHILDREN", "42000"))
    croots, cnrm = sample_surface(co, tris, wfn, NCH, rng)
    cq = croots @ Ahi[:3, :3].T + Ahi[:3, 3]
    ckind = strand_kind(cq)
    # wispy, see-through bangs: thin out children in the bang zone
    keep = (ckind != 1) | (rng.random(len(cq)) < 0.7)
    croots, cnrm, cq, ckind = croots[keep], cnrm[keep], cq[keep], ckind[keep]
    kd = KDTree(len(groots))
    for i, p in enumerate(groots):
        kd.insert(Vector(p), i)
    kd.balance()
    gkind = np.array([m_[0] for m_ in meta])
    gtuck = np.array([m_[4] for m_ in meta])
    glen = [np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(g, axis=0), axis=1))]) for g in guides]
    strands, sk = [], []
    for i in range(len(croots)):
        if ckind[i] == 1:
            near = kd.find_n(Vector(croots[i]), 12)
            near = [nn for nn in near if gkind[nn[1]] == 1] or near
        else:
            near = kd.find_n(Vector(croots[i]), 4)
            near = [nn for nn in near if gkind[nn[1]] == ckind[i]] or near
            near = [nn for nn in near if gtuck[nn[1]] == gtuck[near[0][1]]] or near
        g0 = near[0][1]
        g1 = near[1][1] if len(near) > 1 else g0
        beta = rng.uniform(0.0, 0.45) if ckind[i] != 1 else 0.0
        G0, G1 = guides[g0], guides[g1]
        L0, L1 = glen[g0][-1], glen[g1][-1]
        Lc = ((1 - beta) * L0 + beta * L1) * rng.uniform(0.86, 1.04)
        npts = int(np.clip(Lc / 0.016 + 4, 6, 36))
        s = np.linspace(0, Lc, npts)

        def at(G, gl, ss):
            ss = np.minimum(ss, gl[-1])
            return np.stack([np.interp(ss, gl, G[:, k]) for k in range(3)], 1)
        base = (1 - beta) * at(G0, glen[g0], s) + beta * at(G1, glen[g1], s)
        o = croots[i] - base[0]
        t = s / max(Lc, 1e-6)
        clump = rng.uniform(0.45, 0.85) if ckind[i] != 1 else rng.uniform(0.40, 0.80)
        P = base + o[None, :] * (1 - clump * smoothstep(0.08, 0.9, t))[:, None]
        # frizz / flyaways
        fz = 0.0007 if rng.random() > 0.012 else 0.0028
        P += rng.normal(0, fz, P.shape) * (t ** 1.2)[:, None]
        P[0] = croots[i]
        for k in range(1, len(P)):
            P[k], _ = push_out(bvh, P[k], 0.0012)
        P = clr.apply(P, 0.0022)
        strands.append(P)
        sk.append(ckind[i])
    log("children", len(strands))
    hm = hair_material()
    main_idx = [i for i, k in enumerate(sk) if k != 1]
    bang_idx = [i for i, k in enumerate(sk) if k == 1]
    h1 = make_curves_object("Hair_Strands", [strands[i] for i in main_idx], 0.000085, 0.000030, C["hair"], hm)
    h2 = make_curves_object("Hair_Bangs", [strands[i] for i in bang_idx], 0.000045, 0.000015, C["hair"], hm)
    REPORT["hair"] = {"guides": len(guides), "strands_main": len(main_idx), "strands_bangs": len(bang_idx),
                      "points": int(sum(len(sp) for sp in strands))}
    return h1, h2, bvh


# ----------------------------------------------------------------------------
# 7. hands (fingers emerging from the sweater cuffs) and the phone they hold
# ----------------------------------------------------------------------------
def tube_mesh(name, path, radii, seg=14):
    """Smooth tube along a polyline with per-point radius; ends closed by tiny radii."""
    path = np.asarray(path, float)
    n = len(path)
    tang = np.gradient(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    ref = np.array([0, 0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0, 0])
    nrm = np.cross(tang[0], ref)
    nrm /= np.linalg.norm(nrm)
    verts, faces = [], []
    for i in range(n):
        t = tang[i]
        nrm = nrm - t * (nrm @ t)
        nrm /= np.linalg.norm(nrm)
        bin_ = np.cross(t, nrm)
        for k in range(seg):
            a = 2 * math.pi * k / seg
            verts.append(path[i] + radii[i] * (math.cos(a) * nrm + math.sin(a) * bin_))
    for i in range(n - 1):
        for k in range(seg):
            k2 = (k + 1) % seg
            faces.append((i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k))
    c0 = len(verts)
    verts.append(path[0] - tang[0] * radii[0] * 0.3)
    c1 = len(verts)
    verts.append(path[-1] + tang[-1] * radii[-1] * 0.3)
    for k in range(seg):
        k2 = (k + 1) % seg
        faces.append((k2, k, c0))
        faces.append(((n - 1) * seg + k, (n - 1) * seg + k2, c1))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    for p in me.polygons:
        p.use_smooth = True
    return me


def finger_path(pts, r, n=26):
    """Catmull-Rom through control points; rounded fingertip radius profile."""
    P = np.asarray(pts, float)
    out = []
    for i in range(len(P) - 1):
        p0 = P[max(i - 1, 0)]
        p1, p2 = P[i], P[i + 1]
        p3 = P[min(i + 2, len(P) - 1)]
        for t in np.linspace(0, 1, n // (len(P) - 1), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    out = np.array(out)
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    Ltot = cum[-1]
    dist_end = Ltot - cum
    t = cum / Ltot
    rr = r * (1 - 0.24 * smoothstep(0.25, 1.0, t))            # taper toward the tip
    for tj in (0.42, 0.70):                                    # knuckle bulges (PIP / DIP)
        rr *= 1 + 0.07 * np.exp(-((t - tj) / 0.05) ** 2)
    rt = rr[-1]
    rr = rr * np.where(dist_end < rt, np.sqrt(np.clip(1 - ((rt - dist_end) / rt) ** 2, 0.02, 1)), 1.0)
    return out, rr


def make_nail(name, path, rr, up, mat, coll):
    tip = path[-1]
    tg = path[-1] - path[-4]
    tg /= np.linalg.norm(tg)
    upv = np.array(up) - tg * (np.array(up) @ tg)
    upv /= np.linalg.norm(upv)
    side = np.cross(tg, upv)
    r = rr[-6]
    me = uv_sphere(name, 1.0, 16, 10)
    co = get_co(me)
    c = tip - tg * r * 1.15 + upv * r * 0.80
    co = c + co[:, 0:1] * tg * (r * 1.05) + co[:, 1:2] * side * (r * 0.78) + co[:, 2:3] * upv * (r * 0.22)
    set_co(me, co)
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), coll)


def build_hands_phone(C, T, skin_hand_mat):
    pc, ux, ul, v = phone_frame()
    pc, ux, ul, v = np.array(pc), np.array(ux), np.array(ul), np.array(v)
    nb = -v                                       # phone back normal (away from her)
    objs = []
    # ---- phone: bevelled slab, dark glass screen side, pale pink case back
    pw, ph, pt = PHONE
    me = bpy.data.meshes.new("Phone_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vv in bm.verts:
        vv.co = Vector((vv.co.x * pw, vv.co.y * ph, vv.co.z * pt))
    bmesh.ops.bevel(bm, geom=[e for e in bm.edges], offset=0.0032, segments=4, affect='EDGES', profile=0.5)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    phone = link(bpy.data.objects.new("Phone", me), C["char"])
    M = Matrix(((ux[0], ul[0], -nb[0] * 0 + v[0]), (ux[1], ul[1], v[1]), (ux[2], ul[2], v[2]))).to_4x4()
    M.translation = Vector(pc)
    phone.matrix_world = M                        # local +Z = screen side (toward her)
    m, nt = new_mat("Phone_Case")
    bs = principled(nt)
    gz = N(nt, "ShaderNodeTexCoord", (-600, 0))
    sz = N(nt, "ShaderNodeSeparateXYZ", (-400, 0))
    L(nt, gz.outputs["Object"], sz.inputs[0])
    scr = map_range(nt, sz.outputs["Z"], 0.0025, 0.0035, 0, 1, (-200, 0))
    L(nt, mix_rgb(nt, scr, (*srgb(214, 196, 200), 1), (0.004, 0.004, 0.005, 1), (0, 200)), bs.inputs["Base Color"])
    L(nt, map_range(nt, sz.outputs["Z"], 0.0025, 0.0035, 0.35, 0.05, (0, -200)), bs.inputs["Roughness"])
    output(nt, bs.outputs[0], (300, 0))
    me.materials.append(m)
    objs.append(phone)
    # ---- hands
    nail_mat = simple_material("Nail", (236, 186, 180), 0.18, sss=0.3)
    for sd, sg in (("L", 1), ("R", -1)):
        W2 = np.array(T["_W2." + sd])
        G = np.array(grip_point(sd))
        f = np.array(T["_dirf." + sd])
        edge = pc + ux * sg * (pw / 2)
        back_plane = (pc + nb * (pt / 2)) @ nb
        # back of the hand inside the cuff (ellipsoid), knuckles at the phone edge
        K = W2 + f * 0.088
        hand_axis = K - W2
        hl = np.linalg.norm(hand_axis)
        pm = uv_sphere("Hand_Palm." + sd, 1.0, 24, 16)
        co = get_co(pm)
        hx = hand_axis / hl
        hy = ul - hx * (ul @ hx)
        hy /= np.linalg.norm(hy)
        hz = np.cross(hx, hy)
        co = (co[:, 0:1] * hx * (hl * 0.66) + co[:, 1:2] * hy * 0.041 + co[:, 2:3] * hz * 0.0165) + (W2 + hand_axis * 0.47)
        set_co(pm, co)
        palm = link(bpy.data.objects.new("Hand_Palm." + sd, pm), C["char"])
        pm.materials.append(skin_hand_mat)
        objs.append(palm)
        heights = [0.027, 0.009, -0.009, -0.026] if True else None
        radii = [0.0074, 0.0077, 0.0072, 0.0062]
        reach = [0.030, 0.035, 0.031, 0.022]
        for k, (h, r, rc) in enumerate(zip(heights, radii, reach)):
            hk = h + G @ ul - pc @ ul
            base = K + ul * (hk - (K - pc) @ ul) * 0.85
            # knuckle just outside the phone edge, slightly behind its back plane
            kn = pc + ul * hk + ux * sg * (pw / 2 + 0.011) + nb * (pt / 2 + 0.010)
            mid = pc + ul * hk + ux * sg * (pw / 2 - 0.008) + nb * (pt / 2 + r * 0.95)
            tip = pc + ul * (hk - 0.003) + ux * sg * (pw / 2 - rc) + nb * (pt / 2 + r * 0.85)
            pts = [base - f * 0.02, base, kn, mid, tip]
            path, rr = finger_path(pts, r)
            fm = tube_mesh("Finger.%s.%d" % (sd, k), path, rr)
            fo = link(bpy.data.objects.new("Finger.%s.%d" % (sd, k), fm), C["char"])
            fm.materials.append(skin_hand_mat)
            objs.append(fo)
            objs.append(make_nail("Nail.%s.%d" % (sd, k), path, rr, nb, nail_mat, C["char"]))
        # thumb on the screen side, along the phone edge
        tb = K - f * 0.02 + nb * 0.004
        t1 = pc + ul * (G @ ul - pc @ ul - 0.012) + ux * sg * (pw / 2 + 0.006) - nb * (pt / 2 + 0.006)
        t2 = pc + ul * (G @ ul - pc @ ul + 0.016) + ux * sg * (pw / 2 - 0.010) - nb * (pt / 2 + 0.0085)
        path, rr = finger_path([tb, (tb + t1) / 2 - nb * 0.004, t1, t2], 0.0092)
        tm = tube_mesh("Thumb." + sd, path, rr)
        to = link(bpy.data.objects.new("Thumb." + sd, tm), C["char"])
        tm.materials.append(skin_hand_mat)
        objs.append(to)
    for o in objs:
        if o.type == 'MESH' and o.name != "Phone":
            mod = o.modifiers.new("Subdivision", 'SUBSURF')
            mod.levels, mod.render_levels = 1, 1
    return objs


def make_cameras(C):
    cams = {}
    cd = bpy.data.cameras.new("Cam_Beauty")
    cd.lens = CAM_LENS
    cd.sensor_fit = 'AUTO'
    cd.sensor_width = 36
    cd.clip_start = 0.008
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = 2.6
    cam = link(bpy.data.objects.new("Cam_Beauty", cd), C["cams"])
    cam.location = CAM_BEAUTY
    cam.rotation_euler = CAM_ROT.to_euler()
    focus = bpy.data.objects.new("Focus_Target", None)
    link(focus, C["cams"])
    focus.location = (-0.03, -0.15, 1.49)
    cd.dof.focus_object = focus
    cams["beauty"] = cam
    # diagnostic turntable cameras around the character's vertical axis
    # turntable azimuths 0/45/90/180 about the vertical axis.  The phone is held high in front
    # of the face, so the front and 45 deg cameras are raised (about 33 / 17 deg elevation, like
    # the selfie itself): the phone and forearms then project onto the chest and the face is
    # visible above them.  Side and back views are level.
    centre = Vector((0.0, -0.22, 1.44))
    for name, az, cam_dz, aim_dz in (("front", 0, 1.60, 0.05), ("three-quarter", 45, 0.75, 0.04),
                                     ("side", 90, 0.04, 0.0), ("back", 180, 0.06, 0.0)):
        d = bpy.data.cameras.new("Cam_" + name)
        d.lens = 95
        d.clip_start = 0.05
        c = link(bpy.data.objects.new("Cam_" + name, d), C["cams"])
        a = math.radians(az)
        dist = 2.45
        c.location = centre + Vector((math.sin(a) * dist, -math.cos(a) * dist, cam_dz))
        c.rotation_euler = (centre + Vector((0, 0, aim_dz)) - c.location).to_track_quat('-Z', 'Y').to_euler()
        cams[name] = c
    return cams


def render_settings(sc, cam):
    sc.camera = cam
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 32
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.render.threads_mode = 'FIXED'
    sc.render.threads = 3
    sc.render.resolution_x, sc.render.resolution_y = 960, 1040
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    # Khronos PBR Neutral keeps the hue/saturation of bright albedo (skin, pink knit) close to
    # the photographed colours, where AgX desaturated them toward grey
    sc.view_settings.view_transform = 'Khronos PBR Neutral'
    sc.view_settings.look = 'None'
    sc.view_settings.exposure = -0.25
    sc.render.film_transparent = False


# ----------------------------------------------------------------------------
# 8. eyelashes (rooted on the detected lid margins) and eyebrows
# ----------------------------------------------------------------------------
def lash_material(name="Lash_Hair", c0=(0.008, 0.005, 0.004), c1=(0.022, 0.012, 0.008)):
    m, nt = new_mat(name)
    hb = principled(nt, (200, 0))
    info = N(nt, "ShaderNodeHairInfo", (-400, 0))
    col = mix_rgb(nt, info.outputs["Random"], (*c0, 1), (*c1, 1), (-100, 100))
    L(nt, col, hb.inputs["Base Color"])
    hb.inputs["Roughness"].default_value = 0.42
    hb.inputs["Coat Weight"].default_value = 0.05
    output(nt, hb.outputs[0], (500, 0))
    return m


def build_lashes_brows(C, body, eyes, T, rng):
    Ah = T["head"]
    A = np.array(Ah)
    Ai = np.array(Ah.inverted())
    co, tris = evaluated_mesh_arrays(body)
    loc = co @ Ai[:3, :3].T + Ai[:3, 3]
    bvh = BVHTree.FromPolygons(loc.tolist(), tris.tolist())
    to_w = lambda P: P @ A[:3, :3].T + A[:3, 3]
    lashes, brows, lash_roots = [], [], []
    fwd = np.array([0, -1.0, 0])
    up = np.array([0, 0, 1.0])
    for side, sg in (("L", 1), ("R", -1)):
        cw, r_eye, gaze = eyes[side]
        ec = np.array(Ah.inverted() @ cw)
        lat = np.array([sg, 0, 0.0])
        rr = r_eye + 0.0004

        def margin(phi):
            for a_deg in np.arange(4, 80, 0.5):
                a = math.radians(a_deg)
                d = math.cos(a) * fwd + math.sin(a) * (math.cos(phi) * lat + math.sin(phi) * up)
                hit = bvh.ray_cast(Vector(ec), Vector(d), rr + 0.004)
                if hit[0] is not None:
                    return a, d
            return None, None
        for lid, phis, n_l, Lr in (("upper", np.radians(np.linspace(4, 176, 230)), 230, (0.0060, 0.0095)),
                                   ("lower", np.radians(np.linspace(200, 340, 34)), 34, (0.0020, 0.0036))):
            for phi in phis + rng.normal(0, 0.012, len(phis)):
                a, d = margin(phi)
                if a is None:
                    continue
                a2 = a + math.radians(1.6)
                d2 = math.cos(a2) * fwd + math.sin(a2) * (math.cos(phi) * lat + math.sin(phi) * up)
                root = ec + d2 * (r_eye + 0.0011)
                lash_roots.append(ec + d * (r_eye + 0.0016))
                radial = math.cos(phi) * lat + math.sin(phi) * up
                lateral_t = (math.cos(phi) + 1) / 2         # 1 at the outer corner
                if lid == "upper":
                    L = Lr[0] + (Lr[1] - Lr[0]) * math.sin(math.pi * min(1.0, 0.25 + 0.85 * lateral_t)) ** 0.7
                    L *= rng.uniform(0.85, 1.08)
                    d0 = 0.9 * fwd + 0.28 * radial + 0.05 * up
                    curl = 0.42 * up + 0.12 * radial + 0.18 * lat * (lateral_t - 0.3)
                else:
                    L = rng.uniform(*Lr)
                    d0 = 0.7 * fwd + 0.6 * radial
                    curl = -0.25 * up
                d0 /= np.linalg.norm(d0)
                t = np.linspace(0, 1, 6)
                P = root + L * (t[:, None] * d0 + (t ** 2)[:, None] * curl * 0.6)
                lashes.append(to_w(P))
        # ---- brow: soft, straight-ish natural brow
        nb = 420
        u = rng.random(nb * 3)
        x = sg * (0.012 + u * 0.046)
        zc = 1.5685 + 0.0040 * np.sin(np.pi * np.clip(u * 1.05, 0, 1)) ** 0.8 - 0.005 * np.clip(u - 0.75, 0, 1) / 0.25
        half = 0.0030 * (1 - 0.6 * u) + 0.0006
        z = zc + rng.uniform(-1, 1, len(u)) * half
        dens = np.clip(1.15 - 0.6 * u, 0, 1) * (rng.random(len(u)) < 0.6)
        sel = np.nonzero(dens > rng.random(len(u)))[0][:nb]
        for i in sel:
            o = Vector((x[i], -0.30, z[i]))
            hit = bvh.ray_cast(o, Vector((0, 1, 0)), 0.3)
            if hit[0] is None:
                continue
            p0 = np.array(hit[0])
            n0 = np.array(hit[1])
            ui = u[i]
            # hairs point up-outward at the head, outward along the body, down-outward at the tail
            ang = math.radians(70 - 95 * ui + rng.normal(0, 8))
            tdir = np.array([sg * math.cos(ang), 0.0, math.sin(ang)])
            tdir = tdir - n0 * (tdir @ n0)
            tdir /= np.linalg.norm(tdir)
            L = rng.uniform(0.0045, 0.0072) * (1 - 0.3 * ui)
            t = np.linspace(0, 1, 4)
            P = p0 + n0 * 0.0002 + L * (t[:, None] * (tdir + 0.28 * n0) - (t ** 2)[:, None] * 0.25 * n0)
            brows.append(to_w(P))
    lm = lash_material()
    lo = make_curves_object("Eyelashes", lashes, 0.000105, 0.00002, C["hair"], lm)
    bm_ = lash_material("Brow_Hair", (0.10, 0.055, 0.032), (0.16, 0.09, 0.05))
    bo = make_curves_object("Eyebrows", brows, 0.000048, 0.000014, C["hair"], bm_)
    # darken the skin along the lash line (tight-line where the lashes root)
    me = body.data
    rl = np.array(lash_roots)
    kdl = KDTree(len(rl))
    for i, p in enumerate(rl):
        kdl.insert(Vector(p), i)
    kdl.balance()
    rgba = np.empty(len(me.vertices) * 4, np.float32)
    me.color_attributes["skin_col"].data.foreach_get("color", rgba)
    rgba = rgba.reshape(-1, 4)
    dark = np.array(srgb(92, 58, 48))
    bl = get_co(me) @ Ai[:3, :3].T + Ai[:3, 3]
    lash_line = np.zeros(len(bl))
    for vi in range(len(bl)):
        if abs(bl[vi, 2] - 1.545) > 0.03 or bl[vi, 1] > -0.11:
            continue
        _, _, dd = kdl.find(Vector(bl[vi]))
        w = float(np.clip(1 - dd / 0.0016, 0, 1)) ** 1.5 * 0.85
        if w > 0:
            rgba[vi, :3] = rgba[vi, :3] * (1 - w) + dark * w
        # tight-line used on top of the photo layer (upper lid stronger than lower)
        lash_line[vi] = float(np.clip(1 - dd / 0.0019, 0, 1)) ** 1.3
    me.color_attributes["skin_col"].data.foreach_set("color", rgba.ravel())
    set_attr_float(me, "lash_line", lash_line)
    REPORT["lashes"] = len(lashes)
    REPORT["brow_hairs"] = len(brows)
    log("lashes", len(lashes), "brow hairs", len(brows))
    return lo, bo


# ----------------------------------------------------------------------------
# 9. cafe suggestion (kept simple; defocused in the beauty shot)
# ----------------------------------------------------------------------------
def wood_material(name, c1, c2, scale=6.0):
    m, nt = new_mat(name)
    bs = principled(nt, (300, 0))
    tc = N(nt, "ShaderNodeTexCoord", (-700, 0))
    mp = N(nt, "ShaderNodeMapping", (-500, 0))
    mp.inputs["Scale"].default_value = (1.0, 1.0, 12.0)
    L(nt, tc.outputs["Object"], mp.inputs[0])
    wv = N(nt, "ShaderNodeTexWave", (-300, 0), wave_type='BANDS', bands_direction='X')
    wv.inputs["Scale"].default_value = scale
    wv.inputs["Distortion"].default_value = 6.0
    wv.inputs["Detail"].default_value = 3.0
    L(nt, mp.outputs[0], wv.inputs["Vector"])
    L(nt, mix_rgb(nt, wv.outputs["Fac"], (*srgb(*c1), 1), (*srgb(*c2), 1), (0, 100)), bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.45
    output(nt, bs.outputs[0], (600, 0))
    return m


def simple_material(name, rgb, rough=0.5, emit=None, transmission=0.0, sss=0.0):
    m, nt = new_mat(name)
    bs = principled(nt)
    bs.inputs["Base Color"].default_value = (*srgb(*rgb), 1)
    bs.inputs["Roughness"].default_value = rough
    bs.inputs["Transmission Weight"].default_value = transmission
    if sss:
        bs.inputs["Subsurface Weight"].default_value = sss
        bs.inputs["Subsurface Scale"].default_value = 0.01
    if emit:
        bs.inputs["Emission Color"].default_value = (*emit[0], 1)
        bs.inputs["Emission Strength"].default_value = emit[1]
    output(nt, bs.outputs[0], (300, 0))
    return m


def box(name, coll, size, loc, mat, rot=(0, 0, 0), bevel=0.0):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    if bevel:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=2, affect='EDGES')
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me), coll)
    ob.location = loc
    ob.rotation_euler = rot
    me.materials.append(mat)
    return ob


def cylinder(name, coll, r, h, loc, mat, seg=24):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=h)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(me)
    bm.free()
    ob = link(bpy.data.objects.new(name, me), coll)
    ob.location = loc
    me.materials.append(mat)
    return ob


def leaf_cluster(name, coll, centre, radius, n, mat, rng, size=(0.05, 0.025)):
    """Plant foliage: many small curved leaves filling an ellipsoid crown (denser outside)."""
    verts, faces = [], []
    for i in range(n):
        dvec = rng.normal(0, 1, 3)
        dvec /= np.linalg.norm(dvec)
        p = np.array(centre) + dvec * np.array(radius) * rng.uniform(0.55, 1.0) ** 0.5
        nrm = dvec * 0.6 + rng.normal(0, 0.5, 3) + np.array([0, 0, 0.5])
        nrm /= np.linalg.norm(nrm)
        t = np.cross(nrm, [0, 0, 1.0])
        if np.linalg.norm(t) < 1e-3:
            t = np.array([1.0, 0, 0])
        t /= np.linalg.norm(t)
        b = np.cross(nrm, t)
        l, w = size[0] * rng.uniform(0.7, 1.3), size[1] * rng.uniform(0.7, 1.3)
        base = len(verts)
        for (u, v) in ((-0.5, 0), (-0.2, -0.45), (0.2, -0.4), (0.5, 0), (0.2, 0.4), (-0.2, 0.45)):
            q = p + t * u * l + b * v * w - nrm * (0.25 * l * (u * u))
            verts.append(q)
        faces.append(tuple(range(base, base + 6)))
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    ob = link(bpy.data.objects.new(name, me), coll)
    me.materials.append(mat)
    return ob


def build_cafe(C, rng):
    cc = C["cafe"]
    wall = simple_material("Cafe_Wall", (226, 214, 196), 0.8)
    oak = wood_material("Cafe_Oak", (176, 120, 70), (204, 150, 96))
    dark = wood_material("Cafe_Walnut", (110, 70, 40), (140, 92, 54), 9.0)
    leaf = simple_material("Cafe_Leaf", (84, 132, 52), 0.5, sss=0.3)
    glass = simple_material("Cafe_Glass", (240, 240, 240), 0.05, transmission=1.0)
    ceramic = simple_material("Cafe_Ceramic", (236, 232, 222), 0.3)
    steel = simple_material("Cafe_Steel", (180, 176, 170), 0.25)
    flower = simple_material("Cafe_Flower", (246, 244, 236), 0.6, sss=0.4)
    window = simple_material("Cafe_Window_Glow", (255, 255, 255), 1.0, emit=((1.0, 0.97, 0.9), 6.0))
    # room shell
    box("Floor", cc, (8, 8, 0.02), (0, 1.0, -0.01), dark)
    box("Back_Wall", cc, (8, 0.1, 3.2), (0, 3.2, 1.6), wall)
    box("Right_Wall", cc, (0.1, 8, 3.2), (2.6, 1.0, 1.6), wall)
    # big bright window on her right-front (camera left) with mullions
    box("Window_Light", cc, (0.02, 3.0, 2.2), (-2.4, 0.6, 1.5), window)
    for k in range(4):
        box("Window_Mullion.%d" % k, cc, (0.05, 0.05, 2.3), (-2.36, -0.9 + k * 1.0, 1.5), dark)
    # counter + shelving with jars at the back right
    box("Counter", cc, (2.2, 0.6, 1.0), (1.2, 2.7, 0.5), oak, bevel=0.01)
    for k, z in enumerate((1.45, 1.85, 2.25)):
        box("Shelf.%d" % k, cc, (2.0, 0.28, 0.035), (1.2, 3.0, z), oak)
        for j in range(9):
            x = 0.35 + j * 0.2 + rng.uniform(-0.03, 0.03)
            h = rng.uniform(0.12, 0.22)
            m = [glass, ceramic, steel][int(rng.integers(0, 3))]
            cylinder("Jar.%d.%d" % (k, j), cc, rng.uniform(0.04, 0.06), h, (x, 3.0, z + 0.0175 + h / 2), m)
    # tables & chairs
    for k, (x, y) in enumerate(((1.05, 0.9), (1.6, 2.0), (-0.9, 1.6))):
        box("Table_Top.%d" % k, cc, (0.8, 0.8, 0.04), (x, y, 0.76), oak, bevel=0.008)
        cylinder("Table_Leg.%d" % k, cc, 0.035, 0.74, (x, y, 0.37), dark)
    # table in front of her (lower frame edge in the reference)
    box("Table_Top.near", cc, (1.0, 0.7, 0.04), (0.05, -0.55, 0.74), oak, bevel=0.008)
    # her chair back
    box("Chair_Back", cc, (0.46, 0.035, 0.3), (0.0, 0.26, 0.95), oak, rot=(math.radians(-8), 0, 0), bevel=0.01)
    for sx in (-0.2, 0.2):
        box("Chair_Post.%d" % int(sx * 10), cc, (0.035, 0.035, 0.95), (sx, 0.25, 0.48), oak)
    # vase with white flowers on the right table
    cylinder("Vase", cc, 0.05, 0.18, (0.85, 0.75, 0.87), glass)
    leaf_cluster("Vase_Flowers", cc, (0.85, 0.75, 1.06), (0.08, 0.08, 0.06), 140, flower, rng, (0.018, 0.014))
    leaf_cluster("Vase_Leaves", cc, (0.85, 0.75, 1.0), (0.06, 0.06, 0.06), 60, leaf, rng, (0.04, 0.012))
    # potted plants near the window
    for k, (x, y, hgt) in enumerate(((-1.7, -0.2, 1.7), (-1.6, 1.4, 1.4), (-1.1, 2.6, 1.9))):
        cylinder("Planter.%d" % k, cc, 0.22, 0.45, (x, y, 0.225), ceramic)
        cylinder("Plant_Trunk.%d" % k, cc, 0.02, hgt, (x, y, hgt / 2 + 0.3), dark, 8)
        leaf_cluster("Plant_Leaves.%d" % k, cc, (x, y, 0.45 + hgt * 0.62), (0.38, 0.38, hgt * 0.36), 1600, leaf, rng, (0.075, 0.04))
    return cc


def build_lights(C):
    for ob in list(C["light"].objects):
        bpy.data.objects.remove(ob)

    def area(name, loc, target, size, power, color):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.shape = 'RECTANGLE'
        ld.size, ld.size_y = size
        ld.energy = power
        ld.color = color
        ob = link(bpy.data.objects.new(name, ld), C["light"])
        ob.location = loc
        ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return ob
    # window light from camera-left, soft fill from the phone side, warm cafe rim from behind
    area("Key_Window", (-1.45, -0.75, 2.0), (0, -0.1, 1.5), (0.9, 1.1), 120, (1.0, 0.96, 0.9))
    area("Fill_Front", (0.45, -1.25, 1.95), (0, -0.1, 1.5), (1.2, 1.2), 30, (1.0, 0.93, 0.87))
    area("Rim_Back", (-0.8, 1.2, 2.2), (0, 0, 1.55), (1.4, 1.4), 160, (1.0, 0.86, 0.70))
    area("Rim_Back_R", (1.0, 1.0, 1.9), (0, 0, 1.45), (1.0, 1.0), 70, (1.0, 0.88, 0.76))
    area("Cafe_Ceiling", (0.6, 1.5, 2.9), (0.6, 1.5, 0.0), (3.0, 3.0), 260, (1.0, 0.85, 0.68))
    # small soft source just above the selfie camera: catchlights + gentle frontal fill
    area("Catchlight", tuple(CAM_BEAUTY + Vector((0.05, -0.10, 0.30))), (0, -0.12, 1.53), (0.10, 0.07), 6, (1.0, 0.97, 0.93))
    dl = area("Diag_Fill_Back", (0.0, 1.6, 1.7), (0, 0, 1.4), (1.6, 1.6), 60, (1.0, 0.95, 0.9))
    dl.hide_render = True                         # only enabled for diagnostic turntable views
    w = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.75, 0.68, 0.6, 1)
    bg.inputs[1].default_value = 0.35


def finalize(sc, C):
    # tidy: smooth shading on all character meshes, viewport display of hair
    for ob in C["char"].all_objects:
        if ob.type == 'MESH':
            for p in ob.data.polygons:
                p.use_smooth = True
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    counts, tot_v = {}, 0
    mins, maxs = np.full(3, 1e9), np.full(3, -1e9)
    for ob in C["char"].all_objects:
        ev = ob.evaluated_get(dg)
        if ob.type == 'MESH':
            me = ev.to_mesh()
            co = get_co(me)
            mw = np.array(ob.matrix_world)
            co = co @ mw[:3, :3].T + mw[:3, 3]
            counts[ob.name] = {"base_vertices": len(ob.data.vertices), "evaluated_vertices": len(me.vertices),
                               "evaluated_faces": len(me.polygons)}
            tot_v += len(me.vertices)
            mins, maxs = np.minimum(mins, co.min(0)), np.maximum(maxs, co.max(0))
            ev.to_mesh_clear()
        elif ob.type == 'CURVES':
            n = len(ob.data.points)
            counts[ob.name] = {"curves": len(ob.data.curves), "control_points": n}
            co = np.empty(n * 3, np.float32)
            ob.data.position_data.foreach_get("vector", co)
            co = co.reshape(-1, 3)
            mins, maxs = np.minimum(mins, co.min(0)), np.maximum(maxs, co.max(0))
    REPORT["objects"] = counts
    REPORT["character_evaluated_mesh_vertices"] = tot_v
    REPORT["character_bounds_m"] = {"min": mins.round(4).tolist(), "max": maxs.round(4).tolist(),
                                    "size": (maxs - mins).round(4).tolist()}
    sc.camera = bpy.data.objects["Cam_Beauty"]
    bpy.ops.file.pack_all()
    out = os.path.join(ROOT, "character.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    log("saved", out)
    rep = {
        "blender_version": bpy.app.version_string,
        "render_engine": sc.render.engine,
        "render_device": sc.cycles.device,
        "samples": sc.cycles.samples,
        "denoising": sc.cycles.use_denoising,
        "threads": sc.render.threads,
        "color_management": {"view_transform": sc.view_settings.view_transform, "look": sc.view_settings.look},
        "collections": {c.name: sorted(o.name for o in c.objects) for c in bpy.data.collections},
        "important_objects": ["Body_Skin", "Eyeball.L", "Eyeball.R", "Cornea.L", "Cornea.R", "Eyelashes", "Eyebrows",
                              "Hair_Strands", "Hair_Bangs", "Sweater", "Sweater_Collar", "Hand_Palm.L", "Hand_Palm.R",
                              "Phone", "Cam_Beauty", "Cam_front", "Cam_three-quarter", "Cam_side", "Cam_back"],
        "materials": sorted(m.name for m in bpy.data.materials),
        "packed_images": [i.name for i in bpy.data.images if i.packed_file],
    }
    rep.update(REPORT)
    os.makedirs(os.path.join(ROOT, "previews"), exist_ok=True)
    with open(os.path.join(ROOT, "previews", "build-report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    log("build report written")


def stage(name):
    log("stage done:", name)
    if STOP_AFTER == name:
        out = os.path.join(ROOT, os.environ.get("STAGE_BLEND", "previews/stage.blend"))
        bpy.ops.wm.save_as_mainfile(filepath=out)
        log("saved stage file", out)
        sys.exit(0)


def main():
    sc = reset_scene()
    C = {}
    C["char"] = new_coll("Character")
    C["hair"] = new_coll("Hair", C["char"])
    C["cloth"] = new_coll("Clothing", C["char"])
    C["eyes"] = new_coll("Eyes", C["char"])
    C["cafe"] = new_coll("Cafe_Set")
    C["light"] = new_coll("Lighting")
    C["cams"] = new_coll("Cameras")
    body, T, rest, posed = build_body(C)
    stage("body")
    skin_attributes(body.data)
    skin_mat = skin_material()
    body.data.materials.append(skin_mat)
    sub = body.modifiers.new("Subdivision", 'SUBSURF')
    sub.levels, sub.render_levels = 1, 1
    eyes = build_eyes(C, body, T)
    preview_lights(C)
    stage("eyes")
    sweater, sV, neck_loop, region = build_sweater(C, body, T, attr_vec(body.data, "rest_co"))
    collar = build_collar(C, sV, neck_loop, body)
    imgs = make_knit_images()
    km = sweater_material(imgs)
    sweater.data.materials.append(km)
    collar.data.materials.append(km)
    trim_skin_under_sweater(body)
    stage("sweater")
    rng = np.random.default_rng(2026)
    eye_objs = [bpy.data.objects["Eyeball.L"], bpy.data.objects["Eyeball.R"]]
    hair_main, hair_bangs, bvh = build_hair(C, body, [body, sweater, collar] + eye_objs, T, rng)
    stage("hair")
    build_lashes_brows(C, body, eyes, T, rng)
    hand_mat = skin_material("Skin_Hands", use_attr=False)
    hands = build_hands_phone(C, T, hand_mat)
    cams = make_cameras(C)
    img, med = photo_skin_texture(body, cams["beauty"], T)
    add_photo_layer(skin_mat, img, med)
    photo_iris_layers(cams["beauty"])
    tint_procedural_skin(body, hand_mat, med)
    export_texture_copies()
    build_cafe(C, np.random.default_rng(5))
    build_lights(C)
    render_settings(sc, cams["beauty"])
    stage("hands")
    finalize(sc, C)
    # ---- reopen the saved file, check it, render the finals from it, write the report
    path = os.path.join(ROOT, "character.blend")
    bpy.ops.wm.open_mainfile(filepath=path)
    sc = bpy.context.scene
    unpacked = [i.name for i in bpy.data.images if i.type == 'IMAGE' and not i.packed_file]
    reopen = {"file": "character.blend", "objects": len(bpy.data.objects),
              "scene_camera": sc.camera.name if sc.camera else None,
              "images_not_packed": unpacked,
              "skin_photo_texture_packed": bool(bpy.data.images["Skin_Photo_RGBA"].packed_file),
              "file_size_mb": round(os.path.getsize(path) / 1e6, 1)}
    log("reopened:", reopen)
    assert not unpacked, unpacked
    times = None
    if not os.environ.get("SKIP_RENDER"):
        import render
        times = render.render_views(root=ROOT)
    import scene_report
    scene_report.write_report(ROOT, times, reopen)
    log("scene-report.json written")


def export_texture_copies():
    """Write inspection copies of the generated photo textures (the .blend uses packed copies)."""
    os.makedirs(os.path.join(ROOT, "textures"), exist_ok=True)
    for name in ("Skin_Photo_RGBA", "Iris_Photo.L", "Iris_Photo.R"):
        im = bpy.data.images[name]
        W, H = im.size
        px = np.empty(W * H * 4, np.float32)
        im.pixels.foreach_get(px)
        tmp = bpy.data.images.new(name + "_copy", W, H, alpha=True)
        tmp.pixels.foreach_set(px)
        tmp.filepath_raw = os.path.join(ROOT, "textures", name + ".png")
        tmp.file_format = 'PNG'
        tmp.save()
        bpy.data.images.remove(tmp)


if __name__ == "__main__":
    main()
