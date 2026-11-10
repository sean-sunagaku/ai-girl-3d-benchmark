"""Build-time skeleton from the supplied landmarks, bone-heat weights from Blender, and a numpy pose solver.

The armature is only a construction tool: the pose is baked into the delivered meshes (no production rig).
"""
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector


def _R(axis, deg):
    return np.array(Matrix.Rotation(math.radians(deg), 3, Vector(axis).normalized()))


def rot_between(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    ax = np.cross(a, b); s = np.linalg.norm(ax); c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3)
    return np.array(Matrix.Rotation(math.atan2(s, c), 3, Vector(ax / s)))


class Skeleton:
    def __init__(self, landmarks):
        J = lambda k: np.array(landmarks['joint-' + k], float)
        self.bones = []        # (name, head, tail, parent)
        add = lambda n, a, b, p=None: self.bones.append((n, np.array(a, float), np.array(b, float), p))
        add('spine2', J('spine-3'), J('spine-2'))
        add('spine1', J('spine-2'), J('spine-1'), 'spine2')
        add('chest', J('spine-1'), J('neck'), 'spine1')
        add('neck', J('neck'), J('head'), 'chest')
        add('head', J('head'), J('head-2'), 'neck')
        for s in ('l', 'r'):
            add('clav.' + s, J(s + '-clavicle'), J(s + '-shoulder'), 'chest')
            add('upperarm.' + s, J(s + '-shoulder'), J(s + '-elbow'), 'clav.' + s)
            add('forearm.' + s, J(s + '-elbow'), J(s + '-hand'), 'upperarm.' + s)
            add('hand.' + s, J(s + '-hand'), J(s + '-hand-2'), 'forearm.' + s)
            for f in range(1, 6):
                par = 'hand.' + s
                for k in range(1, 4):
                    nm = 'f%d%d.%s' % (f, k, s)
                    add(nm, J('%s-finger-%d-%d' % (s, f, k)), J('%s-finger-%d-%d' % (s, f, k + 1)), par)
                    par = nm
        self.index = {b[0]: i for i, b in enumerate(self.bones)}
        self.names = [b[0] for b in self.bones]
        self.head = {b[0]: b[1] for b in self.bones}
        self.tail = {b[0]: b[2] for b in self.bones}
        self.parent = {b[0]: b[3] for b in self.bones}
        self.landmarks = landmarks

    def bone_heat_weights(self, body_obj):
        """Let Blender compute automatic (bone heat) weights; return an (n_verts, n_bones) array."""
        arm = bpy.data.armatures.new('BuildRig')
        rig = bpy.data.objects.new('BuildRig', arm)
        bpy.context.scene.collection.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        for name, h, t, p in self.bones:
            eb = arm.edit_bones.new(name); eb.head = tuple(h); eb.tail = tuple(t)
            if p:
                eb.parent = arm.edit_bones[p]
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        body_obj.select_set(True); rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        n = len(body_obj.data.vertices)
        W = np.zeros((n, len(self.bones)))
        gmap = {g.index: self.index[g.name] for g in body_obj.vertex_groups if g.name in self.index}
        for v in body_obj.data.vertices:
            for g in v.groups:
                if g.group in gmap:
                    W[v.index, gmap[g.group]] = g.weight
        s = W.sum(1)
        assert (s > 1e-6).all(), 'bone heat left unweighted vertices'
        W /= s[:, None]
        # the skull, face and jaw must move as one rigid piece (bone heat lets the neck bone pull on the chin)
        V = np.array([v.co[:] for v in body_obj.data.vertices])
        t = V[:, 2] - (1.425 + (V[:, 1] + 0.14) * 0.30)
        r = np.clip((t + 0.010) / 0.022, 0.0, 1.0); r = r * r * (3 - 2 * r)
        r *= (np.abs(V[:, 0]) < 0.12)
        hi = self.index['head']
        W *= (1.0 - r)[:, None]
        W[:, hi] += r
        # remove the construction rig again
        for m in list(body_obj.modifiers):
            if m.type == 'ARMATURE':
                body_obj.modifiers.remove(m)
        body_obj.parent = None
        body_obj.vertex_groups.clear()
        bpy.data.objects.remove(rig, do_unlink=True)
        bpy.data.armatures.remove(arm)
        return W

    # ------------------------------------------------------------------------------- pose solve
    def solve(self, rot):
        """rot: {bone: 3x3 world-axis delta rotation about the bone head}. Returns {bone: 4x4}."""
        M = {}
        for name, h, t, p in self.bones:
            Mp = M[p] if p else np.eye(4)
            L = np.eye(4)
            R = rot.get(name)
            if R is not None:
                L[:3, :3] = R
                L[:3, 3] = h - R @ h
            M[name] = Mp @ L
        return M

    @staticmethod
    def apply(M, p):
        return M[:3, :3] @ np.asarray(p, float) + M[:3, 3]

    def skin(self, V, W, M):
        out = np.zeros_like(V)
        for name, i in self.index.items():
            w = W[:, i]
            nz = w > 1e-7
            if not nz.any():
                continue
            out[nz] += w[nz, None] * (V[nz] @ M[name][:3, :3].T + M[name][:3, 3])
        return out


def arm_ik(sk, rot, side, wrist_target, elbow_hint, hand_fwd=None, hand_palm=None):
    """Analytic two-bone IK in world space. Fills rot[] for upperarm/forearm/hand of one side."""
    s = side
    M = sk.solve(rot)
    Mc = M['clav.' + s]
    S = sk.apply(Mc, sk.head['upperarm.' + s])
    E0 = sk.apply(Mc, sk.head['forearm.' + s])
    W0 = sk.apply(Mc, sk.head['hand.' + s])
    l1 = np.linalg.norm(E0 - S); l2 = np.linalg.norm(W0 - E0)
    T = np.asarray(wrist_target, float)
    d = T - S; dist = np.linalg.norm(d)
    dist_c = min(dist, (l1 + l2) * 0.999)
    T = S + d / dist * dist_c
    a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (2 * dist_c)
    hgt = math.sqrt(max(l1 * l1 - a * a, 0.0))
    ax = d / dist
    hint = np.asarray(elbow_hint, float) - S
    perp = hint - ax * np.dot(hint, ax)
    perp /= np.linalg.norm(perp)
    E = S + ax * a + perp * hgt
    # parent (clavicle) rotation part
    Rc = Mc[:3, :3]
    r_up = rot_between(Rc @ (sk.head['forearm.' + s] - sk.head['upperarm.' + s]), E - S)
    rot['upperarm.' + s] = np.linalg.inv(Rc) @ r_up @ Rc
    M = sk.solve(rot)
    Ru = M['upperarm.' + s][:3, :3]
    r_fo = rot_between(Ru @ (sk.head['hand.' + s] - sk.head['forearm.' + s]), T - E)
    rot['forearm.' + s] = np.linalg.inv(Ru) @ r_fo @ Ru
    if hand_fwd is not None:
        M = sk.solve(rot)
        Rf = M['forearm.' + s][:3, :3]
        L = sk.landmarks
        w = np.array(L['joint-%s-hand' % s]); i = np.array(L['joint-%s-finger-2-1' % s]); k = np.array(L['joint-%s-finger-5-1' % s])
        m = np.array(L['joint-%s-finger-3-1' % s])
        fwd0 = Rf @ (m - w); fwd0 /= np.linalg.norm(fwd0)
        pn0 = np.cross(Rf @ (i - w), Rf @ (k - w)); pn0 /= np.linalg.norm(pn0)
        if s == 'r':
            pn0 = -pn0
        f1 = np.asarray(hand_fwd, float); f1 /= np.linalg.norm(f1)
        r1 = rot_between(fwd0, f1)
        pn1 = r1 @ pn0
        if hand_palm is not None:
            tp = np.asarray(hand_palm, float); tp = tp - f1 * np.dot(tp, f1); tp /= np.linalg.norm(tp)
            ang = math.atan2(np.dot(np.cross(pn1, tp), f1), np.dot(pn1, tp))
            r2 = np.array(Matrix.Rotation(ang, 3, Vector(f1)))
        else:
            r2 = np.eye(3)
        rot['hand.' + s] = np.linalg.inv(Rf) @ (r2 @ r1) @ Rf
    return E


def finger_curl(sk, rot, side, curls, spread=0.0):
    """curls: {finger_no: (mcp, pip, dip) degrees}. Rotation axes are derived from the rest hand plane."""
    s = side
    L = sk.landmarks
    w = np.array(L['joint-%s-hand' % s]); i = np.array(L['joint-%s-finger-2-1' % s]); k = np.array(L['joint-%s-finger-5-1' % s])
    pn = np.cross(i - w, k - w); pn /= np.linalg.norm(pn)
    if s == 'r':
        pn = -pn                                  # pn now points out of the palm on both sides
    for f, angs in curls.items():
        for seg, ang in zip((1, 2, 3), angs):
            nm = 'f%d%d.%s' % (f, seg, s)
            d = sk.tail[nm] - sk.head[nm]; d /= np.linalg.norm(d)
            ax = np.cross(d, pn); ax /= np.linalg.norm(ax)
            rot[nm] = _R(ax, ang)
            if seg == 1 and spread and f != 1:
                rot[nm] = _R(pn, spread * (f - 3.5) * (1 if s == 'l' else -1)) @ rot[nm]


DEFAULT_POSE = dict(
    lean=3.0,                 # torso forward lean (deg)
    torso_roll=4.5,           # torso side lean, positive = toward her left
    head_roll=-28.5,          # lateral tilt, negative = toward her right (-X)
    head_yaw=-6.0,            # negative = turned toward her right
    head_pitch=-2.0,          # negative = chin up
    neck_share=0.42, neck_roll_share=0.82,
    r_wrist=(-0.262, -0.470, 1.350), r_elbow=(-0.385, -0.235, 1.205),
    l_wrist=(0.215, -0.490, 1.400), l_elbow=(0.385, -0.240, 1.215),
    r_hand_fwd=(0.30, -0.80, 0.50), r_hand_palm=(0.75, 0.25, -0.60),
    l_hand_fwd=(-0.22, -0.90, 0.38), l_hand_palm=(-0.65, 0.20, -0.72),
    fingers={1: (8, 14, 10), 2: (18, 26, 14), 3: (24, 32, 18), 4: (30, 36, 20), 5: (36, 38, 22)},
    r_clav_up=7.0, l_clav_up=3.0, clav_fwd=10.0,
)


def build_pose(sk, P=None):
    P = dict(DEFAULT_POSE, **(P or {}))
    rot = {}
    X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    rot['spine1'] = _R(Y, P['torso_roll'] * 0.5) @ _R(X, P['lean'] * 0.5)
    rot['chest'] = _R(Y, P['torso_roll'] * 0.5) @ _R(X, P['lean'] * 0.5)
    ns = P['neck_share']; nr = P['neck_roll_share']
    def head_part(f, fr):
        # order: yaw about Z, then pitch about X, then roll about Y (world axes at rest)
        return _R(Y, (P['head_roll'] - P['torso_roll']) * fr) @ _R(X, P['head_pitch'] * f - (P['lean'] * f)) @ _R(Z, P['head_yaw'] * f)
    rot['neck'] = head_part(ns, nr)
    rot['head'] = head_part(1.0 - ns, 1.0 - nr)
    # shoulders: lift and bring forward (protraction) on the reaching side
    rot['clav.l'] = _R(Z, -P['clav_fwd']) @ _R(Y, -P['l_clav_up'])
    rot['clav.r'] = _R(Z, P['clav_fwd']) @ _R(Y, P['r_clav_up'])
    return rot, P
