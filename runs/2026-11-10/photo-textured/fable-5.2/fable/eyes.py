"""Eyeballs with a refracting cornea, a painted (generated) iris, lashes and brows as real hair curves."""
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector
from . import util
from .sculpt import EYE_C, EYE_R, EYE_C0

R_GLOBE = EYE_R - 0.00008
R_IRIS = 0.00705
R_PUPIL = 0.00265


def _lathe(profile, n_around, close_axis=True):
    """profile: list of (rho, z). Returns V, F for a surface of revolution about +Z."""
    V = []; F = []
    rings = []
    for (rho, z) in profile:
        if rho < 1e-9:
            rings.append([len(V)]); V.append((0.0, 0.0, z))
        else:
            idx = []
            for k in range(n_around):
                a = 2 * math.pi * k / n_around
                idx.append(len(V)); V.append((rho * math.cos(a), rho * math.sin(a), z))
            rings.append(idx)
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        for k in range(n_around):
            k2 = (k + 1) % n_around
            if len(a) == 1:
                F.append((a[0], b[k], b[k2]))
            elif len(b) == 1:
                F.append((a[k], b[0], a[k2]))
            else:
                F.append((a[k], b[k], b[k2], a[k2]))
    return np.array(V), F, rings


def iris_texture(size=512, seed=3, tone=None):
    """Procedural iris albedo (sRGB). In the second pass `tone` is the median iris colour sampled from the photo."""
    y, x = np.mgrid[0:size, 0:size]
    u = (x + 0.5) / size * 2 - 1; v = (y + 0.5) / size * 2 - 1
    r = np.hypot(u, v) / 0.92                     # 1.0 = limbus
    a = np.arctan2(v, u)
    an = a / (2 * math.pi)
    fib = util.fbm2(an * 90.0, r * 2.2, 4, seed)              # fine radial fibres
    fib2 = util.fbm2(an * 34.0 + 5.0, r * 1.1, 3, seed + 5)    # broader streaks
    crypt = util.fbm2(an * 16.0 + 9.0, r * 6.0, 3, seed + 9)
    pr = R_PUPIL / R_IRIS
    t = np.clip((r - pr) / (1 - pr), 0, 1)                    # 0 at pupil edge .. 1 at limbus
    inner = np.array([0.44, 0.235, 0.105])                    # warm brown near the collarette
    mid = np.array([0.290, 0.142, 0.064])
    outer = np.array([0.115, 0.056, 0.032])
    if tone is not None:                                      # keep the authored structure, take the hue/value from the photo
        tone = np.asarray(tone, float)
        tone = np.clip(tone.mean() + (tone - tone.mean()) * 1.20, 0.02, 1)      # the photo's iris is veiled by reflections
        inner = 0.5 * inner + 0.5 * np.clip(tone * 1.00, 0, 1)
        mid = 0.5 * mid + 0.5 * tone * 0.66
        outer = 0.5 * outer + 0.5 * tone * 0.28
    c = np.where(t[..., None] < 0.42, inner + (mid - inner) * (t[..., None] / 0.42),
                 mid + (outer - mid) * ((t[..., None] - 0.42) / 0.58))
    c = c * (0.62 + 0.62 * fib[..., None]) * (0.86 + 0.28 * fib2[..., None])
    coll = np.exp(-((t - 0.20) / 0.085) ** 2) * (0.6 + 0.8 * util.fbm2(an * 40.0, r, 2, seed + 2))
    c = c + coll[..., None] * np.array([0.10, 0.065, 0.03])
    c = c * (1.0 - 0.38 * np.clip((crypt - 0.60) / 0.2, 0, 1) * (t > 0.25) * (t < 0.8))[..., None]
    limbal = np.clip((t - 0.80) / 0.20, 0, 1) ** 1.6
    c = c * (1.0 - 0.80 * limbal)[..., None]
    pup = np.clip((pr * 1.03 - r) / (pr * 0.07), 0, 1)
    c = c * (1.0 - pup)[..., None] + np.array([0.006, 0.005, 0.005]) * pup[..., None]
    out_s = np.clip((r - 1.0) / 0.05, 0, 1)                   # beyond the limbus: sclera tone (blend ring)
    c = c * (1 - out_s)[..., None] + np.array([0.80, 0.76, 0.73]) * out_s[..., None]
    return np.clip(c, 0, 1)


def eye_materials(iris_tone=None):
    img = util.new_image('Iris_Generated_PhotoTone' if iris_tone is not None else 'Iris_Generated', iris_texture(tone=iris_tone), 'sRGB')
    mats = {}
    # iris ------------------------------------------------------------------
    m = bpy.data.materials.new('Eye_Iris'); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes['Principled BSDF']
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img; tex.extension = 'EXTEND'
    nt.links.new(tex.outputs['Color'], p.inputs['Base Color'])
    p.inputs['Roughness'].default_value = 0.42
    p.inputs['Subsurface Weight'].default_value = 0.15
    p.inputs['Subsurface Radius'].default_value = (0.6, 0.3, 0.2)
    p.inputs['Subsurface Scale'].default_value = 0.0006
    m.diffuse_color = (0.25, 0.13, 0.06, 1)
    mats['iris'] = m
    # sclera ----------------------------------------------------------------
    m = bpy.data.materials.new('Eye_Sclera'); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes['Principled BSDF']
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], sep.inputs[0])
    # z of the local frame is the optical axis: redden towards the back/corners
    mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.0125; mr.inputs['From Max'].default_value = 0.004
    nt.links.new(sep.outputs['Z'], mr.inputs['Value'])
    noise = nt.nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 520.0; noise.inputs['Detail'].default_value = 5.0
    noise.inputs['Distortion'].default_value = 2.2
    nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
    vein = nt.nodes.new('ShaderNodeMapRange'); vein.inputs['From Min'].default_value = 0.60; vein.inputs['From Max'].default_value = 0.74
    nt.links.new(noise.outputs['Fac'], vein.inputs['Value'])
    mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
    nt.links.new(vein.outputs['Result'], mul.inputs[0]); nt.links.new(mr.outputs['Result'], mul.inputs[1])
    add = nt.nodes.new('ShaderNodeMath'); add.operation = 'ADD'; add.use_clamp = True
    sc = nt.nodes.new('ShaderNodeMath'); sc.operation = 'MULTIPLY'; sc.inputs[1].default_value = 0.42
    nt.links.new(mr.outputs['Result'], sc.inputs[0])
    nt.links.new(sc.outputs[0], add.inputs[0]); nt.links.new(mul.outputs[0], add.inputs[1])
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (0.80, 0.765, 0.735, 1); mix.inputs['B'].default_value = (0.70, 0.36, 0.30, 1)
    nt.links.new(add.outputs[0], mix.inputs['Factor'])
    nt.links.new(mix.outputs['Result'], p.inputs['Base Color'])
    p.inputs['Roughness'].default_value = 0.38
    p.inputs['Subsurface Weight'].default_value = 0.5
    p.inputs['Subsurface Radius'].default_value = (0.9, 0.5, 0.4)
    p.inputs['Subsurface Scale'].default_value = 0.0015
    m.diffuse_color = (0.80, 0.77, 0.74, 1)
    mats['sclera'] = m
    # cornea / tear film -----------------------------------------------------
    m = bpy.data.materials.new('Eye_Cornea'); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes['Principled BSDF']; out = nt.nodes['Material Output']
    p.inputs['Base Color'].default_value = (1, 1, 1, 1)
    p.inputs['Transmission Weight'].default_value = 1.0
    p.inputs['Roughness'].default_value = 0.015
    p.inputs['IOR'].default_value = 1.376
    lp = nt.nodes.new('ShaderNodeLightPath'); tr = nt.nodes.new('ShaderNodeBsdfTransparent')
    mixs = nt.nodes.new('ShaderNodeMixShader')
    mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'MAXIMUM'
    nt.links.new(lp.outputs['Is Shadow Ray'], mx.inputs[0]); nt.links.new(lp.outputs['Is Diffuse Ray'], mx.inputs[1])
    nt.links.new(mx.outputs[0], mixs.inputs['Fac'])
    nt.links.new(p.outputs['BSDF'], mixs.inputs[1]); nt.links.new(tr.outputs['BSDF'], mixs.inputs[2])
    nt.links.new(mixs.outputs['Shader'], out.inputs['Surface'])
    m.diffuse_color = (1, 1, 1, 0.15)
    mats['cornea'] = m
    # caruncle ----------------------------------------------------------------
    m = bpy.data.materials.new('Eye_Caruncle'); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (0.72, 0.36, 0.33, 1); p.inputs['Roughness'].default_value = 0.22
    p.inputs['Subsurface Weight'].default_value = 0.8; p.inputs['Subsurface Radius'].default_value = (1.0, 0.3, 0.2)
    p.inputs['Subsurface Scale'].default_value = 0.002
    m.diffuse_color = (0.72, 0.36, 0.33, 1)
    mats['caruncle'] = m
    return mats


def build_eyes(col, head_matrix, look_at, mats, quality=1.0):
    """Create both eyes in posed space. head_matrix: 4x4 numpy; look_at: world point both eyes fixate."""
    n_ar = 72
    hl = math.sqrt(R_GLOBE ** 2 - R_IRIS ** 2)
    prof = [(0.0, hl - 0.0016)]
    for k in range(1, 9):
        rho = R_IRIS * k / 8
        prof.append((rho, hl - 0.0005 - 0.0011 * (1 - rho / R_IRIS)))
    th_l = math.asin(R_IRIS / R_GLOBE)
    n_s = 40
    for k in range(0, n_s + 1):
        th = th_l + (math.pi - th_l) * k / n_s
        prof.append((R_GLOBE * math.sin(th) if k < n_s else 0.0, R_GLOBE * math.cos(th)))
    V, F, rings = _lathe(prof, n_ar)
    n_iris_faces = n_ar * 8
    uv = []
    for f in F:
        for i in f:
            uv.append((V[i][0] / (R_IRIS / 0.92) * 0.5 + 0.5, V[i][1] / (R_IRIS / 0.92) * 0.5 + 0.5))
    # cornea + tear film: front part only
    Rf = R_GLOBE + 0.00016
    rho_c = R_IRIS + 0.0004; r_c = 0.0088
    z_join = math.sqrt(Rf ** 2 - rho_c ** 2)
    z_c = z_join - math.sqrt(r_c ** 2 - rho_c ** 2)
    cprof = [(0.0, z_c + r_c)]
    for k in range(1, 15):
        rho = rho_c * k / 14
        zc = z_c + math.sqrt(r_c ** 2 - rho ** 2)
        zs = math.sqrt(Rf ** 2 - rho ** 2)
        b = util_smooth((rho / rho_c - 0.82) / 0.18)
        cprof.append((rho, zc * (1 - b) + max(zc, zs) * b))
    th0 = math.asin(rho_c / Rf)
    for k in range(1, 26):
        th = th0 + (math.radians(112) - th0) * k / 25
        cprof.append((Rf * math.sin(th), Rf * math.cos(th)))
    Vc, Fc, _ = _lathe(cprof, n_ar)
    objs = []
    for side, tag in ((1.0, 'L'), (-1.0, 'R')):
        c = head_matrix[:3, :3] @ (EYE_C * np.array([side, 1, 1])) + head_matrix[:3, 3]
        d = np.asarray(look_at, float) - c; d /= np.linalg.norm(d)
        # local +Z -> gaze; keep local +Y roughly world up
        zax = Vector(d); up = Vector(head_matrix[:3, :3] @ np.array([0, 0, 1.0]))
        xax = up.cross(zax).normalized(); yax = zax.cross(xax).normalized()
        Mw = Matrix(((xax.x, yax.x, zax.x, c[0]), (xax.y, yax.y, zax.y, c[1]), (xax.z, yax.z, zax.z, c[2]), (0, 0, 0, 1)))
        ob = util.mesh_object('Eye_%s' % tag, V, F, col, uvs=uv)
        ob.data.materials.append(mats['iris']); ob.data.materials.append(mats['sclera'])
        mi = np.ones(len(F), dtype=np.int32); mi[:n_iris_faces] = 0
        ob.data.polygons.foreach_set('material_index', mi)
        ob.matrix_world = Mw
        co = util.mesh_object('Eye_Cornea_%s' % tag, Vc, Fc, col)
        co.data.materials.append(mats['cornea'])
        co.matrix_world = Mw
        co.visible_shadow = False
        # caruncle: small glossy pink body at the inner canthus (rest frame -> posed)
        car_rest = EYE_C * np.array([side, 1, 1]) + np.array([-side * 0.0128, -0.0112, -0.0016])
        cw = head_matrix[:3, :3] @ car_rest + head_matrix[:3, 3]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0019, segments=16, ring_count=8, location=tuple(cw))
        car = bpy.context.object; car.name = 'Eye_Caruncle_%s' % tag; car.scale = (1.0, 0.8, 1.15)
        bpy.ops.object.shade_smooth(); util.link(car, col); car.data.materials.append(mats['caruncle'])
        objs += [ob, co, car]
    return objs


def util_smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------- hair-curve helpers

def curves_object(name, strands, radii, col, material=None):
    """strands: (n, k, 3) array; radii: (n, k). Creates a native Curves (hair) object."""
    strands = np.asarray(strands, dtype=np.float32); n, k, _ = strands.shape
    cu = bpy.data.hair_curves.new(name)
    cu.add_curves([k] * n)
    cu.position_data.foreach_set('vector', strands.reshape(-1))
    r = cu.attributes.get('radius') or cu.attributes.new('radius', 'FLOAT', 'POINT')
    r.data.foreach_set('value', np.asarray(radii, dtype=np.float32).reshape(-1))
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    if material is not None:
        cu.materials.append(material)
    return ob


def hair_material(name, color, roughness=0.3, radial=0.35, coat=0.0, melanin=None, redness=0.5, ior=1.55):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nt.nodes.remove(n)
    out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    h = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    if melanin is not None:
        h.parametrization = 'MELANIN'
        h.inputs['Melanin'].default_value = melanin
        h.inputs['Melanin Redness'].default_value = redness
        h.inputs['Tint'].default_value = (*color, 1)
    else:
        h.parametrization = 'COLOR'
        h.inputs['Color'].default_value = (*color, 1)
    h.inputs['Roughness'].default_value = roughness
    h.inputs['Radial Roughness'].default_value = radial
    h.inputs['Coat'].default_value = coat
    h.inputs['IOR'].default_value = ior
    nt.links.new(h.outputs['BSDF'], out.inputs['Surface'])
    m.diffuse_color = (*color, 1)
    return m, h


def lid_margin(body, side):
    """Ordered cage-vertex ids of the outer lid margin for one eye, split into upper and lower (inner -> outer)."""
    c = EYE_C0 * np.array([side, 1, 1])
    V0 = body.V0
    near = np.where(np.linalg.norm(V0 - c, axis=1) < 0.03)[0]
    enc = body.enclosed
    ids = [i for i in near if (not enc[i]) and any(enc[j] for j in body.adj[i]) and V0[i, 1] < -0.125]
    P = V0[ids]
    ang = np.degrees(np.arctan2(P[:, 2] - c[2], (P[:, 0] - c[0]) * side))     # 0 = outer corner, +-180 = inner
    up = sorted([(a, i) for a, i in zip(ang, ids) if a >= 0], key=lambda t: -t[0])
    lo = sorted([(a, i) for a, i in zip(ang, ids) if a < 0], key=lambda t: t[0])
    return [i for _, i in up], [i for _, i in lo]


def _resample(P, n):
    P = np.asarray(P); seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1), t / s[-1]


def build_lashes(body, surf_rest, head_matrix, col, mat, rng, quality=1.0):
    """surf_rest: limit-surface positions of the sculpted rest cage (first len(V) entries = cage verts)."""
    strands = []; radii = []
    R3 = head_matrix[:3, :3]; t3 = head_matrix[:3, 3]
    for side in (1.0, -1.0):
        c = EYE_C * np.array([side, 1, 1])
        up_ids, lo_ids = lid_margin(body, side)
        for ids, upper in ((up_ids, True), (lo_ids, False)):
            n = int((330 if upper else 40) * quality)
            pts, u = _resample(surf_rest[ids], 64)
            for k in range(n):
                uu = rng.uniform(0.04, 0.985) if upper else rng.uniform(0.12, 0.97)
                p = np.array([np.interp(uu, u, pts[:, j]) for j in range(3)])
                tang = np.array([np.interp(min(uu + 0.02, 1), u, pts[:, j]) for j in range(3)]) - p
                tang /= np.linalg.norm(tang) + 1e-12
                radial = p - c; radial /= np.linalg.norm(radial)
                zdir = np.array([0, 0, 1.0])
                env = math.sin(math.pi * min(max((uu - 0.02) / 0.96, 0), 1)) ** 0.6
                if upper:
                    L = (0.0024 + 0.0026 * env * (0.6 + 0.4 * uu)) * rng.uniform(0.80, 1.12)
                    fan = (uu - 0.42) * 0.95
                    d0 = radial * 0.86 + zdir * 0.30 + tang * fan
                    curl = zdir * 0.75 + radial * -0.12
                    root = p - radial * 0.0003 + zdir * -0.0002
                else:
                    L = (0.0012 + 0.0012 * env) * rng.uniform(0.75, 1.15)
                    fan = (uu - 0.45) * 0.8
                    d0 = radial * 0.72 - zdir * 0.62 + tang * fan
                    curl = radial * 0.30 - zdir * 0.05
                    root = p - radial * 0.0003 + zdir * 0.0002
                d0 = d0 + rng.normal(0, 0.07, 3); d0 /= np.linalg.norm(d0)
                ts = np.linspace(0, 1, 6)
                st = root[None] + L * (d0[None] * ts[:, None] + curl[None] * (ts[:, None] ** 2) * 0.55)
                strands.append(st @ R3.T + t3)
                r0 = 0.000085 if upper else 0.000050
                radii.append(r0 * (1 - ts * 0.80))
    return curves_object('Eyelashes', np.array(strands), np.array(radii), col, mat)


BROW_PTS = [  # (x, z) of the brow centre-line for the left side in the rest frame; half-thickness (m)
    (0.0100, 1.5622, 0.0026), (0.0180, 1.5628, 0.0029), (0.0270, 1.5630, 0.0028),
    (0.0360, 1.5624, 0.0025), (0.0460, 1.5610, 0.0018), (0.0555, 1.5584, 0.0009),
]


def build_brows(surf_bvh, head_matrix, col, mat, rng, quality=1.0):
    strands = []; radii = []
    R3 = head_matrix[:3, :3]; t3 = head_matrix[:3, 3]
    bp = np.array(BROW_PTS)
    seg = np.hypot(np.diff(bp[:, 0]), np.diff(bp[:, 1])); s = np.concatenate([[0], np.cumsum(seg)]); s /= s[-1]
    n = int(420 * quality)
    for side in (1.0, -1.0):
        for k in range(n):
            u = rng.beta(1.25, 1.25)
            v = rng.uniform(-1, 1)
            v = v * (abs(v) ** 0.25)
            x = np.interp(u, s, bp[:, 0]); z = np.interp(u, s, bp[:, 1]); hw = np.interp(u, s, bp[:, 2])
            tx = np.interp(min(u + 0.03, 1), s, bp[:, 0]) - np.interp(max(u - 0.03, 0), s, bp[:, 0])
            tz = np.interp(min(u + 0.03, 1), s, bp[:, 1]) - np.interp(max(u - 0.03, 0), s, bp[:, 1])
            tl = math.hypot(tx, tz); tx /= tl; tz /= tl
            px = x + (-tz) * v * hw; pz = z + tx * v * hw
            hit = surf_bvh.ray_cast(Vector((px * side, -0.30, pz)), Vector((0, 1, 0)), 0.3)
            if hit[0] is None:
                continue
            p = np.array(hit[0]); nrm = np.array(hit[1])
            if nrm[1] > 0:
                nrm = -nrm
            # flow: inner hairs stand up, body hairs sweep sideways, both rows converge to the centre line
            lift = (1 - util_smooth(u / 0.30)) * 1.25
            conv = -v * 0.32
            flow = np.array([tx * side, 0, tz]) + np.array([0, 0, 1.0]) * (lift + conv) + np.array([0, 0, -0.25]) * util_smooth((u - 0.6) / 0.4)
            flow = flow - nrm * np.dot(flow, nrm); flow /= np.linalg.norm(flow)
            L = rng.uniform(0.0036, 0.0062) * (1.0 - 0.35 * util_smooth((u - 0.7) / 0.3))
            ts = np.linspace(0, 1, 5)
            lift_n = 0.22 * (ts - ts ** 2 * 0.9)
            st = p[None] + L * (flow[None] * ts[:, None] + nrm[None] * lift_n[:, None]) + nrm[None] * 0.00006
            strands.append(st @ R3.T + t3)
            radii.append(0.000042 * (1 - ts * 0.70))
    return curves_object('Eyebrows', np.array(strands), np.array(radii), col, mat)
