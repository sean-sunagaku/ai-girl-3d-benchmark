"""Generated knit tiles (stockinette and 1x1 rib) and the knit materials. All procedural, authored here."""
import math
import numpy as np
import bpy
from . import util

TILE_ST = 4          # stitches per tile in U and rows per tile in V


def _capsule(u, v, a, b, r):
    """Rounded height profile of a yarn segment a->b (in stitch units) with radius r."""
    ab = (b[0] - a[0], b[1] - a[1])
    t = np.clip(((u - a[0]) * ab[0] + (v - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2), 0, 1)
    du = u - (a[0] + t * ab[0]); dv = v - (a[1] + t * ab[1])
    d2 = (du * du + dv * dv) / (r * r)
    return np.sqrt(np.clip(1 - d2, 0, 1)), t


def stockinette_tile(size=512, aspect=0.76):
    """Height (0..1) and yarn-twist phase for a TILE_ST x TILE_ST patch of V stitches. aspect = row height / stitch width."""
    y, x = np.mgrid[0:size, 0:size]
    U = (x + 0.5) / size * TILE_ST; V = (y + 0.5) / size * TILE_ST
    fu = U - np.floor(U); fv = V - np.floor(V)
    H = np.zeros((size, size)); TW = np.zeros((size, size))
    # each stitch: two legs forming a V; legs of the row above/below overlap so rows interlock
    for dv in (-1, 0, 1):
        vv = (fv - dv) * aspect
        for a, b in (((0.08, -0.10 * aspect), (0.44, 1.18 * aspect)), ((0.92, -0.10 * aspect), (0.56, 1.18 * aspect))):
            for du in (-1, 0, 1):
                h, t = _capsule(fu - du, vv, a, b, 0.215)
                # the upper end of a leg tucks under the next row: taper its height
                h = h * (0.55 + 0.45 * np.clip(1.35 - t * 1.1, 0, 1))
                upd = h > H
                H = np.where(upd, h, H); TW = np.where(upd, t + (0.5 if a[0] > 0.5 else 0.0) + du * 0.37 + dv * 0.21, TW)
    return H, TW


def rib_tile(size=512, aspect=0.76):
    """1x1 rib: knit column (raised V stitches) next to a recessed purl column. Period 2 stitches."""
    H0, TW = stockinette_tile(size, aspect)
    y, x = np.mgrid[0:size, 0:size]
    U = (x + 0.5) / size * TILE_ST; V = (y + 0.5) / size * TILE_ST
    col = np.floor(U) % 2
    fu = U - np.floor(U); fv = V - np.floor(V)
    ridge = np.sin(np.pi * fu) ** 0.7
    purl = 0.30 * np.sqrt(np.clip(1 - ((fv - 0.5) / 0.42) ** 2, 0, 1)) * np.sin(np.pi * fu) ** 0.5
    H = np.where(col == 0, 0.38 + 0.62 * H0 * ridge, 0.04 + purl)
    return H, TW


def knit_images(size=512):
    out = {}
    for name, fn in (('Knit_Stockinette', stockinette_tile), ('Knit_Rib', rib_tile)):
        H, TW = fn(size)
        twist = 0.5 + 0.5 * np.sin(TW * 2 * math.pi * 5.0)
        Hc = np.clip(H * (0.93 + 0.07 * twist), 0, 1)
        ao = np.clip(0.06 + 0.94 * H ** 0.85, 0, 1)
        data = np.stack([Hc, ao, twist], 2).astype(np.float32)
        out[name] = util.new_image(name + '_Height_AO_Twist', data, is_data=True)
    return out


def knit_material(name, image, base=(0.80, 0.350, 0.345), deep=(0.27, 0.060, 0.075), bump=1.0):
    """Sweater yarn. UVs are in stitch units; the tile spans TILE_ST stitches."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; p = nt.nodes['Principled BSDF']
    uv = nt.nodes.new('ShaderNodeUVMap'); uv.uv_map = 'UVMap'
    mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1.0 / TILE_ST, 1.0 / TILE_ST, 1.0)
    nt.links.new(uv.outputs['UV'], mp.inputs['Vector'])
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = image; tex.interpolation = 'Cubic'
    nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
    sep = nt.nodes.new('ShaderNodeSeparateColor'); nt.links.new(tex.outputs['Color'], sep.inputs['Color'])
    # per-stitch / per-row yarn variation (irregular hand-knit look)
    fl = nt.nodes.new('ShaderNodeVectorMath'); fl.operation = 'FLOOR'; nt.links.new(uv.outputs['UV'], fl.inputs[0])
    wn = nt.nodes.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '2D'; nt.links.new(fl.outputs['Vector'], wn.inputs['Vector'])
    geo = nt.nodes.new('ShaderNodeTexCoord')
    big = nt.nodes.new('ShaderNodeTexNoise'); big.inputs['Scale'].default_value = 9.0; big.inputs['Detail'].default_value = 3.0
    nt.links.new(geo.outputs['Object'], big.inputs['Vector'])
    fine = nt.nodes.new('ShaderNodeTexNoise'); fine.inputs['Scale'].default_value = 900.0; fine.inputs['Detail'].default_value = 2.0
    nt.links.new(geo.outputs['Object'], fine.inputs['Vector'])
    # colour = mix(deep, base, ao) * variation
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    mix.inputs['A'].default_value = (*deep, 1); mix.inputs['B'].default_value = (*base, 1)
    nt.links.new(sep.outputs['Green'], mix.inputs['Factor'])
    var = nt.nodes.new('ShaderNodeMath'); var.operation = 'MULTIPLY_ADD'; var.inputs[1].default_value = 0.14; var.inputs[2].default_value = 0.86
    nt.links.new(wn.outputs['Value'], var.inputs[0])
    var2 = nt.nodes.new('ShaderNodeMath'); var2.operation = 'MULTIPLY_ADD'; var2.inputs[1].default_value = 0.20; var2.inputs[2].default_value = 0.90
    nt.links.new(big.outputs['Fac'], var2.inputs[0])
    var3 = nt.nodes.new('ShaderNodeMath'); var3.operation = 'MULTIPLY_ADD'; var3.inputs[1].default_value = 0.16; var3.inputs[2].default_value = 0.92
    nt.links.new(sep.outputs['Blue'], var3.inputs[0])
    m1 = nt.nodes.new('ShaderNodeMath'); m1.operation = 'MULTIPLY'; nt.links.new(var.outputs[0], m1.inputs[0]); nt.links.new(var2.outputs[0], m1.inputs[1])
    m2 = nt.nodes.new('ShaderNodeMath'); m2.operation = 'MULTIPLY'; nt.links.new(m1.outputs[0], m2.inputs[0]); nt.links.new(var3.outputs[0], m2.inputs[1])
    colm = nt.nodes.new('ShaderNodeMix'); colm.data_type = 'RGBA'; colm.blend_type = 'MULTIPLY'; colm.inputs['Factor'].default_value = 1.0
    nt.links.new(mix.outputs['Result'], colm.inputs['A']); nt.links.new(m2.outputs[0], colm.inputs['B'])
    nt.links.new(colm.outputs['Result'], p.inputs['Base Color'])
    p.inputs['Roughness'].default_value = 0.92
    p.inputs['Sheen Weight'].default_value = 0.40
    p.inputs['Sheen Roughness'].default_value = 0.55
    p.inputs['Sheen Tint'].default_value = (1.0, 0.78, 0.76, 1)
    p.inputs['Specular IOR Level'].default_value = 0.15
    p.inputs['Subsurface Weight'].default_value = 0.12
    p.inputs['Subsurface Radius'].default_value = (1.0, 0.5, 0.5)
    p.inputs['Subsurface Scale'].default_value = 0.004
    # relief: stitch height + fine fibre noise
    hsum = nt.nodes.new('ShaderNodeMath'); hsum.operation = 'MULTIPLY_ADD'; hsum.inputs[1].default_value = 0.06
    nt.links.new(fine.outputs['Fac'], hsum.inputs[0]); nt.links.new(sep.outputs['Red'], hsum.inputs[2])
    bn = nt.nodes.new('ShaderNodeBump'); bn.inputs['Strength'].default_value = bump; bn.inputs['Distance'].default_value = 0.0048
    nt.links.new(hsum.outputs[0], bn.inputs['Height'])
    nt.links.new(bn.outputs['Normal'], p.inputs['Normal'])
    m.diffuse_color = (*base, 1)
    return m


def fuzz_material(name='Knit_Fuzz', color=(0.84, 0.46, 0.45)):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nt.nodes.remove(n)
    out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    h = nt.nodes.new('ShaderNodeBsdfHairPrincipled'); h.parametrization = 'COLOR'
    h.inputs['Color'].default_value = (*color, 1); h.inputs['Roughness'].default_value = 0.65; h.inputs['Radial Roughness'].default_value = 0.75
    nt.links.new(h.outputs['BSDF'], out.inputs['Surface'])
    m.diffuse_color = (*color, 1)
    return m
