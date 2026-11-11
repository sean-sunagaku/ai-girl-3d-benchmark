import math
import bpy


def srgb_to_lin(c):
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def rgba_srgb(r, g, b, a=1.0):
    return (srgb_to_lin(r), srgb_to_lin(g), srgb_to_lin(b), a)


def new_material(name):
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    mat = bpy.data.materials.new(name=name)
    if mat.node_tree is None:
        mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat


def _add_gaussian_3d_mask(nt, vec_socket, center, radii, loc=(0, 0), label="ZoneMask"):
    """Create exp(-( ((x-cx)/rx)^2 + ((y-cy)/ry)^2 + ((z-cz)/rz)^2 )) mask in shader nodes."""
    sub = nt.nodes.new('ShaderNodeVectorMath')
    sub.operation = 'SUBTRACT'
    sub.label = f"{label}_Sub"
    sub.location = (loc[0], loc[1])
    sub.inputs[1].default_value = center
    nt.links.new(vec_socket, sub.inputs[0])

    div = nt.nodes.new('ShaderNodeVectorMath')
    div.operation = 'DIVIDE'
    div.label = f"{label}_Scale"
    div.location = (loc[0] + 180, loc[1])
    div.inputs[1].default_value = radii
    nt.links.new(sub.outputs['Vector'], div.inputs[0])

    dot = nt.nodes.new('ShaderNodeVectorMath')
    dot.operation = 'DOT_PRODUCT'
    dot.label = f"{label}_R2"
    dot.location = (loc[0] + 360, loc[1])
    nt.links.new(div.outputs['Vector'], dot.inputs[0])
    nt.links.new(div.outputs['Vector'], dot.inputs[1])

    neg = nt.nodes.new('ShaderNodeMath')
    neg.operation = 'MULTIPLY'
    neg.inputs[1].default_value = -1.0
    neg.location = (loc[0] + 540, loc[1])
    nt.links.new(dot.outputs['Value'], neg.inputs[0])

    expn = nt.nodes.new('ShaderNodeMath')
    expn.operation = 'EXPONENT'
    expn.label = label
    expn.location = (loc[0] + 700, loc[1])
    nt.links.new(neg.outputs['Value'], expn.inputs[0])
    return expn.outputs['Value']


def _mix_rgb(nt, fac_sock, col1_sock_or_val, col2_sock_or_val, loc=(0, 0), label="MixColor"):
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MIX'
    mix.clamp_factor = True
    mix.label = label
    mix.location = loc
    if isinstance(fac_sock, (int, float)):
        mix.inputs['Factor'].default_value = fac_sock
    else:
        nt.links.new(fac_sock, mix.inputs['Factor'])

    if isinstance(col1_sock_or_val, tuple):
        mix.inputs['A'].default_value = col1_sock_or_val
    else:
        nt.links.new(col1_sock_or_val, mix.inputs['A'])

    if isinstance(col2_sock_or_val, tuple):
        mix.inputs['B'].default_value = col2_sock_or_val
    else:
        nt.links.new(col2_sock_or_val, mix.inputs['B'])
    return mix.outputs['Result']


def _mix_float(nt, fac_sock, v1, v2, loc=(0, 0), label="MixFloat"):
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'FLOAT'
    mix.clamp_factor = True
    mix.label = label
    mix.location = loc
    nt.links.new(fac_sock, mix.inputs['Factor'])
    if isinstance(v1, (int, float)):
        mix.inputs['A'].default_value = v1
    else:
        nt.links.new(v1, mix.inputs['A'])
    if isinstance(v2, (int, float)):
        mix.inputs['B'].default_value = v2
    else:
        nt.links.new(v2, mix.inputs['B'])
    return mix.outputs['Result']


def build_skin_material(textures):
    """
    Physically-based human portrait skin material using RANDOM_WALK_SKIN subsurface scattering,
    seamless 3D canonical rest-pose facial zones (blush, coral-pink hydrated lips, nose/ear warmth,
    nostril AO, scalp shadow), and UV micro-pore/roughness detail.
    """
    mat = new_material("Mat_Character_Skin")
    nt = mat.node_tree

    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (1800, 0)

    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (1420, 0)
    bsdf.subsurface_method = 'RANDOM_WALK_SKIN'
    bsdf.inputs['Subsurface Weight'].default_value = 0.34
    bsdf.inputs['Subsurface Scale'].default_value = 0.0060
    bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.38, 0.22)
    bsdf.inputs['Subsurface IOR'].default_value = 1.40
    bsdf.inputs['Subsurface Anisotropy'].default_value = 0.35
    bsdf.inputs['IOR'].default_value = 1.43
    bsdf.inputs['Specular IOR Level'].default_value = 0.44
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    # Read symmetric rest_co = (|rx|, ry, rz) for zero-seam bilateral facial color zones
    attr = nt.nodes.new('ShaderNodeAttribute')
    attr.attribute_name = "rest_co"
    attr.location = (-1400, 200)

    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    sep.location = (-1200, 200)
    nt.links.new(attr.outputs['Vector'], sep.inputs['Vector'])

    abs_x = nt.nodes.new('ShaderNodeMath')
    abs_x.operation = 'ABSOLUTE'
    abs_x.location = (-1020, 300)
    nt.links.new(sep.outputs['X'], abs_x.inputs[0])

    sym_co = nt.nodes.new('ShaderNodeCombineXYZ')
    sym_co.location = (-840, 200)
    nt.links.new(abs_x.outputs['Value'], sym_co.inputs['X'])
    nt.links.new(sep.outputs['Y'], sym_co.inputs['Y'])
    nt.links.new(sep.outputs['Z'], sym_co.inputs['Z'])

    # Read smooth vertex color/zone attributes painted accurately on the sculpted topology
    attr_lip = nt.nodes.new('ShaderNodeAttribute')
    attr_lip.attribute_name = "lip_mask"
    attr_lip.location = (-600, 520)

    attr_blush = nt.nodes.new('ShaderNodeAttribute')
    attr_blush.attribute_name = "blush_mask"
    attr_blush.location = (-600, 340)

    attr_ao = nt.nodes.new('ShaderNodeAttribute')
    attr_ao.attribute_name = "cavity_ao"
    attr_ao.location = (-600, 160)

    attr_scalp = nt.nodes.new('ShaderNodeAttribute')
    attr_scalp.attribute_name = "scalp_mask"
    attr_scalp.location = (-600, -20)

    attr_lash = nt.nodes.new('ShaderNodeAttribute')
    attr_lash.attribute_name = "lashline_mask"
    attr_lash.location = (-600, -140)

    attr_brow = nt.nodes.new('ShaderNodeAttribute')
    attr_brow.attribute_name = "brow_mask"
    attr_brow.location = (-600, -260)

    # Subtle 3D organic mottling so skin has living warmth variation
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 95.0
    noise.inputs['Detail'].default_value = 5.0
    noise.inputs['Roughness'].default_value = 0.55
    noise.location = (-600, -400)
    nt.links.new(attr.outputs['Vector'], noise.inputs['Vector'])

    # Realistic fair-warm human skin reflectance (illuminates to luminous ivory-peach under cafe daylight
    # while preserving soft, natural sculptural shading on nose bridge, eyelids, cheeks, and jaw!)
    col_skin_a = rgba_srgb(0.835, 0.668, 0.598)
    col_skin_b = rgba_srgb(0.805, 0.632, 0.562)
    col_blush = rgba_srgb(0.825, 0.505, 0.478)
    col_lip = rgba_srgb(0.755, 0.375, 0.378)
    col_cavity = rgba_srgb(0.200, 0.095, 0.085)
    col_scalp = rgba_srgb(0.195, 0.120, 0.088)
    col_lashline = rgba_srgb(0.135, 0.078, 0.062)
    col_brow_tint = rgba_srgb(0.420, 0.305, 0.250)

    c0 = _mix_rgb(nt, noise.outputs['Fac'], col_skin_b, col_skin_a, loc=(-260, 420), label="SkinMottle")
    c1 = _mix_rgb(nt, attr_blush.outputs['Fac'], c0, col_blush, loc=(0, 420), label="CheekEarNoseWarmth")

    # Subtle vertical lip micro-lines on lip vermilion
    lip_wave = nt.nodes.new('ShaderNodeTexNoise')
    lip_wave.inputs['Scale'].default_value = 420.0
    lip_wave.inputs['Detail'].default_value = 3.0
    lip_wave.location = (-260, 160)
    lip_scale = nt.nodes.new('ShaderNodeVectorMath')
    lip_scale.operation = 'MULTIPLY'
    lip_scale.inputs[1].default_value = (2.8, 0.5, 0.35)
    lip_scale.location = (-460, 160)
    nt.links.new(attr.outputs['Vector'], lip_scale.inputs[0])
    nt.links.new(lip_scale.outputs['Vector'], lip_wave.inputs['Vector'])

    col_lip_deep = rgba_srgb(0.685, 0.315, 0.325)
    lip_col = _mix_rgb(nt, lip_wave.outputs['Fac'], col_lip_deep, col_lip, loc=(0, 180), label="LipDetailColor")

    c2 = _mix_rgb(nt, attr_lip.outputs['Fac'], c1, lip_col, loc=(260, 360), label="LipVermilion")
    c2b = _mix_rgb(nt, attr_brow.outputs['Fac'], c2, col_brow_tint, loc=(420, 360), label="SoftBrowPowder")
    c3 = _mix_rgb(nt, attr_scalp.outputs['Fac'], c2b, col_scalp, loc=(580, 360), label="ScalpShadow")
    c3b = _mix_rgb(nt, attr_lash.outputs['Fac'], c3, col_lashline, loc=(740, 360), label="UpperLashTightline")
    c4 = _mix_rgb(nt, attr_ao.outputs['Fac'], c3b, col_cavity, loc=(900, 360), label="NostrilMouthAO")
    nt.links.new(c4, bsdf.inputs['Base Color'])

    # UV skin micro-pore detail
    uv_node = nt.nodes.new('ShaderNodeTexCoord')
    uv_node.location = (-900, -520)
    uv_map = nt.nodes.new('ShaderNodeMapping')
    uv_map.inputs['Scale'].default_value = (14.0, 14.0, 14.0)
    uv_map.location = (-700, -520)
    nt.links.new(uv_node.outputs['UV'], uv_map.inputs['Vector'])

    tex_pore = nt.nodes.new('ShaderNodeTexImage')
    tex_pore.image = textures["skin_detail"]
    tex_pore.location = (-460, -520)
    nt.links.new(uv_map.outputs['Vector'], tex_pore.inputs['Vector'])

    sep_pore = nt.nodes.new('ShaderNodeSeparateColor')
    sep_pore.location = (-160, -520)
    nt.links.new(tex_pore.outputs['Color'], sep_pore.inputs['Color'])

    # Roughness: ~0.37 on skin (modulated +-0.05 by micro-pores), ~0.21 hydrated sheen on lips
    skin_rough = nt.nodes.new('ShaderNodeMapRange')
    skin_rough.inputs['From Min'].default_value = 0.0
    skin_rough.inputs['From Max'].default_value = 1.0
    skin_rough.inputs['To Min'].default_value = 0.31
    skin_rough.inputs['To Max'].default_value = 0.43
    skin_rough.location = (120, -420)
    nt.links.new(sep_pore.outputs['Green'], skin_rough.inputs['Value'])

    final_rough = _mix_float(nt, attr_lip.outputs['Fac'], skin_rough.outputs['Result'], 0.21, loc=(440, -360), label="LipRoughness")
    nt.links.new(final_rough, bsdf.inputs['Roughness'])

    # Hydrated lip clearcoat + subtle T-zone/cheek specular sheen
    coat_w = _mix_float(nt, attr_lip.outputs['Fac'], 0.06, 0.32, loc=(440, -160), label="LipCoat")
    nt.links.new(coat_w, bsdf.inputs['Coat Weight'])
    bsdf.inputs['Coat Roughness'].default_value = 0.14

    # Restrained micro-pore + lip vertical micro-groove bump
    lip_groove_h = _mix_float(nt, attr_lip.outputs['Fac'], sep_pore.outputs['Red'], lip_wave.outputs['Fac'], loc=(440, -580), label="PoreOrLipBump")
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.14
    bump.inputs['Distance'].default_value = 0.00018
    bump.location = (960, -440)
    nt.links.new(lip_groove_h, bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    return mat


def build_eye_materials(textures):
    """
    Dual-layer anatomical eye materials:
    1) Mat_Eye_Outer: Sclera with wet tear coat + crystal-clear refractive Cornea dome
       using Cycles Light-Path shadow-ray transparency so natural light illuminates the iris!
    2) Mat_Eye_Iris: Rich chestnut-amber radial iris stroma with pupil and limbal ring.
    3) Mat_Tear_Meniscus: Glossy wet meniscus line along lower eyelid margin.
    """
    # --- 1. Outer Sclera + Cornea ---
    mat_outer = new_material("Mat_Eye_Outer")
    nt = mat_outer.node_tree
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (1300, 0)

    attr_cornea = nt.nodes.new('ShaderNodeAttribute')
    attr_cornea.attribute_name = "cornea_mask"
    attr_cornea.location = (-500, 300)

    # Sclera BSDF (off-white warm vascularized sclera with wet tear film)
    sclera_bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    sclera_bsdf.location = (300, 260)
    sclera_bsdf.subsurface_method = 'RANDOM_WALK'
    sclera_bsdf.inputs['Base Color'].default_value = rgba_srgb(0.925, 0.895, 0.880)
    sclera_bsdf.inputs['Subsurface Weight'].default_value = 0.45
    sclera_bsdf.inputs['Subsurface Scale'].default_value = 0.006
    sclera_bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.25)
    sclera_bsdf.inputs['Roughness'].default_value = 0.05
    sclera_bsdf.inputs['Coat Weight'].default_value = 0.85
    sclera_bsdf.inputs['Coat Roughness'].default_value = 0.015
    sclera_bsdf.inputs['IOR'].default_value = 1.376

    # Cornea BSDF (refractive dome + shadow-ray pass-through for luminous iris!)
    cornea_bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    cornea_bsdf.location = (100, -180)
    cornea_bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    cornea_bsdf.inputs['Transmission Weight'].default_value = 1.0
    cornea_bsdf.inputs['Roughness'].default_value = 0.006
    cornea_bsdf.inputs['IOR'].default_value = 1.376
    cornea_bsdf.inputs['Specular IOR Level'].default_value = 0.55

    transp = nt.nodes.new('ShaderNodeBsdfTransparent')
    transp.location = (100, -440)

    lp = nt.nodes.new('ShaderNodeLightPath')
    lp.location = (-260, -420)
    max_ray = nt.nodes.new('ShaderNodeMath')
    max_ray.operation = 'MAXIMUM'
    max_ray.location = (-40, -420)
    nt.links.new(lp.outputs['Is Shadow Ray'], max_ray.inputs[0])
    nt.links.new(lp.outputs['Is Diffuse Ray'], max_ray.inputs[1])

    cornea_mix = nt.nodes.new('ShaderNodeMixShader')
    cornea_mix.location = (480, -220)
    nt.links.new(max_ray.outputs['Value'], cornea_mix.inputs['Fac'])
    nt.links.new(cornea_bsdf.outputs['BSDF'], cornea_mix.inputs[1])
    nt.links.new(transp.outputs['BSDF'], cornea_mix.inputs[2])

    final_mix = nt.nodes.new('ShaderNodeMixShader')
    final_mix.location = (920, 40)
    nt.links.new(attr_cornea.outputs['Fac'], final_mix.inputs['Fac'])
    nt.links.new(sclera_bsdf.outputs['BSDF'], final_mix.inputs[1])
    nt.links.new(cornea_mix.outputs['Shader'], final_mix.inputs[2])
    nt.links.new(final_mix.outputs['Shader'], out.inputs['Surface'])

    # --- 2. Inner Concave Iris + Pupil ---
    mat_iris = new_material("Mat_Eye_Iris")
    nt2 = mat_iris.node_tree
    out2 = nt2.nodes.new('ShaderNodeOutputMaterial')
    out2.location = (900, 0)

    iris_bsdf = nt2.nodes.new('ShaderNodeBsdfPrincipled')
    iris_bsdf.location = (520, 0)
    iris_bsdf.inputs['Roughness'].default_value = 0.52
    iris_bsdf.inputs['Specular IOR Level'].default_value = 0.22
    iris_bsdf.inputs['Subsurface Weight'].default_value = 0.14
    iris_bsdf.inputs['Subsurface Scale'].default_value = 0.003
    iris_bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.45, 0.20)
    nt2.links.new(iris_bsdf.outputs['BSDF'], out2.inputs['Surface'])

    uv2 = nt2.nodes.new('ShaderNodeTexCoord')
    uv2.location = (-360, 0)

    tex_ia = nt2.nodes.new('ShaderNodeTexImage')
    tex_ia.image = textures["iris_albedo"]
    tex_ia.extension = 'EXTEND'
    tex_ia.location = (-60, 140)
    nt2.links.new(uv2.outputs['UV'], tex_ia.inputs['Vector'])
    nt2.links.new(tex_ia.outputs['Color'], iris_bsdf.inputs['Base Color'])

    tex_ib = nt2.nodes.new('ShaderNodeTexImage')
    tex_ib.image = textures["iris_bump"]
    tex_ib.extension = 'EXTEND'
    tex_ib.location = (-60, -180)
    nt2.links.new(uv2.outputs['UV'], tex_ib.inputs['Vector'])

    ibump = nt2.nodes.new('ShaderNodeBump')
    ibump.inputs['Strength'].default_value = 0.35
    ibump.inputs['Distance'].default_value = 0.0004
    ibump.location = (240, -180)
    nt2.links.new(tex_ib.outputs['Color'], ibump.inputs['Height'])
    nt2.links.new(ibump.outputs['Normal'], iris_bsdf.inputs['Normal'])

    # --- 3. Wet Tear Meniscus ---
    mat_tear = new_material("Mat_Tear_Meniscus")
    nt3 = mat_tear.node_tree
    out3 = nt3.nodes.new('ShaderNodeOutputMaterial')
    out3.location = (640, 0)
    tear_bsdf = nt3.nodes.new('ShaderNodeBsdfPrincipled')
    tear_bsdf.location = (80, 80)
    tear_bsdf.inputs['Transmission Weight'].default_value = 1.0
    tear_bsdf.inputs['Roughness'].default_value = 0.01
    tear_bsdf.inputs['IOR'].default_value = 1.336
    tr3 = nt3.nodes.new('ShaderNodeBsdfTransparent')
    tr3.location = (80, -180)
    lp3 = nt3.nodes.new('ShaderNodeLightPath')
    lp3.location = (-180, -180)
    mx3 = nt3.nodes.new('ShaderNodeMixShader')
    mx3.location = (380, 0)
    nt3.links.new(lp3.outputs['Is Shadow Ray'], mx3.inputs['Fac'])
    nt3.links.new(tear_bsdf.outputs['BSDF'], mx3.inputs[1])
    nt3.links.new(tr3.outputs['BSDF'], mx3.inputs[2])
    nt3.links.new(mx3.outputs['Shader'], out3.inputs['Surface'])

    return mat_outer, mat_iris, mat_tear


def build_hair_materials():
    """
    Physically-based Principled Hair BSDF materials:
    - Mat_Hair_Chestnut: Warm medium-dark chestnut brown with golden-bronze backlit sheen
    - Mat_Hair_Volume_Core: Soft dark-chestnut inner underlayer so crown never shows bald gaps
    - Mat_Eyebrow: Soft natural ash-chestnut brow hair
    - Mat_Eyelash: Rich dark espresso-black lash hair
    """
    # 1. Main Chestnut Hair Strands
    mat_hair = new_material("Mat_Hair_Chestnut")
    nt = mat_hair.node_tree
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (600, 0)

    hair = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    hair.location = (220, 0)
    hair.model = 'CHIANG'
    hair.parametrization = 'MELANIN'
    # Melanin 0.81 + Redness 0.45 produces rich dark chestnut brown with warm bronze backlit glow!
    hair.inputs['Melanin'].default_value = 0.81
    hair.inputs['Melanin Redness'].default_value = 0.45
    hair.inputs['Tint'].default_value = rgba_srgb(0.93, 0.84, 0.78)
    hair.inputs['Roughness'].default_value = 0.24
    hair.inputs['Radial Roughness'].default_value = 0.34
    hair.inputs['Coat'].default_value = 0.28
    hair.inputs['IOR'].default_value = 1.55
    hair.inputs['Offset'].default_value = math.radians(2.0)
    hair.inputs['Random Color'].default_value = 0.08
    hair.inputs['Random Roughness'].default_value = 0.06
    nt.links.new(hair.outputs['BSDF'], out.inputs['Surface'])

    # 2. Inner Hair Volume Core (sits recessed under strands at back/crown)
    mat_core = new_material("Mat_Hair_Volume_Core")
    ntc = mat_core.node_tree
    outc = ntc.nodes.new('ShaderNodeOutputMaterial')
    outc.location = (520, 0)
    bsdfc = ntc.nodes.new('ShaderNodeBsdfPrincipled')
    bsdfc.location = (180, 0)
    bsdfc.inputs['Base Color'].default_value = rgba_srgb(0.115, 0.068, 0.048)
    bsdfc.inputs['Roughness'].default_value = 0.44
    bsdfc.inputs['Anisotropic'].default_value = 0.68
    bsdfc.inputs['Specular IOR Level'].default_value = 0.28
    ntc.links.new(bsdfc.outputs['BSDF'], outc.inputs['Surface'])

    # 3. Eyebrows (soft ash-chestnut brown, natural & feathery)
    mat_brow = new_material("Mat_Eyebrow")
    ntb = mat_brow.node_tree
    outb = ntb.nodes.new('ShaderNodeOutputMaterial')
    outb.location = (520, 0)
    hb = ntb.nodes.new('ShaderNodeBsdfHairPrincipled')
    hb.location = (180, 0)
    hb.model = 'CHIANG'
    hb.parametrization = 'MELANIN'
    hb.inputs['Melanin'].default_value = 0.56
    hb.inputs['Melanin Redness'].default_value = 0.35
    hb.inputs['Roughness'].default_value = 0.44
    hb.inputs['Radial Roughness'].default_value = 0.50
    hb.inputs['Random Color'].default_value = 0.06
    ntb.links.new(hb.outputs['BSDF'], outb.inputs['Surface'])

    # 4. Eyelashes (deep espresso dark brown-black)
    mat_lash = new_material("Mat_Eyelash")
    ntl = mat_lash.node_tree
    outl = ntl.nodes.new('ShaderNodeOutputMaterial')
    outl.location = (520, 0)
    hl = ntl.nodes.new('ShaderNodeBsdfHairPrincipled')
    hl.location = (180, 0)
    hl.model = 'CHIANG'
    hl.parametrization = 'MELANIN'
    hl.inputs['Melanin'].default_value = 0.88
    hl.inputs['Melanin Redness'].default_value = 0.35
    hl.inputs['Roughness'].default_value = 0.26
    hl.inputs['Radial Roughness'].default_value = 0.32
    ntl.links.new(hl.outputs['BSDF'], outl.inputs['Surface'])

    return mat_hair, mat_core, mat_brow, mat_lash


def build_sweater_materials(textures, uv_scale=(3.8, 3.2, 1.0), name="Mat_Sweater_Pink_Knit", use_box_projection=False):
    """
    Dusty pink knitted sweater material with Principled BSDF v2 microfiber sheen,
    warm subsurface yarn scattering, and seamless 3D BOX projection (for torso) or UV normal map.
    """
    mat = new_material(name)
    nt = mat.node_tree
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (1100, 0)

    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (720, 0)
    bsdf.inputs['Roughness'].default_value = 0.78
    bsdf.inputs['Diffuse Roughness'].default_value = 0.45
    bsdf.inputs['Specular IOR Level'].default_value = 0.22
    bsdf.inputs['Subsurface Weight'].default_value = 0.16
    bsdf.inputs['Subsurface Scale'].default_value = 0.005
    bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.55, 0.55)
    bsdf.inputs['Sheen Weight'].default_value = 0.88
    bsdf.inputs['Sheen Roughness'].default_value = 0.48
    bsdf.inputs['Sheen Tint'].default_value = rgba_srgb(0.96, 0.80, 0.79)
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    uv = nt.nodes.new('ShaderNodeTexCoord')
    uv.location = (-660, 0)
    mapping = nt.nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = uv_scale
    mapping.location = (-460, 0)

    if use_box_projection:
        nt.links.new(uv.outputs['Object'], mapping.inputs['Vector'])

        tex_a = nt.nodes.new('ShaderNodeTexImage')
        tex_a.image = textures["knit_albedo"]
        tex_a.projection = 'BOX'
        tex_a.projection_blend = 0.26
        tex_a.location = (-180, 220)
        nt.links.new(mapping.outputs['Vector'], tex_a.inputs['Vector'])
        nt.links.new(tex_a.outputs['Color'], bsdf.inputs['Base Color'])

        tex_h = nt.nodes.new('ShaderNodeTexImage')
        tex_h.image = textures["knit_height"]
        tex_h.projection = 'BOX'
        tex_h.projection_blend = 0.26
        tex_h.location = (-180, -140)
        nt.links.new(mapping.outputs['Vector'], tex_h.inputs['Vector'])

        sep_h = nt.nodes.new('ShaderNodeSeparateColor')
        sep_h.location = (110, -140)
        nt.links.new(tex_h.outputs['Color'], sep_h.inputs['Color'])

        bump = nt.nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = 1.05
        bump.inputs['Distance'].default_value = 0.0038
        bump.location = (340, -140)
        nt.links.new(sep_h.outputs['Red'], bump.inputs['Height'])
        nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    else:
        nt.links.new(uv.outputs['UV'], mapping.inputs['Vector'])

        tex_a = nt.nodes.new('ShaderNodeTexImage')
        tex_a.image = textures["knit_albedo"]
        tex_a.location = (-180, 220)
        nt.links.new(mapping.outputs['Vector'], tex_a.inputs['Vector'])
        nt.links.new(tex_a.outputs['Color'], bsdf.inputs['Base Color'])

        tex_n = nt.nodes.new('ShaderNodeTexImage')
        tex_n.image = textures["knit_normal"]
        tex_n.location = (-180, -140)
        nt.links.new(mapping.outputs['Vector'], tex_n.inputs['Vector'])

        nmap = nt.nodes.new('ShaderNodeNormalMap')
        nmap.inputs['Strength'].default_value = 1.15
        nmap.location = (220, -140)
        nt.links.new(tex_n.outputs['Color'], nmap.inputs['Color'])
        nt.links.new(nmap.outputs['Normal'], bsdf.inputs['Normal'])

    return mat


def build_sweater_fuzz_material():
    """Translucent pastel-pink wool halo fuzz curves for sunlit knitwear rim realism."""
    mat = new_material("Mat_Sweater_Fuzz")
    nt = mat.node_tree
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (520, 0)
    hair = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    hair.location = (160, 0)
    hair.model = 'CHIANG'
    hair.parametrization = 'COLOR'
    hair.inputs['Color'].default_value = rgba_srgb(0.92, 0.72, 0.71)
    hair.inputs['Roughness'].default_value = 0.52
    hair.inputs['Radial Roughness'].default_value = 0.65
    nt.links.new(hair.outputs['BSDF'], out.inputs['Surface'])
    return mat
