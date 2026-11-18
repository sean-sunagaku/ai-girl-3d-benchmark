"""Physically-based Cycles material node graphs and seamless 3D anatomical skin color attributes."""

import math
import bpy
import numpy as np


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gauss3(x, y, z, cx, cy, cz, sx, sy, sz):
    return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2 + ((z - cz) / sz) ** 2))


def populate_skin_color_attributes(body_obj: bpy.types.Object):
    """
    Compute seamless 3D rest-space anatomical skin coloration (fair warm ivory base, malar blush,
    soft eyelid & lash-root shading, nostril interior darkening, ear warmth, coral-rose lips,
    and specular/roughness masks) directly on Character_Body's point attributes.
    """
    mesh = body_obj.data
    n = len(mesh.vertices)

    rest_flat = np.zeros(n * 3, dtype=np.float64)
    mesh.attributes["rest_pos"].data.foreach_get("vector", rest_flat)
    P = rest_flat.reshape((n, 3))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax = np.abs(x)

    lip_mask = np.zeros(n, dtype=np.float64)
    lip_line = np.zeros(n, dtype=np.float64)
    scalp_mask = np.zeros(n, dtype=np.float64)
    mesh.attributes["lip_mask"].data.foreach_get("value", lip_mask)
    mesh.attributes["lip_line_mask"].data.foreach_get("value", lip_line)
    mesh.attributes["scalp_mask"].data.foreach_get("value", scalp_mask)

    # Physically calibrated fair, luminous warm East Asian skin tone (sRGB ~0.78 -> linear ~0.57)
    base_srgb = np.zeros((n, 3), dtype=np.float64)
    base_srgb[:, 0] = 0.788
    base_srgb[:, 1] = 0.652
    base_srgb[:, 2] = 0.592

    # Subtle natural skin organic micro-variation
    mottle = (
        0.5
        + 0.25 * np.sin(x * 190.0 + z * 210.0) * np.cos(y * 175.0 - x * 140.0)
        + 0.25 * np.sin(x * 410.0 - y * 380.0 + z * 350.0)
    )
    base_srgb[:, 0] += 0.008 * (mottle - 0.5)
    base_srgb[:, 1] -= 0.006 * (mottle - 0.5)
    base_srgb[:, 2] -= 0.008 * (mottle - 0.5)

    # 1. Soft rosy peach cheek blush on lifted smiling apple cheeks
    cheek_blush = _gauss3(ax, y, z, 0.038, -0.136, 1.519, 0.022, 0.024, 0.017)
    blush_srgb = np.array([0.780, 0.515, 0.495])
    w_blush = (0.38 * cheek_blush)[:, None]
    base_srgb = base_srgb * (1.0 - w_blush) + blush_srgb[None, :] * w_blush

    # 2. Delicate warmth on nasal tip, alar wings, chin pad, and ear helix
    nose_warm = _gauss3(ax, y, z, 0.0, -0.163, 1.513, 0.015, 0.015, 0.010)
    ear_warm = _gauss3(ax, y, z, 0.076, -0.038, 1.540, 0.016, 0.024, 0.028)
    chin_warm = _gauss3(ax, y, z, 0.0, -0.151, 1.462, 0.018, 0.018, 0.012)
    warm_zone = np.clip(0.24 * nose_warm + 0.30 * ear_warm + 0.14 * chin_warm, 0.0, 0.45)[:, None]
    warm_col = np.array([0.775, 0.545, 0.515])
    base_srgb = base_srgb * (1.0 - warm_zone) + warm_col[None, :] * warm_zone

    # 3. Soft warm peach-taupe upper eyelid eyeshadow & subtle tightline matching widened eyes!
    upper_lid_shade = (
        np.exp(-(((ax - 0.0315) / 0.0145) ** 2))
        * _smoothstep(-0.133, -0.139, y)
        * np.exp(-(((z - 1.5535) / 0.0038) ** 2))
    )
    lid_col = np.array([0.655, 0.495, 0.440])
    base_srgb = base_srgb * (1.0 - 0.44 * upper_lid_shade[:, None]) + lid_col[None, :] * (
        0.44 * upper_lid_shade[:, None]
    )

    rim_xs = np.array([0.0195, 0.0250, 0.0310, 0.0370, 0.0435])
    rim_up_zs = np.array([1.5459, 1.5492, 1.5517, 1.5514, 1.5486])
    lash_arch_z = np.interp(ax, rim_xs, rim_up_zs)
    tightline = (
        _smoothstep(0.0180, 0.0200, ax)
        * (1.0 - _smoothstep(0.0425, 0.0450, ax))
        * _smoothstep(-0.136, -0.1398, y)
        * np.exp(-(((z - lash_arch_z) / 0.00095) ** 2))
    )
    tight_col = np.array([0.20, 0.11, 0.09])
    base_srgb = base_srgb * (1.0 - 0.72 * tightline[:, None]) + tight_col[None, :] * (0.72 * tightline[:, None])

    # Soft ash-brown eyebrow skin under-tint extending to 53.5 mm lateral tail
    brow_u = np.clip((ax - 0.0145) / 0.0390, 0.0, 1.0)
    brow_z_c = 1.5600 + 0.0018 * np.sin(np.pi * (brow_u ** 0.80)) - 0.0022 * (brow_u ** 2.1)
    brow_under = (
        _smoothstep(0.0135, 0.0185, ax)
        * (1.0 - _smoothstep(0.046, 0.0535, ax))
        * _smoothstep(-0.136, -0.142, y)
        * np.exp(-(((z - brow_z_c) / 0.0028) ** 2))
    )
    brow_tint = np.array([0.54, 0.42, 0.36])
    base_srgb = base_srgb * (1.0 - 0.32 * brow_under[:, None]) + brow_tint[None, :] * (0.32 * brow_under[:, None])

    # 4. Subtle nostril interior cavity darkening
    nostril_occ = _gauss3(ax, y, z, 0.0074, -0.1560, 1.5065, 0.0030, 0.0032, 0.0015)
    base_srgb *= (1.0 - 0.48 * nostril_occ[:, None])

    deep_orbit = (ax > 0.014) & (ax < 0.048) & (z > 1.534) & (z < 1.558) & (y > -0.122)
    base_srgb[deep_orbit] *= 0.15

    # 5. Hydrated Warm Coral-Rose Smiling Lips matching reference.png!
    lip_outer_srgb = np.array([0.765, 0.335, 0.340])
    lip_inner_srgb = np.array([0.635, 0.215, 0.230])
    inner_blend = np.clip(lip_line * 1.15, 0.0, 1.0)[:, None]
    lip_col = lip_outer_srgb[None, :] * (1.0 - 0.45 * inner_blend) + lip_inner_srgb[None, :] * (0.45 * inner_blend)
    lip_col *= (1.0 - 0.25 * (lip_line ** 2.0)[:, None])

    lm = np.clip(lip_mask, 0.0, 1.0)[:, None]
    base_srgb = base_srgb * (1.0 - lm) + lip_col * lm

    # 6. Rich espresso-chestnut scalp root shadow inside hair-bearing cranium
    scalp_col = np.array([0.19, 0.12, 0.09])
    sm = (0.88 * np.clip(scalp_mask, 0.0, 1.0))[:, None]
    base_srgb = base_srgb * (1.0 - sm) + scalp_col[None, :] * sm

    # Convert sRGB -> scene-linear RGBA and store in FLOAT_COLOR attribute 'skin_tone'
    lin_rgb = _srgb_to_linear(base_srgb)
    rgba = np.concatenate([lin_rgb, np.ones((n, 1), dtype=np.float64)], axis=-1).astype(np.float32)

    if "skin_tone" in mesh.color_attributes:
        mesh.color_attributes.remove(mesh.color_attributes["skin_tone"])
    col_attr = mesh.color_attributes.new(name="skin_tone", type="FLOAT_COLOR", domain="POINT")
    col_attr.data.foreach_set("color", rgba.ravel())

    # Specular/Coat gloss mask (hydrated sheen on lower/upper lips + subtle T-zone glow on nose tip & cheeks)
    tzone_glow = (
        0.22 * _gauss3(ax, y, z, 0.0, -0.163, 1.510, 0.011, 0.012, 0.009)
        + 0.16 * cheek_blush
        + 0.12 * _gauss3(ax, y, z, 0.0, -0.148, 1.595, 0.038, 0.025, 0.020)
    )
    gloss_mask = np.clip(0.75 * lip_mask + tzone_glow, 0.0, 1.0).astype(np.float32)
    if "gloss_mask" in mesh.attributes:
        mesh.attributes.remove(mesh.attributes["gloss_mask"])
    ga = mesh.attributes.new(name="gloss_mask", type="FLOAT", domain="POINT")
    ga.data.foreach_set("value", gloss_mask)


def _mix_rgba_node(nt, fac_sock, col_a, col_b, loc=(0, 0)):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    n.location = loc
    rgba_in = [s for s in n.inputs if s.type == "RGBA"]
    rgba_out = [s for s in n.outputs if s.type == "RGBA"][0]
    if isinstance(fac_sock, (int, float)):
        n.inputs["Factor"].default_value = float(fac_sock)
    else:
        nt.links.new(fac_sock, n.inputs["Factor"])
    if isinstance(col_a, tuple):
        rgba_in[0].default_value = col_a
    else:
        nt.links.new(col_a, rgba_in[0])
    if isinstance(col_b, tuple):
        rgba_in[1].default_value = col_b
    else:
        nt.links.new(col_b, rgba_in[1])
    return rgba_out


def _mix_float_node(nt, fac_sock, val_a, val_b, loc=(0, 0)):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "FLOAT"
    n.location = loc
    fl_in = [s for s in n.inputs if s.type == "VALUE"]
    fl_out = [s for s in n.outputs if s.type == "VALUE"][0]
    if isinstance(fac_sock, (int, float)):
        n.inputs["Factor"].default_value = float(fac_sock)
    else:
        nt.links.new(fac_sock, fl_in[0])
    if isinstance(val_a, (int, float)):
        fl_in[1].default_value = float(val_a)
    else:
        nt.links.new(val_a, fl_in[1])
    if isinstance(val_b, (int, float)):
        fl_in[2].default_value = float(val_b)
    else:
        nt.links.new(val_b, fl_in[2])
    return fl_out


def build_all_materials(textures: dict) -> dict:
    """Create all editable node-based Cycles materials and return dict by role."""
    mats = {}

    # ==================================================================
    # 1. MAT_SKIN (Random-Walk SSS + 3D seamless micro-pores + hydrated lip sheen)
    # ==================================================================
    mat_skin = bpy.data.materials.new("Mat_Skin")
    mat_skin.use_nodes = True
    nt = mat_skin.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (650, 0)

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (280, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Read seamless anatomical skin_tone attribute
    attr_col = nt.nodes.new("ShaderNodeAttribute")
    attr_col.attribute_name = "skin_tone"
    attr_col.location = (-520, 260)

    attr_lip = nt.nodes.new("ShaderNodeAttribute")
    attr_lip.attribute_name = "lip_mask"
    attr_lip.location = (-520, 60)

    attr_gloss = nt.nodes.new("ShaderNodeAttribute")
    attr_gloss.attribute_name = "gloss_mask"
    attr_gloss.location = (-520, -100)

    attr_rest = nt.nodes.new("ShaderNodeAttribute")
    attr_rest.attribute_name = "rest_pos"
    attr_rest.location = (-820, -320)

    # Seamless 3D Box-projected skin_micro_2k texture in rest_pos space (zero UV seams across head/neck!)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-620, -320)
    mapping.inputs["Scale"].default_value = (14.0, 14.0, 14.0)  # ~7 cm per 2048px tile -> crisp natural pore scale!
    nt.links.new(attr_rest.outputs["Vector"], mapping.inputs["Vector"])

    tex_micro = nt.nodes.new("ShaderNodeTexImage")
    tex_micro.image = textures["skin_micro"][0]
    tex_micro.projection = "BOX"
    tex_micro.projection_blend = 0.35
    tex_micro.location = (-380, -320)
    nt.links.new(mapping.outputs["Vector"], tex_micro.inputs["Vector"])

    sep_micro = nt.nodes.new("ShaderNodeSeparateColor")
    sep_micro.location = (-120, -320)
    nt.links.new(tex_micro.outputs["Color"], sep_micro.inputs["Color"])

    # Base Color
    nt.links.new(attr_col.outputs["Color"], bsdf.inputs["Base Color"])

    # Realistic Human Dermis Random-Walk Subsurface Scattering (in meters!)
    if hasattr(bsdf, "subsurface_method"):
        try:
            bsdf.subsurface_method = "RANDOM_WALK_SKIN"
        except Exception:
            pass
    bsdf.inputs["Subsurface Weight"].default_value = 0.58
    bsdf.inputs["Subsurface Scale"].default_value = 0.0072  # 7.2 mm physical dermis scatter scale
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.36, 0.21)
    if "Subsurface IOR" in bsdf.inputs:
        bsdf.inputs["Subsurface IOR"].default_value = 1.40
    if "Subsurface Anisotropy" in bsdf.inputs:
        bsdf.inputs["Subsurface Anisotropy"].default_value = 0.78
    bsdf.inputs["IOR"].default_value = 1.42
    bsdf.inputs["Specular IOR Level"].default_value = 0.46

    # Roughness: ~0.37 natural velvety skin with pore micro-roughness, dropping to ~0.19 hydrated satin on lips!
    skin_rough_base = nt.nodes.new("ShaderNodeMath")
    skin_rough_base.operation = "MULTIPLY_ADD"
    skin_rough_base.inputs[1].default_value = 0.10
    skin_rough_base.inputs[2].default_value = 0.32
    nt.links.new(sep_micro.outputs["Green"], skin_rough_base.inputs[0])

    final_rough = _mix_float_node(nt, attr_lip.outputs["Fac"], skin_rough_base.outputs[0], 0.19, loc=(60, -80))
    nt.links.new(final_rough, bsdf.inputs["Roughness"])

    # Hydrated Coat Sheen on lips & T-zone highlights
    coat_w = _mix_float_node(nt, attr_gloss.outputs["Fac"], 0.08, 0.42, loc=(60, -220))
    nt.links.new(coat_w, bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.22

    # Soft microscopic vellus peach-fuzz sheen on cheeks
    bsdf.inputs["Sheen Weight"].default_value = 0.14
    bsdf.inputs["Sheen Roughness"].default_value = 0.68
    bsdf.inputs["Sheen Tint"].default_value = (1.0, 0.88, 0.84, 1.0)

    # Restrained micro-pore & fine vertical lip-line bump
    # Subtle vertical lip sulci driven by rest_pos.x * 950
    sep_rest = nt.nodes.new("ShaderNodeSeparateXYZ")
    sep_rest.location = (-620, -540)
    nt.links.new(attr_rest.outputs["Vector"], sep_rest.inputs["Vector"])

    lip_freq = nt.nodes.new("ShaderNodeMath")
    lip_freq.operation = "MULTIPLY"
    lip_freq.inputs[1].default_value = 1350.0
    nt.links.new(sep_rest.outputs["X"], lip_freq.inputs[0])

    lip_sin = nt.nodes.new("ShaderNodeMath")
    lip_sin.operation = "SINE"
    nt.links.new(lip_freq.outputs[0], lip_sin.inputs[0])

    bump_h = _mix_float_node(nt, attr_lip.outputs["Fac"], sep_micro.outputs["Red"], lip_sin.outputs[0], loc=(-20, -440))

    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (110, -400)
    bump.inputs["Strength"].default_value = 0.075
    bump.inputs["Distance"].default_value = 0.00028  # 0.28 mm subtle realistic micro-relief!
    nt.links.new(bump_h, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    mats["skin"] = mat_skin

    # ==================================================================
    # 2. MAT_SCLERA_IRIS & MAT_CORNEA & MAT_CARUNCLE
    # ==================================================================
    mat_eye_in = bpy.data.materials.new("Mat_Sclera_Iris")
    mat_eye_in.use_nodes = True
    nte = mat_eye_in.node_tree
    nte.nodes.clear()
    oute = nte.nodes.new("ShaderNodeOutputMaterial")
    bsdfe = nte.nodes.new("ShaderNodeBsdfPrincipled")
    nte.links.new(bsdfe.outputs["BSDF"], oute.inputs["Surface"])

    tex_iris = nte.nodes.new("ShaderNodeTexImage")
    tex_iris.image = textures["iris_albedo"][0]
    tex_iris.extension = "EXTEND"
    tex_iris.location = (-380, 160)

    tex_ibump = nte.nodes.new("ShaderNodeTexImage")
    tex_ibump.image = textures["iris_bump"][0]
    tex_ibump.extension = "EXTEND"
    tex_ibump.location = (-380, -140)

    ibump = nte.nodes.new("ShaderNodeBump")
    ibump.inputs["Strength"].default_value = 0.28
    ibump.inputs["Distance"].default_value = 0.0005
    nte.links.new(tex_ibump.outputs["Color"], ibump.inputs["Height"])

    nte.links.new(tex_iris.outputs["Color"], bsdfe.inputs["Base Color"])
    nte.links.new(ibump.outputs["Normal"], bsdfe.inputs["Normal"])
    bsdfe.inputs["Roughness"].default_value = 0.35
    bsdfe.inputs["Subsurface Weight"].default_value = 0.35
    bsdfe.inputs["Subsurface Scale"].default_value = 0.0035
    bsdfe.inputs["Subsurface Radius"].default_value = (1.0, 0.42, 0.28)
    bsdfe.inputs["Specular IOR Level"].default_value = 0.25
    mats["eye_inner"] = mat_eye_in

    # Cornea & Tear Meniscus: physically accurate Fresnel IOR=1.376 crisp specular catchlights + noise-free transparency!
    mat_cornea = bpy.data.materials.new("Mat_Cornea")
    mat_cornea.use_nodes = True
    ntc = mat_cornea.node_tree
    ntc.nodes.clear()
    outc = ntc.nodes.new("ShaderNodeOutputMaterial")

    transp = ntc.nodes.new("ShaderNodeBsdfTransparent")
    gloss = ntc.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    gloss.inputs["Roughness"].default_value = 0.008  # razor-crisp wet corneal reflection!

    fres = ntc.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.376

    # Boost grazing & catchlight reflection slightly so window catchlights sparkle just like reference.png!
    fres_boost = ntc.nodes.new("ShaderNodeMath")
    fres_boost.operation = "MAXIMUM"
    fres_boost.inputs[1].default_value = 0.032
    ntc.links.new(fres.outputs["Fac"], fres_boost.inputs[0])

    mix_sh = ntc.nodes.new("ShaderNodeMixShader")
    ntc.links.new(fres_boost.outputs[0], mix_sh.inputs["Fac"])
    ntc.links.new(transp.outputs["BSDF"], mix_sh.inputs[1])
    ntc.links.new(gloss.outputs["BSDF"], mix_sh.inputs[2])

    # Shadow rays pass straight through cornea cleanly
    lp = ntc.nodes.new("ShaderNodeLightPath")
    mix_shad = ntc.nodes.new("ShaderNodeMixShader")
    ntc.links.new(lp.outputs["Is Shadow Ray"], mix_shad.inputs["Fac"])
    ntc.links.new(mix_sh.outputs["Shader"], mix_shad.inputs[1])
    ntc.links.new(transp.outputs["BSDF"], mix_shad.inputs[2])
    ntc.links.new(mix_shad.outputs["Shader"], outc.inputs["Surface"])
    mats["cornea"] = mat_cornea

    # Lacrimal Caruncle (subtle moist rose medial tear-duct mucosa)
    mat_car = bpy.data.materials.new("Mat_Caruncle")
    mat_car.use_nodes = True
    bsdf_car = mat_car.node_tree.nodes["Principled BSDF"]
    bsdf_car.inputs["Base Color"].default_value = (0.38, 0.16, 0.15, 1.0)
    bsdf_car.inputs["Roughness"].default_value = 0.12
    bsdf_car.inputs["Subsurface Weight"].default_value = 0.65
    bsdf_car.inputs["Subsurface Scale"].default_value = 0.003
    bsdf_car.inputs["Coat Weight"].default_value = 0.70
    bsdf_car.inputs["Coat Roughness"].default_value = 0.05
    mats["caruncle"] = mat_car

    # ==================================================================
    # 3. MAT_HAIR_CHESTNUT, MAT_HAIR_BROWS, MAT_HAIR_LASHES
    # ==================================================================
    mat_hair = bpy.data.materials.new("Mat_Hair_Chestnut")
    mat_hair.use_nodes = True
    nth = mat_hair.node_tree
    nth.nodes.clear()
    outh = nth.nodes.new("ShaderNodeOutputMaterial")

    hair_bsdf = nth.nodes.new("ShaderNodeBsdfHairPrincipled")
    hair_bsdf.parametrization = "MELANIN"
    nth.links.new(hair_bsdf.outputs["BSDF"], outh.inputs["Surface"])

    hinfo = nth.nodes.new("ShaderNodeHairInfo")
    hinfo.location = (-360, 120)

    # Rich East Asian dark-medium warm chestnut brown: roots (0.87) -> sunlit mid/tips (0.78)
    mel_mix = _mix_float_node(nth, hinfo.outputs["Intercept"], 0.87, 0.78, loc=(-120, 140))
    nth.links.new(mel_mix, hair_bsdf.inputs["Melanin"])

    hair_bsdf.inputs["Melanin Redness"].default_value = 0.44
    hair_bsdf.inputs["Tint"].default_value = (0.94, 0.82, 0.74, 1.0)
    hair_bsdf.inputs["Roughness"].default_value = 0.23
    hair_bsdf.inputs["Radial Roughness"].default_value = 0.32
    hair_bsdf.inputs["Coat"].default_value = 0.26
    hair_bsdf.inputs["IOR"].default_value = 1.55
    hair_bsdf.inputs["Offset"].default_value = math.radians(2.0)
    hair_bsdf.inputs["Random Color"].default_value = 0.11
    hair_bsdf.inputs["Random Roughness"].default_value = 0.08
    mats["hair"] = mat_hair

    # Eyebrows: soft light-medium ash-chestnut brown
    mat_brow = bpy.data.materials.new("Mat_Hair_Brows")
    mat_brow.use_nodes = True
    ntb = mat_brow.node_tree
    ntb.nodes.clear()
    outb = ntb.nodes.new("ShaderNodeOutputMaterial")
    hb_brow = ntb.nodes.new("ShaderNodeBsdfHairPrincipled")
    hb_brow.parametrization = "MELANIN"
    hb_brow.inputs["Melanin"].default_value = 0.68
    hb_brow.inputs["Melanin Redness"].default_value = 0.34
    hb_brow.inputs["Roughness"].default_value = 0.38
    hb_brow.inputs["Radial Roughness"].default_value = 0.46
    hb_brow.inputs["Random Color"].default_value = 0.08
    ntb.links.new(hb_brow.outputs["BSDF"], outb.inputs["Surface"])
    mats["brows"] = mat_brow

    # Eyelashes: deep espresso-black
    mat_lash = bpy.data.materials.new("Mat_Hair_Lashes")
    mat_lash.use_nodes = True
    ntl = mat_lash.node_tree
    ntl.nodes.clear()
    outl = ntl.nodes.new("ShaderNodeOutputMaterial")
    hb_lash = ntl.nodes.new("ShaderNodeBsdfHairPrincipled")
    hb_lash.parametrization = "MELANIN"
    hb_lash.inputs["Melanin"].default_value = 0.95
    hb_lash.inputs["Melanin Redness"].default_value = 0.22
    hb_lash.inputs["Roughness"].default_value = 0.25
    hb_lash.inputs["Radial Roughness"].default_value = 0.35
    ntl.links.new(hb_lash.outputs["BSDF"], outl.inputs["Surface"])
    mats["lashes"] = mat_lash

    # ==================================================================
    # 4. MAT_SWEATER_KNIT & MAT_SWEATER_RIB (Dusty Pink Plush Knitwear)
    # ==================================================================
    def make_knit_mat(mat_name: str, height_img_key: str, bump_strength: float, uv_scale=(1.0, 1.0)):
        m = bpy.data.materials.new(mat_name)
        m.use_nodes = True
        ntk = m.node_tree
        ntk.nodes.clear()
        outk = ntk.nodes.new("ShaderNodeOutputMaterial")
        outk.location = (550, 0)

        bsdfk = ntk.nodes.new("ShaderNodeBsdfPrincipled")
        bsdfk.location = (220, 0)
        ntk.links.new(bsdfk.outputs["BSDF"], outk.inputs["Surface"])

        uvn = ntk.nodes.new("ShaderNodeTexCoord")
        uvn.location = (-760, 0)
        mapk = ntk.nodes.new("ShaderNodeMapping")
        mapk.location = (-560, 0)
        mapk.inputs["Scale"].default_value = (uv_scale[0], uv_scale[1], 1.0)
        ntk.links.new(uvn.outputs["UV"], mapk.inputs["Vector"])

        t_alb = ntk.nodes.new("ShaderNodeTexImage")
        t_alb.image = textures["knit_albedo"][0]
        t_alb.location = (-320, 240)
        ntk.links.new(mapk.outputs["Vector"], t_alb.inputs["Vector"])

        t_hgt = ntk.nodes.new("ShaderNodeTexImage")
        t_hgt.image = textures[height_img_key][0]
        t_hgt.location = (-320, -40)
        ntk.links.new(mapk.outputs["Vector"], t_hgt.inputs["Vector"])

        t_rgh = ntk.nodes.new("ShaderNodeTexImage")
        t_rgh.image = textures["knit_roughness"][0]
        t_rgh.location = (-320, -300)
        ntk.links.new(mapk.outputs["Vector"], t_rgh.inputs["Vector"])

        # For ribbed trim, modulate albedo subtly with rib height AO so chunky vertical 1x1 ribs pop naturally!
        if height_img_key == "rib_height":
            rib_ao = _mix_rgba_node(
                ntk,
                t_hgt.outputs["Color"],
                (0.42, 0.20, 0.22, 1.0),
                (0.76, 0.44, 0.47, 1.0),
                loc=(-40, 200),
            )
            alb_mix = _mix_rgba_node(ntk, 0.38, t_alb.outputs["Color"], rib_ao, loc=(60, 220))
            ntk.links.new(alb_mix, bsdfk.inputs["Base Color"])
        else:
            ntk.links.new(t_alb.outputs["Color"], bsdfk.inputs["Base Color"])

        ntk.links.new(t_rgh.outputs["Color"], bsdfk.inputs["Roughness"])
        bsdfk.inputs["Diffuse Roughness"].default_value = 0.45  # Oren-Nayar soft wool diffuse!
        bsdfk.inputs["Specular IOR Level"].default_value = 0.28

        # Plush knit yarn fuzz sheen (huge realism boost in grazing cafe window sunlight!)
        bsdfk.inputs["Sheen Weight"].default_value = 0.78
        bsdfk.inputs["Sheen Roughness"].default_value = 0.56
        bsdfk.inputs["Sheen Tint"].default_value = (0.98, 0.80, 0.82, 1.0)

        # Subtle warm yarn fiber subsurface translucency when backlit by cafe window!
        bsdfk.inputs["Subsurface Weight"].default_value = 0.18
        bsdfk.inputs["Subsurface Scale"].default_value = 0.0035
        bsdfk.inputs["Subsurface Radius"].default_value = (1.0, 0.55, 0.58)

        bmpk = ntk.nodes.new("ShaderNodeBump")
        bmpk.location = (-20, -140)
        bmpk.inputs["Strength"].default_value = bump_strength
        bmpk.inputs["Distance"].default_value = 0.0024
        ntk.links.new(t_hgt.outputs["Color"], bmpk.inputs["Height"])
        ntk.links.new(bmpk.outputs["Normal"], bsdfk.inputs["Normal"])
        return m

    mats["sweater_knit"] = make_knit_mat("Mat_Sweater_Knit", "knit_height", bump_strength=0.58, uv_scale=(1.25, 1.25))
    mats["sweater_rib"] = make_knit_mat("Mat_Sweater_Rib", "rib_height", bump_strength=0.72, uv_scale=(1.15, 1.15))

    mat_inner = bpy.data.materials.new("Mat_Sweater_Inner")
    mat_inner.use_nodes = True
    bi = mat_inner.node_tree.nodes["Principled BSDF"]
    bi.inputs["Base Color"].default_value = (0.025, 0.012, 0.014, 1.0)
    bi.inputs["Roughness"].default_value = 1.0
    bi.inputs["Specular IOR Level"].default_value = 0.0
    mats["sweater_inner"] = mat_inner

    return mats
