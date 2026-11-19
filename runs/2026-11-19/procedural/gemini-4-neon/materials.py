"""
materials.py - Physically based Cycles materials for skin (Random Walk SSS),
eyes (iris/sclera + glossy cornea + tear meniscus), hair (Principled Hair BSDF),
chunky dusty-pink knitted wool sweater, and cafe background props.
"""
import bpy


def new_material(name):
    if name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[name])
    mat = bpy.data.materials.new(name=name)
    # In Blender 5.1 materials use node trees by default
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, nt


def create_skin_material(skin_imgs):
    """Photorealistic fair warm skin with Random Walk SSS, restrained pores, and soft glossy lips."""
    mat, nt = new_material("Mat_Skin")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Random Walk Subsurface Scattering
    bsdf.subsurface_method = 'RANDOM_WALK'
    bsdf.inputs["Subsurface Weight"].default_value = 0.24
    bsdf.inputs["Subsurface Scale"].default_value = 0.011
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.36, 0.20)
    if "Subsurface IOR" in bsdf.inputs:
        bsdf.inputs["Subsurface IOR"].default_value = 1.4
    if "Subsurface Anisotropy" in bsdf.inputs:
        bsdf.inputs["Subsurface Anisotropy"].default_value = 0.25
    bsdf.inputs["IOR"].default_value = 1.40
    bsdf.inputs["Specular IOR Level"].default_value = 0.44
    bsdf.inputs["Sheen Weight"].default_value = 0.06
    bsdf.inputs["Sheen Roughness"].default_value = 0.65

    # UV coordinates
    texcoord = nt.nodes.new("ShaderNodeTexCoord")

    # Albedo
    tex_alb = nt.nodes.new("ShaderNodeTexImage")
    tex_alb.image = skin_imgs["skin_albedo"]
    nt.links.new(texcoord.outputs["UV"], tex_alb.inputs["Vector"])
    nt.links.new(tex_alb.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness
    tex_rgh = nt.nodes.new("ShaderNodeTexImage")
    tex_rgh.image = skin_imgs["skin_roughness"]
    nt.links.new(texcoord.outputs["UV"], tex_rgh.inputs["Vector"])
    nt.links.new(tex_rgh.outputs["Color"], bsdf.inputs["Roughness"])

    # Restrained Micro-Bump (pores + delicate lip creases)
    tex_bmp = nt.nodes.new("ShaderNodeTexImage")
    tex_bmp.image = skin_imgs["skin_bump"]
    nt.links.new(texcoord.outputs["UV"], tex_bmp.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.065
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(tex_bmp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Soft moist lip-balm sheen via Coat Weight driven by lip mask
    tex_lip = nt.nodes.new("ShaderNodeTexImage")
    tex_lip.image = skin_imgs["skin_lipmask"]
    nt.links.new(texcoord.outputs["UV"], tex_lip.inputs["Vector"])
    lip_scale = nt.nodes.new("ShaderNodeMath")
    lip_scale.operation = 'MULTIPLY'
    lip_scale.inputs[1].default_value = 0.42
    nt.links.new(tex_lip.outputs["Color"], lip_scale.inputs[0])
    nt.links.new(lip_scale.outputs["Value"], bsdf.inputs["Coat Weight"])
    bsdf.inputs["Coat Roughness"].default_value = 0.14
    bsdf.inputs["Coat IOR"].default_value = 1.46

    return mat


def create_eye_materials(iris_img):
    """Create Eyeball (sclera+iris), Cornea (clear glossy shell), TearLine, and Caruncle materials."""
    # --- 1. Inner Eyeball ---
    mat_eye, nt = new_material("Mat_Eyeball")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    texcoord = nt.nodes.new("ShaderNodeTexCoord")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = iris_img
    nt.links.new(texcoord.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Roughness"].default_value = 0.18
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    bsdf.inputs["Subsurface Weight"].default_value = 0.12
    bsdf.inputs["Subsurface Scale"].default_value = 0.004
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.55, 0.40)

    # --- 2. Outer Cornea (Transparent + Fresnel Glossy for crisp catchlights without darkening iris!) ---
    mat_cornea, nt = new_material("Mat_Cornea")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    transp.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy = nt.nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs["Roughness"].default_value = 0.012

    fresnel = nt.nodes.new("ShaderNodeLayerWeight")
    fresnel.inputs["Blend"].default_value = 0.16
    # Boost subtle reflection floor slightly so window catchlights sparkle brightly like reference.png
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.0
    ramp.inputs["From Max"].default_value = 1.0
    ramp.inputs["To Min"].default_value = 0.065
    ramp.inputs["To Max"].default_value = 0.55
    nt.links.new(fresnel.outputs["Facing"], ramp.inputs["Value"])

    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(ramp.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(glossy.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # --- 3. Tear Meniscus ---
    mat_tear, nt = new_material("Mat_TearLine")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    glossy = nt.nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Roughness"].default_value = 0.025
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.25
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lw.outputs["Facing"], mix.inputs["Fac"])
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(glossy.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # --- 4. Caruncle ---
    mat_car, nt = new_material("Mat_Caruncle")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.58, 0.22, 0.20, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.12
    bsdf.inputs["Coat Weight"].default_value = 0.7
    bsdf.inputs["Coat Roughness"].default_value = 0.05
    bsdf.inputs["Subsurface Weight"].default_value = 0.35
    bsdf.inputs["Subsurface Scale"].default_value = 0.003
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return {
        "eyeball": mat_eye,
        "cornea": mat_cornea,
        "tear": mat_tear,
        "caruncle": mat_car,
    }


def create_hair_materials():
    """
    Create Principled Hair BSDF materials for:
      - Chestnut brown main hair (warm chocolate-chestnut with sunlit golden-brown sheen & strand randomness)
      - Dark espresso eyelashes
      - Soft medium-dark chestnut eyebrows
      - Inner scalp volume cap (dark chestnut matte so no bald gaps ever appear)
    """
    # --- 1. Main Chestnut Brown Hair ---
    mat_hair, nt = new_material("Mat_Hair_Chestnut")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    hair.parametrization = 'MELANIN'
    hair.model = 'CHIANG'
    # Rich dark-medium warm chestnut brown matching input/reference.png
    hair.inputs["Melanin"].default_value = 0.80
    hair.inputs["Melanin Redness"].default_value = 0.55
    hair.inputs["Tint"].default_value = (0.88, 0.70, 0.58, 1.0)
    hair.inputs["Roughness"].default_value = 0.28
    if "Radial Roughness" in hair.inputs:
        hair.inputs["Radial Roughness"].default_value = 0.36
    if "Coat" in hair.inputs:
        hair.inputs["Coat"].default_value = 0.14
    hair.inputs["IOR"].default_value = 1.55
    hair.inputs["Offset"].default_value = 0.032
    hair.inputs["Random Color"].default_value = 0.16
    hair.inputs["Random Roughness"].default_value = 0.16

    # Subtle root-to-tip warmth variation via Curves Info intercept
    hinfo = nt.nodes.new("ShaderNodeHairInfo")
    ramp = nt.nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.0
    ramp.inputs["From Max"].default_value = 1.0
    ramp.inputs["To Min"].default_value = 0.86  # Deeper espresso-chestnut at roots
    ramp.inputs["To Max"].default_value = 0.72  # Rich warm chestnut toward ends
    nt.links.new(hinfo.outputs["Intercept"], ramp.inputs["Value"])
    nt.links.new(ramp.outputs["Result"], hair.inputs["Melanin"])
    nt.links.new(hinfo.outputs["Random"], hair.inputs["Random"])
    nt.links.new(hair.outputs["BSDF"], out.inputs["Surface"])

    # --- 2. Eyelashes (rich dark espresso-black) ---
    mat_lash, nt = new_material("Mat_Eyelashes")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    hair.parametrization = 'MELANIN'
    hair.model = 'CHIANG'
    hair.inputs["Melanin"].default_value = 0.94
    hair.inputs["Melanin Redness"].default_value = 0.25
    hair.inputs["Roughness"].default_value = 0.30
    if "Radial Roughness" in hair.inputs:
        hair.inputs["Radial Roughness"].default_value = 0.36
    if "Coat" in hair.inputs:
        hair.inputs["Coat"].default_value = 0.12
    nt.links.new(hair.outputs["BSDF"], out.inputs["Surface"])

    # --- 3. Eyebrows (soft natural medium-dark chestnut brown) ---
    mat_brow, nt = new_material("Mat_Eyebrows")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    hair = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    hair.parametrization = 'MELANIN'
    hair.model = 'CHIANG'
    hair.inputs["Melanin"].default_value = 0.80
    hair.inputs["Melanin Redness"].default_value = 0.45
    hair.inputs["Roughness"].default_value = 0.38
    if "Radial Roughness" in hair.inputs:
        hair.inputs["Radial Roughness"].default_value = 0.44
    hair.inputs["Random Color"].default_value = 0.12
    nt.links.new(hair.outputs["BSDF"], out.inputs["Surface"])

    # --- 4. Inner Scalp Volume Cap (matte dark chestnut under-layer) ---
    mat_scalp, nt = new_material("Mat_Scalp_Under")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.025, 0.013, 0.008, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.90
    bsdf.inputs["Specular IOR Level"].default_value = 0.08
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return {
        "hair": mat_hair,
        "lashes": mat_lash,
        "brows": mat_brow,
        "scalp": mat_scalp,
    }


def create_sweater_materials(knit_imgs):
    """
    Create dusty blush-pink knitted wool sweater materials:
      - Mat_Sweater_Knit: stockinette V-stitch chunky knit with bump, subtle melange, warm wool sheen
      - Mat_Sweater_Rib: chunky ribbed knit for crew-neck collar, cuffs, and waist hem
      - Mat_Sweater_Fuzz: soft pale-pink translucent wool halo fibers
    """
    def _build_knit(name, height_img, uv_scale=(7.0, 5.5), bump_str=0.85, color_mult=(1.0, 1.0, 1.0)):
        mat, nt = new_material(name)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        texcoord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (uv_scale[0], uv_scale[1], 1.0)
        nt.links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])

        tex_col = nt.nodes.new("ShaderNodeTexImage")
        tex_col.image = knit_imgs["knit_color"]
        nt.links.new(mapping.outputs["Vector"], tex_col.inputs["Vector"])

        # Subtle color tint adjustment if needed (e.g. collar slightly deeper crevice)
        if color_mult != (1.0, 1.0, 1.0):
            mul = nt.nodes.new("ShaderNodeMix")
            mul.data_type = 'RGBA'
            mul.blend_type = 'MULTIPLY'
            mul.inputs["Factor"].default_value = 1.0
            mul.inputs[7].default_value = (*color_mult, 1.0)
            nt.links.new(tex_col.outputs["Color"], mul.inputs[6])
            nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])
        else:
            nt.links.new(tex_col.outputs["Color"], bsdf.inputs["Base Color"])

        tex_h = nt.nodes.new("ShaderNodeTexImage")
        tex_h.image = height_img
        nt.links.new(mapping.outputs["Vector"], tex_h.inputs["Vector"])

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_str
        bump.inputs["Distance"].default_value = 0.0035
        nt.links.new(tex_h.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Wool fabric optical response: matte diffuse + warm rose-tinted soft fuzz sheen
        bsdf.inputs["Roughness"].default_value = 0.88
        bsdf.inputs["Specular IOR Level"].default_value = 0.12
        bsdf.inputs["Sheen Weight"].default_value = 0.48
        bsdf.inputs["Sheen Roughness"].default_value = 0.55
        bsdf.inputs["Sheen Tint"].default_value = (0.95, 0.72, 0.74, 1.0)
        bsdf.inputs["Subsurface Weight"].default_value = 0.05
        bsdf.inputs["Subsurface Scale"].default_value = 0.003
        bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.50, 0.52)
        return mat

    mat_knit = _build_knit("Mat_Sweater_Knit", knit_imgs["knit_height"], uv_scale=(4.6, 3.6), bump_str=0.95)
    mat_rib = _build_knit("Mat_Sweater_Rib", knit_imgs["knit_rib_height"], uv_scale=(3.2, 0.85), bump_str=0.95, color_mult=(0.94, 0.92, 0.92))

    # Recessed shadowed cuff interior so wrist openings read as hollow dark knit cuffs
    mat_cuff_in, nt = new_material("Mat_Sweater_CuffInterior")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.14, 0.075, 0.080, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Specular IOR Level"].default_value = 0.05
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Fuzzy wool halo fibers
    mat_fuzz, nt = new_material("Mat_Sweater_Fuzz")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.76, 0.46, 0.49, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.80
    bsdf.inputs["Sheen Weight"].default_value = 0.75
    bsdf.inputs["Sheen Roughness"].default_value = 0.45
    bsdf.inputs["Transmission Weight"].default_value = 0.15
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return {
        "sweater": mat_knit,
        "sweater_rib": mat_rib,
        "sweater_cuff_interior": mat_cuff_in,
        "sweater_fuzz": mat_fuzz,
    }


def create_cafe_materials():
    """Create simple physically plausible materials for the blurred sunlit cafe background."""
    mats = {}

    def simple_pbr(name, color, rough=0.5, metal=0.0, trans=0.0, coat=0.0, sss=0.0, emit=0.0):
        mat, nt = new_material(name)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
        bsdf.inputs["Transmission Weight"].default_value = trans
        bsdf.inputs["Coat Weight"].default_value = coat
        bsdf.inputs["Subsurface Weight"].default_value = sss
        if emit > 0.0:
            bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emit
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat

    mats["wood_oak"] = simple_pbr("Mat_Cafe_WoodOak", (0.48, 0.29, 0.14), rough=0.42, coat=0.15)
    mats["wood_dark"] = simple_pbr("Mat_Cafe_WoodDark", (0.22, 0.13, 0.07), rough=0.48)
    mats["wall_cream"] = simple_pbr("Mat_Cafe_WallCream", (0.86, 0.82, 0.75), rough=0.82)
    mats["pot_ceramic"] = simple_pbr("Mat_Cafe_PotCeramic", (0.80, 0.77, 0.72), rough=0.55)
    mats["leaf_green"] = simple_pbr("Mat_Cafe_LeafGreen", (0.14, 0.34, 0.08), rough=0.35, sss=0.35)
    mats["glass_vase"] = simple_pbr("Mat_Cafe_GlassVase", (0.95, 0.98, 0.97), rough=0.04, trans=0.92)
    mats["flower_white"] = simple_pbr("Mat_Cafe_FlowerWhite", (0.95, 0.93, 0.86), rough=0.55, sss=0.4)
    mats["metal_cafe"] = simple_pbr("Mat_Cafe_Metal", (0.68, 0.67, 0.65), rough=0.28, metal=0.88)
    mats["window_glow"] = simple_pbr("Mat_Cafe_WindowGlow", (1.0, 0.97, 0.91), rough=0.1, emit=8.5)
    return mats
