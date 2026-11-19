"""
textures.py - Procedural high-resolution (<=2048px) packed image textures generated
via NumPy: 3D-to-UV baked facial skin (albedo, roughness, bump, lip mask),
photorealistic warm brown iris/sclera, and chunky stockinette + ribbed wool knit maps.
"""
import math
import bpy
import numpy as np


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_blender_image(name, rgba_float, colorspace="sRGB"):
    """Create a packed bpy.types.Image from an (H, W, 4) float32 array in [0, 1]."""
    h, w, _ = rgba_float.shape
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    img = bpy.data.images.new(name=name, width=w, height=h, alpha=True, float_buffer=False)
    img.colorspace_settings.name = colorspace
    # Blender expects flat bottom-to-top RGBA floats
    flat = np.ascontiguousarray(rgba_float, dtype=np.float32).ravel()
    img.pixels.foreach_set(flat)
    img.pack()
    return img


def hash_noise_2d(grid_x, grid_y, seed=0):
    """Fast deterministic value noise on 2D coordinates."""
    ix = np.floor(grid_x).astype(np.int64)
    iy = np.floor(grid_y).astype(np.int64)
    fx = (grid_x - ix).astype(np.float32)
    fy = (grid_y - iy).astype(np.float32)
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    def h(a, b):
        n = (a * 374761393 + b * 668265263 + int(seed) * 1442695041) & 0x7FFFFFFF
        n = ((n ^ (n >> 13)) * 1274126177) & 0x7FFFFFFF
        return (((n ^ (n >> 16)) & 0xFFFF).astype(np.float32)) / 65535.0

    v00 = h(ix, iy)
    v10 = h(ix + 1, iy)
    v01 = h(ix, iy + 1)
    v11 = h(ix + 1, iy + 1)
    return (v00 * (1.0 - ux) + v10 * ux) * (1.0 - uy) + (v01 * (1.0 - ux) + v11 * ux) * uy


def fbm_3d_fast(x, y, z, freq=60.0, octaves=3, seed=11):
    """Fast seam-free 3D pseudo-fBM using projected hash layers."""
    out = np.zeros_like(x, dtype=np.float32)
    amp = 0.5
    f = freq
    for i in range(octaves):
        # Tri-planar style combination for isotropy without UV seams
        n1 = hash_noise_2d(x * f + z * (f * 0.37), y * f - z * (f * 0.29), seed + i * 31)
        n2 = hash_noise_2d(y * f + x * (f * 0.41), z * f + x * (f * 0.23), seed + i * 31 + 17)
        out += amp * (0.5 * (n1 + n2))
        f *= 2.13
        amp *= 0.5
    return out


def rasterize_3d_pos_to_uv(mesh, canonical_verts, size=2048):
    """
    Rasterize canonical 3D vertex positions onto the mesh's UV space (size x size)
    and dilate UV island edges so texture filtering never bleeds background pixels.
    Returns pos_map (size, size, 3) float32 and mask (size, size) bool.
    """
    uv_layer = mesh.uv_layers.active.data
    pos_map = np.zeros((size, size, 3), dtype=np.float32)
    mask = np.zeros((size, size), dtype=bool)

    # Extract triangles (split quads into 2 tris)
    for poly in mesh.polygons:
        loops = list(poly.loop_indices)
        verts = [mesh.loops[li].vertex_index for li in loops]
        uvs = [uv_layer[li].uv for li in loops]
        if len(loops) == 3:
            tris = [(0, 1, 2)]
        elif len(loops) == 4:
            tris = [(0, 1, 2), (0, 2, 3)]
        else:
            tris = [(0, i, i + 1) for i in range(1, len(loops) - 1)]

        for i0, i1, i2 in tris:
            u0, v0 = uvs[i0].x * (size - 1), uvs[i0].y * (size - 1)
            u1, v1 = uvs[i1].x * (size - 1), uvs[i1].y * (size - 1)
            u2, v2 = uvs[i2].x * (size - 1), uvs[i2].y * (size - 1)

            min_x = max(0, int(math.floor(min(u0, u1, u2))))
            max_x = min(size - 1, int(math.ceil(max(u0, u1, u2))))
            min_y = max(0, int(math.floor(min(v0, v1, v2))))
            max_y = min(size - 1, int(math.ceil(max(v0, v1, v2))))
            if max_x < min_x or max_y < min_y:
                continue

            denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
            if abs(denom) < 1e-8:
                continue
            inv_d = 1.0 / denom

            xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
            ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
            gx, gy = np.meshgrid(xs, ys)

            w0 = ((v1 - v2) * (gx - u2) + (u2 - u1) * (gy - v2)) * inv_d
            w1 = ((v2 - v0) * (gx - u2) + (u0 - u2) * (gy - v2)) * inv_d
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-3) & (w1 >= -1e-3) & (w2 >= -1e-3)
            if not np.any(inside):
                continue

            p0 = canonical_verts[verts[i0]]
            p1 = canonical_verts[verts[i1]]
            p2 = canonical_verts[verts[i2]]

            sub_pos = pos_map[min_y:max_y + 1, min_x:max_x + 1]
            sub_mask = mask[min_y:max_y + 1, min_x:max_x + 1]
            px = w0 * p0[0] + w1 * p1[0] + w2 * p2[0]
            py = w0 * p0[1] + w1 * p1[1] + w2 * p2[1]
            pz = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]

            sub_pos[inside, 0] = px[inside]
            sub_pos[inside, 1] = py[inside]
            sub_pos[inside, 2] = pz[inside]
            sub_mask[inside] = True

    # Multi-pass 4-neighbor dilation to eliminate UV seam bleed
    dilated_mask = mask.copy()
    for _ in range(12):
        need = ~dilated_mask
        if not np.any(need):
            break
        acc = np.zeros_like(pos_map)
        cnt = np.zeros((size, size), dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_m = np.roll(np.roll(dilated_mask, dy, axis=0), dx, axis=1)
            shifted_p = np.roll(np.roll(pos_map, dy, axis=0), dx, axis=1)
            valid = need & shifted_m
            acc[valid] += shifted_p[valid]
            cnt[valid] += 1.0
        fill = cnt > 0
        pos_map[fill] = acc[fill] / cnt[fill, None]
        dilated_mask[fill] = True

    return pos_map, dilated_mask


def generate_skin_textures(body_obj, canonical_verts, eye_c_l, eye_c_r, size=1536):
    """
    Generate anatomically accurate packed facial & upper-body skin textures:
      - Skin_Albedo (sRGB): fair warm porcelain-peach skin, rosy cheek/nose blush,
        delicate warm orbital shading & upper lash tightline, soft coral-rose lips,
        subtle ear warmth, nostril occlusion, seam-free organic skin mottling.
      - Skin_Roughness (Non-Color): glossy-satin lips, subtle nose/cheek sheen, matte jaw/neck.
      - Skin_Bump (Non-Color): ultra-restrained fine pores, micro-skin grain, subtle vertical lip creases.
      - Skin_LipMask (Non-Color): clean mask for subtle clearcoat lip gloss.
    """
    pos, mask = rasterize_3d_pos_to_uv(body_obj.data, canonical_verts, size=size)
    x = pos[:, :, 0]
    y = pos[:, :, 1]
    z = pos[:, :, 2]
    ax = np.abs(x)

    # Organic multi-scale seam-free skin noise
    n_macro = fbm_3d_fast(x, y, z, freq=28.0, octaves=3, seed=7)
    n_mid = fbm_3d_fast(x, y, z, freq=140.0, octaves=3, seed=23)
    n_pore = fbm_3d_fast(x, y, z, freq=520.0, octaves=2, seed=59)

    # ------------------------------------------------------------------
    # 1. BASE SKIN ALBEDO (Fair luminous warm porcelain-peach like reference.png)
    # ------------------------------------------------------------------
    # Calibrated sRGB base (~0.848, 0.678, 0.600) so soft daylight renders warm, fair & lifelike (never chalky!)
    r = np.full((size, size), 0.848, dtype=np.float32)
    g = np.full((size, size), 0.678, dtype=np.float32)
    b = np.full((size, size), 0.600, dtype=np.float32)

    # Subtle natural skin color micro-variation (prevents CG plastic uniformity)
    mottle = (n_macro - 0.5) * 0.036 + (n_mid - 0.5) * 0.018
    r += mottle * 0.7
    g += mottle * 1.0 - (n_macro - 0.5) * 0.012
    b += mottle * 1.1

    # Subtle warm ivory brightness on center forehead, nose bridge, and cheeks
    tzone = smoothstep(-0.135, -0.155, y) * np.exp(-0.5 * ((ax / 0.042) ** 2 + ((z - 1.545) / 0.055) ** 2))
    r += 0.024 * tzone
    g += 0.025 * tzone
    b += 0.020 * tzone

    # Very gentle natural neck & jaw warmth / subtle shadow tone
    neck_zone = smoothstep(1.465, 1.385, z)
    r -= 0.032 * neck_zone
    g -= 0.042 * neck_zone
    b -= 0.044 * neck_zone

    # ------------------------------------------------------------------
    # 2. SOFT ROSY PEACH BLUSH (Cheek apples, subtle nose tip, chin, ear rims)
    # ------------------------------------------------------------------
    front_face = smoothstep(-0.105, -0.135, y)
    blush_cheek = front_face * np.exp(
        -0.5 * (((ax - 0.036) / 0.022) ** 2 + ((z - 1.512) / 0.017) ** 2)
    )
    blush_nose = smoothstep(-0.152, -0.163, y) * np.exp(
        -0.5 * ((ax / 0.011) ** 2 + ((z - 1.509) / 0.008) ** 2)
    )
    blush_chin = smoothstep(-0.138, -0.149, y) * np.exp(
        -0.5 * ((ax / 0.015) ** 2 + ((z - 1.452) / 0.010) ** 2)
    )
    blush_ear = smoothstep(0.068, 0.080, ax) * smoothstep(1.495, 1.525, z) * smoothstep(1.585, 1.555, z)

    blush = np.clip(0.68 * blush_cheek + 0.34 * blush_nose + 0.18 * blush_chin + 0.45 * blush_ear, 0.0, 1.0)
    r = r * (1.0 - 0.12 * blush) + 0.92 * (0.12 * blush)
    g = g * (1.0 - 0.24 * blush) + 0.54 * (0.24 * blush)
    b = b * (1.0 - 0.26 * blush) + 0.50 * (0.26 * blush)

    # ------------------------------------------------------------------
    # 3. ORBITAL SHADING, DOUBLE-EYELID CREASE, UPPER LASH TIGHTLINE & BROW SHADOW
    # ------------------------------------------------------------------
    for side, ec in [(1.0, eye_c_l), (-1.0, eye_c_r)]:
        dx = x - ec[0]
        sx = side * dx
        dz = z - ec[2]
        efront = smoothstep(ec[1] + 0.002, ec[1] - 0.011, y)

        # Soft warm peach-brown eyeshadow glow across upper eyelid & crease
        shadow_z = ec[2] + 0.0080 - 0.0020 * ((sx - 0.001) / 0.013) ** 2
        eyeshadow = efront * np.exp(
            -0.5 * (((sx - 0.0015) / 0.0135) ** 2 + ((z - shadow_z) / 0.0046) ** 2)
        )
        r = r * (1.0 - 0.18 * eyeshadow) + 0.64 * (0.18 * eyeshadow)
        g = g * (1.0 - 0.24 * eyeshadow) + 0.44 * (0.24 * eyeshadow)
        b = b * (1.0 - 0.26 * eyeshadow) + 0.38 * (0.26 * eyeshadow)

        # Crisp crease line accent right in the double-eyelid fold
        crease_z = ec[2] + 0.0094 - 0.0020 * ((sx - 0.001) / 0.013) ** 2
        crease_line = efront * np.exp(
            -0.5 * (((sx - 0.001) / 0.0125) ** 2 + ((z - crease_z) / 0.0011) ** 2)
        )
        r *= (1.0 - 0.20 * crease_line)
        g *= (1.0 - 0.27 * crease_line)
        b *= (1.0 - 0.30 * crease_line)

        # Natural dark espresso lash-line tightline along upper lid margin
        upper_rim_z = ec[2] + 0.0058 - 0.0026 * ((sx - 0.001) / 0.013) ** 2
        tightline = (
            smoothstep(ec[1] - 0.006, ec[1] - 0.012, y)
            * np.exp(-0.5 * (((sx - 0.0015) / 0.0125) ** 2 + ((z - upper_rim_z) / 0.0014) ** 2))
        )
        r = r * (1.0 - 0.60 * tightline) + 0.18 * (0.60 * tightline)
        g = g * (1.0 - 0.64 * tightline) + 0.11 * (0.64 * tightline)
        b = b * (1.0 - 0.66 * tightline) + 0.09 * (0.66 * tightline)

        # Soft warm aegyo-sal glow & subtle under-line
        aegyo_z = ec[2] - 0.0088 + 0.0015 * (sx / 0.012) ** 2
        aegyo_glow = efront * np.exp(
            -0.5 * (((sx - 0.0005) / 0.011) ** 2 + ((z - aegyo_z) / 0.0022) ** 2)
        )
        r += 0.025 * aegyo_glow
        g += 0.010 * aegyo_glow
        b += 0.008 * aegyo_glow

        under_z = aegyo_z - 0.0032
        under_s = efront * np.exp(
            -0.5 * (((sx - 0.0005) / 0.011) ** 2 + ((z - under_z) / 0.0014) ** 2)
        )
        r *= (1.0 - 0.09 * under_s)
        g *= (1.0 - 0.13 * under_s)
        b *= (1.0 - 0.14 * under_s)

        # Inner socket deep occlusion
        deep_socket = (
            smoothstep(ec[1] - 0.009, ec[1] - 0.002, y)
            * smoothstep(ec[1] + 0.020, ec[1] + 0.004, y)
            * np.exp(-0.5 * ((dx / 0.016) ** 2 + (dz / 0.014) ** 2))
        )
        r *= (1.0 - 0.60 * deep_socket)
        g *= (1.0 - 0.68 * deep_socket)
        b *= (1.0 - 0.70 * deep_socket)

        # Soft feathered eyebrow skin undertone sitting ~16.8mm above eye center
        brow_t = (sx + 0.0155) / 0.0415
        brow_arch_z = ec[2] + 0.0168 + 0.0020 * np.sin(np.clip(brow_t, 0.0, 1.0) * math.pi * 0.88) - 0.0018 * np.maximum(brow_t - 0.65, 0.0)
        brow_half_h = 0.0029 * (1.0 - 0.52 * np.clip(brow_t, 0.0, 1.0)) * smoothstep(-0.08, 0.14, brow_t) * smoothstep(1.06, 0.80, brow_t)
        brow_w = (
            efront
            * smoothstep(-0.05, 0.12, brow_t)
            * smoothstep(1.05, 0.85, brow_t)
            * np.exp(-0.5 * ((z - brow_arch_z) / (brow_half_h + 1e-4)) ** 2)
        )
        bt = 0.28 * brow_w * (0.75 + 0.25 * n_mid)
        r = r * (1.0 - bt) + 0.30 * bt
        g = g * (1.0 - bt) + 0.20 * bt
        b = b * (1.0 - bt) + 0.14 * bt

    # ------------------------------------------------------------------
    # 4. SHAPED NATURAL CORAL-ROSE LIPS (matching sculpted closed smile)
    # ------------------------------------------------------------------
    # Exact lip seam curve after sculpt.py's smile lift
    lip_base_z = 1.4816
    smile_curve = 0.0052 * smoothstep(0.004, 0.028, ax) * np.exp(-0.5 * ((ax - 0.027) / 0.018) ** 2)
    z_seam = lip_base_z + smile_curve

    # Lip horizontal envelope (half-width ~24mm)
    lip_x_env = smoothstep(0.0275, 0.0205, ax)
    # Upper lip silhouette with gentle Cupid's bow peaks around |x| = 0.0055
    cupid = 0.0011 * np.exp(-0.5 * ((ax - 0.0055) / 0.0040) ** 2) - 0.0005 * np.exp(-0.5 * (ax / 0.0030) ** 2)
    upper_h = (0.0064 * np.sqrt(np.clip(1.0 - (ax / 0.0265) ** 2, 0.0, 1.0)) + cupid) * lip_x_env
    # Fuller pillowy lower lip
    lower_h = (0.0092 * np.sqrt(np.clip(1.0 - (ax / 0.0255) ** 2, 0.0, 1.0))) * lip_x_env

    dz_lip = z - z_seam
    lip_front = smoothstep(-0.145, -0.153, y)
    # Upper & lower lip masks with soft natural feathered vermilion edge
    upper_mask = lip_front * smoothstep(-0.0015, 0.0008, dz_lip) * smoothstep(upper_h + 0.0016, upper_h - 0.0012, dz_lip)
    lower_mask = lip_front * smoothstep(0.0015, -0.0008, dz_lip) * smoothstep(-(lower_h + 0.0018), -(lower_h - 0.0014), dz_lip)
    # Center band across seam so seam itself is rich colored
    seam_band = lip_front * lip_x_env * np.exp(-0.5 * (dz_lip / 0.0022) ** 2)
    lip_mask = np.clip(np.maximum(np.maximum(upper_mask, lower_mask), seam_band) * smoothstep(0.0285, 0.0225, ax), 0.0, 1.0)

    # Lip color gradient: warm coral-rose with slightly deeper inner parting line & soft peachy center
    inner_line = lip_front * lip_x_env * np.exp(-0.5 * (dz_lip / 0.0013) ** 2)
    lip_r = 0.875 - 0.14 * inner_line + 0.02 * (n_mid - 0.5)
    lip_g = 0.475 - 0.16 * inner_line + 0.02 * (n_mid - 0.5)
    lip_b = 0.445 - 0.15 * inner_line + 0.02 * (n_mid - 0.5)

    r = r * (1.0 - lip_mask) + lip_r * lip_mask
    g = g * (1.0 - lip_mask) + lip_g * lip_mask
    b = b * (1.0 - lip_mask) + lip_b * lip_mask

    # Soft mouth-corner smile crease darkening (tiny realistic shadow right at smiling corners)
    corner_sh = lip_front * np.exp(-0.5 * (((ax - 0.0255) / 0.0038) ** 2 + ((z - (z_seam + 0.0005)) / 0.0025) ** 2))
    r *= (1.0 - 0.16 * corner_sh)
    g *= (1.0 - 0.22 * corner_sh)
    b *= (1.0 - 0.24 * corner_sh)

    # ------------------------------------------------------------------
    # 5. NOSTRIL CAVITIES & EAR CONCHA OCCLUSION
    # ------------------------------------------------------------------
    nostril_occ = (
        smoothstep(-0.150, -0.159, y)
        * smoothstep(-0.165, -0.159, y)
        * np.exp(-0.5 * (((ax - 0.0065) / 0.0036) ** 2 + ((z - 1.5005) / 0.0026) ** 2))
    )
    r *= (1.0 - 0.42 * nostril_occ)
    g *= (1.0 - 0.52 * nostril_occ)
    b *= (1.0 - 0.55 * nostril_occ)

    # Subtle ear fold AO
    ear_occ = (
        smoothstep(0.064, 0.072, ax)
        * smoothstep(0.080, 0.074, ax)
        * np.exp(-0.5 * (((y + 0.048) / 0.012) ** 2 + ((z - 1.535) / 0.018) ** 2))
    )
    r *= (1.0 - 0.18 * ear_occ)
    g *= (1.0 - 0.24 * ear_occ)
    b *= (1.0 - 0.26 * ear_occ)

    # Assemble Albedo RGBA
    albedo = np.stack([
        np.clip(r, 0.0, 1.0),
        np.clip(g, 0.0, 1.0),
        np.clip(b, 0.0, 1.0),
        np.ones((size, size), dtype=np.float32),
    ], axis=-1)

    # ------------------------------------------------------------------
    # 6. SKIN ROUGHNESS MAP (Non-Color)
    # ------------------------------------------------------------------
    # Youthful satin-dewy skin: 0.36 on nose/cheeks/forehead, 0.44 on neck/jaw, 0.17 on glossy-satin lips
    rough = 0.43 - 0.07 * tzone - 0.04 * blush_cheek + 0.03 * (n_mid - 0.5)
    # Slight sheen on nose tip & upper cheeks
    nose_sheen = smoothstep(-0.155, -0.164, y) * np.exp(-0.5 * ((ax / 0.010) ** 2 + ((z - 1.512) / 0.012) ** 2))
    rough -= 0.07 * nose_sheen
    # Glossy-satin lips (matches reference's natural moist lip highlight!)
    rough = rough * (1.0 - lip_mask) + (0.175 + 0.03 * (n_pore - 0.5)) * lip_mask
    rough = np.clip(rough, 0.12, 0.65)
    rough_rgba = np.stack([rough, rough, rough, np.ones_like(rough)], axis=-1)

    # ------------------------------------------------------------------
    # 7. RESTRAINED SKIN MICRO-BUMP / PORES & DELICATE LIP LINES (Non-Color)
    # ------------------------------------------------------------------
    # Keep pores very subtle/youthful so face looks fresh and photographic, never orange-peel
    pore_zone = smoothstep(-0.130, -0.155, y) * np.exp(-0.5 * ((ax / 0.040) ** 2 + ((z - 1.515) / 0.035) ** 2))
    skin_grain = (n_pore - 0.5) * (0.28 + 0.32 * pore_zone) + (n_mid - 0.5) * 0.14
    # Fine vertical lip micro-lines
    lip_creases = np.sin(x * 1650.0 + n_mid * 3.5) * 0.25 * lip_mask
    bump = np.clip(0.5 + skin_grain * (1.0 - 0.7 * lip_mask) + lip_creases, 0.0, 1.0)
    bump_rgba = np.stack([bump, bump, bump, np.ones_like(bump)], axis=-1)

    # Lip gloss coat mask
    lip_rgba = np.stack([lip_mask, lip_mask, lip_mask, np.ones_like(lip_mask)], axis=-1)

    return {
        "skin_albedo": make_blender_image("Skin_Albedo", albedo, colorspace="sRGB"),
        "skin_roughness": make_blender_image("Skin_Roughness", rough_rgba, colorspace="Non-Color"),
        "skin_bump": make_blender_image("Skin_Bump", bump_rgba, colorspace="Non-Color"),
        "skin_lipmask": make_blender_image("Skin_LipMask", lip_rgba, colorspace="Non-Color"),
    }


def generate_eye_iris_texture(size=1024):
    """
    Generate a photorealistic warm chestnut-brown iris & natural off-white sclera texture
    matching the polar UV layout of Eyeball_L/R (center = (0.5, 0.5), radius 0.495 = pi radians).
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    u = (xx + 0.5) / size - 0.5
    v = (yy + 0.5) / size - 0.5
    rad_uv = np.hypot(u, v)
    ang = (rad_uv / 0.495) * math.pi  # radians from -Y optical axis
    phi = np.arctan2(v, u)

    # Expressive warm dark-brown iris filling the natural eyelid opening
    r_pupil = math.radians(11.8)
    r_collarette = math.radians(18.0)
    r_iris = math.radians(31.8)
    r_limbal = math.radians(33.5)

    # Radial & angular iris fiber noise (seamless periodic in phi!)
    c1, s1 = np.cos(phi), np.sin(phi)
    fibers = np.zeros_like(ang)
    rng = np.random.default_rng(42)
    for k in range(1, 28):
        freq = k * 3 + rng.integers(0, 3)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amp = 1.0 / (0.6 + 0.14 * k)
        wiggle = 0.08 * np.sin(ang * 34.0 + phase * 1.7)
        fibers += amp * np.sin(freq * (phi + wiggle) + phase)
    fibers = fibers / (np.max(np.abs(fibers)) + 1e-6)

    n_fine = hash_noise_2d(c1 * 18.0 + ang * 45.0, s1 * 18.0 + ang * 45.0, seed=101)
    n_sclera = hash_noise_2d(u * 16.0, v * 16.0, seed=202)

    # --- Sclera (soft natural warm off-white) ---
    scl_r = 0.88 - 0.05 * smoothstep(r_limbal, math.radians(65.0), ang) + 0.015 * (n_sclera - 0.5)
    scl_g = 0.85 - 0.11 * smoothstep(r_limbal, math.radians(65.0), ang) + 0.015 * (n_sclera - 0.5)
    scl_b = 0.82 - 0.14 * smoothstep(r_limbal, math.radians(65.0), ang) + 0.015 * (n_sclera - 0.5)

    vein = np.exp(-0.5 * ((np.sin(phi * 11.0 + n_sclera * 5.0)) / 0.10) ** 2) * smoothstep(math.radians(36.0), math.radians(58.0), ang)
    scl_r = scl_r * (1.0 - 0.14 * vein) + 0.82 * (0.14 * vein)
    scl_g = scl_g * (1.0 - 0.20 * vein) + 0.52 * (0.20 * vein)
    scl_b = scl_b * (1.0 - 0.20 * vein) + 0.50 * (0.20 * vein)

    # --- Deep Soulful Warm Espresso-Chestnut Iris (matching reference.png!) ---
    t_iris = np.clip((ang - r_pupil) / (r_iris - r_pupil + 1e-6), 0.0, 1.0)
    # Subtle warm honey-chestnut bounce on lower half of iris (natural optical caustics!)
    lower_glow = smoothstep(0.02, -0.09, v) * 0.09
    ir_r = 0.21 + lower_glow + 0.09 * np.exp(-0.5 * ((t_iris - 0.35) / 0.24) ** 2) - 0.13 * smoothstep(0.68, 1.0, t_iris)
    ir_g = 0.10 + lower_glow * 0.55 + 0.05 * np.exp(-0.5 * ((t_iris - 0.33) / 0.22) ** 2) - 0.07 * smoothstep(0.68, 1.0, t_iris)
    ir_b = 0.055 + lower_glow * 0.25 + 0.025 * np.exp(-0.5 * ((t_iris - 0.30) / 0.20) ** 2) - 0.035 * smoothstep(0.68, 1.0, t_iris)

    fiber_mod = 0.065 * fibers + 0.035 * (n_fine - 0.5)
    ir_r = np.clip(ir_r + fiber_mod * 1.15, 0.015, 0.58)
    ir_g = np.clip(ir_g + fiber_mod * 0.72, 0.010, 0.38)
    ir_b = np.clip(ir_b + fiber_mod * 0.38, 0.008, 0.24)

    # Subtle upper-lid ambient shadow baked gently on top of iris
    top_shade = smoothstep(0.01, 0.14, v) * 0.32
    ir_r *= (1.0 - top_shade)
    ir_g *= (1.0 - top_shade)
    ir_b *= (1.0 - top_shade)

    # Rich dark espresso limbal ring
    limbal_w = np.exp(-0.5 * ((ang - r_iris) / math.radians(2.1)) ** 2)
    ir_r = ir_r * (1.0 - 0.78 * limbal_w) + 0.045 * (0.78 * limbal_w)
    ir_g = ir_g * (1.0 - 0.80 * limbal_w) + 0.025 * (0.80 * limbal_w)
    ir_b = ir_b * (1.0 - 0.82 * limbal_w) + 0.018 * (0.82 * limbal_w)

    # Blend iris into sclera cleanly at limbal border
    iris_mask = smoothstep(r_limbal + math.radians(1.0), r_iris - math.radians(0.6), ang)
    r = scl_r * (1.0 - iris_mask) + ir_r * iris_mask
    g = scl_g * (1.0 - iris_mask) + ir_g * iris_mask
    b = scl_b * (1.0 - iris_mask) + ir_b * iris_mask

    # Crisp dark pupil with ultra-soft 0.7 deg feathered edge
    pupil_mask = smoothstep(r_pupil + math.radians(0.7), r_pupil - math.radians(0.5), ang)
    r = r * (1.0 - pupil_mask) + 0.015 * pupil_mask
    g = g * (1.0 - pupil_mask) + 0.012 * pupil_mask
    b = b * (1.0 - pupil_mask) + 0.012 * pupil_mask

    rgba = np.stack([
        np.clip(r, 0.0, 1.0),
        np.clip(g, 0.0, 1.0),
        np.clip(b, 0.0, 1.0),
        np.ones((size, size), dtype=np.float32),
    ], axis=-1)
    return make_blender_image("Eye_Iris_Albedo", rgba, colorspace="sRGB")


def generate_knit_textures(size=1024):
    """
    Generate seamless tileable 1024x1024 chunky stockinette wool knit maps + ribbed collar knit map:
      - Knit_Height (Non-Color): deep 3D interlocking V-stitch yarn loops with twisted 2-ply striations.
      - Knit_Color (sRGB): dusty rose/blush-pink melange yarn with subtle crevice AO and warm fiber mottling.
      - Knit_Rib_Height (Non-Color): chunky 1x1 ribbed knit for collar, cuffs, and hem.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    u = xx / size  # horizontal (columns / wales)
    v = yy / size  # vertical (rows / courses)

    # Tile counts per UV unit: 24 wales wide x 32 courses tall per repeat tile
    nw = 24.0
    nc = 32.0
    wu = (u * nw) % 1.0  # [0, 1) within a single knit stitch column
    cv = (v * nc) % 1.0  # [0, 1) within a single knit stitch row

    # Stockinette V-stitch geometry:
    # Left half of stitch (wu < 0.5): diagonal yarn leg leaning right as v goes up
    # Right half of stitch (wu >= 0.5): diagonal yarn leg leaning left as v goes up
    # Distance to left leg & right leg (with vertical wrap for interlocking loops)
    # Left leg center line: x_l(v) = 0.25 + 0.14 * (v - 0.5)  (forming a 'V' pointing down)
    h_stitch = np.zeros((size, size), dtype=np.float32)
    for dv in (-1.0, 0.0, 1.0):
        vv = cv + dv
        # Left leg of V-loop
        xl = 0.26 + 0.18 * (vv - 0.45)
        dl = np.abs(wu - xl)
        leg_l = np.clip(1.0 - (dl / 0.21) ** 2, 0.0, 1.0) * np.exp(-0.5 * ((vv - 0.48) / 0.42) ** 2)
        # Twist striations along left yarn strand
        twist_l = 0.5 + 0.5 * np.sin((vv * 5.0 + wu * 8.0) * 2.0 * math.pi)
        leg_l = np.sqrt(np.maximum(leg_l, 0.0)) * (0.86 + 0.14 * twist_l)

        # Right leg of V-loop
        xr = 0.74 - 0.18 * (vv - 0.45)
        dr = np.abs(wu - xr)
        leg_r = np.clip(1.0 - (dr / 0.21) ** 2, 0.0, 1.0) * np.exp(-0.5 * ((vv - 0.48) / 0.42) ** 2)
        twist_r = 0.5 + 0.5 * np.sin((vv * 5.0 - wu * 8.0) * 2.0 * math.pi)
        leg_r = np.sqrt(np.maximum(leg_r, 0.0)) * (0.86 + 0.14 * twist_r)

        h_stitch = np.maximum(h_stitch, np.maximum(leg_l, leg_r))

    # Add subtle column vertical rib undulation & fuzzy wool micro-fiber noise
    # Tileable noise using periodic trig coordinates
    cu, su = np.cos(2.0 * math.pi * u), np.sin(2.0 * math.pi * u)
    cvp, svp = np.cos(2.0 * math.pi * v), np.sin(2.0 * math.pi * v)
    n_yarn = hash_noise_2d(cu * 14.0 + cvp * 5.0, su * 14.0 + svp * 5.0, seed=303)
    n_fuzz = hash_noise_2d(cu * 55.0 + svp * 17.0, su * 55.0 + cvp * 17.0, seed=404)

    knit_h = np.clip(0.12 + 0.78 * h_stitch + 0.06 * (n_yarn - 0.5) + 0.05 * (n_fuzz - 0.5), 0.0, 1.0)

    # Dusty blush-pink / rose-pink wool melange color (matching input/reference.png!)
    ao = 0.54 + 0.46 * knit_h
    melange = (n_yarn - 0.5) * 0.065 + (n_fuzz - 0.5) * 0.035
    kr = np.clip((0.855 + melange * 0.8) * ao, 0.0, 1.0)
    kg = np.clip((0.555 + melange * 1.0) * (ao ** 1.15), 0.0, 1.0)
    kb = np.clip((0.580 + melange * 1.0) * (ao ** 1.12), 0.0, 1.0)

    # Chunky 1x1 ribbed knit height (for crew-neck collar, cuffs, hem)
    # Alternating raised vertical knit ribs + purl troughs
    rib_wave = 0.5 + 0.5 * np.cos(u * 20.0 * 2.0 * math.pi)
    rib_h = np.clip(0.15 + 0.55 * (rib_wave ** 0.7) + 0.25 * h_stitch * (0.4 + 0.6 * rib_wave) + 0.05 * (n_fuzz - 0.5), 0.0, 1.0)

    h_rgba = np.stack([knit_h, knit_h, knit_h, np.ones_like(knit_h)], axis=-1)
    c_rgba = np.stack([kr, kg, kb, np.ones_like(kr)], axis=-1)
    rib_rgba = np.stack([rib_h, rib_h, rib_h, np.ones_like(rib_h)], axis=-1)

    return {
        "knit_height": make_blender_image("Knit_Height", h_rgba, colorspace="Non-Color"),
        "knit_color": make_blender_image("Knit_Color", c_rgba, colorspace="sRGB"),
        "knit_rib_height": make_blender_image("Knit_Rib_Height", rib_rgba, colorspace="Non-Color"),
    }
