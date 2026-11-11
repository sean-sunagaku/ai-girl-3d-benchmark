import os
import math
import numpy as np
import bpy


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save_rgba_png(filepath, rgba_float, is_data=False):
    """Save HxWx4 float32 [0..1] array as PNG and return a loaded, packed Blender Image."""
    h, w, _ = rgba_float.shape
    name = os.path.basename(filepath)
    for existing in list(bpy.data.images):
        if existing.name == name or existing.name.startswith(name + "."):
            bpy.data.images.remove(existing)

    tmp_img = bpy.data.images.new(name="_tmp_" + name, width=w, height=h, alpha=True, float_buffer=False)
    # Set Non-Color BEFORE writing pixels so save() writes exact 8-bit values without double-gamma
    tmp_img.colorspace_settings.name = 'Non-Color'
    flipped = np.clip(rgba_float[::-1, :, :], 0.0, 1.0).astype(np.float32)
    tmp_img.pixels.foreach_set(flipped.ravel())
    tmp_img.filepath_raw = filepath
    tmp_img.file_format = 'PNG'
    tmp_img.save()
    bpy.data.images.remove(tmp_img)

    img = bpy.data.images.load(filepath)
    img.name = name
    img.colorspace_settings.name = 'Non-Color' if is_data else 'sRGB'
    img.pack()
    return img


def _tileable_noise(res, freq_x, freq_y, seed=0):
    """Generate smooth tileable 2D periodic Perlin-like harmonic noise in [0, 1]."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)[:, None]
    x = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)[None, :]
    acc = np.zeros((res, res), dtype=np.float32)
    for k in range(10):
        fx = rng.integers(1, max(2, freq_x + 1)) * rng.choice([-1, 1])
        fy = rng.integers(1, max(2, freq_y + 1)) * rng.choice([-1, 1])
        ph = rng.uniform(0, 2.0 * np.pi)
        amp = 1.0 / (1.0 + 0.25 * math.hypot(fx / max(1, freq_x), fy / max(1, freq_y)))
        acc += amp * np.sin(fx * x + fy * y + ph)
    acc = (acc - acc.min()) / max(1e-6, (acc.max() - acc.min()))
    return acc


def generate_iris_textures(out_dir, res=1024):
    """Generate warm chestnut-brown human iris albedo and stroma bump maps."""
    u = np.linspace(-1.0, 1.0, res, dtype=np.float32)[None, :]
    v = np.linspace(-1.0, 1.0, res, dtype=np.float32)[:, None]
    r = np.hypot(u, v)
    theta = np.arctan2(v, u)

    # Pupil ~0.31 of iris radius; collarette ~0.46; limbal ring ~0.88..1.0
    pupil_r = 0.305
    pupil_mask = np.clip((r - pupil_r) / 0.022, 0.0, 1.0)

    # Multi-scale wavy radial stroma fibers
    warp = 0.035 * np.sin(7.0 * theta + 5.0 * r) + 0.018 * np.cos(19.0 * theta - 9.0 * r)
    th_w = theta + warp
    f1 = 0.5 + 0.5 * np.sin(68.0 * th_w + 1.5 * np.sin(13.0 * theta))
    f2 = 0.5 + 0.5 * np.sin(142.0 * (th_w + 0.012 * np.sin(18.0 * r)) + 0.8)
    f3 = 0.5 + 0.5 * np.cos(256.0 * th_w - 3.0 * r)
    stroma = 0.48 * f1 + 0.34 * f2 + 0.18 * f3

    # Collarette ring around r=0.45 with organic zig-zag
    col_r = 0.455 + 0.026 * np.sin(16.0 * theta) + 0.014 * np.cos(29.0 * theta + 1.1)
    collarette = np.exp(-((r - col_r) / 0.068) ** 2)

    # Subtle radial crypts (darker pockets between fibers)
    crypt_ang = 0.5 + 0.5 * np.sin(23.0 * theta + 2.1 * np.cos(9.0 * theta))
    crypt_rad = np.exp(-((r - 0.56) / 0.18) ** 2)
    crypts = (crypt_ang ** 3) * crypt_rad

    # Outer dark brown limbal ring
    limbal = np.clip((r - 0.81) / 0.17, 0.0, 1.0) ** 1.5

    # Base warm chestnut-amber colors (in sRGB space, saved as sRGB PNG)
    # Rich dark chestnut base -> warm amber-brown collarette -> deep espresso limbal ring
    base_rgb = np.zeros((res, res, 3), dtype=np.float32)
    c_mid = np.array([0.355, 0.205, 0.125], dtype=np.float32)      # rich chestnut
    c_amber = np.array([0.525, 0.320, 0.185], dtype=np.float32)    # warm golden-chestnut highlights
    c_crypt = np.array([0.205, 0.110, 0.068], dtype=np.float32)    # deep cocoa crypts
    c_limbal = np.array([0.115, 0.065, 0.045], dtype=np.float32)   # soft dark limbal ring
    c_pupil = np.array([0.018, 0.014, 0.014], dtype=np.float32)    # velvet pupil

    fiber_mix = np.clip(0.35 * stroma + 0.52 * collarette * (0.55 + 0.45 * f1), 0.0, 1.0)
    iris_rgb = c_mid[None, None, :] * (1.0 - fiber_mix[..., None]) + c_amber[None, None, :] * fiber_mix[..., None]
    iris_rgb = iris_rgb * (1.0 - 0.42 * crypts[..., None]) + c_crypt[None, None, :] * (0.42 * crypts[..., None])
    iris_rgb = iris_rgb * (1.0 - 0.82 * limbal[..., None]) + c_limbal[None, None, :] * (0.82 * limbal[..., None])

    # Subtle radial gradient (slightly darker near pupil edge and outer rim)
    inner_shade = 0.82 + 0.18 * np.clip((r - pupil_r) / 0.12, 0.0, 1.0)
    iris_rgb *= inner_shade[..., None]

    # Combine pupil and iris, fade outside r=1.0 into warm sclera edge
    final_rgb = c_pupil[None, None, :] * (1.0 - pupil_mask[..., None]) + iris_rgb * pupil_mask[..., None]
    sclera_edge = np.clip((r - 0.98) / 0.04, 0.0, 1.0)
    c_sclera = np.array([0.92, 0.89, 0.87], dtype=np.float32)
    final_rgb = final_rgb * (1.0 - sclera_edge[..., None]) + c_sclera[None, None, :] * sclera_edge[..., None]

    albedo_rgba = np.concatenate([final_rgb, np.ones((res, res, 1), dtype=np.float32)], axis=-1)
    bump_val = np.clip((0.55 * stroma + 0.35 * collarette - 0.30 * crypts) * pupil_mask * (1.0 - limbal * 0.7), 0.0, 1.0)
    bump_rgba = np.stack([bump_val, bump_val, bump_val, np.ones_like(bump_val)], axis=-1)

    img_albedo = _save_rgba_png(os.path.join(out_dir, "iris_albedo.png"), albedo_rgba, is_data=False)
    img_bump = _save_rgba_png(os.path.join(out_dir, "iris_bump.png"), bump_rgba, is_data=True)
    return img_albedo, img_bump


def generate_skin_micro_detail(out_dir, res=1024):
    """Generate seamless tileable human skin micro-pore and polygonal skin-cell relief map."""
    rng = np.random.default_rng(42)
    # Periodic grid of jittered Voronoi cell centers for skin micro-relief
    grid_n = 48
    gy, gx = np.meshgrid(np.arange(grid_n), np.arange(grid_n), indexing='ij')
    pts_x = (gx + rng.uniform(0.18, 0.82, size=(grid_n, grid_n))).astype(np.float32)
    pts_y = (gy + rng.uniform(0.18, 0.82, size=(grid_n, grid_n))).astype(np.float32)

    py = np.linspace(0.0, grid_n, res, endpoint=False, dtype=np.float32)[:, None]
    px = np.linspace(0.0, grid_n, res, endpoint=False, dtype=np.float32)[None, :]
    ix = np.floor(px).astype(np.int32)
    iy = np.floor(py).astype(np.int32)

    d1 = np.full((res, res), 1e9, dtype=np.float32)
    d2 = np.full((res, res), 1e9, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx = (ix + dx) % grid_n
            ny = (iy + dy) % grid_n
            cx = pts_x[ny, nx] + ((ix + dx) - nx)
            cy = pts_y[ny, nx] + ((iy + dy) - ny)
            dist = np.hypot(px - cx, py - cy)
            swap = dist < d1
            d2 = np.where(swap, d1, np.minimum(d2, dist))
            d1 = np.where(swap, dist, d1)

    # Soft pillowy skin micro-cells + tiny pore dimples at cell centers
    edge_groove = np.clip((d2 - d1) / 0.28, 0.0, 1.0) ** 0.55
    pore_dimple = 1.0 - 0.55 * np.exp(-((d1 / 0.13) ** 2))
    macro_var = _tileable_noise(res, 6, 6, seed=101)
    height = np.clip(0.55 * edge_groove * pore_dimple + 0.25 * macro_var + 0.15, 0.0, 1.0)

    # Pack height in R, micro-roughness variation in G, subtle pore AO in B
    rough_mod = np.clip(0.5 + 0.28 * (1.0 - edge_groove) + 0.18 * (macro_var - 0.5), 0.0, 1.0)
    ao_mod = np.clip(0.85 + 0.15 * height, 0.0, 1.0)
    rgba = np.stack([height, rough_mod, ao_mod, np.ones_like(height)], axis=-1)
    return _save_rgba_png(os.path.join(out_dir, "skin_micro_detail.png"), rgba, is_data=True)


def generate_sweater_knit_textures(out_dir, res=1024):
    """
    Generate realistic tileable Fisherman's Rib / Brioche Knit maps:
    - Height map (for real 3D mesh displacement + bump)
    - Normal map (for crisp 2-ply yarn twist & stitch crevices)
    - Albedo map (soft dusty-pink yarn with realistic stitch-crevice self-shadowing)
    """
    # 16 vertical rib columns (wales) x 20 horizontal rows (courses) per UV tile
    n_wales = 16.0
    n_courses = 22.0

    u = np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32)[:, None]

    wx = u * n_wales
    wy = v * n_courses

    # Local coordinates within each rib column: s in [-0.5, 0.5], row phase t in [0, 1)
    col_idx = np.floor(wx)
    s = (wx - col_idx) - 0.5

    # Fisherman's rib V-stitch: left leg (s < 0) slopes up-left, right leg (s > 0) slopes up-right
    # Leg center sits around |s| = 0.185, creating plump twin yarn strands forming each vertical wale
    abs_s = np.abs(s)
    leg_center = 0.175
    leg_width = 0.155

    # V-chevron vertical shift along the wale
    v_shift = wy - 1.35 * abs_s
    loop_arch = 0.5 + 0.5 * np.cos(2.0 * np.pi * v_shift)

    # Cross-sectional roundness of the two yarn legs of the V-stitch
    leg_profile = np.clip(1.0 - ((abs_s - leg_center) / leg_width) ** 2, 0.0, 1.0) ** 0.65
    # Slight central Valley between left and right leg of the same V-stitch
    center_pinch = 1.0 - 0.22 * np.exp(-((s / 0.048) ** 2))
    wale_height = leg_profile * center_pinch * (0.76 + 0.24 * loop_arch)

    # Horizontal purl / tuck stitches visible in the groove between vertical rib columns (|s| > 0.28)
    groove_mask = np.clip((abs_s - 0.24) / 0.14, 0.0, 1.0)
    purl_phase = wy + 0.5 * (col_idx % 2.0) * 0.0  # aligned horizontal rib bridges like reference.png!
    purl_bar = (0.5 + 0.5 * np.cos(2.0 * np.pi * purl_phase)) ** 0.75
    purl_height = 0.52 * groove_mask * (0.28 + 0.72 * purl_bar)

    # 2-ply diagonal yarn twist micro-ridges along each strand
    sign_s = np.where(s >= 0.0, 1.0, -1.0)
    ply_angle = 18.0 * (s * sign_s) + 3.0 * v_shift
    yarn_ply = 0.5 + 0.5 * np.sin(2.0 * np.pi * ply_angle)

    # Fine fuzzy wool fiber noise (tileable)
    fuzz = _tileable_noise(res, 32, 32, seed=77)
    macro = _tileable_noise(res, 4, 4, seed=88)

    raw_h = np.maximum(wale_height, purl_height)
    height = np.clip(0.84 * raw_h + 0.09 * yarn_ply * (raw_h > 0.2) + 0.07 * fuzz, 0.0, 1.0)

    # Compute tangent-space Normal map from height (periodic wrap)
    strength = 4.2
    dh_du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * strength
    dh_dv = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * strength
    nx = -dh_du
    ny = -dh_dv
    nz = np.ones_like(height)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    norm_rgb = np.stack([0.5 + 0.5 * nx * inv_len, 0.5 + 0.5 * ny * inv_len, 0.5 + 0.5 * nz * inv_len], axis=-1)
    norm_rgba = np.concatenate([norm_rgb, np.ones((res, res, 1), dtype=np.float32)], axis=-1)

    # Dusty rose / soft blush-pink yarn albedo with stitch-crevice warm AO
    # Reference sweater sRGB ~ (0.875, 0.645, 0.640) on crests, (0.695, 0.455, 0.455) in deep purl crevices
    ao = np.clip(0.28 + 0.72 * (raw_h ** 0.72), 0.0, 1.0)
    c_crest = np.array([0.895, 0.665, 0.658], dtype=np.float32)
    c_trough = np.array([0.625, 0.392, 0.398], dtype=np.float32)
    yarn_var = 0.96 + 0.08 * (macro - 0.5) + 0.04 * (fuzz - 0.5)
    albedo_rgb = (c_trough[None, None, :] * (1.0 - ao[..., None]) + c_crest[None, None, :] * ao[..., None]) * yarn_var[..., None]
    albedo_rgba = np.concatenate([np.clip(albedo_rgb, 0.0, 1.0), np.ones((res, res, 1), dtype=np.float32)], axis=-1)

    height_rgba = np.stack([height, ao, fuzz, np.ones_like(height)], axis=-1)

    img_h = _save_rgba_png(os.path.join(out_dir, "sweater_knit_height.png"), height_rgba, is_data=True)
    img_n = _save_rgba_png(os.path.join(out_dir, "sweater_knit_normal.png"), norm_rgba, is_data=True)
    img_a = _save_rgba_png(os.path.join(out_dir, "sweater_knit_albedo.png"), albedo_rgba, is_data=False)
    return img_h, img_n, img_a


def generate_cafe_wood_texture(out_dir, res=512):
    """Generate warm honey-oak wood grain texture for cafe tables, chairs, and shelves."""
    u = np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, res, endpoint=False, dtype=np.float32)[:, None]
    n1 = _tileable_noise(res, 3, 12, seed=21)
    n2 = _tileable_noise(res, 8, 24, seed=22)
    rings = 0.5 + 0.5 * np.sin(2.0 * np.pi * (14.0 * u + 1.8 * n1 + 0.4 * v))
    grain = np.clip(0.65 * rings + 0.35 * n2, 0.0, 1.0)
    c_light = np.array([0.74, 0.55, 0.36], dtype=np.float32)
    c_dark = np.array([0.52, 0.35, 0.20], dtype=np.float32)
    rgb = c_dark[None, None, :] * (1.0 - grain[..., None]) + c_light[None, None, :] * grain[..., None]
    rgba = np.concatenate([rgb, np.ones((res, res, 1), dtype=np.float32)], axis=-1)
    return _save_rgba_png(os.path.join(out_dir, "cafe_wood.png"), rgba, is_data=False)


def build_all_textures(work_dir):
    tex_dir = _ensure_dir(os.path.join(work_dir, "generated_textures"))
    iris_albedo, iris_bump = generate_iris_textures(tex_dir, res=1024)
    skin_detail = generate_skin_micro_detail(tex_dir, res=1024)
    knit_h, knit_n, knit_a = generate_sweater_knit_textures(tex_dir, res=1024)
    wood_a = generate_cafe_wood_texture(tex_dir, res=512)
    return {
        "iris_albedo": iris_albedo,
        "iris_bump": iris_bump,
        "skin_detail": skin_detail,
        "knit_height": knit_h,
        "knit_normal": knit_n,
        "knit_albedo": knit_a,
        "cafe_wood": wood_a,
    }
