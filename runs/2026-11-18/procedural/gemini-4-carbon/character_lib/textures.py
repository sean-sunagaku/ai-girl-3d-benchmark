"""Procedural <=2048px texture generators (iris, skin micro-pores, chunky knitwear)."""

import os
import bpy
import numpy as np


def _save_array_as_image(name: str, arr_rgba: np.ndarray, out_dir: str, colorspace: str = "sRGB"):
    """Create/replace a Blender image from an HxWx4 float32 [0..1] array, save to disk, and pack."""
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.abspath(os.path.join(out_dir, f"{name}.png"))
    h, w, c = arr_rgba.shape
    assert c == 4 and w <= 2048 and h <= 2048

    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])

    img = bpy.data.images.new(name=name, width=w, height=h, alpha=True, float_buffer=False)
    img.colorspace_settings.name = colorspace
    flat = np.clip(arr_rgba, 0.0, 1.0).astype(np.float32).ravel()
    img.pixels.foreach_set(flat)
    img.filepath_raw = filepath
    img.file_format = "PNG"
    img.save()
    img.pack()
    return img, filepath


def _fbm_tileable(h: int, w: int, freqs=(4, 8, 16, 32), seed: int = 42) -> np.ndarray:
    """Fast tileable band-limited multi-octave noise."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    yy = np.linspace(0.0, 2.0 * np.pi, h, endpoint=False, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 2.0 * np.pi, w, endpoint=False, dtype=np.float32)[None, :]
    total_w = 0.0
    for oct_idx, f in enumerate(freqs):
        amp = 0.5 ** oct_idx
        sub = np.zeros((h, w), dtype=np.float32)
        for _ in range(10):
            kx = int(rng.integers(-f, f + 1))
            ky = int(rng.integers(-f, f + 1))
            if kx == 0 and ky == 0:
                kx = f
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            sub += np.cos(kx * xx + ky * yy + phase)
        sub /= np.maximum(np.std(sub), 1e-5)
        acc += amp * sub
        total_w += amp
    acc /= total_w
    acc = (acc - acc.min()) / np.maximum(acc.max() - acc.min(), 1e-6)
    return acc


def generate_iris_textures(out_dir: str):
    """Generate 1024x1024 warm dark-chestnut iris albedo and radial stroma bump maps."""
    size = 1024
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin)
    r = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)

    rng = np.random.default_rng(108)
    fibers_fine = np.zeros_like(theta)
    fibers_coarse = np.zeros_like(theta)
    for k in range(1, 28):
        f1 = int(rng.integers(55, 190))
        ph1 = float(rng.uniform(0, 2 * np.pi))
        fibers_fine += (1.0 / (k ** 0.55)) * np.cos(f1 * theta + ph1 + 1.4 * np.sin(5 * theta + r * 3.0))
    for k in range(1, 16):
        f2 = int(rng.integers(12, 46))
        ph2 = float(rng.uniform(0, 2 * np.pi))
        fibers_coarse += (1.0 / (k ** 0.6)) * np.cos(f2 * theta + ph2)

    fibers_fine = (fibers_fine - fibers_fine.mean()) / (fibers_fine.std() + 1e-6)
    fibers_coarse = (fibers_coarse - fibers_coarse.mean()) / (fibers_coarse.std() + 1e-6)

    collarette_r = 0.47 + 0.028 * np.sin(11 * theta + 0.7) + 0.016 * np.cos(23 * theta - 1.2)
    collarette_ridge = np.exp(-((r - collarette_r) / 0.045) ** 2)

    crypt_wave = np.sin(17 * theta + 1.1) * np.cos(29 * theta - 0.5)
    crypt_mask = np.clip(crypt_wave, 0.0, 1.0) ** 2.2 * np.exp(-((r - 0.56) / 0.14) ** 2)

    pupil_r = 0.295 + 0.004 * np.sin(32 * theta)
    pupil_mask = np.clip((pupil_r + 0.012 - r) / 0.018, 0.0, 1.0)

    limbal_ring = np.exp(-((r - 0.885) / 0.072) ** 2)
    sclera_mask = np.clip((r - 0.915) / 0.052, 0.0, 1.0)
    sclera_mask = sclera_mask * sclera_mask * (3.0 - 2.0 * sclera_mask)

    # Rich East Asian warm dark chestnut brown iris + realistic soft off-white sclera (not glowing white!)
    deep_umber = np.array([0.12, 0.062, 0.038], dtype=np.float32)
    chestnut_mid = np.array([0.31, 0.172, 0.098], dtype=np.float32)
    warm_amber = np.array([0.46, 0.275, 0.148], dtype=np.float32)
    pupillary_gold = np.array([0.38, 0.215, 0.115], dtype=np.float32)
    limbal_dark = np.array([0.075, 0.042, 0.028], dtype=np.float32)
    sclera_col = np.array([0.74, 0.70, 0.68], dtype=np.float32)

    stroma_t = np.clip(0.5 + 0.22 * fibers_fine + 0.18 * fibers_coarse, 0.0, 1.0)
    radial_warmth = np.exp(-((r - 0.54) / 0.24) ** 2)

    rgb = (
        deep_umber[None, None, :] * (1.0 - stroma_t[:, :, None])
        + chestnut_mid[None, None, :] * stroma_t[:, :, None]
    )
    amber_boost = (0.36 * collarette_ridge + 0.26 * radial_warmth * stroma_t) * (1.0 - 0.65 * crypt_mask)
    rgb = rgb * (1.0 - amber_boost[:, :, None]) + warm_amber[None, None, :] * amber_boost[:, :, None]

    ruff = np.exp(-((r - 0.315) / 0.024) ** 2) * 0.42
    rgb = rgb * (1.0 - ruff[:, :, None]) + pupillary_gold[None, None, :] * ruff[:, :, None]
    rgb *= (1.0 - 0.42 * crypt_mask[:, :, None])

    limbal_w = np.clip(0.82 * limbal_ring, 0.0, 0.88)
    rgb = rgb * (1.0 - limbal_w[:, :, None]) + limbal_dark[None, None, :] * limbal_w[:, :, None]

    pupil_col = np.array([0.010, 0.008, 0.008], dtype=np.float32)
    rgb = rgb * (1.0 - pupil_mask[:, :, None]) + pupil_col[None, None, :] * pupil_mask[:, :, None]

    # Subtle sclera shadow toward upper/outer periphery so eyes sit softly inside lids
    sclera_ao = np.clip(1.0 - 0.18 * np.clip((r - 1.05) / 0.35, 0.0, 1.0), 0.78, 1.0)
    sclera_rgb = sclera_col[None, None, :] * sclera_ao[:, :, None]
    rgb = rgb * (1.0 - sclera_mask[:, :, None]) + sclera_rgb * sclera_mask[:, :, None]

    albedo_rgba = np.concatenate([rgb, np.ones((size, size, 1), dtype=np.float32)], axis=-1)

    iris_zone = (1.0 - pupil_mask) * (1.0 - sclera_mask)
    bump = np.clip(0.5 + (0.20 * (stroma_t - 0.5) + 0.22 * collarette_ridge - 0.20 * crypt_mask) * iris_zone, 0.0, 1.0)
    bump_rgba = np.stack([bump, bump, bump, np.ones_like(bump)], axis=-1)

    img_alb, path_alb = _save_array_as_image("iris_albedo_1k", albedo_rgba, out_dir, "sRGB")
    img_bmp, path_bmp = _save_array_as_image("iris_bump_1k", bump_rgba, out_dir, "Non-Color")
    return {"iris_albedo": (img_alb, path_alb), "iris_bump": (img_bmp, path_bmp)}


def generate_skin_micro_texture(out_dir: str):
    """Generate a restrained 2048x2048 tileable skin micro-pore & epidermal sulcus map."""
    size = 2048
    yy = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[None, :]

    fbm_low = _fbm_tileable(size, size, freqs=(6, 14, 28), seed=21)
    fbm_hi = _fbm_tileable(size, size, freqs=(48, 96, 160), seed=77)

    wx = xx * 128.0 + 1.35 * fbm_low
    wy = yy * 128.0 + 1.35 * np.roll(fbm_low, 257, axis=1)
    cx = np.cos(2.0 * np.pi * wx)
    cy = np.cos(2.0 * np.pi * wy)
    cxy = np.cos(2.0 * np.pi * (wx * 0.5 + wy * 0.866))
    pores = np.clip((cx + cy + cxy) / 3.0, 0.0, 1.0) ** 2.6

    langer1 = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx * 180.0 + yy * 140.0 + 0.9 * fbm_low))
    langer2 = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx * 150.0 - yy * 190.0 + 0.9 * fbm_hi))
    sulci = (langer1 * langer2) ** 1.6

    height = 0.58 - 0.24 * pores - 0.08 * (1.0 - sulci) + 0.12 * (fbm_hi - 0.5)
    rough_var = 0.50 + 0.22 * (fbm_low - 0.5) + 0.16 * pores
    subsurf_var = 0.50 + 0.20 * (fbm_low - 0.5)

    rgba = np.stack(
        [
            np.clip(height, 0.0, 1.0),
            np.clip(rough_var, 0.0, 1.0),
            np.clip(subsurf_var, 0.0, 1.0),
            np.ones((size, size), dtype=np.float32),
        ],
        axis=-1,
    )
    img, path = _save_array_as_image("skin_micro_2k", rgba, out_dir, "Non-Color")
    return {"skin_micro": (img, path)}


def generate_knit_textures(out_dir: str):
    """
    Generate 2048x2048 chunky fisherman's-rib / brioche knitwear textures matching reference.png:
    prominent vertical wale columns separated by deep soft purl valleys, with clear V-loop stitches
    along every vertical rib column!
    """
    size = 2048
    # 52 vertical wale columns across U, 112 tight horizontal courses up V -> crisp VERTICAL knit columns!
    wales = 52.0
    courses = 112.0

    u = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32)[:, None]

    fbm_yarn = _fbm_tileable(size, size, freqs=(8, 18, 42), seed=303)
    fbm_fuzz = _fbm_tileable(size, size, freqs=(64, 140), seed=509)

    su = u * wales + 0.045 * (fbm_yarn - 0.5)
    sv = v * courses + 0.045 * (np.roll(fbm_yarn, 311, axis=0) - 0.5)

    # Fractional position across a wale column [-0.5, 0.5] and course [0, 1)
    fx = (su % 1.0) - 0.5
    fy = sv % 1.0
    abs_x = np.abs(fx)

    # Prominent vertical rib wale envelope: raised rounded twin-leg column in |fx| < 0.38, deep valley in |fx| > 0.38!
    wale_col = np.clip(1.0 - (abs_x / 0.43) ** 2.4, 0.0, 1.0)
    # Central cleft between left leg (fx ~ -0.17) and right leg (fx ~ +0.17) of each knit V-stitch!
    center_cleft = 1.0 - 0.26 * np.exp(-((fx / 0.075) ** 2))

    # Interlocking steep V-stitches stacked tightly along each vertical wale column!
    # Notice steep V-slope: as |fx| goes 0 -> 0.36, y rises by 0.85 of a course -> unmistakable vertical V braids!
    v_phase = fy - 2.15 * abs_x
    v_loop = 0.5 + 0.5 * np.cos(2.0 * np.pi * v_phase)
    v_plump = (v_loop ** 0.62)

    # 2-ply twisted yarn micro-grooves along each leg
    sign_x = np.where(fx >= 0.0, 1.0, -1.0)
    ply = 0.5 + 0.5 * np.cos(2.0 * np.pi * (sv * 2.0 + fx * 5.5 * sign_x))

    # Recessed purl stitch horizontal bumps deep inside the valleys between wales
    purl_bumps = (1.0 - wale_col) * (0.35 + 0.35 * (0.5 + 0.5 * np.cos(2.0 * np.pi * sv)))

    stitch_h = np.clip(
        wale_col * center_cleft * (0.58 + 0.36 * v_plump + 0.08 * (ply - 0.5))
        + 0.24 * purl_bumps
        + 0.05 * (fbm_fuzz - 0.5),
        0.0,
        1.0,
    )

    # Rich stitch-valley ambient occlusion so vertical rib columns & V-stitches read clearly from beauty & turnaround distances!
    ao = np.clip(0.34 + 0.66 * (stitch_h ** 0.78), 0.24, 1.0)

    # Calibrated dusty-pink yarn palette (realistic mid-value sRGB so it never washes out in bright cafe sun!)
    pink_hi = np.array([0.805, 0.535, 0.560], dtype=np.float32)
    pink_mid = np.array([0.725, 0.448, 0.476], dtype=np.float32)
    pink_deep = np.array([0.465, 0.245, 0.275], dtype=np.float32)

    heather = np.clip(0.5 + 0.30 * (fbm_yarn - 0.5) + 0.24 * (fbm_fuzz - 0.5) + 0.14 * (ply - 0.5), 0.0, 1.0)
    yarn_col = pink_mid[None, None, :] * (1.0 - 0.38 * heather[:, :, None]) + pink_hi[None, None, :] * (
        0.38 * heather[:, :, None]
    )
    albedo_rgb = pink_deep[None, None, :] * (1.0 - ao[:, :, None]) + yarn_col * ao[:, :, None]

    albedo_rgba = np.concatenate([albedo_rgb, np.ones((size, size, 1), dtype=np.float32)], axis=-1)
    height_rgba = np.stack([stitch_h, stitch_h, stitch_h, np.ones_like(stitch_h)], axis=-1)

    rough_size = 1024
    sh_small = stitch_h[::2, ::2]
    fz_small = fbm_fuzz[::2, ::2]
    rough = np.clip(0.85 - 0.14 * sh_small + 0.08 * (fz_small - 0.5), 0.60, 0.94)
    rough_rgba = np.stack([rough, rough, rough, np.ones((rough_size, rough_size), dtype=np.float32)], axis=-1)

    # Chunky 1x1 rib knit height map (1024x1024) for crew-neck collar, cuffs, and waist hem
    rsize = 1024
    ru = np.linspace(0.0, 1.0, rsize, endpoint=False, dtype=np.float32)[None, :]
    rv = np.linspace(0.0, 1.0, rsize, endpoint=False, dtype=np.float32)[:, None]
    rfbm = _fbm_tileable(rsize, rsize, freqs=(8, 24, 64), seed=811)
    rib_cols = 42.0
    rib_rows = 88.0
    r_phase = ru * rib_cols + 0.035 * (rfbm - 0.5)
    rib_wave = np.clip(0.5 + 0.5 * np.cos(2.0 * np.pi * r_phase), 0.0, 1.0) ** 0.68
    v_courses = 0.5 + 0.5 * np.cos(2.0 * np.pi * (rv * rib_rows - 1.4 * np.abs((r_phase % 1.0) - 0.5)))
    purl_valley = 0.5 + 0.5 * np.cos(2.0 * np.pi * (rv * rib_rows))
    rib_h = np.clip(
        0.72 * rib_wave
        + 0.20 * rib_wave * v_courses
        + 0.08 * (1.0 - rib_wave) * purl_valley
        + 0.05 * (rfbm - 0.5),
        0.0,
        1.0,
    )
    rib_rgba = np.stack([rib_h, rib_h, rib_h, np.ones_like(rib_h)], axis=-1)

    img_ka, path_ka = _save_array_as_image("knit_albedo_2k", albedo_rgba, out_dir, "sRGB")
    img_kh, path_kh = _save_array_as_image("knit_height_2k", height_rgba, out_dir, "Non-Color")
    img_kr, path_kr = _save_array_as_image("knit_roughness_1k", rough_rgba, out_dir, "Non-Color")
    img_rh, path_rh = _save_array_as_image("rib_knit_height_1k", rib_rgba, out_dir, "Non-Color")

    return {
        "knit_albedo": (img_ka, path_ka),
        "knit_height": (img_kh, path_kh),
        "knit_roughness": (img_kr, path_kr),
        "rib_height": (img_rh, path_rh),
    }


def generate_all_textures(out_dir: str = "textures"):
    """Generate and pack all procedural textures required by the scene."""
    out = {}
    out.update(generate_iris_textures(out_dir))
    out.update(generate_skin_micro_texture(out_dir))
    out.update(generate_knit_textures(out_dir))
    return out
