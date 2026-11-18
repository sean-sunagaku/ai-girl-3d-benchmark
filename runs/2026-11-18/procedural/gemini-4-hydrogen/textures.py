"""Procedural texture generation (<= 2048x2048) for skin, eyes, knit sweater, and wood."""

import os
import math
import numpy as np
import bpy


TEX_DIR = os.path.abspath("generated_textures")


def _ensure_dir():
    os.makedirs(TEX_DIR, exist_ok=True)


def _save_and_load_image(name: str, rgba: np.ndarray, colorspace: str = "sRGB"):
    """Save RGBA float32 array (H, W, 4) in [0, 1] as PNG and load into Blender."""
    _ensure_dir()
    h, w, _ = rgba.shape
    filepath = os.path.join(TEX_DIR, f"{name}.png")

    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])

    img = bpy.data.images.new(name=name, width=w, height=h, alpha=True, float_buffer=False)
    # Must set colorspace BEFORE pixels.foreach_set; changing colorspace afterwards clears buffer in Blender 5.1.
    # Row 0 of Blender img.pixels corresponds to UV v=0 (bottom of saved PNG).
    img.colorspace_settings.name = colorspace
    buf = np.ascontiguousarray(rgba, dtype=np.float32)
    img.pixels.foreach_set(buf.ravel())
    img.filepath_raw = filepath
    img.file_format = "PNG"
    img.save()
    img.pack()
    return img


def _fbm_2d(h: int, w: int, octaves=(4, 8, 16, 32, 64), seed: int = 42, tileable: bool = True) -> np.ndarray:
    """Fast vectorized multi-octave smooth noise in [0, 1]."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp_sum = 0.0
    for idx, freq in enumerate(octaves):
        amp = 0.55 ** idx
        grid = rng.random((freq, freq), dtype=np.float32)
        if tileable:
            ys = (np.arange(h, dtype=np.float32) / h) * freq
            xs = (np.arange(w, dtype=np.float32) / w) * freq
            y0 = np.floor(ys).astype(np.int32) % freq
            x0 = np.floor(xs).astype(np.int32) % freq
            y1 = (y0 + 1) % freq
            x1 = (x0 + 1) % freq
        else:
            ys = (np.arange(h, dtype=np.float32) / max(1, h - 1)) * (freq - 1)
            xs = (np.arange(w, dtype=np.float32) / max(1, w - 1)) * (freq - 1)
            y0 = np.clip(np.floor(ys).astype(np.int32), 0, freq - 2)
            x0 = np.clip(np.floor(xs).astype(np.int32), 0, freq - 2)
            y1 = y0 + 1
            x1 = x0 + 1
        fy = (ys - np.floor(ys))[:, None]
        fx = (xs - np.floor(xs))[None, :]
        # Smoothstep interpolation
        fy = fy * fy * (3.0 - 2.0 * fy)
        fx = fx * fx * (3.0 - 2.0 * fx)

        c00 = grid[y0[:, None], x0[None, :]]
        c10 = grid[y0[:, None], x1[None, :]]
        c01 = grid[y1[:, None], x0[None, :]]
        c11 = grid[y1[:, None], x1[None, :]]
        val = (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy
        acc += amp * val
        amp_sum += amp
    return np.clip(acc / amp_sum, 0.0, 1.0)


def generate_iris_texture(size: int = 1024):
    """Generate warm chestnut-brown portrait iris + natural off-white sclera texture."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    u = (x + 0.5) / size - 0.5
    v = (y + 0.5) / size - 0.5
    r = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)

    # Iris occupies radius 0.296 in UV space (~15.3mm diameter on 27.8mm eyeball - expressive portrait ratio)
    r_iris = 0.296
    r_pupil = r_iris * 0.285
    rn = r / r_iris

    # Sclera base: clean warm ivory off-white with very subtle peripheral warmth
    sclera_base = np.zeros((size, size, 3), dtype=np.float32)
    sclera_noise = _fbm_2d(size, size, octaves=(6, 14, 30), seed=11, tileable=False)
    sclera_base[:, :, 0] = 0.960 - 0.018 * sclera_noise
    sclera_base[:, :, 1] = 0.945 - 0.020 * sclera_noise
    sclera_base[:, :, 2] = 0.935 - 0.020 * sclera_noise

    # Very subtle pink peripheral sclera edge (mostly hidden under lids)
    periph = np.clip((rn - 1.38) / 0.65, 0.0, 1.0)
    veins = np.clip(np.sin(theta * 18.0 + sclera_noise * 8.0) * 0.5 + 0.5, 0.0, 1.0) ** 8
    sclera_base[:, :, 0] += periph * (0.025 + 0.025 * veins)
    sclera_base[:, :, 1] -= periph * (0.015 + 0.020 * veins)
    sclera_base[:, :, 2] -= periph * (0.020 + 0.020 * veins)

    # Iris radial collagen fiber structure
    radial_n1 = np.zeros((size, size), dtype=np.float32)
    rng = np.random.default_rng(101)
    for k in (19, 37, 61, 97, 149):
        phase = rng.uniform(0, 2 * math.pi)
        wobble = 0.07 * np.sin(rn * 11.0 + phase)
        radial_n1 += np.sin(k * (theta + wobble) + phase)
    radial_n1 = np.clip(radial_n1 * 0.12 + 0.5, 0.0, 1.0)

    fine_noise = _fbm_2d(size, size, octaves=(12, 28, 64), seed=17, tileable=False)
    crypts = np.clip((fine_noise - 0.48) * 1.9, 0.0, 1.0)

    # Rich warm dark chestnut brown matching reference.png
    collarette = np.exp(-((rn - 0.38) / 0.12) ** 2)
    limbus = np.clip((rn - 0.80) / 0.18, 0.0, 1.0)

    iris_rgb = np.zeros((size, size, 3), dtype=np.float32)
    iris_rgb[:, :, 0] = 0.31 + 0.16 * radial_n1 + 0.13 * collarette - 0.16 * limbus - 0.06 * crypts
    iris_rgb[:, :, 1] = 0.155 + 0.090 * radial_n1 + 0.075 * collarette - 0.090 * limbus - 0.035 * crypts
    iris_rgb[:, :, 2] = 0.072 + 0.045 * radial_n1 + 0.035 * collarette - 0.045 * limbus - 0.018 * crypts

    # Subtle lower-iris light scatter crescent
    bottom_glow = np.clip((-v / r_iris) * 0.45 + 0.30, 0.0, 1.0) * (1.0 - limbus)
    iris_rgb[:, :, 0] += 0.07 * bottom_glow
    iris_rgb[:, :, 1] += 0.038 * bottom_glow
    iris_rgb[:, :, 2] += 0.014 * bottom_glow
    iris_rgb = np.clip(iris_rgb, 0.015, 0.85)

    # Pupil blend
    pupil_alpha = np.clip((r - r_pupil) / 0.004, 0.0, 1.0)
    pupil_rgb = np.array([0.015, 0.010, 0.009], dtype=np.float32)
    iris_rgb = pupil_rgb[None, None, :] * (1.0 - pupil_alpha[:, :, None]) + iris_rgb * pupil_alpha[:, :, None]

    # Smooth limbus-to-sclera edge
    iris_mask = 1.0 - np.clip((r - (r_iris - 0.004)) / 0.008, 0.0, 1.0)
    rgb = sclera_base * (1.0 - iris_mask[:, :, None]) + iris_rgb * iris_mask[:, :, None]
    rgb = np.clip(rgb, 0.0, 1.0)

    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb
    return _save_and_load_image("iris_eye", rgba, colorspace="sRGB")


def _gaussian_blob(uu: np.ndarray, vv: np.ndarray, cu: float, cv: float, su: float, sv: float, rot: float = 0.0) -> np.ndarray:
    du = uu - cu
    dv = vv - cv
    if rot != 0.0:
        c, s = math.cos(rot), math.sin(rot)
        du, dv = du * c - dv * s, du * s + dv * c
    return np.exp(-0.5 * ((du / su) ** 2 + (dv / sv) ** 2))


def _curve_distance_mask(uu: np.ndarray, vv: np.ndarray, pts_uv: np.ndarray, width: float) -> np.ndarray:
    """Minimum distance field from a polyline in UV space."""
    d2 = np.full(uu.shape, 1e9, dtype=np.float32)
    for k in range(len(pts_uv) - 1):
        ax, ay = pts_uv[k]
        bx, by = pts_uv[k + 1]
        vx, vy = bx - ax, by - ay
        len2 = vx * vx + vy * vy + 1e-12
        t = np.clip(((uu - ax) * vx + (vv - ay) * vy) / len2, 0.0, 1.0)
        px = ax + t * vx
        py = ay + t * vy
        dist2 = (uu - px) ** 2 + (vv - py) ** 2
        d2 = np.minimum(d2, dist2)
    dist = np.sqrt(d2)
    return np.clip(1.0 - dist / width, 0.0, 1.0) ** 1.6


def generate_skin_textures(landmarks_uv: dict, size: int = 2048):
    """Generate pixel-registered 2048x2048 face/neck albedo, roughness, and pore bump maps."""
    y_idx, x_idx = np.mgrid[0:size, 0:size].astype(np.float32)
    uu = (x_idx + 0.5) / size
    vv = (y_idx + 0.5) / size

    # Multi-scale skin noise (macro tone warmth, mottled blood flow, medium grain, fine pores)
    macro_n = _fbm_2d(size, size, octaves=(6, 12, 24), seed=7, tileable=False)
    med_n = _fbm_2d(size, size, octaves=(32, 64, 128), seed=19, tileable=False)
    pore_n = _fbm_2d(size, size, octaves=(128, 256, 384), seed=31, tileable=False)

    # Base fair warm porcelain-peach skin (natural warmth under AgX)
    albedo = np.zeros((size, size, 3), dtype=np.float32)
    albedo[:, :, 0] = 0.905 - 0.030 * macro_n - 0.012 * med_n
    albedo[:, :, 1] = 0.718 - 0.032 * macro_n - 0.014 * med_n
    albedo[:, :, 2] = 0.612 - 0.030 * macro_n - 0.014 * med_n

    # Subtle golden-warm undertone variation on forehead & chin
    albedo[:, :, 0] += 0.015 * (macro_n - 0.5)
    albedo[:, :, 1] += 0.010 * (macro_n - 0.5)

    # Extract exact UV landmarks
    cheek_l = landmarks_uv["cheek_l"]
    cheek_r = landmarks_uv["cheek_r"]
    nose_tip = landmarks_uv["nose_tip"]
    chin = landmarks_uv["chin"]
    forehead = landmarks_uv["forehead"]
    ear_l = landmarks_uv["ear_l"]
    ear_r = landmarks_uv["ear_r"]

    # Rosy-peach cheek blush (soft feminine flush matching reference.png)
    blush_l = _gaussian_blob(uu, vv, cheek_l[0], cheek_l[1], 0.068, 0.042, rot=-0.18)
    blush_r = _gaussian_blob(uu, vv, cheek_r[0], cheek_r[1], 0.068, 0.042, rot=0.18)
    nose_blush = _gaussian_blob(uu, vv, nose_tip[0], nose_tip[1], 0.026, 0.018)
    chin_warm = _gaussian_blob(uu, vv, chin[0], chin[1], 0.034, 0.022)
    ear_warm_l = _gaussian_blob(uu, vv, ear_l[0], ear_l[1], 0.045, 0.055)
    ear_warm_r = _gaussian_blob(uu, vv, ear_r[0], ear_r[1], 0.045, 0.055)

    flush_mask = np.clip(1.05 * (blush_l + blush_r) + 0.55 * nose_blush + 0.30 * chin_warm + 0.55 * (ear_warm_l + ear_warm_r), 0.0, 1.0)
    albedo[:, :, 0] = albedo[:, :, 0] * (1.0 - 0.06 * flush_mask) + 0.94 * (0.06 * flush_mask)
    albedo[:, :, 1] -= 0.098 * flush_mask
    albedo[:, :, 2] -= 0.090 * flush_mask

    # Nostril shadow & subtle depth under nose wings
    for n_pt in landmarks_uv["nostrils"]:
        nd = _gaussian_blob(uu, vv, n_pt[0], n_pt[1], 0.011, 0.009)
        albedo[:, :, 0] -= 0.14 * nd
        albedo[:, :, 1] -= 0.16 * nd
        albedo[:, :, 2] -= 0.14 * nd

    # Soft natural upper/lower eyelid warmth & subtle brown lash-line depth
    for side in ("l", "r"):
        up_pts = np.array(landmarks_uv[f"eyelid_upper_{side}"], dtype=np.float32)
        lo_pts = np.array(landmarks_uv[f"eyelid_lower_{side}"], dtype=np.float32)
        eye_c = landmarks_uv[f"eye_center_{side}"]

        # Soft warm peach-taupe eyeshadow glow on upper lid
        lid_glow = _gaussian_blob(uu, vv, eye_c[0], eye_c[1] + 0.014, 0.044, 0.019)
        albedo[:, :, 0] -= 0.065 * lid_glow
        albedo[:, :, 1] -= 0.105 * lid_glow
        albedo[:, :, 2] -= 0.105 * lid_glow

        # Soft feathered lash-line espresso-brown shadow framing expressive almond lids
        lash_up = _curve_distance_mask(uu, vv, up_pts, 0.0056)
        lash_lo = _curve_distance_mask(uu, vv, lo_pts, 0.0028)
        lash_mask = np.clip(lash_up * 0.80 + lash_lo * 0.22, 0.0, 1.0)
        albedo[:, :, 0] *= (1.0 - 0.62 * lash_mask)
        albedo[:, :, 1] *= (1.0 - 0.70 * lash_mask)
        albedo[:, :, 2] *= (1.0 - 0.74 * lash_mask)

        # Subtle soft brow root tint under 3D brow hairs
        brow_pts = np.array(landmarks_uv[f"brow_{side}"], dtype=np.float32)
        brow_m = _curve_distance_mask(uu, vv, brow_pts, 0.010) * 0.28
        albedo[:, :, 0] = albedo[:, :, 0] * (1.0 - brow_m) + 0.32 * brow_m
        albedo[:, :, 1] = albedo[:, :, 1] * (1.0 - brow_m) + 0.21 * brow_m
        albedo[:, :, 2] = albedo[:, :, 2] * (1.0 - brow_m) + 0.15 * brow_m

    # Lips: soft coral-rose-peach natural gradient + darker warm crease along mouth seam
    upper_lip_pts = np.array(landmarks_uv["lip_upper_outline"], dtype=np.float32)
    lower_lip_pts = np.array(landmarks_uv["lip_lower_outline"], dtype=np.float32)
    seam_pts = np.array(landmarks_uv["lip_seam"], dtype=np.float32)

    # Build lip polygon mask using signed vertical bounds interpolated along U
    lip_u_min = min(upper_lip_pts[:, 0].min(), lower_lip_pts[:, 0].min())
    lip_u_max = max(upper_lip_pts[:, 0].max(), lower_lip_pts[:, 0].max())
    in_u = (uu >= lip_u_min) & (uu <= lip_u_max)
    u_row = np.clip(uu[0, :], lip_u_min, lip_u_max)
    top_v = np.interp(u_row, upper_lip_pts[:, 0], upper_lip_pts[:, 1])[None, :]
    bot_v = np.interp(u_row, lower_lip_pts[:, 0], lower_lip_pts[:, 1])[None, :]
    seam_v = np.interp(u_row, seam_pts[:, 0], seam_pts[:, 1])[None, :]

    # Distance from vermilion border (soft feathered edge)
    d_top = (top_v - vv) / 0.0052
    d_bot = (vv - bot_v) / 0.0052
    # Taper softly near mouth corners
    u_mid = 0.5 * (lip_u_min + lip_u_max)
    u_half = max(1e-5, 0.5 * (lip_u_max - lip_u_min))
    corner_taper = np.clip((1.0 - np.abs(uu - u_mid) / (u_half * 1.03)) / 0.14, 0.0, 1.0)
    lip_mask = np.where(in_u, np.clip(np.minimum(d_top, d_bot), 0.0, 1.0) * corner_taper, 0.0)

    # Lip vertical crease texture & center saturation
    lip_lines = np.sin((uu - u_mid) * 750.0 + med_n * 5.0) * 0.5 + 0.5
    inner_lip = np.exp(-((vv - seam_v) / 0.0055) ** 2) * lip_mask
    lip_color = np.zeros((size, size, 3), dtype=np.float32)
    # Soft coral-rose lip color matching reference.png (warm glossy inner tint)
    lip_color[:, :, 0] = 0.89 - 0.11 * inner_lip - 0.02 * lip_lines
    lip_color[:, :, 1] = 0.41 - 0.14 * inner_lip - 0.02 * lip_lines
    lip_color[:, :, 2] = 0.38 - 0.12 * inner_lip - 0.015 * lip_lines
    albedo = albedo * (1.0 - 0.90 * lip_mask[:, :, None]) + lip_color * (0.90 * lip_mask[:, :, None])

    # Subtle natural lip seam crease (avoid dark corner line!)
    seam_center_w = np.exp(-((uu - u_mid) / (u_half * 0.72)) ** 4)
    seam_line = _curve_distance_mask(uu, vv, seam_pts, 0.0016) * seam_center_w
    albedo[:, :, 0] *= (1.0 - 0.14 * seam_line)
    albedo[:, :, 1] *= (1.0 - 0.19 * seam_line)
    albedo[:, :, 2] *= (1.0 - 0.19 * seam_line)

    # Subtle collarbone/neck warmth
    neck_shadow = np.clip((0.28 - vv) / 0.20, 0.0, 1.0)
    albedo[:, :, 0] -= 0.018 * neck_shadow
    albedo[:, :, 1] -= 0.022 * neck_shadow
    albedo[:, :, 2] -= 0.020 * neck_shadow

    albedo = np.clip(albedo, 0.02, 0.99)

    # Roughness map (lower = glossier): lips ~0.18, nose/forehead/cheekbone highlights ~0.31, base ~0.40
    t_zone = (
        _gaussian_blob(uu, vv, nose_tip[0], nose_tip[1] + 0.02, 0.022, 0.055)
        + 0.75 * _gaussian_blob(uu, vv, forehead[0], forehead[1], 0.085, 0.045)
        + 0.65 * _gaussian_blob(uu, vv, cheek_l[0], cheek_l[1] + 0.015, 0.052, 0.032)
        + 0.65 * _gaussian_blob(uu, vv, cheek_r[0], cheek_r[1] + 0.015, 0.052, 0.032)
    )
    t_zone = np.clip(t_zone, 0.0, 1.0)
    rough = 0.41 - 0.085 * t_zone + 0.045 * (pore_n - 0.5) + 0.025 * (med_n - 0.5)
    rough = rough * (1.0 - lip_mask) + (0.19 + 0.04 * lip_lines) * lip_mask
    rough = np.clip(rough, 0.14, 0.62)

    # Micro-bump map (neutral 0.5, pores dip slightly below 0.5, restrained for youthful smooth skin)
    pore_zone = np.clip(
        _gaussian_blob(uu, vv, nose_tip[0], nose_tip[1] + 0.01, 0.05, 0.04)
        + _gaussian_blob(uu, vv, cheek_l[0] - 0.02, cheek_l[1], 0.055, 0.045)
        + _gaussian_blob(uu, vv, cheek_r[0] + 0.02, cheek_r[1], 0.055, 0.045)
        + 0.6 * _gaussian_blob(uu, vv, forehead[0], forehead[1], 0.09, 0.05),
        0.2,
        1.0,
    )
    bump = 0.5 + (pore_n - 0.5) * (0.28 * pore_zone) + (med_n - 0.5) * 0.10
    # Lip micro-creases
    bump = bump * (1.0 - lip_mask) + (0.5 + (lip_lines - 0.5) * 0.35) * lip_mask
    bump = np.clip(bump, 0.0, 1.0)

    rgba_a = np.ones((size, size, 4), dtype=np.float32)
    rgba_a[:, :, :3] = albedo
    img_albedo = _save_and_load_image("skin_albedo", rgba_a, colorspace="sRGB")

    rgba_r = np.ones((size, size, 4), dtype=np.float32)
    rgba_r[:, :, :3] = rough[:, :, None]
    img_rough = _save_and_load_image("skin_roughness", rgba_r, colorspace="Non-Color")

    rgba_b = np.ones((size, size, 4), dtype=np.float32)
    rgba_b[:, :, :3] = bump[:, :, None]
    img_bump = _save_and_load_image("skin_bump", rgba_b, colorspace="Non-Color")

    return img_albedo, img_rough, img_bump


def generate_knit_textures(size: int = 1024):
    """Generate tileable chunky stockinette V-stitch & 2x2 rib knit textures for dusty pink sweater."""
    y_idx, x_idx = np.mgrid[0:size, 0:size].astype(np.float32)
    # 16 stitches horizontally x 20 courses vertically per UV tile (chunky cozy knit gauge)
    nx, ny = 16.0, 20.0
    u_st = (x_idx / size) * nx
    v_st = (y_idx / size) * ny

    col_idx = np.floor(u_st)
    sx = u_st - col_idx
    sy = v_st - np.floor(v_st)

    # Stockinette V-stitch: crisp diagonal left/right yarn legs forming distinct V chevrons
    left_center = 0.33 - 0.26 * (sy - 0.5)
    right_center = 0.67 + 0.26 * (sy - 0.5)

    d_left = np.abs(sx - left_center)
    d_right = np.abs(sx - right_center)

    r_yarn = 0.195
    h_left = np.sqrt(np.clip(1.0 - (d_left / r_yarn) ** 2, 0.0, 1.0))
    h_right = np.sqrt(np.clip(1.0 - (d_right / r_yarn) ** 2, 0.0, 1.0))

    loop_arch = 0.52 + 0.48 * np.sin(math.pi * sy)
    ply_l = 0.84 + 0.16 * np.sin((sy * 3.0 + sx * 1.4) * 2.0 * math.pi)
    ply_r = 0.84 + 0.16 * np.sin((sy * 3.0 - sx * 1.4) * 2.0 * math.pi)

    h_stock = np.maximum(h_left * ply_l, h_right * ply_r) * loop_arch
    fuzz = _fbm_2d(size, size, octaves=(16, 48, 96, 192), seed=73, tileable=True)
    melange = _fbm_2d(size, size, octaves=(8, 24, 64), seed=89, tileable=True)

    height = np.clip(0.08 + 0.82 * h_stock + 0.10 * (fuzz - 0.5), 0.0, 1.0)

    # Warm dusty rose-pink melange yarn albedo matching reference.png
    albedo = np.zeros((size, size, 3), dtype=np.float32)
    ao = np.clip(0.30 + 0.70 * h_stock, 0.0, 1.0)
    fiber_var = 0.90 + 0.18 * melange + 0.06 * (fuzz - 0.5)
    albedo[:, :, 0] = (0.50 + 0.34 * ao) * fiber_var
    albedo[:, :, 1] = (0.28 + 0.27 * ao) * fiber_var
    albedo[:, :, 2] = (0.30 + 0.27 * ao) * fiber_var
    albedo = np.clip(albedo, 0.04, 0.95)

    rough = np.clip(0.86 - 0.10 * h_stock + 0.12 * (fuzz - 0.5), 0.64, 0.98)

    rgba_a = np.ones((size, size, 4), dtype=np.float32)
    rgba_a[:, :, :3] = albedo
    img_albedo = _save_and_load_image("knit_albedo", rgba_a, colorspace="sRGB")

    rgba_h = np.ones((size, size, 4), dtype=np.float32)
    rgba_h[:, :, :3] = height[:, :, None]
    img_height = _save_and_load_image("knit_height", rgba_h, colorspace="Non-Color")

    rgba_r = np.ones((size, size, 4), dtype=np.float32)
    rgba_r[:, :, :3] = rough[:, :, None]
    img_rough = _save_and_load_image("knit_roughness", rgba_r, colorspace="Non-Color")

    # 2x2 Ribbed knit texture for collar, cuffs, hem
    rib_phase = (x_idx / size) * 14.0 * 2.0 * math.pi
    rib_wave = 0.5 + 0.5 * np.cos(rib_phase)
    rib_h = np.clip(0.18 + 0.58 * rib_wave + 0.18 * h_stock + 0.08 * (fuzz - 0.5), 0.0, 1.0)
    rib_ao = np.clip(0.35 + 0.65 * rib_h, 0.0, 1.0)
    rib_alb = np.zeros((size, size, 3), dtype=np.float32)
    rib_alb[:, :, 0] = np.clip((0.44 + 0.37 * rib_ao) * fiber_var, 0.04, 0.95)
    rib_alb[:, :, 1] = np.clip((0.24 + 0.28 * rib_ao) * fiber_var, 0.04, 0.95)
    rib_alb[:, :, 2] = np.clip((0.26 + 0.28 * rib_ao) * fiber_var, 0.04, 0.95)

    rgba_ra = np.ones((size, size, 4), dtype=np.float32)
    rgba_ra[:, :, :3] = rib_alb
    img_rib_a = _save_and_load_image("knit_rib_albedo", rgba_ra, colorspace="sRGB")

    rgba_rh = np.ones((size, size, 4), dtype=np.float32)
    rgba_rh[:, :, :3] = rib_h[:, :, None]
    img_rib_h = _save_and_load_image("knit_rib_height", rgba_rh, colorspace="Non-Color")

    return {
        "albedo": img_albedo,
        "height": img_height,
        "roughness": img_rough,
        "rib_albedo": img_rib_a,
        "rib_height": img_rib_h,
    }


def generate_oak_wood_texture(size: int = 512):
    """Generate warm oak wood texture for cafe furniture."""
    y_idx, x_idx = np.mgrid[0:size, 0:size].astype(np.float32)
    u = x_idx / size
    v = y_idx / size
    n1 = _fbm_2d(size, size, octaves=(4, 10, 24), seed=55, tileable=True)
    rings = np.sin((u * 14.0 + 1.8 * n1 + 0.4 * v) * 2.0 * math.pi) * 0.5 + 0.5
    fine = _fbm_2d(size, size, octaves=(32, 96), seed=59, tileable=True)
    grain = 0.65 * rings + 0.35 * fine

    alb = np.zeros((size, size, 3), dtype=np.float32)
    alb[:, :, 0] = 0.56 + 0.20 * grain
    alb[:, :, 1] = 0.36 + 0.16 * grain
    alb[:, :, 2] = 0.20 + 0.10 * grain
    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = np.clip(alb, 0.0, 1.0)
    return _save_and_load_image("cafe_oak_wood", rgba, colorspace="sRGB")
