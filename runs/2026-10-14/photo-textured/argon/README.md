# 3D Upper-Body Portrait & Turnaround Scene (`character.blend`) — Second Finishing Pass

## 1. How to Build and Open the Scene

### Reproducible Headless Build & Render
To regenerate `character.blend`, `scene-report.json`, and all 5 final-resolution renders (`renders/beauty.png` at `960x1040`, `renders/front.png`, `renders/three-quarter.png`, `renders/side.png`, and `renders/back.png` at `768x768`) from scratch:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

For a fast preview iteration (`480x520` beauty, `384x384` turnaround views, 10 Cycles samples):

```bash
PREVIEW=1 /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

### Opening the Saved Scene (`character.blend`)
Open `character.blend` in Blender 5.1.1 (`/Applications/Blender.app`):
- The scene opens on `BeautyCamera` (`960x1040`), configured for `CYCLES` (`CPU`, `3` threads, `32` samples, `OPENIMAGEDENOISE`, `AgX` view transform).
- All 6 generated hybrid photographic + procedural textures (`<=2048x2048`) are packed directly into `character.blend` (the raw `input/reference.png` is used only in-memory during texture synthesis and removed before packing).
- The first-pass scene is preserved unmodified at `source-character.blend` for comparison.
- The Outliner is organized into four clean collections:
  - `Character`: `CharacterHead`, `Eyeball_L`, `Eyeball_R`, `Cornea_L`, `Cornea_R`, `EyeTearline_L`, `EyeTearline_R`, `Eyelashes_L`, `Eyelashes_R`, `Eyebrows_L`, `Eyebrows_R`, `HairScalpCap`, `HairVolumeLocks`, `HairStrands_Bangs`, `HairStrands_FaceFraming`, `HairStrands_Main`, `SweaterBody`, and `SweaterCollar`.
  - `Environment`: 3D out-of-focus cafe background (`CafeWindow_Left`, `CafePlant_*`, `CafeTable_Left`, `CafeChair_Left`, `CafeCup_Left`, `CafeWall_Back`, `CafeCounter_Right`, `CafeEspressoMachine`, `CafeShelf_UpperRight`, `CafeShelf_MidRight`, `CafeJar_1..3`), hidden during the 4 neutral-studio turnaround renders and restored before saving `character.blend`.
  - `Lighting`: `KeyLight_Window`, `FillLight_CafeBounce`, `RimLight_TopBack`, and `BounceLight_Table`.
  - `Cameras`: `BeautyCamera` (`960x1040`) and the 4 turntable cameras around the character's vertical axis (`Cam_Front` `0°`, `Cam_ThreeQuarter` `45°`, `Cam_Side` `90°`, `Cam_Back` `180°` at `768x768`).

---

## 2. How Reference & Rendered Images Were Accessed

- **Direct `Read` Tool Inspection**: `input/reference.png` (`1205x1306` RGBA PNG), the first-pass renders, and all 5 final second-pass renders (`renders/beauty.png`, `renders/front.png`, `renders/three-quarter.png`, `renders/side.png`, `renders/back.png`) were inspected with the `Read` tool (which returns a vision-language description of each image in this CLI harness).
- **Direct Pixel-Buffer & Landmark Analysis in Blender Python**: Because the `Read` tool returns text descriptions rather than raw pixel arrays, we also loaded `input/reference.png` and all 5 output renders directly via `bpy.data.images.load(...)` in Blender Python to measure exact 2D landmark coordinates, sRGB/luminance distributions across facial zones (forehead, malar cheeks, nose tip, lips, iris, hair roots/highlights, and unoccluded sweater patches), 1-pixel-stride ASCII luminance maps of the eyes and forehead, and 3D-to-2D camera projections (`world_to_camera_view`).

---

## 3. What Changed from the First Pass (`source-character.blend`)

1. **Hybrid Photographic + Procedural UV Texture Pipeline**:
   - Replaced the first-pass purely procedural skin and sweater textures with landmark-aligned photographic textures derived from `input/reference.png` combined with seamless procedural synthesis for unobserved regions (`360°` temples, ears, neck, back of head, and full sweater body/sleeves).
   - Fixed the washed-out pale highlight appearance of the first pass by calibrating skin base chromaticity (`sRGB ≈ (0.818, 0.612, 0.542)`), subsurface scattering (`Subsurface Weight = 0.16`, `Subsurface Scale = 0.005 m`), roughness (`0.32–0.43`), and specular level (`0.38`) under `AgX` tone mapping.
2. **Smiling Almond Eyes, Centered Gaze & Dark Pupil Readability**:
   - In `sculpt_face_canonical()`, lowered the upper eyelid margin (`-2.2 mm` with outer canthus curvature) and gently lifted the smiling lower eyelid (`+0.7 mm`) to eliminate the wide-eyed/surprised expression from the first pass.
   - In `build_eyes_lashes_brows()`, calibrated the eyeball center (`z = 1.5462`), horizontal camera-directed gaze (`canon_gaze = [0.036 / 0.006, -0.9988, 0.036]`), and `11.8 mm` iris diameter (`r_iris = 0.365, r_pupil = 0.152`) so the upper eyelid covers the top ~25% of the iris with zero white sclera visible above the iris.
   - Reduced `Mat_Cornea` Fresnel `IOR` from `1.24` to `1.12` so large area lights produce crisp catchlights without veiling the dark pupil in a milky reflection.
3. **Continuous Wispy 3D Bangs, Fluted Volume Locks & Tucked Front Scalp Cap**:
   - Removed the 13 discrete polygon forehead bang strips and root stubs from the first pass.
   - Distributed 78 4-lobed fluted 8-sided `HairVolumeLocks` along the full front-to-back cranial parting curve (`part_y_lock` from `-0.128` at the front hairline to `-0.012` at the back crown), wired to `HairStrandTex` (`Roughness = 0.50`, `Specular IOR Level = 0.11`, longitudinal wave bump `0.52`), and veiled beneath `5,700` native 3D `HairCurves` strands (`380` in `HairStrands_Bangs`, `1,120` in `HairStrands_FaceFraming`, `4,200` in `HairStrands_Main`).
   - Tucked the front hemisphere of `HairScalpCap` (`cy < -0.060`) `2 mm` inside `CharacterHead` while keeping its back/side dome at `cran_scale = 1.018`, eliminating any visible dark scalp-cap step on the forehead while preventing bald gaps in side/back views.
4. **Extended Forearm Sleeves & Open Ribbed Cuffs**:
   - Adjusted arm pose (`-9.5°` pitch, `22.0°` adduction) and extended the `SweaterBody` sleeves `8.2 cm` downward along the forearm axis to `z ≈ 0.947 m` (aligning flush with the `z = 0.958 m` waist hem across all turnaround views) with double holding loops and a recessed dark interior cuff tunnel.

---

## 4. Exact Texture Generation & UV Projection Methods

All 6 textures are generated in `generate_and_pack_textures()` in `build.py` and packed into `character.blend`:

| Texture Name | Resolution | Color Space | Source & UV Projection Method |
| :--- | :--- | :--- | :--- |
| `SkinAlbedoTex` | `2048x2048` | `sRGB` | **Hybrid (`input/reference.png` + procedural skin)** on `FaceCylUV`. Fits a 25-landmark Thin-Plate Spline (TPS) warp from canonical `(theta, cz)` on `CharacterHead` to `input/reference.png`. Applies an 80th-percentile local skin envelope filter outside tight feature masks (brows, lids, nose, lips) to de-hair stray forehead/cheek wisps, applies 68% low-frequency delighting, strictly excludes the inner eye socket openings (`1.0 - eye_aperture_excl` so zero double-eye artifacts occur), and cosine-feathers (`78%` max weight) into color-matched 360° procedural skin across the upper forehead, temples, ears, neck, and back of head. |
| `SkinRoughBumpTex` | `2048x2048` | `Non-Color` | **Hybrid (`input/reference.png` high-pass + procedural micro-relief)** on `FaceCylUV`. Packed RGB map: `R` = TPS-warped photographic high-pass skin/lip micro-relief (`photo_micro`) combined with procedural pores and vertical lip grooves; `G` = spatially varying roughness (`0.32` on lips/nose tip, `0.43` on cheeks/forehead, `0.48` on neck); `B` = subsurface scattering weight mask. |
| `EyeIrisTex` | `1024x1024` | `sRGB` | **Reference-calibrated radial synthesis** on `Eyeball_L/R` `UVMap`. Calibrated to sampled `input/reference.png` dark espresso-chestnut iris chromaticity with radial stroma fibers, collarette ring, dark limbal ring (`r_iris = 0.365`), dark pupil (`r_pupil = 0.152`), dual window catchlights, and warm off-white sclera. |
| `HairStrandTex` | `1024x1024` | `sRGB` | **Reference-calibrated longitudinal strand texture** on `HairVolumeLocks` / `HairScalpCap` `UVMap`. Uses multi-frequency longitudinal fiber striations and a root-to-tip tonal gradient calibrated to `input/reference.png` warm chestnut roots (`(0.245, 0.142, 0.098)`) and sunlit caramel-chestnut highlights (`(0.365, 0.220, 0.150)`). |
| `SweaterKnitHeightTex` | `2048x2048` | `Non-Color` | **Hybrid (`input/reference.png` yarn patch + procedural stitch relief)** on `KnitUV`. Extracts the unoccluded sweater patch (`x ∈ [0.63, 0.77], y ∈ [0.59, 0.73]`) from `input/reference.png`, removes low-frequency lighting gradients, mirror-tiles it (`2x2` symmetric tiling, repeated `4x4` across `KnitUV`), and blends its high-frequency yarn grain with a `64x80` interlocking waffle/seed-stitch heightmap driving both mesh `Displace` (`Midlevel = 0.5, Strength = 0.0038`) and shader `Bump`. |
| `SweaterKnitAlbedoTex` | `2048x2048` | `sRGB` | **Hybrid (`input/reference.png` sweater chromaticity + stitch AO)** on `KnitUV`. Combines the mirror-tiled photographic yarn variation from `input/reference.png` with dusty-rose base chromaticity (`sRGB ≈ (0.76, 0.51, 0.54)`) and stitch-crevice ambient occlusion across torso, shoulders, collar, and sleeves. |

---

## 5. Assumptions for Unobserved Regions

Because `input/reference.png` is a single frontal portrait cropped at the chest and upper arms:
- **Side & Back Cranium, Ears, and Neck (`360°` Skin)**: Unobserved side/back skin surfaces are shaded with seamless cylindrical procedural skin matched to the delighted frontal complexion (`sRGB ≈ (0.818, 0.612, 0.542)`) with organic multi-frequency chromophore mottling and neck shading.
- **Side & Back Hair Arrangement**: The 78 `HairVolumeLocks` and `5,700` `HairCurves` strands are distributed around the full `360°` perimeter (`azim ∈ [-177°, +177°]`) cascading down to mid-back (`z ≈ 1.14 m`) outside the dilated 3D body envelope.
- **Back Sweater Drapery & Lower Hems**: The waffle/seed-stitch `KnitUV` parameterization wraps seamlessly around the back torso and sleeves; the truncated starter mesh boundaries at the waist and forearms are extended into clean, rolled horizontal hems (`z ≈ 0.947–0.958 m`).

---

## 6. Attribution & Provenance

- **Starter Base Mesh (`input/upper-body.obj`)**: Derived from the MakeHuman project (`hm08` female base mesh), licensed under **CC0 1.0 Universal (Public Domain Dedication)**. See `input/PROVENANCE.md` for full details.
- **Reference Photograph (`input/reference.png`)**: Provided in the task workspace (`input/PROVENANCE.md`) as the visual and texture source for the second finishing pass.

---

## 7. Honest Self-Review & Remaining Limitations

- **What Worked Well**:
  - The 25-landmark TPS facial albedo and high-pass micro-relief projection (`SkinAlbedoTex`, `SkinRoughBumpTex`) with 80th-percentile local skin envelope de-hairing and strict eye-aperture exclusion gives `CharacterHead` warm, natural peachy skin tones, soft lip/cheek coloration, and seamless transitions into the 360° neck and ear surfaces without any double-eye projection artifacts.
  - Lowering the upper eyelids (`-2.2 mm`), lifting the lower eyelids (`+0.7 mm`), centering the `11.8 mm` iris (`r_iris = 0.365`), and lowering `Mat_Cornea` `IOR` to `1.12` eliminated the wide-eyed expression and upper scleral show from the first pass.
  - The hybrid photographic + procedural knit displacement (`SweaterKnitHeightTex`, `SweaterKnitAlbedoTex`) and extended forearm sleeves (`z ≈ 0.947 m`) give the sweater crisp waffle/seed-stitch relief and balanced proportions across all 5 views.
- **Remaining Limitations**:
  - **Single-View 3D Sculpt Approximation**: Facial geometry is sculpted by deforming the CC0 MakeHuman `hm08` topology rather than a multi-camera photogrammetry scan, so subtle 3D facial planes remain an approximation of the subject.
  - **Hybrid Lock + Curve Hair Representation**: To stay well within the `<1,000,000` evaluated vertex limit (`133,321` evaluated mesh vertices, `129,384` evaluated faces, plus `5,700` `HairCurves` with `154,660` control points = `287,981` total) and CPU render time (`~98 s` for all 5 final renders), the hair uses 78 fluted 3D volume locks veiled under `5,700` curve strands rather than `>100,000` individual scalp follicles. As a result, underlying volume locks remain partially discernible as thicker sculpted bundles in side/back views, and where the front-crown hair locks part and cast contact shadows near the upper forehead/temples at 32 Cycles samples with `OPENIMAGEDENOISE`, subtle shadow streaking remains visible under the wispy bangs.
  - **Truncated Bust Geometry**: As an upper-body bust, the torso and open sleeve cuffs terminate cleanly near the waist plane (`z ≈ 0.947–0.958 m`) without a lower body or articulated fingers.
