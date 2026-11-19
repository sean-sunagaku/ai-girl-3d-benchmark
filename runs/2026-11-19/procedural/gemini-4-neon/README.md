# 3D Upper-Body Portrait (Blender 5.1.1 / Cycles CPU)

## Reference Image Access
I directly inspected and viewed the decoded photograph `input/reference.png` using the `read_file` image-reading tool, alongside `input/environment.json` and `input/ASSET-LICENSE.md`. The photograph was used solely as an unrendered visual appearance reference; no photograph pixels, crops, or camera-facing billboards are used anywhere in the scene geometry, textures, or background.

## How to Build and Open
All scene geometry, procedural textures, materials, hair curves, lighting, cameras, and renders are generated reproducibly from the supplied inputs (`input/upper-body.obj`, `input/base.obj`) using Blender 5.1.1:

```bash
# Full reproducible build: constructs scene, packs textures, saves character.blend,
# reopens character.blend to verify portability, renders all 5 final PNGs, and writes scene-report.json
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py

# Optional fast preview mode (renders 12-sample half-res previews into previews/)
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --preview
```

To inspect or edit interactively, open `character.blend` directly in Blender 5.1.1. All 8 generated image textures are packed inside `character.blend`, collections are organized cleanly (`Character/Character_Body_And_Eyes`, `Character/Character_Clothing`, `Character/Character_Hair`, `Cafe_Environment`, `Lighting`, `Cameras`), and `Camera_Beauty` is set as the active scene camera.

## Architecture & Module Breakdown
- `build.py`: Entry script orchestrating the full build, texture packing, saving/reopening `character.blend`, Cycles CPU rendering (`--threads 3`, 32 samples, OpenImageDenoise), and `scene-report.json` generation.
- `sculpt.py`: Imports `input/upper-body.obj`, transfers exact 1-to-1 UVs from `input/base.obj`, sculpts youthful feminine facial proportions, eyelid arches, double-eyelid creases, subtle aegyo-sal, petite nose, soft rosy smile, and graceful neck/head pose (`-14.5°` roll, `+3.5°` pitch, `-6°` yaw), trims covered torso skin beneath the sweater collar, and builds layered 3D eyes (`Eyeball_L/R`, refractive bulge `Cornea_L/R`, `Caruncle_L/R`, and `TearLine_L/R`).
- `textures.py`: Procedurally generates 8 packed textures (all `<= 1536×1536`) using NumPy:
  - 3D-to-UV rasterized skin maps (`Skin_Albedo`, `Skin_Roughness`, `Skin_Bump`, `Skin_LipMask`) with fair warm porcelain-peach base, soft rosy blush, delicate orbital/crease shading, coral-rose lip vermilion, subtle lip gloss mask, and restrained micro-pores.
  - Polar-mapped `Eye_Iris_Albedo` (1024×1024) with rich dark warm espresso-chestnut iris fibers, subtle lower-iris honey warmth, limbal ring, and natural off-white sclera.
  - Seamless tileable chunky knit maps (`Knit_Color`, `Knit_Height`, `Knit_Rib_Height`, 1024×1024) modeling interlocking stockinette V-stitches and 1×1 ribbed knit for the collar, cuffs, and hem.
- `materials.py`: Physically based Cycles node trees (`Mat_Skin` with Random Walk SSS and lip coat sheen, `Mat_Eyeball` + Fresnel `Mat_Cornea`, Chiang `Principled Hair BSDF` for chestnut hair/brows/lashes, wool knit + sheen + bump/displacement materials, and cafe background materials).
- `sweater.py`: Constructs a 3D chunky dusty-pink knitted crew-neck sweater (`Sweater_Torso`, `Sweater_Collar`, `Sweater_Sleeve_R/L`) with smooth dropped-shoulder junctions, forward-reaching ruched sleeves ending in recessed hollow ribbed cuffs, Subdivision Surface + UV `Displace` stitch geometry, and 4,200 translucent `Sweater_WoolFuzz` micro-fibers.
- `hair.py`: Multi-layer Blender 5.1 `Curves` (`hair_curves`) system (~15,400 head/brow/lash strands) with a fitted `Hair_Scalp_Cap`, scalp-hugging geodesic + gravity-draped `Hair_Main_Volume`, 5 wavy front-left chestnut locks (`Hair_Front_Cascade`), airy see-through `Hair_Bangs_Wispy`, backlit `Hair_Flyaways`, and surface-snapped 3D `Eyebrows_Curves` and `Eyelashes_Curves`. Every layer runs `BVHTree` collision resolution against both `Character_Body` and the sweater meshes.
- `cafe_env.py`: Builds lightweight 3D sunlit cafe bokeh geometry (sunlit left window, potted plants, oak chairs/tables, glass flower vase, rear espresso bar and shelves), a 5-light soft daylight portrait lighting rig, `Camera_Beauty` (38mm, shallow DoF f/2.6 focused on eyes), and four turnaround diagnostic cameras (`0°`, `45°`, `90°`, `180°`).

## Inferred Unseen Anatomy & Clothing
Because `input/reference.png` is a single front-facing selfie photograph, the following elements are inferred consistently across all 360° diagnostic views:
- Side facial profile, nose bridge/chin depth, right ear, and back of the skull/nape follow balanced youthful feminine proportions derived from the MakeHuman hm08 topology.
- Back and side hair volume continues the visible chestnut wavy texture down to mid-back, tucking smoothly behind the left ear so the ear and profile remain visible in the `three-quarter` and `side` views.
- The unseen lower back of the dusty-pink sweater finishes in a matching chunky ribbed waist hem, and the forward-reaching sleeves finish cleanly in closed, dark-recessed ribbed cuffs rather than modeling unseen hands or lower torso/legs.

## Self-Review & Honest Limitations
**What works well:**
- The overall composition, gentle head tilt, fair warm skin tone with Random Walk SSS, warm brown eyes with crisp corneal catchlights, soft glossy closed-mouth smile, dusty-pink chunky V-stitch knitted crew-neck sweater with wool fuzz, and sunlit cafe bokeh closely capture the mood and color harmony of `input/reference.png`.
- All four turnaround diagnostic views (`front`, `three-quarter`, `side`, `back`) use the exact same unmodified character geometry and materials against a plain neutral studio backdrop, confirming clean 3D facial profile depth, seamless sweater shoulders, recessed cuffs, and full back hair/sweater coverage without bald scalp patches or back-collar clipping.
- Final asset stays well within resource limits (`243,703` evaluated mesh vertices, `19,596` curve strands, all textures `<= 1536px`, folder well under 1.5 GB).

**Remaining limitations & approximations:**
- **Facial likeness:** Procedural parametric sculpting on the hm08 base mesh approximates the reference's youthful East Asian facial structure and gentle smile, but does not achieve exact photographic identity likeness; the lower cheeks, philtrum, and eye shape remain slightly more generic than the reference photograph.
- **Hair intersections & micro-flow:** Hair is generated from analytical scalp-hugging guides and BVH collision rather than strand physics simulation. On close inspection in `beauty.png`, a small dark edge shadow appears near the right outer temple/cheekbone where dense front-cascade strands pass closely along the cheek, and crown strands show slight procedural grouping regularity compared to real hair.
- **Left ear reveal & wispy bangs:** The left ear is subtly framed rather than as openly tucked behind hair as in the reference photo, and the see-through bangs are slightly softer/thinner at the forehead tips.
- **Hands & lower body:** Arms terminate cleanly inside hollow ribbed sweater cuffs; unseen hands and lower body are not modeled.

## Third-Party Base Mesh Attribution
- **MakeHuman hm08 Base Mesh (`input/upper-body.obj`, `input/base.obj`)**: Released under **CC0** (Public Domain) by the MakeHuman Community.
- Pinned upstream source (per `input/ASSET-LICENSE.md`): `https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj`
