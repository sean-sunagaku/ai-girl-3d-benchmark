# Procedural 3D Upper-Body Portrait in Blender 5.1.1

## Reference Photograph Access
Yes — I directly inspected and visually decoded `input/reference.png` (`1205x1306`) using an actual image-reading tool before designing or writing any geometry/shader code, and I also visually inspected five rounds of intermediate Blender preview renders plus all five final full-resolution renders (`renders/*.png`). The reference photograph was used strictly for visual observation; no crop, projection, texture sample, camera-facing plane, or background plate from `input/reference.png` is used anywhere in the scene.

## How to Build, Open, and Render

### Full Reproducible Build & Render (from Supplied Inputs)
Run from this working directory:
```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```
This command:
1. Generates seven procedural `<=1024px` texture maps in `generated_textures/` (`lib_textures.py`).
2. Builds physically based Cycles node materials (`lib_materials.py`).
3. Loads `input/upper-body.obj` (mapping canonical UVs from `input/base.obj`), refines subdivision, sculpts the facial likeness, constructs multi-layer eyeballs, brows, and lashes, and poses the gentle head tilt (`lib_anatomy.py`).
4. Constructs the dusty-pink knitted crew-neck sweater, forward-extending bent-elbow sleeves, cuffs, hands, and wool halo fuzz (`lib_sweater.py`).
5. Grooms 23,488 collision-resolved 3D Catmull-Rom hair, brow, lash, and fuzz curves (`lib_hair.py`).
6. Builds the 3D sunlit cafe interior, natural area lighting, `Beauty_Camera`, and four turnaround cameras (`lib_scene.py`).
7. Packs all generated textures into `character.blend` and saves `character.blend` **before** rendering.
8. Renders `renders/beauty.png` (`960x1040`) and `renders/front.png`, `renders/three-quarter.png`, `renders/side.png`, `renders/back.png` (`768x768`), then writes `scene-report.json`.

### Optional Fast Preview Mode / Build-Only Mode
```bash
# Fast lower-sample preview pass into previews/
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --preview

# Construct and save character.blend without rendering
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --no-render
```

### Opening Interactively in Blender
```bash
/Applications/Blender.app/Contents/MacOS/Blender character.blend
```
All collections (`Character`, `Cafe_Environment`, `Lighting`, `Cameras`), live `SUBSURF` and `DISPLACE` modifiers, `CURVES` hair objects, and shader node trees remain directly editable, with all generated textures packed inside `character.blend`.

## Geometry & Material Techniques

- **Continuous Sculpted Face & Neck (`Character_Head_Neck`)**: Starting from `input/upper-body.obj` (with clean UV coordinates recovered via 1-to-1 vertex/face topology inspection of `input/base.obj`), one level of Catmull-Clark refinement is applied before smooth 3D anatomical deformation fields sculpt soft East Asian feminine proportions: compact facial thirds, softened brow bossing, gentle V-oval jaw taper, fuller malar apple cheeks, a petite integrated nose with soft nostrils, subtle supratarsal double-eyelid creases, gentle pretarsal (*aegyo-sal*) fullness, and a closed-mouth smile with lifted commissures. Topological BFS flood-filling isolates the internal oral cavity so outer cheeks remain clean, and a live `SkinSubsurf` modifier produces a smooth continuous surface.
- **Skin Shading (`Mat_Character_Skin`)**: Uses Blender 5.1's `Principled BSDF` with `RANDOM_WALK_SKIN` subsurface scattering (`Subsurface Scale = 6 mm`, RGB scattering radii `(1.0, 0.38, 0.22)`), seamless 3D canonical `rest_co` vertex attributes (rosy cheek/ear/nose warmth, feathered rose-coral lip vermilion with vertical lip micro-creases, upper lash-line definition, and nostril AO — avoiding any neck UV seam), and a tileable `1024x1024` Voronoi cellular skin micro-pore/roughness map (`skin_micro_detail.png`).
- **Multi-Layer Anatomical Eyes & Wrapping Eyelids**: Each eye combines an outer sclera/cornea mesh (`Eye_Outer_L/R`, `13.6 mm` radius with a `+0.85 mm` anterior corneal dome using `IOR 1.376` refraction mixed with `Transparent BSDF` on Cycles shadow/diffuse rays so natural cafe window light illuminates the iris while keeping crisp specular catchlights), a concave inner iris/pupil disc (`Eye_Iris_L/R`, `13.0 mm` warm chestnut-amber radial stroma & dark limbal ring), and a wet lower-lid `Tear_Meniscus_L/R`. Inner eyelid rims are radially conformed to the eyeball spheres (`0.2 mm` clearance), accented with curved 3D `CURVES` upper/lower eyelashes and directional ray-cast eyebrows.
- **Collision-Resolved 3D Chestnut Hair (`CURVES` + `CHIANG` Melanin Shader)**: Native Blender 5.1 Catmull-Rom `CURVES` objects (`Character_Hair_Bangs`, `Character_Hair_Cascade_Right`, `Character_Hair_Drape_Left`, `Character_Hair_Back`, `Character_Hair_Flyaways`, over a recessed posterior `Hair_Volume_Core`) are routed across the posed skull and projected outside `BVHTree` colliders for both the head and sweater. Airy see-through wisps and a right-swept diagonal temple lock frame the forehead while keeping the left ear and 90° facial profile clear; loose S-waves cascade over the right shoulder onto the front sweater chest. Shaded with `Principled Hair BSDF` (`CHIANG`, `Melanin = 0.81`, `Melanin Redness = 0.45`).
- **Dusty-Pink Knitted Crew-Neck Sweater**: Parametric quad meshes (`Sweater_Torso`, `Sweater_Collar`, `Sweater_Sleeve_R/L`) use live `SUBSURF` + true 3D `DISPLACE` modifiers driven by a procedural `1024x1024` fisherman-rib V-stitch and purl-bar height map (`sweater_knit_height.png`), seamless 3D `BOX` projection on the torso to avoid radial collar distortion, `Principled BSDF` microfiber sheen, subsurface yarn scattering, and `Sweater_Wool_Fuzz` halo fibers for sunlit rim softness.

## Approximations & Inferred Unseen Anatomy
- **Inferred Features**: Because `input/reference.png` is a close upper-body selfie cropped above the waist and outside the wrists/hands, the following elements are inferred procedurally to form a complete 3D asset:
  - Back of the cranium, nape, and full cascading back hair (`Character_Hair_Back`, visible in `back.png` and `side.png`).
  - Side and back panels of the knitted sweater, lower ribbed waist hem (`Z = 0.90 m`), elbow bend geometry, ribbed sleeve cuffs, and softly curled hands/fingers (`Character_Hand_L/R`) at the distal sleeve ends outside the beauty camera frame.
  - 3D cafe furniture, sunlit window panels, potted foliage, counter/shelves, and tabletop flower vase positioned in depth for natural `f/2.2` portrait bokeh.
  - All unseen lower-body anatomy below the sweater hem is omitted by design.

## Honest Self-Review
- **Strengths**: Matches the overall upper-body selfie composition, ~18.5° head tilt, raised right shoulder, forward-extending dusty-pink ribbed sleeves, visible fisherman-rib knit texture, exposed left ear, airy see-through bangs with diagonal right-swept wisps, rich chestnut chest waves, warm brown irises with window catchlights, gentle closed-mouth smile, and sunlit cafe bokeh atmosphere without any photographic textures or billboards. All diagnostic views (`0°, 45°, 90°, 180°`) use the exact same character geometry and materials and show an unobstructed side profile and complete back hair/sweater.
- **Remaining Limits**: Compared to a real photograph, smooth parametric deformation of the MakeHuman base topology still looks slightly idealized/CG-smooth in close inspection of the midface and eyelids; sculpted cuff hands in the zoomed-out diagnostic turnarounds are simplified softly curled forms; and procedural spline hair/cloth lacks full strand-to-strand physical clumping and micro-wrinkle complexity of dynamic simulation.

## Base-Mesh Attribution
- `input/upper-body.obj` and `input/base.obj` derive from the **MakeHuman `hm08` base mesh** (Copyright (C) 2020 Jonas Hauquier / MakeHuman Community), explicitly released under **CC0 (Public Domain)** as documented in `input/ASSET-LICENSE.md`.
