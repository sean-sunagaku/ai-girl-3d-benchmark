# Procedural 3D Upper-Body Portrait (Blender 5.1.1)

## Reference Photograph Access Statement

Yes — `input/reference.png` (`1205x1306`) was directly inspected as a decoded photograph using an image-reading tool before designing or writing any scene code, alongside `input/environment.json` and `input/ASSET-LICENSE.md`. Six rounds of rendered Blender previews (`beauty`, `front`, `three-quarter`, `side`, `back`) were also visually inspected as decoded PNGs to iteratively correct proportions, rigging, hair flow, and shading.

The supplied photograph is **never** loaded, cropped, projected, or packed into `character.blend` or any material, texture, or background plane. Every visible surface, strand, and texture in `character.blend` and `renders/*.png` is genuine 3D geometry and procedural material/texture synthesis rendered directly by Blender 5.1.1 Cycles CPU without post-processing.

---

## Reproducible Build, Open, and Render Commands

Run from this working directory:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

What `build.py` does end-to-end:
1. Synthesizes seven procedural PNG maps (`<= 2048x2048`) into `textures/` and packs them into Blender datablocks.
2. Loads `input/upper-body.obj` (plus the contiguous waist margin ring down to `Z=1.04 m` and exact MakeHuman `UVMap` coordinates from `input/base.obj`), sculpts the continuous feminine facial and upper-body proportions, builds multi-layer 3D eyes, creates `Character_Armature`, and poses the character into the tilted-head cafe selfie posture.
3. Constructs the 3D `Character_Sweater` mesh with rolled ribbed crew-neck collar, cuffs, closed waistband hem, sculpted folds, and physical stitch displacement.
4. Generates collision-free 3D `bpy.data.hair_curves` (`Hair_Underlayer`, `Hair_Main_Waves`, `Hair_Wispy_Bangs`, `Hair_Framing_Tendrils`, `Hair_Flyaways`, `Character_Eyebrows`, `Character_Eyelashes`), `Cafe_Environment`, `Diagnostic_Backdrop`, `Lighting`, and all five cameras.
5. Packs all resources (`bpy.ops.file.pack_all()`) and **saves `character.blend` before rendering** with `Cam_Beauty` active and the character + cafe environment ready to inspect.
6. **Reopens `character.blend` from disk** to verify portability, measures base and evaluated geometry counts into `scene-report.json`, and renders:
   - `renders/beauty.png` (`960x1040`, `Cam_Beauty` with 3D `Cafe_Environment` and `f/2.0` optical DOF)
   - `renders/front.png`, `renders/three-quarter.png`, `renders/side.png`, `renders/back.png` (`768x768` at `0°`, `45°`, `90°`, and `180°` around the upper-body asset against the plain neutral `Diagnostic_Backdrop`, with zero changes to character geometry or materials)

### Fast Preview Flag (Optional)
For rapid iteration at lower resolution/samples:
```bash
PREVIEW=1 SAMPLES=16 RENDER_SCALE=0.5 /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

### Opening Interactively in Blender
```bash
/Applications/Blender.app/Contents/MacOS/Blender character.blend
```
All collections (`Character`, `Eyes`, `Hair`, `Cafe_Environment`, `Diagnostic_Backdrop`, `Lighting`, `Cameras`), modifiers (`Armature`, `CorrectiveSmooth`, `Subdivision`, `KnitRelief`), curve hair objects, and node-based Cycles materials remain live and editable.

---

## Architecture, Geometry & Material Techniques

- **Continuous Anatomical Facial & Upper-Body Sculpting (`character_lib/anatomy.py`)**:
  - Transforms the neutral androgynous MakeHuman `hm08` mesh into balanced young East Asian feminine proportions: shortened midface and philtrum anchored at eye level (`Z = 1.5453 m`), softened supraorbital bossing, rounded forehead, fuller apple/buccal cheeks with 24-pass selective Laplacian relaxation over MakeHuman's adult marionette/nasolabial loops, V-oval jawline and softly rounded chin, slender neck, and narrower feminine shoulders/arms.
  - Sculpted palpebral apertures with tapered East Asian double-eyelid creases, subtle lower-lid *aegyo-sal* fullness, snug waterline wrap around the 3D eyeballs, refined nasal bridge/alar width (`~28 mm`) with recessed nostrils, and plump sealed lips in a gentle closed-mouth smile.
- **Editable Armature Rig (`Character_Armature`)**:
  - Places the `head` bone pivot at the anatomical C1 skull base (`Z = 1.442 m`) with rigid skull/mandible skinning weights so the `~17.5°` head tilt toward her right shoulder never shears the lower face or smile, plus `body_smooth_only` vertex-group masking so `CorrectiveSmooth` relaxes neck/shoulder bends without blurring sculpted lips or eyelids.
- **Multi-Layer 3D Eyes (`Eyes` collection)**:
  - `Eyeball_Inner_L/R` (concave anterior iris chamber mapped to `1024x1024` procedural warm dark-chestnut stroma, crypts, collarette, and dark limbal ring) + `Eyeball_Cornea_L/R` (anterior corneal dome bulge using Fresnel `IOR = 1.376` specular reflections and shadow-ray pass-through for crisp 10:30 window catchlights at 32 samples) + `Eye_Caruncle_L/R` + `Eye_TearLine_L/R` wet lower-lid meniscus strips.
- **Stratified Scalp-Sampled, Collision-Aware 3D Hair (`character_lib/hair.py`)**:
  - `14,138` editable Catmull-Rom 3D curves (`280,205` control points) rendered as thick 3D curves in Cycles with `Principled Hair BSDF` (`MELANIN` parametrization, warm dark chestnut `0.78–0.87` melanin with `0.44` redness and strand-to-strand randomness).
  - Follicles are stratified directly across the posed scalp mesh triangles (`scalp_mask`) to guarantee uniform root density without crown or back bald gaps, projected outside both `Character_Body` and `Character_Sweater` via `BVHTree` signed-distance checks so waves drape over the raised right shoulder onto the front-left sweater bodice while tucking behind the visible left ear on the right side.
  - Separate editable layers for `Hair_Underlayer`, `Hair_Main_Waves`, whisper-fine `Hair_Wispy_Bangs` ("air bangs"), `Hair_Framing_Tendrils`, `Hair_Flyaways`, `Character_Eyebrows`, and eyelid-rim-interpolated `Character_Eyelashes`.
- **3D Knitted Crew-Neck Sweater (`character_lib/clothing.py`)**:
  - Relaxed all-quad garment shell lofted `10–26 mm` off the torso/arms with 28-pass garment span relaxation, recessed drop-shoulder seams, sculpted shoulder/elbow compression folds, 3D extruded rolled ribbed crew-neck collar, wide cylindrical chunky ribbed cuffs, and closed ribbed waistband hem.
  - Subdivided (`Subdivision` level 2) with a physical `Displace` modifier (`KnitRelief`) driven by `textures/knit_height_2k.png`, paired with `Mat_Sweater_Knit` and `Mat_Sweater_Rib` (vertical fisherman's-rib/stockinette wales, Oren-Nayar diffuse roughness, fuzzy `Sheen`, and subtle warm yarn subsurface scattering).
- **Skin Subsurface Scattering & Seamless Coloration (`character_lib/materials.py`)**:
  - `Mat_Skin` uses `Principled BSDF` (`RANDOM_WALK_SKIN`, physical `Subsurface Scale = 0.0072 m`, `Subsurface Radius = (1.0, 0.36, 0.21)`), seamless 3D rest-space facial color zones (`skin_tone` attribute: fair warm ivory complexion, malar peach blush, subtle taupe upper-lid & tightline shading, hydrated coral-rose lips, recessed nostril darkening, dark scalp root shadow) with zero neck/jaw UV seams, and box-projected `2048x2048` cellular micro-pore bump/roughness (`textures/skin_micro_2k.png`).

---

## Observed vs. Inferred Anatomy and Scene Elements

- **Directly observed in `input/reference.png`**: Frontal-oblique facial proportions, ~17–19° head tilt toward her right shoulder, warm dark-brown irises with upper-left window catchlights, double eyelids and subtle *aegyo-sal*, light see-through bangs, right-side wavy chestnut hair cascading over the front-left sweater, visible left ear with tucked side hair, gentle closed-mouth coral-pink smile, dusty-pink ribbed crew-neck pullover sweater with forward-reaching selfie sleeves, and softly blurred sunlit cafe interior (window plant on left; wooden chair, table, white flower vase, counter, and shelves on right).
- **Inferred / Approximated (not visible in single frontal photo)**:
  - True 90° lateral nose/lip/chin depth profile (`side.png`) and posterior cranial curvature.
  - Full back-of-head hair layering and wave distribution (`back.png`).
  - Posterior sweater back panel drape, ribbed waist hem termination (`Z ~ 0.99 m`, cleanly capped since lower body/legs are unseen and out of scope), and forward sleeve cuff terminations outside the selfie crop (`front.png`, `three-quarter.png`, `side.png`).
  - Simplified 3D cafe furniture, window mullions, potted ficus leaves, shelves, jars, and flower vase constructed purely to provide authentic 3D depth-of-field bokeh and warm bounce lighting.

---

## Iterative Self-Review & Honest Likeness Assessment

During development we rendered and inspected six preview iterations before producing the final 32-sample renders:
1. **Preview 1** caught cumulative pose-bone angles lifting the right arm in front of the face, excessive upward hair launch, ginger-light hair melanin, and overexposed skin albedo (`sRGB 0.94`).
2. **Preview 2** verified calibrated skin exposure, rich chestnut hair melanin, and vertical knitwear columns, while revealing that `lip_mask` targeted `Z = 1.493 m` (philtrum) instead of MakeHuman's actual stomion at `Z = 1.4771 m`, plus sparse crown/temple hair angles.
3. **Preview 3 & 4** aligned the lips to the exact stomion, switched to stratified scalp-triangle follicle sampling (eliminating crown and back bald gaps), replaced clumped bangs with continuous whisper-fine air bangs, and moved the `head` bone pivot to `Z = 1.442 m` with rigid skull/jaw weights so head tilt stopped shearing the jaw.
4. **Preview 5 & 6** shortened the neutral Western midface/philtrum for youthful East Asian proportions, erased residual perioral marionette creases via selective cheek Laplacian relaxation, widened the eyebrows, draped S-wave curls onto the front-left sweater bodice, and interpolated eyelash roots to `0.1 mm` precision along the sculpted eyelid rim.

### Honest Closeness & Remaining Limits
- **What matches well**: Upper-body selfie framing, head tilt, warm sunlit cafe mood and bokeh, dusty-pink chunky vertical-ribbed crew-neck sweater and folds, chestnut hair color with backlit rim glow, airy see-through bangs, smooth fair-warm SSS skin, dark-chestnut irises with catchlights, and clean 360° turnaround completeness (`149,232` character evaluated mesh vertices + `280,205` curve points; `69 MB` total folder size).
- **Remaining limits**: Because facial features are procedurally sculpted over subdivided MakeHuman `hm08` topology rather than a high-resolution scan, subtle individual likeness nuances—such as exact soft tissue fullness around the nasal alae, natural micro-asymmetry in the smile corners, and ultra-fine loose curl clumping—remain stylized 3D approximations and still read slightly more CG-smooth than a real photograph, especially under the shared 32-sample CPU + OpenImageDenoise budget.

---

## Third-Party Asset Attribution

- **Base Mesh**: `input/upper-body.obj` and `input/base.obj` derive from the **MakeHuman `hm08` base mesh**, released by the MakeHuman Community under **CC0 1.0 Universal (Public Domain Dedication)** as documented in `input/ASSET-LICENSE.md` (`https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj`).
- All sculpting deformations, clothing geometry, 3D hair/brow/lash curves, eye assemblies, cafe props, procedural textures, and Cycles shader networks were created procedurally from scratch in this workspace.
