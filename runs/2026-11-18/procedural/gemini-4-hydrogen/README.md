# Upper-Body Procedural Portrait Scene (Blender 5.1.1)

## Visual Inspection of `input/reference.png`

`input/reference.png` (1205×1306) was decoded and inspected directly before sculpting, texturing, rigging, and lighting. The reference photograph depicts a young East Asian woman seated in a softly backlit cafe in an upper-body selfie pose:

- **Pose & framing**: Head gently tilted toward her right shoulder (viewer's left) with a subtle leftward turn exposing a glimpse of her left ear (viewer's right), direct warm gaze into the lens, and arms reaching slightly forward toward the bottom corners of the frame.
- **Expression & skin**: Fair warm porcelain-peach skin with soft rosy cheek blush, small delicate nose, large expressive warm chestnut-brown eyes with the upper eyelid gently grazing the top of the iris and crisp left-side window catchlights, and a relaxed closed-mouth smile with coral-rose lips.
- **Hair**: Long wavy dark-chestnut hair parted lightly across the crown with delicate wispy see-through bangs above/near the brows, soft face-framing strands tucked toward the left side behind the ear, and a thick cascade draping forward over her right shoulder (viewer's left).
- **Garment & backdrop**: A chunky dusty rose-pink crew-neck knitted sweater with 2×2 ribbing at the collar, cuffs, and waist hem, set against a bright sunlit left-side cafe window with greenery, a curved wooden chair rail behind her right shoulder, oak shelves/counter, and a small white floral vase blurred in the right background.

---

## Build and Open Commands

Run from this project directory using Blender 5.1.1 (CPU Cycles, 3 threads, factory startup):

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

Open the packed scene interactively or inspect it directly in Blender:

```bash
/Applications/Blender.app/Contents/MacOS/Blender character.blend
```

---

## Pipeline Architecture & Design Choices

All scene geometry, materials, procedural textures, hair curves, rig weights, cafe props, and lighting are built deterministically by `build.py` and its modular helpers:

1. **Facial Sculpting, Hands, UVs & Multi-Layer Eyes ([`character_mesh.py`](character_mesh.py))**
   - Imports `input/upper-body.obj` (MakeHuman `hm08` CC0 base mesh) with `forward_axis='Y', up_axis='Z'` so `-Y` faces forward in Z-up space.
   - Procedurally sculpts the jawline toward a softer V-oval contour, shapes fullness in the cheeks, refines the nasal bridge/tip, curves a closed-mouth smile seam (`lip_inner` closure with zero gap), adds gentle under-eye *aegyo-sal* contours and double-eyelid creases, and sets natural almond-rounded eyelids hugging the eyeballs with radial sphere clearance.
   - Extends the truncated forearm rims cleanly to wrists with manifold 5-digit relaxed hands, caps the lower torso edge safely inside the sweater, and generates seam-free cylindrical UVs covering the head/neck.
   - Constructs four-component eyes per side (`Eye_Inner`, `Eye_Cornea`, `Eye_TearLine`, `Eye_Caruncle`): `Eye_Inner` maps a forward `-Y` pole to the procedural iris/sclera texture, while `Eye_Cornea` uses a `Transparent + Fresnel Glossy` node mix to deliver crisp window catchlights without refractive shadow/caustic noise.

2. **Procedural Textures ([`textures.py`](textures.py))**
   - Generates and packs 10 PNG textures directly into `character.blend` (and saves copies under `generated_textures/`):
     - `skin_albedo.png`, `skin_roughness.png`, `skin_bump.png` (2048×2048): landmark-registered cheek blush, warm coral-rose lip gradient, subtle lash-line espresso root warmth, T-zone/lower-lip glossiness variation, and fine pore relief.
     - `iris_eye.png` (1024×1024): warm chestnut-brown iris (`r_iris = 0.296`) with radial collagen striations, amber collarette, dark limbal ring, lower light-scatter crescent, and natural ivory sclera.
     - `knit_albedo.png`, `knit_height.png`, `knit_roughness.png`, `knit_rib_albedo.png`, `knit_rib_height.png` (1024×1024): tileable dusty rose-pink melange V-stitch stockinette and 2×2 rib textures with wool fiber fuzz noise.
     - `cafe_oak_wood.png` (512×512): procedural warm oak grain for furniture surfaces.

3. **Armature Skinning & Selfie Posing ([`rig_and_pose.py`](rig_and_pose.py))**
   - Builds `Character_Rig` from `input/environment.json` joint landmarks (`spine_base`, `chest`, `neck`, `head`, `eye.L/R`, `clavicle.L/R`, `upper_arm.L/R`, `forearm.L/R`, `hand.L/R`), computes smooth analytical capsule skinning weights, and attaches eye components rigidly to `eye.L/R` and `head`.
   - Applies the portrait head tilt (~19° roll toward her right shoulder, ~7° yaw, gentle forward lean), reaches both arms forward toward the lower foreground corners, and aims both eyes directly at the `Beauty_Camera` optical axis.

4. **Tailored Chunky Knit Sweater & Wool Fuzz ([`sweater_builder.py`](sweater_builder.py))**
   - Constructs the sweater torso, closed shoulder yoke, flat-lying 2×2 ribbed crew-neck collar, ribbed waist hem, parallel-transported zero-twist sleeves, and 2×2 ribbed cuffs directly in posed space around the evaluated `Character_Body` `BVHTree` with surface normal checks and a final skin-clearance push-out pass (`>= 10.5 mm`).
   - Adds yarn relief bump maps (`ShaderNodeBump` + Principled BSDF wool sheen and subtle SSS) plus 3,600 procedural `Sweater_Fuzz` curves for soft backlit silhouette fuzziness.

5. **Chestnut Hair, Wispy Bangs, Brows & Lashes ([`hair_builder.py`](hair_builder.py))**
   - Uses a recessed dark `Hair_VolumeCap` beneath procedural `CURVES` primitives shaded with `Principled Hair BSDF` (`Melanin` absorption):
     - `Hair_Main`: multi-zone clumped guide waves with BVH collision sliding (right-shoulder chest cascade, left temple-to-behind-ear sweep, and full back drape).
     - `Hair_Bangs_Sheer`: upper crown-to-forehead fringe blended over the skull contour plus 11 wispy see-through bang locks.
     - `Hair_FaceFraming` & `Hair_Flyaways`: soft outer cheekbone tendrils and backlit silhouette flyaways.
     - `Eyebrows_3D` & `Eyelashes_3D`: curved, tapered brow hairs and dense short upper/lower eyelashes extracted directly from the sculpted eyelid lash margins.

6. **Cafe Backdrop, Lighting & Cameras ([`cafe_scene.py`](cafe_scene.py))**
   - Builds out-of-focus background props in `CafeScenery` (curved oak chair back, foreground table edge, sunlit left window with potted plant leaves, oak counter, wall shelves with ceramic mugs, and a glass vase with white stems/blossoms).
   - Lights the portrait with a warm sunlit left window area key (`Key_CafeWindow`), backlit rim (`Rim_BackWindow`), soft interior fill (`Fill_CafeInterior`), and ceiling bounce (`Fill_TopBounce`).
   - Sets up `Beauty_Camera` (33 mm lens, f/2.6 depth of field focused on the midpoint between the eyes) for `renders/beauty.png` (960×1040), plus 50 mm turnaround cameras (`Cam_Front`, `Cam_ThreeQuarter`, `Cam_Side`, `Cam_Back`) rendered at 768×768 after hiding `CafeScenery` on a neutral studio background (without overwriting the already-saved `character.blend`).

---

## Self-Review & Approximations

### Comparison Against `input/reference.png`
- **Matches well**: Head tilt angle and relaxed neck framing, sunlit cafe window-left illumination with crisp corneal catchlights, warm chestnut iris tones, gentle closed-mouth coral smile, wispy see-through bangs parting lightly across the brows, asymmetrical hair cascade draping over her right shoulder, chunky dusty-pink knit sweater texture/ribbing, and softly defocused cafe depth (chair rail, greenery, warm wood, white floral accent).
- **Remaining gaps vs. photo**:
  - Procedural sculpting from the MakeHuman base retains slightly smoother, more generalized facial plane transitions around the malar cheeks, eyelids, and philtrum compared with real photographic skin micro-contrast.
  - Procedural curve bangs and `Principled Hair BSDF` under 32-sample Cycles CPU + denoising read slightly more uniform/matte along the hair ends than individual backlit photographic strands.
  - Upper eyelashes and brow tails are softer and more restrained at portrait distance than macro photographic hair definition.

### Inferred / Unseen Elements
Because `input/reference.png` shows only a front-facing upper-body portrait:
- **Back of head, hair length, and back sweater drape**: Inferred as full shoulder-blade-length layered wavy hair flowing cleanly over a matching continuous back knit panel without gaps.
- **Side profiles (`90°` & `45°`)**: Ear contours, occiput volume, shoulder yoke slope, and elbow sleeve folds are constructed cleanly for full 360° diagnostic inspection.
- **Lower arms, hands, and waist termination**: The reaching arms continue into knit 2×2 ribbed cuffs and relaxed 5-digit posed hands, while the sweater terminates cleanly with a ribbed waist hem around `z ~ 1.00 m`.
- **Cafe background**: Only the softly blurred chair rail, window frame, greenery, warm wooden counter/shelves, and white flowers behind the subject are represented as minimal background geometry.

---

## Measured Scene Outputs

- **Saved Scene**: `character.blend` (saved and reopened prior to rendering, all 10 generated textures packed)
- **Renders** (`CYCLES` CPU, 32 samples with denoising, `AgX` view transform):
  - `renders/beauty.png` (960×1040)
  - `renders/front.png` (768×768)
  - `renders/three-quarter.png` (768×768)
  - `renders/side.png` (768×768)
  - `renders/back.png` (768×768)
- **Scene Report**: `scene-report.json` (17,239 base mesh vertices / 68,225 evaluated vertices; 18,550 total curves / 331,404 control points across hair, brows/lashes, and sweater fuzz)

---

## Asset Attribution

The base upper-body mesh topology (`input/upper-body.obj`) and anatomical joint landmarks (`input/environment.json`) originate from **MakeHuman `hm08`** under the **CC0** public domain dedication, as documented in [`input/ASSET-LICENSE.md`](input/ASSET-LICENSE.md) (pinned commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`).
