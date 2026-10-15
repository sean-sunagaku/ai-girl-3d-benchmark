# Cafe selfie portrait — Blender 5.1.1 3D character (second pass, photo-textured)

This is an editable 3D upper-body character built and rendered in the installed Blender 5.1.1
(`/Applications/Blender.app/Contents/MacOS/Blender`) with bpy scripts only. Finals use Cycles on the CPU with
3 threads, 32 samples and OpenImageDenoise.

The first pass was procedural only, because photographic character textures were not allowed then. That restriction
has been lifted for this pass. The supplied photograph is now used in three ways:
- to fit the camera and the face shape
- as a camera-projected skin texture, baked into a real UV layout
- for the iris colour and the knit colour

The geometry is a sculpted, posed 3D mesh with curve hair, separate eyeballs and solid knit clothing, so every view
renders the same model. Nothing in the final images is a photo plane or a composite. The person is not identified,
and no attempt was made to identify her.

## Image access (how I actually looked at things)
- **Reference photo – direct visual access.** I opened `input/reference.png` with the image-reading tool (a 922×1000
  copy, `previews/ref_small.png`, made with `sips`). I also wrote `tools/crop_grid.py`, which makes magnified crops
  with a pixel grid. I picked every landmark by eye on those crops: eye corners, lids, irises, nose, nostrils, mouth
  corners, lips, chin, cheek contour, ear and collar.
- **Numerical decoding.** I sampled colours numerically in Blender: skin, lips, iris, hair, the knit median and the
  sclera.
- **First-pass images – direct visual access.** The first-pass PNGs were not copied into this directory. I re-rendered
  them at half resolution from the copied first-pass scene (`previews/first-pass/pv_*.png`) and viewed them directly.
  They showed the face fully hidden behind the phone and arms in the front view, a doll-like procedural face, and a
  straight-cut wig-like fringe.
- **New images – direct visual access.** I viewed every intermediate preview, debug render and final render in
  `renders/` directly. These include texture-only emission renders used to check alignment, seams and stretching.

## Files
| file | content |
|---|---|
| `build.py` | Builds everything from `input/`, saves the packed `character.blend`, reopens and checks it, renders the five finals and writes `scene-report.json`. |
| `photo_fit.py` | Hand-picked photo landmarks, mesh landmark detection, camera projection, Gauss-Newton camera fit, RBF/similarity helpers. Imported by `build.py`. |
| `photo_texture.py` | Photo masking, pull-push inpainting, de-lighting, Cycles position/normal bake, visibility, thin-plate morph, iris texture. Imported by `build.py`. |
| `render.py` | Renders the five views from the saved scene. It is called by `build.py` and can also be run standalone (see below). |
| `scene_report.py` | Writes `scene-report.json` from the **reopened** `.blend`. |
| `character.blend` | The final editable scene (~23 MB). `Cam_Beauty` is active and all five images are packed. |
| `renders/beauty.png` | 960×1040 beauty render. |
| `renders/front.png`, `three-quarter.png`, `side.png`, `back.png` | 768×768 turntable views at 0/45/90/180°. |
| `scene-report.json` | Versions, engine/device/threads/samples, cameras, objects, bounds, measured counts, textures, fit errors and limitations. |
| `textures/` | Inspection copies of the generated images: `Skin_Photo_RGBA`, `Iris_Photo.L/R` and the two knit height maps. The `.blend` uses packed copies. |
| `source-character.blend` | The first-pass scene, kept **unmodified** for reference. `build.py` does **not** read it; the build starts from `input/`. |
| `previews/` | `ref_small.png`, the first-pass previews, `build.log` and the intermediate `build-report.json`. |
| `tools/` | Development helpers: landmark crops, face/eye close-ups, texture-debug renders, fit evaluation, and some first-pass inspection scripts. None of them is needed for the build. |

## Build / render
```sh
B=/Applications/Blender.app/Contents/MacOS/Blender
$B --background --factory-startup --threads 3 --python build.py        # full rebuild + 5 finals + report (~2 min 40 s here)
SKIP_RENDER=1 $B --background --factory-startup --threads 3 --python build.py   # scene only (~30 s)
$B --background --factory-startup --threads 3 character.blend --python render.py                        # re-render finals from the saved scene
$B --background --factory-startup --threads 3 character.blend --python render.py -- beauty side --preview  # half-res previews into previews/
```
The build reads only files in this directory. It is deterministic because it uses fixed RNG seeds and fixed
landmark data.

## What changed from my first pass
1. **Camera and face fitted to the photo** (`fit_to_photo` in `build.py`). The first pass was hand-tuned only. This
   pass works as follows:
   - **Landmarks.** 19 landmarks are picked on the photo. The matching points on the mesh are detected automatically:
     - Nose and mouth points are ray-cast on the canonical MakeHuman surface and carried through the sculpt by
       barycentric mapping.
     - Eye corners and lids are the extremes of the opening that is *actually visible from the camera*. This is
       measured on the subdivided surface by ray-casting against the fitted eyeball sphere.
     - The iris is the projected eyeball centre, because the eyes look into the lens.
   - **Camera fit.** A 6-DoF camera is fitted (Gauss-Newton, 26 mm-equivalent phone lens). It uses the irises, nose,
     mouth, eye corners and the visible ear, which pins the head yaw.
   - **Warps.** Three warps follow, all applied before posing, so the eyes, lashes, hair and sweater are built on the
     fitted face:
     - Per eye, a 3D similarity of the eye region: rotation about the view axis, uniform scale about the eyeball
       centre, and an image-plane shift.
     - Radius-preserving lid warps: opening the lids, and rotating the lids around the eyeball so the iris sits where
       it sits in the photo. The lids keep wrapping the eyeball.
     - A compact Wendland-RBF warp for the nose, lips, chin, five cheek-contour silhouette points and the ear.
   - **Error.** The mean landmark error drops from 10.4 px after the rigid fit to **1.5 px** (max 6.8 px) in the
     1205 px photo. The face became rounder and wider at the cheeks, and her left ear moved to where the photo shows it.
2. **Photographic skin layer.** See the texture method below. The skin is no longer only vertex colours.
3. **Photographic iris** with real catchlights from the cornea shader. The sclera was darkened, with an AO contact
   shadow. The lashes are denser and darker, and a tight-line attribute was added over the photo layer.
4. **Composition fixes.**
   - The phone is the camera. The 26 mm lens lets both hands actually reach it; the previous 35 mm fit put the
     camera out of arm's reach.
   - Her right hand grips the phone's lower corner, so no cuff opening appears in the selfie frame. Both arms leave
     the frame at the bottom corners, as in the photo.
   - The front and 45° turntable cameras are raised (~33°/17° elevation). The phone and hands now project onto the
     chest instead of hiding the face, which was a visible failure in the first pass.
5. **Hair.**
   - Her left side is tucked behind the ear (ear visible, as in the photo).
   - The bangs are longer and styled down the forehead. They hug the forehead instead of forming a visor once the
     head is tilted.
   - The bang strands are thinner and less clumped. There are 42k strands instead of 29k.
   - The colour is darker chestnut, and the waves are looser with a drifting wavelength.
   - A first attempt to also rotate the neck to match the photographed collar height broke the hair curtain at the
     nape, so it was removed. The ~60 px collar offset is reported instead.
6. **Clothing.**
   - A lower, wider crew neck with a thinner rib band.
   - More skin kept under the neckline, which removes a dark cut edge that was visible from the high camera.
   - The knit colour is sampled from the photo.
7. **Colour management.** Khronos PBR Neutral (exposure −0.25) replaced AgX Punchy, because AgX turned the photo-derived
   skin and the pink knit grey.

## Exact texture method
### `Skin_Photo_RGBA` (2048², UV map `SkinUV` on `Body_Skin`)
1. **UVs.** `SkinUV` is made with Smart UV Project (62°). The face splits into two mirrored islands.
2. **Photo-space cleaning** (`photo_texture.photo_skin_layers`).
   - **Skin mask.** A pixel counts as skin if it is bright and low in saturation, with R ≥ G ≥ B. Pixels darker than
     their neighbourhood (thin hair strands) are rejected.
   - **Added to the mask.** The lips (a polygon from the lip landmarks) and a thin ring around each eye opening (lash
     line and lid shading).
   - **Removed from the mask.**
     - the eye openings themselves, so sclera and iris pixels never land on the lids
     - everything above the brow line, where the photo's bangs read as skin
     - everything at or below the collar
     - everything outside a face ellipse
   - **Filling.** The masked pixels are filled by multiresolution pull-push from the surrounding skin.
   - **Partial de-lighting.** The photo is divided by a 22 px low-pass of the *skin-only* luminance (normalised
     convolution, power 0.55), so lashes and lips are not washed out.
3. **Projection into UV space.**
   - **Bake.** Cycles bakes the world position and normal of the posed, subdivided mesh into SkinUV (emission bake,
     with bake margins kept for seamless filtering).
   - **Projection.** Each texel is projected through `Cam_Beauty`, the same camera the face was fitted to. Self-occluded
     texels are removed by ray casts from the camera.
   - **Sampling.** Texels are sampled bilinearly through a thin-plate-spline morph built from the residual landmark
     offsets.
4. **Confidence (alpha).** Alpha is the product of four factors:
   - facing: a smoothstep of N·V from 0.25 to 0.55
   - visibility
   - a soft face ellipse
   - no scalp (`scalp_density`)

   Alpha also stops above the photographed collar line, so the collar shadow is not painted onto the neck. It is then
   slightly dilated to close hairline gaps at nostril rims and lid creases.
5. **Albedo scale.** Colours are scaled by 0.74 in linear space, because the photo is bright and over-exposed. The RGB
   is pull-push dilated into alpha-0 texels for clean filtering.
6. **Shader.** In the `Skin` material, Base Color = mix(procedural layer, photo RGB, photo alpha) × lash tight-line.
   Random-walk skin SSS, pores, micro bump and roughness variation are kept from the first pass.

### `Iris_Photo.L` / `Iris_Photo.R` (512²)
- **Mapping.** Planar along each eyeball's gaze axis (object coordinates through a Mapping node). Each texel is a point
  on the eyeball front, projected into the photo through the same camera and morph.
- **Catchlights.** Photo catchlights (luminance > 0.42) are removed and pull-push filled; the cornea shader makes real
  ones.
- **Alpha.** Alpha is the iris disc inside the photo's eye opening. The procedural iris and sclera show elsewhere.
  Colour is scaled by 0.46 (linear).

### Procedural / generated parts (no photo pixels)
- The skin colour/roughness vertex attributes from the first pass: blush, lips, ears, wet lid margin, mouth and scalp
  paint. They are tinted by the photo's median skin albedo `[0.784, 0.833, 0.924]` and form the layer under the photo
  everywhere.
- The `lash_line` attribute.
- The sclera, the cornea, and hair, brows and lashes (Principled Hair, Chiang).
- The two knit height tiles. Their colour ramp is built from the photo's sweater median, sRGB (231,169,165), scaled to
  an albedo.
- Hands, the phone and the cafe.

## Photo vs. unobserved areas
**Taken from the photograph:**
- the camera direction and the framing of the face
- 2D positions of eyes, lids, nose, mouth, chin, left cheek contour and left ear
- the frontal skin colour and shading of the face and upper neck, the lips, lid/lash-line makeup and irises
- the median complexion
- the knit colour

**Inferred (not visible, or not measurable from one image):**
- all depth: nose projection, cheek and jaw depth, skull, profile
- the back, sides and top of the head and the hidden right ear
- the scalp, the skin under the chin and the forehead under the bangs (inpainted)
- hair length, waves and behaviour at the back
- the torso shape, the sweater back, sides, hem and cuffs
- hands, fingers and the phone

## Editable resources in `character.blend`
- **Collections.** `Character` (with `Hair`, `Clothing`, `Eyes`), `Cafe_Set`, `Lighting`, `Cameras`.
- **Meshes.**
  - `Body_Skin`: UV map `SkinUV`, attributes `skin_col`, `skin_rough`, `lash_line`, `orig_co`, `rest_co`, and a
    Subdivision modifier.
  - `Eyeball.L/R`, `Cornea.L/R`.
  - `Sweater` and `Sweater_Collar`: `KnitUV` and a `rib` attribute.
  - `Hand_Palm.*`, `Finger.*`, `Thumb.*`, `Nail.*`, `Phone`, and the cafe props.
- **Native hair Curves.** `Hair_Strands`, `Hair_Bangs`, `Eyelashes`, `Eyebrows`.
- **Node materials.** `Skin` (the photo layer is the image node `Skin_Photo_RGBA`), `Skin_Hands`,
  `Eye_Sclera_Iris.L/R` (image node `Iris_Photo`), `Eye_Cornea`, `Hair_Chestnut`, `Lash_Hair`, `Brow_Hair`,
  `Knit_DustyPink`, `Nail`, `Phone_Case`, and the `Cafe_*` materials.
- **Images.** All five are packed: `Skin_Photo_RGBA` 2048², `Iris_Photo.L/R` 512², and the knit height maps 1024².
- **Lights.** `Key_Window`, `Fill_Front`, `Catchlight`, `Rim_Back`, `Rim_Back_R`, `Cafe_Ceiling`, and `Diag_Fill_Back`
  (turntable views only).
- **Cameras.** `Cam_Beauty` (26 mm, f/2.6 DOF on `Focus_Target`), plus `Cam_front`, `Cam_three-quarter`, `Cam_side`
  and `Cam_back` (95 mm).

The turntable renders use exactly the same objects, materials and textures as the beauty shot. They only exclude
`Cafe_Set`, switch to a grey world and enable `Diag_Fill_Back`.

## Counts (from `scene-report.json`, measured after reopening)
- **Mesh vertices.** The character has about 272k evaluated mesh vertices (272,455). The whole scene, including the
  cafe, has 304k.
- **Hair.** Native hair is counted separately: 42,347 curves with 1,339,162 control points. This covers head hair
  (38,639 strands plus 2,340 bang strands from 1,889 guides), 528 lashes and 840 brow hairs.
- **Size.** Character bounds are about 0.47 × 0.71 × 0.79 m, including the outstretched arms and the phone.

## Base attribution
Base mesh: **MakeHuman hm08 base mesh**, **CC0**, pinned at commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`:
https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj
I used the supplied cropped `input/upper-body.obj`. Sculpting, fitting, hair, clothing, materials and generated
textures are my own work in the scripts here. The only image source is the supplied `input/reference.png`. No other
models, textures, reconstruction models or generation services were used.

## Honest self-review
**Better than the first pass:**
- The face now lines up with the photo: eye placement and shape, irises, nose, smile line, chin, a rounder and wider
  face, and the visible left ear.
- The skin carries the photo's own complexion, lip colour and soft lid makeup, with no doubled eyes or features. I
  checked this with texture-only emission renders from the beauty camera.
- There is no stretching on the sides of the face; those areas fall back to the matched procedural layer.
- The irises are the photo's warm brown.
- The beauty frame reads like the reference: high close selfie, tilted head, arms leaving the bottom corners,
  fringe, long dark chestnut hair over her right shoulder, the ear tuck, and a dusty-pink crew-neck knit.
- All four turntable views show the same solid model. The face is visible in front and 45° views, there is a clear
  profile, and the back hair is continuous.

**Still visibly CG / limitations:**
- This is not a scan and not an exact likeness. All depth is inferred, and the fit uses only about 25 points from a
  single view.
- **Eyes.** They still read slightly large and "doll-like". The right (far) eye remains a few pixels off.
- **Skin.** It is smooth and airbrushed, because the source photo is beauty-filtered. Some photo shading is still
  baked in. Under the bright key light the nose bridge gets a strong highlight.
- **Hair.** It is a procedural groom, not photo-textured:
  - The crown reads a little flat and dark.
  - The wave and clump structure is more regular than real hair.
  - A white-ish backlit tuft over her left ear and a few stray strands show in the turntable views.
- **Torso and collar.** The 26 mm selfie perspective makes the torso smaller relative to the head than in the
  reference, and the hem is visible. The collar front sits about 60 px (in the photo) from the photographed neckline.
- **Sweater.** No cloth simulation; folds are procedural. The knit is a generated tile with only the colour sampled
  from the photo. A faint shadow line is still visible just inside the front of the collar, where the high camera
  looks down into the neckline.
- **Hands.** Crude tube fingers. They are out of frame in the beauty shot but visible in the turntables.
- **Scene and rig.** The cafe is a simple block set. There is no production rig.
