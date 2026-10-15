# Cafe selfie portrait – procedural Blender 5.1.1 character

Everything here was built and rendered in the installed Blender 5.1.1
(`/Applications/Blender.app/Contents/MacOS/Blender`) using bpy scripts, Cycles on the CPU with 3 threads, 32 samples and OpenImageDenoise.
No external models, textures, reconstruction tools or generation services were used.

## Reference image access
I **did see the decoded photograph**. I opened `input/reference.png` with the image-reading tool; it also came through as a
downscaled copy (`previews/ref_small.jpg`, made with `sips`). It shows a selfie taken with both arms reaching toward the camera.
The subject has a gently tilted head, long chestnut wavy hair with light bangs, warm brown eyes, a soft closed-mouth smile and a dusty-pink chunky
crew-neck sweater, in a bright cafe with plants, shelves and a flower vase. I used the photo only as a visual reference.
It is not used as a texture, projection, billboard or background, and it does not appear anywhere in the .blend. I made no attempt to identify the person.

I also read `input/environment.json` (landmarks/coordinate frame) and `input/ASSET-LICENSE.md`.

## Files
| file | content |
|---|---|
| `build.py` | builds the whole scene from `input/` and saves `character.blend` (packed) |
| `render.py` | renders the saved .blend to `renders/*.png` (or half-res previews) |
| `character.blend` | final editable scene, Cam_Beauty active, all images packed (~15 MB) |
| `renders/beauty.png` | 960×1040 portrait |
| `renders/front.png`, `three-quarter.png`, `side.png`, `back.png` | 768×768 turntable views at 0/45/90/180° |
| `scene-report.json` | versions, engine/device, objects, bounds, counts, textures, limitations (written after reopening the .blend) |
| `tools/` | inspection/preview helpers I used during development (`verify_report.py` writes the report) |
| `textures/` | the two generated knit height maps (also packed in the .blend) |
| `previews/` | development previews and logs |

## Build / render commands
```sh
B=/Applications/Blender.app/Contents/MacOS/Blender
$B --background --factory-startup --threads 3 --python build.py                 # ~15 s, writes character.blend
$B --background --factory-startup --threads 3 character.blend --python render.py # all 5 finals (~2.7 min)
$B --background --factory-startup --threads 3 character.blend --python render.py -- beauty side --preview   # half-res, 12 spp
$B --background --factory-startup --threads 3 character.blend --python tools/verify_report.py              # reopen check + scene-report.json
```
The build is deterministic because it uses fixed RNG seeds. You can use `STOP_AFTER=body|eyes|sweater|hair|hands` to save intermediate stages.

## Scene organisation (in the .blend)
- **Character**: `Body_Skin` (sculpted and posed MakeHuman mesh with a Subdivision modifier), `Phone`, `Hand_Palm.L/R`, `Finger.*`, `Thumb.*`, `Nail.*`
  - **Eyes**: `Eyeball.L/R` (sclera+iris+pupil shader), `Cornea.L/R` (separate domed glossy/transparent shell)
  - **Hair**: `Hair_Strands`, `Hair_Bangs`, `Eyelashes`, `Eyebrows` (native Blender hair *Curves* objects)
  - **Clothing**: `Sweater`, `Sweater_Collar`
- **Cafe_Set**: room, window, counter, shelves/jars, tables, chair, vase with flowers, potted plants (all simple geometry)
- **Lighting**: `Key_Window`, `Fill_Front`, `Catchlight`, `Rim_Back`, `Rim_Back_R`, `Cafe_Ceiling`, `Diag_Fill_Back` (the last one is used only for the diagnostic views)
- **Cameras**: `Cam_Beauty` (35 mm lens on a 36 mm sensor, wide selfie look; DOF focused on the face via `Focus_Target`), `Cam_front`, `Cam_three-quarter`, `Cam_side`, `Cam_back`

All materials are node-based and editable: Skin, Skin_Hands, Eye_Sclera_Iris, Eye_Cornea, Hair_Chestnut, Lash_Hair, Brow_Hair, Knit_DustyPink, Nail, Phone_Case and the Cafe_* materials.

## How it was made (geometry and material techniques)
1. **Face/body sculpt** (`sculpt()`): the CC0 MakeHuman upper body gets many smooth, mirrored, compact-support deformations:
   - slimmer neck and narrower shoulders
   - a narrower jaw and a soft V chin
   - a shortened, youthful lower face
   - fuller, higher cheeks and a softer brow ridge
   - larger almond eyes. An angular warp about each socket centre opens the lids while keeping them at the same radius, so they still wrap the eyeball.
   - a smaller nose with narrower alae and a slightly lifted tip
   - a narrower mouth with the seam closed and the corners lifted, for the slight closed-mouth smile
   - a modest bust with the base-mesh nipples flattened

   Then one Catmull–Clark level is applied to the mesh and another is kept as a modifier.
2. **Pose**: a temporary armature is built from the landmark joints. Its vertex weights are computed analytically: the head is cut by a tilted plane under the jaw, so the whole face moves rigidly. Dual-quaternion skinning applies the pose:
   - head roll −26°, chin up 14°, a slight turn
   - protracted clavicles
   - two-bone IK, so both wrists reach a phone held just below the beauty camera

   The pose is baked and the rig deleted.
3. **Eyes**: a sphere is least-squares fitted to the lid/socket vertices around each eye, giving r ≈ 15.6 mm. The eyeball sits inside the lids and its gaze is aimed at the camera.
   - The iris is procedural: angular/radial noise fibres, an amber collarette, a dark limbal ring and the pupil.
   - The cornea is a separate shell with a domed bulge over the iris, using a Fresnel glossy/transparent material.
   - The lid margins are found by ray casting from the eye centre. 150 upper and 42 lower lashes are rooted on them as curves, and the lash line on the skin is darkened.
   - Eyebrows have 420 per side short hairs, rooted on the skin by ray casting, with direction changing from head to tail.
4. **Skin**: `Principled BSDF` with Random-Walk (Skin) subsurface (radius 1/0.42/0.24, scale 3.2 mm).
   - Colour and roughness come from per-vertex attributes painted procedurally in sculpt space: cheek and nose blush, lip shape (from a measured lip-seam curve), lid tint, ear redness, wet lid margin, mouth interior, and a hair-brown scalp under the hair.
   - Faint mottling and freckle noise are added in the shader.
   - Restrained Voronoi pores plus fine-noise bump give micro-relief, with roughness micro-variation.
5. **Hair**: about 1,900 guide strands are Poisson-sampled on a scalp-density field that leaves the hairline, temples and ears clear. They are integrated under gravity, with nearest-surface collision against skin, sweater, collar and eyeballs.
   - Guides have per-zone comb directions: a slightly off-centre part, wispy bangs lifted off the forehead, face-framing pieces, back hair, and left-side hair routed behind the raised arm.
   - Loose sinusoidal 3-D waves are added after about 12 cm from the root.
   - About 29,000 child strands are interpolated from 2 guides each, with clumping, frizz and rare flyaways. Bang children are deliberately sparse.
   - A polar "collar clearance" map keeps hair on top of the knit collar instead of slipping between collar and neck.
   - Shader: Principled Hair (Chiang) with melanin 0.80–0.86, redness 0.70, a warm tint, random colour/roughness, and roughness 0.33 / radial 0.55.
6. **Sweater**:
   - The posed torso and arm faces covered by a crew neck are extracted.
   - Sleeves are planarised and extended to a gathered rib cuff that opens over the palm. A rolled lip at the cuff gives the opening thickness.
   - A hem rib is added.
   - The shell is subdivided, then smoothed outward only so it bridges anatomy.
   - It is pushed onto a blurred cylindrical envelope that hangs down from the bust, with ease added on top.
   - The sleeves sag under gravity.
   - Procedural folds are displaced along the normals: irregular compression rings on the sleeves, strongest at the elbows, plus a few drape and hem ripples.
   - Knit UVs are computed analytically (around/along the body and the arms).
   - The `Sweater_Collar` is a separate 1×1 rib band swept along a low-pass-filtered neckline, with skin clearance.
   - Material: two generated, packed 1024² height maps (stockinette V-stitch with ply twist and fibre fuzz, and a 1×1 rib), with the rib selected by a `rib` attribute. They drive both the colour ramp and the bump. The fabric uses dusty pink with sheen, fuzz noise, and slight SSS.
7. **Hands/phone**:
   - Simple anatomically placed geometry: a flattened palm ellipsoid inside each cuff, tapered finger tubes with knuckle bulges and nails wrapping the back of the phone, and thumbs on the screen side.
   - The phone is a bevelled slab with a pale case and a dark glass screen.
8. **Lighting/camera**: a large window area light from camera-left, soft fill, a small catchlight above the lens, warm rim lights and the cafe ceiling. The camera uses AgX with the Punchy look and +0.15 exposure.
   - The beauty camera is a high, close selfie viewpoint at f/2.6, with a slight roll so the head tilt reads like the photo.
   - The arms run out of the frame toward the lens, as in the reference.

## Self-review (honest)
**Works reasonably:**
- the overall composition: high selfie angle, tilted head, both arms reaching out of frame, and long hair falling over one shoulder and behind the other
- the soft cafe context
- a continuous sculpted face with no seams or detached features
- eyeballs that actually sit inside the lid sockets
- brown irises with corneal highlights
- a small nose with nostrils
- a closed, gently smiling mouth
- real ears from the base mesh, partially covered by hair
- brows and lashes
- a believable neck
- around 29k waved hair strands with bangs, with no bald gaps
- a ribbed crew-neck sweater with visible knit stitches, cuffs and hem
- correct-looking profile and back views of the same geometry

**Approximations / weaker than the reference:**
- **Likeness is only approximate.** The face is hand-tuned procedural sculpting, not photo reconstruction. Compared with the photo:
  - the face is slightly longer and the eyes slightly smaller and less luminous
  - the cheeks are less full
  - the smile is subtler
  - the skin reads smoother and a little pinker or more doll-like
- Bangs are thin and wispy but more uniform and straight than in the photo. The crown/part shows a light sheen.
- Waves are regular sinusoids rather than natural S-waves. On her raised-arm side, some strands bunch near the collar, which shows as a darker band in the side and back views.
- The sweater is not a cloth simulation. Folds are procedural, and the torso front is smooth and a little flat. The knit is a bump-mapped tile rather than modelled yarn, and its scale only approximates the chunky knit in the photo.
- Hands are simplified tube fingers and are not anatomically detailed. In the beauty shot they are out of frame, as in the photo; the diagnostic views show them.
- The cafe is a defocused suggestion only: blocks, cylinders and leaf cards.
- There is no production rig.

**Inferred (not visible in the reference):**
- the back and sides of the head and hair
- hair length and waves at the back
- the sweater back, sides, hem and cuffs
- the torso shape under the sweater
- hands, fingers and the phone
- ear detail

The diagnostic renders use the same geometry and materials as the beauty shot. They hide only the `Cafe_Set` collection and switch to a plain grey world plus `Diag_Fill_Back`.

## Counts (from `scene-report.json`, measured after reopening)
- Character evaluated mesh vertices: about 272k (whole scene about 304k)
- Hair curves: 30,471 curves / 937,765 control points (head hair, bangs, lashes, brows)
- Character bounds: about 0.50 × 0.75 × 0.80 m (x/y/z, including the outstretched arms and phone)

## Third-party attribution
Base mesh: **MakeHuman hm08 base mesh**, released under **CC0** by the MakeHuman project. The source file is pinned at commit
`a8bc2d54ff0ac92e78ff71431b1023eda42bf482`:
https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj
I used the supplied cropped and converted `input/upper-body.obj`. `base.obj` was not used in the final scene. All sculpting, hair, clothing, materials, textures and scenery are my own procedural work in `build.py`.
