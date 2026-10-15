# Upper-body portrait

Editable clothed bust in Blender 5.1.1. The cage is the supplied MakeHuman body. Skin, lips, irises and the sweater yarn color are taken from `input/reference.png`. The person is not identified.

## Image access

`input/reference.png` (1205×1306) was opened as a decoded image. The picture itself was visible: a cafe selfie, long chestnut hair with wispy bangs, warm brown eyes, a small nose, a closed-mouth smile, a slight head tilt, fair warm skin, and a dusty pink knitted crew neck, with one arm reaching toward the camera.

The same file was also read numerically (Pillow). Catchlights sit near pixels (643, 309) and (511, 386). A lip-colored cluster centers near (638, 514). Cheek, iris, lip and a hair-free knit patch were averaged from those regions. The five final PNGs were opened the same way after rendering and checked at 960×1040 and 768×768.

`source-character.blend` is the previous procedural-only scene. This pass does not overwrite it.

## What changed from the first pass

The first pass could not use the photograph as a texture. That limit is lifted here.

The first render showed a pale mannequin face, a bald scalp, and thin strands crossing the eyes and nose. This pass sculpts a smaller nose, a softer jaw, a closed smile and a stronger head roll; seats smaller eyeballs deeper in the lids; keeps long hair beside the face instead of across it; and colors the skin and sweater from the photograph. The scalp is still a smooth cap. That failure is not hidden by a photo.

## Build and open

From this directory:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

That imports `input/upper-body.obj`, sculpts and poses the body, samples `input/reference.png`, saves `character.blend`, then renders the five PNGs with Cycles CPU, 3 threads, 32 samples and OpenImageDenoise.

A lower-resolution look (8 samples, `preview/*.png`, still saves the blend):

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --preview
```

Open `character.blend` in Blender 5.1. The scene camera is `BeautyCamera`. Materials, the knit images, hair curves and modifiers are in the file. `ProjCamera` is hidden and was used only to place skin colors.

## Texture method

No scan and no outside model. Generated images are 1024 and packed.

**Skin.** After the pose, a hidden camera is fit so the posed eye centers, nose and mouth land on the measured photo landmarks. Mean error is about 18 pixels. The eyes match within about 6 pixels; the mouth is about 50 pixels off, so the mouth is not copied as a picture. Front-facing vertices sample the photo only when the pixel passes a skin test (not hair, not the pink sweater, not a deep shadow, not a white highlight). Eye sockets are left unpainted so the photo's eyes are not stamped onto the lids. Samples are flattened toward a cheek albedo measured from the photo, then darkened so the scene lights do the shading. Sides, ears, neck and the back of the head use that cheek color. Lips use the sampled lip color. The result is the `SkinColor` point attribute on `Body`, plus a cylindrical `UVMap` for editing. It is not a UV-projected face photograph.

**Eyes.** Separate spheres and corneas. The iris shader uses browns measured around the photo catchlights, about linear (0.20, 0.08, 0.06) before the shader's own mix. Lids are the body mesh, pulled onto the eyeball and then eased shut so the iris is not a full circle of sclera.

**Sweater.** `KnitColor` is the 160×160 patch at photo pixel (780, 860). Hair-colored pixels in that crop are replaced, the low-frequency lighting is divided out, and the tile edges are feathered. It repeats on the sweater UV (mapping scale 3.2, 8.0, 1). `KnitHeight` is still the procedural stockinette map and drives displacement and bump.

**Hair.** Chestnut colors measured from the dark and mid hair pixels, used as a Principled color with a lighter noise mix. Geometry is a scalp cap (cloud displacement), 915 beveled poly curves and 273 thinner flyaways. Strands are rooted on the cranium and fall beside the face. The cap covers the crown so the skin does not show through, and it still reads as a smooth wig base.

## Photo versus inferred surfaces

Taken from the photograph: front skin color within the mask, lip and iris averages, sweater yarn color, hair color, and the pose idea (tilt, closed smile, an arm toward the lens).

Inferred: the side and back of the skull, the ears under the hair, the hair mass and waves, the neck under the collar, sweater folds on the far side, and both hands. The cafe is a few boxes and a plant, not the photographed room. Turnaround views hide that set and use a flat background.

## Editable resources

- `Body`: sculpted mesh, `SkinColor`, `SkinRough`, cylindrical UV, Skin material, subdivision
- `Sweater`: shell mesh, `UVMap`, KnitColor, KnitHeight, displacement and solidify
- `HairCap`, `HairStrands`, `HairFlyaways`: cap mesh and native poly curves
- `Eye_L`, `Eye_R`, `Cornea_L`, `Cornea_R`, `LashesBrows`
- `BeautyCamera` and the four turnaround cameras
- `ProjCamera` (hidden, not used for the delivered frames)

Evaluated mesh vertices are about 255k, including the beveled hair curves. Native curves are 915 + 273 splines (12250 + 3262 points) and are also listed separately in `scene-report.json`.

## Base mesh

MakeHuman hm08, commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`, CC0. `input/upper-body.obj` only crops that body and converts it to meters, Z-up, face toward -Y. See `input/ASSET-LICENSE.md`.

## Self-review

The bust is a real mesh: profile, ears, skull, crew neck and sleeves stay present from the front, the three-quarter, the side and the back. The sweater is dusty pink with a visible knit. The eyes are brown. The mouth is closed. The head is tilted. Skin is warmer than the first pass.

It is still obviously CG. The scalp is a smooth brown shell with a hard fringe, and the long hair is a few hundred thick curves rather than waves. The face is a softened MakeHuman head with painted color, not this person's features. The lids and nose do not match the photograph. Shoulders and cuffs have gaps where the shell meets the skin. Hands are blocky. Nothing here is a scan or a likeness.
