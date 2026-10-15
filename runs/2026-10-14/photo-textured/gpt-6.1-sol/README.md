# Editable Blender portrait — photographic finishing pass

Open **[character.blend](character.blend)** in Blender **5.1.1**. It contains a continuous anatomical head and neck, separate three-dimensional eyes, a fitted volumetric hair foundation, editable hair fibers, and solid sweater/sleeve geometry. The beauty camera is active. **The result remains visibly CG; it is not a scan or a perfect likeness.**

The five files in **[renders/](renders/)** are raw Blender renders of the same character. There is no photograph plane, reference-photo background, photographic post-compositing, retouching, or view-dependent geometry replacement. The saved scene uses a plain warm studio background; the first-pass geometric cafe remains available in a hidden collection.

## Open and rebuild

Run from this directory:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

`build.py` opens **`source-character.blend`**, the preserved first-pass scene supplied in this directory. That stable source is never overwritten. The script reads the supplied `input/reference.png`, repairs the source character, recreates the generated textures and groom, packs the images, saves `character.blend`, genuinely reopens it, and renders all five views. Only files in this working directory are used as task source files. Blender supplies `bpy`, `mathutils` and NumPy; no installation is required.

Final rendering: **Cycles CPU, three fixed threads, 32 samples, denoising**. Beauty is **960×1040**. Front, three-quarter, side and back are **768×768**, with cameras at **0°, 45°, 90° and 180°** around the same tilted upper-body character. Materials, geometry, world and lights remain identical in every view.

For small previews:

```sh
PORTRAIT_PREVIEW=1 /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

This uses 12 samples and writes to `previews/`; it also rebuilds the working `character.blend`. Run the normal command afterward to restore the final 32-sample deliverables.

Independent verification:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --threads 3 --python verify.py
```

`verify.py` checks the reopened scene, render dimensions, packed image sizes, geometry counts, resource budget and render settings. It writes `verification.json` and adds its measured results to `scene-report.json`. Direct visual review is separately recorded; the verification script does not claim to see images by itself.

## Actual image access

I **directly saw the decoded reference photograph through the `view_image` tool** before editing. The copied directory did **not** contain the previous beauty/side PNGs. I therefore rendered the preserved first-pass scene into `previews/previous-beauty.png`, `previous-side.png` and `previous-face.png`, then directly opened all three through the image-reading tool before changing the model.

Multiple low-resolution finishing passes were actually rendered and directly inspected, including all five views. All five final PNGs were also opened for direct visual inspection. Blender additionally decoded the reference pixels for texture construction. This was visual access plus numerical processing, not an image description substituted for seeing the photograph.

I worked independently in this directory, did not read other entrants or parent comparison results, did not modify the previous original directory, and used no additional agents, downloads, pretrained reconstruction models, or external image/3D-generation services.

## Changes from the procedural first pass

The first pass used procedural skin, iris, hair and knit appearance. **The second-pass instruction explicitly removed the restriction on photographic character textures.** This version uses carefully restricted pixels from the supplied photograph as genuine character textures while retaining the original continuous 3D anatomy.

Changes include refined integrated nose/mouth geometry, a gently increased head tilt, corrected eye placement and curved optical surfaces, source-derived iris color/detail with the original catchlight and pupil removed from the texture, a rebuilt fitted scalp, repaired hanging hair roots, new crown and fringe fibers, more restrained rim lighting, an opposite shoulder slope, and collision correction where front hair met the collar. The useful tubular hanging strands and sculpted wave cores from my first pass are retained and repaired. Faulty first-pass crown strands are discarded. Camera framing is closer to the upper-body selfie composition.

## Exact texture method

| Resource | Size | Method and editable mapping |
|---|---:|---|
| `textures/face-albedo.png` | 1024² | Supplied photo remapped with a thin-plate spline using manually inspected eye, nose, mouth, chin and cheek landmarks. Anatomical x/z UVs are stored on the continuous face. Alpha feathers the face boundary, excludes the source eye areas and rejects dark/non-skin hair pixels. A vertex attribute further restricts projection by frontal depth. Low-frequency luminance correction attenuates photographic illumination; albedo is reduced to suit scene lighting. |
| `textures/iris-albedo.png` | 256² | Isolated brown iris crop from the reference. Bright/desaturated source glints and the central source pupil are filled with brown. Planar UVs map the remaining iris detail onto curved iris geometry. Separate dark pupils and refractive corneal caps produce real scene reflections. |
| `textures/hair-crown-albedo.png` | 1024² | Source hair remapped onto a fitted, curved cranial mesh through anatomical x/z UVs. Luminance/chroma and frontal-depth support reject skin/background and blend into the inferred hair material. The scalp has actual skull depth and is supplemented by real crown/fringe fibers. |
| `textures/hair-strand-tile.png` | 512² | A cropped patch of supplied long hair, mirrored into a repeat tile, supplies strand detail on inferred scalp sides/back through cylindrical UVs. It is not a photographed silhouette or hair card. |
| `textures/knit-albedo.png` | 512² | A sweater-only crop from the reference, normalized for broad shading and mirrored into a seamless repeat. Triplanar object-space mapping applies the knit across the solid torso and sleeves. |
| `textures/knit-height.png` | 512² | Normalized luminance from the same sweater crop supplies fine yarn bump, with conservative relief. |

All generated texture dimensions are below 2048. All used images are packed into the `.blend`; the external PNGs are also retained for editing. The entire original photograph is **not** packed or used as a rendered background. The retained 256² mathematical knit tile exists in the old material nodes, but the final cloth output uses the photographic knit textures.

Unphotographed skin blends into a warm procedural base with subsurface scattering, pore bump and variable roughness. Brows, lashes, eyelid folds, lip crease, collar ribs, hanging tubular strands and wave cores remain editable geometry. Crown/fringe fibers use native Blender hair curves. Cloth has additional editable cylindrical UVs alongside its triplanar material. Collections separate skin, eyes, hair, sweater, lighting, cafe and cameras.

## Measurement, assumptions and self-review

The completed character has **598,205 evaluated vertices excluding native hair**, including the evaluated tubular hair and rib curves. Native hair is reported separately: **3,600 curves / 106,400 points**. The mesh/converted-curve vertex total is below one million. Exact object counts, bounds, packed resources and output hashes are in `scene-report.json` and `verification.json`.

The photograph informs the observed complexion, lip color, iris, front crown detail and knit appearance. Skull depth, ears, profile, rear scalp, back hair, garment construction, sleeve paths and cropped sleeve ends are inferred. No hands or lower body are modeled. The face/neck pose is baked; this is an editable portrait asset rather than a production facial rig.

The final images preserve actual cheek, nose, lip, eyelid, jaw, ear and skull depth. The profile and back views show the same clothes and groom. However, **human realism and likeness remain limited**: the eyes/lids and corneal optics are simplified, the face is smoother and more doll-like than the reference, the crown still reads as a fitted CG hair mass, hanging locks are too organized with some coarse tips, and the sweater's shoulder shapes/drape are approximate. Some photographic lighting remains in the face/crown albedo. The inferred scalp texture repeats, the front-to-side texture blend can be noticeable, and the photographed knit tile repeats and is bump relief rather than fully modeled yarn. These limitations are not hidden with a photographic background or alternate-view assets.

## Base attribution

The continuous anatomical mesh derives from the supplied **MakeHuman hm08 common upper-body topology, CC0**, pinned commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`.

[Original MakeHuman base asset](https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj). See **[input/ASSET-LICENSE.md](input/ASSET-LICENSE.md)**. Supplied preparation cropped the upper body and converted it to meters, Z-up, front −Y. The initial facial edits, hair, clothing and scene came from my preserved first-pass asset; this script implements the finishing changes described above. The photograph is the user-supplied reference, not an additional downloaded asset.
