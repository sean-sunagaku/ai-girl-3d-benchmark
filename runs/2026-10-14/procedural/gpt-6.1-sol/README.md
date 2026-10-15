# Editable upper-body portrait

The completed asset is **character.blend**. The five PNGs in **renders/** are direct Blender Cycles renders, with no photographic compositing or retouching.

I **actually saw the decoded reference photograph**, using the image-reading tool before construction. I also read `input/environment.json` and `input/ASSET-LICENSE.md`. I worked only in this entrant's directory. No downloaded models, textures, reconstruction systems, image-generation services, or additional agents were used.

## Build and open

Open `character.blend` in Blender 5.1.1. It is saved with the beauty camera, cafe, materials and portrait lighting active. Collections separate the continuous skin, eyes, hair, sweater, lighting, cafe and cameras. The generated knit image is packed.

From this directory, rebuild and render all five views:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

For smaller previews, using the same character geometry:

```sh
PORTRAIT_PREVIEW=1 /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

The final saved scene was independently reopened and checked with `verify.py`: 893,837 evaluated character vertices, a packed 256×256 knit texture, all five correct PNG dimensions, and approximately 19 MB for the entire directory. The final five images were also decoded and visually inspected.

The script saves the scene before rendering, genuinely reopens it in Blender, and then renders. Default rendering uses Cycles CPU, 32 samples, denoising and three threads. Beauty is 960×1040; front, three-quarter, side and back are 768×768. Diagnostic cameras orbit at 0°, 45°, 90° and 180°. They share the character, materials and lighting; only the cafe is removed for the plain diagnostic background.

## Construction

The CC0 anatomical mesh supplies a continuous face, native eyelids, lips, nostril recesses, ears and neck. Vertex deformations soften the cheeks and jaw, refine the nose, enlarge the orbital openings, close the tiny mouth gap and lift the smile corners. Subdivision smooths the surface. The head tilt is 20°, with a gradual deformation through the neck.

The eyes have actual spherical sclerae, curved brown irises with radial shader fibers, dark pupils, limbal transitions and refractive corneal caps. Brows, lashes and upper-lid creases are editable curves. Skin uses painted vertex complexion, restrained blush/lip variation, subsurface scattering, pore bump and micro-roughness.

Chestnut hair uses a fitted scalp, flattened underlying locks and thousands of individually tapered 3D curve strands. Extra crown fibers, separated fringe wisps and flyaways supplement the long waves. Root tilt fades through the hanging lengths to suggest gravity. All hair remains geometry, with editable brown anisotropic materials.

The sweater is a joined, smoothed cloth volume with rounded extended sleeves, modeled folds and a real neck opening. Its crew-neck band has physical ribbing. A generated 256×256 looping-yarn texture provides fine knit relief and color; selected front stitches and collar ribs are actual raised curve geometry. Cafe shelves, cups, furniture and a plant are simple 3D scenery.

## Self-review and approximations

Small rendered previews were decoded and inspected through several revisions. Corrections addressed crown clipping, overly smooth hair cores, exposed shoulders, neck-band fit, abrupt hair/clothing intersections, mouth darkness and diagnostic framing.

The render captures the warm feminine appearance, brown eyes, closed smile, tilted head, long chestnut waves and dusty pink sweater. **It remains visibly CG and does not achieve photographic likeness.** The skin is too uniform, eye/lid details are simplified, the groom has organized lock patterns and some coarse tips, and the sweater lacks the photograph's intricate fabric and drape. The facial proportions and cafe composition are approximations.

Side/back anatomy, ear depth, skull shape, back hair, garment construction, sleeve paths and the cropped sleeve ends are inferred from one front photograph. Hands and unseen lower anatomy are not modeled. This is an editable portrait asset rather than a production rig; the face/neck pose is baked into the mesh. Measurement details and remaining limitations are in `scene-report.json`.

## Attribution

MakeHuman hm08 common upper-body topology, **CC0**. Pinned source commit: `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`.

Source: <https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj>

The supplied preparation cropped the upper body and converted it to meters, Z-up, front −Y. Facial changes, clothing, eyes, hair, procedural materials and scenery were created for this entry. See `input/ASSET-LICENSE.md` for the supplied license statement.
