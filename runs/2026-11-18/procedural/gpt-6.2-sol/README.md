# Chestnut and rose

I opened `input/reference.png` with an image-reading tool and saw the decoded 1205 × 1306 photograph before modeling. I used it to guide the tilted pose, gaze, chestnut waves, wispy bangs, pink knit and window-lit café composition. The scene uses modeled geometry and authored materials throughout.

## Open and reproduce

Open `character.blend` in Blender 5.1.1. It opens on **Beauty • café selfie**. The **Character** collection contains the editable face, eyes, groom, sweater and hands. The four diagnostic cameras use the same character under neutral studio lighting.

Run these commands from this directory. The build saves and packs the scene. The render command reopens it and writes the five PNGs.

```sh
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
"$BLENDER" --background --factory-startup --threads 3 --python build.py
"$BLENDER" --background --factory-startup character.blend --threads 3 --python render.py -- beauty front three-quarter side back
"$BLENDER" --background --factory-startup character.blend --threads 3 --python report_scene.py
```

## Geometry and materials

- The reshaped hm08 surface connects the face, ears and neck. Separate eye globes have radial iris pigment, a raised corneal surface and tear-film gloss. Skin uses subsurface scattering, local pigment and procedural pores.
- Editable Blender Curves form the brows, lashes, parted crown, bangs and long waves. Shaped scalp and back meshes support the dense groom.
- The sweater has joined sleeves, modeled ribbed edges and a packed 384 × 384 yarn-height tile traced from interlocked loops. The torso and sleeves use their own knit UV directions.

## Self-review

I inspected the Blender previews and revised the lighting, groom coverage, sleeve joins and collar fit. The final PNGs are raw Cycles renders on CPU at 32 samples with denoising. The likeness is strongest in the head tilt, palette and styling. The skin and hairline still read as CG in close-up. The temple shows a narrow area of scalp between swept locks in profile. The face shape and smile are approximate. The profile, rear hair, unseen sweater construction and hand pose are inferred from the portrait setup.

MakeHuman hm08 is CC0. The supplied `upper-body.obj` was the starting point for the face and neck, and the supplied source mesh provided the hand topology. [Pinned upstream asset](https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj)
