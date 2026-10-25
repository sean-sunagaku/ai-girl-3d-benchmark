# Editable upper-body portrait

## Build, open, render

Run from this directory with installed Blender 5.1.1:

`/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py`

This builds the character, writes `character.blend`, packs resources, reopens the saved file, and renders all five required PNGs with Cycles CPU, 32 samples and denoising. For a faster two-view check, append `-- --preview`. Open `character.blend` in Blender; the active camera is **Camera • beauty selfie**. Character geometry is organized in collections 01–04, with editable meshes, materials, curve splines and modifiers.

## Reference access and design

The decoded `input/reference.png` **was actually seen and visually inspected with an image-reading tool** during production. `input/environment.json` and `input/ASSET-LICENSE.md` were also read. The reference guided the warm fair skin, brown eyes, slight closed-mouth smile, diagonal wispy bangs, long chestnut waves, dusty-pink crew neck, forward arm pose and gentle head tilt. No photograph, crop, generated portrait, billboard or photographic texture is used by the scene or renders.

The continuous MakeHuman facial surface is conservatively sculpted at the jaw, cheeks, nose, eye region and mouth corners. Real scleral globes sit behind custom almond eyelid surfaces; radial brown irises, pupils, clear corneas, lashes and individual brow hairs are separate editable geometry. The fitted scalp carries hundreds of combed strands, with additional bangs, wavy solid lock volumes, surface fibers and flyaways. The sweater expands and reposes anatomical topology into a loose garment with modeled folds, ribbed collar, cuffs and hem; procedural wool color and crossed-wave bump provide knit detail. Lighting uses large soft areas and a simple genuinely 3D cafe background. Diagnostic views hide only the cafe collection and use identical character geometry and materials.

## Inference and self-review

The unseen side/back silhouette, rear hair, garment construction and arm/hand posing are inferred. This is a procedural likeness study, not a scan or exact reconstruction. The face, expression and eye/lid blending remain approximations; lock-based hair and procedural knit cannot match every fine feature of the photograph. The simple cafe is secondary to the portrait. The hands are reposed base-mesh anatomy, not a production hand rig, and there is no full-body or animation rig. All final PNGs are direct Blender renders without photographic retouching or compositing.

## Base-mesh attribution

MakeHuman **hm08**, CC0, pinned commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`. Source supplied in `input/base.obj`; prepared topology in `input/upper-body.obj`. Source: https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj . The original header names Data Collection AB, Joel Palmius and Jonas Hauquier at the CC0 release. Geometry was cropped to the clothed upper body/hands, sculpted, reposed and supplemented here with original procedural hair, eyes, clothing, materials and set geometry. No additional models or textures were downloaded.
