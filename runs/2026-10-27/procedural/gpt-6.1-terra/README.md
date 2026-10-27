# Editable upper-body portrait

`character.blend` is an editable Blender 5.1.1 scene. The five PNGs in `renders/` are unretouched Cycles renders made from that saved scene. I opened and **actually saw the decoded photograph** in `input/reference.png` before modeling: it shows a gently tilted, smiling face, long chestnut hair with fine bangs, brown eyes, and a dusty pink sweater in a softly lit cafe. The photograph was used for visual guidance only; it is not in the scene, used as a texture, projected onto the model, or used as a background.

## Build, open and render

From this directory, with the specified installed Blender:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
/Applications/Blender.app/Contents/MacOS/Blender --background character.blend --threads 3 --python render_views.py
/Applications/Blender.app/Contents/MacOS/Blender --background character.blend --threads 3 --python report.py
```

The first command constructs and saves `character.blend` **before** rendering. Open that file normally in Blender to inspect the beauty camera, materials, lights, collections, the tilted head parent, face, individual eye components, hair curves and sweater meshes. The scene uses Cycles CPU, 32 samples, denoising and three threads. The render command reopens the saved scene, makes `beauty.png` at 960 × 1040, then temporarily removes the cafe objects in memory, supplies a neutral studio background and lights, and renders the same character and materials at 0°, 45°, 90° and 180° at 768 × 768. It does not save those temporary studio changes over the beauty scene. `report.py` measures the saved scene and refreshes the JSON report. `render_views.py -- --preview` produces small diagnostic previews instead.

## Construction

- The continuous face is made from the supplied CC0 MakeHuman upper-body mesh, cropped to the head and locally deformed around the eyes, mouth, jaw and one ear. A separate tapered neck overlaps the cropped portion. The head and its features are parented to an editable tilt pivot.
- The eyes have spherical sclera, modeled radial brown irises and pupils, curved transparent/glossy corneal shells, eyelid strips, lashes and individual eyebrow curves. The nose is part of the underlying face; very small curves suggest the nostril and mouth creases. Skin uses a vertex-painted complexion, subsurface scattering and procedural micro-bump.
- Hair consists of a scalp cap, an underlying flowing shell and thousands of editable beveled curves for long hair, fringe and strands on the scalp and shoulder. It uses several chestnut materials and slight coat highlights. One ear is partly exposed by a tuck in the hair.
- The sweater is a ring-lofted torso, ribbed crew neck and two shaped forward-reaching sleeves. An original, locally generated 512 × 512 repeating knit relief image drives its color and bump; that image is packed in the blend. The end of each sleeve is covered and closed, because the hands lie beyond the photograph's frame.
- The cafe is simplified 3D geometry, lit with broad area lights and photographed with depth of field. No external models, portrait textures, reconstruction services or downloaded assets were used.

## Self-review and inferred areas

This is a **stylized procedural reconstruction, not a convincing photorealistic likeness**. The face and expression remain generic and smoother than the photograph; the eyelids, lips, nose, ears and eye reflections are approximations. The hair contains fine 3D strands but its larger masses are too smooth and straight, and some intersections and simplified strand flow remain. The knit and folds suggest fabric rather than reproducing individual photographed stitches or simulated cloth. The sleeve caps, shoulders, back of the head, hair behind the subject, profile, neck, back of the sweater and all other unseen anatomy and clothing are inferred from a single frontal photograph, not measured from it. The cafe is schematic. There is no lower body, modeled hand or production rig. The corneal shell approximates a glossy clear surface rather than a complete volumetric eye.

The exact saved-scene object names, measured bounds, evaluated geometry counts, packed image and output dimensions are recorded in `scene-report.json`.

## Base asset attribution

The supplied `input/upper-body.obj` is the cropped, coordinate-converted MakeHuman hm08 base mesh; its source is released under CC0 as documented in `input/ASSET-LICENSE.md`. Pinned source: <https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj>. The common base was not a likeness; the visible modifications, hair, clothing, shaders and set were constructed for this scene.
