# Window seat

An editable, fully clothed upper-body portrait built in Blender 5.1.1. Open `character.blend` and press **F12** for the saved beauty view. The five finished PNGs are in `renders/`.

## Reference and interpretation

I actually saw the decoded `input/reference.png` before building the scene. I used the diagonal from the crown to the chin, the exposed ear, the long hair falling over the raised shoulder, and the two sleeves reaching toward the camera to compose the portrait. The photograph served as a visual reference only. The scene’s single packed raster image is an original 384 × 384 repeating yarn pattern.

The frontal features are a visual approximation. I inferred the profile, skull and ear depth, back hair, garment panels, forearms and hands. The hands complete the pose beyond the edges of the selfie. The café is a small modeled set with a window, branching plant, shelving, ceramics and an espresso machine.

## Editing the portrait

- **Face and eyes.** The MakeHuman face, ears and neck form a continuous skin mesh. Relative shape keys hold the portrait sculpt and cervical tilt. The original eyelid loops wrap separate eye globes. Each eye has a pigmented iris with radial variation, pupil, convex corneal film and lacrimal tissue. Brows and lashes have editable fibres. Paintable attributes control complexion, local gloss and the distribution of pigment. The skin shader adds subsurface scattering, pore relief and fine vertical lip furrows.
- **Hair.** The chestnut groom uses native Blender hair curves. The outer and inner lengths, tapered fringe, ear tendrils, flyaways and back wave each have their own object. A shaped scalp carries the part beneath the fibres. The Huang hair shader varies colour across locks and individual strands.
- **Sweater.** The torso and forward sleeves share vertices around their armholes. The separate ribbed crew neck meets a continuous skin yoke. The shoulders carry modeled seam stitches. A packed procedural stockinette tile supplies yarn colour and relief, and short curves give the garment a woolly edge.
- **Cameras.** The beauty camera uses perspective and depth of field. Four orthographic cameras show the entire posed character at 0°, 45°, 90° and 180°. For these views the render script switches off the café collection and uses the plain world background. Each PNG comes directly from Cycles with CPU denoising.

## Reproduce

Run these commands from this directory. For a source rebuild, keep `atelier.py`, `face.py`, `groom.py`, `hands.py`, `knit.py`, `room.py` and the supplied `input/` beside `build.py`. The build saves the scene before any rendering. The render and inspection commands reopen it in a fresh Blender process.

```sh
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
"$BLENDER" --background --factory-startup --threads 3 --python build.py
"$BLENDER" --background character.blend --threads 3 --python render.py -- --preview
"$BLENDER" --background character.blend --threads 3 --python render.py
"$BLENDER" --background character.blend --threads 3 --python ../../scripts/inspect_scene.py -- --output validation.json
"$BLENDER" --background character.blend --threads 3 --python report.py
```

`render.py` also accepts individual view names after `--`, such as `beauty` or `three-quarter`. `validation.json` records the common inspector’s findings. `scene-report.json` adds measured character bounds, mesh and curve counts, camera coverage and the five PNG hashes.

## Self-review

Preview renders revealed skin coming through the first collar and a sharp bend in the hair over the shoulder. I rebuilt the neckline from a shared skin boundary, reshaped the long hair and adjusted the forearms after checking the side and back. The final portrait captures the tilted smile, brown eyes, asymmetric hair and pink crew neck. The skin is smoother than the photograph, the brows and waves are more regular, and the change of hair direction at the nape remains visible. The sweater and individual hairs are authored for this fixed pose.

## Attribution

The common anatomical starting mesh is MakeHuman **hm08**, released under **CC0**. The source header credits Data Collection AB, Joel Palmius and Jonas Hauquier. This build uses the supplied cropped topology in metres. The pinned source revision is `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`. [Asset licence](input/ASSET-LICENSE.md) · [Pinned MakeHuman source](https://raw.githubusercontent.com/makehumancommunity/makehuman/a8bc2d54ff0ac92e78ff71431b1023eda42bf482/makehuman/data/3dobjs/base.obj)
