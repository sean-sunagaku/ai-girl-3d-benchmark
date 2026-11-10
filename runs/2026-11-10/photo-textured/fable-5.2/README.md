# Claude Fable 5.2 — cafe-selfie-v1, second pass (photo textures allowed)

Upper-body portrait built with Blender 5.1.1 Python (`bpy`) from the supplied CC0 MakeHuman base, the landmark
file and the reference photograph. This is the finishing pass over my own procedural-only first pass. The
restriction on photographic character textures was lifted for this pass, so **the photograph is now used as a
texture source on the real 3D head** (and for a few sampled colours). It is never used as a plane, a background
or a composite; every pixel of the five PNGs is a raw Cycles render of the saved 3D scene.
Nothing was downloaded, no image/3D generation service and no additional agents were used.

## Image access (honest statement)

Direct visual access. I opened the decoded `input/reference.png` with an image-reading tool, and I opened my own
previous `renders/beauty.png` and `renders/side.png` the same way before starting this pass. Numbers used by the
scripts (14 landmark pixel positions, a few region medians) were read from gridded zoom crops of the photo or
are computed from its pixels inside `build.py`. I did not try to identify the person.

## Open / rebuild

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

`build.py` (helpers in `fable/`) reads only files in this directory: `input/base.obj`, `input/upper-body.obj`,
`input/environment.json` and `input/reference.png`. It rebuilds the whole character, packs the generated
textures, saves `character.blend`, re-opens that saved file, renders the five views from it and writes
`scene-report.json`. Cycles, CPU, 3 threads, 32 samples, OpenImageDenoise; no GPU. About 5 to 8 minutes here,
depending on what else the machine is doing.

`source-character.blend` is my unmodified first-pass scene (byte-identical copy). **`build.py` does not read
it** — the second pass regenerates everything from the supplied inputs with the same pipeline — it is kept only
as the reference for what changed, and is never overwritten.

Open `character.blend` in Blender 5.1: saved with `Cam_Beauty`, the cafe set and the portrait lights active.
`Cam_diagnostic_*` are orthographic cameras at 0°, 45°, 90° and 180°. For those four views the script only
switches the environment (cafe and portrait lights off, grey studio world and lights on); character geometry,
textures, materials and pose are identical in all five images.

## What changed from my first pass

1. **Photo-projected face texture** (`fable/photo.py`, `fable/skin.py`). The photograph is projected through the
   fitted selfie camera onto the posed, sculpted head and baked per texel into the MakeHuman UV layout
   (`Skin_Color_Generated`, 2048², packed):
   - *Registration.* 14 landmarks (pupils, eye corners, nose tip, subnasale, lip top/centre/bottom, mouth
     corners, chin) are projected from the 3D model and paired with their hand-measured photo positions; a
     thin-plate spline maps model image space to photo space (mean residual before the warp ≈ 8 px, max ≈ 17 px
     at 1205×1306). This is what keeps lips on the lips and avoids doubled or shifted features.
   - *Masking.* A texel takes photo colour only if it faces the camera, is not hidden by other geometry
     (ray-cast against the posed head), falls on clean skin in the photo (bright warm pixels; hair, lashes and
     background are eroded away) and lies in the central face below the eye line. Lips are admitted by their
     geometric vermilion mask. Lid margins and the eye openings are excluded (eyeballs are separate geometry).
   - *Unobserved or covered areas* — forehead and brows (hidden by the fringe in the photo), lids, ears, neck,
     scalp and every surface turned away from the camera — keep the generated complexion from the first pass.
   - *Levels.* The photo is already lit and exposed, so its values are not albedo: both sources are brought to
     one common level (generated value, part of the photographed hue), and the remaining low-frequency
     difference is removed towards the edge of the projected area so there is no visible mask outline.
2. **Sampled colours.** Iris palette (authored fibre structure, palette mixed 50/50 with the median photographed
   iris), hair shader (melanin/redness derived from the median photographed hair colour) and yarn colours
   (lit and shaded medians of the photographed sweater) now come from the photo. The knit relief itself and all
   other maps remain generated.
3. **Repair.** The hair tucked behind her left ear used to drop in front of the left shoulder and cover the neck
   and collar; it now falls behind the shoulder, as in the photo.

Everything else is the first-pass pipeline (see the source comments): analytic re-shaping of the CC0 base with
lids that keep wrapping the enlarged eyeballs, refracting corneas, lash/brow curves on the real lid margins,
random-walk skin shading, a structured raglan sweater with stitch-unit UVs settled on the posed body plus a
fibre halo, ~1,700 collision-aware locks expanded to ≈98,000 strands, and a minimal 3D cafe behind a shallow
depth of field. The background is geometry only; no photo pixels are in it.

## Photographic vs. generated, at a glance

| Part | Source |
| --- | --- |
| Cheeks, nose, lips, chin (camera-facing, below the eye line) | supplied photo, projected and baked to UV |
| Forehead, lids, brows region, ears, neck, scalp, side/back of head, hands | generated, tone-matched |
| Iris, hair, yarn colour | generated structure with colours sampled from the photo |
| Eyeball geometry, corneas, lashes, brows, hair strands, sweater, cafe | geometry and procedural materials |

## Editable resources

`Body` keeps its Subdivision modifier and `UVMap`; `Skin_Color_Generated` / `Skin_Data_Generated` are packed
images that can be repainted. `Hair` is a native Curves object with a Geometry Nodes children modifier
(count and spread exposed), `Sweater_Fuzz` is a Geometry Nodes scatter (density and length exposed), the
sweater has a live Subdivision modifier and per-panel stitch UVs. All materials are node graphs.
`Hair_Export_Proxy` is a ribbon mesh that is invisible to every Cycles ray; it only exists for mesh-only
exchange formats and contributes to no render.

## Honest self-review

Closer than the first pass, still visibly CG, and not a scan or a perfect likeness.

- The mouth is the clear gain: real lip colour, gloss and the shape of the smile now sit on the modelled lips.
  Cheeks and nose carry the photo's soft flush instead of painted blobs.
- The face is still a generic base re-shaped analytically; the photographed features are stretched over
  proportions that only approximate the person, and the upper face (forehead, lids, brows) is not photographic
  at all because the fringe covers it in the photo.
- The photo's soft shading is baked into the albedo and adds to the scene lighting (only level/hue matched,
  not de-lit). Under the neutral studio light of the diagnostic views the texture reads slightly flatter than
  in the beauty shot.
- Side and back views show generated skin beyond the cheeks; the hand-over is soft but it is a hand-over.
- Hair, knit, hands and the cafe have the limits described in the first pass: a tidy procedural groom, a
  bump/colour knit tile, loosely posed hands outside the beauty frame, a stage-set cafe.

Unobserved-area assumptions: the side and back of the head, the right ear, the hair behind the shoulders, the
back and lower part of the sweater, hands and forearms, the chair and the room are inferred.

## Files

- `build.py`, `fable/*.py` — the build; `tools/*.py` — development helpers (previews, landmark comparison).
- `character.blend` — final editable scene; `source-character.blend` — my first-pass scene, untouched.
- `renders/beauty.png` (960×1040), `renders/front.png`, `three-quarter.png`, `side.png`, `back.png` (768×768).
- `scene-report.json` — measured facts incl. texture sources/projection, sampled colours and limitations.
- `previews/` — contact sheets from both passes (`11_`…`15_` are from this pass; `11_` shows the baked UV map
  next to the photo blend weight).

## Third-party attribution

The body topology is the MakeHuman hm08 base mesh (`input/base.obj`, pinned commit
`a8bc2d54ff0ac92e78ff71431b1023eda42bf482`), released under CC0 by its header; see `input/ASSET-LICENSE.md`.
The reference photograph is the supplied input. Everything else here was written or generated by the entrant.
