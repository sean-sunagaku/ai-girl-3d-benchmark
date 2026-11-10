# Claude Fable 5.2 — cafe-selfie-v1, procedural stage

Upper-body portrait built entirely with Blender 5.1.1 Python (`bpy`) from the supplied CC0 MakeHuman base,
the landmark file and the reference photograph (looked at and measured only). No photographic texture is used
anywhere, nothing was downloaded, no image/3D generation service and no additional agents were used.

## Did I actually see the photograph?

Yes. I opened `input/reference.png` with an image-reading tool and saw the decoded picture directly (a softly
smiling young woman with long chestnut hair and wispy bangs, head tilted toward her right shoulder, dusty-pink
chunky knit crew-neck, both sleeves reaching toward the camera, bright cafe with window plants, wooden
furniture, a small vase on a table). I also made gridded zoom crops to read 2D landmark pixel positions
(pupils, eye corners, nose tip, mouth, chin, collar) and coarse region colour averages. Those numbers only
guided hand-set constants in the scripts; no pixel of the photo is in any texture, material, background or render.
I did not try to identify the person.

## Build / open

```sh
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

`build.py` (helpers in `fable/`) reads only `input/base.obj`, `input/upper-body.obj` and
`input/environment.json`, builds everything, packs the generated images, saves `character.blend`, re-opens that
saved file, renders the five PNGs from it and writes `scene-report.json`. One run takes about 4.5 minutes here
(≈25 s build, ≈2.5 min beauty render, ≈20 s per diagnostic view). Cycles, CPU, 3 threads, 32 samples,
OpenImageDenoise; no GPU.

Development switches after `--`: `--quick` (lower densities), `--skip sweater,hair,cafe`, `--no-render`,
`--views beauty,front`. `tools/` holds the preview/measurement scripts I used while iterating
(`shot.py`, `landmarks.py`, `compare.py`, …); they are not needed for the build.

Open `character.blend` in Blender 5.1: it is saved with `Cam_Beauty`, the cafe set and the portrait lights
active. Collections: `Character` (`Body`, `Eyes_Brows_Lashes`, `Sweater`, `Hair`), `Cafe_Set`,
`Lights_Portrait`, `Lights_Diagnostic_Studio` (render-disabled in the saved state), `Cameras`.
The four `Cam_diagnostic_*` cameras are orthographic at 0°, 45°, 90° and 180° around the character
(0° = front, positive toward her left). For those four views the script only switches the *environment*
(cafe + portrait lights off, neutral grey world + studio lights on); character geometry, materials and
pose are identical in all five images.

## How it is made

**Body and face.** The body is the supplied MakeHuman hm08 mesh: the organisers' upper-body crop plus the
forearms and hands of the same supplied `base.obj` (same topology, same CC0 source), with MakeHuman's UVs.
`fable/sculpt.py` reshapes the rest-pose head with smooth analytic brushes: shorter mid-face, softer and wider
cheeks, smaller and lower nose, fuller lips with a slightly lop-sided closed-mouth smile, slimmer neck. The eyes
are enlarged by scaling each whole eye region about the eyeball centre and re-opening the lids in angle space
around the globe, so the lids always keep wrapping the eyeball; the inner lid surface is re-seated on the globe.
A build-time skeleton from the landmark file gets Blender's automatic (bone heat) weights; the selfie pose
(head tilt carried mostly by the neck, slight torso lean, both arms reaching forward by analytic two-bone IK,
relaxed fingers) is then baked into the mesh. There is no production rig.

**Eyes.** Separate eyeballs (sclera + recessed iris with a generated radial-fibre texture) under a refracting
cornea/tear-film shell with a real bulge, caruncles, ~740 eyelash curves rooted on the true lid-margin loops of
the topology and ~840 brow hairs with a flow field. Both eyes fixate the selfie camera. Light linking keeps the
two broad fill lights out of the corneas (they would mirror as a milky veil); a small catch light is added.

**Skin.** One generated 2K colour map and one data map (roughness / coat / lip-line height) painted by
`fable/skin.py` in MakeHuman UV space. A small rasteriser maps every texel to its rest-space position, so all
masks are analytic functions of anatomy: complexion mottling, blush, lip colour inside vermilion borders
measured on the topology, lash line, eyelid fold, contact shading around the nose, scalp under-tone with a
feathered hairline, moles. Shader: Principled BSDF, random-walk (skin) subsurface, coat on the lips, two scales
of restrained procedural micro-bump.

**Sweater.** `fable/sweater.py` builds one structured quad mesh in the rest pose the way a raglan sweater is
knitted: a yoke from the collar outward, a body tube and two sleeve tubes that share the under-arm stitches,
ribbed collar band, cuffs and hem with rolled rims. Every panel gets UVs in *stitch units* (arc length along
rows and columns), so the generated stockinette / 1x1-rib tiles (height, ambient occlusion and yarn twist
from `fable/knit.py`) run with the knit direction and wrap seamlessly; the only seams are the four raglan lines.
The garment is posed with weights transferred from the body, hangs under gravity on the sleeves and is settled
against the posed body with a small membrane-relaxation/collision solver. A Geometry Nodes modifier scatters
short fibres over it at render time for the mohair halo.

**Hair.** `fable/hair.py` samples roots on the scalp, integrates ~1,700 lock centre-lines from a comb field
(parting, fringe section, her left side swept over the temple and tucked behind the ear) into gravity, sliding
over a signed-distance grid of the posed body and sweater so strands never pass through skin or knit. Locks get
loose phase-coherent waves; ~11,000 authored strands are clumped around them with frizz, fly-aways and uneven
ends, and the fringe is gathered into wisps. They are stored as a native `Curves` object; a Geometry Nodes
modifier multiplies them ×9 (≈98,000 rendered strands, Principled Hair BSDF, melanin model).

**Previews first.** I rendered low-resolution previews after every stage and looked at the actual images
(a few contact sheets are kept in `previews/`). Problems found and fixed that way include: the jaw opening
because the neck bone pulled on the chin (head weights made rigid), a gap at the outer eye corners (inner lid
re-seated on the globe), broad fill lights veiling the corneas (light linking), hair shooting off the scalp at
the parting (adhesion to the hair layer), locks orbiting the ear way-point, ribbon geometry showing through the
strands (made ray-invisible), an inverted distance-field sign from the open sweater shell, the knit following
body detail on the chest, a step at the lower edge of the supplied body crop, and stitches that were far too
faint at final resolution.

**Scene.** A minimal 3D cafe corner (glazed corner with mullions, three potted plants built from leaf cards,
wooden chairs and tables, a glass vase with flowers, counter and shelves) seen through a shallow depth of field.
The selfie camera is placed from the posed eyes so that the pupils land where they are in the reference frame
(same 960:1040 framing, 74° vertical field of view, about 34 cm from the face).

## Honest self-review

How close is it? The *arrangement* is close: framing, head tilt, the big foreground sleeve, long chestnut hair
with a see-through fringe, a clearly knitted dusty-pink sweater, a bright cafe behind. The facial *proportions*
measured through the camera match my 2D landmark measurements of the photo within a few percent of the pupil
distance. It is nevertheless visibly CG and not a likeness:

- The face reads as a smooth, doll-like character. The surface is an analytic re-shaping of a generic base
  with painted colour only: no pores, fine lines, natural asymmetry or the soft detail a sculptor would add.
- Hair is a combed/settled procedural groom, not a simulation: the top is too tidy, the fringe is blunter and
  the lengths are less airy than in the photo, and under the chin the mass is denser than it should be.
- The knit is a bump/colour tile on a smooth cloth surface; silhouettes are softened only by the fibre halo and
  the folds are simple. The hem line and the lower sweater are invented.
- Hands are loosely posed and intentionally outside the beauty frame; they would not survive a close-up.
- The cafe is a stage set that only works out of focus.

Inferred (not visible in the photo): the side and back of the head, the right ear, most of the left ear's
shape, all hair behind the shoulders, the back and lower part of the sweater, the hands and forearms, the
chair, the room beyond the frame.

`Hair_Export_Proxy` is a thin ribbon mesh along every third authored strand. It is invisible to every Cycles
ray and does not contribute to any render; it is there so mesh-only exchange formats can carry the hairstyle.

## Files

- `build.py`, `fable/*.py` — the build. `tools/*.py` — development helpers.
- `character.blend` — final editable scene (generated images packed).
- `renders/beauty.png` (960×1040), `renders/front.png`, `three-quarter.png`, `side.png`, `back.png` (768×768):
  raw Cycles renders of the re-opened saved scene, no compositing or retouching.
- `scene-report.json` — measured facts (versions, engine, objects, bounds, vertex/curve counts, generated images).

## Third-party attribution

The body topology is the MakeHuman hm08 base mesh (`input/base.obj`, pinned commit
`a8bc2d54ff0ac92e78ff71431b1023eda42bf482`), which its header releases under CC0; see `input/ASSET-LICENSE.md`.
Everything else in this directory was written or generated by the entrant.
