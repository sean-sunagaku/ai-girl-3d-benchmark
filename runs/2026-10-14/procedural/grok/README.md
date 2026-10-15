# Upper-body portrait

Editable clothed bust built in Blender 5.1.1 from the supplied MakeHuman cage and the reference photograph.

## Reference image

Yes. `input/reference.png` was opened with an image reader and the decoded photograph was visible: a softly smiling person in a dusty pink knit sweater, long chestnut hair with wispy bangs, warm brown eyes, a slight head tilt, arms reaching toward the camera, in a bright cafe. The file was not projected, used as a texture, or sent to an external generator. The person is not identified. Shape of the unseen side, back, ears, and the sweater under the arms is inferred.

## Build and open

From this directory:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

That imports `input/upper-body.obj`, sculpts and poses the character, saves `character.blend`, then renders the five PNGs with Cycles CPU, 32 samples, and OpenImageDenoise.

A faster look is:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py -- --preview
```

Open `character.blend` in Blender 5.1. The scene camera is `BeautyCamera`. Materials, the knit image, hair curves, and modifiers are packed or stored in the file.

## Design choices

The MakeHuman hm08 upper body is the anatomical cage (meters, Z-up, face toward -Y). Blender's OBJ importer adds a Y-up rotation; `build.py` clears it so the file coordinates stay the visual coordinates.

Rest-pose edits narrow and shorten the nose, soften the jaw, open the eyes, lift the mouth corners into a closed smile, slim the neck, and add a modest chest under the sweater. A two-bone IK pose leans the torso toward the camera, rolls the head slightly toward her right shoulder, and reaches both arms forward. Her right arm (negative X) comes closer to the lens. The pose is baked into the mesh.

Eyeballs are UV spheres with a procedural brown iris. A slightly larger cornea uses transmission. Lids are pulled onto a shell just outside each ball. Brows and lashes are small mesh strips.

The sweater is a shell of the torso and arms, thickened, with a packed 1024 stockinette height map driving displacement and bump, plus a fuzzy sheen. Hands stay the MakeHuman surface.

Hair is a scalp shell plus about a thousand beveled curve strands (chestnut Principled BSDF, with a thinner flyaway set). Strands are meant to hang, wave, and leave the eyes readable. The cafe is a few primitive shapes and warm area lights, hidden for the gray diagnostic views.

## Self-review

The bust reads as a person in a pink sweater taking a selfie: brown irises, a small nose, a slight closed-mouth smile, a crew neck, and arms in front. Knit ribs show on the sweater, especially from the back and side. Evaluated geometry is about 241k vertices.

It is not close to the photograph. The face is still a softened MakeHuman head, pale and mannequin-like next to the reference. Hair is the largest miss: the scalp shell shows as a smooth cap, strands are too straight and too thin, many cross the eyes and nose, and the back of the head is only partly covered. Bangs are not the light wisps in the photo. Hands stay blocky. The cafe is only a suggestion.

## Approximations

- Likeness, smile, and eye size are sculpted on the neutral cage, not a scan.
- Side and back of the skull, the hair mass behind the shoulders, the ears, and sweater folds on the unseen side are inferred.
- Iris color and catchlights are a shader on a sphere.
- Knit stitches are a procedural height map.
- Finger curl was left off because it pinched the base hands.

## Base mesh

MakeHuman hm08, commit `a8bc2d54ff0ac92e78ff71431b1023eda42bf482`, explicitly CC0. `input/upper-body.obj` is that body cropped to the upper body and converted to meters, Z-up, face toward -Y, without a likeness sculpt. See `input/ASSET-LICENSE.md`.
