# 3D Upper-Body Portrait & Turnaround Scene (`character.blend`)

## 1. How to Build and Open the Scene

### Reproducible Build & Render (Headless CLI)
To regenerate `character.blend`, `scene-report.json`, and all 5 required renders (`renders/beauty.png`, `renders/front.png`, `renders/three-quarter.png`, `renders/side.png`, and `renders/back.png`) from scratch:

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

For a fast preview iteration (`480x520` beauty, `384x384` turnaround views, 10 Cycles samples):

```bash
PREVIEW=1 /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 3 --python build.py
```

### Opening the Saved Scene
Open `character.blend` in Blender 5.1.1 (`/Applications/Blender.app`).
- The scene is saved with `BeautyCamera` as the active camera (`960x1040` resolution), full Cycles node materials, packed procedural textures, and all character objects visible (`hide_viewport = False`, `hide_render = False`).
- The Outliner is organized into three collections:
  - `Character`: `CharacterHead`, `Eyeball_L`, `Eyeball_R`, `Cornea_L`, `Cornea_R`, `EyeTearline_L`, `EyeTearline_R`, `Eyelashes_L`, `Eyelashes_R`, `Eyebrows_L`, `Eyebrows_R`, `HairScalpCap`, `HairVolumeLocks`, `HairStrands_Bangs`, `HairStrands_FaceFraming`, `HairStrands_Main`, `SweaterBody`, and `SweaterCollar`.
  - `CafeEnvironment`: 3D out-of-focus cafe background (`CafeWindow_Left`, `CafePlants`, `CafeTable_Left`, `CafeChair_Left`, `CafeCup_Left`, `CafeWall_Back`, `CafeCounter_Right`, `CafeEspressoMachine`, `CafeShelf_UpperRight`, `CafeShelf_MidRight`, `CafeJar_1..3`), hidden during diagnostic turnaround renders and restored before saving `character.blend`.
  - `LightsAndCameras`: `BeautyCamera`, `Cam_Front`, `Cam_ThreeQuarter`, `Cam_Side`, `Cam_Back`, `KeyLight_Window`, `FillLight_CafeBounce`, `RimLight_TopBack`, and `BounceLight_Table`.

---

## 2. Reference-Image Access & Design Choices

- **Reference Image Access**: `input/reference.png` (`960x1040` portrait photograph) was inspected directly and analyzed quantitatively in Blender Python (`bpy.data.images.load("input/reference.png")`) to measure 2D landmark coordinates, head roll angle (`18.4°` toward the character's right shoulder), eye color/contrast (`min sRGB ≈ 0.08..0.15`), skin tone, and dusty-rose sweater chromaticity.
- **Starter Asset Separation & Sculpting (`build_head_and_sweater_meshes`)**:
  - `input/upper-body.obj` (CC0 1.0 MakeHuman `hm08` female upper-body mesh) was loaded and cleanly partitioned into `CharacterHead` (head and neck above the collar curve) and `SweaterBody` (torso and sleeves below the collar curve) rather than painting clothing onto skin.
  - Facial likeness was sculpted onto `CharacterHead` via smooth spatial radial-basis fields: fuller youthful cheeks, soft v-line jaw taper, refined nose bridge/tip, closed-mouth smile with lifted oral commissures (`+4.2 mm`), and wide open almond eye apertures (`>11 mm` vertical opening) with aegyo-sal lower lid fullness.
- **3D Pose & Clean Garment Boundaries (`pose_point_array`)**:
  - The character is posed in 3D with an `18.5°` lateral head tilt toward her right shoulder (`-X`), `-4.0°` pitch, `+4.5°` yaw, a gentle right-shoulder shrug, and natural arm adduction (`22°` downward from the starter `45°` A-pose, `-18°` forward pitch, `-16°` inward yaw).
  - The raw polygon cut at `z = 1.106` on the starter mesh was regularized: the waist boundary was leveled and extruded into 4 smooth rings terminating on a horizontal plane (`z = 0.962`, `0.0 mm` sawtooth variation) with an inward-rolled ribbed hem, and the forearm sleeve cuts were projected perpendicular to the forearm axis and rolled inward into finished cuffs.

---

## 3. Geometry & Material Techniques

- **Two-Layer Anatomical Eyes, Lashes & Brows (`build_eyes_lashes_brows`)**:
  - Each eye consists of `Eyeball_L/R` (sclera + concave iris + dark pupil mapped with a `1024x1024` procedural `EyeIrisTex` featuring radial stroma fibers, collarette ring, and limbal ring) and an outer `Cornea_L/R` shell with anterior corneal bulge.
  - `Mat_Cornea` uses a camera-ray-only Fresnel (`IOR = 1.24`) + low-roughness Glossy BSDF (`0.008`) mix over Transparent BSDF, producing crisp window catchlights without veiling the underlying pupil or casting shadows onto the iris.
  - 3D tapered mesh eyelashes (`Eyelashes_L/R`), wet tear meniscus (`EyeTearline_L/R`), and 3D arched eyebrows (`Eyebrows_L/R`) frame both eyes.
- **Multi-Layered 3D Hair System (`build_hair_system`)**:
  - Combines `HairScalpCap`, `HairVolumeLocks` (74 perimeter locks + 11 crown-to-bangs locks), and three Blender 5.1 `CURVES` objects (`HairStrands_Bangs`, `HairStrands_FaceFraming`, `HairStrands_Main` totaling ~4,500 3D hair curves rendered as `3D_CURVES` ribbons).
  - Cross-sectional binormals on `HairVolumeLocks` are orthogonalized against the true 3D cranial surface normal so ribbons lie flat across the crown dome (`0` crown gaps across all camera views), while a dilated 2D cylindrical heightfield envelope (`build_posed_body_radial_envelope`) keeps all locks outside the posed `SweaterBody` and `SweaterCollar`.
  - Central see-through bangs terminate at eyebrow level (`z >= 1.560`) with explicit eye-corridor clearance so `0` bang curve points occlude either eye.
- **Chunky Dusty-Rose Knit Sweater (`SweaterBody`, `SweaterCollar`, `Mat_Sweater`)**:
  - `SweaterBody` uses `Solidify` (`4.5 mm`) and `Subdivision` (`levels=1`, `render_levels=2`) modifiers paired with a `3D` ribbed crew-neck collar (`SweaterCollar`).
  - `Mat_Sweater` combines a `2048x2048` procedural tileable moss/seed-stitch heightmap (`SweaterKnitHeightTex`) and stitch-AO dusty-rose albedo map (`SweaterKnitAlbedoTex`) with `Principled BSDF` fabric sheen (`Sheen Weight = 0.28`).

---

## 4. Self-Review & Remaining Limitations

- **Self-Review**:
  - Verified all 5 output renders at final resolution (`renders/beauty.png` at `960x1040`, `renders/{front,three-quarter,side,back}.png` at `768x768`) via both visual inspection and raycast/pixel-level diagnostics in Blender.
  - Confirmed `0` crown scalp gaps across `BeautyCamera`, `Cam_Side`, and `Cam_Back`, `0` hair-to-sweater mesh penetrations, `0` bang strands blocking the eye apertures, dark pupil / warm chestnut iris readout (`min sRGB ≈ 0.06..0.09`), and clean horizontal waist/cuff hems.
- **Approximations & Remaining Limitations**:
  - **Single-View Inference**: Because `input/reference.png` is a single front portrait cropped at the chest/upper arms, the exact side profile, back hair wave arrangement, back sweater drapery, and lower waist/cuff terminations are anatomically inferred.
  - **Sculpted Base Topology**: Facial likeness is achieved by deforming the CC0 MakeHuman `hm08` mesh topology rather than a 3D photogrammetry scan of the subject.
  - **Hybrid Hair Representation**: To stay well within the `<1,000,000` evaluated vertex limit (`234,401` evaluated vertices, `111,464` evaluated faces) and CPU render budget (`~75 s` for all 5 production renders), the hair uses 85 sculpted 3D volume locks layered with ~4,500 `CURVES` strands rather than ~100,000 individual follicle curves.
  - **Stylized 3D Cafe Environment**: The background in `renders/beauty.png` is a stylized 3D geometric cafe scene blurred via camera depth of field (`f/1.8`).

---

## 5. Attribution & Provenance

- **Starter Base Mesh (`input/upper-body.obj`)**: Derived from the MakeHuman project (`hm08` female base mesh), licensed under **CC0 1.0 Universal (Public Domain Dedication)**. See `input/PROVENANCE.md` for full details.
- **Reference Image (`input/reference.png`)**: Provided in the task workspace (`input/PROVENANCE.md`) as the visual target for pose, framing, lighting, hair, and knitwear styling.
