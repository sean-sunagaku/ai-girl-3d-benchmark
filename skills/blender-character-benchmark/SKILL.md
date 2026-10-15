---
name: blender-character-benchmark
description: Use to view or reproduce the published AI Girl 3D Benchmark, or to run the same single-image Blender portrait task with another model using the shared inputs and prompts. Do not score appearance or rerun existing artifacts unless explicitly requested.
---

# Blender character benchmark

Find the repository root containing `benchmarks/cafe-selfie-v1` and read `README.md` and `docs/PROTOCOL.md`.

## View published results

1. Show the original `benchmarks/cafe-selfie-v1/inputs/reference.png` and the requested model/stage's images in `runs/2026-10-14`.
2. Use `GALLERY.md` for a comparison of all four models across both stages and five views.
3. For editable scenes, follow `docs/REPRODUCE.md` to download `blender-scenes.zip` from the same repository's Release.
4. Explain that GLBs are simplified existing exports; do not generate missing exports automatically.

Do not assign appearance scores, rankings or critiques. The benchmark lets people inspect the artifacts themselves. Existing model-authored self-reviews are historical source material.

## Prepare a new model run

Only when the user requests a new run, use `scripts/prepare_case.py` to copy the original inputs and prompt into a new directory. For the second stage, provide only that entrant's own completed first-stage workspace with `--previous`.

Give the model the complete unchanged `PROMPT.md`. Record the selected model ID, harness version, run time, input files, any adaptations to executable paths, and any recovery instructions. Keep all work in the new workspace. Never overwrite published submissions or borrow another entrant's work.

Actual generation requires a user-authorized model harness and installed Blender. Do not silently change providers or call an image-generation service instead of building a Blender scene.

## Optional reproduction or technical inspection

Run existing Blender code, rerender, or perform technical validation only when the user requests it. Work on a copy, preserve the original submission, and follow the entrant's README. The publication snapshot was intentionally shipped without new validation. Do not interpret historical success records as a fresh check.

If requested, record whether files open, dependencies are present, expected views are produced, and the source remained unchanged. Keep technical findings distinct from appearance; leave unmeasured values unknown.

