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

For a metered API entrant, confirm the exact API model ID against the provider's
model list and record the harness's reasoning/output settings. Keep task prompts
byte-identical and record routing/retry adaptations separately. Enforce the
user's approved spending limit before paid requests; a deposited balance or
voucher is not permission for extra purchases or a different spending limit.
Never put credentials in prompts, source files, HTML, run records, or Git.

For cache-aware metering, use the provider's reported cached-input count and
published cache-hit price after a completed request; reasoning tokens are
already part of completion tokens. Reserve the maximum possible next-request
cost before calling, and retain the reservation if usage is unknown. Check
request/day quotas as well as balance. If a provider daily quota stops a later
stage, preserve completed stages and label the missing stage as not executed;
do not display a copied earlier asset as a completed later result or silently
switch accounts/providers. Document coordinator compatibility fixes and actual
script/image execution separately from model-authored claims.

When the user also requests an HTML result view, link the new entrant's genuine
renders and exported GLB from `viewer/results.json`. Verify both stage selectors,
all five saved views, orbit/zoom, download links, failure handling, and a narrow
mobile layout. Export only explicitly selected character objects/collections
from an opened copy/in-memory scene, preserve the source `.blend`, and disclose
Blender-to-glTF material/hair differences. `scripts/inspect_scene.py` measures a
new scene and can create this requested export; do not apply it to historical
submissions unless separately requested. Keep new run metadata and technical
validation distinct from an appearance score.

## Optional reproduction or technical inspection

Run existing Blender code, rerender, or perform technical validation only when the user requests it. Work on a copy, preserve the original submission, and follow the entrant's README. The publication snapshot was intentionally shipped without new validation. Do not interpret historical success records as a fresh check.

If requested, record whether files open, dependencies are present, expected views are produced, and the source remained unchanged. Keep technical findings distinct from appearance; leave unmeasured values unknown.

