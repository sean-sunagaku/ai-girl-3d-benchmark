# Additional 3D result viewer

This static HTML page displays the GPT-6.1 Terra and Kimi K3.1 Pro submissions.
It reads `results.json`, lets visitors switch between the two models, shows the
shared original reference and each model's raw Blender renders, and loads the
selected model's exported GLB for orbit/zoom inspection. It never calls an AI API.

Both models have a completed procedural first stage. Terra's photo-textured
second stage was outside this run's scope. Kimi's second stage was not executed
because the official API reached its daily token quota. Each model's second-stage
selector is disabled with its own reason; there is no substituted or copied
second-stage model. A future completed stage can be added to that model's
`results.json` entry to enable it.

From the repository root:

```sh
python3 -m http.server 8765 --bind localhost
```

Open `http://localhost:8765/viewer/`. Serve the repository root, since the page
links to the shared inputs and run artifacts. A direct `file://` open may block
JSON/GLB fetching. JavaScript and WebGL are required for the interactive model;
the saved PNG files and run documentation remain separately accessible. The
`?model=kimi-k3.1-pro` query selects Kimi directly; Terra is the default.

`vendor/model-viewer.min.js` is Google's model-viewer 4.1.0, copied from the
official Google Hosted Libraries distribution. Its Apache 2.0 license is in
`vendor/MODEL-VIEWER-LICENSE.txt`. No package installation or CDN connection is
needed to use this page. GLBs are uncompressed so no remote decoder is needed.

The viewer preserves the generated Blender output. A GLB is a web export,
so procedural shaders, fine hair, illumination and material appearance can
differ from the editable `.blend` and its raw renders. Keep both when comparing.

## Design baseline

The user's requested HTML result display is the written baseline: original photo,
model and stage selectors, actual rotatable 3D output, five saved views per
completed stage, download/source links, and the factual run record. The layout uses a white/gray
surface, a muted green accent, system typography and a 2-column desktop work area;
mobile stacks the reference above the result. There are no illustrative mockups
or generated replacement portraits. The model's own results supply every preview.
