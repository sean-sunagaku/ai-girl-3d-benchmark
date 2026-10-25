# Kimi 3D result viewer

This static HTML page displays the new Kimi K3.1 Pro submission. It reads
`results.json`, shows the original reference and raw Blender renders, and loads
the entrant's exported GLB for orbit/zoom inspection. It never calls an AI API.

The current completed result is the procedural first stage. The photo-textured
second stage has not been executed because the official API reached its daily
token quota. Its selector is disabled and the reason is visible; there is no
substituted or copied second-stage model. A future completed stage can be added
to `results.json` to enable it.

From the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/viewer/`. Serve the repository root, since the page
links to the shared inputs and run artifacts. A direct `file://` open may block
JSON/GLB fetching. JavaScript and WebGL are required for the interactive model;
the saved PNG files and run documentation remain separately accessible.

`vendor/model-viewer.min.js` is Google's model-viewer 4.1.0, copied from the
official Google Hosted Libraries distribution. Its Apache 2.0 license is in
`vendor/MODEL-VIEWER-LICENSE.txt`. No package installation or CDN connection is
needed to use this page. GLBs are uncompressed so no remote decoder is needed.

The viewer preserves the generated Blender output. A GLB is a web export,
so procedural shaders, fine hair, illumination and material appearance can
differ from the editable `.blend` and its raw renders. Keep both when comparing.

## Design baseline

The user's requested HTML result display is the written baseline: original photo,
two stage selectors, actual rotatable 3D output, five saved views per stage,
download/source links, and the factual run record. The layout uses a white/gray
surface, a muted green accent, system typography and a 2-column desktop work area;
mobile stacks the reference above the result. There are no illustrative mockups
or generated replacement portraits. The model's own results supply every preview.
