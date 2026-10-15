"""Development helper: crop input/reference.png with a pixel grid for landmark picking.

    Blender --background --factory-startup --python tools/crop_grid.py -- x0 y0 x1 y1 scale out.png [pts.json]

Coordinates are reference-photo pixels, origin top-left.  Minor grid every 10 px (dark),
major every 50 px (cyan).  Optional pts.json = {"name": [x, y], ...} is drawn as red dots.
"""
import bpy, sys, os, json
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
x0, y0, x1, y1, s = (int(v) for v in a[:5])
out = a[5]
pts = json.load(open(a[6])) if len(a) > 6 else {}
img = bpy.data.images.load(os.path.abspath("input/reference.png"))
W, H = img.size
px = np.array(img.pixels[:], np.float32).reshape(H, W, 4)[::-1]      # top-left origin
crop = px[y0:y1, x0:x1].copy()
crop = np.repeat(np.repeat(crop, s, 0), s, 1)
h, w = crop.shape[:2]
for gx in range(x0 - x0 % 10 + 10, x1, 10):
    c = (gx - x0) * s
    major = gx % 50 == 0
    crop[:, c, :3] = (0, 1, 1) if major else crop[:, c, :3] * 0.55
for gy in range(y0 - y0 % 10 + 10, y1, 10):
    r = (gy - y0) * s
    major = gy % 50 == 0
    crop[r, :, :3] = (0, 1, 1) if major else crop[r, :, :3] * 0.55
for k, (x, y) in pts.items():
    cx, cy = int((x - x0) * s), int((y - y0) * s)
    if 0 <= cx < w and 0 <= cy < h:
        crop[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3, :3] = (1, 0, 0)
o = bpy.data.images.new("crop", w, h)
o.pixels[:] = crop[::-1].ravel()
o.filepath_raw = os.path.abspath(out)
o.file_format = 'PNG'
o.save()
print("wrote", out, w, h)
