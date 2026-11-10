"""Tile preview PNGs into one JPEG contact sheet: sheet.py out.jpg cols cell_w a.png b.png ..."""
import sys
from PIL import Image
out, cols, cw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); files = sys.argv[4:]
ims = []
for f in files:
    im = Image.open(f).convert('RGB'); im = im.resize((cw, round(im.size[1] * cw / im.size[0])), Image.LANCZOS); ims.append(im)
rows = (len(ims) + cols - 1) // cols
rh = [max(im.size[1] for im in ims[r * cols:(r + 1) * cols]) for r in range(rows)]
W = Image.new('RGB', (cw * cols, sum(rh)), (24, 24, 24)); y = 0
for r in range(rows):
    for c, im in enumerate(ims[r * cols:(r + 1) * cols]): W.paste(im, (c * cw, y))
    y += rh[r]
W.save(out, quality=90); print(out, W.size)
