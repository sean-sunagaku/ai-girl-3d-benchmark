"""Side-by-side of the reference photo and a beauty render at equal height: compare.py render.png out.jpg [crop x0 y0 x1 y1 in 1205x1306 space]"""
import sys
from PIL import Image
r = Image.open(sys.argv[1]).convert('RGB'); ref = Image.open('input/reference.png').convert('RGB')
r = r.resize(ref.size, Image.LANCZOS)
if len(sys.argv) > 3:
    box = tuple(int(v) for v in sys.argv[3:7]); r = r.crop(box); ref = ref.crop(box)
h = 900; w = round(ref.size[0] * h / ref.size[1])
out = Image.new('RGB', (w * 2 + 8, h), (0, 0, 0)); out.paste(ref.resize((w, h), Image.LANCZOS), (0, 0)); out.paste(r.resize((w, h), Image.LANCZOS), (w + 8, 0))
out.save(sys.argv[2], quality=90); print(out.size)
