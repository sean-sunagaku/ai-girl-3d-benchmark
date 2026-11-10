"""Make gridded zoom crops of an image for reading landmark pixel coordinates (reference only)."""
import sys
from PIL import Image, ImageDraw, ImageFont
def crop(src, out, x0, y0, x1, y1, scale=2, step=50):
    im = Image.open(src).convert('RGB').crop((x0, y0, x1, y1))
    im = im.resize(((x1-x0)*scale, (y1-y0)*scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    try: font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 13)
    except Exception: font = ImageFont.load_default()
    for x in range((x0//step+1)*step, x1, step):
        X = (x-x0)*scale; d.line([(X,0),(X,im.size[1])], fill=(0,255,255), width=1); d.text((X+2,2), str(x), fill=(255,255,0), font=font)
    for y in range((y0//step+1)*step, y1, step):
        Y = (y-y0)*scale; d.line([(0,Y),(im.size[0],Y)], fill=(0,255,255), width=1); d.text((2,Y+2), str(y), fill=(255,255,0), font=font)
    im.save(out, quality=92)
if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]; a = [int(v) for v in sys.argv[3:7]]
    scale = int(sys.argv[7]) if len(sys.argv) > 7 else 2; step = int(sys.argv[8]) if len(sys.argv) > 8 else 50
    crop(src, out, *a, scale=scale, step=step)
