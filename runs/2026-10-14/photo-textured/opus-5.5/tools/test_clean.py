"""Development helper: write the photo-space skin mask / cleaned photo for inspection."""
import bpy, sys, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import photo_fit as pf, photo_texture as pt
photo = pt.load_rgb(os.path.join(ROOT, "input", "reference.png"))
clean, skin, lips = pt.photo_skin_layers(photo, pf.PHOTO_LM)
def save(a, name):
    H, W = a.shape[:2]
    if a.ndim == 2: a = np.repeat(a[..., None], 3, 2)
    o = bpy.data.images.new(name, W, H)
    rgba = np.ones((H, W, 4), np.float32); rgba[..., :3] = a
    o.pixels.foreach_set(rgba[::-1].ravel())
    o.filepath_raw = os.path.join(ROOT, "work", name); o.file_format = 'PNG'; o.save()
x0,y0,x1,y1 = 380,150,920,720
save(clean[y0:y1,x0:x1], "clean.png")
save(skin[y0:y1,x0:x1], "skinmask.png")
