"""Development helper: dump Skin_Photo_RGBA (rgb / alpha) next to the opened .blend for inspection."""
import bpy, numpy as np, os
img = bpy.data.images["Skin_Photo_RGBA"]
W, H = img.size
a = np.empty(W*H*4, np.float32); img.pixels.foreach_get(a); a = a.reshape(H, W, 4)
print("alpha stats", a[...,3].min(), a[...,3].max(), (a[...,3]>0.5).sum(), "rgb mean", a[...,:3].mean((0,1)))
for nm, arr in (("tex_rgb.png", a[..., :3]), ("tex_alpha.png", np.repeat(a[..., 3:4], 3, 2))):
    o = bpy.data.images.new(nm, W, H); rgba = np.ones((H, W, 4), np.float32); rgba[..., :3] = arr
    o.pixels.foreach_set(rgba.ravel()); o.filepath_raw = os.path.abspath("../work/" + nm) if False else os.path.join(os.path.dirname(bpy.data.filepath), nm)
    o.file_format = 'PNG'; o.save()
m = bpy.data.materials["Skin"]
for l in m.node_tree.links:
    if "Mix" in l.to_node.name or l.from_node.name.startswith("Skin_Photo"): print("LINK", l.from_node.name, l.from_socket.name, "->", l.to_node.name, l.to_socket.identifier)
