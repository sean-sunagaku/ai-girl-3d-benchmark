import bpy, numpy as np, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath="input/upper-body.obj", forward_axis='Y', up_axis='Z')
ob = bpy.context.selected_objects[0]; me=ob.data
co = np.array([v.co for v in me.vertices]); ed=np.array([e.vertices[:] for e in me.edges])
def plot(name, sel_fn, ax0, ax1, lo, hi, W=800):
    img=np.ones((W,W,4),np.float32)
    def px(p):
        u=(p[ax0]-lo[0])/(hi[0]-lo[0]); v=(p[ax1]-lo[1])/(hi[1]-lo[1]); return u*(W-1), v*(W-1)
    m=sel_fn(co)
    for a,b in ed:
        if m[a] and m[b]:
            (u0,v0),(u1,v1)=px(co[a]),px(co[b])
            for t in np.linspace(0,1,40):
                u=int(u0+(u1-u0)*t); v=int(v0+(v1-v0)*t)
                if 0<=u<W and 0<=v<W: img[v,u,:3]=0
    # grid every 5mm
    for g in np.arange(lo[0],hi[0],0.005):
        u=int((g-lo[0])/(hi[0]-lo[0])*(W-1)); img[:,u,0]=np.minimum(img[:,u,0],0.8)
    for g in np.arange(lo[1],hi[1],0.005):
        v=int((g-lo[1])/(hi[1]-lo[1])*(W-1)); img[v,:,0]=np.minimum(img[v,:,0],0.8)
    im=bpy.data.images.new(name,W,W); im.pixels=img.ravel(); im.filepath_raw="previews/%s.png"%name; im.file_format='PNG'; im.save()
# side section through eye: x in [0.029,0.033]; axes (-y) horizontal? use y,z
plot("sec_eye", lambda c: np.abs(c[:,0]-0.031)<0.0025, 1,2, (-0.17,1.51),(-0.10,1.58))
plot("sec_mouth", lambda c: np.abs(c[:,0]-0.0)<0.0012, 1,2, (-0.17,1.45),(-0.10,1.52))
plot("sec_eye_top", lambda c: np.abs(c[:,2]-1.546)<0.002, 0,1, (0.005,-0.17),(0.06,-0.10))
