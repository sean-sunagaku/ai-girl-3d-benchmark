"""Geometry helpers. Authoring coordinates are millimetres; the Blender scene is in metres."""
import math
import bpy
import numpy as np
from mathutils import Vector, Euler
from math import sin, cos, pi, exp, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sat(x):
    return min(1.0, max(0.0, x))

def smooth(a, b, x):
    t = sat((x-a)/(b-a))
    return t*t*(3-2*t)

def gauss(x, width):
    return exp(-(x/width)**2)

def lerp(a,b,t):
    return a*(1-t)+b*t

def vec(p):
    return Vector(p)

def meter(p):
    return tuple(float(c)*.001 for c in p)

def col(name, parent=None):
    coll=bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(coll)
    return coll

def mesh(name, points, faces, material, collection, uvs=None, smooth_shade=True):
    data=bpy.data.meshes.new(name)
    data.from_pydata([meter(p) for p in points], [], faces)
    data.update()
    if smooth_shade:
        data.polygons.foreach_set('use_smooth', np.ones(len(faces),dtype=bool))
    obj=bpy.data.objects.new(name,data)
    collection.objects.link(obj)
    if material:
        data.materials.append(material)
    if uvs is not None:
        uv=data.uv_layers.new(name='Pattern coordinates')
        uv.uv.foreach_set('vector',np.asarray(uvs, dtype=np.float32).reshape(-1))
    return obj

def subdiv(obj, n=2):
    s=obj.modifiers.new('Silhouette smoothing','SUBSURF')
    s.levels=n
    s.render_levels=n
    return s

def principle(name, color, rough=.45, **values):
    m=bpy.data.materials.new(name)
    m.diffuse_color=(*color,1)
    m.use_nodes=True
    b=m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value=(*color,1)
    b.inputs['Roughness'].default_value=rough
    for k,v in values.items():
        b.inputs[k].default_value=v
    return m,b

def node(m,kind,name,x,y):
    n=m.node_tree.nodes.new(kind)
    n.label=name
    n.location=(x,y)
    return n

def link(m, a, b):
    m.node_tree.links.new(a,b)

def attr_color(obj,name,colors):
    a=obj.data.color_attributes.new(name=name,type='FLOAT_COLOR',domain='POINT')
    a.data.foreach_set('color',np.array(colors,dtype=np.float32).reshape(-1))
    obj.data.color_attributes.active_color=a

def strands(name,paths,radii,material,collection):
    data=bpy.data.hair_curves.new(name)
    sizes=np.array([len(path) for path in paths],dtype=np.int32)
    data.add_curves(sizes.tolist())
    pos=np.concatenate(paths,axis=0).astype(np.float32)*.001
    data.position_data.foreach_set('vector',pos.reshape(-1))
    radius=data.attributes.new('radius','FLOAT','POINT')
    radius.data.foreach_set('value', np.concatenate(radii).astype(np.float32)*.001)
    if material.name=='Hair | chestnut fibre scattering':
        mid=np.array([path[len(path)//2] for path in paths]);tone=np.clip(.52+.22*np.sin(.073*mid[:,0]+.025*mid[:,2])+.17*np.sin(.155*mid[:,1]-.038*mid[:,2])+.10*np.cos(.225*mid[:,0]+.08*mid[:,1]),0,1)
        dye=data.attributes.new('Lock pigment','FLOAT','CURVE');dye.data.foreach_set('value',tone.astype(np.float32))
    obj=bpy.data.objects.new(name,data)
    collection.objects.link(obj)
    data.materials.append(material)
    obj['curve_count']=len(sizes)
    return obj

def aim(obj, point):
    obj.rotation_euler=(Vector(point)-obj.location).to_track_quat('-Z','Y').to_euler()

PIVOT=Vector((0,-25,1404))
HEAD_R=Euler(tuple(math.radians(i) for i in (1.5,-26.1,-6.0)),'XYZ').to_matrix()
HEAD_M=np.asarray(HEAD_R,dtype=float)
HEAD_OFFSET=np.array((-2.,-3.,-16.))

def head(p):
    return HEAD_M@(np.asarray(p)-np.asarray(PIVOT))+np.asarray(PIVOT)+HEAD_OFFSET

def head_array(p):
    return (np.asarray(p)-np.asarray(PIVOT))@HEAD_M.T+np.asarray(PIVOT)+HEAD_OFFSET

def pose(p):
    p=np.asarray(p,dtype=float)
    w=smooth(1358,1450,p[2])
    return lerp(p,head(p),w)

def catmull(knots,t):
    """Uniform Catmull-Rom interpolation with clamped ends, scalar or array parameter."""
    p=np.asarray(knots,dtype=float)
    q=np.clip(np.asarray(t),0,1)*(len(p)-1)
    j=np.minimum(np.floor(q).astype(int),len(p)-2)
    u=(q-j)[...,None]
    a=p[np.maximum(j-1,0)]; b=p[j]; c=p[j+1]; d=p[np.minimum(j+2,len(p)-1)]
    return .5*((2*b)+(-a+c)*u+(2*a-5*b+4*c-d)*u*u+(-a+3*b-3*c+d)*u*u*u)
