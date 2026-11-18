"""A shallow-focus café set and a neutral studio for the orbit cameras."""
from math import cos, pi, sin
from random import Random

import bpy
import numpy as np
from mathutils import Vector

from portrait_common import aim, catmull, collection, linear, mesh_object, node, surface

BEAUTY_LOC = (.014,-.685,1.616)
BEAUTY_TARGET = (-.044,-.045,1.412)
ORBIT_CENTER = (.002,-.300,1.343)
ORBITS = {'front':0,'three-quarter':45,'side':90,'back':180}


def move_col(o,col):
    for old in list(o.users_collection):
        old.objects.unlink(o)
    col.objects.link(o)
    return o


def box(name,col,loc,dim,mat,bevel=.006):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc)
    o=move_col(bpy.context.object,col)
    o.name=name
    for v in o.data.vertices:
        for i in range(3):
            v.co[i]*=dim[i]
    o.data.materials.append(mat)
    if bevel:
        ed=o.modifiers.new('Rounded edge','BEVEL')
        ed.width=bevel
        ed.segments=3
        mod=o.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
        mod.keep_sharp=True
    return o


def revolving(name,col,x,y,z,profile,mat):
    n=48
    verts=[(x+r*sin(i*2*pi/n),y-r*cos(i*2*pi/n),z+h) for r,h in profile for i in range(n)]
    faces=[(j*n+i,j*n+(i+1)%n,(j+1)*n+(i+1)%n,(j+1)*n+i) for j in range(len(profile)-1) for i in range(n)]
    faces.extend((tuple(reversed(range(n))),tuple(range(len(verts)-n,len(verts)))))
    return mesh_object(name,col,verts,faces,mat)


def area(name,col,coord,target,power,dimensions,color):
    d=bpy.data.lights.new(name,'AREA')
    d.energy=power
    d.shape='RECTANGLE'
    d.size,d.size_y=dimensions
    d.color=color
    o=bpy.data.objects.new(name,d)
    col.objects.link(o)
    o.location=coord
    aim(o,target)
    return o


def camera(name,col,coord,target,ortho=None):
    d=bpy.data.cameras.new(name)
    d.clip_start=.015
    d.clip_end=45
    c=bpy.data.objects.new(name,d)
    col.objects.link(c)
    c.location=coord
    aim(c,target)
    if ortho:
        d.type='ORTHO'
        d.ortho_scale=ortho
    return c


def grain_material():
    mat,nt,p=surface('Café | satin honey oak',(.57,.35,.19),.39,
                     **{'Coat Weight':.15,'Coat Roughness':.46})
    coords=node(nt,'ShaderNodeTexCoord','Wood coordinates',(-760,130))
    stretch=node(nt,'ShaderNodeVectorMath','Grain direction',(-575,150))
    stretch.operation='MULTIPLY'
    stretch.inputs[1].default_value=(12,220,290)
    nt.links.new(coords.outputs['Object'],stretch.inputs[0])
    grain=node(nt,'ShaderNodeTexNoise','Oak growth variations',(-405,150))
    grain.inputs['Scale'].default_value=1.
    grain.inputs['Detail'].default_value=3.2
    nt.links.new(stretch.outputs['Vector'],grain.inputs['Vector'])
    ramp=node(nt,'ShaderNodeValToRGB','Honey wood grain',(-212,200))
    ramp.color_ramp.elements[0].color=(*linear((.39,.225,.109)),1)
    ramp.color_ramp.elements[1].color=(*linear((.73,.517,.288)),1)
    nt.links.new(grain.outputs['Fac'],ramp.inputs[0])
    nt.links.new(ramp.outputs[0],p.inputs['Base Color'])
    return mat


def bent_tube(name,col,points,radii,mat,n=9):
    verts=[]
    for i,(p,r) in enumerate(zip(points,radii)):
        p=Vector(p)
        t=Vector(points[min(len(points)-1,i+1)])-Vector(points[max(0,i-1)])
        a=t.cross(Vector((.44,.72,.27))).normalized()
        b=t.normalized().cross(a).normalized()
        verts.extend(tuple(p+float(r)*(a*cos(j*2*pi/n)+b*sin(j*2*pi/n))) for j in range(n))
    faces=[(i*n+k,i*n+(k+1)%n,(i+1)*n+(k+1)%n,(i+1)*n+k) for i in range(len(points)-1) for k in range(n)]
    return mesh_object(name,col,verts,faces,mat)


def plant(name,col,origin,height,rng):
    clay,_,_=surface('Café | matte oat planter',(.68,.59,.475),.64)
    stem_mat,_,_=surface('Café | dry olive stems',(.29,.235,.12),.60)
    leaf_mat=[]
    for s in ((.283,.377,.152),(.37,.48,.185),(.25,.34,.12),(.44,.51,.22),(.335,.47,.178)):
        leaf_mat.append(surface('Café | leaf',s,.50,**{'Subsurface Weight':.10,'Subsurface Scale':.0018})[0])
    x,y,z=origin
    revolving(name+' • planter',col,x,y,z,[(.076,0),(.079,.007),(.098,.137),(.102,.146),(.102,.153),(.094,.155),(.090,.143),(.079,.086)],clay)
    leaf_v,leaf_f,mat_ids=[],[],[]
    for k in range(13):
        a=rng.uniform(0,2*pi)
        dx=rng.uniform(.090,.250)*cos(a)
        dy=rng.uniform(.045,.17)*sin(a)
        ht=rng.uniform(.48,1.0)*height
        base=Vector((x+dx*.19,y+dy*.19,z+.131))
        cap=base+Vector((dx,dy,ht))
        path=catmull([base,base+Vector((dx*.26,dy*.31,.26*ht)),base+Vector((dx*.72,dy*.73,.71*ht)),cap],24)
        bent_tube(name+' • stem',col,path,np.linspace(.0055,.0011,len(path)),stem_mat)
        for j in range(9,24):
            p=Vector(path[j])
            phi=rng.uniform(0,2*pi)
            forward=Vector((cos(phi),sin(phi),rng.uniform(-.35,.5))).normalized()
            sideways=forward.cross(Vector((.02,-.04,1.))).normalized()
            width=rng.uniform(.025,.055)
            length=rng.uniform(.061,.122)*(.68+.32*(24-j)/18)
            start=len(leaf_v)
            for v in np.linspace(0,1,9):
                for u in np.linspace(-1,1,5):
                    q=p+forward*length*v+sideways*u*width*sin(pi*v)**.8
                    q.z+=.009*sin(v*pi)-.012*u*u*sin(v*pi)
                    leaf_v.append(tuple(q))
            for f in range(8):
                for i in range(4):
                    a0=start+f*5+i
                    leaf_f.append((a0,a0+5,a0+6,a0+1))
                    mat_ids.append((k+j)%len(leaf_mat))
    foliage=mesh_object(name+' • curved leaves',col,leaf_v,leaf_f,leaf_mat[0])
    for m in leaf_mat[1:]:
        foliage.data.materials.append(m)
    for p,i in zip(foliage.data.polygons,mat_ids):
        p.material_index=i


def cafe(col):
    rng=Random(89)
    wood=grain_material()
    plaster,_,_=surface('Café | warm lime plaster',(.69,.58,.462),.83)
    taupe,_,_=surface('Café | slatted cabinet',(.425,.336,.254),.61)
    metal,_,_=surface('Café | dark bronze joinery',(.173,.181,.168),.39,**{'Metallic':.38})
    concrete,_,_=surface('Café | terrazzo floor',(.49,.43,.362),.74)
    sky,nts,glow=surface('Café | sunlit frosted window',(.86,.93,.99),.45)
    glow.inputs['Emission Color'].default_value=(.91,.958,1.,1.)
    camera_ray=node(nts,'ShaderNodeLightPath','Window brightness in the camera',(-475,-160))
    brightness=node(nts,'ShaderNodeMath','White sky exposure',(-269,-175))
    brightness.operation='MULTIPLY'
    brightness.inputs[1].default_value=2.05
    nts.links.new(camera_ray.outputs['Is Camera Ray'],brightness.inputs[0])
    nts.links.new(brightness.outputs[0],glow.inputs['Emission Strength'])
    pottery=[surface('Café | hand-glazed ceramic',c,.27,**{'Coat Weight':.25})[0] for c in
             [( .815,.76,.659),(.48,.48,.395),(.283,.313,.292),(.633,.526,.380),(.784,.702,.59)]]
    box('Limewashed wall',col,(.54,.885,1.38),(2.02,.095,2.89),plaster,.008)
    box('Window glass • diffused sky',col,(-.675,.820,1.49),(.94,.040,2.72),sky,.006)
    for x in (-1.15,-.840,-.520,-.249):
        box('Window vertical mullion',col,(x,.776,1.51),(.025,.049,2.63),metal,.003)
    for z in (.66,1.25,2.08):
        box('Window horizontal mullion',col,(-.70,.767,z),(.935,.041,.022),metal,.002)
    box('Terrazzo café floor',col,(0,.33,.57),(6.00,4.0,.092),concrete,.009)
    box('Counter carcass',col,(.494,.624,.938),(1.41,.370,.64),taupe,.008)
    for i in range(50):
        box('Fluted bar facing',col,(-.182+i*.027,.435,.948),(.021,.013,.60),taupe,.006)
    box('Bar counter • thick oak top',col,(.50,.585,1.263),(1.48,.45,.069),wood,.017)
    for z in (1.476,1.699,1.927):
        box('Oak wall shelf',col,(.560,.719,z),(1.20,.259,.041),wood,.008)
        for j in range(rng.randint(9,14)):
            x=-.005+j*.080+rng.uniform(-.019,.022)
            y=.671+rng.uniform(-.039,.048)
            h=rng.uniform(.048,.137)
            r=rng.uniform(.022,.033)
            revolving('Shelf ware • ceramic jars and cups',col,x,y,z+.020,
                      [(r*.77,0),(r*.80,.009),(r,h*.79),(r*.94,h),(r*.79,h+.001),(r*.70,h*.83),(r*.63,.015)],
                      rng.choice(pottery))
    for j in range(6):
        x=.22+j*.092
        h=rng.uniform(.079,.191)
        r=rng.uniform(.024,.038)
        revolving('Café counter vessels',col,x,.568,1.297,
                  [(r*.8,0),(r,.009),(r,h*.82),(r*.55,h*.94),(r*.48,h),(r*.41,h),(r*.41,h*.81)],
                  rng.choice(pottery))
    for x,y in ((-.230,.108),(.433,.123)):
        for sign in (-1,1):
            o=box('Chair • tapered side rail',col,(x+sign*.158,y,.956),(.043,.054,.63),wood,.016)
            o.rotation_euler[0]=-.115
        o=box('Chair • curved top rail',col,(x,y-.030,1.264),(.358,.047,.071),wood,.019)
        o.rotation_euler[1]=.055 if x<0 else -.04
        box('Chair • second slat',col,(x,y-.022,1.147),(.313,.031,.023),wood,.012)
    box('Rear café table',col,(.492,.214,1.115),(.69,.55,.045),wood,.021)
    box('Near table edge',col,(.049,-.440,1.008),(.999,.556,.037),wood,.022)
    plant('Window fig',col,(-.395,.340,.759),1.24,rng)
    plant('Small counter greenery',col,(.540,.319,1.134),.30,rng)
    area('Large window • warm daylight',col,(-.430,-.445,1.900),(0,-.090,1.441),29,(.49,.63),(1.,.931,.841))
    area('Room bounce • soft fill',col,(.318,-.429,1.597),(0,-.105,1.475),8.5,(.45,.42),(.848,.909,1.))
    area('Back window • hair rim',col,(-.331,.237,1.867),(0,-.051,1.507),6.5,(.385,.52),(1.,.843,.685))
    area('Shelf light',col,(.492,.174,1.874),(.473,.694,1.431),32,(.55,.38),(1.,.835,.682))
    area('Soft white catchlight',col,(.069,-.613,1.714),(0,-.135,1.553),2.1,(.179,.136),(1.,.959,.909))


def build_staging(scene):
    cam_col=collection('Cameras')
    beauty=camera('Beauty • café selfie',cam_col,BEAUTY_LOC,BEAUTY_TARGET)
    beauty.data.sensor_fit='HORIZONTAL'
    beauty.data.lens=48.0
    beauty.data.dof.use_dof=True
    beauty.data.dof.focus_distance=.548
    beauty.data.dof.aperture_fstop=4.30
    beauty.data.dof.aperture_blades=8
    scene.camera=beauty
    for label,deg in ORBITS.items():
        phi=deg*pi/180
        center=np.asarray(ORBIT_CENTER)
        coord=center+np.array((1.51*sin(phi),-1.51*cos(phi),.022))
        camera('diagnostic • '+label,cam_col,coord,center,ortho=.929)
    world=bpy.data.worlds.new('Café daylight')
    world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.52,.47,.42,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.25
    scene.world=world
    cafe(collection('Café set and beauty lighting'))
    studio=collection('Neutral orbit lighting')
    shades=[
        ('Studio • overhead',(-.31,-.28,1.960),(0,-.21,1.42),34,(.59,.61),(1.,.969,.926)),
        ('Studio • fill',(.410,-.432,1.611),(0,-.150,1.435),14,(.46,.63),(.928,.958,1.)),
        ('Studio • rear',(.323,.288,1.811),(0,-.095,1.42),20,(.63,.66),(1.,.966,.941)),
        ('Studio • bounce',(-.53,.21,1.60),(0,-.083,1.35),15,(.53,.58),(1.,.98,.96)),
    ]
    for args in shades:
        area(args[0],studio,*args[1:])
    studio.hide_render=True
    studio.hide_viewport=True
