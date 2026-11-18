"""Connected jersey sweater, knitted yarn relief, ribbed trims and hands."""
from math import atan2, cos, pi, sin

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from portrait_common import (
    ROOT, catmull, ease, gauss, linear, mesh_object, mix, node, noise_bump,
    read_obj, subdivide, subset, surface,
)
from portrait_face import skin_material


def jersey_tile():
    """Rasterize a periodic height tile from the paths of interlocking yarns."""
    size, cols, rows = 384, 3, 4
    height = np.zeros((size,size), np.float32)
    path = catmull([
        (-.093,-.15,.18),(.05,.02,.23),(.235,.215,.44),(.342,.417,.70),
        (.234,.715,.86),(.29,.955,.80),(.50,1.079,.59),(.71,.955,.80),
        (.766,.715,.86),(.658,.417,.70),(.765,.215,.44),(.95,.02,.23),
        (1.093,-.15,.18),
    ],144)
    for j in range(rows):
        for i in range(cols):
            skew=.032*sin((i*3+j*7)*1.49)
            fiber=.126+.009*sin(i*7+j*17)
            for x,y,h in path:
                x=(x+i+skew*(y-.4))*size/cols
                y=(y+j)*size/rows
                ax=fiber*size/cols
                ay=fiber*1.09*size/rows
                u=np.arange(int(x-ax)-1,int(x+ax)+2)
                v=np.arange(int(y-ay)-1,int(y+ay)+2)
                solid=1-((u[None,:]-x)/ax)**2-((v[:,None]-y)/ay)**2
                shape=.22+.39*np.sqrt(np.maximum(0,solid))+.30*h
                shape=np.where(solid>0,shape*np.clip(solid*8,0,1),0)
                pixels = np.ix_(v%size,u%size)
                height[pixels]=np.maximum(height[pixels],shape)
    img=bpy.data.images.new('Jersey knit • interlocked yarn height',size,size,alpha=True)
    img.colorspace_settings.name='Non-Color'
    rgba=np.dstack([height,height,height,np.ones_like(height)])
    img.pixels.foreach_set(rgba.reshape(-1))
    img.pack()
    return img


def knit_material(name, tile, swatch, rib=False):
    mat,nt,p=surface(name,swatch,.70,**{
        'Sheen Weight':.38, 'Sheen Roughness':.76,
        'Sheen Tint':(*linear((.96,.70,.68)),1),
        'Specular IOR Level':.24,
        'Subsurface Weight':.025,'Subsurface Scale':.002,
    })
    uv=node(nt,'ShaderNodeTexCoord','Wales and courses',(-900,110))
    img=node(nt,'ShaderNodeTexImage','Knitted loop relief',(-705,155))
    img.image=tile
    img.extension='REPEAT'
    nt.links.new(uv.outputs['UV'],img.inputs['Vector'])
    ramp=node(nt,'ShaderNodeValToRGB','Yarn strand and recess colors',(-403,220))
    ramp.color_ramp.elements[0].position=.0
    ramp.color_ramp.elements[0].color=(*linear(tuple(c*.77 for c in swatch)),1)
    ramp.color_ramp.elements[1].position=.93
    ramp.color_ramp.elements[1].color=(*linear(tuple(min(1,c*1.08) for c in swatch)),1)
    ramp.color_ramp.elements.new(.42).color=(*linear(tuple(c*.976 for c in swatch)),1)
    nt.links.new(img.outputs['Color'],ramp.inputs['Fac'])
    flecks=node(nt,'ShaderNodeTexNoise','Small variation between fibers',(-710,-95))
    flecks.inputs['Scale'].default_value=710
    flecks.inputs['Roughness'].default_value=.63
    nt.links.new(uv.outputs['Object'],flecks.inputs['Vector'])
    mottle=node(nt,'ShaderNodeValToRGB','Mélange',(-450,-95))
    mottle.color_ramp.elements[0].color=(.58,.44,.46,1)
    mottle.color_ramp.elements[1].color=(1.21,1.13,1.12,1)
    nt.links.new(flecks.outputs['Fac'],mottle.inputs['Fac'])
    mixcolor=node(nt,'ShaderNodeMixRGB','Dyed pink wool',(-150,180))
    mixcolor.blend_type='MULTIPLY'
    mixcolor.inputs[0].default_value=.36
    nt.links.new(ramp.outputs['Color'],mixcolor.inputs[1])
    nt.links.new(mottle.outputs[0],mixcolor.inputs[2])
    nt.links.new(mixcolor.outputs['Color'],p.inputs['Base Color'])
    textile=noise_bump(nt,p,5300,.29,.00012,uv.outputs['Object'],'Twisted yarn fibers')
    relief=node(nt,'ShaderNodeBump','Knit height',(-41,-126))
    relief.inputs['Strength'].default_value=.415 if not rib else .30
    relief.inputs['Distance'].default_value=.00110 if not rib else .00092
    nt.links.new(img.outputs['Color'],relief.inputs['Height'])
    nt.links.new(textile.outputs['Normal'],relief.inputs['Normal'])
    nt.links.new(relief.outputs['Normal'],p.inputs['Normal'])
    return mat


def add_uv(obj, face_uvs):
    uv=obj.data.uv_layers.new(name="Knit coordinates")
    uv.uv.foreach_set('vector',np.asarray([co for poly in face_uvs for co in poly],dtype=np.float32).reshape(-1))


def rings_object(name,col,rings,material,u_scale,v_positions,closed=False):
    v=[tuple(co) for ring in rings for co in ring]
    n=len(rings[0])
    faces,uvs=[],[]
    for j in range(len(rings)-1+int(closed)):
        row=(j+1)%len(rings)
        for i in range(n):
            b=(i+1)%n
            faces.append((j*n+i,row*n+i,row*n+b,j*n+b))
            u=i*u_scale/n
            w=(i+1)*u_scale/n
            tv=v_positions[j]
            bv=v_positions[row] if row else tv+.18
            uvs.append(((u,tv),(u,bv),(w,bv),(w,tv)))
    o=mesh_object(name,col,v,faces,material)
    add_uv(o,uvs)
    return o


# Rows from the inner neck edge to the lower torso.
TORSO = [
    (0,1.373,.067,-.019,.070,.060),
    (3,1.369,.100,-.015,.085,.070),
    (8,1.344,.178,-.010,.096,.084),
    (13,1.310,.186,-.013,.107,.088),
    (19,1.247,.182,-.013,.121,.091),
    (24,1.197,.171,-.012,.108,.087),
    (31,1.111,.158,-.010,.092,.081),
    (40,1.014,.162,-.007,.091,.081),
]


def torso_point(row, theta):
    j=np.array([r[0] for r in TORSO])
    fit=np.array(TORSO)[:,1:]
    samples=((-.0,.28),(-.92,.22),(.92,.22),(-1.84,.105),(1.84,.105),(-2.76,.035),(2.76,.035))
    z,rx,cy,front,back=(
        sum(w*np.interp(row+x,j,fit[:,k]) for x,w in samples) for k in range(5)
    )
    z+=.028*(1-ease(0,27,row))
    c=cos(theta)
    s=sin(theta)
    y=cy-c*(front if c>=0 else back)
    x=s*rx
    z -= .024*max(0,c)**1.75*(1-ease(0,10,row))
    z += .009*max(0,-c)*(1-ease(0,10,row))
    z += .008*sin(theta+.3)*gauss(row,11,8)
    knit_folds=(.0037*sin(6*theta+.15*row) + .0045*sin(12*theta-.11*row))*ease(11,37,row)
    knit_folds+=.0047*gauss(row,23,5)*gauss(abs(s),.84,.19)*sin(row*.95+2*theta)
    knit_folds+=.0015*sin(4*theta)**2*gauss(row,7,7)
    # Shallow tension folds run out of the lower sleeve seams into the chest.
    for side,offset in ((.80,-.25),(-.85,.36)):
        phi=(theta+pi)%(2*pi)-pi
        diagonal=(row-19.5)+9.2*(phi-side)+offset
        knit_folds+=.0024*sin(diagonal*1.34)*gauss(row,19.7,9.7)*gauss(phi,side,.47)
    knit_folds+=.0024*sin(17*theta+.12*row)*gauss(row,39,5)
    return Vector((x+.004*sin(row*.15)*ease(8,32,row)+s*knit_folds,y-c*knit_folds,z))


SLEEVE_KNOTS = {
    -1:[(-.170,-.034,1.305),(-.181,-.094,1.299),(-.191,-.181,1.273),
        (-.188,-.273,1.255),(-.161,-.353,1.262),(-.131,-.432,1.290),
        (-.099,-.528,1.323),(-.083,-.575,1.342)],
     1:[(.172,-.033,1.290),(.192,-.089,1.278),(.211,-.186,1.252),
        (.211,-.291,1.233),(.186,-.367,1.247),(.155,-.451,1.267),
        (.126,-.541,1.293),(.107,-.590,1.308)],
}


def local_frame(direction):
    up=Vector((0,0,1))
    side=direction.cross(up).normalized()
    top=side.cross(direction).normalized()
    return side,top


def add_sleeve(verts,faces,uvs,rim,sign,arm_widths,mat_col,trim):
    """Connect sleeve quads to their actual armhole edges."""
    knots=[(x,y,z+.023) for x,y,z in SLEEVE_KNOTS[sign]]
    centers=[Vector(p) for p in catmull(knots,59)]
    directions=[(centers[min(58,j+1)]-centers[max(0,j-1)]).normalized() for j in range(59)]
    # Follow the boundary orientation and distribute the samples around a plane.
    mean=sum((Vector(verts[i]) for i in rim),Vector())/len(rim)
    frame=local_frame(directions[0])
    angles=[]
    for i in rim:
        rel=Vector(verts[i])-mean
        angles.append(atan2(rel.dot(frame[1]),rel.dot(frame[0])))
    # A uniformly sampled tube meets the armhole through transitional quads.
    turns=np.unwrap(angles)
    uniform=np.linspace(turns[0],turns[0]+np.sign(turns[-1]-turns[0])*2*pi,len(rim),endpoint=False)
    angles=uniform
    n=len(rim)
    end_width=.0304
    rings=[]
    arclength=0.
    prev=rim
    prev_v=0.
    for j in range(51):
        t=j/58
        center=centers[j]
        normal,up=local_frame(directions[j])
        radius=np.interp(t,[0,.16,.31,.51,.68,.87,1],arm_widths)
        current=[]
        if j:
            arclength+=(centers[j]-centers[j-1]).length
        cur_v=(arclength+.012)/.0122
        for k,a in enumerate(angles):
            armpit=(.0011+.0032*gauss(t,.51,.22))*sin(16*pi*t+2.1*cos(a))
            fibers=.0024*sin(6*a+t*9)*ease(.05,.37,t)
            forearm=.0026*sin(25*pi*t+.6*cos(3*a))*gauss(t,.79,.215)
            fold=.0051*gauss(t,.47,.14)*gauss(cos(a),-.55,.57)*sin(t*88+2*cos(a))
            offset=radius+armpit+fibers+forearm+fold
            p=center+normal*(cos(a)*offset)+up*(sin(a)*(offset*.92))
            if t<.18:
                rim_vec=Vector(verts[rim[k]])
                p=rim_vec.lerp(p,.29+.71*ease(0,.18,t))
            current.append(len(verts))
            verts.append(tuple(p))
        for k in range(n):
            h=(k+1)%n
            u=k/n*42
            w=(k+1)/n*42
            faces.append((prev[k],prev[h],current[h],current[k]))
            uvs.append(((u,prev_v),(w,prev_v),(w,cur_v),(u,cur_v)))
        prev=current
        prev_v=cur_v
    # Rolled rib knit cuffs. Return the wrist pose for the inferred hands.
    n=192
    cuff=[]
    for j,t in enumerate(np.linspace(0,1,22)):
        f=50+8*t
        k=min(57,int(f))
        center=centers[k].lerp(centers[k+1],f-k)
        normal,up=local_frame(directions[k])
        radius=mix(.0474,end_width,t)+.0019*sin(pi*t)**2
        ring=[]
        for i in range(n):
            a=2*pi*i/n
            r=radius+.0011*cos(a*64)*(.7+.3*sin(pi*t))
            ring.append(center+normal*(cos(a)*r)+up*(sin(a)*r*.92))
        cuff.append(ring)
    for extra,depth in ((1,.0022),(.91,.0043),(.24,.0064),(0,.006)):
        f=50+8*extra
        k=min(57,int(f))
        center=centers[k].lerp(centers[k+1],f-k)
        normal,up=local_frame(directions[k])
        radius=mix(.0474,end_width,extra)-depth
        cuff.append([center+radius*(normal*cos(2*pi*i/n)+up*.92*sin(2*pi*i/n)) for i in range(n)])
    label='left' if sign<0 else 'right'
    obj=rings_object("Ribbed cuff • image "+label,mat_col,cuff,trim,64,[j*.178 for j in range(len(cuff))],closed=True)
    subdivide(obj,1,'Soft rib edges')
    return centers[-1],directions[-1]


def hands(col,env,poses):
    vertices,faces=read_obj(ROOT/'input/base.obj',lambda p:(p[0]*.1,-p[2]*.1,(p[1]+8.168)*.1),'body')
    mat=skin_material(colored=False)
    landmarks=env['landmarks']
    for sign,(wrist,forward) in poses.items():
        label='l' if sign>0 else 'r'
        s=Vector(landmarks['joint-'+label+'-hand'])
        mid=Vector(landmarks['joint-'+label+'-finger-3-3'])
        across=(Vector(landmarks['joint-'+label+'-finger-5-1'])-Vector(landmarks['joint-'+label+'-finger-1-1'])).normalized()
        along=(mid-s).normalized()
        across=(across-along*across.dot(along)).normalized()
        up=along.cross(across).normalized()
        fs=[f for f in faces if all(sign*vertices[i][0]>.350 and (Vector(vertices[i])-s).dot(along)>-.024 for i in f)]
        points,fs=subset(vertices,fs)
        side,normal=local_frame(forward)
        spread=(side*.48+normal*.877)*(-sign)
        upper=forward.cross(spread).normalized()
        # Preserve handedness and the knuckles, with a slight flex at the fingertips.
        transformed=[]
        for p in points:
            d=Vector(p)-s
            l=d.dot(along)
            across_pos=d.dot(across)
            recess=d.dot(up)
            flex=ease(.021,.139,l)
            curl=(.80+.20*gauss(across_pos,.005,.035))*flex**1.65
            q=(wrist-forward*.016+forward*(l*.78-.032*curl)
               +spread*(across_pos*(.75-.20*flex))
               +upper*(recess*.80-.047*curl))
            transformed.append(q)
        obj=mesh_object("Relaxed hand • image "+('right' if sign>0 else 'left'),col,transformed,fs,mat)
        subdivide(obj,2,'Knuckles and fingers')


def fitted_neckline(head):
    """Fit the inner knit edge against horizontal sections of the posed neck."""
    bpy.context.view_layer.update()
    tree=BVHTree.FromObject(head,bpy.context.evaluated_depsgraph_get())
    steps=360
    offsets=[]
    for a in np.linspace(0,2*pi,steps,endpoint=False):
        base=torso_point(0,a)
        radial=Vector((sin(a),-cos(a),0))
        axis=Vector((-.003,-.019,base.z+.005))
        start=axis+radial*.156
        hit=tree.ray_cast(start,-radial,.160)[0]
        if hit:
            distance=(hit-base).dot(radial)
            offsets.append(min(.007,max(-.022,distance+.0004)))
        else:
            offsets.append(-.012)
    weights=np.array((.13,.22,.30,.22,.13))
    offsets=sum(w*np.roll(offsets,k) for k,w in zip((-2,-1,0,1,2),weights))

    def point(row,a):
        base=torso_point(row,a)
        shift=np.interp(a%(2*pi),np.linspace(0,2*pi,steps+1),np.r_[offsets,offsets[0]])
        return base+Vector((sin(a),-cos(a),0))*shift*(1-ease(0,4.7,row))
    return point


def build_wardrobe(col,env,head):
    tile=jersey_tile()
    yarn=knit_material("Dusty rose • plain stockinette",tile,(.755,.438,.469))
    trim=knit_material("Dusty rose • rib knit",tile,(.751,.440,.459),rib=True)
    inside,_,_=surface("Garment lining • muted rose",(.405,.212,.236),.84)
    size=128
    tailored=fitted_neckline(head)
    vertices=[tuple(tailored(j,2*pi*i/size)) for j in range(41) for i in range(size)]
    faces,uvs=[],[]
    holes=[(19,45,7,23,1),(83,109,7,23,-1)]
    for j in range(40):
        for i in range(size):
            if any(a<=i<b and c<=j<d for a,b,c,d,_ in holes):
                continue
            k=(i+1)%size
            faces.append((j*size+i,(j+1)*size+i,(j+1)*size+k,j*size+k))
            # The neck rings compress the wales continuously into the collar.
            u=i/size*67
            w=(i+1)/size*67
            a=j*.815
            b=(j+1)*.815
            uvs.append(((u,a),(u,b),(w,b),(w,a)))
    poses={}
    for a,b,c,d,sign in holes:
        rim=[c*size+i for i in range(a,b)]+[j*size+b for j in range(c,d)]
        rim += [d*size+i for i in range(b,a,-1)]+[j*size+a for j in range(d,c,-1)]
        poses[sign]=add_sleeve(vertices,faces,uvs,rim,sign,[.074,.070,.065,.060,.057,.047,.041],col,trim)
    vertices,faces=subset(vertices,faces)
    garment=mesh_object("Sweater • torso and continuous set-in sleeves",col,vertices,faces,yarn)
    add_uv(garment,uvs)
    bpy.context.view_layer.objects.active=garment
    garment.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    garment.select_set(False)
    subdivide(garment,1,'Draped cloth')
    garment.data.materials.append(inside)
    solid=garment.modifiers.new('Soft wool thickness','SOLIDIFY')
    solid.thickness=.0028
    solid.offset=-1.
    solid.use_rim_only=True
    solid.material_offset=1
    # Neckband curves up into the back of a classic crew-neck opening.
    collar=[]
    n=240
    for j in range(15):
        s=j/14
        strip=[]
        for i in range(n):
            a=2*pi*i/n
            p=tailored(3.05*(1-s),a)
            ripple=.00095*cos(a*80)*(.72+.28*sin(pi*s))
            p+=Vector((sin(a)*.85,-cos(a)*.90,.78))*(ripple+.00145+.0015*sin(pi*s))
            p.z+=.006*s
            strip.append(p)
        collar.append(strip)
    for blend,h in ((1,.001),(.91,-.002),(.50,-.003),(0,-.002)):
        collar.append([tailored(3.05*(1-blend),2*pi*i/n)+Vector((0,0,h+.004*blend)) for i in range(n)])
    neckband=rings_object("Crew neck • raised ribbed binding",col,collar,trim,68,[j*.21 for j in range(len(collar))],True)
    subdivide(neckband,1,'Round the knitted binding')
    hem=[]
    for j in range(16):
        s=j/15
        strip=[]
        for i in range(256):
            a=2*pi*i/256
            p=torso_point(38.0+2*s,a)
            r=.0016+.0010*cos(a*128)+.001*sin(pi*s)
            p+=Vector((sin(a)*r,-cos(a)*r,-.012*s))
            strip.append(p)
        hem.append(strip)
    for t,h in ((1,-.012),(.95,-.009),(.2,-.003),(0,0)):
        hem.append([torso_point(38.0+2*t,2*pi*i/256)+Vector((-.004*sin(2*pi*i/256),.004*cos(2*pi*i/256),h)) for i in range(256)])
    hem_obj=rings_object("Sweater hem • ribbed welt",col,hem,trim,91,[j*.148 for j in range(len(hem))],True)
    subdivide(hem_obj,1,'Round the knitted welt')
    # Closed interior at the presentation cut, entirely inside the welt.
    plug=[torso_point(40,2*pi*i/128)+Vector((-.003*sin(2*pi*i/128),.003*cos(2*pi*i/128),-.006)) for i in range(128)]
    mesh_object("Inward-facing sweater hem finish",col,plug,[tuple(reversed(range(128)))],inside,smooth=False)
    hands(col,env,poses)
    return garment
