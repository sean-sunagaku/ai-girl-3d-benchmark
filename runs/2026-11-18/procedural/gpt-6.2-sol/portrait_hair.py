"""An asymmetric, strand-based chestnut groom."""
from math import cos, pi, sin
from random import Random

import numpy as np
from mathutils import Vector

from portrait_common import catmull, ease, gauss, linear, mesh_object, native_strands, node, surface
from portrait_face import pose_array, visible_surface

F_MIN, F_MAX = .455, 2.995


def fiber_shader(name, color, roughness):
    mat,nt,p=surface(name,color,.40)
    p.location=(-145,-173)
    p.inputs['IOR'].default_value=1.47
    p.inputs['Specular IOR Level'].default_value=.26
    p.inputs['Roughness'].default_value=.50
    hair=node(nt,'ShaderNodeBsdfHairPrincipled','Pigment scattering along hair',(-145,80))
    hair.parametrization='COLOR'
    guide=node(nt,'ShaderNodeHairInfo','Strand variation',(-680,140))
    ramp=node(nt,'ShaderNodeValToRGB','Chestnut to warm umber',(-443,165))
    tone=ramp.color_ramp
    dark,light=tone.elements[0],tone.elements[1]
    for entry,pigment,position in (
        (dark,(.143,.064,.040),.0),
        (tone.elements.new(.41),color,.41),
        (tone.elements.new(.76),tuple(c*1.27 for c in color),.76),
        (light,tuple(c*1.65 for c in color),1.),
    ):
        entry.position=position
        entry.color=(*linear(pigment),1.)
    nt.links.new(guide.outputs['Random'],ramp.inputs[0])
    nt.links.new(ramp.outputs['Color'],hair.inputs['Color'])
    hair.inputs['Roughness'].default_value=roughness
    hair.inputs['Radial Roughness'].default_value=.386
    hair.inputs['Coat'].default_value=.07
    hair.inputs['Random Roughness'].default_value=.19
    nt.links.new(ramp.outputs['Color'],p.inputs['Base Color'])
    mixed=node(nt,'ShaderNodeMixShader','Directional sheen and opaque fiber core',(121,56))
    mixed.inputs[0].default_value=.32
    nt.links.new(p.outputs[0],mixed.inputs[1])
    nt.links.new(hair.outputs[0],mixed.inputs[2])
    out=nt.nodes.get('Material Output')
    out.location=(350,62)
    nt.links.new(mixed.outputs[0],out.inputs['Surface'])
    return mat


def shell_direction(phi,f,sign):
    # The part follows a gently irregular near-sagittal line.
    part=-.321-.051*sin(phi*2.45)+.014*sin(phi*9.7)
    width=1.81-.44*gauss(phi,.47,.52)+.88*ease(1.94,2.90,phi)
    if sign>0:
        width-=.41*gauss(phi,1.10,.46)
    theta=part+sign*(width-sign*part)*f
    swing=(.32 if sign<0 else .66)*gauss(phi,.60,.64)
    a=phi+swing*f**1.5+.041*sin(f*pi)*sin(2.5*phi)
    v=Vector((.079*sin(theta)*sin(a), -.113*cos(a), .118*cos(theta)*sin(a)))
    return v.normalized()


def scalp_sheet(col,skin):
    cap,nt,p=surface('Scalp underlayer | deep chestnut',(.161,.073,.041),.44,**{
        'Coat Weight':.115,'Coat Roughness':.48,'Specular IOR Level':.28,
    })
    uv=node(nt,'ShaderNodeTexCoord','Crown flow UV',(-715,180))
    sep=node(nt,'ShaderNodeSeparateXYZ','Across the locks',(-550,150))
    nt.links.new(uv.outputs['UV'],sep.inputs[0])
    wave=node(nt,'ShaderNodeMath','Fine roots',(-400,145))
    wave.operation='MULTIPLY'
    wave.inputs[1].default_value=4171
    nt.links.new(sep.outputs['X'],wave.inputs[0])
    sine=node(nt,'ShaderNodeMath','Parallel fiber grooves',(-260,113))
    sine.operation='SINE'
    nt.links.new(wave.outputs[0],sine.inputs[0])
    bump=node(nt,'ShaderNodeBump','Crown grooves',(-120,-70))
    bump.inputs['Strength'].default_value=.42
    bump.inputs['Distance'].default_value=.000075
    nt.links.new(sine.outputs[0],bump.inputs['Height'])
    nt.links.new(bump.outputs[0],p.inputs['Normal'])
    grids={}
    p_count, t_count = 129, 51
    center=Vector((0.,-.048,1.551))
    for sign,label in ((-1,'left sweep'),(1,'right sweep')):
        points, normals = [], []
        for phi in np.linspace(F_MIN,F_MAX,p_count):
            for t in np.linspace(0,1,t_count):
                t=.0022+.9978*t
                direction=shell_direction(phi,t,sign)
                hit=skin.ray_cast(center+direction*.26,-direction,.30)[0]
                if hit is None:
                    hit=center+direction*.10
                loft=.0015+.0065*sin(pi*t*.89)**.86
                loft+=.0041*gauss(phi,.78,.57)*sin(pi*t)**1.3
                loft+=.0029*gauss(phi,1.32,1.10)*sin(pi*t)**.75
                loft*= .64+.36*ease(F_MIN,F_MIN+.16,phi)
                loft += .00028*sin(phi*42.2+sin(t*5)*.9)*sin(pi*t)**.62
                points.append(hit+direction*loft)
                normals.append(direction)
        grid=np.asarray(points,dtype=np.float64).reshape(p_count,t_count,3)
        for _ in range(3):
            grid[1:-1,1:-1]=(.47*grid[1:-1,1:-1]+.1325*(
                grid[:-2,1:-1]+grid[2:,1:-1]+grid[1:-1,:-2]+grid[1:-1,2:]))
        raw=grid.reshape(-1,3)
        faces=[]
        uv_data=[]
        for j in range(p_count-1):
            for i in range(t_count-1):
                a=j*t_count+i
                cell=(a,a+1,a+t_count+1,a+t_count)
                if sign<0:
                    cell=tuple(reversed(cell))
                faces.append(cell)
                uv_data.extend((v//t_count/(p_count-1),v%t_count/(t_count-1)) for v in cell)
        o=mesh_object('Scalp underlayer • '+label,col,pose_array(raw),faces,cap)
        tex=o.data.uv_layers.new(name='Direction of growth')
        tex.uv.foreach_set('vector',np.asarray(uv_data,dtype=np.float32).reshape(-1))
        grids[sign]=(raw.reshape(p_count,t_count,3),np.asarray(normals).reshape(p_count,t_count,3))
    return grids


def sample_grid(grid,phi,t):
    x,y,_=grid.shape
    a=np.clip((np.asarray(phi)-F_MIN)/(F_MAX-F_MIN)*(x-1),0,x-1-1e-7)
    b=np.clip(np.asarray(t)*(y-1),0,y-1-1e-7)
    i,j=a.astype(int),b.astype(int)
    u,v=(a-i)[...,None],(b-j)[...,None]
    return grid[i,j]*(1-u)*(1-v)+grid[i+1,j]*u*(1-v)+grid[i,j+1]*(1-u)*v+grid[i+1,j+1]*u*v


def radius_profile(start,n):
    t=np.linspace(0,1,n)
    return start*(.94+.12*np.sin(pi*t))*np.maximum(.065,(1-t)**.32)*np.clip(9*(1-t),.08,1)


def varied_pose(p):
    w=np.clip((p[:,2]-1.385)/(1.575-1.385),0,1)
    w=w*w*(3-2*w)
    return pose_array(p,w[:,None])


def crown_hairs(col,grids,mat,rng):
    paths,radii=[],[]
    s=np.linspace(0,1,22)
    for sign in (-1,1):
        points,normals=grids[sign]
        for i in range(2880):
            phi=rng.uniform(F_MIN+.003,F_MAX-.005)
            start=rng.random()**2*.67
            end=rng.uniform(.892,1.)
            t=start+(end-start)*s
            phase=rng.uniform(0,2*pi)
            offset=rng.uniform(.00012,.00285)
            p=phi+(.022*np.sin(s*2.9+phase)+.009*np.sin(s*9+phase))*np.sin(pi*s)**.7
            n=sample_grid(normals,p,t)
            track=sample_grid(points,p,t)+n*offset*(.18+.82*np.sin(pi*s*.68)**.7)[:,None]
            paths.append(pose_array(track))
            radii.append(radius_profile(rng.uniform(.000038,.000075),len(track)))
    native_strands('Crown • combed individual hairs',col,paths,radii,mat)


def front_locks():
    # Locks across the far shoulder are deliberately shorter than the dense
    # sweep that falls over the near side of the chest.
    return [
        (-1,.49,.019,.009,425,[(-.066,-.140,1.571),(-.082,-.141,1.498),(-.123,-.106,1.435),(-.150,-.153,1.352),(-.137,-.176,1.282),(-.109,-.181,1.212),(-.110,-.139,1.120)]),
        (-1,.63,.037,.012,455,[(-.077,-.139,1.548),(-.094,-.135,1.485),(-.124,-.101,1.430),(-.142,-.151,1.356),(-.114,-.173,1.281),(-.088,-.161,1.199),(-.088,-.132,1.136),(-.061,-.147,1.099)]),
        (-1,.78,.046,.014,460,[(-.081,-.122,1.540),(-.108,-.117,1.478),(-.144,-.093,1.407),(-.128,-.143,1.342),(-.101,-.169,1.268),(-.089,-.152,1.187),(-.079,-.151,1.144)]),
        (-1,.94,.053,.013,490,[(-.094,-.110,1.520),(-.109,-.105,1.465),(-.151,-.098,1.407),(-.142,-.137,1.340),(-.110,-.164,1.251),(-.119,-.151,1.190),(-.082,-.141,1.079)]),
        (-1,1.13,.069,.016,530,[(-.095,-.079,1.519),(-.112,-.096,1.463),(-.157,-.095,1.401),(-.145,-.122,1.332),(-.126,-.166,1.263),(-.131,-.146,1.201),(-.115,-.143,1.095)]),
        (-1,1.28,.079,.016,490,[(-.097,-.053,1.521),(-.119,-.060,1.457),(-.165,-.108,1.397),(-.148,-.143,1.350),(-.130,-.152,1.273),(-.155,-.140,1.185),(-.129,-.134,1.120)]),
        (-1,1.40,.053,.014,390,[(-.100,-.050,1.517),(-.114,-.054,1.455),(-.150,-.086,1.394),(-.154,-.136,1.347),(-.132,-.145,1.298),(-.099,-.129,1.229),(-.113,-.106,1.157)]),
        (-1,1.54,.071,.013,350,[(-.098,-.030,1.517),(-.124,-.027,1.458),(-.161,-.065,1.404),(-.174,-.106,1.365),(-.125,-.134,1.308),(-.120,-.134,1.232),(-.077,-.155,1.164)]),
        (-1,.59,.039,.014,560,[(-.068,-.147,1.565),(-.081,-.140,1.506),(-.107,-.117,1.448),(-.132,-.119,1.382),(-.105,-.165,1.312),(-.065,-.173,1.246),(-.070,-.143,1.176),(-.051,-.137,1.120)]),
        (-1,.70,.046,.016,515,[(-.073,-.142,1.554),(-.090,-.136,1.488),(-.110,-.114,1.428),(-.117,-.144,1.368),(-.077,-.174,1.300),(-.068,-.159,1.224),(-.046,-.132,1.158)]),
        (-1,.82,.037,.015,480,[(-.091,-.120,1.538),(-.095,-.119,1.478),(-.112,-.122,1.423),(-.107,-.148,1.368),(-.063,-.162,1.308),(-.045,-.162,1.241),(-.057,-.148,1.187)]),
        (1,.49,.018,.009,320,[(.078,-.017,1.577),(.092,.010,1.514),(.105,-.025,1.440),(.152,-.063,1.388),(.166,-.085,1.359),(.150,-.081,1.331),(.123,-.104,1.316)]),
        (1,.70,.047,.012,431,[(.087,-.014,1.569),(.097,.022,1.497),(.127,-.024,1.418),(.166,-.064,1.367),(.172,-.066,1.340),(.160,-.093,1.308),(.105,-.125,1.296)]),
        (1,.90,.054,.011,425,[(.091,.014,1.552),(.108,.030,1.471),(.147,-.011,1.413),(.174,-.017,1.372),(.193,-.025,1.344),(.202,-.025,1.320),(.187,-.037,1.299)]),
        (1,1.10,.062,.012,355,[(.089,.021,1.509),(.104,.025,1.448),(.146,.031,1.403),(.176,.029,1.368),(.184,.042,1.328),(.175,.045,1.291)]),
    ]


def back_drape(col,tree,mat,rng):
    """A continuous lower volume makes the strands lie over the nape and back."""
    base,nt,p=surface('Hair mass | dark chestnut',(.177,.087,.055),.455,
                       **{'Specular IOR Level':.26,'Coat Weight':.12,'Coat Roughness':.40})
    tex=node(nt,'ShaderNodeTexCoord','Downward comb directions',(-770,140))
    sep=node(nt,'ShaderNodeSeparateXYZ','Growth flow',(-570,140))
    nt.links.new(tex.outputs['UV'],sep.inputs[0])
    ridges=node(nt,'ShaderNodeMath','Fiber bundles',(-422,165))
    ridges.operation='MULTIPLY'
    ridges.inputs[1].default_value=2770
    nt.links.new(sep.outputs[0],ridges.inputs[0])
    wave=node(nt,'ShaderNodeMath','Linear microtexture',(-265,115))
    wave.operation='SINE'
    nt.links.new(ridges.outputs[0],wave.inputs[0])
    b=node(nt,'ShaderNodeBump','Hair bundle ridges',(-115,-45))
    b.inputs['Strength'].default_value=.44
    b.inputs['Distance'].default_value=.000104
    nt.links.new(wave.outputs[0],b.inputs['Height'])
    nt.links.new(b.outputs[0],p.inputs['Normal'])

    def drape(u,t):
        shape=((0,1.646,.043,.015,-.007),(.10,1.598,.064,.055,.024),
               (.22,1.543,.087,.071,.028),(.39,1.456,.092,.083,.046),
               (.59,1.353,.121,.102,.057),(.81,1.253,.136,.114,.067),
               (1.,1.175,.115,.105,.065))
        control=np.array(shape)
        z,width,center,edge=(np.interp(t,control[:,0],control[:,j]) for j in range(1,5))
        u=min(1,max(-1,u))
        wave=(.006+.014*t)*sin(t*12+u*7)*sin(t*pi*.96)
        x=u*width+wave + .008*sin(t*7.3)*ease(.3,.7,t)
        y=center-(center-edge)*abs(u)**2.45
        y+=ease(.055,.38,t)*((.0012+.0068*t)*sin(u*28.9+t*8)+.0032*t*sin(u*52-t*11.6))
        z+=(.017*cos(u*20.4)+.014*cos(u*11.3+.8)+.018*abs(u))*ease(.47,1,t)
        hit=tree.ray_cast(Vector((x,.25,z)),Vector((0,-1,0)))[0] if t<.38 else None
        if hit:
            y=max(y,hit.y-.0037+.0080*ease(.028,.34,t))
        return x,y,z

    n,m=123,73
    u=np.linspace(-1,1,n)
    t=np.linspace(0,1,m)
    positions=np.asarray([[drape(x,y) for x in u] for y in t])
    back=[]
    for j,s in enumerate(t):
        for i,q in enumerate(u):
            x,y,z=positions[j,i]
            thickness=(.003+.020*ease(.15,.43,s))*sin(pi*(q+1)/2)**.64
            thickness*=1-ease(.84,1.06,s)
            back.append((x,y-thickness,z))
    raw=np.concatenate((positions.reshape(-1,3),np.asarray(back)))
    faces=[]
    for j in range(m-1):
        for i in range(n-1):
            a=j*n+i
            faces.append((a,a+1,a+n+1,a+n))
            faces.append(tuple(m*n+k for k in (a+n,a+n+1,a+1,a)))
    for i in range(n-1):
        for a in (i,(m-1)*n+i):
            faces.append((a,a+1,a+1+m*n,a+m*n))
    for j in range(m-1):
        for a in (j*n,j*n+n-1):
            faces.append((a,a+n,a+n+m*n,a+m*n))
    obj=mesh_object('Nape and back • layered hair volume',col,varied_pose(raw),faces,base)
    coords=[((v%(n*m))%n/(n-1), (v%(n*m))//n/(m-1)) for f in faces for v in f]
    attr=obj.data.uv_layers.new(name='Growth direction')
    attr.uv.foreach_set('vector',np.asarray(coords,dtype=np.float32).reshape(-1))
    # Cover the underlying volume with dense independent hairs, including
    # offset waves and lengths following the shoulder blades.
    paths,radii=[],[]
    for i in range(5900):
        u=rng.uniform(-.999,.999)
        start=rng.uniform(.0,.46)**1.9
        end=rng.uniform(.76,1.08)
        s=np.linspace(start,min(end,1),30)
        drift=rng.uniform(-.11,.11)*np.sin(s*pi*.82)**2
        q=np.clip(u+drift+.020*np.sin(s*11+rng.uniform(0,pi)),-.999,.999)
        a=(q+1)/2*(n-1)
        b0=np.clip(s*(m-1),0,m-1-.000001)
        ax,bx=np.minimum(n-2,a.astype(int)),np.minimum(m-2,b0.astype(int))
        da,db=(a-ax)[:,None],(b0-bx)[:,None]
        path=((1-da)*(1-db)*positions[bx,ax]+da*(1-db)*positions[bx,ax+1]
              +(1-da)*db*positions[bx+1,ax]+da*db*positions[bx+1,ax+1])
        path[:,1]+=(.00030+rng.random()**1.77*.0085)*(np.sin(pi*np.linspace(.08,.94,len(s)))+.17)*np.clip(s*7,0,1)
        path[:,0]+=.0021*np.sin(s*18+rng.uniform(0,2*pi))*rng.random()*np.clip(s*3,0,1)
        path[:,2] -= max(0,end-1)*.31*np.linspace(0,1,len(s))**3
        paths.append(varied_pose(path))
        radii.append(radius_profile(rng.uniform(.000036,.000081),len(path)))
    native_strands('Back waves • individually combed fibers',col,paths,radii,mat)


def length_group(col,name,specs,grids,mat,rng):
    paths,radii=[],[]
    a=np.linspace(0,1,15)
    for sign,phi,spread,radius,number,knots in specs:
        roots,normals=grids[sign]
        end=sample_grid(roots,phi,np.array(1.))
        # A common path gives the clump a direction with distributed roots
        # and loosely diverging tapered ends.
        m=24
        ext=catmull([end,*knots],m)[1:]
        s=np.linspace(0,1,m)[1:]
        perps=np.c_[np.gradient(ext[:,2]),np.zeros(m-1),-np.gradient(ext[:,0])]
        perps/=np.maximum(.00001,np.linalg.norm(perps,axis=1))[:,None]
        for k in range(number):
            root_offset=rng.gauss(0,.41)
            side=rng.gauss(0,.51)
            base_angle=np.clip(phi+spread*root_offset,F_MIN+.001,F_MAX-.001)
            st=rng.random()**1.9*.71
            t=np.linspace(st,1,15)
            q=base_angle+np.sin(pi*a)*rng.uniform(-.032,.032)
            n=sample_grid(normals,q,t)
            points=sample_grid(roots,q,t)+n*rng.uniform(.00024,.0032)
            lateral_spread=radius*(.87*side*(.52+.50*s)+rng.uniform(-.67,.67)*np.sin(2*pi*s+.3)*s)
            swept=ext+perps*lateral_spread[:,None]
            swept[:,1]+=radius*(rng.gauss(0,.37)*np.sin(pi*s*.76)**.8+root_offset*.44)
            swept+=(points[-1]-end)*np.exp(-6*s)[:,None]
            j=rng.randint(0,7)
            if j:
                swept=swept[:-j]
            all_points=np.concatenate((points,swept))
            if sign<0:
                all_points[:,0]+=.022*np.clip((1.355-all_points[:,2])/.198,0,1)**1.3
            paths.append(varied_pose(all_points))
            radii.append(radius_profile(rng.uniform(.000043,.000085),len(all_points)))
    native_strands(name,col,paths,radii,mat)


def fringe_hairs(col,tree,mat,rng):
    paths,radii=[],[]
    t=np.linspace(0,1,33)
    styles=[
        (-.029,1.637,-.012,1.562,163),
        (-.025,1.635,.012,1.554,150),
        (-.021,1.639,.032,1.564,131),
        (-.016,1.641,.051,1.578,151),
        (-.012,1.640,.072,1.549,102),
        (-.034,1.636,-.043,1.550,118),
        (-.040,1.642,-.063,1.541,183),
        (-.046,1.644,-.072,1.520,167),
    ]
    for a,z0,b,z1,cnt in styles:
        for i in range(cnt):
            jitter=rng.gauss(0,.0038)
            start_x=a+jitter
            end_x=b+jitter+rng.gauss(0,.0047)
            z_end=z1+rng.gauss(0,.0051)
            height=z0+rng.gauss(0,.0017)
            lift=rng.uniform(.0012,.006)
            xs=start_x+(end_x-start_x)*(.15*t+.85*t**1.64)+rng.uniform(-.005,.003)*np.sin(t*pi)
            zs=height+(z_end-height)*(1.50*t-.5*t*t)
            y=[]
            for f,x,z in zip(t,xs,zs):
                face=visible_surface(tree,x,z,-.111)
                y.append(face-.0016-lift*(.26+.79*sin(pi*f)**.75)-.0029*(1-f)**3+rng.uniform(-.00015,.00015))
            p=np.c_[xs,y,zs]
            paths.append(pose_array(p))
            radii.append(radius_profile(rng.uniform(.000034,.000061),len(t)))
    native_strands('Fringe • separated wispy bangs',col,paths,radii,mat)


def swept_temples(col,grids,mat,rng):
    """Comb out of the scalp surface before loosening along the temples."""
    guides=[
        (1,.663,.641,238,[(.053,-.120,1.620),(.067,-.101,1.586),(.079,-.072,1.555),(.088,-.030,1.530),(.097,.016,1.479)]),
        (1,.928,.719,247,[(.075,-.079,1.603),(.084,-.042,1.572),(.093,.000,1.539),(.099,.024,1.481)]),
        (1,.607,.474,174,[(.038,-.137,1.624),(.055,-.122,1.595),(.070,-.096,1.567),(.082,-.057,1.547),(.087,-.035,1.507)]),
        (-1,.827,.795,284,[(-.076,-.124,1.566),(-.085,-.126,1.527),(-.098,-.109,1.481),(-.128,-.125,1.405)]),
    ]
    paths,radii=[],[]
    for sign,phi,turn,count,knots in guides:
        points,normals=grids[sign]
        gate=sample_grid(points,phi,np.array(turn))
        template=catmull([gate,*knots],26)[1:]
        steps=np.linspace(0,1,len(template))
        for j in range(count):
            phase=phi+rng.gauss(0,.024)
            fraction=np.linspace(rng.uniform(.04,.30),turn+rng.uniform(-.035,.035),12)
            scalp=sample_grid(points,phase,fraction)
            lift=np.linspace(-.00054,.0015,len(scalp))[:,None]
            scalp+=sample_grid(normals,phase,fraction)*lift
            p=template.copy()
            p[:,0]+=rng.gauss(0,.0034)*np.sin(pi*steps*.87)
            p[:,1]+=rng.gauss(0,.0029)*np.sin(pi*steps*.90)
            p[:,2]+=rng.gauss(0,.0036)*np.sin(pi*steps*.86)
            p+=(scalp[-1]-gate)*np.exp(-5*steps)[:,None]
            cutoff=rng.randint(21,25)
            path=np.concatenate((scalp,p[:cutoff]))
            paths.append(varied_pose(path))
            radii.append(radius_profile(rng.uniform(.000039,.000069),len(path)))
    native_strands('Temple sweeps • crown-to-length transitions',col,paths,radii,mat)


def flyaway_hairs(col,grids,mat,rng):
    paths,radii=[],[]
    t=np.linspace(0,1,32)
    for i in range(330):
        sign=rng.choice([-1,1])
        pts,norm=grids[sign]
        phi=rng.uniform(.48,2.85)
        f=rng.uniform(.04,.56)
        stop=rng.uniform(.73,1.)
        f=f+(stop-f)*t
        ramp=np.sin(pi*t)**1.34
        p=phi+rng.uniform(-.13,.13)*(t*t)+.018*np.sin(t*6)
        q=sample_grid(pts,p,f)+sample_grid(norm,p,f)*(rng.uniform(.0035,.015)*ramp+.001)[:,None]
        paths.append(pose_array(q))
        radii.append(radius_profile(rng.uniform(.000020,.000036),len(t)))
    stray=[
        [(-.106,-.136,1.477),(-.132,-.139,1.412),(-.178,-.152,1.360),(-.191,-.144,1.327),(-.146,-.165,1.292),(-.090,-.161,1.297)],
        [(-.121,-.106,1.444),(-.179,-.092,1.398),(-.196,-.099,1.362),(-.189,-.122,1.338),(-.145,-.145,1.324)],
        [(.097,-.043,1.477),(.118,-.074,1.417),(.151,-.125,1.376),(.142,-.114,1.327),(.108,-.132,1.302),(.076,-.130,1.316)],
        [(.091,-.033,1.501),(.099,-.058,1.449),(.108,-.098,1.393),(.134,-.136,1.341),(.159,-.123,1.323)],
    ]
    for knots in stray:
        p=catmull(knots,37)
        for i in range(40):
            j=rng.uniform(-.006,.006)
            q=p.copy()
            q[:,0]+=j*np.linspace(.4,1,37)
            q[:,1]+=rng.gauss(0,.0025)
            q[:,2]+=rng.gauss(0,.0036)
            paths.append(varied_pose(q))
            radii.append(radius_profile(rng.uniform(.000030,.000052),37))
    native_strands('Flyaways • crown and loose ends',col,paths,radii,mat)


def build_hair(col,tree):
    rng=Random(99324)
    center=fiber_shader('Hair | chestnut scattering',(.208,.094,.054),.415)
    tips=fiber_shader('Hair | warm chestnut lengths',(.236,.109,.061),.409)
    faint=fiber_shader('Hair | light fine fibers',(.275,.139,.089),.436)
    grids=scalp_sheet(col,tree)
    crown_hairs(col,grids,center,rng)
    length_group(col,'Long waves • draped over shoulders',front_locks(),grids,tips,rng)
    back_drape(col,tree,center,rng)
    fringe_hairs(col,tree,tips,rng)
    swept_temples(col,grids,center,rng)
    flyaway_hairs(col,grids,faint,rng)
