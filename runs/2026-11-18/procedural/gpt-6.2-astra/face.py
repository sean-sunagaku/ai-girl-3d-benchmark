"""Sculpt, skin and eye construction for the portrait."""
from atelier import *
from mathutils.bvhtree import BVHTree

V=[]
F=[]
with (ROOT/'input/upper-body.obj').open() as source:
    for line in source:
        if line.startswith('v '):
            V.append([float(v)*1000 for v in line.split()[1:4]])
        elif line.startswith('f '):
            F.append(tuple(int(v.split('/')[0])-1 for v in line.split()[1:]))
V=np.array(V)
base_tree=BVHTree.FromPolygons(V.tolist(),F)
upper_tree=BVHTree.FromPolygons(V.tolist(),[f for f in F if max(V[i,2] for i in f)<1425])

def front_y(x,z):
    hit=base_tree.ray_cast(Vector((x,-250,z)),Vector((0,1,0)))[0]
    return hit.y if hit is not None else -130

def neck_top(x,y):
    hit=upper_tree.ray_cast(Vector((x,y,1450)),Vector((0,0,-1)))[0]
    return hit.z if hit else 1370

def collar_height(angle):
    return 1362+24*sin(angle)**2+7*(1-cos(angle))

def rim_z(a,upper=True):
    ids=([66,63,60,57,54,1,0,6,9,12,18,21] if upper else [69,72,53,50,47,44,41,36,33,30,24,21])
    pts=sorted([(abs(V[i,0]),V[i,2]) for i in ids]);return np.interp(a,[p[0] for p in pts],[p[1] for p in pts])

def brow_axes(a):
    t=sat((a-14.4)/38.8);z=1558.8+6.0*sin(pi*(.055+.90*t))+.35*t
    return t,z

def sculpt(p):
    x,y,z=p
    a=abs(x); s=1 if x>=0 else -1
    f=smooth(-85,-130,y)
    # Smooth jaw/chin taper, malar volume, and a softer supraorbital plane.
    xx=x*(1-.094*gauss(z-1462,37)*smooth(-65,-105,y))
    yy=y-3.7*gauss(a-43,19)*gauss(z-1515,22)*f
    yy+=2.8*gauss(a-28,26)*gauss(z-1564,9)*f
    zz=z+12.5*gauss(z-1432,37)*smooth(-58,-112,y)
    zz+=3.5*gauss(z-1476,17)*gauss(a,33)*f
    xx+=s*1.7*gauss(a-49,17)*gauss(z-1510,28)*f
    yy-=1.45*gauss(a-36,18)*gauss(z-1518,18)*f
    # Shorter alae, a gently raised lobule and less projection through the bridge.
    nose=gauss(z-1509,17)*gauss(a,18)*f
    xx-=.027*x*nose
    zz+=2.15*nose
    yy+=(2.5*gauss(z-1523,22)*gauss(a,12)-.65*gauss(z-1507,5)*gauss(a,6))*f
    # Smile travels through the perioral surface into both cheeks.
    mouth=gauss(z-1477.5,14)*f
    xx+=s*2.6*gauss(a-21,13)*mouth
    zz+=(4.3+.32*s)*gauss(a-23,10.8)*gauss(z-1477.8,13)*f
    zz-=.24*(z-1477.6)*gauss(z-1477.6,14)*gauss(a,25)*f
    yy+=1.8*gauss(a-22.8,5.2)*gauss(z-1478,6)*f
    yy+=1.45*gauss(a,19)*gauss(z-1484,6)*f
    yy-=1.2*gauss(a,21)*gauss(z-1471,4.2)*f
    closed=gauss(z-1476.9,2.7)*smooth(1477.7,1476.6,z)*gauss(a,22)*f
    zz+=1.04*closed;yy-=.91*closed
    lidspan=smooth(17,23,a)*smooth(46,41,a)*f
    yy+=1.15*gauss(z-rim_z(a)-3.9,1.55)*lidspan
    yy-=1.02*gauss(z-rim_z(a,False)+2.8,1.9)*lidspan
    yy+=.63*gauss(z-rim_z(a,False)+5.0,1.2)*lidspan
    # Squint and canthal tilt preserve all of the source eye loops.
    eye=gauss(a-31,25)*gauss(z-1545.3,14)*smooth(-96,-125,y)
    xx+=s*(a-31)*.085*eye
    zz+=((z-1545.3)*.105+(a-31)*.092)*eye
    # Turn the exposed ear slightly out of the tuck, retaining the helix and bowl.
    ear=smooth(64,77,a)*gauss(z-1531,34)*gauss(y+52,33)
    xx+=s*(1.8 if s>0 else -1.6)*ear
    yy-=2.3*ear*(s>0)
    return np.array((xx,yy,zz))

def face_point(p):
    return pose(sculpt(p))

def skin_color(p):
    x,y,z=p; a=abs(x); f=smooth(-75,-117,y)
    c=np.array([.64,.347,.270])
    def wash(target,weight):
        nonlocal c
        c=lerp(c,np.array(target),sat(weight))
    wash((.66,.231,.197),.70*gauss(a-45,20)*gauss(z-1519,17)*f)
    wash((.64,.248,.193),.68*gauss(a,17)*gauss(z-1508,14)*f)
    wash((.41,.169,.143),.51*gauss(a-30,16)*gauss(z-1551,8)*f)
    wash((.54,.201,.174),.51*gauss(a-32,18)*gauss(z-1537,5.8)*f)
    wash((.60,.270,.218),.42*smooth(65,78,a)*gauss(y+50,30)*gauss(z-1530,35))
    wash((.55,.282,.226),.19*gauss(z-1443,32))
    # Vermilion follows the original lip loops, with a soft irregular border.
    upper=np.interp(a,[0,2.3,5.4,8,11,17.8,22,25.1],[1485.2,1485.7,1487.0,1487.5,1486.1,1483.4,1480.2,1477.65])
    lower=np.interp(a,[0,5.4,12.8,18.4,22,25.1],[1465.9,1466.7,1468.7,1472.7,1475.7,1477.45])
    border=smooth(lower-1.2,lower+1.6,z)*smooth(upper+.95,upper-1.05,z)*smooth(25.9,24.1,a)*f
    lc=lerp(np.array([.594,.116,.135]),np.array([.442,.084,.107]),smooth(1475,1482,z)*.82)
    wash(lc,border*.96)
    seam=np.interp(a,[0,6,12,18,23,25],[1477.34,1477.5,1477.8,1477.8,1477.6,1477.6])
    wash((.303,.085,.075), .30*gauss(z-seam,.87)*smooth(25.9,23,a)*f)
    wash((.478,.160,.132),.48*smooth(71,77,a)*smooth(87,78,a)*gauss(y+54,12)*gauss(z-1530,15))
    # Deepest alar recesses and lacrimal corners hold slightly more colour.
    wash((.195,.070,.048),.44*gauss(a-9.2,3.2)*gauss(z-1501.4,2.6)*f)
    wash((.52,.189,.171),.53*gauss(a-18.3,2.7)*gauss(z-1544,2)*f)
    span=smooth(16.9,22.4,a)*smooth(46.7,42.4,a)*f
    wash((.347,.143,.120),.63*gauss(z-rim_z(a)-3.9,1.2)*span)
    wash((.496,.173,.143),.42*gauss(z-rim_z(a,False)+1.4,2.1)*span)
    wash((.46,.203,.146),.31*gauss(a-15,4)*gauss(z-1503,7)*f)
    wash((.50,.244,.168),.22*gauss(a-19,8)*gauss(z-1544,14)*f)
    t,bz=brow_axes(a)
    wash((.24,.118,.075),.52*gauss(z-bz-.7,2.6)*smooth(13.2,16.5,a)*smooth(54.8,50,a)*f)
    wash((.68,.345,.256),.28*gauss(a-42,19)*gauss(z-1475,17)*f)
    return (*c,1)

def skin_material():
    m,b=principle('Skin | apricot, rose and vermilion',(.69,.39,.31),.425,**{'Subsurface Weight':.105,'Subsurface Scale':.00115,'Subsurface Radius':(1.,.47,.27),'Specular IOR Level':.42})
    c=node(m,'ShaderNodeVertexColor','Paintable circulation and lip tint',-760,300);c.layer_name='Complexion'
    tc=node(m,'ShaderNodeTexCoord','Metric microstructure',-970,-80)
    broad=node(m,'ShaderNodeTexNoise','Natural chromatic mottling',-745,65);broad.inputs['Scale'].default_value=260; broad.inputs['Detail'].default_value=3
    link(m,tc.outputs['Object'],broad.inputs['Vector'])
    ramp=node(m,'ShaderNodeValToRGB','Restrained complexion variation',-510,95)
    ramp.color_ramp.elements[0].color=(.84,.83,.84,1);ramp.color_ramp.elements[1].color=(1,.985,.96,1)
    link(m,broad.outputs['Fac'],ramp.inputs['Fac'])
    mix=node(m,'ShaderNodeMixRGB','Layer over anatomical colour',-255,210);mix.blend_type='MULTIPLY';mix.inputs[0].default_value=.66
    link(m,c.outputs['Color'],mix.inputs[1]);link(m,ramp.outputs['Color'],mix.inputs[2]);link(m,mix.outputs[0],b.inputs['Base Color'])
    pores=node(m,'ShaderNodeTexNoise','Fine pores',-745,-235); pores.inputs['Scale'].default_value=2800; pores.inputs['Detail'].default_value=2.1; pores.inputs['Roughness'].default_value=.66
    link(m,tc.outputs['Object'],pores.inputs['Vector'])
    bump=node(m,'ShaderNodeBump','Fingertip-scale relief',-260,-135);bump.inputs['Strength'].default_value=.21;bump.inputs['Distance'].default_value=.00012
    link(m,pores.outputs['Fac'],bump.inputs['Height']); link(m,bump.outputs['Normal'],b.inputs['Normal'])
    sheen=node(m,'ShaderNodeVertexColor','Local hydration',-735,-470);sheen.layer_name='Hydration'
    rough=node(m,'ShaderNodeMapRange','Skin to satin lip',-264,-405);rough.inputs['To Min'].default_value=.48;rough.inputs['To Max'].default_value=.255
    link(m,sheen.outputs['Color'],rough.inputs[0]);link(m,rough.outputs[0],b.inputs['Roughness'])
    link(m,sheen.outputs['Color'],b.inputs['Coat Weight']);b.inputs['Coat Roughness'].default_value=.23
    # Sparse melanin flecks are evaluated in object space, restricted to the malar area.
    malar=node(m,'ShaderNodeVertexColor','Cheek and nasal distribution',-930,-665);malar.layer_name='Pigment region'
    v=node(m,'ShaderNodeTexVoronoi','Irregular pigment sites',-746,-840);v.inputs['Scale'].default_value=212;link(m,tc.outputs['Object'],v.inputs['Vector'])
    size=node(m,'ShaderNodeMapRange','Soft submillimetre edges',-489,-764);size.inputs['From Min'].default_value=.043;size.inputs['From Max'].default_value=.174;size.inputs['To Min'].default_value=.38;size.inputs['To Max'].default_value=0;link(m,v.outputs['Distance'],size.inputs[0])
    choose=node(m,'ShaderNodeSeparateColor','Site variation',-490,-952);link(m,v.outputs['Color'],choose.inputs[0]);rnd=node(m,'ShaderNodeMath','Fleck opacity',-238,-840);rnd.operation='MULTIPLY';link(m,choose.outputs['Red'],rnd.inputs[0]);link(m,size.outputs[0],rnd.inputs[1]);mask=node(m,'ShaderNodeMath','Localized pigment',-33,-640);mask.operation='MULTIPLY';link(m,malar.outputs['Color'],mask.inputs[0]);link(m,rnd.outputs[0],mask.inputs[1]);result=node(m,'ShaderNodeMixRGB','Diffuse melanin flecks',135,238);result.inputs[2].default_value=(.245,.098,.053,1);link(m,mask.outputs[0],result.inputs[0]);link(m,mix.outputs[0],result.inputs[1]);link(m,result.outputs[0],b.inputs['Base Color'])
    rest=node(m,'ShaderNodeAttribute','Unposed metric skin coordinate',-972,-1208);rest.attribute_name='Skin coordinate'
    verm=node(m,'ShaderNodeVertexColor','Vermilion relief mask',-984,-1482);verm.layer_name='Vermilion'
    stretch=node(m,'ShaderNodeVectorMath','Fine vertical lip furrows',-730,-1210);stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(2400,380,450);link(m,rest.outputs['Vector'],stretch.inputs[0]);furrows=node(m,'ShaderNodeTexNoise','Natural vermilion grain',-510,-1190);furrows.inputs['Scale'].default_value=1.0;furrows.inputs['Detail'].default_value=2.3;link(m,stretch.outputs[0],furrows.inputs[0]);localbump=node(m,'ShaderNodeBump','Softly furrowed lip surface',-80,-1115);localbump.inputs['Distance'].default_value=.000147;link(m,verm.outputs['Color'],localbump.inputs['Strength']);link(m,furrows.outputs['Fac'],localbump.inputs['Height']);link(m,bump.outputs['Normal'],localbump.inputs['Normal']);link(m,localbump.outputs[0],b.inputs['Normal'])
    # A second, broader pigment scale survives at portrait distance.
    detail=node(m,'ShaderNodeTexNoise','Dermal colour at 1–3 millimetres',-744,-1460);detail.inputs['Scale'].default_value=600;detail.inputs['Detail'].default_value=2.5;link(m,tc.outputs['Object'],detail.inputs[0]);dr=node(m,'ShaderNodeValToRGB','Peach and warm-neutral variation',-475,-1450);dr.color_ramp.elements[0].position=.28;dr.color_ramp.elements[0].color=(.88,.793,.755,1);dr.color_ramp.elements[1].position=.72;dr.color_ramp.elements[1].color=(1,1,1,1);link(m,detail.outputs['Fac'],dr.inputs[0]);var=node(m,'ShaderNodeMixRGB','Skin microcolour',351,231);var.blend_type='MULTIPLY';var.inputs[0].default_value=.32;link(m,result.outputs[0],var.inputs[1]);link(m,dr.outputs[0],var.inputs[2]);link(m,var.outputs[0],b.inputs['Base Color'])
    return m

def make_skin(C):
    # Clip the source at a closed neck section, retaining shared intersections on
    # crossing edges. Grow an integral yoke from that same boundary into the collar.
    cut=1418.0;raw=[];faces=[];idx={}
    def add(key,point):
        if key not in idx:idx[key]=len(raw);raw.append(np.array(point))
        return idx[key]
    for poly in F:
        out=[]
        for i,j in zip(poly,poly[1:]+poly[:1]):
            pi0,pj=V[i],V[j];inside=pi0[2]>=cut;insidej=pj[2]>=cut
            if inside:out.append(add(i,pi0))
            if inside!=insidej:
                t=(cut-pi0[2])/(pj[2]-pi0[2]);out.append(add(tuple(sorted((i,j))),lerp(pi0,pj,t)))
        if len(out)>=3:faces.append(out)
    from collections import Counter
    counts=Counter(tuple(sorted((a,b))) for f in faces for a,b in zip(f,f[1:]+f[:1]))
    edges=[(a,b) for f in faces for a,b in zip(f,f[1:]+f[:1]) if counts[tuple(sorted((a,b)))]==1]
    boundary=sorted({i for e in edges for i in e});sculpted=[sculpt(p) for p in raw];posedpoints=[pose(p) for p in sculpted];last={i:i for i in boundary}
    for t in (.10,.25,.46,.66,.83,.95,1.):
        cur={}
        for i in boundary:
            origin=raw[i]; ang=math.atan2(origin[0]/57,-(origin[1]+21)/51);si,co=sin(ang),cos(ang)
            foot=np.array((68.5*si,-21-(72.5 if co>0 else 70)*co,collar_height(ang)-5.9));r=t**1.46
            q=lerp(origin,foot,r);q[2]=lerp(cut,foot[2],t)
            qs=lerp(sculpted[i],foot,r);qs[2]=lerp(sculpted[i][2],foot[2],t)
            qp=qs+(posedpoints[i]-sculpted[i])*(1-smooth(0,.92,t));cur[i]=len(raw);raw.append(q);sculpted.append(qs);posedpoints.append(qp)
        for a,b in edges:faces.append((last[b],last[a],cur[a],cur[b]))
        last=cur
    points=np.asarray(raw);obj=mesh('01 • Facial sculpt and neckline',points,faces,skin_material(),C)
    obj.shape_key_add(name='Basis • hm08 and joined neckline');sk=obj.shape_key_add(name='Portrait • cheeks, nose, lids, smile')
    sk.data.foreach_set('co',(np.asarray(sculpted)*.001).astype(np.float32).reshape(-1));sk.value=1
    posed=obj.shape_key_add(name='Gesture • head and cervical tilt');posed.relative_key=sk
    posed.data.foreach_set('co',(np.asarray(posedpoints)*.001).astype(np.float32).reshape(-1));posed.value=1
    attr_color(obj,'Complexion',[skin_color(p) for p in points]); hyd=[]
    for x,y,z in points:
        lip=gauss(z-1476.5,7.7)*gauss(x,21)*smooth(-122,-140,y)
        w=sat(.09+lip*.89+(.63*gauss(x,14)*gauss(z-1509,12)+.27*gauss(abs(x)-36,21)*gauss(z-1520,11))*smooth(-110,-143,y));hyd.append((w,w,w,1))
    attr_color(obj,'Hydration',hyd);malar=[]
    for x,y,z in points:
        w=sat((.87*gauss(abs(x)-38,20)*gauss(z-1518,14)+.76*gauss(x,17)*gauss(z-1514,14))*smooth(-90,-130,y));malar.append((w,w,w,1))
    rest=obj.data.attributes.new('Skin coordinate','FLOAT_VECTOR','POINT');rest.data.foreach_set('vector',(points*.001).astype(np.float32).ravel());vm=[]
    for x,y,z in points:
        a=abs(x);w=.53*gauss(z-1476,8.4)*gauss(a,21)*smooth(-129,-145,y);vm.append((w,w,w,1))
    attr_color(obj,'Vermilion',vm);attr_color(obj,'Pigment region',malar);obj.data.color_attributes.active_color=obj.data.color_attributes['Complexion']
    subdiv(obj,2)
    return obj

# Dimensions are fitted to the recessed sockets, with a very slightly flatter scleral front.
def globe_point(side, raw_x, raw_z, lift=0):
    a=(raw_x-side*30.9)/16.15; b=(raw_z-1545.23)/15.05
    return np.array((raw_x,-124.75-16.15*sqrt(max(.002,1-a*a-b*b))+.072*side*(raw_x-side*30.9)-lift,raw_z))

def build_eyes(C):
    white,b=principle('Eyes | ivory sclera',(.623,.571,.517),.245,**{'Subsurface Weight':.085,'Subsurface Scale':.00055,'Coat Weight':.26,'Coat Roughness':.13})
    scl=node(white,'ShaderNodeVertexColor','Perilimbal cool and warm corners',-270,140);scl.layer_name='Scleral tint';link(white,scl.outputs[0],b.inputs['Base Color'])
    iris,b=principle('Eyes | radial hazel-brown stroma',(.17,.067,.026),.35,**{'Specular IOR Level':.16})
    ca=node(iris,'ShaderNodeVertexColor','Radial pigment and collarette',-260,150);ca.layer_name='Stroma';link(iris,ca.outputs[0],b.inputs['Base Color'])
    pupil,_=principle('Eyes | pupil',(.004,.0025,.0023),.21)
    cornea=bpy.data.materials.new('Eyes | clear convex corneal film');cornea.use_nodes=True
    tr=cornea.node_tree;tr.nodes.clear();out=tr.nodes.new('ShaderNodeOutputMaterial');out.location=(380,0)
    fres=node(cornea,'ShaderNodeFresnel','Air to tear film',-480,0);fres.inputs['IOR'].default_value=1.376
    clear=node(cornea,'ShaderNodeBsdfTransparent','Transmitted light',-250,40)
    refl=node(cornea,'ShaderNodeBsdfGlossy','Corneal highlights',-250,-110);refl.inputs['Roughness'].default_value=.037
    mx=node(cornea,'ShaderNodeMixShader','Clear tear film',5,10);link(cornea,fres.outputs[0],mx.inputs[0]);link(cornea,clear.outputs[0],mx.inputs[1]);link(cornea,refl.outputs[0],mx.inputs[2]);link(cornea,mx.outputs[0],out.inputs['Surface'])
    tear,_=principle('Eyes | lacrimal tissue',(.49,.198,.171),.29,**{'Subsurface Weight':.16,'Subsurface Scale':.00055,'Coat Weight':.25})
    for side,suffix in [(-1,'R'),(1,'L')]:
        seg=80;rows=36;points=[];faces=[];scl_tint=[]
        for j in range(rows+1):
            v=.0001+(pi-.0002)*j/rows
            for i in range(seg):
                u=2*pi*i/seg;dx=16.15*sin(v)*cos(u)
                dz=15.05*sin(v)*sin(u);points.append(face_point((side*30.9+dx,-124.75-16.15*cos(v)+.072*side*dx,1545.23+dz)))
                nasal=smooth(5.8,13.5,-dx*side);lateral=smooth(6.7,13.5,dx*side);occlusion=smooth(2.0,7.7,dz)
                c=lerp(np.array((.66,.635,.587)),np.array((.525,.347,.327)),.62*nasal+.35*lateral);c*=1-.13*occlusion;scl_tint.append((*c,1))
        for j in range(rows):
            for i in range(seg):
                k=j*seg+i;n=j*seg+(i+1)%seg;faces.append((k,k+seg,n+seg,n))
        globe=mesh('02 • Sclera.'+suffix,points,faces,white,C);attr_color(globe,'Scleral tint',scl_tint)
        rad=6.20;pr=2.61
        direction=(np.array((-35,-860,1570))-head((side*30.9,-124.75,1545.23)))@HEAD_M;direction/=np.linalg.norm(direction)
        xc=side*30.9+direction[0]*13.2;zc=1545.34+direction[2]*13.2
        pts=[];clr=[];fs=[];rings=29;seg=176
        rads=np.r_[np.linspace(.003,pr,4),np.linspace(pr+.018,rad,rings-4)]
        rng=np.random.default_rng(80+(side+1)*3)
        phases=rng.uniform(0,2*pi,6)
        for j,rr in enumerate(rads):
            for i in range(seg):
                u=2*pi*i/seg;t=sat((rr-pr)/(rad-pr))
                ripple=sin(39*u+2.5*t+phases[0])*.33+sin(71*u-3*t+phases[1])*.24+sin(95*u+5.3*t+phases[2])*.17+sin(18*u+sin(9*u)*1.2+phases[3])*.26
                s=.40+.36*ripple+.20*sin(pi*t)+.095*sin(7*u+phases[4])
                limbal=.34+.66*smooth(1,.84,t)
                base=np.array([.064+.170*s,.024+.073*s,.010+.030*s])*limbal
                if .06<t<.39:base*=1+.16*sin(20*t+2*sin(15*u))
                if j<4:base=np.array([.005,.003,.0028])
                clr.append((*base,1));px=xc+rr*cos(u);pz=zc+rr*sin(u)
                pts.append(face_point(globe_point(side,px,pz,.075+.075*(1-t))))
        for j in range(len(rads)-1):
            for i in range(seg):
                k=j*seg+i;n=j*seg+(i+1)%seg;fs.append((k,k+seg,n+seg,n))
        io=mesh('03 • Iris and pupil.'+suffix,pts,fs,iris,C);attr_color(io,'Stroma',clr);io.data.materials.append(pupil)
        for poly in list(io.data.polygons)[:3*seg]:poly.material_index=1
        pts=[];fs=[];ng=80;nr=12
        for j in range(nr+1):
            r=.003+(rad+.75)*j/nr
            for i in range(ng):
                u=i*2*pi/ng; px=xc+r*cos(u);pz=zc+r*sin(u)
                pts.append(face_point(globe_point(side,px,pz,.14+1.28*max(0,1-(r/(rad+.76))**2)**1.55)))
        for j in range(nr):
            for i in range(ng):
                k=j*ng+i;n=j*ng+(i+1)%ng;fs.append((k,k+ng,n+ng,n))
        mesh('04 • Corneal meniscus.'+suffix,pts,fs,cornea,C)
        # A tapered caruncle fills the nasal end, behind the wet lid margin.
        p=[];f=[];nx=14;ny=9
        for i in range(nx):
            t=i/(nx-1);x=17.05+3.22*t;cz=1544.0+.18*t;w=.14+1.15*sin(pi*t)**.7
            for j in range(ny):
                v=j/(ny-1)*2-1;z=cz+w*v;y=-137.0-.73*sin(pi*t)*sqrt(max(0,1-v*v));p.append(face_point((side*x,y,z)))
        for i in range(nx-1):
            for j in range(ny-1):
                k=i*ny+j;f.append((k,k+ny,k+ny+1,k+1) if side>0 else (k+1,k+ny+1,k+ny,k))
        mesh('05 • Lacrimal caruncle.'+suffix,p,f,tear,C)

UPPER=[66,63,60,57,54,1,0,6,9,12,15,18,21]
LOWER=[66,69,72,53,50,47,44,41,36,33,30,27,24,21]

def build_features(C):
    lash,_=principle('Lashes | espresso',(.032,.014,.008),.43)
    brow,_=principle('Brows | soft walnut',(.085,.039,.023),.56)
    rim,_=principle('Lids | upper lash roots',(.070,.027,.022),.50)
    wet,_=principle('Lids | warm tear margin',(.52,.235,.193),.285,**{'Coat Weight':.25})
    shade,bs=principle('Brows | translucent root colour',(.115,.060,.039),.69)
    paint=node(shade,'ShaderNodeVertexColor','Weight at the brow root',-240,80);paint.layer_name='Root softness';link(shade,paint.outputs['Color'],bs.inputs['Alpha'])
    rng=np.random.default_rng(620)
    for side,name in [(-1,'R'),(1,'L')]:
        def line(indices,ts):
            pts=V[indices].copy();pts[:,0]=abs(pts[:,0])*side
            # Length parameterization prevents crowding near the temporal canthus.
            dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))];dist/=dist[-1]
            return np.stack([np.interp(ts,dist,pts[:,i]) for i in range(3)],axis=-1)
        for ids,upper in [(UPPER,True),(LOWER,False)]:
            ts=np.linspace(.025,.99,63);pts=line(ids,ts);pts[:,1]-=.40
            pts[:,2]+= .08 if upper else -.03
            strands(('Upper lash bed.' if upper else 'Lower wet margin.')+name,[[face_point(p) for p in pts]],[.045+(.39 if upper else .20)*np.maximum(0,np.sin(pi*ts))**.6],rim if upper else wet,C)
            paths=[];rads=[];n=83 if upper else 44
            for k in range(n):
                t=(k+.44*sin(k*1.55)+rng.uniform(.05,.72))/n;t=.065+.915*t
                p=line(ids,[t])[0];p[1]-=.42
                length=(2.4+(4.3 if upper else 1.80)*sin(pi*t*.88))*rng.uniform(.60,1.16)
                drift=side*(-.30+.94*t+rng.uniform(-.14,.14));w=np.linspace(0,1,6)
                path=[]
                for u in w:
                    d=np.array([drift*u,-.81*u+.30*u*u,(.20*u+.7*u*u if upper else -.64*u+.10*u*u)])*length
                    path.append(face_point(p+d))
                paths.append(path);rads.append((.010+(.139 if upper else .078)*(1-w)**.73)*rng.uniform(.63,1.09))
            strands(('Upper lashes.' if upper else 'Lower lashes.')+name,paths,rads,lash,C)
        # A soft root colour is laid over the curved skin beneath the individual hairs.
        pts=[];fs=[];weights=[];nx=59;ny=9
        for i in range(nx):
            t=i/(nx-1);x=13.7+40.2*t;bz=brow_axes(x)[1];width=(1.6+1.1*sin(pi*t)**.8)*(1-.69*t**2)
            for j in range(ny):
                v=2*j/(ny-1)-1;px=side*x;pz=bz+.67+width*v;py=front_y(px,pz)-.23
                pts.append(face_point((px,py,pz)));w=.36*(1-abs(v)**1.45)*smooth(0,.079,t)*smooth(1,.81,t)*(.95+.05*sin(43*t+6*v));weights.append((w,w,w,1))
        for i in range(nx-1):
            for j in range(ny-1):
                k=i*ny+j;fs.append((k,k+ny,k+ny+1,k+1) if side>0 else (k+1,k+ny+1,k+ny,k))
        ob=mesh('Brow root shading.'+name,pts,fs,shade,C);attr_color(ob,'Root softness',weights)
        paths=[];rads=[]
        for i in range(555):
            t=rng.uniform(0,1);profile=(.25+.75*sin(pi*(.10+.89*t))**.68)*(1-.62*t)
            x=14.6+38.2*t;z=brow_axes(x)[1]
            z+=rng.normal(0,1.37)*profile
            du=(1.0+3.0*t)*rng.uniform(.63,1.25);dz=(3.4*(1-t)-1.1*t)*rng.uniform(.52,1.19)
            us=np.linspace(0,1,5);path=[]
            for u in us:
                px=side*(x+du*u);pz=z+dz*u+.35*sin(u*pi);py=front_y(px,pz)-.44-.24*sin(pi*u)
                path.append(face_point((px,py,pz)))
            paths.append(path);rads.append((.014+.098*(1-us)**.75)*rng.uniform(.47,1.10)*(.85+.15*profile))
        strands('Brows • individually swept hairs.'+name,paths,rads,brow,C)
