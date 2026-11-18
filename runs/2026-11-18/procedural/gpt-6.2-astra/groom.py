"""A parted scalp groom, asymmetrical lengths and a separate wispy fringe."""
from atelier import *
from face import front_y

CENTRE=np.array((0.,-56.,1554.));RAD=np.array((84.,105.,123.))

def hair_shader():
    m=bpy.data.materials.new('Hair | chestnut fibre scattering');m.diffuse_color=(.094,.039,.022,1);m.use_nodes=True
    tree=m.node_tree;tree.nodes.clear();out=tree.nodes.new('ShaderNodeOutputMaterial');out.location=(380,50)
    h=node(m,'ShaderNodeBsdfHairPrincipled','Chestnut fibres',40,60);h.parametrization='COLOR';h.inputs['Roughness'].default_value=.38;h.inputs['Radial Roughness'].default_value=.51;h.inputs['Coat'].default_value=.11;h.inputs['Random Roughness'].default_value=.15;h.model='HUANG';h.inputs['Reflection'].default_value=.69;h.inputs['Transmission'].default_value=.83;h.inputs['Secondary Reflection'].default_value=.92;h.inputs['Aspect Ratio'].default_value=.88
    info=node(m,'ShaderNodeHairInfo','Per-strand pigment',-695,140)
    r=node(m,'ShaderNodeValToRGB','Walnut roots and amber threads',-440,175);r.color_ramp.elements[0].color=(.009,.0033,.0021,1);r.color_ramp.elements[1].color=(.105,.041,.020,1);mid=r.color_ramp.elements.new(.56);mid.color=(.033,.0117,.0068,1)
    lock=node(m,'ShaderNodeAttribute','Colour shared by neighbouring fibres',-720,-120);lock.attribute_name='Lock pigment'
    dye=node(m,'ShaderNodeMixRGB','Individual and lock colour variation',-520,12);dye.inputs[0].default_value=.54;link(m,info.outputs['Random'],dye.inputs[1]);link(m,lock.outputs['Fac'],dye.inputs[2]);link(m,dye.outputs[0],r.inputs[0]);link(m,r.outputs[0],h.inputs['Color']);link(m,h.outputs[0],out.inputs['Surface']);return m

def scalp(C):
    mat,b=principle('Hair | shaded roots and parted scalp',(.071,.030,.020),.57)
    a=node(mat,'ShaderNodeVertexColor','Part and hairline',-300,130);a.layer_name='Root pigment';link(mat,a.outputs[0],b.inputs['Base Color'])
    points=[];colors=[];fs=[];n=176;nt=33
    for j in range(nt+1):
        for i in range(n):
            a=i*2*pi/n;back=(1-cos(a))*.5
            # A slightly uneven hairline, with recession beside the temples.
            zlim=1606-107*back**.64-10*gauss(abs(sin(a))-.81,.18)*(1-back)+1.3*cos(9*a)+.65*sin(15*a)
            theta=.0002+(math.acos((zlim-CENTRE[2])/RAD[2])-.0002)*j/nt
            normal=np.array((sin(theta)*sin(a),-sin(theta)*cos(a),cos(theta)))
            p=CENTRE+(RAD-np.array((2.1,2.0,2.1)))*normal
            part_x=-12.2+7.0*smooth(-150,17,p[1]);part=gauss(p[0]-part_x,1.65)*smooth(-151,-131,p[1])*smooth(26,-2,p[1])*smooth(1590,1612,p[2]);colr=lerp(np.array((.054,.022,.016)),np.array((.422,.214,.151)),part*.88)
            colors.append((*colr,1));points.append(head(p))
    for j in range(nt):
        for i in range(n):
            k=j*n+i;l=j*n+(i+1)%n;fs.append((k,k+n,l+n,l))
    obj=mesh('20 • Scalp | shaped hairline and part',points,fs,mat,C);attr_color(obj,'Root pigment',colors)

def section(side,r,layer):
    q=-1.115+2.62*r-.70*smooth(.40,1,r);part=(-12.2+7.0*r+1.3*sin(4*pi*r)+.65*sin(12*pi*r))/RAD[0]
    a0=np.array((part,sin(q)*sqrt(1-part*part),cos(q)*sqrt(1-part*part)))
    angle=.69+2.445*r+.075*sin(2*pi*r)+.060*smooth(.82,1,r)
    ze=.118-.666*r+.048*gauss(r-.20,.17)*(side>0)
    ze=lerp(ze,.073+.035*sin(5.6*r),smooth(.35,.76,r))
    hor=sqrt(1-ze*ze);a1=np.array((side*sin(angle)*hor,-cos(angle)*hor,ze));angle01=math.acos(np.clip(np.dot(a0,a1),-1,1))
    us=np.linspace(0,1,17); ns=(np.sin((1-us)*angle01)[:,None]*a0+np.sin(us*angle01)[:,None]*a1)/sin(angle01)
    loft=(2.6+(6.3+3.2*sin(7*r+.65*side))*np.sin(pi*us)**.7+layer+np.sin(pi*us)**.55*(1.55*np.sin(67*r+2*us)+.8*np.sin(112*r-2*us)))
    cap=CENTRE+ns*(RAD+loft[:,None]);cap[:,0]+=side*(.65+1.1*np.sin(us*pi));cap=head_array(cap)
    edge=CENTRE+a1*(RAD+loft[-1]);x,y,z=edge
    if side>0:
        tuck=1-smooth(.25,.52,r)
        ordinary=np.array((x*1.08+side*4,y+4,z-27))
        a=lerp(ordinary,np.array((74.6+5.4*r,-78+97*r,1562+5*r)),tuck)
        b=lerp(np.array((x*1.12+side*8,y+12,z-60)),np.array((75+9*r,-35+62*r,1554-15*r)),tuck)
        c=lerp(np.array((x*1.15+side*11,y+15,1450)),np.array((70.5+19*r,-13+38*r,1497)),tuck)
    else:
        a=np.array((x*1.075+side*(7-3*r),y-3+7*r,z-23))
        b=np.array((x*1.13+side*(12-8*r),y-3+10*r,z-55))
        c=np.array((x*1.14+side*(16-12*r),y-3+16*r,1448-10*r))
    # The left front section comes over the knit; on the tucked side most falls behind it.
    par=np.interp(r,[0,.13,.29,.47,.68,.84,1],[.08,.24,.48,1.62,2.23,2.70,pi]) if side<0 else np.interp(r,[0,.045,.13,.25,.48,.69,.85,1],[.43,1.12,1.98,2.12,2.31,2.48,2.84,pi])
    front=max(0,cos(par));back=max(0,-cos(par));lat=sin(par)
    def dest(z,w,d,offset=0):
        return np.array((side*(w*lat+(78+offset)*front),-32-(d+12)*front+(d*.86)*back,z))
    over=dest(1390-10*back,165,98,-2);over[0]+=-12*(1-r);over[1]+=9*back
    on=dest(1341+7*sin(5*r+side),185,120,5);on[1]+=(10*back-3*front)
    mid=dest(1251+12*cos(4*r),183,149,18);mid[1]+=3*back
    lower=dest(1183+12*sin(7*r),171,151,18);lower[1]+=5*back
    tipz=1103+24*sin(6*r+1.4*side)+23*sin(14*r+.45*side)+16*back+(23 if side>0 else 0)
    tip=dest(tipz,149,143,9);tip[0]+=side*8*sin(10*r+.8);tip[1]+=6*sin(15*r)
    e=cap[-1];tangent=e-cap[-2];tangent/=np.linalg.norm(tangent)
    if side<0:
        first=e+14*tangent;first[2]=min(first[2],e[2]-5.5)
        on[0]+=19*front;mid[0]+=18*front;lower[0]+=16*front;tip[0]+=15*front
        firstfall=lerp(e,on,.30);firstfall[0]-=7+5*sin(r*9);firstfall[1]+=7*cos(par*1.2)
        secondfall=lerp(e,on,.69);secondfall[0]-=7*sin(8*r);secondfall[1]-=5*front
        tailknots=[e,first,firstfall,secondfall,on,mid,lower,tip]
    else:
        # The right-hand lengths drop behind the visible ear and shoulder.
        backangle=1.94+1.20*r
        def tucked(z,w,depth):
            return np.array((w*sin(backangle),-15-depth*cos(backangle),z))
        on=tucked(1334+7*sin(3*r),139,111);mid=tucked(1262,147,127);lower=tucked(1179,145,130);tip=tucked(tipz,123,118)
        near=1-smooth(.25,.53,r)
        first=lerp(e+14*tangent,head(a),near);second=lerp(lerp(e,on,.24),head(b),near)
        third=lerp(lerp(e,on,.51),head(c),near);third[2]=min(third[2],second[2]-14)
        aboveshoulder=lerp(third,on,.46);aboveshoulder[0]+=3.5*sin(10*r)
        tailknots=[e,first,second,third,aboveshoulder,on,mid,lower,tip]
    t=np.linspace(0,1,36);tail=catmull(tailknots,t)
    # Changing wavelength and phase gives broad, non-periodic shifts in direction.
    env=np.sin(t*pi*.60)**1.25;ph=(16.6*r+1.35*sin(10.1*r) if side<0 else 8.9*r)+1.3*side
    tail[:,0]+=env*(16.3*np.sin(12.9*t+ph)+7.2*np.sin(21.5*t+6.0*r+.7*side));tail[:,1]+=env*(9.4*np.sin(14.2*t+ph-1)+6.6*np.sin(22*t+4.6*r))
    return np.vstack((cap,tail[1:]))

def build_lengths(C,mat):
    rng=np.random.default_rng(4115)
    for side,name in [(-1,'R • over shoulder'),(1,'L • tucked')]:
        for under in [False,True]:
            paths=[];widths=[];ng=170 if not under else 78;children=32 if not under else 24;npts=37 if not under else 31
            # A family of smooth envelope curves is re-sampled by neighbouring children.
            grid=np.linspace(0,1,301);ref=np.array([section(side,r,-.9 if under else 1.8) for r in grid]);total=ref.shape[1]
            for k in range(ng):
                g=(k+rng.uniform(.14,.86))/ng;phase=rng.uniform(0,2*pi);phase2=rng.uniform(0,2*pi)
                wave=rng.uniform(1.4,5.1);lockend=rng.normal(0,.020);lockshift=rng.uniform(-.9,.9);depthshift=rng.uniform(-5.7,5.7)
                for h in range(children if g<.56 else int(children*.52)):
                    r=np.clip(g+rng.normal(0,.005 if under else .0033),0,1);ri=r*(len(grid)-1);ii=min(int(ri),len(grid)-2)
                    p=lerp(ref[ii],ref[ii+1],ri-ii)
                    # Roots are spread over the scalp beneath the outside sweep.
                    start=(rng.random()**2.5)*(.30 if under else .21);end=np.clip(.974+lockend+rng.normal(0,.020),.87,1.0)
                    ts=np.linspace(start,end,npts);ix=ts*(total-1);a=np.minimum(ix.astype(int),total-2);q=lerp(p[a],p[a+1],(ix-a)[:,None]);local=np.linspace(0,1,npts)
                    envelope=np.maximum(0,np.sin(np.minimum(1,ts/.40)*pi/2));dz=rng.normal(0,3.2);dy=rng.normal(0,3.7);dx=rng.normal(0,3.4)
                    capweight=np.clip(1-ts/.38,0,1);zfall=np.clip((ts-.33)/.67,0,1)
                    q[:,0]+=envelope*(dx*(.4+.7*zfall)+wave*np.sin(ts*17+phase)+.78*np.sin(45*ts+phase2))
                    q[:,1]+=envelope*(dy*(.45+.78*zfall)+wave*.9*np.sin(16*ts+phase2)+depthshift*zfall)
                    q[:,2]+=envelope*(dz*(.5+zfall)+lockshift*2*np.sin(13*ts+phase))+capweight*.48*sin(phase)
                    q[:,2]+=zfall**3*rng.normal(0,12.4)
                    if under:q[:,1]+=zfall*(4 if r<.30 and side<0 else -2.0)
                    base=rng.uniform(.064,.112)*(.96 if under else 1.0)
                    rad=base*(.86+.14*np.sin(np.minimum(1,local*3)*pi/2))*(1-.95*np.clip((local-.73)/.27,0,1)**1.55)
                    paths.append(q);widths.append(rad)
            strands('21 • Lengths | '+name+(' | inner layer' if under else ' | outer layer'),paths,widths,mat,C)

# Wisps follow the actual forehead surface as they move clear of the scalp shell.
def make_bangs(C,mat):
    rng=np.random.default_rng(4177);paths=[];rads=[]
    # Each lock has its own fall line, curl at the ends and thinning pattern.
    # Knots below describe centre lines on the actual forehead in unposed X/Z.
    locks=[
        ([(-24,1644),(-31,1623),(-35,1596),(-34,1574),(-30,1560),(-23,1555)],2.1,85),
        ([(-21,1643),(-26,1621),(-24,1598),(-19,1581),(-10,1566),(3,1553)],2.1,76),
        ([(-19,1647),(-22,1625),(-20,1601),(-13,1580),(0,1564),(12,1557)],1.7,64),
        ([(-17,1648),(-15,1624),(-8,1601),(3,1584),(18,1573),(27,1567)],2.1,65),
        ([(-13,1650),(-9,1627),(2,1606),(16,1590),(31,1581),(37,1570)],2.5,85),
        ([(-14,1652),(0,1631),(15,1610),(32,1591),(48,1574),(59,1556)],3.4,185),
        ([(-8,1652),(8,1633),(26,1611),(42,1593),(56,1576),(66,1550)],4.3,235),
        ([(-6,1650),(14,1635),(33,1616),(54,1596),(66,1574),(71,1555)],3.3,180),
        ([(-25,1647),(-39,1624),(-47,1600),(-48,1580),(-46,1560),(-43,1548)],2.0,76),
        ([(-26,1648),(-44,1629),(-57,1603),(-61,1577),(-60,1553),(-53,1534)],3.8,162),
        ([(-29,1643),(-47,1624),(-62,1601),(-70,1571),(-72,1542),(-66,1511)],3.5,177),
        ([(-28,1647),(-47,1630),(-64,1606),(-74,1585),(-78,1551),(-77,1524)],3.0,130),
    ]
    sheets=[]
    for knots,halfwidth,count in locks:
        grid=np.linspace(0,1,69);xz=catmull(knots,grid);sheet=[]
        for t,(x,z) in zip(grid,xz):
            shell=-56-109*sqrt(max(.012,1-(x/88)**2-((z-1554)/129.)**2))
            face=front_y(x,z);follows=face-(5.9+7.1*sin(pi*t));y=lerp(min(shell-6.0,follows),follows,smooth(.40,.90,t))
            if abs(x)>62 and z<1555:y=min(y,-81-10*gauss(z-1540,20))
            sheet.append((x,y,z))
        sheet=np.array(sheet);sheets.append(sheet)
        for k in range(count):
            n=27;first=rng.uniform(0,.13);last=np.clip(rng.normal(.906,.091),.67,1);ts=np.linspace(first,last,n);q=catmull(sheet,ts)
            w=(.58+.68*np.sin(pi*ts))*(1-.68*ts**3);offs=rng.normal(0,halfwidth*.65);sw=rng.uniform(-1.45,1.45)
            q[:,0]+=w*offs+.54*np.sin(13*ts+sw)+sw*ts**2;q[:,1]+=rng.uniform(-1.6,1.2)*w+.67*np.sin(9*ts+sw);q[:,2]+=w*offs*.4+rng.uniform(-1.5,1.4)*ts
            paths.append(head_array(q));u=np.linspace(0,1,n);rads.append(rng.uniform(.052,.103)*(1-.95*u**2.65))
    # Downy interstitial fibres have different stop lengths and occasionally cross a lock.
    for a,b in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(8,9),(9,10),(10,11),(0,8)]:
        for i in range(60):
            mix=rng.uniform(.02,.98);base=lerp(sheets[a],sheets[b],mix);n=23;u=np.linspace(0,1,n);ts=np.linspace(rng.uniform(.05,.20),rng.uniform(.53,1),n);pts=catmull(base,ts);drift=rng.normal(0,1.8)
            pts[:,0]+=drift*np.sin(pi*ts)+1.7*np.sin(6*ts+rng.uniform(-1,1))*ts;pts[:,1]-=.65+1.6*sin(pi*mix);pts[:,2]+=rng.uniform(-2.6,2.6)*ts
            paths.append(head_array(pts));rads.append(rng.uniform(.039,.077)*(1-.96*u**2.25))
    strands('22 • Fringe | tapered curtain and brow-skimming wisps',paths,rads,mat,C)
    # A few long, loose locks bridge the ear tuck and the neckline. The opposite
    # shoulder carries a second, staggered layer of S-shaped chest-length waves.
    paths=[];rads=[]
    for side,count in [(-1,38),(1,13)]:
        for j in range(count):
            g=(j+.3)/count;phase=rng.uniform(0,2*pi)
            if side<0:
                ref=section(-1,.008+.30*g,2.5);start=rng.uniform(.15,.35);tail=rng.uniform(.84,1);ts=np.linspace(start,tail,41);q=catmull(ref,ts);u=np.linspace(0,1,len(q));q[:,1]-=3+9*np.sin(u*pi)**.8
                q[:,0]+=(6+3*sin(phase))*np.sin(13*u+phase)*np.sin(pi*u*.65);q[:,2]+=3*sin(phase)*u;children=12
            else:
                root=head((59+5*g,-108+8*g,1569+9*g));p1=head((67+4*g,-84+2*g,1543+5*g));p2=head((69+7*g,-74+13*g,1499+12*g));
                knots=[root,p1,p2,(-1+14*g,-91+11*g,1447-17*g),(12+37*g,-90-12*g,1399-18*g),(37+51*g,-109-10*g,1360-13*g),(61+60*g,-104+8*g,1353-10*g),(78+54*g,-82+14*g,1366-14*g)]
                q=catmull(knots,np.linspace(0,rng.uniform(.64,1),34));u=np.linspace(0,1,len(q));q[:,0]+=3.2*np.sin(9*u+phase)*np.sin(pi*u);q[:,1]+=2.0*np.sin(6*u+phase)*u;children=8 if j%2 else 12
            for k in range(children):
                end=rng.uniform(.82,1);w=np.linspace(0,end,len(q));pts=catmull(q,w);pts[:,0]+=rng.normal(0,1.4)*np.sin(pi*w*.85)+.5*np.sin(15*w+phase);pts[:,1]+=rng.normal(0,1);paths.append(pts);rads.append(rng.uniform(.043,.083)*(1-.94*np.linspace(0,1,len(q))**2.6))
    strands('23 • Free lengths | ear tendrils and separated waves',paths,rads,mat,C)
    # Loose single fibres turn out of the groom at the temples and against the light.
    paths=[];rads=[]
    for i in range(266):
        s=rng.choice([-1,1]);r=rng.random();base=section(s,r,2.3);n=36;u=np.linspace(0,1,n)
        t0=rng.uniform(.015,.31);te=rng.uniform(.51,1);q=catmull(base,lerp(t0,te,u));amp=rng.uniform(3.5,11.3);ph=rng.uniform(0,2*pi)
        q[:,0]+=s*amp*np.sin(pi*u)**.75+2.1*np.sin(16*u+ph)*u;q[:,1]+=amp*.50*np.sin(pi*u+ph)*u;q[:,2]+=amp*.58*np.sin(pi*u*1.25)
        paths.append(q);rads.append(rng.uniform(.045,.089)*(1-.94*u**2))
    for i in range(143):
        side=rng.choice([-1,1]);r=rng.uniform(.00,.88);p=section(side,r,2.6)[:17];u=np.linspace(0,1,21);ts=np.linspace(rng.uniform(.00,.23),rng.uniform(.67,1),len(u));q=catmull(p,ts);relative=q-head(CENTRE);relative/=np.linalg.norm(relative,axis=1)[:,None];lift=rng.uniform(2.,12.)*np.sin(pi*u)**.85;q+=relative*lift[:,None];q[:,0]+=rng.uniform(-3,3)*np.sin(2*pi*u);paths.append(q);rads.append(rng.uniform(.043,.083)*(1-.95*u**2.5))
    strands('23 • Flyaways | loose temple and back fibres',paths,rads,mat,C)

def back_guide(f,phase):
    # Hair leaves the round occiput at its widest section and then falls with
    # gravity. The lengths retain depth behind the nape and over the sweater.
    f=.78*f
    u=np.linspace(0,1,13);q0=.45+.29*abs(f)+.13*sin(3.1*f);q=q0+(1.59-q0)*u
    xf=(-.095+.30*f)*(1-u)+(.943*f)*u+.045*np.sin(pi*u)*sin(3.3*f+.8)
    h=np.sqrt(1-xf*xf);normal=np.stack((xf,h*np.sin(q),h*np.cos(q)),axis=1)
    loft=-1.6+(6.4+2.8*(1-smooth(.50,.78,abs(f))))*(.5-.5*np.cos(pi*np.clip(u/.79,0,1)))+1.45*np.sin(pi*u)*sin(phase)
    cap=head_array(CENTRE+normal*(RAD+loft[:,None]));edge=cap[-1];tangent=edge-cap[-2];tangent/=np.linalg.norm(tangent)
    p1=edge+tangent*20;p2=edge+tangent*43+np.array((3*f,5.4,0))
    shoulder=np.array((141*f-19,45+48*(1-f*f),1358+5*sin(4*f)))
    startfall=lerp(p2,shoulder,.52);startfall[1]+=3.5+4.8*cos(f*4)
    mid=np.array((145*f-14,52+57*(1-f*f),1261+12*sin(3*f)))
    low=np.array((143*f-14,49+61*(1-f*f),1177+12*sin(5*f)))
    tip=np.array((132*f-7,49+45*(1-f*f),1083+71*abs(f)**1.4+12*sin(8*f+1)))
    v=np.linspace(0,1,26);tail=catmull([edge,p1,p2,startfall,shoulder,mid,low,tip],v)
    env=np.sin(v*pi*.55)**1.4;tail[:,0]+=env*(10*np.sin(12*v+phase)+6.2*np.sin(20*v+3.4*f));tail[:,1]+=env*(8.8*np.sin(12.2*v+phase+1.1)+3.4*np.sin(22*v+f))
    return np.vstack((cap,tail[1:]))

def back_and_inner(C,mat):
    rng=np.random.default_rng(851);paths=[];rads=[];grid=np.linspace(-1,1,320);base=np.array([back_guide(f,6.1*f+1.2*sin(7*f)) for f in grid]);n=32;u=np.linspace(0,1,n)
    for j in range(142):
        f=-1+2*(j+.47)/142;phase=rng.uniform(0,2*pi);lock=rng.uniform(-.020,.023)
        for k in range(25):
            idx=np.clip((f+rng.normal(0,.0058)+1)*.5*(len(grid)-1),0,len(grid)-1);i=min(int(idx),len(grid)-2);guide=lerp(base[i],base[i+1],idx-i)
            ts=np.linspace(rng.uniform(0,.20),np.clip(.968+lock+rng.normal(0,.017),.84,1),n);q=catmull(guide,ts);fade=np.sin(np.minimum(1,ts/.33)*pi/2);q[:,0]+=fade*(rng.normal(0,2.1)+3.2*np.sin(14*ts+phase));q[:,1]+=fade*(rng.normal(0,2.4)+2.8*np.sin(12*ts+phase));q[:,2]+=rng.normal(0,8.)*ts**3
            paths.append(q);rads.append(rng.uniform(.060,.106)*(1-.95*u**3.6))
    strands('24 • Occipital sweep | continuous back waves',paths,rads,mat,C)
    paths=[];rads=[]
    for j in range(48):
        g=(j+.5)/48;ph=5.6*g+rng.uniform(-.6,.6);roots=section(-1,.14+.33*g,-2);e=roots[14];a=head((-74-7*sin(pi*g),-63+41*g,1519));b=head((-69-10*g,-67+41*g,1479));c=np.array((-66-44*g,-74-25*g,1406+11*g));d=np.array((-65-55*g,-116-21*g,1366+9*g));mid=np.array((-64-57*g,-160-12*g,1273));end=np.array((-64-56*g,-160-17*g,1134+40*g+14*sin(9*g)))
        guide=np.vstack((roots[7:15],catmull([e,a,b,c,d,mid,end],np.linspace(0,1,23))[1:]));n=30;u=np.linspace(0,1,n)
        for k in range(22):
            ts=np.linspace(rng.uniform(0,.15),rng.uniform(.86,1),n);q=catmull(guide,ts);env=np.sin(ts*pi*.63);q[:,0]+=env*(rng.normal(0,2.4)+8*np.sin(12*ts+ph));q[:,1]+=env*(rng.normal(0,2.5)+5.6*np.sin(15*ts+ph));paths.append(q);rads.append(rng.uniform(.064,.110)*(1-.95*u**3.2))
    strands('25 • Underlayer | behind the tilted jaw',paths,rads,mat,C)

def make_hair(C):
    mat=hair_shader();scalp(C);build_lengths(C,mat);back_and_inner(C,mat);make_bangs(C,mat)
