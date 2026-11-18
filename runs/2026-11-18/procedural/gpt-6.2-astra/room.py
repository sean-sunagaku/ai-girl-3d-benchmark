"""Small physical café set, daylight rig and the five saved cameras."""
from atelier import *

# The set is built at metre scale; the procedural object helper accepts mm.
def box(name,center,size,mat,C,bevel=0):
    x,y,z=center;a,b,c=np.array(size)*.5
    vs=[(x+i*a,y+j*b,z+k*c) for i,j,k in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
    obj=mesh(name,np.array(vs)*1000,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],mat,C,smooth_shade=False)
    if bevel:
        m=obj.modifiers.new('Rounded furniture edge','BEVEL');m.width=bevel;m.segments=3
    return obj

def lathe(name,center,profile,mat,C,n=40):
    p=[];fs=[]
    for r,h in profile:
        for i in range(n):
            a=i*2*pi/n;p.append(((center[0]+r*cos(a))*1000,(center[1]+r*sin(a))*1000,(center[2]+h)*1000))
    for j in range(len(profile)-1):
        for i in range(n):
            k=j*n+i;l=j*n+(i+1)%n;fs.append((k,l,l+n,k+n))
    return mesh(name,p,fs,mat,C)

def wood():
    m,b=principle('Café | oiled oak',(.32,.147,.057),.43);tc=node(m,'ShaderNodeTexCoord','Timber coordinates',-800,140);stretch=node(m,'ShaderNodeVectorMath','Long grain',-590,140);stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(2,30,53);link(m,tc.outputs['Object'],stretch.inputs[0]);noise=node(m,'ShaderNodeTexNoise','Oak figure',-382,140);noise.inputs['Scale'].default_value=3.1;noise.inputs['Detail'].default_value=2.7;noise.inputs['Distortion'].default_value=.72;link(m,stretch.outputs[0],noise.inputs[0]);r=node(m,'ShaderNodeValToRGB','Honey and umber',-170,165);r.color_ramp.elements[0].color=(.14,.065,.025,1);r.color_ramp.elements[1].color=(.52,.295,.113,1);link(m,noise.outputs['Fac'],r.inputs[0]);link(m,r.outputs[0],b.inputs[0]);return m

def make_room(C):
    oak=wood();plaster,_=principle('Café | warm plaster',(.66,.55,.45),.89);metal,_=principle('Café | window bronze',(.069,.066,.051),.42,**{'Metallic':.37});ivory,_=principle('Café | cream porcelain',(.81,.75,.62),.24);olive,_=principle('Café | tea-glazed stoneware',(.22,.27,.177),.35);terracotta,_=principle('Café | cinnamon ceramic',(.37,.175,.105),.34);dark,_=principle('Café | coffee canister',(.050,.047,.038),.33,**{'Metallic':.27})
    pane,_=principle('Café | daylight through frosted glazing',(.77,.86,.88),.39,**{'Emission Color':(.84,.93,1,1),'Emission Strength':1.5})
    box('Limewash • rear wall',(0,1.52,1.52),(4,.075,3.4),plaster,C)
    box('Window • diffusing pane',(-.81,1.42,1.65),(1.5,.035,2.6),pane,C)
    for i,x in enumerate((-1.35,-.79,-.14)):
        box('Window • narrow upright '+str(i),(x,1.39,1.61),(.026,.045,2.55),metal,C,.003)
    box('Window • cross rail',(-.78,1.39,1.54),(1.32,.041,.017),metal,C,.002)
    for i,z in enumerate([1.39,1.68,1.99]):
        box('Oak shelving '+str(i),(.56,1.32,z),(1.24,.25,.029),oak,C,.004)
    rng=np.random.default_rng(396);mats=[ivory,olive,ivory,terracotta,dark,ivory]
    for k in range(19):
        row=k//7;i=k%7;z=[1.405,1.695,2.005][row];x=.002+i*.151+rng.uniform(-.022,.024);y=1.27+rng.uniform(-.03,.04);r=rng.uniform(.029,.048);h=rng.uniform(.073,.151)
        if k%4==0:prof=[(.78*r,0),(.88*r,.006),(r,.21*h),(r,.85*h),(.69*r,.92*h),(.69*r,h),(.04*r,h)]
        else:prof=[(.64*r,0),(.85*r,.009),(r,.86*h),(.98*r,h),(.86*r,h),(.74*r,.23*h),(.02*r,.22*h)]
        lathe('Shelf pottery '+str(k+1),(x,y,z),prof,mats[k%len(mats)],C,32)
    box('Side table',(.47,.59,1.115),(.58,.52,.037),oak,C,.012)
    box('Chair • curved-back rail',(.32,.69,1.135),(.48,.047,.083),oak,C,.025)
    for x in (.13,.51):box('Chair • oak stile',(x,.69,.89),(.035,.038,.54),oak,C,.008)
    box('Window banquette',(-.63,.75,.86),(.86,.29,.055),oak,C,.012)
    # Small silhouettes at the counter break up the rows of cups when defocused.
    chrome,_=principle('Café | brushed steel',(.49,.49,.44),.31,**{'Metallic':.80})
    box('Espresso machine • feet and tray',(.185,1.21,1.419),(.205,.168,.022),chrome,C,.007)
    box('Espresso machine • back',(.185,1.272,1.479),(.205,.046,.129),dark,C,.009)
    box('Espresso machine • boiler hood',(.185,1.215,1.550),(.219,.149,.064),chrome,C,.013)
    for j in [-1,1]:
        lathe('Espresso machine • portafilter '+str(j),(.185+j*.053,1.169,1.49),[(.016,0),(.022,.007),(.025,.025)],dark,C,24)
    vase,_=principle('Café | celadon vase',(.285,.475,.357),.22,**{'Coat Weight':.34})
    lathe('Table • bud vase',(.252,.62,1.134),[(.020,0),(.033,.013),(.039,.046),(.037,.077),(.014,.109),(.012,.145),(.008,.145)],vase,C)
    stem,_=principle('Café | flower stems',(.134,.233,.09),.70);petal,_=principle('Café | cream blossoms',(.83,.731,.554),.6);paths=[];widths=[];petals=[];flower_faces=[]
    for j in range(15):
        end=np.array((.252+rng.uniform(-.062,.073),.62+rng.uniform(-.034,.042),1.42+rng.uniform(-.055,.078)))*1000;start=np.array((.252,.62,1.24))*1000;curve=catmull([start,lerp(start,end,.63)+[0,0,12],end],np.linspace(0,1,7));paths.append(curve);widths.append(np.linspace(.85,.37,7))
        for k in range(5):
            a=2*pi*k/5+j;v0=len(petals);d=np.array((cos(a),sin(a),.55));cross=np.array((-sin(a),cos(a),.2));size=rng.uniform(3.6,6.7);petals.extend([end,end+d*size*.57+cross*2.1,end+d*size,end+d*size*.57-cross*2.1]);flower_faces.append((v0,v0+1,v0+2,v0+3))
    strands('Table flowers • stems',paths,widths,stem,C);mesh('Table flowers • petals',petals,flower_faces,petal,C)

    # Crassula-like branching foliage beyond the portrait focal plane.
    leafm=[]
    for i,c in enumerate([(.065,.157,.035),(.158,.263,.067),(.206,.32,.088),(.098,.194,.057)]):leafm.append(principle('Café | foliage '+str(i+1),c,.51,**{'Subsurface Weight':.075,'Subsurface Scale':.0006})[0])
    bark,_=principle('Café | slender branches',(.155,.108,.060),.76);pot,_=principle('Café | oatmeal planter',(.54,.47,.359),.76)
    lathe('Window planter',(-.48,.73,.89),[(.051,0),(.074,.018),(.092,.16),(.089,.173),(.080,.173),(.068,.135),(.009,.135)],pot,C)
    paths=[];rads=[];vv=[];ff=[];indices=[];base=np.array((-.48,.73,1.037))*1000
    for j in range(22):
        end=base+np.array((rng.uniform(-295,175),rng.uniform(-120,130),rng.uniform(355,870)));mid=lerp(base,end,.46)+[rng.uniform(-32,32),0,27]
        p=catmull([base,mid,end],np.linspace(0,1,10));paths.append(p);rads.append(np.linspace(3.1,.65,10))
        for h in range(3,10):
            pos=p[h];sg=(-1)**h;tip=pos+[sg*rng.uniform(25,91),rng.uniform(-33,33),rng.uniform(23,91)]
            paths.append(np.array([pos,lerp(pos,tip,.54)+[0,0,5],tip]));rads.append(np.array([1.15,.85,.29]))
            for q in range(3):
                pt=lerp(pos,tip,.42+.27*q);a=rng.uniform(0,2*pi);length=rng.uniform(28,50);wid=rng.uniform(10,20);direction=np.array([cos(a)*length,.58*sin(a)*length,rng.uniform(11,34)]);across=np.array([-sin(a)*wid,cos(a)*wid,.26*wid]);v0=len(vv);vv.extend([pt,pt+direction*.49-across,pt+direction*.54+[0,0,4],pt+direction*.49+across,pt+direction]);ff.extend([(v0,v0+1,v0+2),(v0,v0+2,v0+3),(v0+1,v0+4,v0+2),(v0+2,v0+4,v0+3)]);indices.extend([rng.integers(0,4)]*4)
    strands('Window plant • wood stems',paths,rads,bark,C);ob=mesh('Window plant • leaf blades',vv,ff,leafm[0],C);[ob.data.materials.append(m) for m in leafm[1:]];ob.data.polygons.foreach_set('material_index',np.array(indices,dtype=np.int32))

def light(C,name,loc,target,power,color,size,second=None):
    d=bpy.data.lights.new(name,'AREA');d.energy=power;d.color=color;d.shape='RECTANGLE';d.size=size;d.size_y=second or size
    o=bpy.data.objects.new(name,d);C.objects.link(o);o.location=loc;aim(o,target);return o

def make_rig(C):
    light(C,'Daylight • broad left window',(-.54,-.56,1.98),(-.055,-.09,1.48),20.0,(1,.949,.892),.48,.75)
    light(C,'Bounce • cream card',(.47,-.41,1.7),(-.04,-.09,1.49),4.3,(1,.87,.81),.48,.65)
    light(C,'Hair light • high rear window',(-.38,.27,1.94),(-.1,-.05,1.43),11,(1,.836,.673),.43,.62)
    light(C,'Back wall bounce',(.48,.23,2.36),(.13,1.38,1.54),43,(1,.922,.818),1.18,.83)
    light(C,'Eyes • small reflected window',(.045,-.95,1.78),(-.04,-.13,1.54),1.2,(.89,.949,1),.09,.12)
    w=bpy.data.worlds.new('Daylight studio');bpy.context.scene.world=w;w.use_nodes=True
    bg=w.node_tree.nodes.get('Background');bg.name='Room ambience';bg.inputs['Color'].default_value=(.71,.772,.827,1);bg.inputs['Strength'].default_value=.18
    plate=node_world=w.node_tree.nodes.new('ShaderNodeBackground');plate.name='Diagnostic background';plate.inputs[0].default_value=(.36,.393,.388,1);plate.inputs[1].default_value=.8;plate.location=(-120,-130)
    path=w.node_tree.nodes.new('ShaderNodeLightPath');path.location=(-480,-160)
    mix=w.node_tree.nodes.new('ShaderNodeMixShader');mix.location=(120,0);w.node_tree.links.new(path.outputs['Is Camera Ray'],mix.inputs[0]);w.node_tree.links.new(bg.outputs[0],mix.inputs[1]);w.node_tree.links.new(plate.outputs[0],mix.inputs[2]);w.node_tree.links.new(mix.outputs[0],w.node_tree.nodes.get('World Output').inputs[0])

def camera(C,name,loc,target):
    data=bpy.data.cameras.new(name);ob=bpy.data.objects.new(name,data);C.objects.link(ob);ob.location=loc;aim(ob,target);data.clip_start=.025;data.clip_end=25;return ob

def make_cameras(C):
    beauty=camera(C,'Portrait • window seat',(-.035,-.86,1.570),(-.067,-.03,1.369));beauty.data.lens=56.5;beauty.data.dof.use_dof=True;beauty.data.dof.focus_distance=(beauty.location-Vector((-.084,-.150,1.499))).length;beauty.data.dof.aperture_fstop=5.6;beauty.data.dof.aperture_blades=9
    for key,deg in [('front',0),('three-quarter',45),('side',90),('back',180)]:
        a=math.radians(deg);center=Vector((-.005,-.225,1.307));direction=Vector((sin(a),-cos(a),.045));ob=camera(C,'diagnostic • '+key,center+direction*2.25,center);ob.data.type='ORTHO';ob.data.ortho_scale=.792
    return beauty
