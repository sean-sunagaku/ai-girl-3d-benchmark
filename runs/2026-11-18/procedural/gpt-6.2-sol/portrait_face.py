"""Continuous skin sculpt, orbital geometry, expression and facial fibers."""
from math import atan2, cos, exp, pi, sin, sqrt
from random import Random

import bpy
import numpy as np
from mathutils import Euler, Vector
from mathutils.bvhtree import BVHTree

from portrait_common import (
    ROOT, ease, gauss, linear, mesh_object, mix, native_strands, node, noise_bump,
    point_color, read_obj, subdivide, subset, surface,
)

PIVOT = Vector((0., -.056, 1.437))
ROTATION = Euler((-.041, -.440, -.048), 'XYZ').to_matrix()
R = np.array(ROTATION)
CENTER = np.array(PIVOT)
EYE_Z = 1.5461
EYE_RADIUS = .01488


def pose(point, weight=1.):
    p = Vector(point)
    return p.lerp(PIVOT + ROTATION @ (p - PIVOT), weight)


def pose_array(path, weight=1.):
    path = np.asarray(path)
    return path + ((path-CENTER) @ R.T + CENTER - path) * weight


def lip_outline(x):
    """Original-space contours for vermilion pigment and the contact seam."""
    x = abs(x)
    t = min(x/.0258, 1.)
    seam = 1.47822 + .00075 * t - .00115 * sin(pi*t)**1.3
    upper = seam + (.0074 + .00165 * gauss(t, .29, .16) - .00085 * gauss(t, 0, .15)) * (1-t**1.66)
    lower = seam - .0107 * (1-t**1.8) * (1-.15*t)
    return lower, seam, upper


def morph(co, eye_x):
    x, y, z = co
    a = abs(x)
    sign = 1 if x >= 0 else -1
    front = ease(.065, .113, -y)
    eyelid = exp(-((a-eye_x)/.022)**4 - ((z-EYE_Z)/.019)**4) * front
    # Broader almond aperture, with room under the lower lid.
    x += sign * (.0009 + (a-eye_x)*.228) * eyelid
    z += ((z-EYE_Z)*.578 + .00145 - (a-eye_x)*.053) * eyelid
    y -= .00145 * eyelid

    # Soften the upper orbit, build the malar pad, and narrow the lower jaw.
    cheek = gauss(a, .043, .024)*gauss(co[2], 1.514, .024)*front
    y -= .0047*cheek
    x += sign * .0031*cheek
    y += .0016*gauss(a, .029, .028)*gauss(co[2], 1.572, .016)*front
    x -= sign*.0052*gauss(a, .055, .027)*gauss(co[2], 1.450, .023)*ease(.075,.105,-co[1])
    z += .0040*gauss(co[2],1.437,.021)*gauss(a,0,.043)*ease(.071,.115,-co[1])
    y -= .0019*gauss(co[2],1.447,.012)*gauss(a,0,.024)*front

    # Preserve the nasal folds and the recessed nostril topology.
    tip = gauss(co[2],1.510,.016)*gauss(a,0,.016)*ease(.142,.157,-co[1])
    x *= 1 - .105*gauss(co[2],1.508,.016)*gauss(a,0,.028)*front
    y += .0035*tip
    z += .0016*tip
    z += .0007*gauss(co[2],1.506,.012)*gauss(a,.012,.008)*front
    y -= .0012*gauss(co[2],1.532,.014)*gauss(a,0,.011)*front

    # The lips share the same facial mesh as the cheeks and chin.
    lip_zone = gauss(co[2],1.479,.018)*ease(.130,.152,-co[1])
    y += .00225*lip_zone*gauss(a,0,.019)
    z += (.0046*gauss(a,.024,.013)-.00105)*lip_zone
    z -= .89*(co[2]-1.47816)*gauss(co[2],1.47816,.0034)*gauss(a,0,.035)*front
    # Shorter proportions through the lower face, blended into the throat.
    jaw_shortening=.189*min(.116,max(0,1.549-co[2]))
    z+=jaw_shortening*ease(.060,.116,-co[1])*(1-.31*ease(.043,.076,a))*ease(1.407,1.435,co[2])
    # Very small left/right differences in the soft tissue.
    z += .00075*gauss(co[2],1.480,.013)*gauss(co[0],.022,.016)*front
    x += .0007*gauss(co[2],1.453,.023)*front
    neck = gauss(co[2],1.411,.031)*ease(-.105,-.055,co[1])
    x *= 1-.065*neck
    under_collar = 1-ease(1.347,1.397,co[2])
    x *= 1-.18*under_collar
    y += (.004 + (-.022-y)*.18)*under_collar
    radius=sqrt((x/.066)**2+((y+.019)/(.070 if y<-.019 else .058))**2)
    z -= .075*ease(.76,1.38,radius)*(1-ease(1.387,1.422,co[2]))
    return x,y,z


def skin_pigment(co, eye_x):
    x,y,z = co
    a = abs(x)
    forward = ease(.071,.130,-y)
    skin = np.array((.824,.599,.516))
    blush_mask = gauss(a,.044,.022) * gauss(z,1.516,.025)*forward
    orbital = gauss(a,eye_x,.022)*gauss(z,1.554,.012)*forward
    lower_eye = gauss(a,eye_x,.022)*gauss(z,1.536,.006)*forward
    nose_tip = gauss(a,0,.019)*gauss(z,1.506,.016)*ease(.135,.160,-y)
    pinna = ease(.065,.086,a)*gauss(z,1.531,.04)*ease(-.09,-.068,y)
    skin = mix(skin, np.array((.822,.400,.358)), .51*blush_mask + .19*pinna)
    skin = mix(skin, np.array((.657,.403,.336)), .23*orbital + .14*lower_eye)
    skin = mix(skin, np.array((.814,.465,.398)), .28*nose_tip)
    skin += .021*gauss(z,1.575,.045)*gauss(a,0,.048)
    brow_x=(a-.0135)/.046
    arch=1.5657+.0076*sin(pi*(.10+.84*brow_x))+.00135*brow_x-.0028*brow_x*brow_x
    brow_shape=gauss(z,arch,.00164)*ease(.011,.019,a)*(1-ease(.047,.060,a))*forward
    skin=mix(skin,np.array((.408,.248,.193)),.31*brow_shape)
    nostril = gauss(a,.0083,.0051)*gauss(z,1.503,.005)*ease(-.162,-.151,y)*ease(.129,.148,-y)
    skin=mix(skin,np.array((.646,.346,.280)),.27*nostril)
    skin += .007*sin(x*83+sin(z*117))*sin(z*160+y*20)
    low,seam,high = lip_outline(a)
    lip = ease(low-.0011, low+.00095,z) * (1-ease(high-.00065,high+.0013,z))
    lip *= ease(.142,.151,-y)*(1-ease(.0242,.0263,a))
    lipstick = mix(np.array((.784,.407,.369)),np.array((.757,.302,.286)),ease(seam-.003,seam+.006,z))
    lipstick += .034*gauss(z,seam-.005,.004)*gauss(a,0,.014)
    skin = mix(skin, lipstick, lip*.88)
    inner = gauss(z,seam,.00124)*ease(.022,.004,a)*ease(.141,.152,-y)
    skin = mix(skin,np.array((.299,.105,.093)), .47*inner)
    oral = gauss(z,seam,.0049)*ease(.024,.012,a)*ease(-.154,-.146,y)
    skin = mix(skin,np.array((.36,.146,.144)), .86*oral)
    freckle=(.42*gauss(a,.035,.030)*gauss(z,1.529,.015)+.21*gauss(a,0,.016)*gauss(z,1.522,.017))*forward
    return (*linear(np.clip(skin, 0., 1.)),1.), lip, freckle


def skin_material(colored=True):
    mat, nt, bsdf = surface("Skin | warm peach, perfusion and pores" if colored else "Skin | hands",
        (.84,.628,.543), .435, **{
            'Subsurface Weight':.12, 'Subsurface Radius':(1.,.47,.28),
            'Subsurface Scale':.008, 'Specular IOR Level':.34})
    coord = node(nt, 'ShaderNodeTexCoord', 'Meter-scale pore coordinates',(-720,-165))
    if colored:
        attr = node(nt,'ShaderNodeVertexColor','Hand-painted anatomical color masks',(-710,205))
        attr.layer_name = 'Skin tone'
        source = attr.outputs['Color']
    else:
        rgb = node(nt,'ShaderNodeRGB','Warm skin',(-710,205))
        rgb.outputs[0].default_value = mat.diffuse_color
        source = rgb.outputs[0]
    noise = node(nt,'ShaderNodeTexNoise','Quiet mottling and capillary color',(-710,20))
    noise.inputs['Scale'].default_value = 238
    noise.inputs['Detail'].default_value = 3
    nt.links.new(coord.outputs['Object'],noise.inputs['Vector'])
    ramp = node(nt,'ShaderNodeValToRGB','Restrained tone variation',(-460,80))
    ramp.color_ramp.elements[0].color = (.75,.72,.70,1)
    ramp.color_ramp.elements[1].color = (1.07,1.04,1.01,1)
    nt.links.new(noise.outputs['Fac'],ramp.inputs[0])
    mult = node(nt,'ShaderNodeMixRGB','Skin tone and mottling',(-203,246))
    mult.blend_type = 'MULTIPLY'
    mult.inputs[0].default_value = .48
    nt.links.new(source,mult.inputs[1])
    nt.links.new(ramp.outputs[0],mult.inputs[2])
    if colored:
        spread=node(nt,'ShaderNodeTexVoronoi','Sparse pigment flecks',(-705,-465))
        spread.inputs['Scale'].default_value=830
        nt.links.new(coord.outputs['Object'],spread.inputs['Vector'])
        specks=node(nt,'ShaderNodeMapRange','Speck radius falloff',(-466,-550))
        specks.inputs['From Min'].default_value=.075
        specks.inputs['From Max'].default_value=.23
        specks.inputs['To Min'].default_value=.79
        specks.inputs['To Max'].default_value=0
        nt.links.new(spread.outputs['Distance'],specks.inputs['Value'])
        location=node(nt,'ShaderNodeAttribute','Freckle veil',(-707,-670))
        location.attribute_name='Freckle veil'
        fader=node(nt,'ShaderNodeMath','Local freckle amount',(-208,-555))
        fader.operation='MULTIPLY'
        nt.links.new(specks.outputs[0],fader.inputs[0])
        nt.links.new(location.outputs['Fac'],fader.inputs[1])
        pigment=node(nt,'ShaderNodeMixRGB','Warm pigment flecks', (3,290))
        pigment.blend_type='MULTIPLY'
        pigment.inputs[2].default_value=(.52,.33,.25,1)
        nt.links.new(fader.outputs[0],pigment.inputs[0])
        nt.links.new(mult.outputs[0],pigment.inputs[1])
        nt.links.new(pigment.outputs[0],bsdf.inputs['Base Color'])
    else:
        nt.links.new(mult.outputs[0],bsdf.inputs['Base Color'])
    noise_bump(nt,bsdf,3650,.139,.00011,coord.outputs['Object'],'Fine pores')
    if colored:
        tint=node(nt,'ShaderNodeAttribute','Vermilion mask',(-416,-430))
        tint.attribute_name='Vermilion'
        gloss=node(nt,'ShaderNodeMapRange','Lip hydration',(-176,-430))
        gloss.inputs['To Min'].default_value=.435
        gloss.inputs['To Max'].default_value=.254
        nt.links.new(tint.outputs['Fac'],gloss.inputs['Value'])
        nt.links.new(gloss.outputs[0],bsdf.inputs['Roughness'])
        wet=node(nt,'ShaderNodeMath','Thin lip highlights',(34,-405))
        wet.operation='MULTIPLY'
        wet.inputs[1].default_value=.225
        nt.links.new(tint.outputs['Fac'],wet.inputs[0])
        nt.links.new(wet.outputs[0],bsdf.inputs['Coat Weight'])
        bsdf.inputs['Coat Roughness'].default_value=.21
    return mat


def visible_surface(tree, x, z, fallback=-.14):
    p = tree.ray_cast(Vector((x,-.42,z)),Vector((0,1,0)))[0]
    return p.y if p else fallback


def eye_surfaces(col, center, gaze, label):
    # The white, iris and pupil are one continuous radial globe, with a
    # shallow corneal profile and a local wet-coat lobe over the optical zone.
    gaze = Vector(gaze).normalized()
    right = Vector((0,0,1)).cross(gaze).normalized()
    up = gaze.cross(right).normalized()
    white, nt, bsdf = surface("Eye | moist sclera and radial iris "+label,
                              (.74,.688,.632),.287,**{
                              'IOR':1.38,'Specular IOR Level':.24,
                              'Subsurface Weight':.035,'Subsurface Scale':.0035})
    colors = node(nt,'ShaderNodeVertexColor','Iris stroma and limbus',(-365,155))
    colors.layer_name = 'Ocular pigment'
    nt.links.new(colors.outputs[0],bsdf.inputs['Base Color'])
    mask = node(nt,'ShaderNodeAttribute','Corneal tear film',(-365,-70))
    mask.attribute_name = 'Corneal gloss'
    nt.links.new(mask.outputs['Fac'],bsdf.inputs['Coat Weight'])
    bsdf.inputs['Coat Roughness'].default_value=.075
    bsdf.inputs['Coat IOR'].default_value=1.38
    pupil=.00264
    iris=.00668
    rim=EYE_RADIUS
    rings = [atan2(r, sqrt(rim*rim-r*r)) for r in np.linspace(.00016,.0071,50)]
    rings += list(np.linspace(rings[-1]+.038,pi-.046,33))
    n = 144
    verts, faces, pigments, coat = [], [], [], []
    verts.append(pose(center + gaze*(rim+.00069)))
    pigments.append((*linear((.061,.029,.022)),1))
    coat.append(.83)
    for j, angle in enumerate(rings):
        rd = rim*sin(angle)
        axial = rim*cos(angle) + .00069*(1-ease(.0,iris+.0011,rd)) if angle < pi/2 else rim*cos(angle)
        for i in range(n):
            a = 2*pi*i/n
            stretch=1+.219*ease(.00645,.012,rd)
            verts.append(pose(center + axial*gaze + rd*(cos(a)*right*stretch + sin(a)*up)))
            if angle < pi/2 and rd < iris:
                f = ((1 + sin(73*a + 4*rd/iris))*.17
                     + sin(37*a + 12*rd/iris + .3*sin(7*a))*.14
                     + sin(111*a + 9*rd/iris)*.092
                     + sin(19*a + 13*rd/iris)*.11
                     + sin(13*a + 9*rd/iris)*.14
                     + sin(48*a-25*rd/iris)*.085)
                base=np.array((.190,.085,.048)) + f*np.array((.144,.093,.055))
                collar = gauss(rd,.0036,.00112)
                base += collar * np.array((.047,.037,.024))
                base *= 1-.58*ease(iris-.00084,iris+.00005,rd)
                base = mix(np.array((.044,.023,.020)),base,ease(pupil-.0001,pupil+.00028,rd))
                color = mix(base,np.array((.794,.750,.715)),ease(iris-.00010,iris+.00010,rd))
            else:
                vein = ease(.008,.014,rd) * max(0,sin(a*10+2.5*sin(a*18+rd*250)))**24
                edge = ease(.007,.018,rd)*.25+vein*.090
                color = mix(np.array((.739,.684,.647)),np.array((.749,.447,.418)),edge)
                if angle < pi/2:
                    color = mix(np.array((.733,.687,.644)),color,ease(iris,iris+.00054,rd))
            pigments.append((*linear(color),1.))
            coat.append(.14+.34*(1-ease(.0060,.0076,rd)) if angle<pi/2 else .16)
        if j == 0:
            faces.extend((0,1+i,1+(i+1)%n) for i in range(n))
        else:
            a0=1+(j-1)*n
            b0=1+j*n
            faces.extend((a0+i,b0+i,b0+(i+1)%n,a0+(i+1)%n) for i in range(n))
    end=len(verts)
    verts.append(pose(center-gaze*rim))
    pigments.append((*linear((.74,.66,.62)),1.))
    coat.append(.16)
    a0=end-n
    faces.extend((a0+i,end,a0+(i+1)%n) for i in range(n))
    eyeball = mesh_object("Globe, iris and corneal dome • "+label,col,verts,faces,white)
    point_color(eyeball.data,'Ocular pigment',pigments)
    mask = eyeball.data.attributes.new(name='Corneal gloss',type='FLOAT',domain='POINT')
    mask.data.foreach_set('value',coat)


def aperture(tree, sign, cx):
    """Take sections through the subdivided socket to attach fibers to its lips."""
    gap=[]
    for a in np.linspace(cx-.0175,cx+.0175,87):
        x=sign*a
        values=np.linspace(1.536,1.562,100)
        open_z=[z for z in values if visible_surface(tree,x,z,fallback=-.104)>-.122]
        if open_z:
            gap.append((x,min(open_z),max(open_z)))
    return np.array(gap)


def facial_fibers(col, tree, cx):
    rng=Random(3754)
    lash, _, _ = surface("Lashes | espresso",(.108,.048,.033),.37)
    brow, _, _ = surface("Brow hairs | cool walnut",(.203,.116,.080),.52)
    fine, _, _ = surface("Lower lashes | light chestnut",(.344,.201,.145),.49)
    wet, _, _ = surface("Lower waterline | rose",(.548,.283,.249),.344,
                       **{'Subsurface Weight':.087, 'Subsurface Scale':.002})
    upper_ink,_,_=surface("Upper lash roots | deep brown",(.235,.109,.083),.47)
    lash_paths, lash_widths, low_paths, low_widths, brows, brow_widths = [],[],[],[],[],[]
    liner_paths, liner_widths, rims, rim_widths = [],[],[],[]
    for sign in (-1,1):
        edge = aperture(tree,sign,cx)
        if len(edge)<2:
            continue
        for upper in (True,False):
            colz=2 if upper else 1
            xs=np.interp(np.linspace(.06,.97,55 if upper else 19), np.linspace(0,1,len(edge)),edge[:,0])
            zs=np.interp(np.linspace(.06,.97,len(xs)), np.linspace(0,1,len(edge)),edge[:,colz])
            liner=[]
            for j,(x,z) in enumerate(zip(xs,zs)):
                t=j/(len(xs)-1)
                offset=.00114 if upper else -.00084
                y = visible_surface(tree,x,z+offset)-(.00058 if upper else .00029)
                z += .00028 if upper else -.00029
                liner.append(pose((x,y+.00019,z)))
                for _ in range(2 if upper and t>.35 else 1):
                    h = (rng.uniform(.0033,.0071)*(.55+.47*t)) if upper else rng.uniform(.0012,.003)
                    f = sign*h*(-.034+.82*t)
                    c = .72 if upper else -1
                    origin = np.array((x+sign*rng.uniform(-.00031,.00031),y-.00003,z))
                    points=[]
                    for k in np.linspace(0,1,7):
                        points.append(origin+np.array((f*k*k,-h*(1.12*k-.41*k*k),h*c*(-.10*k+.98*k*k))))
                    (lash_paths if upper else low_paths).append(pose_array(points))
                    r0=rng.uniform(.000068,.000110) if upper else rng.uniform(.000032,.000052)
                    (lash_widths if upper else low_widths).append(r0*np.linspace(1.0,.08,7))
            if upper:
                liner_paths.append(np.asarray(liner))
                liner_widths.append(.00049*np.sin(np.linspace(.12,pi-.12,len(liner)))**.45)
            else:
                rims.append(np.asarray(liner))
                rim_widths.append(.00027*np.sin(np.linspace(.15,pi-.15,len(liner)))**.62)
        for j in range(465):
            t=rng.random()
            a=.0135+t*.046
            x=sign*a
            envelope=sin(pi*min(.999, .125+.875*t))**.54
            z=1.5650+.0076*sin(pi*(.10+.84*t))+.00135*t-.0028*t*t
            z+=rng.gauss(0,.00131)*envelope
            length=rng.uniform(.003,.0063)*(.92-.47*t)
            points=[]
            for k in np.linspace(0.,1.,7):
                dx=sign*length*(.21+.76*t)*k
                dz=length*(.92-1.16*t)*k - .00064*k*k
                front=visible_surface(tree,x+dx,z+dz)
                points.append(pose((x+dx,front-.00005-.00022*sin(pi*k),z+dz)))
            brows.append(np.asarray(points))
            brow_widths.append(rng.uniform(.000054,.000089)*np.linspace(1,.08,7))
    for name,paths,widths,mat in (
        ('Upper lashes • individually curled',lash_paths,lash_widths,lash),
        ('Lower lashes • individually curled',low_paths,low_widths,fine),
        ('Upper lash beds',liner_paths,liner_widths,upper_ink),
        ('Lower tear margins',rims,rim_widths,wet),
        ('Brows • feathered strands',brows,brow_widths,brow),
    ):
        if paths:
            native_strands(name,col,paths,widths,mat)


def lip_contact(col,tree,eye_x):
    """The lips meet over a shallow, softly pigmented inner lip surface."""
    mat,_,_=surface('Lip contact | muted warm rose',(.564,.246,.217),.40)
    v=[]
    for x in np.linspace(-.0244,.0244,85):
        _,_,estimate=morph((x,-.153,1.47816),eye_x)
        scan=np.linspace(estimate-.0027,estimate+.0027,61)
        depths=[visible_surface(tree,x,z) for z in scan]
        opening=[z for z,y in zip(scan,depths) if y>-.133]
        low=(min(opening) if opening else estimate)-.00058
        high=(max(opening) if opening else estimate)+.00058
        surface_y=min(visible_surface(tree,x,low-.0005),visible_surface(tree,x,high+.0005))
        y=surface_y+.00135
        v.extend((pose((x,y,low)),pose((x,y,high))))
    f=[(i,i+2,i+3,i+1) for i in range(0,len(v)-2,2)]
    mesh_object('Lip contact • recessed closure',col,v,f,mat)


def build_face(col, env, camera_location):
    verts,faces = read_obj(ROOT/'input/upper-body.obj')
    # Keep the neck connected to the skull, under the fabric collar.
    faces = [f for f in faces if all(verts[i][2] > 1.294 and abs(verts[i][0]) < .130 for i in f)]
    verts,faces = subset(verts,faces)
    landmark = env['landmarks']['joint-l-eye']
    eye_x = landmark[0]
    material = skin_material()
    morphed = [morph(v,eye_x) for v in verts]
    head = mesh_object("Face, ears and neck • continuous sculpt",col,morphed,faces,material)
    colors,lips,freckles=zip(*(skin_pigment(v,eye_x) for v in verts))
    point_color(head.data,'Skin tone',colors)
    paint=head.data.attributes.new(name='Vermilion',type='FLOAT',domain='POINT')
    paint.data.foreach_set('value',lips)
    flecks=head.data.attributes.new(name='Freckle veil',type='FLOAT',domain='POINT')
    flecks.data.foreach_set('value',freckles)
    subdivide(head,2,'Continuous facial surface')
    bpy.context.view_layer.update()
    tree = BVHTree.FromObject(head,bpy.context.evaluated_depsgraph_get())
    # Keep bind space available for the strand root and crease attachments.
    cx=eye_x+.0013
    for sign,label in ((-1,'image left'),(1,'image right')):
        center = Vector((sign*cx,landmark[1]-.0032,EYE_Z+.0018))
        aim_at = PIVOT+ROTATION.transposed()@(Vector(camera_location)-PIVOT)
        gaze=(aim_at-center).normalized()+Vector((0,-.39,-.005))
        eye_surfaces(col,center,gaze,label)
    facial_fibers(col,tree,cx)
    lip_contact(col,tree,eye_x)
    for vert,p in zip(head.data.vertices,morphed):
        vert.co = pose(p,ease(1.357,1.464,p[2]))
    head.data.update()
    return head,tree
