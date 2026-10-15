p='build.py';s=open(p).read()
s=s.replace("12 degree", "20 degree")
s=s.replace("default_value=.055;p.inputs['Subsurface Radius']", "default_value=.16;p.inputs['Subsurface Radius']")
s=s.replace("default_value=.010\n", "default_value=.0028\n")
s=s.replace("(.72,.72,.64)","(.64,.64,.58)")
s=s.replace("iris_surface('Radial brown iris '+label,.00245", "iris_surface('Radial brown iris '+label,.00305")
s=s.replace("iris_surface('Deep pupil '+label,0,.0025", "iris_surface('Deep pupil '+label,0,.0031")
s=s.replace("z=1.374+.023*q;rx=.080-.026*q;ry=.100-.031*q", "z=1.375+.017*q;rx=.125-.071*q;ry=.094-.026*q")
s=s.replace("rx=.080-.026*q;ry=.100-.031*q", "rx=.125-.071*q;ry=.094-.026*q")
s=s.replace("1.374+.023*q", "1.375+.017*q")
s=s.replace("locks.append((catmull(points,30),random.uniform(.005,.010),s,k))", "path=catmull(points,30)\n  for v in path:\n   if v.z<1.41:\n    hit=cloth_bvh.ray_cast(Vector((v.x,-.45,v.z)),Vector((0,1,0)))\n    if hit[0]:v.y=min(v.y,hit[0].y-.018)\n  locks.append((path,random.uniform(.005,.010),s,k))")
s=s.replace("(s*(.010+.018*a),-.158,1.608),(s*(.013+.036*a),-.160,1.590)","(s*(.022+.023*a),-.158,1.608),(s*(.029+.026*a),-.160,1.590)")
marker="# Wispy curtain bangs: swept outside the facial surface, distinct from long side locks."
extra="""# Additional individual frontal crown fibers prevent a smooth helmet surface.
paths=[]
for k in range(360):
 th=random.uniform(-2.75,-.39);fr=max(0,-math.sin(th));end=1.83-.75*fr**2+random.uniform(-.015,.035);start=random.uniform(.08,.40);arr=[]
 for j in range(24):
  q=j/23;ph=start+(end-start)*q;theta=th+.10*math.sin(q*math.pi)
  v=Vector((.0922*math.sin(ph)*math.cos(theta),-.043+.1244*math.sin(ph)*math.sin(theta),1.55+.1384*math.cos(ph)))
  arr.append((*v,.72*(1-q**8)+.03))
 paths.append(arr)
curves('Hair | fine frontal crown fibers',paths,.000125,hairm[1])
"""
s=s.replace(marker,extra+marker)
s=s.replace("angle=math.radians(-12);rot=Matrix.Rotation(angle,4,'Y');pivot=head.location.copy()", "angle=math.radians(-20);rot=Matrix.Rotation(angle,4,'Y');pivot=head.location.copy()\n# Long hair follows gravity: root tilt fades into the unrotated hanging lengths.\nfor o in list(collections['03 Hair'].objects):\n if o.type not in ('MESH','CURVE'):continue\n points=o.data.vertices if o.type=='MESH' else [p for sp in o.data.splines for p in sp.points]\n for p in points:\n  v=Vector(p.co[:3]);w=max(0,min(1,(v.z-1.28)/.18));v=pivot+Matrix.Rotation(-angle*(1-w),4,'Y')@(v-pivot)-Matrix.Rotation(-angle,4,'Y')@Vector((.025*(1-w),0,0))\n  if o.type=='MESH':p.co=v\n  else:p.co=(*v,1)")
s=s.replace("v.co=pivot+r@(v.co-pivot)", "v.co=pivot+r@(v.co-pivot)+Vector((.025*w,0,0))")
s=s.replace("head.rotation_euler[1]=angle", "head.rotation_euler[1]=angle;head.location.x+=.025")
# Actual twig stems and a pot anchor the otherwise distant geometric cafe leaves.
s=s.replace("for k in range(40):", "leafstems=[]\nfor k in range(40):")
s=s.replace("if o in CHAR:CHAR.remove(o)", "if o in CHAR:CHAR.remove(o)\n leafstems.append([(-.54,.59,1.13,.8),(x*.5-.27,(y+.59)*.5,z-.13,.6),(x,y,z,.15)])")
s=s.replace("# Cameras and lights.", "stems=curves('Cafe | plant stems',leafstems,.0012,green,'06 Cafe',False);CHAR.remove(stems)\nbpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=.14,depth=.23,location=(-.54,.59,1.02));o=bpy.context.object;o.name='Cafe | ceramic planter';o.data.materials.append(ceramic);place(o,'06 Cafe',False)\n# Cameras and lights.")
open(p,'w').write(s)
