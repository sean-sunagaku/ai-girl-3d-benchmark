import sys, os, time
sys.path.insert(0, os.getcwd())
import bpy, numpy as np
from fable.base import Body
bpy.ops.wm.read_factory_settings(use_empty=True)
t=time.time(); b=Body('input'); print('BODY', len(b.V), len(b.F), 'enclosed', int(b.enclosed.sum()), 'upper', b.n_upper_faces, 'arm', b.n_arm_faces, 't=%.2f'%(time.time()-t))
ob=b.make_object('Body')
L=b.landmarks
arm=bpy.data.armatures.new('Rig'); rig=bpy.data.objects.new('Rig',arm); bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active=rig; bpy.ops.object.mode_set(mode='EDIT')
def bone(name,a,b_,parent=None,connect=False):
    eb=arm.edit_bones.new(name); eb.head=tuple(a); eb.tail=tuple(b_)
    if parent: eb.parent=arm.edit_bones[parent]; eb.use_connect=connect
    return eb
J=lambda k:L['joint-'+k]
bone('spine2',J('spine-3'),J('spine-2'))
bone('spine1',J('spine-2'),J('spine-1'),'spine2',True)
bone('chest',J('spine-1'),J('neck'),'spine1',True)
bone('neck',J('neck'),J('head'),'chest',True)
bone('head',J('head'),J('head-2'),'neck',True)
for s,sg in (('l',1),('r',-1)):
    bone('clav.'+s,J(s+'-clavicle'),J(s+'-shoulder'),'chest')
    bone('upperarm.'+s,J(s+'-shoulder'),J(s+'-elbow'),'clav.'+s,True)
    bone('forearm.'+s,J(s+'-elbow'),J(s+'-hand'),'upperarm.'+s,True)
    bone('hand.'+s,J(s+'-hand'),J(s+'-hand-2'),'forearm.'+s,True)
    for f in range(1,6):
        par='hand.'+s
        for k in range(1,4):
            nm='f%d%d.%s'%(f,k,s); bone(nm,J('%s-finger-%d-%d'%(s,f,k)),J('%s-finger-%d-%d'%(s,f,k+1)),par,k>1); par=nm
bpy.ops.object.mode_set(mode='OBJECT')
print('BONES',len(arm.bones))
bpy.ops.object.select_all(action='DESELECT'); ob.select_set(True); rig.select_set(True); bpy.context.view_layer.objects.active=rig
t=time.time(); r=bpy.ops.object.parent_set(type='ARMATURE_AUTO'); print('PARENT',r,'t=%.2f'%(time.time()-t))
print('VGROUPS',len(ob.vertex_groups),[m.type for m in ob.modifiers])
# weight stats
W=np.zeros((len(b.V),len(ob.vertex_groups)))
for v in ob.data.vertices:
    for g in v.groups: W[v.index,g.group]=g.weight
print('unweighted verts',int((W.sum(1)<1e-6).sum()))
names=[g.name for g in ob.vertex_groups]
for nm in ('head','neck','chest','clav.l','upperarm.l','forearm.l','hand.l','f21.l','f23.l'):
    j=names.index(nm); print(' ',nm,'n>0.5:',int((W[:,j]>0.5).sum()),'max',round(float(W[:,j].max()),3))
