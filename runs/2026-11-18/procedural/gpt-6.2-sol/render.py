"""Render camera views directly from the reopened character.blend."""
import argparse
import sys
from pathlib import Path

import bpy

ROOT=Path(__file__).resolve().parent
VIEWS=('beauty','front','three-quarter','side','back')


def render_view(label,out,preview):
    scene=bpy.context.scene
    beauty=label=='beauty'
    bpy.data.collections['Café set and beauty lighting'].hide_render=not beauty
    studio=bpy.data.collections['Neutral orbit lighting']
    studio.hide_render=beauty
    if not beauty:
        studio.hide_viewport=False
    scene.camera=bpy.data.objects['Beauty • café selfie' if beauty else 'diagnostic • '+label]
    world=scene.world.node_tree
    bg=world.nodes['Background']
    bg.inputs['Color'].default_value=(.52,.47,.42,1) if beauty else (.64,.615,.588,1)
    bg.inputs['Strength'].default_value=.25 if beauty else .40
    scene.view_settings.exposure=-.37 if beauty else -.38
    scene.render.resolution_x=960 if beauty else 768
    scene.render.resolution_y=1040 if beauty else 768
    scene.render.resolution_percentage=50 if preview else 100
    scene.cycles.samples=13 if preview else 32
    scene.cycles.device='CPU'
    scene.render.threads_mode='FIXED'
    scene.render.threads=3
    scene.cycles.use_denoising=True
    bpy.context.view_layer.update()
    scene.render.filepath=str(out/(label+'.png'))
    bpy.ops.render.render(write_still=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('views',nargs='*',choices=VIEWS,default=None)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    out=ROOT/('.qa/previews' if args.preview else 'renders')
    out.mkdir(parents=True,exist_ok=True)
    if not bpy.data.filepath or Path(bpy.data.filepath).name!='character.blend':
        raise RuntimeError('Open character.blend before running render.py')
    for label in args.views or VIEWS:
        render_view(label,out,args.preview)


if __name__=='__main__':
    main()
