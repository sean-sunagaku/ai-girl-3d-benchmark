"""Coordinator-side collection and technical validation for the 2026-11-10 Claude Fable 5.2 run.

Run from the repository root, e.g.

    python3 runs/2026-11-10/harness/coordinator_validate.py --stage procedural \
        --workspace .work/fable-5.2-procedural --rerun .work/fable-5.2-rerun-procedural

It never edits the entrant's scene: the workspace is copied, the copy of character.blend is only opened
(scripts/inspect_scene.py, no save) and an independent rebuild happens in a third directory.
This is a technical check, not an appearance score.
"""
import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / 'benchmarks' / 'cafe-selfie-v1'
RUN = ROOT / 'runs' / '2026-11-10'
BLENDER = '/Applications/Blender.app/Contents/MacOS/Blender'
EXPECTED = {'beauty.png': (960, 1040), 'front.png': (768, 768), 'three-quarter.png': (768, 768), 'side.png': (768, 768), 'back.png': (768, 768)}
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '*.blend1', '*.log', '.DS_Store', '.env', '.env.*', 'source-character.blend', 'dbg_*')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def png_info(path):
    d = open(path, 'rb').read()
    assert d[:8] == b'\x89PNG\r\n\x1a\n', path
    w, h = struct.unpack('>II', d[16:24])
    i = 8; text = {}; idat = hashlib.sha256()
    while i < len(d):
        n, t = struct.unpack('>I4s', d[i:i + 8]); c = d[i + 8:i + 8 + n]
        if t == b'tEXt':
            k, _, v = c.partition(b'\x00'); text[k.decode('latin1')] = v.decode('latin1')
        elif t == b'IDAT':
            idat.update(c)
        i += 12 + n
    return dict(width=w, height=h, text_chunks=text, idat_sha256=idat.hexdigest())


def personal_patterns():
    pats = {getpass.getuser(), str(Path.home())}
    for key in ('user.name', 'user.email'):
        try:
            v = subprocess.run(['git', 'config', key], capture_output=True, text=True, cwd=ROOT).stdout.strip()
            if v:
                pats.add(v)
                if '@' in v:
                    pats.add(v.split('@')[0])
        except Exception:
            pass
    return sorted(p for p in pats if len(p) >= 4)


def scan_privacy(folder):
    """Search every collected file (and the decompressed .blend payload) for personal strings and secrets."""
    pats = personal_patterns()
    generic = [rb'/Users/[A-Za-z0-9_.-]+', rb'/home/[A-Za-z0-9_.-]+', rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
               rb'\b(?:\d{1,3}\.){3}\d{1,3}\b', rb'(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*[\'"][^\'"]{6,}']
    allow = [rb'noreply@anthropic\.com', rb'/Applications/Blender\.app']
    hits = []; scanned = 0
    for path in sorted(Path(folder).rglob('*')):
        if not path.is_file():
            continue
        data = path.read_bytes(); scanned += 1
        payloads = [('raw', data)]
        if path.suffix == '.blend' and data[:4] == b'\x28\xb5\x2f\xfd':
            out = subprocess.run(['zstd', '-dc', str(path)], capture_output=True)
            if out.returncode == 0:
                payloads.append(('zstd-decompressed', out.stdout))
        binary = path.suffix in ('.blend', '.glb', '.png', '.jpg', '.obj')
        for label, blob in payloads:
            for p in pats:
                if p.encode() in blob:
                    hits.append(dict(file=str(path.relative_to(folder)), payload=label, kind='personal-string'))
            for g in generic[:2] + ([] if binary else generic[2:]):
                for m in re.finditer(g, blob):
                    s = m.group(0)
                    if any(re.search(a, s) for a in allow):
                        continue
                    if g == generic[3] and not path.suffix in ('.md', '.json', '.py', '.txt'):
                        continue
                    hits.append(dict(file=str(path.relative_to(folder)), payload=label, kind='pattern', pattern=g.decode('latin1')[:40]))
                    break
    return dict(files_scanned=scanned, personal_strings_checked=len(pats), hits=hits, clean=not hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['procedural', 'photo-textured'], required=True)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--rerun', type=Path, required=True)
    ap.add_argument('--entrant', default='fable-5.2')
    ap.add_argument('--include', default='Character')
    ap.add_argument('--skip-rerun', action='store_true')
    a = ap.parse_args()
    ws = (ROOT / a.workspace).resolve(); rr = (ROOT / a.rerun).resolve()
    dest = RUN / a.stage / a.entrant
    out = dict(stage=a.stage, entrant=a.entrant, checked_on_jst=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               appearance_scoring=False, note='Technical validation by the coordinator; not an appearance evaluation.')

    # 1. shared inputs against the manifest, workspace copies against the shared inputs
    manifest = json.loads((CASE / 'manifest.json').read_text())
    out['manifest_checks'] = [dict(path=f['path'], sha256_matches=sha(CASE / f['path']) == f['sha256']) for f in manifest['files']]
    prompt = '01-procedural.md' if a.stage == 'procedural' else '02-photo-textured.md'
    ws_checks = [dict(path='PROMPT.md', byte_identical_to='prompts/' + prompt, matches=(ws / 'PROMPT.md').read_bytes() == (CASE / 'prompts' / prompt).read_bytes())]
    for f in sorted((CASE / 'inputs').iterdir()):
        ws_checks.append(dict(path='input/' + f.name, matches=(ws / 'input' / f.name).read_bytes() == f.read_bytes()))
    out['workspace_input_checks'] = ws_checks

    # 2. collect the submission as-is
    previous = {}
    if (dest / 'validation.json').is_file():
        previous = json.loads((dest / 'validation.json').read_text())
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ws, dest, ignore=IGNORE)
    blend = dest / 'character.blend'
    out['collected'] = dict(path=str(dest.relative_to(ROOT)), files=sorted(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()),
                            character_blend=dict(bytes=blend.stat().st_size, sha256=sha(blend), identical_to_workspace=sha(blend) == sha(ws / 'character.blend')))
    out['original_renders'] = {}
    for name, size in EXPECTED.items():
        info = png_info(dest / 'renders' / name)
        out['original_renders'][name] = dict(sha256=sha(dest / 'renders' / name), **info, expected=list(size), dimensions_ok=(info['width'], info['height']) == size)

    # 3. independent rebuild in a separate directory: only the code and inputs that are also published
    #    (no previous outputs, and not the entrant's unpublished source-character.blend)
    if not a.skip_rerun:
        if rr.exists():
            shutil.rmtree(rr)
        rr.mkdir(parents=True)
        for item in ws.iterdir():
            if item.name in ('renders', 'character.blend', 'scene-report.json', 'previews', 'character.glb', 'validation.json',
                             'source-character.blend') or item.suffix in ('.blend1', '.log'):
                continue
            if item.is_dir():
                shutil.copytree(item, rr / item.name, ignore=IGNORE)
            else:
                shutil.copy2(item, rr / item.name)
        t0 = time.time()
        proc = subprocess.run([BLENDER, '--background', '--factory-startup', '--threads', '3', '--python', 'build.py'], cwd=rr, capture_output=True, text=True)
        (rr / 'coordinator-build.log').write_text(proc.stdout[-200000:] + '\n--- stderr ---\n' + proc.stderr[-20000:])
        rebuilt = {}
        for name, size in EXPECTED.items():
            p = rr / 'renders' / name
            if p.is_file():
                info = png_info(p)
                rebuilt[name] = dict(sha256=sha(p), width=info['width'], height=info['height'], dimensions_ok=(info['width'], info['height']) == size,
                                     byte_identical_to_original=sha(p) == out['original_renders'][name]['sha256'],
                                     pixel_data_identical_to_original=info['idat_sha256'] == out['original_renders'][name]['idat_sha256'])
            else:
                rebuilt[name] = dict(missing=True)
        out['independent_rebuild'] = dict(command='Blender --background --factory-startup --threads 3 --python build.py', exit_code=proc.returncode,
                                          seconds=round(time.time() - t0, 1), produced_character_blend=(rr / 'character.blend').is_file(),
                                          produced_scene_report=(rr / 'scene-report.json').is_file(), renders=rebuilt,
                                          all_five_expected_dimensions=all(v.get('dimensions_ok') for v in rebuilt.values()))

    elif 'independent_rebuild' in previous:
        out['independent_rebuild'] = previous['independent_rebuild']      # keep the earlier rebuild record

    # 4. re-open the collected scene (no save) + character-only GLB
    insp = dest / 'validation.inspect.json'; glb = dest / 'character.glb'
    proc = subprocess.run([BLENDER, '--background', str(blend), '--threads', '3', '--python', str(ROOT / 'scripts' / 'inspect_scene.py'), '--',
                           '--output', str(insp), '--glb', str(glb), '--include', a.include], capture_output=True, text=True)
    out['inspect_scene'] = dict(exit_code=proc.returncode, script='scripts/inspect_scene.py', include=a.include)
    if insp.is_file():
        out['scene'] = json.loads(insp.read_text()); insp.unlink()
    out['collected']['character_blend']['sha256_after_inspection'] = sha(blend)
    out['collected']['character_blend']['unchanged_by_inspection'] = sha(blend) == out['collected']['character_blend']['sha256']

    # 5. the exported GLB must load again
    if glb.is_file():
        code = ("import bpy,json,sys;bpy.ops.wm.read_factory_settings(use_empty=True);"
                "bpy.ops.import_scene.gltf(filepath=sys.argv[-1]);"
                "print('GLBCHECK',json.dumps(dict(meshes=len([o for o in bpy.data.objects if o.type=='MESH']),"
                "vertices=sum(len(o.data.vertices) for o in bpy.data.objects if o.type=='MESH'),"
                "materials=len(bpy.data.materials),images=len([i for i in bpy.data.images if i.size[0]>0]))))")
        proc = subprocess.run([BLENDER, '--background', '--factory-startup', '--python-expr', code, '--', str(glb)], capture_output=True, text=True)
        m = re.search(r'GLBCHECK (\{.*\})', proc.stdout)
        head = glb.read_bytes()[:12]
        out['glb'] = dict(path='character.glb', bytes=glb.stat().st_size, sha256=sha(glb), container_magic_ok=head[:4] == b'glTF',
                          container_version=struct.unpack('<I', head[4:8])[0], reimport=json.loads(m.group(1)) if m else None, reimport_succeeded=bool(m))

    # 6. privacy / secret scan of everything collected
    out['privacy_scan'] = scan_privacy(dest)
    (dest / 'validation.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    summary = dict(manifest_ok=all(c['sha256_matches'] for c in out['manifest_checks']), workspace_ok=all(c['matches'] for c in ws_checks),
                   rebuild=out.get('independent_rebuild', {}).get('exit_code'), inspect=out['inspect_scene']['exit_code'],
                   blend_unchanged=out['collected']['character_blend']['unchanged_by_inspection'],
                   glb=out.get('glb', {}).get('reimport_succeeded'), privacy_clean=out['privacy_scan']['clean'])
    print('VALIDATION', json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
