#!/usr/bin/env python3
"""Benchmark harness for the fx styles: fixed inputs, frozen renders, a change check.

  bench.py render DIR [--fx FILE] [--styles glow,streak,dust] [--set work|held|all]
      frame 0 of each style on each benchmark input, as DIR/<style>_<name>.png,
      plus DIR/hashes.json (a hash of the first frames of every clip)
  bench.py freeze     render the repo's renderer and store its hashes as the baseline
  bench.py check      render again and list every style/input whose frames changed
  bench.py sheet OUT.png DIR [DIR...] [--style glow] [--set ...]
      contact sheet: INPUT | one column per DIR

The inputs are not in the repo: put <name>.540.png files (540 x 540, opaque on black or
with alpha) in out/bench/inputs, or point PHOSPHOR_BENCH at a folder of them.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
FX = ROOT / 'tools' / 'phosphor_fx.py'
INPUTS = Path(os.environ.get('PHOSPHOR_BENCH', ROOT / 'out' / 'bench' / 'inputs'))
BASELINE = Path(__file__).with_name('baseline.json')
SEED, FRAMES, HASHED = 7, 8, 4

# A change must win on both: WORK is what it is developed on, HELD is only looked at to confirm.
WORK = ['close_helmet', 'helmet_dragon', 'armor_suit', 'helmet', 'dragon_bronze', 'magdalen_flame',
        'artemis_rocket', 'eclipse2017', 'solar_prominence', 'dragon_woodcut', 'angel_wings', 'orrery']
HELD = ['burgonet', 'gauntlet', 'sword_met', 'skull_met', 'neon_sign', 'saturn_smiled', 'lion_durer',
        'dore_satan']
SETS = {'work': WORK, 'held': HELD, 'all': WORK + HELD}


def load(path: Path):
    sys.path.insert(0, str(ROOT / 'tools'))
    spec = importlib.util.spec_from_file_location('fx_' + hashlib.md5(str(path).encode()).hexdigest()[:8], path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def clip(fx, style: str, name: str):
    P = fx.params(style, frames=FRAMES)
    im = Image.open(INPUTS / f'{name}.540.png').convert('RGBA')
    if im.width != P.long:
        im = im.resize((P.long, P.long), Image.LANCZOS)
    return fx.STYLES[style][0](fx.analyse(np.asarray(im)), P, SEED)


def job(a):
    fxp, style, name, out = a
    idx, pal = clip(load(Path(fxp)), style, name)
    h = hashlib.sha256(np.ascontiguousarray(idx[:HASHED]).tobytes() + np.ascontiguousarray(pal).tobytes()).hexdigest()[:16]
    if out:
        im = Image.fromarray(pal[idx[0]])
        k = 1080 // im.width
        im.resize((im.width * k, im.height * k), Image.NEAREST).save(Path(out) / f'{style}_{name}.png')
    return f'{style}/{name}', h


def render(out, fx=FX, styles=('glow', 'streak', 'dust'), names=SETS['all']) -> dict:
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
    jobs = [(str(fx), s, n, str(out) if out else None) for s in styles for n in names]
    with Pool(min(8, os.cpu_count() or 2)) as p:
        hashes = dict(p.map(job, jobs))
    if out:
        (Path(out) / 'hashes.json').write_text(json.dumps(hashes, indent=1, sort_keys=True))
    return hashes


def sheet(out: Path, dirs: list[Path], style: str, names: list[str], tile: int = 360) -> None:
    cols = 1 + len(dirs)
    sh = Image.new('RGB', (cols * tile, len(names) * (tile + 16)), (40, 40, 40))
    d = ImageDraw.Draw(sh)
    for r, n in enumerate(names):
        y = r * (tile + 16)
        a = np.asarray(Image.open(INPUTS / f'{n}.540.png').convert('RGBA')).astype(float)
        rgb = a[..., :3] * (a[..., 3:] / 255) + np.array([255, 0, 255]) * (1 - a[..., 3:] / 255)
        cells = [('INPUT ' + n, Image.fromarray(rgb.astype(np.uint8)))]
        cells += [(p.name, Image.open(p / f'{style}_{n}.png').convert('RGB')) for p in dirs]
        for c, (label, im) in enumerate(cells):
            sh.paste(im.resize((tile, tile), Image.LANCZOS), (c * tile, y + 16))
            d.text((c * tile + 3, y + 2), label, fill=(255, 255, 0))
    sh.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['render', 'freeze', 'check', 'sheet'])
    ap.add_argument('paths', nargs='*', type=Path)
    ap.add_argument('--fx', type=Path, default=FX)
    ap.add_argument('--styles', default='glow,streak,dust')
    ap.add_argument('--style', default='glow')
    ap.add_argument('--set', default='all', choices=list(SETS))
    a = ap.parse_args()
    styles, names = a.styles.split(','), SETS[a.set]
    if a.cmd == 'render':
        render(a.paths[0], a.fx, styles, names)
    elif a.cmd == 'freeze':
        BASELINE.write_text(json.dumps(render(ROOT / 'out' / 'bench' / 'base'), indent=1, sort_keys=True) + '\n')
        print('baseline frozen:', BASELINE)
    elif a.cmd == 'check':
        base, now = json.loads(BASELINE.read_text()), render(None, a.fx, styles, names)
        changed = sorted(k for k, h in now.items() if base.get(k) != h)
        by = {s: [k.split('/')[1] for k in changed if k.startswith(s + '/')] for s in styles}
        for s in styles:
            print(f'{s}: {len(by[s])} of {len(names)} changed', *by[s])
        sys.exit(1 if changed else 0)
    else:
        sheet(a.paths[0], a.paths[1:], a.style, names)


if __name__ == '__main__':
    main()
