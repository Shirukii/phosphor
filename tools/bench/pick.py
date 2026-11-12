#!/usr/bin/env python3
"""A blind picker page for the user: pairs shown left and right in a shuffled order.

  pick.py TAG SPEC     SPEC is a text file, one pair per line: name|top picture|picture A|picture B
                       (the top picture is what both are judged against: the input, or a reference)

Writes out/pick_TAG/ (served by the "picks" entry of .claude/launch.json at
http://localhost:8351/pick_TAG/) and the key to out/bench/keys/pick_TAG.json. The picks are
kept in the page's localStorage under pick_TAG.
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    tag, spec = sys.argv[1], Path(sys.argv[2])
    out = ROOT / 'out' / f'pick_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    rows = [ln.split('|') for ln in spec.read_text().splitlines() if ln.strip()]
    rng = random.Random(tag)
    rng.shuffle(rows)
    key, pairs = {}, []
    for i, (name, top, a, b) in enumerate(rows):
        k = f'{i:02d}'
        Image.open(top).convert('RGB').save(out / f'{k}_in.jpg', quality=92, subsampling=0)
        swap = rng.random() < 0.5
        left, right = (b, a) if swap else (a, b)
        Image.open(left).convert('RGB').save(out / f'{k}_L.png')
        Image.open(right).convert('RGB').save(out / f'{k}_R.png')
        key[k] = {'name': name, 'L': 'B' if swap else 'A', 'R': 'A' if swap else 'B'}
        pairs.append(k)
    keys = ROOT / 'out' / 'bench' / 'keys'
    keys.mkdir(parents=True, exist_ok=True)
    (keys / f'pick_{tag}.json').write_text(json.dumps(key, indent=1))
    page = Path(__file__).with_name('pick_template.html').read_text()
    (out / 'index.html').write_text(page.replace('__PAIRS__', json.dumps(pairs)).replace('__TAG__', tag))
    print(out, len(pairs))


if __name__ == '__main__':
    main()
