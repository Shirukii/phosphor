#!/usr/bin/env python3
"""A side-by-side page: a reference clip beside ours, row by row, with a rating and a note each.

  compare.py TAG SPEC     SPEC is a text file, one row per line: label|reference clip|our clip
                          (clips are .mp4, .webp or .gif; they are copied as they are)

Writes out/compare_TAG/ (served by the "picks" entry of .claude/launch.json at
http://localhost:8351/compare_TAG/). Ratings and notes are kept in the page's localStorage
under compare_TAG. out/ is not in the repo: reference clips stay on this machine.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    tag, spec = sys.argv[1], Path(sys.argv[2])
    out = ROOT / 'out' / f'compare_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ln in enumerate(x for x in spec.read_text().splitlines() if x.strip()):
        label, ref, ours = ln.split('|')
        names = []
        for side, f in (('ref', ref), ('ours', ours)):
            name = f'{i:02d}_{side}{Path(f).suffix}'
            shutil.copy(f, out / name)
            names.append(name)
        rows.append([f'{i:02d}', label, *names])
    page = Path(__file__).with_name('compare_template.html').read_text()
    (out / 'index.html').write_text(page.replace('__ROWS__', json.dumps(rows)).replace('__TAG__', tag))
    print(out, len(rows))


if __name__ == '__main__':
    main()
