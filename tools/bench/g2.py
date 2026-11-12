#!/usr/bin/env python3
"""Render the glow 2 prototype on the benchmark inputs: g2.py OUTDIR [k=v ...] -> OUTDIR/glow_<name>.png"""
import ast
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench  # noqa: E402
import phosphor_glow2 as g2  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
P = g2.Glow2()
for kv in sys.argv[2:]:
    k, v = kv.split('=', 1)
    setattr(P, k, ast.literal_eval(v))
for n in bench.SETS['all']:
    rgba = np.asarray(Image.open(bench.INPUTS / f'{n}.540.png').convert('RGBA'))
    idx, pal = g2.glow2(rgba, P)
    Image.fromarray(pal[idx]).resize((1080, 1080), Image.NEAREST).save(out / f'glow_{n}.png')
