#!/usr/bin/env python3
"""Loops of the benchmark inputs for a picker: loops.py OUTDIR glow|glow2 [NAME ...] -> OUTDIR/<name>.webp"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench  # noqa: E402
import phosphor_fx as fx  # noqa: E402
import phosphor_glow2 as g2  # noqa: E402

out, which = Path(sys.argv[1]), sys.argv[2]
out.mkdir(parents=True, exist_ok=True)
for n in sys.argv[3:] or bench.SETS['all']:
    rgba = np.asarray(Image.open(bench.INPUTS / f'{n}.540.png').convert('RGBA'))
    if which == 'glow2':
        g2.save(g2.glow2(rgba, seed=bench.SEED), str(out / f'{n}.webp'))
    else:
        (out / f'{n}.webp').write_bytes(fx.encode(fx.glow(fx.analyse(rgba), fx.params('glow'), bench.SEED), 'webp'))
    print(n, flush=True)
