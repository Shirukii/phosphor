#!/usr/bin/env python3
"""Loops of the benchmark inputs for a picker: loops.py OUTDIR STYLE [--fx FILE] [NAME ...] -> OUTDIR/<name>.webp

--fx renders with another copy of the renderer (an earlier commit's, a candidate's), so two
versions can be put side by side in pick.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

args = sys.argv[1:]
fxp = bench.FX
if '--fx' in args:
    i = args.index('--fx')
    fxp = Path(args[i + 1])
    del args[i:i + 2]
out, style, names = Path(args[0]), args[1], args[2:] or bench.SETS['all']
out.mkdir(parents=True, exist_ok=True)
fx = bench.load(fxp)
for n in names:
    P = fx.params(style)
    frames = P.frames
    saved, bench.FRAMES = bench.FRAMES, frames
    fr = bench.clip(fx, style, n)
    bench.FRAMES = saved
    (out / f'{n}.webp').write_bytes(fx.encode(fr, 'webp', P.fps, P.scale))
    print(n, flush=True)
