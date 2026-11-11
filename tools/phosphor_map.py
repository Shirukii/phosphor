#!/usr/bin/env python3
"""Turn a shaded drawing into the brightness map Phosphor draws from.

    phosphor_map.py SHADED [--frame NAME=LINEART ...] [--idle NAME] [--size WxH]
                    [--eye X0,X1,Y0,Y1] [--out map.ts] [--export LEDMAP]
                    [--png DIR | --no-png] [--dots N[,N...]] [--backgrounds none,black,white,grey]

SHADED is the picture lit the way the dots should be lit, on black (any image; converted to grey).
Each --frame is the line art of one frame, on a transparent background (RGBA), framed exactly like
SHADED: the first (or --idle) is the frame SHADED shows; the others (a blink, a glance) differ only
in small areas, and need no shaded version of their own: their change is taken from their line
art and shaded to match. The idle frame's alpha is the figure's silhouette, which the background
(the ray burst) stays out of. Without any --frame, the silhouette is wherever SHADED is lit.
Everything is resized to --size (default 340x280), the map's size: one value per display px.

Writes a TypeScript module (default map.ts) exporting the map: 4 bits per value, base64, plus a
small patch per other frame and the silhouette (1 bit). No image files, so the picture is pure code.

Also writes a PNG of every frame (state) in every colour preset (src/presets.ts: blue, green,
amber, rose, grey), as the renderer draws it at rest: the same dots and colours, on each
background (transparent, black, white, grey), a folder per frame and one per background in it:
<png>/<frame>/<background>/<preset>.png, with --png defaulting to the folder of --out. Plus
black.png: black dots like ink, where the picture is dark (its light inverted, inside the
silhouette), for light backgrounds (not on black). --dots picks the dot sizes (default 1,2): 1 px dots, and 2x2 px dots as the
renderer's `dot: 2` draws them (same picture size, a coarser grid), as <preset>_2px.png.
--png-scale N draws every PNG px as NxN (a bigger image, no more detail; --size adds detail).

The map holds the light; the renderer turns it into an ordered (Bayer) dot pattern. Nothing is
outlined: an edge is just where the dots stop. Before that, the tone is boosted so the dark parts
keep their detail: in the deep shadows only, adaptive equalisation brings out detail, the shadows
are lifted, then local contrast is added (each value pushed away from the average around it).

Also a module: build() takes arrays (for a pipeline that frames its images itself), to_ts() writes.
Needs numpy, opencv-python(-headless) and pillow.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
from PIL import Image, ImageFilter


@dataclass
class Tone:
    black: float = 0.04          # black point
    white: float = 99.5          # the percentile of the figure's light taken as white
    gamma: float = 0.8           # tone curve after normalising
    lift: float = 0.6            # shadow lift (gamma; lower brings more of the dark up)
    local: float = 1.5           # local contrast: amount
    local_r: float = 6           # and the radius it compares against (px)
    floor: float = 0.03          # below this after the lift: empty
    # Before any of that, the deep shadows are brought up toward an adaptive histogram equalisation
    # (CLAHE) of the drawing, so their detail comes out instead of a hole. Only where the drawing is
    # dark (blurred tone below `shadow`), and only `amount` of the way, so the lit parts and the
    # overall light stay as drawn. Clip limit, and tiles across the panel.
    clahe_clip: float = 2.0
    clahe_tiles: int = 6
    shadow: float = 0.15
    shadow_r: float = 6
    amount: float = 0.6
    # Frames: a patch covers where a frame's line art differs from idle by more than patch_min,
    # plus patch_margin px. The silhouette reaches body_pad px past the figure.
    patch_min: float = 0.03
    patch_margin: int = 8
    body_pad: int = 3


def unit(img: Image.Image) -> np.ndarray:
    return np.asarray(img, np.float32) / 255


def as_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def tone(shaded: np.ndarray, mask: np.ndarray, wp: float, T: Tone) -> np.ndarray:
    """The shaded drawing (0..1) -> the light, 0..15."""
    eq = cv2.createCLAHE(clipLimit=T.clahe_clip, tileGridSize=(T.clahe_tiles, T.clahe_tiles)).apply(
        (shaded * 255).round().astype(np.uint8))
    t = shaded * mask
    dark = np.clip(1 - unit(as_img(t).filter(ImageFilter.GaussianBlur(T.shadow_r))) / T.shadow, 0, 1)
    t = t + np.maximum(eq.astype(np.float32) / 255 * mask - t, 0) * dark * T.amount
    t = np.clip((t - T.black) / (wp - T.black), 0, 1) ** T.gamma
    t = t ** T.lift
    t = np.clip(t + T.local * (t - unit(as_img(t).filter(ImageFilter.GaussianBlur(T.local_r)))), 0, 1) * (t > T.floor)
    return np.clip(np.round(t * 15), 0, 15).astype(np.uint8)


def pack(vals: np.ndarray) -> str:
    v = vals.ravel()
    if v.size % 2:
        v = np.append(v, np.zeros(1, v.dtype))  # (np.append(v, 0) would widen to int64)
    return base64.b64encode((v[0::2] << 4 | v[1::2]).tobytes()).decode()


# DEFAULT_LOOK in src/phosphor.ts: what decides which dots are lit and their colour.
LOOK = dict(gain=1.0, curve=1.0, solid=1.01, fill=1.0, rampA=2.0, rampB=3.0)
PRESETS = Path(__file__).resolve().parent.parent / 'src' / 'presets.ts'


def presets() -> dict[str, list[str]]:
    """The colour presets, from src/presets.ts (one `name: ['#dim', '#mid', '#bright'],` per line)."""
    import re
    hexes = r"'(#[0-9a-fA-F]{6})'"
    found = {m[0]: list(m[1:]) for m in re.findall(rf"^\s*(\w+):\s*\[{hexes},\s*{hexes},\s*{hexes}\]", PRESETS.read_text(), re.M)}
    if not found:
        raise ValueError(f'no presets in {PRESETS}')
    return found


def bayer8(h: int, w: int) -> np.ndarray:
    """The shader's 8x8 Bayer threshold (bayer8 + 0.5/64) for every px."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    b2 = lambda x, y: np.mod(np.floor(x) / 2 + np.floor(y) ** 2 * 0.75, 1)
    b4 = lambda x, y: b2(x / 2, y / 2) * 0.25 + b2(x, y)
    return b4(x / 2, y / 2) * 0.25 + b2(x, y) + 0.5 / 64


def sample(tex: np.ndarray, dot: int) -> np.ndarray:
    """The map as the shader reads it with dots of `dot` px: at each dot's centre, linearly filtered
    (clamped at the edges). For 1, the px itself; for 2, the average of the 2x2 px under the dot."""
    if dot == 1:
        return tex
    H, W = tex.shape

    def axis(n, size):
        x = (np.arange(n) + 0.5) * dot - 0.5   # the centre, in texel index space
        i0 = np.floor(x).astype(int)
        f = x - i0
        return np.clip(i0, 0, size - 1), np.clip(i0 + 1, 0, size - 1), f
    y0, y1, fy = axis(-(-H // dot), H)
    x0, x1, fx = axis(-(-W // dot), W)
    top = tex[y0][:, x0] * (1 - fx) + tex[y0][:, x1] * fx
    bot = tex[y1][:, x0] * (1 - fx) + tex[y1][:, x1] * fx
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def render(vals: np.ndarray, colors: list[str], L: dict | None = None, dot: int = 1) -> Image.Image:
    """A frame (0..15) as the renderer draws it at rest (no breath, the drifting light at its mean):
    RGBA, the dots in their colours on transparent, at the map's size. dot: dot size, px.
    Full light is solid, as the renderer draws it with look.full = 1. By default it leaves 1 dot in
    64 dark there, which the drifting light keeps filling in, so it reads as solid in the animation;
    a still would show it as a grid of pinholes."""
    L = {**LOOK, **(L or {})}
    H0, W0 = vals.shape
    b = sample((vals.astype(np.float64) * 17).round() / 255, dot)  # as the texture holds it
    H, W = b.shape
    v = b ** L['curve'] * L['gain']
    lit = (b >= 0.02) & ((b >= 1 - 1e-6) | (v >= L['solid']) | (v / L['solid'] * L['fill'] > bayer8(H, W)))
    pal = [np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.float64) / 255 for c in colors]

    def smooth(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    lo = pal[0] + (pal[1] - pal[0]) * smooth(0.2, L['rampA'], v)[..., None]
    hi = pal[1] + (pal[2] - pal[1]) * smooth(L['rampA'], L['rampB'], v)[..., None]
    col = np.where((v < L['rampA'])[..., None], lo, hi)
    out = np.zeros((H, W, 4), np.uint8)
    out[..., :3] = (col * 255).round().astype(np.uint8)
    out[..., 3] = lit * 255
    out[~lit, :3] = 0
    img = Image.fromarray(out, 'RGBA')
    return img.resize((W * dot, H * dot), Image.NEAREST).crop((0, 0, W0, H0)) if dot > 1 else img


BACKGROUNDS = {'none': None, 'black': '#000000', 'white': '#ffffff', 'grey': '#808080'}
INK = ['#000000'] * 3  # the black version: dots where the picture is dark


def ink(vals: np.ndarray, body: np.ndarray) -> np.ndarray:
    """A frame's light inverted inside the silhouette (dark is dense), for black dots on light."""
    return np.where(body, 15 - vals, 0).astype(np.uint8)


def pngs(m: dict, dots: Iterable[int] = (1, 2), backgrounds: Iterable[str] = tuple(BACKGROUNDS),
         colours: Iterable[str] | None = None, scale: int = 1) -> Iterator[tuple[str, Image.Image]]:
    """Every frame of a built map in each preset (`colours`: preset names, plus 'black' for the ink
    version; default all), dot size and background, as (path, image): <frame>/<background>/<preset>.png
    (1 px dots), .../<preset>_<N>px.png (N px dots). Background 'none' (folder `transparent`) keeps
    the alpha; black ink is not drawn on black. scale: each px as scale x scale px (no more detail)."""
    table = {**presets(), 'black': INK}
    names = list(table) if colours is None else [c for c in colours if c in table]
    for name, vals in m['maps'].items():
        for preset in names:
            colors = table[preset]
            for d in dots:
                img = render(ink(vals, m['bodymask']) if colors is INK else vals, colors, dot=d)
                if scale > 1:
                    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
                for bg in backgrounds:
                    col = BACKGROUNDS[bg]
                    if colors is INK and col == '#000000':
                        continue
                    path = f'{name}/{"transparent" if col is None else bg}/{preset}{"" if d == 1 else f"_{d}px"}.png'
                    if col is None:
                        yield path, img
                    else:
                        flat = Image.new('RGBA', img.size, col)
                        flat.alpha_composite(img)
                        yield path, flat.convert('RGB')


def write_pngs(m: dict, folder: Path, dots: Iterable[int] = (1, 2),
               backgrounds: Iterable[str] = tuple(BACKGROUNDS), scale: int = 1) -> list[Path]:
    """pngs(), saved under folder."""
    out = []
    for path, img in pngs(m, dots, backgrounds, scale=scale):
        f = folder / path
        f.parent.mkdir(parents=True, exist_ok=True)
        img.save(f)
        out.append(f)
    return out


def build(shaded: np.ndarray, frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
          idle: str | None = None, eye: tuple[int, int, int, int] | None = None, T: Tone | None = None) -> dict:
    """shaded: the light, 0..1, HxW. frames: name -> (line art on black, alpha), 0..1, the same
    size; the idle frame is `idle` (default: the first). eye: the area (x0, x1, y0, y1) where the
    shading's response to the line art is measured (default: around everything the frames change).
    Returns the map as a dict (see to_ts) plus `stats`, and `maps`: every frame in full (0..15)."""
    T = T or Tone()
    H, W = shaded.shape
    frames = frames or {}
    idle = idle or next(iter(frames), 'idle')
    if frames:
        line0, alpha = frames[idle]
    else:
        line0, alpha = np.zeros_like(shaded), (shaded > 0.02).astype(np.float32)
    mask = unit(as_img(alpha).filter(ImageFilter.MaxFilter(5)))
    body = unit(as_img(alpha).filter(ImageFilter.MaxFilter(T.body_pad * 2 + 1))) > 0.3
    wp = float(np.percentile((shaded * mask)[shaded * mask > 0.02], T.white))  # white point, before the lift
    base = tone(shaded, mask, wp, T)
    # The other frames are the same drawing with something small changed. Each gets the shaded
    # drawing plus its own line art's change from idle, scaled by how strongly the shading follows
    # the line art there; only that patch (where the frames differ) is stored.
    moved = {}
    for n, (line, _) in frames.items():
        if n == idle:
            continue
        delta = line - line0
        mv = unit(as_img(np.abs(delta)).filter(ImageFilter.GaussianBlur(2))) > T.patch_min
        if mv.any():
            moved[n] = (delta, mv)
    if moved and eye is None:
        ys, xs = np.nonzero(np.any([mv for _, mv in moved.values()], axis=0))
        pad = 20
        eye = (max(0, xs.min() - pad), min(W, xs.max() + pad), max(0, ys.min() - pad), min(H, ys.max() + pad))
    gain = 0.0
    if moved:
        ey, ex = slice(eye[2], eye[3]), slice(eye[0], eye[1])
        a, b = line0[ey, ex].ravel(), shaded[ey, ex].ravel()
        gain = float(np.cov(a, b)[0, 1] / a.var()) if a.var() > 0 else 0.0
    patches, maps = {}, {idle: base}
    for n, (delta, mv) in moved.items():
        soft = unit(as_img(mv.astype(np.float32)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2)))
        vals = tone(np.clip(shaded + gain * delta * soft, 0, 1), mask, wp, T)
        # keep the tone's own changes (it works locally) to a margin around the patch
        near = unit(as_img(mv.astype(np.float32)).filter(ImageFilter.MaxFilter(T.patch_margin * 2 + 1))) > 0.5
        vals = np.where(near, vals, base)
        maps[n] = vals
        ys, xs = np.nonzero(vals != base)
        if not len(ys):
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        patches[n] = {'x': int(x0), 'y': int(y0), 'w': int(x1 - x0), 'h': int(y1 - y0), 'data': pack(vals[y0:y1, x0:x1])}
    return {'w': W, 'h': H, 'idle': idle, 'data': pack(base),
            'body': base64.b64encode(np.packbits(body.ravel())).decode(), 'frames': patches, 'maps': maps, 'bodymask': body,
            'stats': {'lit': float((base > 0).mean()), 'mean': float(base[base > 0].mean() / 15) if (base > 0).any() else 0.0,
                      'gain': gain}}


def to_ts(m: dict, export: str = 'LEDMAP', header: str = '') -> str:
    """The map as a TypeScript module; `header` (comment lines) goes on top."""
    frames = ''.join(f'\n    {n}: {{ x: {p["x"]}, y: {p["y"]}, w: {p["w"]}, h: {p["h"]}, data: \'{p["data"]}\' }},'
                     for n, p in m['frames'].items())
    return (header +
            f'export const {export} = {{\n  w: {m["w"]}, h: {m["h"]}, idle: \'{m["idle"]}\',\n  data: \'{m["data"]}\',\n'
            f'  body: \'{m["body"]}\',\n'
            f'  frames: {{{frames}\n  }} as Record<string, {{ x: number; y: number; w: number; h: number; data: string }}>,\n}};\n')


HEADER = ('// Written by phosphor_map.py: a brightness map, 4 bits per value.\n'
          '// `data` is the idle frame; each of `frames` is a patch (a rect of the map) where that frame differs;\n'
          '// `body` is the silhouette, 1 bit per px (MSB first), which the background stays out of.\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('shaded', type=Path)
    ap.add_argument('--frame', action='append', default=[], metavar='NAME=PATH', help='line art of a frame (RGBA)')
    ap.add_argument('--idle', help='the frame SHADED shows (default: the first --frame)')
    ap.add_argument('--size', default='340x280', help='map size, WxH (default 340x280)')
    ap.add_argument('--eye', help='x0,x1,y0,y1: where the shading follows the line art (default: auto)')
    ap.add_argument('--out', type=Path, default=Path('map.ts'))
    ap.add_argument('--export', default='LEDMAP', help='name of the exported constant')
    ap.add_argument('--png', type=Path, help='folder for the PNGs, a subfolder per frame (default: the folder of --out)')
    ap.add_argument('--no-png', action='store_true', help='write no PNGs')
    ap.add_argument('--dots', default='1,2', help='dot sizes for the PNGs, px (default 1,2)')
    ap.add_argument('--png-scale', type=int, default=1, help='draw each PNG px as NxN (bigger, no more detail; default 1)')
    ap.add_argument('--backgrounds', default=','.join(BACKGROUNDS),
                    help=f'PNG backgrounds, of {",".join(BACKGROUNDS)} (none: transparent; default all)')
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.lower().split('x'))
    shaded = unit(Image.open(args.shaded).convert('L').resize((W, H), Image.LANCZOS))
    frames = {}
    for f in args.frame:
        name, _, path = f.partition('=')
        im = Image.open(path).convert('RGBA').resize((W, H), Image.LANCZOS)
        a = unit(im.getchannel('A'))
        frames[name] = (unit(im.convert('L')) * a, a)
    eye = tuple(int(v) for v in args.eye.split(',')) if args.eye else None
    m = build(shaded, frames, args.idle, eye)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_ts(m, args.export, HEADER))
    s = m['stats']
    print(f'{args.out}: {W}x{H}, {args.out.stat().st_size} bytes, {s["lit"]:.0%} lit, mean {s["mean"]:.2f}'
          + (f'; line-art gain {s["gain"]:.2f}' if m['frames'] else ''))
    for n, p in m['frames'].items():
        print(f'  {n}: patch {p["w"]}x{p["h"]} at {p["x"]},{p["y"]}')
    if not args.no_png:
        folder = args.png or args.out.parent
        dots = tuple(sorted({int(k) for k in args.dots.split(',')}))
        bgs = tuple(b.strip() for b in args.backgrounds.split(','))
        bad = [b for b in bgs if b not in BACKGROUNDS]
        if bad:
            ap.error(f'unknown background {", ".join(bad)}: use {", ".join(BACKGROUNDS)}')
        pngs = write_pngs(m, folder, dots, bgs, max(1, args.png_scale))
        print(f'{folder}: {len(pngs)} PNGs, a folder per frame and background '
              f'({len(presets())} presets and black ink x {len(dots)} dot sizes x {len(bgs)} backgrounds)')


if __name__ == '__main__':
    main()
