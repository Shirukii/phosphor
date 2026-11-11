#!/usr/bin/env python3
"""Turn a shaded drawing into the brightness map Phosphor draws from.

    phosphor_map.py SHADED [--frame NAME=LINEART ...] [--idle NAME] [--size WxH]
                    [--eye X0,X1,Y0,Y1] [--out map.ts] [--export LEDMAP]

SHADED is the picture lit the way the dots should be lit, on black (any image; converted to grey).
Each --frame is the line art of one frame, on a transparent background (RGBA), framed exactly like
SHADED: the first (or --idle) is the frame SHADED shows; the others (a blink, a glance) differ only
in small areas, and need no shaded version of their own: their change is taken from their line
art and shaded to match. The idle frame's alpha is the figure's silhouette, which the background
(the ray burst) stays out of. Without any --frame, the silhouette is wherever SHADED is lit.
Everything is resized to --size (default 340x280), the map's size: one value per display px.

Writes a TypeScript module (default map.ts) exporting the map: 4 bits per value, base64, plus a
small patch per other frame and the silhouette (1 bit). No image files, so the picture is pure code.

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


def build(shaded: np.ndarray, frames: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
          idle: str | None = None, eye: tuple[int, int, int, int] | None = None, T: Tone | None = None) -> dict:
    """shaded: the light, 0..1, HxW. frames: name -> (line art on black, alpha), 0..1, the same
    size; the idle frame is `idle` (default: the first). eye: the area (x0, x1, y0, y1) where the
    shading's response to the line art is measured (default: around everything the frames change).
    Returns the map as a dict (see to_ts) plus `stats`."""
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
    patches = {}
    for n, (delta, mv) in moved.items():
        soft = unit(as_img(mv.astype(np.float32)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2)))
        vals = tone(np.clip(shaded + gain * delta * soft, 0, 1), mask, wp, T)
        # keep the tone's own changes (it works locally) to a margin around the patch
        near = unit(as_img(mv.astype(np.float32)).filter(ImageFilter.MaxFilter(T.patch_margin * 2 + 1))) > 0.5
        vals = np.where(near, vals, base)
        ys, xs = np.nonzero(vals != base)
        if not len(ys):
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        patches[n] = {'x': int(x0), 'y': int(y0), 'w': int(x1 - x0), 'h': int(y1 - y0), 'data': pack(vals[y0:y1, x0:x1])}
    return {'w': W, 'h': H, 'idle': idle, 'data': pack(base),
            'body': base64.b64encode(np.packbits(body.ravel())).decode(), 'frames': patches,
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
    args.out.write_text(to_ts(m, args.export, HEADER))
    s = m['stats']
    print(f'{args.out}: {W}x{H}, {args.out.stat().st_size} bytes, {s["lit"]:.0%} lit, mean {s["mean"]:.2f}'
          + (f'; line-art gain {s["gain"]:.2f}' if m['frames'] else ''))
    for n, p in m['frames'].items():
        print(f'  {n}: patch {p["w"]}x{p["h"]} at {p["x"]},{p["y"]}')


if __name__ == '__main__':
    main()
