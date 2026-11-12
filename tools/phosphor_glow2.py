#!/usr/bin/env python3
"""Glow 2, a prototype: a smooth coloured picture, pattern-dithered onto a fixed palette.

What the reference pieces show, dot by dot: a full tone is a strict checker of two palette
colours with a third on every fourth dot, and the average colour of any small block runs along
one curve from black through blue and teal to salmon. That is an ordered (pattern) dither of a
continuous picture onto the palette, not dots placed by rule. So:

  light  -> the picture's light at the dot grid (white-paper line art is inverted)
  drawn  -> a tonal picture (a photo, a painting) is first drawn: its contours as lines one dot
            wide, and of its surfaces only the lit planes; inside, the shade is black
  expose -> each picture's light brought to one level: its brightest light to full, its
            middle light lifted if it sits too low (a dim picture is not left dim)
  heat   -> light, plus a soft bloom of it (dense or bright places run hotter)
  source -> one light source is found (the most compact strong light, by brightness or by
            colour strength, off the frame's edge) and runs hot, the heat falling off round it
  colour -> heat through one ramp: black, blue, teal, salmon, pale
  dots   -> the colour pattern-dithered (Knoll, 4x4 Bayer) onto the palette
  motion -> the picture holds; the sparse dots of the shade blink: each of them is given a fresh
            place in the dither pattern every couple of frames, so the light stays the same

  phosphor_glow2.py PICTURE OUT.png [--long 540] [--invert auto|yes|no]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

PALETTE = np.array([(0, 0, 0), (44, 114, 222), (79, 200, 82), (181, 73, 195), (237, 141, 55), (255, 255, 255)], np.float32)
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])


@dataclass
class Glow2:
    frames: int = 72           # frames in the loop (1: a still)
    fps: int = 24
    twinkle: float = 0.5       # share of the shade's sparse dots that blink (0: the picture is still)
    twinkle_hold: int = 2      # frames a blink lasts
    expose: float = 1.0        # how far each picture's light is brought to one level (0: as it is)
    expose_high: float = 0.9   # where the brightest light (the top half percent of the subject) is put
    expose_mid: float = 0.36   # the least the middle light (its median) is lifted to
    black: float = 0.3        # light below this is black
    white: float = 1.25        # light at this or more is full
    gamma: float = 1.0         # tone curve between them
    lines: float = 0.6         # unsharp amount: thin lines and edges keep their light
    bloom: float = 1.0         # how much hotter light runs where much light is gathered round it
    halo: float = 0.15         # how much of the bloom spills into the black round the light
    bloom_size: float = 6.0    # its radius (dots)
    draw: float | None = 0.0   # draw the picture as lines and lit planes: 1 yes, 0 as it is, None by its look (off: lost the user's pick 0 to 12)
    line_level: float = 0.85   # which edges become lines (percentile of edge strength in the subject)
    line_heat: tuple = (0.40, 0.75)   # a line's heat in the shade, and in full light
    plane_from: float = 0.70   # the light (percentile in the subject) where lit planes start
    plane_heat: float = 0.80   # the heat the brightest plane reaches
    plane_gamma: float = 1.6   # how fast the planes thin out below it
    line_min: int = 14         # the shortest line kept (dots): no worms on rough texture
    source: float = 0.4        # how much the picture's light source lifts the light round it (0: none)
    source_size: float = 0.04  # how far that heat reaches round it, as a share of the frame
    source_spill: float = 0.04 # how much of it lights the black round the source
    source_xy: tuple | None = None   # the source set by hand (x, y, 0..1 of the frame)
    spread: float = 0.6        # how far the dither may reach for colours off the target
    # heat -> colour: black, blue, teal, salmon, pale
    ramp: tuple = ((0.0, (0, 0, 0)), (0.45, (44, 114, 222)), (0.7, (61, 157, 152)), (0.82, (79, 190, 100)),
                   (0.92, (226, 140, 99)), (1.0, (245, 190, 170)))


def light(rgba: np.ndarray, invert: bool | None = None) -> np.ndarray:
    """The picture's light, 0..1. Mostly-white pictures (line art on paper) are inverted."""
    a = rgba[..., 3:].astype(np.float32) / 255
    lum = (rgba[..., :3].astype(np.float32) * a) @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    if invert is None:
        invert = float(np.median(lum)) > 0.6
    return 1 - lum if invert else lum


def line_art(rgba: np.ndarray, v: np.ndarray) -> bool:
    """Is the picture already a drawing: colourless, its light in thin strokes?"""
    subject = v > 0.06
    if not subject.any():
        return True
    sat = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2HSV)[..., 1][subject].mean() / 255
    opened = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return bool(sat < 0.02 and (v - opened).sum() / max(float(v.sum()), 1e-6) >= 0.18)


def drawn(v: np.ndarray, P: Glow2) -> np.ndarray:
    """A tonal picture as heat: contours one dot wide, and the lit planes; black elsewhere."""
    soft = cv2.GaussianBlur(v, (0, 0), 1.4)
    subject = soft > 0.04
    if not subject.any():
        return np.zeros_like(v)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8)
    gx, gy = cv2.Sobel(soft, cv2.CV_32F, 1, 0), cv2.Sobel(soft, cv2.CV_32F, 0, 1)
    hi = float(np.percentile(np.hypot(gx, gy)[subject], P.line_level * 100)) * 255
    edges = cv2.Canny(u8, 0.4 * hi, hi, L2gradient=True)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    edges = (stats[:, cv2.CC_STAT_AREA] >= P.line_min)[lab] & (lab > 0)
    near = cv2.GaussianBlur(v, (0, 0), 3.0)
    near = np.clip(near / max(float(np.percentile(near[subject], 99)), 1e-3), 0, 1)
    lines = np.where(edges, P.line_heat[0] + (P.line_heat[1] - P.line_heat[0]) * near, 0)
    lo, top = np.percentile(v[subject], [P.plane_from * 100, 99.7])
    planes = np.clip((v - lo) / max(top - lo, 1e-3), 0, 1) ** P.plane_gamma * P.plane_heat
    return np.maximum(lines, planes).astype(np.float32)


def emissive(rgba: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Light as a source gives it: bright, or strongly coloured (a red neon tube is dim by luminance)."""
    if line_art(rgba, v):
        return v
    return np.maximum(v, rgba[..., :3].max(-1).astype(np.float32) / 255 * (rgba[..., 3].astype(np.float32) / 255))


def source(rgba: np.ndarray, v: np.ndarray, P: Glow2) -> np.ndarray:
    """The light source as a field, 1 on it and falling off round it; zeros if there is none."""
    h, w = v.shape
    long = max(h, w)
    if P.source_xy is not None:
        region = np.zeros((h, w), np.uint8)
        cv2.circle(region, (round(P.source_xy[0] * (w - 1)), round(P.source_xy[1] * (h - 1))), max(2, round(0.012 * long)), 1, -1)
    else:
        e = emissive(rgba, v)
        small, large = cv2.GaussianBlur(e, (0, 0), 0.012 * long), cv2.GaussianBlur(e, (0, 0), 0.06 * long)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        centre = np.exp(-(((xx - w / 2) / (0.45 * w)) ** 2 + ((yy - h / 2) / (0.45 * h)) ** 2))
        # compact light stands out of its surround (a broad flat white does not); nearer the middle counts more
        score = (small - 0.4 * large) * (0.5 + 0.5 * centre)
        m = round(0.04 * long)
        score[:m], score[-m:], score[:, :m], score[:, -m:] = -1, -1, -1, -1
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        if score[y, x] < 0.05:
            return np.zeros_like(v)
        n, lab = cv2.connectedComponents((small > 0.8 * small[y, x]).astype(np.uint8))
        region = (lab == lab[y, x]).astype(np.uint8)
        if region.sum() > 0.012 * h * w:   # too broad to be one source: keep its heart only
            near = np.zeros((h, w), np.uint8)
            cv2.circle(near, (int(x), int(y)), round(0.035 * long), 1, -1)
            region &= near
    d = cv2.distanceTransform(1 - region, cv2.DIST_L2, 3)
    return np.exp(-(d / (P.source_size * long)) ** 2).astype(np.float32)


def exposed(v: np.ndarray, P: Glow2) -> np.ndarray:
    """The light at one level: a gain for the brightest light, a curve lifting a low middle."""
    subject = v > 0.06
    if not P.expose or subject.sum() < 50:
        return v
    mid, high = np.percentile(v[subject], [50, 99.5])
    gain = float(np.clip(P.expose_high / max(high, 1e-3), 1.0, 1.8))
    gain = 1 + (gain - 1) * float(np.clip((0.5 - mid) / 0.2, 0, 1))   # a picture bright in the middle needs none
    m = min(mid * gain, 0.99)
    gamma = float(np.clip(np.log(P.expose_mid) / np.log(m), 0.65, 1.0)) if m < P.expose_mid else 1.0
    out = np.clip(v * gain, 0, 1) ** gamma
    return (v + P.expose * (out - v)).astype(np.float32)


def knoll(target: np.ndarray, pal: np.ndarray, spread: float, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Pattern dither: per dot, n palette picks (dark to light) whose mix nears the target, and its Bayer cell."""
    h, w, _ = target.shape
    t = target.reshape(-1, 3)
    err = np.zeros_like(t)
    picks = np.empty((n, t.shape[0]), np.int16)
    for i in range(n):
        want = t + err * spread
        picks[i] = ((want[:, None, :] - pal[None]) ** 2).sum(-1).argmin(1)
        err += t - pal[picks[i]]
    lum = pal @ np.array([0.299, 0.587, 0.114], np.float32)
    order = np.argsort(lum[picks], axis=0, kind='stable')
    picks = np.take_along_axis(picks, order, 0)
    cell = np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w].reshape(-1)
    return picks, cell


def animate(picks: np.ndarray, cell: np.ndarray, shape: tuple, P: Glow2, seed: int) -> np.ndarray:
    """The frames: each dot takes the pick its Bayer cell names; blinking dots take a fresh cell in turn."""
    h, w = shape
    dots = np.arange(cell.size)
    still = picks[cell, dots]
    if P.frames <= 1 or not P.twinkle:
        return np.repeat(still.reshape(1, h, w), max(P.frames, 1), 0).astype(np.uint8)
    rng = np.random.default_rng(seed)
    # the shade: dots whose mix holds black, so they are lit only some of the time
    sparse = (picks == 0).any(0) & (picks != 0).any(0)
    blink = np.where(sparse & (rng.random(cell.size) < P.twinkle))[0]
    out = np.empty((P.frames, cell.size), np.uint8)
    steps = -(-P.frames // P.twinkle_hold)
    for k in range(steps):
        f = still.copy()
        f[blink] = picks[rng.integers(0, picks.shape[0], blink.size), blink]
        out[k * P.twinkle_hold:(k + 1) * P.twinkle_hold] = f
    return out.reshape(P.frames, h, w)


def glow2(rgba: np.ndarray, P: Glow2 | None = None, invert: bool | None = None, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The clip: (frames of palette indices (N, H, W), palette)."""
    P = P or Glow2()
    v0 = light(rgba, invert)
    v = exposed(v0, P)
    draw = P.draw if P.draw is not None else float(not line_art(rgba, v))
    if draw:
        v = drawn(v, P)
    else:
        if P.lines:
            v = v + P.lines * (v - cv2.GaussianBlur(v, (0, 0), 1.2))
        v = np.clip((v - P.black) / (P.white - P.black), 0, 1) ** P.gamma
    near = cv2.GaussianBlur(v, (0, 0), P.bloom_size)
    heat = v * (1 + P.bloom * near) + P.halo * near
    if P.source:
        # the source heats the light that is there, in proportion: the surface's detail stays, nothing fills flat
        near_source = source(rgba, v0, P)
        heat = heat * (1 + P.source * near_source) \
            + P.source_spill * P.source * near_source
    heat = np.clip(heat, 0, 1)
    xs = np.array([s for s, _ in P.ramp], np.float32)
    cs = np.array([c for _, c in P.ramp], np.float32)
    target = np.stack([np.interp(heat, xs, cs[:, k]) for k in range(3)], -1).astype(np.float32)
    picks, cell = knoll(target, PALETTE, P.spread)
    return animate(picks, cell, heat.shape, P, seed), PALETTE.astype(np.uint8)


def frame(img: Image.Image, long: int) -> np.ndarray:
    k = long / max(img.size)
    return np.asarray(img.convert('RGBA').resize((round(img.width * k), round(img.height * k)), Image.LANCZOS))


def save(fr: tuple[np.ndarray, np.ndarray], out: str, fps: int = 24, scale: int = 2) -> None:
    """The clip as a file by its ending: .png (a still, or a loop when it has frames: APNG), .gif, .webp."""
    idx, pal = fr
    flat = np.zeros(768, np.uint8)
    flat[:pal.size] = pal.ravel()
    ims = []
    for f in idx:
        im = Image.fromarray(f, 'P')
        im.putpalette(flat.tolist())
        ims.append(im.resize((im.width * scale, im.height * scale), Image.NEAREST))
    if len(ims) == 1:
        ims[0].convert('RGB').save(out)
    elif out.endswith('.webp'):
        ims[0].save(out, save_all=True, append_images=ims[1:], loop=0, duration=round(1000 / fps), lossless=True, method=1)
    else:
        ims[0].save(out, save_all=True, append_images=ims[1:], loop=0, duration=round(1000 / fps), disposal=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('picture')
    ap.add_argument('out', help='.png for a still, .webp or .gif for the loop')
    ap.add_argument('--long', type=int, default=540)
    ap.add_argument('--invert', default='auto', choices=['auto', 'yes', 'no'])
    ap.add_argument('--frames', type=int)
    a = ap.parse_args()
    P = Glow2()
    P.frames = a.frames or (1 if a.out.endswith('.png') else P.frames)
    save(glow2(frame(Image.open(a.picture), a.long), P, {'auto': None, 'yes': True, 'no': False}[a.invert]), a.out, P.fps)


if __name__ == '__main__':
    main()
