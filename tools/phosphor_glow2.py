#!/usr/bin/env python3
"""Glow 2, a prototype: a smooth coloured picture, pattern-dithered onto a fixed palette.

What the reference pieces show, dot by dot: a full tone is a strict checker of two palette
colours with a third on every fourth dot, and the average colour of any small block runs along
one curve from black through blue and teal to salmon. That is an ordered (pattern) dither of a
continuous picture onto the palette, not dots placed by rule. So:

  light  -> the picture's light at the dot grid (white-paper line art is inverted)
  drawn  -> a tonal picture (a photo, a painting) is first drawn: its contours as lines one dot
            wide, and of its surfaces only the lit planes; inside, the shade is black
  heat   -> light, plus a soft bloom of it (dense or bright places run hotter)
  colour -> heat through one ramp: black, blue, teal, salmon, pale
  dots   -> the colour pattern-dithered (Knoll, 4x4 Bayer) onto the palette

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
    black: float = 0.3        # light below this is black
    white: float = 1.25        # light at this or more is full
    gamma: float = 1.0         # tone curve between them
    lines: float = 0.6         # unsharp amount: thin lines and edges keep their light
    bloom: float = 1.0         # how much hotter light runs where much light is gathered round it
    halo: float = 0.15         # how much of the bloom spills into the black round the light
    bloom_size: float = 6.0    # its radius (dots)
    draw: float | None = None  # draw the picture as lines and lit planes: 1 yes, 0 as it is, None by its look
    line_level: float = 0.85   # which edges become lines (percentile of edge strength in the subject)
    line_heat: tuple = (0.40, 0.75)   # a line's heat in the shade, and in full light
    plane_from: float = 0.70   # the light (percentile in the subject) where lit planes start
    plane_heat: float = 0.80   # the heat the brightest plane reaches
    plane_gamma: float = 1.6   # how fast the planes thin out below it
    line_min: int = 14         # the shortest line kept (dots): no worms on rough texture
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


def knoll(target: np.ndarray, pal: np.ndarray, spread: float, n: int = 16) -> np.ndarray:
    """Pattern dither: per dot, n palette picks whose mix nears the target; the Bayer cell picks one."""
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
    return picks[cell, np.arange(t.shape[0])].reshape(h, w)


def glow2(rgba: np.ndarray, P: Glow2 | None = None, invert: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One frame: (palette index per dot, palette)."""
    P = P or Glow2()
    v = light(rgba, invert)
    draw = P.draw if P.draw is not None else float(not line_art(rgba, v))
    if draw:
        v = drawn(v, P)
    else:
        if P.lines:
            v = v + P.lines * (v - cv2.GaussianBlur(v, (0, 0), 1.2))
        v = np.clip((v - P.black) / (P.white - P.black), 0, 1) ** P.gamma
    near = cv2.GaussianBlur(v, (0, 0), P.bloom_size)
    heat = np.clip(v * (1 + P.bloom * near) + P.halo * near, 0, 1)
    xs = np.array([s for s, _ in P.ramp], np.float32)
    cs = np.array([c for _, c in P.ramp], np.float32)
    target = np.stack([np.interp(heat, xs, cs[:, k]) for k in range(3)], -1).astype(np.float32)
    return knoll(target, PALETTE, P.spread), PALETTE.astype(np.uint8)


def frame(img: Image.Image, long: int) -> np.ndarray:
    k = long / max(img.size)
    return np.asarray(img.convert('RGBA').resize((round(img.width * k), round(img.height * k)), Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('picture')
    ap.add_argument('out')
    ap.add_argument('--long', type=int, default=540)
    ap.add_argument('--invert', default='auto', choices=['auto', 'yes', 'no'])
    a = ap.parse_args()
    idx, pal = glow2(frame(Image.open(a.picture), a.long), invert={'auto': None, 'yes': True, 'no': False}[a.invert])
    im = Image.fromarray(pal[idx])
    im.resize((im.width * 2, im.height * 2), Image.NEAREST).save(a.out)


if __name__ == '__main__':
    main()
