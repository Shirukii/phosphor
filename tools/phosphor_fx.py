#!/usr/bin/env python3
"""Render a picture as a looping clip of dots on black, in one of three styles.

    phosphor_fx.py PICTURE [--style glow|streak|dust|all] [--long N] [--scale N] [--frames N] [--fps N]
                   [--cutout] [--colour C] [--angle DEG] [--flat] [--seed N] [--out DIR] [--formats gif,webp,png]

  glow    six colours on black: the picture's light, levelled, as one smooth ramp of colour
          (black, blue, teal, salmon, pale) pattern-dithered onto the palette, so the shade is a
          sparse lattice of blue dots and full tones are checkers of two or three colours. Light
          runs hotter where much of it gathers, and round the picture's one light source. The
          picture holds still; the sparse dots of the shade blink.
  streak  one colour (--colour, default red) on a coarse grid: the figure error-diffused afresh
          every frame so it boils, short dashes along its shapes (and over the texture of its
          shaded parts, which the dither leaves black), loose specks gathering towards the bottom,
          and slanted rain flashing up in new places every frame.
  dust    grey specks in small grains, brighter in the light, the outline and strongest edges
          as bright strokes, short flecks along the shapes and a few glints; from its darker side
          along --angle the figure comes apart in lanes of dashes that drift off and fade.

These follow the looks of the references: glow's dots, their colours block by block and their
blinking were measured off them (the light constant, a tenth to a fifth of the lit dots changing
from one frame to the next);
streak a coarse, boiling stipple on clean
black, in a half- to one-second loop; dust a sparse field of grey flecks (~17% lit in the face),
the hair drifting off in strands.

Each is drawn at --long dots across the longer side (glow 540, streak 270, dust 540), and saved at
--scale times that (glow 2, streak 4, dust 2: each dot as NxN px). The clip loops without a seam:
everything that moves comes back to where it started after --frames frames (glow 72, streak 24,
dust 48; all at 24 fps). Without --seed, the seed comes from the picture: the same picture comes
out the same.

A picture with transparency is taken as cut out already; --cutout cuts the figure out of any other
(GrabCut, as phosphor_prep does). Otherwise the whole picture is used, framed to cover the size.

Writes DIR/<style>.gif, DIR/<style>.webp (animated, lossless: the same dots, a fraction of the
size) and DIR/<style>/0000.png... plus DIR/<style>.png (the first frame), where DIR defaults to
out/<picture name>/fx. Lossy video (MP4, WebM) smears 1-dot detail and comes out bigger than the
GIF, so there is none.

Also a module: analyse() reads a framed picture, glow() and streak() return the frames (palette
indices plus the palette), render() does both steps, encode() / save() write. Needs numpy,
opencv-python(-headless) and pillow.
"""
from __future__ import annotations

import argparse
import io
import math
import zlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phosphor_prep as prep  # noqa: E402

Frames = tuple[np.ndarray, np.ndarray]  # (N, H, W) uint8 palette indices, (K, 3) uint8 RGB palette


# ---- the picture, read

def form(lum: np.ndarray, a: np.ndarray, amount: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """The light with shading added where the picture has none (flat colour: cel-shaded art, a
    woodblock print, a flat background), and how flat each px is. Flat areas are lit softly from
    above and fall off a little towards the figure's outline; shaded and detailed parts
    are left as they are. Without it a flat area comes out one even texture: a block."""
    H, W = lum.shape
    R = 0.1 * max(H, W)
    # how flat: the variation left once thin drawn lines are taken out (they are marks on a flat
    # area, not shading in it)
    k = 2 * max(2, round(max(H, W) / 180)) + 1
    base = cv2.medianBlur((np.clip(lum, 0, 1) * 255).astype(np.uint8), k).astype(np.float32) / 255
    m = cv2.GaussianBlur(base, (0, 0), 3)
    sd = np.sqrt(np.clip(cv2.GaussianBlur(base * base, (0, 0), 3) - m * m, 0, None))
    flat = cv2.GaussianBlur(np.clip(1 - sd / 0.03, 0, 1), (0, 0), 3)
    # one soft light over the figure from above (a little left): faces sit high in a picture; and the figure a little fuller in its
    # middle than at its outline. (Turning each flat area on its own was tried: in flat art the
    # areas' edges are too ragged, and it bruises the skin round eyes and mouths.)
    m8 = (a > 0.5).astype(np.uint8)
    if m8.all():   # not cut out: the frame's edge stands for the outline
        m8 = np.pad(m8[1:-1, 1:-1], 1)
    body = cv2.GaussianBlur(1 - np.exp(-cv2.distanceTransform(m8, cv2.DIST_L2, 5) / R), (0, 0), R / 3)
    ys, xs = np.nonzero(m8)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ramp = 0.25 * (xx - xs.min()) / max(1, xs.max() - xs.min()) + 0.95 * (yy - ys.min()) / max(1, ys.max() - ys.min())
    ramp = np.clip(ramp / 1.2, 0, 1)                               # 0 at the top (a little left), 1 at the bottom
    # it only takes light away, on the shaded side: the lit side keeps the picture's own light,
    # so flat skin matches the same skin next to it that isn't flat
    shape = np.clip(1 - 0.6 * ramp ** 1.6 - 0.25 * (1 - body), 0.3, 1)
    f = amount * flat
    return (lum * (1 - f + f * shape)).astype(np.float32), flat.astype(np.float32)


def analyse(rgba: np.ndarray) -> dict:
    """What both styles draw from, for a picture already at the output size (RGBA uint8):
    tone   0..1, the light (locally equalised a little, 0 outside the figure)
    plain  0..1, the light as it is (only stretched), 0 outside the figure
    alpha  0..1, the figure
    warm   -1..1, how warm (red over blue) the picture is there
    tx, ty the direction along the shapes at each px (unit; an axis, so its sign means nothing)
    coh    0..1, how strongly the shapes there run one way
    edge   0..1, how sharp the light changes there
    flat   0..1, how flat (unshaded) the picture is there"""
    rgb = rgba[..., :3].astype(np.float32) / 255
    a = rgba[..., 3].astype(np.float32) / 255
    lum, flat = form(rgb @ np.float32([0.299, 0.587, 0.114]), a)
    l8 = (np.clip(lum, 0, 1) * 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l8).astype(np.float32) / 255
    t = 0.7 * lum + 0.3 * eq
    inside = a > 0.5
    if inside.any():
        lo, hi = np.percentile(t[inside], [1, 99.7])
        t = np.clip((t - lo) / max(hi - lo, 1e-3), 0, 1)
    tone = (t * a).astype(np.float32)
    if inside.any():  # the plain light too, without the equalising (which evens out the lighting)
        lo, hi = np.percentile(lum[inside], [1, 99.7])
        plain = (np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1) * a).astype(np.float32)
    else:
        plain = tone
    warm = np.clip((rgb[..., 0] - rgb[..., 2]) * 2.5, -1, 1).astype(np.float32)
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)   # H in degrees
    hue_warm = ((hsv[..., 0] > 5) & (hsv[..., 0] < 40) & (hsv[..., 1] > 0.12) & (hsv[..., 1] < 0.65) & (hsv[..., 2] > 0.4)).astype(np.float32)   # skin-like: light, moderately saturated, red-orange

    g = cv2.GaussianBlur(tone, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jxy, jyy = (cv2.GaussianBlur(v, (0, 0), 3) + 0.3 * cv2.GaussianBlur(v, (0, 0), 14)
                     for v in (gx * gx, gx * gy, gy * gy))
    jxx = jxx + 1e-7  # nothing there at all: lie horizontal
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # across the gradient
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    mag = np.hypot(gx, gy)
    edge = np.clip(mag / (np.percentile(mag[inside], 97) if inside.any() else mag.max() or 1), 0, 1)
    return dict(rgba=rgba, tone=tone, plain=plain, alpha=a, warm=warm, hue_warm=hue_warm, lum=(lum * (a > 0.5)).astype(np.float32), tx=np.cos(ang).astype(np.float32),
                ty=np.sin(ang).astype(np.float32), coh=(lam / (jxx + jyy)).astype(np.float32),
                edge=edge.astype(np.float32), flat=(flat * (a > 0.5)).astype(np.float32))


# ---- small parts

def bayer(h: int, w: int) -> np.ndarray:
    """The 8x8 ordered-dither thresholds (0..1) tiled over h x w."""
    m = np.zeros((1, 1), int)
    for _ in range(3):
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    m = (m + 0.5) / 64
    return np.tile(m, (h // 8 + 1, w // 8 + 1))[:h, :w].astype(np.float32)


def plot(img: np.ndarray, pts: np.ndarray, val) -> None:
    """Set the px at pts (M, 2) float, those inside, to val (scalar or (M,) array)."""
    H, W = img.shape
    x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    img[y[ok], x[ok]] = val[ok] if isinstance(val, np.ndarray) else val


def seed_where(weight: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n px positions (x, y), drawn with probability following weight (>= 0)."""
    w = weight.ravel().astype(np.float64)
    if n <= 0 or w.sum() <= 0:
        return np.zeros((0, 2), np.float32)
    i = rng.choice(w.size, size=n, p=w / w.sum())
    W = weight.shape[1]
    return np.stack([i % W, i // W], 1).astype(np.float32)


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# ---- glow
#
# What the reference pieces show, dot by dot: a full tone is a strict checker of two palette colours
# with a third on every fourth dot, and the average colour of any small block runs along one curve
# from black through blue and teal to salmon. That is an ordered (pattern) dither of a smooth
# coloured picture onto the palette, not dots placed by rule. And frame by frame: the picture holds,
# its light stays the same, and the sparse dots of the shade blink, each on for a couple of frames.

GLOW_PALETTE = np.array([(0, 0, 0), (44, 114, 222), (79, 200, 82), (181, 73, 195), (237, 141, 55), (255, 255, 255)], np.float32)
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])


@dataclass
class Glow:
    frames: int = 72           # frames in the loop (1: a still)
    fps: int = 24
    long: int = 540            # dots across the longer side
    scale: int = 2
    gain: float = 1.0          # more or less light, after the levelling
    twinkle: float = 0.5       # share of the shade's sparse dots that blink (0: the picture is still)
    twinkle_hold: int = 2      # frames a blink lasts
    twinkle_rate: float = 0.3  # how often a blinking dot changes: its chance each time
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
    source: float = 0.4        # how much the picture's light source lifts the light round it (0: none)
    source_size: float = 0.04  # how far that heat reaches round it, as a share of the frame
    source_spill: float = 0.04 # how much of it lights the black round the source
    source_xy: tuple | None = None   # the source set by hand (x, y, 0..1 of the frame)
    warm: float = 1.0          # how much of the mix is warm (magenta, orange): the Warmth control
    # heat -> how many of the dots are lit there, and their mix of blue, green, magenta, orange, white.
    # Measured off the references: even the sparsest dots are only five-sixths blue, with a few of
    # every other colour among them; round the light a band rich in green; the light itself orange
    # and magenta with a little white, hardly any blue.
    ramp: tuple = ((0.0, 0.0, (0.83, 0.06, 0.05, 0.045, 0.015)),
                   (0.45, 1.0, (0.83, 0.06, 0.05, 0.045, 0.015)),
                   (0.6, 1.0, (0.68, 0.16, 0.07, 0.07, 0.02)),
                   (0.75, 1.0, (0.42, 0.38, 0.08, 0.10, 0.02)),
                   (0.88, 0.97, (0.30, 0.10, 0.26, 0.32, 0.02)),
                   (1.0, 0.94, (0.22, 0.03, 0.30, 0.42, 0.03)))
    floor: float = 0.06        # light fainter than this share of full is left black
    drips: int = 60            # dotted lines falling from the picture's lower edges (0: none)
    drip_len: tuple = (10, 40)  # how far one falls (dots)
    drip_speed: float = 0.5    # dots a frame
    drip_bundle: float = 0.6   # chance of another beside it, two dots over
    shake: float = 0.25        # how far a dot's place in the dither pattern is shaken (of one step): small shares still show


def light(rgba: np.ndarray) -> np.ndarray:
    """The picture's light, 0..1. A drawing on white paper (colourless, mostly white) is inverted:
    its lines are the light."""
    a = rgba[..., 3:].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) * a
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    paper = a[..., 0] > 0.5
    if paper.mean() > 0.5 and float(np.median(lum[paper])) > 0.6 and float((rgb.max(-1) - rgb.min(-1))[paper].mean()) < 8:
        return ((1 - lum) * a[..., 0]).astype(np.float32)
    return lum


def line_art(rgba: np.ndarray, v: np.ndarray) -> bool:
    """Is the picture already a drawing: colourless, its light in thin strokes?"""
    subject = v > 0.06
    if not subject.any():
        return True
    sat = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2HSV)[..., 1][subject].mean() / 255
    opened = cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return bool(sat < 0.02 and (v - opened).sum() / max(float(v.sum()), 1e-6) >= 0.18)


def emissive(rgba: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Light as a source gives it: bright, or strongly coloured (a red neon tube is dim by luminance)."""
    if line_art(rgba, v):
        return v
    return np.maximum(v, rgba[..., :3].max(-1).astype(np.float32) / 255 * (rgba[..., 3].astype(np.float32) / 255))


def source(rgba: np.ndarray, v: np.ndarray, P: Glow) -> np.ndarray:
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


def exposed(v: np.ndarray, P: Glow) -> np.ndarray:
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


ORDER = np.array([0, 1, 3, 2, 4, 5])   # the palette dark to light: black, blue, magenta, green, orange, white


def pattern(cover: np.ndarray, mix: np.ndarray, seed: int, shake: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Pattern dither of a mix. Per dot: how much of it is lit (cover, 0..1) and the shares of the five
    colours among the lit (mix, (..., 5): blue, green, magenta, orange, white). Returns, per dot, the
    thresholds at which it steps up from black through the colours dark to light, and its own
    threshold: its place in the 8x8 Bayer pattern, shaken a little so small shares still show."""
    h, w = cover.shape
    share = np.concatenate([1 - cover[..., None], cover[..., None] * mix], -1).reshape(-1, 6)[:, ORDER]
    steps = np.cumsum(share, 1)[:, :-1].astype(np.float32)
    rng = np.random.default_rng(seed)
    cell = (bayer(h, w).reshape(-1) * 64 - 0.5)   # 0..63
    return steps, (((cell + 0.5 + shake * 64 * (rng.random(cell.size) - 0.5)) / 64) % 1.0).astype(np.float32)


def dots(steps: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The palette index of each dot, for its threshold t."""
    return ORDER[(t[:, None] >= steps).sum(1)].astype(np.uint8)


def animate(steps: np.ndarray, t: np.ndarray, shape: tuple, P: Glow, seed: int) -> np.ndarray:
    """The frames: the picture holds; blinking dots take a fresh threshold in turn."""
    h, w = shape
    still = dots(steps, t)
    if P.frames <= 1 or not P.twinkle:
        return np.repeat(still.reshape(1, h, w), max(P.frames, 1), 0)
    rng = np.random.default_rng(seed + 1)
    # the shade: dots lit only some of the time
    sparse = (steps[:, 0] > 0.02) & (steps[:, 0] < 0.98)
    blink = np.where(sparse & (rng.random(t.size) < P.twinkle))[0]
    out = np.empty((P.frames, t.size), np.uint8)
    # each blinking dot changes at its own moments and runs through its own few thresholds in a ring,
    # its place in the pattern first: the last change of the loop brings it back to where it started
    n_steps = -(-P.frames // P.twinkle_hold)
    count = np.cumsum(rng.random((n_steps, blink.size)) < P.twinkle_rate, 0)
    total = np.maximum(count[-1], 1)
    ring = rng.random((int(total.max()), blink.size)).astype(np.float32)
    ring[0] = t[blink]
    f = still.copy()
    for k in range(n_steps):
        f[blink] = dots(steps[blink], ring[count[k] % total, np.arange(blink.size)])
        out[k * P.twinkle_hold:(k + 1) * P.twinkle_hold] = f
    return out.reshape(P.frames, h, w)


def drip(frames: np.ndarray, P: Glow, seed: int) -> np.ndarray:
    """Dotted lines falling from the picture's lower edges: every other dot of a column lit, in the
    colour of the dot it hangs from, growing downward, then thinning out; each in its own time, so
    the loop has no seam. Drawn on black only."""
    T, h, w = frames.shape
    still = frames[0]
    lit = still > 0
    if not P.drips or T < 2 or not lit.any():
        return frames
    rng = np.random.default_rng(seed + 2)
    # a lower edge: a lit dot of the picture (not a stray one) with open black under it, the more the better
    solid = cv2.blur(lit.astype(np.float32), (5, 5)) > 0.3
    ys = xs = np.zeros(0, int)
    for reach in (P.drip_len[1] // 2, P.drip_len[0]):
        clear = np.ones((h, w), bool)
        for k in range(1, reach + 1):
            clear[:h - k] &= ~lit[k:]
            clear[h - k:] = False
        ys, xs = np.where(lit & clear & solid & (np.arange(w)[None, :] % 2 == 0))
        if len(ys) >= P.drips:
            break
    if not len(ys):
        return frames
    pick = rng.choice(len(ys), size=min(P.drips, len(ys)), replace=False)
    cols = []
    for y, x in zip(ys[pick], xs[pick]):
        cols.append((y, x, still[y, x]))
        if rng.random() < P.drip_bundle and x + 2 < w:
            cols.append((y + int(rng.integers(-1, 2)), x + 2, still[y, x]))
    out = frames.copy()
    black = ~lit
    for y, x, colour in cols:
        # one life per loop: it falls, hangs, thins out, and is gone when the loop comes round
        fade = 0.25 * T
        length = min(int(rng.integers(P.drip_len[0], P.drip_len[1] + 1)), max(2, int(0.6 * T * P.drip_speed)))
        grow = length / P.drip_speed
        hold = T - grow - fade
        phase, stagger = rng.random() * T, int(rng.integers(0, 2))
        order = rng.permutation(length)                    # the order its dots go out in
        for f in range(T):
            t = (f + phase) % T
            n = min(length, int(t * P.drip_speed))
            gone = int(np.clip((t - grow - hold) / fade, 0, 1) * length)
            for k in range(stagger, n, 2):
                yy = y + 1 + k
                if yy < h and black[yy, x] and order[k] >= gone:
                    out[f, yy, x] = colour
    return out


def glow(A: dict, P: Glow | None = None, seed: int = 1) -> Frames:
    """The clip: the light levelled, run hotter where it gathers and round its source, coloured
    through one ramp, pattern-dithered onto the palette; the shade's sparse dots blink."""
    P = P or Glow()
    rgba = A['rgba']
    v0 = light(rgba)
    v = np.clip(exposed(v0, P) * P.gain, 0, 1)
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
    xs = np.array([s for s, _, _ in P.ramp], np.float32)
    mix = np.array([m for _, _, m in P.ramp], np.float32)
    mix[:, 2:4] *= P.warm                      # more or less of it warm, the rest made up in blue
    mix[:, 0] = np.maximum(1 - mix[:, 1:].sum(1), 0)
    mix = mix / mix.sum(1, keepdims=True)
    cs = np.array([c for _, c, _ in P.ramp], np.float32)
    cover = np.interp(heat, xs, cs).astype(np.float32)
    cover = np.clip((cover - P.floor) / (1 - P.floor), 0, 1)   # the faintest light is black: no stray dots in the void
    shares = np.stack([np.interp(heat, xs, mix[:, k]) for k in range(5)], -1).astype(np.float32)
    steps, t = pattern(cover, shares, seed, P.shake)
    return drip(animate(steps, t, heat.shape, P, seed), P, seed), GLOW_PALETTE.astype(np.uint8)


# ---- streak

@dataclass
class Streak:
    frames: int = 24
    fps: int = 24
    long: int = 270
    scale: int = 4
    colour: str = '#ee1111'
    gain: float = 1.0           # more or less light
    gamma: float = 1.15         # tone curve: above 1, darker mid-tones and more contrast
    local: float = 0.5          # local contrast before the dither (light against its surroundings, 8 dots)
    most: float = 0.75          # the densest the dots get: dense in the highlights, never solid
    shades: int = 1             # 1: one ink, tone by how many dots only; 3: three shades of it
    black: float = 0.12         # light below this is black (no haze of stray dots in the shade)
    floor: float = 0.0          # and after the tone curve, light below this (no even sprinkle in the dim)
    peak: float = 0.0           # share of the figure, its brightest, dithered near solid
    top: float = 0.05           # share of the figure, its brightest, dithered denser than `most`
    top_most: float = 0.97      # how dense that gets at its brightest
    knee: float = 0.55          # above this light the density climbs smoothly past `most` to `top_most` (0: the `top` share instead)
    features: float = 1.0       # how much darker small dark spots (eyes, mouth) go before the dither
    rim: float = 0.15           # light added along the inside of the silhouette, so it holds
    smooth: float = 0.7         # blur on the light before the dither (dots): clean worms, not blotches
    even: float = 0.5           # how much of the locally evened light (CLAHE) the tone takes; the rest plain
    jitter: float = 0.2         # threshold shake in the dither: maze-like chains, not stripes and checks
    jitter_plain: float = 0.3   # the shake in plain tone, away from edges: looser specks, fewer touching
    edge_pct: float = 85.0      # the edge strength (percentile in the figure) that counts as a full edge
    kernel: str = 'atkinson'    # error diffusion: 'floyd' (Floyd-Steinberg) or 'atkinson' (diagonal weave, more contrast)
    mesh: float = 0.0           # Atkinson: how much harder a dot lights when the dot above and the one before are lit
    clump: float = 0.0          # in the brightest, holes from noise blurred this much (dots): loose pairs, not a grid
    plane: int = 120            # the least dots a patch of bright light needs to go near solid
    hold: float = 0.0           # share of the figure's dots held from one dither all loop (less boil)
    boil: float = 0.05          # noise added before each frame's dither: how much it boils
    stray: float = 0.0          # how much of that noise is let into the background
    specks: float = 0.016       # loose dots over the background, more towards the bottom
    dashes: int = 70            # short straight dashes along the shapes, each frame
    dash_len: tuple[int, int] = (3, 7)
    field: int = 40             # dashes drawing the texture the dither leaves black (waves, hair, cloth in shade)
    field_len: tuple[int, int] = (2, 5)
    rain: int = 320             # bundles of streaks across the frame
    strands: tuple[int, int] = (1, 1)   # streaks in a bundle, side by side
    rain_len: tuple[int, int] = (12, 48)
    rain_angle: float = -24     # degrees from the right, upwards negative (screen y is down)
    rain_gaps: float = 0.4      # gaps in a streak (0: solid): dotted, broken runs
    rain_alt: float = 0.1       # share of streaks with every other dot gone


def diffuse(img: np.ndarray, jitter: float = 0.0, rng=None, hold_j: float = 0.0) -> np.ndarray:
    """Floyd-Steinberg error diffusion, serpentine (each row the other way), so the dots run no
    one way: plain left-to-right scanning spaces them out across and stacks them up and down.
    `jitter` shakes the threshold, which breaks up the stripes and checks it settles into at mid grey."""
    H, W = img.shape
    e = np.zeros((H + 1, W + 2), np.float32)
    e[:H, 1:W + 1] = img
    rows = e.tolist()
    th = 0.5 + jitter * (rng.random((H, W + 2)) - 0.5) if jitter > 0 else np.full((H, W + 2), 0.5)
    th[:, 1:W + 1] = np.where(img > 0.8, 0.5 + hold_j * (th[:, 1:W + 1] - 0.5), th[:, 1:W + 1])   # the brightest calmer: solid, its pinholes loose, not a grid
    th = th.tolist()
    out = np.zeros((H, W), bool)
    for y in range(H):
        r0, r1, t0 = rows[y], rows[y + 1], th[y]
        if y % 2 == 0:
            xs, d = range(1, W + 1), 1
        else:
            xs, d = range(W, 0, -1), -1
        for x in xs:
            v = r0[x]
            if v >= t0[x]:
                out[y, x - 1] = True
                v -= 1
            if v:
                r0[x + d] += v * 0.4375; r1[x - d] += v * 0.1875; r1[x] += v * 0.3125; r1[x + d] += v * 0.0625
    return out


def diffuse_atkinson(img: np.ndarray, jitter: float | np.ndarray = 0.0, rng=None, mesh: float = 0.0) -> np.ndarray:
    """Atkinson error diffusion, serpentine: only three quarters of the error is passed on, so
    tone pulls towards dark and bright, and mid tones settle into a diagonal weave, not noise."""
    H, W = img.shape
    e = np.zeros((H + 2, W + 4), np.float32)
    e[:H, 2:W + 2] = img
    rows = e.tolist()
    if np.ndim(jitter):   # a shake per dot
        jp = np.zeros((H, W + 4), np.float32); jp[:, 2:W + 2] = jitter
        th = np.maximum(0.5 + jp * (rng.random((H, W + 4)) - 0.5), 0.04).tolist()   # never below ~0: a shake past 0 lights black dots, an even sprinkle over black ground
    else:
        th = (0.5 + jitter * (rng.random((H, W + 4)) - 0.5)).tolist() if jitter > 0 else [[0.5] * (W + 4)] * H
    out = np.zeros((H, W), bool)
    for y in range(H):
        r0, r1, r2, t0 = rows[y], rows[y + 1], rows[y + 2], th[y]
        xs, d = (range(2, W + 2), 1) if y % 2 == 0 else (range(W + 1, 1, -1), -1)
        prev = out[y - 1] if y else None
        for x in xs:
            v = r0[x]
            t = t0[x]
            if mesh and prev is not None and prev[x - 2] and out[y, x - 2 - d] if 0 <= x - 2 - d < W else False:
                t += mesh   # its side neighbours already lit: a little less likely, so the mid tones don't knit into a mesh
            if v >= t:
                out[y, x - 2] = True
                v -= 1
            if v:
                v *= 0.125
                r0[x + d] += v; r0[x + 2 * d] += v
                r1[x - d] += v; r1[x] += v; r1[x + d] += v
                r2[x] += v
    return out


def streak(A: dict, P: Streak | None = None, seed: int = 1) -> Frames:
    P = P or Streak()
    rng = np.random.default_rng(seed)
    a = A['alpha']
    t = (1 - P.even) * A['plain'] + P.even * A['tone']   # some of the plain light: real shadows stay dark
    H, W = t.shape
    c = np.float32(hex_rgb(P.colour))
    pal = np.array([[0, 0, 0], c * 0.62, c * 0.8, c], np.uint8)       # three shades of the one colour
    tt = np.clip(P.gain * np.clip((t - P.black) / (1 - P.black), 0, 1) ** P.gamma, 0, 1)
    if P.local > 0:   # local contrast: tone pulls apart into lit planes and shade, not one flat mid band
        tt = np.clip(tt + P.local * (tt - cv2.GaussianBlur(tt, (0, 0), 8)), 0, 1)
    if (a > 0.5).any():   # the brightest of the figure at full light, whatever the picture
        tt = np.clip(tt / max(np.percentile(tt[a > 0.5], 99), 0.3), 0, 1)
    if P.peak > 0 and (a > 0.5).any():   # the brightest light near solid, as on a hat brim or a lit cheek
        q = np.percentile(tt[a > 0.5], 100 * (1 - P.peak))
        tt = np.where(tt > q, np.maximum(tt, np.clip(0.88 + (tt - q) / max(1 - q, 1e-3), 0, 1)), tt)
    top = np.zeros_like(tt)
    if P.top > 0 and (a > 0.5).any():   # the very brightest a little denser than the cap, still not solid
        qt = np.percentile(tt[a > 0.5], 100 * (1 - P.top))
        top = np.clip((tt - qt) / max(1 - qt, 1e-3), 0, 1)
    if P.knee > 0:   # a shoulder: from the knee up, the light climbs past `most` to `top_most`: lit planes go near solid, the mid-tones stay as they were
        x = np.clip((tt - P.knee) / (1 - P.knee), 0, 1)
        tt = (np.where(tt < P.floor, 0, tt) * P.most + (P.top_most - P.most) * x * x * (3 - 2 * x)).astype(np.float32)
    else:
        tt = (np.where(tt < P.floor, 0, tt) * (P.most + (P.top_most - P.most) * top)).astype(np.float32)
    if P.smooth > 0:   # the source's fine grain out first, or the diffusion draws it as blotches
        tt = cv2.GaussianBlur(tt, (0, 0), P.smooth)
    if P.features > 0:   # eyes, nostrils, mouth: small dark spots go darker, so they read as holes
        k = max(3, int(round(0.03 * max(H, W))) | 1)
        th = cv2.morphologyEx(cv2.GaussianBlur(tt, (0, 0), 1.5), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if (a > 0.5).any():   # only the real features, not the fine texture all over
            th = np.where(th > np.percentile(th[a > 0.5], 85), th, 0)
        tt = np.clip(tt - P.features * th, 0, 1).astype(np.float32)
    if P.rim > 0 and 0.02 < (a > 0.5).mean() < 0.97:   # the silhouette diffuses solid, not frayed
        m = (a > 0.5).astype(np.uint8)
        band = (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool) & (tt > 0.1)
        tt = np.where(band, np.minimum(tt + P.rim, P.most), tt).astype(np.float32)
    shade = (1 + (tt > 0.42 * P.most) + (tt > 0.72 * P.most)).astype(np.uint8) if P.shades > 1 else np.full((H, W), 3, np.uint8)
    # the boil only where there's light (and a `stray` share of it anywhere): black stays black,
    # cut out or not
    room = (P.stray + (1 - P.stray) * a * np.clip(tt * 10, 0, 1)).astype(np.float32)

    w_dash = (A['edge'] ** 3 * A['coh'] * (tt > 0.15)).astype(np.float32)
    w_field = (A['edge'] * A['coh'] * (tt <= 0.15) * (A['plain'] > 0.02) * (a > 0.5)
               * (cv2.GaussianBlur(tt, (0, 0), 3) > 0.05)).astype(np.float32)
    d = math.radians(P.rain_angle)
    dv = np.float32([math.cos(d), math.sin(d)])
    span = math.hypot(W, H) + 2 * P.rain_len[1]
    perp = np.float32([-dv[1], dv[0]])
    mid = np.float32([W / 2, H / 2])
    # bundles of 1-3 strands side by side, each a dotted run whose gaps travel with it
    nb = rng.integers(P.strands[0], P.strands[1] + 1, P.rain)
    bi = np.repeat(np.arange(P.rain), nb)
    k_in = np.concatenate([np.arange(n) for n in nb]) if P.rain else np.zeros(0, int)
    nr = len(bi)
    across = (rng.uniform(-span / 2, span / 2, P.rain)[bi] + 1.0 * k_in).astype(np.float32)
    r_across, r_along = across.copy(), ((rng.random(len(across)) - 0.5) * span).astype(np.float32)
    rl = rng.integers(P.rain_len[0], P.rain_len[1] + 1, nr)
    jj = np.arange(P.rain_len[1])[None, :]
    alt = (rng.random(nr) < P.rain_alt)[:, None] & (jj % 2 == 1) & (rng.random((nr, 1 + 0 * jj.shape[1])) > 0.25)          # some every other dot
    gaps = rng.random((nr, P.rain_len[1])) < P.rain_gaps + 0.8 * np.abs(2 * jj / rl[:, None] - 1) ** 3   # thinning at both ends
    gaps |= alt
    yy = np.mgrid[0:H, 0:W][0]
    low = (0.2 + 1.8 * (yy / H) ** 2).astype(np.float32)          # rain and specks heavier lower down
    hold_m = rng.random((P.frames, H, W)) < P.hold if P.hold > 0 else None
    busy = np.clip(cv2.GaussianBlur(tt, (0, 0), 6) * 3, 0, 1)        # where the figure is lit and dense

    big_light = np.zeros((H, W), bool)
    if P.clump > 0:   # solid only on broad planes of light: small bright islands stay dithered
        bl = (cv2.GaussianBlur(tt, (0, 0), 1.5) > 0.78 * P.most).astype(np.uint8)
        nb_, lab_, st_, _ = cv2.connectedComponentsWithStats(bl, connectivity=8)
        keep = st_[:, 4] >= P.plane; keep[0] = False
        big_light = cv2.dilate((keep[lab_]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    # the threshold shake: more in plain tone (loose, maze-like specks), less on edges (crisp features)
    ev = cv2.GaussianBlur(A['edge'], (0, 0), 1.5)
    ev = np.clip(ev / (np.percentile(ev[a > 0.5], P.edge_pct) + 1e-6), 0, 1) ** 2 if (a > 0.5).any() else np.zeros_like(ev)
    jmap = (P.jitter + (P.jitter_plain - P.jitter) * (1 - ev) * (a > 0.5)).astype(np.float32)   # off the figure: as ever
    out = np.zeros((P.frames, H, W), np.uint8)
    for f in range(P.frames):
        s = f / P.frames
        noisy = np.clip(tt + rng.normal(0, P.boil, (H, W)).astype(np.float32) * room, 0, 1)
        o = f % 4                       # Floyd-Steinberg runs one way: turn it each frame so no direction shows
        nz = noisy.T if o & 2 else noisy
        nz = nz[:, ::-1] if o & 1 else nz
        jz = jmap.T if o & 2 else jmap
        jz = jz[:, ::-1] if o & 1 else jz
        bits = (diffuse_atkinson(np.ascontiguousarray(nz), np.ascontiguousarray(jz), rng, P.mesh) if P.kernel == 'atkinson'
                else diffuse(np.ascontiguousarray(nz), P.jitter, rng))
        bits = bits[:, ::-1] if o & 1 else bits
        bits = bits.T if o & 2 else bits
        if P.clump > 0:   # the brightest: holes from lightly clumped noise, in pairs and short runs, not a lattice
            cn = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), P.clump)
            cn = (np.argsort(np.argsort(cn, axis=None)).reshape(H, W) / cn.size).astype(np.float32)
            dense = (tt > P.most * (0.74 + 0.12 * rng.random((H, W)))) & big_light   # its edge ragged, not a contour
            bits = np.where(dense, cn < np.clip(tt / P.most, 0, 1) * 0.9, bits)
        if P.hold > 0:   # some of the figure holds still all loop: it shimmers rather than boils
            if f == 0:
                base = bits.copy()
            bits = np.where(hold_m[f], base, bits)
        img = np.where(bits, shade, 0).astype(np.uint8)
        # dashes: placed afresh each frame along the shapes; bright on the lit figure, and in the
        # mid shade where they stand for texture the dither left black
        for w, n, (l0, l1), sh in ((w_field, P.field, P.field_len, 2 if P.shades > 1 else 3), (w_dash, P.dashes, P.dash_len, 3)):
            seeds = seed_where(w, n, rng)
            if not len(seeds):
                continue
            ix, iy = seeds[:, 0].astype(int), seeds[:, 1].astype(int)
            tv = np.stack([A['tx'][iy, ix], A['ty'][iy, ix]], 1)
            ln = rng.integers(l0, l1 + 1, len(seeds))
            for j in range(l1):
                on = j < ln
                plot(img, (seeds + tv * (j - ln[:, None] / 2))[on], sh)
        # rain, straight across, wrapping round in its own time
        # most streaks stay near where they were (a dot across, a few along), a third start afresh
        new = rng.random(nr) < 0.3
        r_along = np.where(new, (rng.random(nr) - 0.5) * span, r_along + rng.uniform(-6, 6, nr)).astype(np.float32)
        r_across = np.where(new, rng.uniform(-span / 2, span / 2, nr), r_across + rng.integers(-1, 2, nr)).astype(np.float32)
        along, across = r_along, r_across
        # each streak shows or not this frame, more often lower down
        cy = np.clip(np.round(mid[1] + perp[1] * across + dv[1] * along), 0, H - 1).astype(int)
        cx = np.clip(np.round(mid[0] + perp[0] * across + dv[0] * along), 0, W - 1).astype(int)
        show = rng.random(nr) < np.clip(0.4 + 0.8 * cy / H, 0, 1) * (1 - 0.85 * busy[cy, cx])   # mostly off the figure's mass
        for j in range(P.rain_len[1]):
            pts = mid + perp * across[:, None] + dv * (along - j)[:, None]
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            on = (j < rl) & ~gaps[:, j] & show
            plot(img, pts[on], 3)
        sp = (img == 0) & (rng.random((H, W)) < P.specks * low)
        img[sp] = 3
        # some specks are short flat dashes, two or three dots
        sy, sx = np.nonzero(sp & (rng.random((H, W)) < 0.15))
        for k in (1, 2):
            keep = rng.random(len(sx)) < 0.6 ** k
            sx, sy = sx[keep], sy[keep]
            plot(img, np.stack([sx, sy], 1) + dv * k, 3)   # along the rain
        out[f] = img
    return out, pal


@dataclass
class Dust:
    frames: int = 48
    fps: int = 24
    long: int = 540
    scale: int = 2
    colour: str = '#f2f2f2'
    gain: float = 1.0           # more or less light
    gamma: float = 1.6          # tone curve: lower keeps the mid-tones, so the form reads as in the references
    black: float = 0.12         # light below this is black
    most: float = 0.75          # the densest the specks get, in the brightest light
    core: float = 0.5           # extra density in the brightest light (on tt^4): hot cores, near solid, as the references' lit faces
    break_flecks: int = 2500    # where the figure comes apart, its light as streaks along the drift (count for 540x540): solid, then streaks, then motes
    break_len: tuple[int, int] = (3, 7)
    beads: int = 0            # larger round bright dots over the grain, in the light (count for 540x540): a few big marks among many fine
    tiles: float = 0.0          # share of the lit body's light carried by small solid tiles (2x2, 3x3 in the brightest), each one grey, black round it
    tile_from: float = 0.3      # the light (tt) where tiles start, full by tile_from + 0.5
    core_most: float = 0.95     # the densest anything gets (was 0.7)
    clump: float = 1.0          # how much the specks clump (0: even white noise): grains, not a sieve
    clump_size: float = 0.5     # the blur that clumps them (dots): finer keeps small features
    grain_wide: float = 1.0     # how much wider than tall the grains are (1: round)
    lone: float = 1.0           # share of specks touching no other at their sides that go
    unplus: float = 1.0         # share of '+'-shaped grains given a corner, so they read as blocks
    near_rim: float = 0.0       # more of the silhouette drawn on the side that holds (away from the dissolve): crisp there, blown out on the far side
    outline: float = 0.3        # how much of the silhouette and strongest edges is drawn as bright strokes
    dust_floor: float = 0.1     # the least density in the intact figure (a cut-out only): its shade doesn't go black
    pair: float = 0.6           # share of lone body specks given a neighbour along the shapes
    shade_lev: float = 0.6      # how far each speck's grey follows the light (0: any grey)
    grey_spread: float = 0.15   # how far each speck's grey strays from its step (0: a few flat steps)
    grey_shift: float = -0.05   # all the specks' greys moved this far (darker below 0)
    grey_patch: float = 0.16    # how far greys shift patch by patch (0: no patches)
    grey_patch_size: float = 12.0  # how wide the grey patches are (dots of blur)
    order: float = 0.15         # how much of the body is an ordered (Bayer) dither, not noise: a little keeps features crisp
    edges: float = 0.35         # specks gathering on the outline (and less, the edges inside)
    flecks: int = 5000          # short flecks, placed afresh each frame (counts are for 540x540)
    fleck_len: tuple[int, int] = (2, 3)
    fleck_turn: float = 0.5     # share of flecks pointing any way, not along the shapes
    fleck_join: float = 1.0     # share of slanted flecks drawn as steps that touch side to side, not corner to corner
    fleck_lo: float = 0.5       # the dimmest a fleck's grey goes (share of the greys; 0: any)
    angle: float = 0            # the way the dust drifts off (degrees: 0 right, 90 down)
    dissolve: float = 0.97      # how much of the far side's body is gone (to motes)
    spare_lit: bool = True      # drift the other way if the far side holds the most light
    side: float = 0.7           # how much of the figure, from its far side along the angle, dissolves
    motes: int = 2500           # particles drifting off it
    reach: int = 150            # how far they go in a loop, in dots
    trail: int = 16             # motes to a trail, following one another
    curl: float = 0.1           # how much they follow the shapes they come from, not the angle
    blobs: int = 60             # soft grey blots among them
    sparkles: int = 3           # four-point glints on the brightest light, each for part of the loop
    specks_glint: int = 15      # small glints out in the drift, each twinkling for part of the loop
    haze: float = 0.12          # faint dark-grey dots round the lit ones (0: none)
    glow: float = 0.35          # a dim glow round the specks instead, as bright as theirs blurred (0: the sparse haze)
    glow_size: float = 2.0      # its blur (dots)
    fine: int = 64              # greys in the fine ramp the specks are shaken onto
    fine_reflect: bool = True   # specks shaken under the ramp bounce back up, not pile at its foot
    flat: bool = False          # one flat grey for everything (no shades, no haze), as some dust art is
    checker: float = 0.25       # flat only: share of the mid-tone body drawn as patches of checkerboard dither
    checker_from: float = 0.2   # the local speck density where mid tones start (share of dots lit, 9x9)
    mote_len: tuple[int, int] = (3, 14)  # dots in a mote, least and most: a short dash along its way
    mote_tail: float = 0.3     # how quickly long motes get rare (geometric: lower, more long ones)
    accel: float = 1.8          # motes move as u**accel round the loop (above 1: slow off the figure, then faster)
    throw: float = 0.03         # share of trails thrown 1.5-2.5x as far
    falloff: float = 2.5        # how fast motes thin out as they go
    swirl: float = 12.0         # how far the drift turns region by region (degrees, typical)
    thick: float = 0.2          # share of motes two dots thick
    lanes: float = 0.7          # how much the motes gather in lanes along the drift (0: evenly)
    jitter_size: float = 15.0   # how wide the regions are that stray together (dots)
    jitter: float = 6.0        # how far each mote's heading strays from its lane's (degrees, typical)
    bend: float = 0.2           # how far the lanes curve off the angle
    comb: float = 0.2           # how far the flecks lie along the drift, not the shapes, inside the figure


def dust(A: dict, P: Dust | None = None, seed: int = 1) -> Frames:
    """The figure as boiling specks of many greys, densest in the light, with short flecks along
    its shapes; from one side it comes apart into particles drifting off along the angle."""
    P = P or Dust()
    rng = np.random.default_rng(seed)
    t, a = A['tone'], A['alpha']
    H, W = t.shape
    c = np.float32(hex_rgb(P.colour))
    cmax = max(float(c.max()), 1.0)
    lv = np.float32([0, 0.2, 0.32, 0.45, 0.6, 0.8, 1.0])
    NL = len(lv) - 1
    pal = (np.concatenate([lv, [0.06, 0.13]])[:, None] * c[None]).astype(np.uint8)   # and two faint greys past the ramp, for the haze
    HAZE = (NL + 1, NL + 2)
    if P.flat:   # one flat tone for every speck, stroke and mote, and no haze: the even grey kind of dust
        pal[1:NL + 1] = (0.86 * c).astype(np.uint8)
    FINE, FINE_LO = (0 if P.flat or P.grey_spread <= 0 else P.fine), 0.25
    if FINE:   # a fine ramp of greys past the steps (and the haze), that each speck's grey is shaken onto
        pal = np.concatenate([pal, (np.linspace(FINE_LO, 1.0, FINE)[:, None] * c[None]).astype(np.uint8)])
        gshake = (P.grey_shift + rng.normal(0, P.grey_spread, (H, W))).astype(np.float32)   # fixed: a still speck keeps its grey
        if P.grey_patch > 0:   # and patches of lighter and darker grey, a few dots across, under the speckle
            gp = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), P.grey_patch_size)
            gshake += P.grey_patch * gp / (gp.std() + 1e-6)
    GL0, GLN = len(pal), 12
    if P.glow > 0 and not P.flat:   # and a ramp of dim greys for the glow
        pal = np.concatenate([pal, (np.linspace(0.04, 0.2, GLN)[:, None] * c[None]).astype(np.uint8)])
    tt = np.clip(P.gain * np.clip((t - P.black) / (1 - P.black), 0, 1) ** P.gamma, 0, 1).astype(np.float32)
    if (a > 0.5).any():   # the figure's brightest at full light, whatever the picture
        tt = np.clip(tt / max(float(np.percentile(tt[a > 0.5], 98)), 0.2), 0, 1).astype(np.float32)

    # how far each px is along the angle across the figure: its far side comes apart
    th = math.radians(P.angle)
    dv = np.float32([math.cos(th), math.sin(th)])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    proj = xx * dv[0] + yy * dv[1]
    inside = a > 0.5
    if inside.any():
        lo, hi = np.percentile(proj[inside], [2, 98])
    else:
        lo, hi = proj.min(), proj.max()
    pn = np.clip((proj - lo) / max(hi - lo, 1), 0, 1)
    if P.spare_lit and inside.any():   # the lit side (a face, mostly) stays; the darker side goes
        g = tt * inside
        far, near = float((g * (pn > 0.5)).sum()), float((g * (pn <= 0.5)).sum())
        if far > near:
            dv = -dv
            proj = -proj
            lo, hi = -hi, -lo
            pn = np.clip((proj - lo) / max(hi - lo, 1), 0, 1)
    sw = (np.clip((pn - (1 - P.side)) / max(P.side, 1e-3), 0, 1) ** 1.3 * (a > 0.2)).astype(np.float32)
    ed = A['edge'] * np.clip(A['plain'] * 4, 0, 1) * (a > 0.5)
    sil = cv2.GaussianBlur(np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)), (0, 0), 0.7)
    sil = np.clip(sil / (sil.max() + 1e-6) * 3, 0, 1) * (a > 0.3)
    sil[:3], sil[-3:], sil[:, :3], sil[:, -3:] = 0, 0, 0, 0   # the frame's edge is no outline
    m_ = a > 0.5
    if m_.any():   # a whole picture framed smaller (its alpha a plain rectangle) is not cut out either
        ys_, xs_ = np.nonzero(m_)
        boxed = m_.sum() / ((ys_.max() - ys_.min() + 1) * (xs_.max() - xs_.min() + 1)) > 0.97
    else:
        boxed = False
    cut = (not boxed) and m_.mean() < 0.97
    if not cut:   # nothing cut out: no silhouette to draw
        sil[:] = 0
    dens = (np.clip(tt * P.most + P.core * tt ** 4 + P.edges * (0.7 * sil + 0.4 * ed ** 1.5 * (0.25 + 0.75 * tt)), 0, P.core_most) * (1 - P.dissolve * sw)).astype(np.float32)
    if P.dust_floor > 0 and cut and 0.02 < (a > 0.5).mean():   # the shade of the intact figure keeps a sparse grain
        fz = (dens < P.dust_floor) & (a > 0.5) & (sw < 0.2)   # where the floor holds: its specks stay, lone or not
        dens = np.maximum(dens, P.dust_floor * (a > 0.5) * np.clip(1 - 3 * sw, 0, 1)).astype(np.float32)
    else:
        fz = np.zeros((H, W), bool)
    many = H * W / 540 ** 2                                  # the counts, for this size

    bay = bayer(H, W)
    ck = (np.add.outer(np.arange(H), np.arange(W)) % 2 == 0)
    k8 = np.ones((3, 3), np.float32); k8[1, 1] = 0
    strong_ed = (ed > np.percentile(ed[a > 0.5], 97)) & (sw < 0.5) if (a > 0.5).any() else np.zeros_like(a, bool)
    if not cut:   # nor the edge of the picture's own frame
        strong_ed &= cv2.erode(m_.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    rim = ((sil > 0.5) | strong_ed) & (a > 0.3)   # the outline holds even where the figure comes apart

    def flat_noise():
        n = rng.random((H, W)).astype(np.float32)
        if P.clump > 0:
            n = (1 - P.clump) * n + P.clump * cv2.GaussianBlur(n, (0, 0), P.clump_size * P.grain_wide, sigmaY=P.clump_size)
            n = (np.argsort(np.argsort(n, axis=None)).reshape(H, W) / n.size).astype(np.float32)
        return n

    # the motes: born on the figure (mostly its far side), drifting off along the angle, bending
    # with the shapes they leave, fading in and out; each lives once round the loop, so it loops
    wm = (tt + 0.15 * a + 0.5 * sil) * (0.01 + sw) * (a > 0.2) * (0.4 + 1.2 * yy / H)   # more of them lower down
    # they go in trails: a few share each starting point and path, at different times round it
    nm = int(P.motes * many)
    if P.lanes > 0:   # they gather in lanes along the drift, with empty lanes between
        acr = xx * -dv[1] + yy * dv[0]
        kn = np.arange(-2, int((W + H) / 10) + 3) * 10.0
        band = np.interp(acr - acr.min(), kn - kn[0] + 0, cv2.GaussianBlur(rng.random((len(kn), 1)).astype(np.float32), (0, 0), 1.2).ravel())
        band = (band - band.min()) / (np.ptp(band) + 1e-6)
        wm = wm * ((1 - P.lanes) + P.lanes * 2 * np.clip(band - 0.25, 0, 1))
    src = seed_where(wm.astype(np.float32), max(1, nm // P.trail), rng)
    k = len(src)
    ix, iy = src[:, 0].astype(int), src[:, 1].astype(int)
    # the shapes' broad run (orientations, smoothed, so neighbouring trails go side by side)
    c2 = cv2.GaussianBlur(A['tx'] ** 2 - A['ty'] ** 2, (0, 0), 60)
    s2 = cv2.GaussianBlur(2 * A['tx'] * A['ty'], (0, 0), 60)
    ang2 = 0.5 * np.arctan2(s2, c2)
    tv = np.stack([np.cos(ang2)[iy, ix], np.sin(ang2)[iy, ix]], 1).astype(np.float32)
    tv *= np.where((tv @ dv) < 0, -1, 1)[:, None]          # the tangent pointing the way of the drift
    tv = np.where((np.abs(tv @ dv) > 0.6)[:, None], tv, dv)   # and only where it runs roughly that way
    sv = (1 - P.curl) * dv + P.curl * tv
    if P.swirl > 0:   # the field turns a little region by region: lanes swirl, not ripple in step
        sn = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), max(H, W) / 8)
        sn = sn / (sn.std() + 1e-6) * math.radians(P.swirl)
        ang = sn[iy, ix]
        c, s_ = np.cos(ang)[:, None], np.sin(ang)[:, None]
        sv = np.concatenate([sv[:, :1] * c - sv[:, 1:] * s_, sv[:, :1] * s_ + sv[:, 1:] * c], 1)
    sv /= np.linalg.norm(sv, axis=1, keepdims=True) + 1e-6
    sfar = (P.reach * math.sqrt(many) * rng.uniform(0.4, 1, k) * np.where(rng.random(k) < P.throw, rng.uniform(1.5, 2.5, k), 1) * (0.4 + 0.6 * sw[iy, ix])).astype(np.float32)
    sbend = rng.normal(0, P.bend, k).astype(np.float32)     # a slow drift sideways too: the lanes curve
    who = rng.integers(0, k, nm) if len(src) else np.zeros(0, int)
    n = len(who)
    m0 = src[who] + rng.normal(0, 0.6, (n, 2)).astype(np.float32)
    swob = rng.random(k).astype(np.float32) * 6.283           # and a slow wave across the way they go
    mv, far, bend, wob = sv[who], sfar[who], sbend[who], swob[who]
    if P.jitter > 0:   # each its own heading, a little off its lane's: not blinds
        jn = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), P.jitter_size)   # smooth: neighbours run together
        jn = jn / (jn.std() + 1e-6) * math.radians(P.jitter)
        q = np.clip(np.round(m0).astype(int), 0, [W - 1, H - 1])
        ja = (jn[q[:, 1], q[:, 0]] + rng.normal(0, math.radians(P.jitter) * 0.15, n)).astype(np.float32)
        c, s_ = np.cos(ja)[:, None], np.sin(ja)[:, None]
        mv = np.concatenate([mv[:, :1] * c - mv[:, 1:] * s_, mv[:, :1] * s_ + mv[:, 1:] * c], 1)
    perp = np.float32([-dv[1], dv[0]])
    mph = rng.random(n).astype(np.float32)
    mlen = np.minimum(P.mote_len[0] - 1 + rng.geometric(P.mote_tail, n), P.mote_len[1])   # mostly short, a long tail of long ones
    mlev = np.full(n, NL - 2)   # a mid grey, under the figure's highlights: the drift never upstages the face
    mhead = rng.random(n) < 0.5   # but half of them lead with a bright dot, as the references' sweep does
    thick = rng.random(n) < P.thick
    blob = rng.random(n) < P.blobs * many / max(n, 1)
    bdots = rng.uniform(-3.5, 3.5, (n, 12, 2)).astype(np.float32)   # a blob: a loose round cluster
    bdots[np.hypot(bdots[..., 0], bdots[..., 1]) > 3.5] = np.nan
    blev = rng.integers(2, 5, (n, 12)).astype(np.uint8)
    # flecks run mostly the way of the drift, bent a little to the shapes
    fdir = np.stack([A['tx'], A['ty']], -1)
    fdir *= np.where((fdir @ dv) < 0, -1, 1)[..., None]
    fdir = (P.comb + (1 - P.comb) * sw)[..., None] * dv + ((1 - P.comb) * (1 - sw))[..., None] * fdir   # combed by the drift where it comes apart
    fdir /= np.linalg.norm(fdir, axis=-1, keepdims=True) + 1e-6

    top_lit = tt > (np.percentile(tt[a > 0.5], 90) if (a > 0.5).any() else 1)
    fstep = np.round(fdir * 1.2).astype(int)   # the neighbouring dot along the shapes, combed by the drift where it comes apart
    fstep[(fstep[..., 0] == 0) & (fstep[..., 1] == 0)] = (1, 0)
    spk = seed_where((tt ** 4 * (a > 0.5) * (sw < 0.6)).astype(np.float32), max(0, int(round(P.sparkles * many))), rng)
    sph, sln = rng.random(len(spk)), rng.integers(2, 7, len(spk))   # a few long-armed, most small
    sln = np.minimum(sln, rng.integers(2, 7, len(spk)))
    if n and P.specks_glint > 0:   # small glints where the motes go, out in the drift
        gi = rng.choice(n, min(n, int(round(P.specks_glint * many))), replace=False)
        gpos = m0[gi] + mv[gi] * (far[gi] * rng.uniform(0.15, 0.6, len(gi)))[:, None]
        gph, garm, glev = rng.random(len(gi)), rng.integers(1, 3, len(gi)), rng.integers(NL - 2, NL, len(gi))
    else:
        gpos = np.zeros((0, 2), np.float32); gph = garm = glev = np.zeros(0)
    inner = cv2.erode((a > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (sw < 0.3)
    off_near = cv2.dilate((a > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    nearw = (np.clip(1 - pn / 0.6, 0, 1) * (sil > 0.5)).astype(np.float32)   # the near side of the outline, fading by 60% across
    bd = seed_where((np.sqrt(tt) * np.clip(0.6 - dens, 0, 1) * (dens > 0.08) * (a > 0.5) * np.clip(1 - 2 * sw, 0, 1)).astype(np.float32), int(P.beads * many), rng) if P.beads > 0 else np.zeros((0, 2), np.float32)
    bd = np.round(bd).astype(int) if len(bd) else np.zeros((0, 2), int)
    bph = rng.random(len(bd)); bbig = rng.random(len(bd)) < 0.35
    bz = (P.tiles * np.clip((tt - P.tile_from) / 0.5, 0, 1) * (a > 0.5) * np.clip(1 - 2 * sw, 0, 1)).astype(np.float32) if P.tiles > 0 else None
    out = np.zeros((P.frames, H, W), np.uint8)
    for f in range(P.frames):
        s = f / P.frames
        # the body: specks where noise falls under the light, each its own grey
        lit = ((1 - P.order) * flat_noise() + P.order * bay) < dens
        lev = rng.integers(1, NL + 1, (H, W))
        if P.shade_lev > 0:   # brighter specks where it's lit: the greys follow the light, not chance
            lev = np.clip(np.round((1 - P.shade_lev) * lev + P.shade_lev * (1.5 + (NL - 1) * np.sqrt(tt))), 1, NL).astype(int)
            lev = np.where(top_lit, lev, np.minimum(lev, NL - 1))   # white only in the very brightest
        if P.outline > 0:   # the silhouette and strongest edges as bright strokes; stray dim specks go
            lit &= ~((lev <= 3) & (cv2.filter2D(lit.astype(np.uint8), -1, k8, borderType=cv2.BORDER_CONSTANT) == 0) & ~fz)
            stroke = rim & (rng.random((H, W)) < (P.outline + P.near_rim * nearw) * (1 - 0.8 * sw))   # thinning where the figure comes apart
            lit |= stroke
            lev = np.where(stroke, np.where(top_lit, NL, NL - 1), lev)
        if P.pair > 0:   # lone body specks move next to a neighbour along the shapes: short dashes, not a checker mesh
            k4 = np.float32([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            lone = lit & (cv2.filter2D(lit.astype(np.uint8), -1, k4, borderType=cv2.BORDER_CONSTANT) == 0) & (rng.random((H, W)) < P.pair)
            ly, lx = np.nonzero(lone)
            ty_, tx_ = np.clip(ly + fstep[ly, lx, 1], 0, H - 1), np.clip(lx + fstep[ly, lx, 0], 0, W - 1)
            lit[ly, lx] = False   # moved, not added: the same light
            lit[ty_, tx_] = True
            lev[ty_, tx_] = lev[ly, lx]
        if P.lone > 0:   # a speck on its own, touching none: mostly gone, so the body is grains, not a sieve
            k4_ = np.float32([[0, 1, 0], [1, 0, 1], [0, 1, 0]])   # none at its sides: alone, or only touching at a corner
            alone = lit & (cv2.filter2D(lit.astype(np.uint8), -1, k4_, borderType=cv2.BORDER_CONSTANT) == 0) & (rng.random((H, W)) < P.lone)
            lit &= ~alone
        if P.flat and P.checker > 0:   # mid tones in patches of checkerboard, as flat dust art dithers them
            ld = cv2.blur(lit.astype(np.float32), (9, 9))
            pn = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 3)
            mid = (ld > P.checker_from) & (ld < 0.6) & inner
            cz = mid & (pn > np.quantile(pn[mid], 1 - P.checker)) if mid.any() else mid
            cz &= rng.random((H, W)) < 0.92   # a few gaps: worn, not printed
            lit = np.where(cz, ck, lit)
            lev = np.where(cz, NL, lev)
        img = np.where(lit, lev, 0).astype(np.uint8)
        tl = []
        if bz is not None:   # part of the light as small solid tiles, each one grey with black round it: marks, not grain
            lit_k = rng.random((H, W)) >= bz
            img[~lit_k & ~rim] = 0
            ty0, tx0 = np.nonzero(rng.random((H, W)) < bz * dens / 4.5)
            if len(ty0):
                sz = np.where(tt[ty0, tx0] > 0.75, np.where(rng.random(len(ty0)) < 0.4, 3, 2), 2)
                for z in (2, 3):   # a black ring round each, then the tile
                    m = sz == z
                    for oy in range(-1, z + 1):
                        for ox in range(-1, z + 1):
                            img[np.clip(ty0[m] + oy, 0, H - 1), np.clip(tx0[m] + ox, 0, W - 1)] = 0
                tlev = np.clip(lev[ty0, tx0] + 1, 1, NL).astype(np.uint8)
                for z in (2, 3):
                    m = sz == z
                    for oy in range(z):
                        for ox in range(z):
                            img[np.clip(ty0[m] + oy, 0, H - 1), np.clip(tx0[m] + ox, 0, W - 1)] = tlev[m]
                tl = (ty0, tx0, sz)
        # flecks along the shapes, mostly in the lit and on edges
        wf = (A['edge'] * A['coh'] * tt + 0.3 * tt) * dens
        fs = seed_where(wf.astype(np.float32), int(P.flecks * many), rng)
        if len(fs):
            fx, fy = fs[:, 0].astype(int), fs[:, 1].astype(int)
            fv = fdir[fy, fx].copy()
            if P.fleck_turn > 0:   # some point any way: ticks, not a combed hatch
                tw = rng.random(len(fs)) < P.fleck_turn
                an = rng.uniform(0, np.pi, tw.sum())
                fv[tw] = np.stack([np.cos(an), np.sin(an)], 1)
            ln = rng.integers(P.fleck_len[0], P.fleck_len[1] + 1, len(fs))
            fl = rng.integers(max(2, int(P.fleck_lo * NL)), NL, len(fs)).astype(np.uint8)   # never white: that's for the brightest
            wig = rng.normal(0, 0.15, (len(fs), 2)).astype(np.float32)
            jn = rng.random(len(fs)) < P.fleck_join if P.fleck_join > 0 else np.zeros(len(fs), bool)
            prev = None
            for j in range(P.fleck_len[1]):
                on = j < ln
                pt = np.round(fs + (fv + wig) * (j - ln[:, None] / 2))
                plot(img, pt[on], fl[on])
                if prev is not None:   # a step between corner-to-corner dots, so the fleck is one piece
                    st = on & jn & (np.abs(pt - prev).sum(1) > 1.5)
                    plot(img, np.stack([pt[:, 0], prev[:, 1]], 1)[st], fl[st])
                prev = pt
        if P.break_flecks > 0 and cut:   # (a cut-out only: on a whole picture the far side is just its frame) where it comes apart: streaks along the drift, brighter where the light was
            bw = (tt ** 0.7 + 0.15 * (a > 0.2) * cut) * np.clip((sw - 0.35) / 0.3, 0, 1) * np.clip((1 - sw) * 3, 0, 1) * (a > 0.2) * np.where(top_lit & (sw < 0.7), 0.15, 1)   # from the trailing part only, and mostly off the brightest light (a face)
            bs = seed_where(bw.astype(np.float32), int(P.break_flecks * many), rng)
            if len(bs):
                bx, by = bs[:, 0].astype(int), bs[:, 1].astype(int)
                an = rng.normal(0, math.radians(8), len(bs))
                bv = np.stack([dv[0] * np.cos(an) - dv[1] * np.sin(an), dv[0] * np.sin(an) + dv[1] * np.cos(an)], 1).astype(np.float32)
                bl = rng.integers(P.break_len[0], P.break_len[1] + 1, len(bs))
                blv = np.clip(np.round(2 + (NL - 2) * np.sqrt(tt[by, bx]) + rng.normal(0, 0.7, len(bs))), 2, NL - 1).astype(np.uint8)
                prev = None
                for j in range(P.break_len[1]):
                    on = j < bl
                    pt = np.round(bs + bv * j)
                    plot(img, pt[on], np.where(j == bl[on] - 1, np.maximum(blv[on].astype(int) - 1, 1), blv[on]).astype(np.uint8))   # the tail a step dimmer
                    if prev is not None:
                        st = on & (np.abs(pt - prev).sum(1) > 1.5)
                        plot(img, np.stack([pt[:, 0], prev[:, 1]], 1)[st], blv[st])
                    prev = pt
        # the motes
        u = np.mod(mph + s, 1)
        ua = u ** P.accel   # they start slow and pick up speed: crowded near the figure, thinning as they go
        pos = m0 + mv * (far * ua)[:, None] + perp * (bend * far * ua * ua + 3 * np.sin(wob + 6.283 * u) - 3 * np.sin(wob))[:, None]
        fade = np.minimum(1, u * 8) * (1 - u) ** P.falloff   # thickest as they leave, thinning out as they go
        vel = mv * far[:, None] + perp * (2 * bend * far * ua + 6 * np.pi * np.cos(wob + 6.283 * u))[:, None]
        vel /= np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6   # each dash along the way it is going
        vis = rng.random(n) < fade ** 0.7                 # they fade by thinning out, each one staying crisp
        pi_ = np.clip(np.round(pos).astype(int), 0, [W - 1, H - 1])
        vis &= ~inner[pi_[:, 1], pi_[:, 0]]               # none over the whole of the figure: it stays clean
        ml = np.where(vis, np.clip(mlev, 1, NL), 0).astype(np.uint8)
        side = np.stack([-vel[:, 1], vel[:, 0]], 1)
        mjn = rng.random(n) if P.fleck_join > 0 else np.ones(n)
        for j in range(P.mote_len[1]):
            on = vis & (j < mlen) & ~blob
            mj = np.where(j >= mlen - 1 - (mlen > 3), np.maximum(ml.astype(int) - 1, 1), ml)   # the tail a step dimmer
            mj = np.where((j == 0) & mhead & (ml > 0), np.minimum(ml.astype(int) + 1, NL), mj).astype(np.uint8)   # the head a step brighter
            pt = np.round(pos - vel * j)
            plot(img, pt[on], mj[on])
            if j > 0 and P.fleck_join > 0:   # corner-to-corner steps joined, so a slanted dash is one piece
                st = on & (np.abs(pt - mprev).sum(1) > 1.5) & (mjn < P.fleck_join)
                plot(img, np.stack([pt[:, 0], mprev[:, 1]], 1)[st], mj[st])
            mprev = pt
            on2 = on & thick
            plot(img, (pos - vel * j + side)[on2], ml[on2])   # some two dots thick
        bi = np.nonzero(blob & vis)[0]
        if len(bi):
            pts = (pos[bi, None] + bdots[bi]).reshape(-1, 2)
            lvv = np.minimum(blev[bi], ml[bi, None]).reshape(-1)
            ok = ~np.isnan(pts[:, 0])
            plot(img, pts[ok], lvv[ok])
        if len(bd):   # beads: each lit for part of the loop, a 2x2 core (a rounded 4x4 for some) with black round it
            on = np.mod(s - bph, 1) < 0.6
            for big_ in (False, True):
                m = on & (bbig == big_)
                cx_, cy_ = bd[m, 0], bd[m, 1]
                r0, r1 = (0, 2) if not big_ else (-1, 3)   # a black ring only round the big ones' rounded 4x4
                for oy in range(r0 - big_, r1 + big_):
                    for ox in range(r0 - big_, r1 + big_):
                        img[np.clip(cy_ + oy, 0, H - 1), np.clip(cx_ + ox, 0, W - 1)] = 0
                cells = [(ox, oy) for oy in range(2) for ox in range(2)]
                if big_:
                    cells = [(ox, oy) for oy in range(-1, 3) for ox in range(-1, 3) if not (ox in (-1, 2) and oy in (-1, 2))]
                for ox, oy in cells:
                    edge_ = big_ and (ox in (-1, 2) or oy in (-1, 2))
                    img[np.clip(cy_ + oy, 0, H - 1), np.clip(cx_ + ox, 0, W - 1)] = NL - 1 if edge_ else NL
        # sparkles: a few four-point glints on the brightest light, each lit for part of the loop
        for (sx, sy), sp, sl in zip(spk, sph, sln):
            ph_ = np.mod(s - sp, 1)
            if ph_ < 0.5:   # lit half the loop, swelling and shrinking
                arm = max(1, int(round(sl * math.sin(math.pi * ph_ / 0.5))))
                pts = [(sx, sy)] + [(sx + d * j, sy) for d in (-1, 1) for j in range(1, arm + 1)] + [(sx, sy + d * j) for d in (-1, 1) for j in range(1, arm + 1)]
                lv8 = [NL] + [NL - 1] * (len(pts) - 1)
                pa = np.round(np.float32(pts)).astype(int)
                ok_ = (pa[:, 0] >= 1) & (pa[:, 0] < W - 1) & (pa[:, 1] >= 1) & (pa[:, 1] < H - 1)
                for ox in (-1, 0, 1):   # a black ring round it, so it stands out of the body
                    for oy in (-1, 0, 1):
                        img[pa[ok_, 1] + oy, pa[ok_, 0] + ox] = 0
                plot(img, np.float32(pts), np.uint8(lv8))
        if not off_near.all():   # off the figure, no lone crumbs: only dashes and blobs
            n_, lab, st, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8), connectivity=8)
            crumb = (st[:, 4] < 3); crumb[0] = False
            img[crumb[lab] & ~off_near] = 0
        for (gx_, gy_), gp, ga, gl in zip(gpos, gph, garm, glev):   # after the crumbs go: they are tiny
            ph_ = np.mod(s - gp, 1)
            if ph_ < 0.35:
                # a soft puff: a bright dot in a loose ring of dimmer greys, a 3x3 at its height
                big = 0.07 <= ph_ <= 0.28
                pts = [(gx_, gy_)] + [(gx_ + ox, gy_ + oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1) if (ox or oy) and (big or ox == 0 or oy == 0) and rng.random() < 0.75]
                plot(img, np.float32(pts), np.uint8([int(gl)] + [int(rng.integers(2, int(gl))) for _ in pts[1:]]))
        if P.unplus > 0:   # a speck with four side neighbours and no corners is a '+': fill a corner, so grains are blocks
            on_ = img > 0
            pd = np.pad(on_, 1)
            arms = pd[1:-1, 2:] & pd[1:-1, :-2] & pd[2:, 1:-1] & pd[:-2, 1:-1]
            corn = pd[2:, 2:] | pd[2:, :-2] | pd[:-2, 2:] | pd[:-2, :-2]
            py, px = np.nonzero(on_ & arms & ~corn & (rng.random((H, W)) < P.unplus))
            if len(py):
                cy = np.clip(py + rng.choice([-1, 1], len(py)), 0, H - 1); cx = np.clip(px + rng.choice([-1, 1], len(py)), 0, W - 1)
                img[cy, cx] = img[py, px]
        if P.haze > 0 and not P.flat and P.glow <= 0:   # a faint dark-grey haze hugging the lit dots, a dot or two out: the specks glow
            hz = cv2.GaussianBlur((img > 0).astype(np.float32), (0, 0), 1.2)
            hr = rng.random((H, W))
            on = (img == 0) & (hr < P.haze * np.clip(hz * 2, 0, 1))
            img[on] = np.where(hr[on] < P.haze * np.clip(hz[on] * 2, 0, 1) * 0.4, HAZE[1], HAZE[0])
        if FINE:   # each speck its own grey near its step's: a fine spread of greys, not a few bands
            sp = (img >= 1) & (img <= NL)
            g = lv[img[sp]] + gshake[sp]
            g = np.clip(np.where(g < FINE_LO, 2 * FINE_LO - g, g) if P.fine_reflect else g, FINE_LO, 1.0)   # those shaken under the ramp bounce back: no pile at its foot
            img[sp] = (NL + 3 + np.round((g - FINE_LO) / (1 - FINE_LO) * (FINE - 1))).astype(np.uint8)
            if len(tl):   # each tile one grey: its first dot's
                ty0, tx0, sz = tl
                for z in (2, 3):
                    m = sz == z
                    v0 = img[ty0[m], tx0[m]]
                    for oy in range(z):
                        for ox in range(z):
                            yy_, xx_ = np.clip(ty0[m] + oy, 0, H - 1), np.clip(tx0[m] + ox, 0, W - 1)
                            keep_ = img[yy_, xx_] > 0
                            img[yy_[keep_], xx_[keep_]] = v0[keep_]
        if P.glow > 0 and not P.flat:   # the specks glow a little into the black round them, brighter round brighter ones
            v = pal[img].max(-1).astype(np.float32) / cmax
            b = cv2.GaussianBlur(v, (0, 0), P.glow_size) * P.glow
            on = (img == 0) & (b >= 0.04) & (rng.random((H, W)) < np.clip(b * 6, 0, 1))
            img[on] = (GL0 + np.round((np.clip(b[on], 0.04, 0.2) - 0.04) / 0.16 * (GLN - 1))).astype(np.uint8)
        out[f] = img
    return out, pal


STYLES = {'glow': (glow, Glow), 'streak': (streak, Streak), 'dust': (dust, Dust)}


# ---- in and out

def prepare(img: Image.Image, size: tuple[int, int], cut: bool = False) -> np.ndarray:
    """The picture framed at size (W, H) as RGBA: its own alpha if it has one, else (with cut)
    the figure GrabCut finds, framed round it; else the whole picture, covering the frame."""
    has_alpha = img.mode in ('RGBA', 'LA', 'PA') or 'transparency' in img.info
    rgba = np.asarray(img.convert('RGBA'))
    rgb, a = rgba[..., :3], rgba[..., 3].astype(np.float32) / 255
    if not has_alpha:
        a = prep.cutout(rgb) if cut else np.ones(a.shape, np.float32)
    if has_alpha or cut:
        view = prep.auto_view(a, size)
    else:
        h, w = a.shape
        vw = min(w, h * size[0] / size[1])
        view = ((w - vw) / 2, (h - vw * size[1] / size[0]) / 2, vw)
    return prep.frame(rgb, a, view, size)


def images(fr: Frames, scale: int = 1) -> list[Image.Image]:
    """The frames as palette images, each dot scale x scale px."""
    idx, pal = fr
    flat = np.zeros(768, np.uint8); flat[:pal.size] = pal.ravel()
    out = []
    for a in idx:
        im = Image.fromarray(a.repeat(scale, 0).repeat(scale, 1) if scale > 1 else a, 'P')
        im.putpalette(flat.tolist())
        out.append(im)
    return out


def encode(fr: Frames, fmt: str, fps: int = 24, scale: int = 2) -> bytes:
    """The clip as a file: 'gif' or 'webp' (looping; WebP lossless and a lot smaller), or 'png' (the first frame)."""
    ims = images(fr, scale)
    buf = io.BytesIO()
    if fmt == 'png':
        ims[0].convert('RGB').save(buf, 'PNG')
    elif fmt == 'gif':
        ims[0].save(buf, 'GIF', save_all=True, append_images=ims[1:], loop=0, duration=round(1000 / fps),
                    disposal=1, optimize=False)
    elif fmt == 'webp':  # lossless: every dot exact, and far smaller than the GIF
        ims[0].save(buf, 'WEBP', save_all=True, append_images=ims[1:], loop=0, duration=round(1000 / fps),
                    lossless=True, method=1)
    else:
        raise ValueError(f'format: gif, webp or png, not {fmt}')
    return buf.getvalue()


def save(fr: Frames, folder: Path, name: str, fps: int = 24, scale: int = 2,
         formats=('gif', 'webp', 'png')) -> list[Path]:
    """Writes folder/name.png (the first frame) and, per format, name.gif, name.webp, name/0000.png..."""
    folder.mkdir(parents=True, exist_ok=True)
    done = [folder / f'{name}.png']
    done[0].write_bytes(encode(fr, 'png', fps, scale))
    for fmt in ('gif', 'webp'):
        if fmt in formats:
            (folder / f'{name}.{fmt}').write_bytes(encode(fr, fmt, fps, scale))
            done.append(folder / f'{name}.{fmt}')
    if 'png' in formats:
        d = folder / name
        d.mkdir(exist_ok=True)
        for i, im in enumerate(images(fr, scale)):
            im.save(d / f'{i:04d}.png')
        done.append(d)
    return done


COLOURS = {'red': '#ee1111', 'white': '#f2f2f2', 'grey': '#d8d8d8', 'green': '#4fe08a', 'blue': '#6f93e8', 'amber': '#ffb347',
           'rose': '#f07aa6'}


TUNE = {'exposure': (0.3, 3.0), 'highlights': (0.0, 3.0), 'warmth': (0.0, 3.0)}   # the hand controls, and their ranges


def tune(P, t: dict | None):
    """Hand controls on top of a style's settings, each 1 (or absent) for no change: exposure (more or
    less light), highlights (how big and dense the brightest light gets), warmth (glow: how much of
    the light turns warm), focus ((x, y), 0..1 of the frame: glow's light source is put there)."""
    t = t or {}
    e, h, w = (float(t.get(k, 1.0)) for k in ('exposure', 'highlights', 'warmth'))
    P.gain *= e ** 0.6 if isinstance(P, Glow) else e   # glow's light is cut by a black point: a gentler hand
    if isinstance(P, Glow):
        P.white = P.black + (P.white - P.black) / max(h, 0.25) ** 0.5   # more of the light reaches the top of the ramp
        P.bloom *= h
        P.source *= w
        P.warm *= w
        if t.get('focus') is not None:
            P.source_xy = (float(t['focus'][0]), float(t['focus'][1]))
    elif isinstance(P, Dust):
        P.core *= h
    elif isinstance(P, Streak):
        P.most = min(1.0, P.most * (0.5 + 0.5 * h))
    return P


def params(style: str, frames: int | None = None, colour: str | None = None, angle: float | None = None,
           flat: bool = False):
    """A style's settings, with the given ones changed."""
    P = STYLES[style][1]()
    if flat and isinstance(P, Dust):
        P.flat = True
        P.grain_wide = 2.0   # flat dust art's grains run level
        P.clump_size, P.order, P.comb = 0.5, 0.3, 0.6   # finer grain, flecks combed with the drift
        P.fleck_turn, P.specks_glint, P.unplus, P.outline, P.sparkles = 0.0, 40, 0.0, 0.6, 9
        P.fleck_join, P.fleck_lo, P.mote_tail = 0.0, 0.0, 0.3
    if frames:
        P.frames = frames
    if colour and isinstance(P, (Streak, Dust)):
        P.colour = colour if colour.startswith('#') else COLOURS.get(colour, P.colour)
    if angle is not None and isinstance(P, Dust):
        P.angle = angle
        P.spare_lit = False   # a direction asked for is kept: no turning round to spare the lit side
    return P


def render(rgba: np.ndarray, style: str, frames: int | None = None, colour: str | None = None,
           angle: float | None = None, seed: int | None = None, flat: bool = False, hand: dict | None = None) -> Frames:
    """A framed picture (RGBA at the output size) as a clip in a style. Without a seed, one from
    the picture: the same picture comes out the same, and no two pictures share their rain."""
    if seed is None:
        seed = zlib.crc32(np.ascontiguousarray(rgba[::7, ::7]).tobytes())
    return STYLES[style][0](analyse(rgba), tune(params(style, frames, colour, angle, flat), hand), seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('picture', type=Path)
    ap.add_argument('--style', choices=list(STYLES) + ['all'], default='all')
    ap.add_argument('--long', type=int, help='dots across the longer side (glow 540, streak 270, dust 540)')
    ap.add_argument('--aspect', default='1x1', help='W x H of the frame (default 1x1, square)')
    ap.add_argument('--scale', type=int, help='each dot as NxN px in the files (glow 2, streak 4, dust 2)')
    ap.add_argument('--frames', type=int)
    ap.add_argument('--fps', type=int)
    ap.add_argument('--cutout', action='store_true', help='cut the figure out (GrabCut) first')
    ap.add_argument('--colour', help='streak, dust: the colour (hex, or ' + ', '.join(COLOURS) + ')')
    ap.add_argument('--angle', type=float, help='dust: which way the dust goes (degrees, 0 right, 90 down)')
    ap.add_argument('--flat', action='store_true', help='dust: one flat grey, no shades or haze')
    ap.add_argument('--seed', type=int, help='default: one from the picture')
    ap.add_argument('--out', type=Path, help='default out/<picture>/fx')
    ap.add_argument('--formats', default='gif,webp,png')
    a = ap.parse_args()
    aw, ah = (float(v) for v in a.aspect.lower().split('x'))
    out = a.out or Path('out') / a.picture.stem / 'fx'
    img = Image.open(a.picture)
    for name in (STYLES if a.style == 'all' else [a.style]):
        P = params(name, a.frames, a.colour, a.angle, a.flat)
        long = a.long or P.long
        k = long / max(aw, ah)
        rgba = prepare(img, (max(8, round(aw * k)), max(8, round(ah * k))), a.cutout)
        fr = STYLES[name][0](analyse(rgba), P, a.seed if a.seed is not None else zlib.crc32(np.ascontiguousarray(rgba[::7, ::7]).tobytes()))
        for p in save(fr, out, name, a.fps or P.fps, a.scale or P.scale, tuple(a.formats.split(','))):
            print(p)


if __name__ == '__main__':
    main()
