#!/usr/bin/env python3
"""Render a picture as a looping clip of dots on black, in one of two styles.

    phosphor_fx.py PICTURE [--style glow|streak|dust|all] [--long N] [--scale N] [--frames N] [--fps N]
                   [--cutout] [--colour C] [--angle DEG] [--seed N] [--out DIR] [--formats gif,webp,png]

  glow    six colours on black, ordered-dithered: a sparse field of blue dots in the shade, green,
          orange and white where the light gathers, a sharpened halo along every edge with magenta
          on its dark side, the outline a dense warm rim. The shape holds still while light runs
          through it in streaks, and a few dotted trails leave it, all along one line (--angle,
          degrees: 0 across, 90 up and down; default 0).
  streak  one colour (--colour, default red) in three shades, on a coarse grid: the figure
          error-diffused afresh every frame so it boils, short dashes along its shapes (and
          over the texture of its shaded parts, which the dither leaves black), a few
          stray specks, and rain streaking across the frame.
  dust    streak's engine, sparse and grey and without rain: the figure as specks and dashes
          along its form, its dim parts dissolving.

These follow the two looks of the references: glow measured at ~9% of dots lit, two thirds of
them blue, ~1.4% of the frame changing from one frame to the next; streak a coarse, boiling
stipple on clean black, ~14% changing, in a half- to one-second loop.

Each is drawn at --long dots across the longer side (glow 540, streak 270), and saved at --scale
times that (glow 2, streak 4: each dot as NxN px). The clip loops without a seam: everything that
moves comes back to where it started after --frames frames (glow 72, streak 24; both at 24 fps).

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

def analyse(rgba: np.ndarray) -> dict:
    """What both styles draw from, for a picture already at the output size (RGBA uint8):
    tone   0..1, the light (locally equalised a little, 0 outside the figure)
    plain  0..1, the light as it is (only stretched), 0 outside the figure
    alpha  0..1, the figure
    warm   -1..1, how warm (red over blue) the picture is there
    tx, ty the direction along the shapes at each px (unit; an axis, so its sign means nothing)
    coh    0..1, how strongly the shapes there run one way
    edge   0..1, how sharp the light changes there"""
    rgb = rgba[..., :3].astype(np.float32) / 255
    a = rgba[..., 3].astype(np.float32) / 255
    lum = rgb @ np.float32([0.299, 0.587, 0.114])
    l8 = (lum * 255).astype(np.uint8)
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
    return dict(tone=tone, plain=plain, alpha=a, warm=warm, tx=np.cos(ang).astype(np.float32),
                ty=np.sin(ang).astype(np.float32), coh=(lam / (jxx + jyy)).astype(np.float32),
                edge=edge.astype(np.float32))


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

# measured off the references: two blues carry two thirds of the lit dots
GLOW_PALETTE = ['#000000', '#3e72b4', '#2c75dc', '#59c077', '#e0904c', '#ae56a8', '#faf5f3']
BLUE, BLUE2, GREEN, ORANGE, MAGENTA, WHITE = range(1, 7)


@dataclass
class Glow:
    frames: int = 72
    fps: int = 24
    long: int = 540             # dots across the longer side
    scale: int = 2
    density: float = 0.2        # dots lit in the figure, on average (0: no auto exposure)
    gain: float = 1.0           # more or less light than that
    gamma: float = 3.2          # tone curve (higher: the shade empties out more)
    most: float = 0.8           # the densest the dots get, so bright areas stay a pattern
    sharpen: float = 1.8        # unsharp amount: the halo along edges
    radius: float = 1.6         # and its radius (dots)
    order: float = 0.55         # ordered (Bayer) against random dither: 1 all ordered
    hue: float = 0.045          # how much neighbouring dots' colours scatter
    zones: float = 3.0          # how broad the light is that sets the colour (dots)
    lines: float = 0.8          # how much of the contours is drawn as a line
    rim: float = 0.8            # how dense the rim band is
    rim_width: int = 3          # and how wide (dots)
    rim_level: float = 0.985    # which edges get one (percentile of edge strength; the outline always)
    line_level: float = 0.965   # which edges count as contours (percentile of edge strength)
    mix: tuple[float, float, float] = (0.73, 0.88, 0.975)  # lit dots' shares up to blue, green, orange
    fringe: float = 0.35        # magenta on the dark side of bright edges
    sprinkle: float = 0.06      # stray magenta among the mid tones
    warmth: float = 0.15        # how far the picture's own warm light leans to orange
    shimmer: float = 0.02       # share of dots that flicker over the loop
    flow: float = 0.6           # light running through the figure in streaks along the angle
    flow_speed: float = 3.0     # px per frame
    flow_len: float = 10.0      # how long a streak of it is (px, soft)
    streams: int = 700          # dotted lines streaming out of the picture
    stream_len: tuple[int, int] = (3, 32)  # dots in one
    gap: int = 2                # px from one dot of a stream to the next
    reach: float = 110          # how far a stream travels in its cycle (px)
    angle: float = 0            # the way they go (degrees: 0 right, 90 down)
    both: bool = True           # and the opposite way too
    palette: list = field(default_factory=lambda: list(GLOW_PALETTE))


def glow(A: dict, P: Glow | None = None, seed: int = 1) -> Frames:
    P = P or Glow()
    rng = np.random.default_rng(seed)
    t, a = A['plain'], A['alpha']
    H, W = t.shape
    pal = np.array([hex_rgb(c) for c in P.palette], np.uint8)

    # the light, sharpened: the overshoot either side of an edge is the halo
    soft = cv2.GaussianBlur(t, (0, 0), P.radius)
    hp = t - soft
    ts = np.clip(t + P.sharpen * hp, 0, 1)
    # exposure: the gain that puts the figure at `density` dots, so dark and bright pictures alike
    # come out as sparse as the references (then `gain` on top)
    body = a > 0.5
    k = 1.0
    if P.density > 0 and body.any():
        tb = ts[body] ** P.gamma
        lo, hi = 0.2, 8.0
        for _ in range(24):
            k = math.sqrt(lo * hi)
            if np.clip(k * tb, 0, P.most).mean() < P.density:
                lo = k
            else:
                hi = k
    p = np.clip(P.gain * k * ts ** P.gamma, 0, P.most)
    dark_side = np.clip(-hp * 7, 0, 1) * np.clip(soft * 3, 0, 1)   # just off something bright
    p = np.maximum(p, P.fringe * 0.7 * dark_side)
    thr = P.order * bayer(H, W) + (1 - P.order) * rng.random((H, W), np.float32)

    # colour per dot: by its light, scattered a little, warm light leaning orange, the bright side
    # of an edge pushed up the scale, its dark side magenta
    # The steps between colours are set by rank among the lit dots, not by fixed levels: lit dots
    # are the bright ones, so fixed levels would make most of them white. By rank, every picture
    # gets the references' mix (blue the most by far, white the least).
    zone = cv2.GaussianBlur(ts, (0, 0), P.zones)   # colour follows the broad light: it comes in zones
    v = zone + rng.normal(0, P.hue, (H, W)) + 0.25 * np.clip(hp * 5, 0, 1) + P.warmth * A['warm'] * zone
    lit0 = thr < p
    qb, qg, qo = np.quantile(v[lit0], P.mix) if lit0.any() else (0.5, 0.7, 0.9)
    col = np.select([v < qb, v < qg, v < qo], [np.where(rng.random((H, W)) < 0.45, BLUE2, BLUE), GREEN, ORANGE],
                    WHITE).astype(np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    col[(col == WHITE) & ((xx + yy) % 2 == 1)] = ORANGE   # the brightest is a checker of white and orange
    col[(rng.random((H, W)) < P.fringe * dark_side) | ((rng.random((H, W)) < P.sprinkle) & (v > 0.35) & (v < 0.8))] = MAGENTA

    # contours: thin continuous lines, their dots in the warm colours (and a little white, a little blue)
    l8 = (cv2.GaussianBlur(t, (0, 0), 2.0) * 255).astype(np.uint8)
    gm = np.hypot(cv2.Sobel(l8, cv2.CV_32F, 1, 0), cv2.Sobel(l8, cv2.CV_32F, 0, 1))
    hi = float(np.percentile(gm[a > 0.5], P.line_level * 100)) if (a > 0.5).any() else 100.0
    contour = (cv2.Canny(l8, max(4, hi * 0.35), max(8, hi)) > 0) & (rng.random((H, W)) < P.lines)
    p = np.where(contour, 1.0, p)
    col[contour] = rng.choice(np.array([ORANGE, MAGENTA, WHITE, BLUE2], np.uint8), int(contour.sum()), p=[0.25, 0.3, 0.15, 0.3])
    # the rim: where the light meets the dark hardest (and round the figure), a band a few dots
    # wide, dense: pink on its outer, dark side, orange and white on its lit side
    strong = gm > np.percentile(gm[a > 0.5], P.rim_level * 100) if (a > 0.5).any() else np.zeros_like(contour)
    if not (a > 0.5).all():
        m8 = (a > 0.5).astype(np.uint8)
        strong |= (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0
        strong[[0, -1], :] = strong[:, [0, -1]] = False
    band = cv2.dilate(strong.astype(np.uint8), np.ones((P.rim_width, P.rim_width), np.uint8)) > 0
    outer = band & (hp < 0)
    inner = band & ~outer
    p = np.where(band, np.maximum(p, P.rim), p)
    col[outer] = rng.choice(np.array([MAGENTA, MAGENTA, BLUE, ORANGE], np.uint8), int(outer.sum()))
    col[inner] = rng.choice(np.array([ORANGE, WHITE, MAGENTA, GREEN, BLUE2], np.uint8), int(inner.sum()))
    lit0 = thr < p

    th0 = math.radians(P.angle)
    flick = np.where(rng.random((H, W)) < P.shimmer, rng.choice([-1, 1], (H, W)), 0).astype(np.float32)

    # streams: from lit dots, straight out along the angle, as dotted lines thinning as they go
    base0 = lit0
    srcs = seed_where((base0 * (0.2 + ts) * (a > 0.3)).astype(np.float32), P.streams, rng)
    n = len(srcs)
    ix, iy = srcs[:, 0].astype(int), srcs[:, 1].astype(int)
    scol = np.where(base0[iy, ix], col[iy, ix], BLUE).astype(np.uint8)
    th = math.radians(P.angle)
    sign = np.where(rng.random(n) < 0.5, -1, 1) if P.both else np.ones(n)
    dv = np.stack([np.cos(th) * sign, np.sin(th) * sign], 1).astype(np.float32)
    ph = rng.random(n).astype(np.float32)
    ln = rng.integers(P.stream_len[0], P.stream_len[1] + 1, n)
    reach = (P.reach * rng.uniform(0.45, 1, n)).astype(np.float32)

    # the flow: a streaky field, periodic along the angle so it scrolls round in one loop
    Lu, Lv = max(8, int(round(P.flow_speed * P.frames))), 256
    fld = cv2.GaussianBlur(np.tile(rng.random((Lv, Lu)).astype(np.float32), (3, 3)), (0, 0), sigmaX=P.flow_len,
                           sigmaY=1.3)[Lv:2 * Lv, Lu:2 * Lu]   # tiled: it wraps round seamlessly
    fld = (np.argsort(np.argsort(fld, axis=None)).reshape(fld.shape) / fld.size).astype(np.float32)  # flat 0..1
    fld = np.clip((fld - 0.5) * 2.2 + 0.5, 0, 1)
    fu = (xx * math.cos(th0) + yy * math.sin(th0)).astype(np.float32)
    fv = (np.round(-xx * math.sin(th0) + yy * math.cos(th0)).astype(int)) % Lv
    held = band | contour                                   # lines and rims hold still
    # it runs through texture (feathers, hair, folds), and only faintly over flat tone, where it
    # would read as scanlines
    tex = cv2.GaussianBlur(np.abs(hp), (0, 0), 4)
    tex = np.clip(tex / (np.percentile(tex[a > 0.5], 80) + 1e-6 if (a > 0.5).any() else 1), 0, 1)
    amp = (P.flow * 2 * (0.25 + 0.75 * tex)).astype(np.float32)

    out = np.zeros((P.frames, H, W), np.uint8)
    for f in range(P.frames):
        s = f / P.frames
        if P.flow > 0:
            m = 1 + amp * (fld[fv, np.round(fu - P.flow_speed * f).astype(int) % Lu] - 0.5)
            pf = np.where(held, p, p * m)
        else:
            pf = p
        lit = np.mod(thr + flick * s, 1) < pf
        img = np.where(lit, col, 0).astype(np.uint8)
        u = np.mod(ph + s, 1)
        trav = u * reach                                   # how far the head has gone
        keep = np.ceil(ln * (1 - u ** 1.5)).astype(int)    # it thins out as it goes
        for j in range(P.stream_len[1]):
            d = np.round(trav / P.gap) * P.gap - j * P.gap  # on a 'gap' grid: evenly dotted
            on = (j < keep) & (d > 1)
            plot(img, (srcs + dv * d[:, None])[on], scol[on])
        out[f] = img
    return out, pal


# ---- streak

@dataclass
class Streak:
    frames: int = 24
    fps: int = 24
    long: int = 270
    scale: int = 4
    colour: str = '#ee1111'
    density: float = 0.2        # dots lit in the figure, on average (0: no auto exposure)
    gain: float = 1.0           # more or less light than that
    gamma: float = 1.25         # tone curve
    black: float = 0.18         # light below this is black (no haze of stray dots in the shade)
    floor: float = 0.0          # and after the tone curve, light below this (no even sprinkle in the dim)
    boil: float = 0.05          # noise added before each frame's dither: how much it boils
    stray: float = 0.25         # how much of that noise is let into the background
    dashes: int = 140           # short straight dashes along the shapes, each frame
    dash_len: tuple[int, int] = (3, 7)
    field: int = 500            # dashes drawing the texture the dither leaves black (waves, hair, cloth in shade)
    field_len: tuple[int, int] = (2, 5)
    rain: int = 130             # streaks across the frame
    rain_len: tuple[int, int] = (8, 26)
    rain_angle: float = -24     # degrees from the right, upwards negative (screen y is down)
    rain_gaps: float = 0.2      # gaps in a streak (0: solid)


def streak(A: dict, P: Streak | None = None, seed: int = 1) -> Frames:
    P = P or Streak()
    rng = np.random.default_rng(seed)
    t, a = A['tone'], A['alpha']
    H, W = t.shape
    c = np.float32(hex_rgb(P.colour))
    pal = np.array([[0, 0, 0], c * 0.62, c * 0.8, c], np.uint8)       # three shades of the one colour
    tt = np.clip(P.gain * np.clip((t - P.black) / (1 - P.black), 0, 1) ** P.gamma, 0, 1)
    tt = np.where(tt < P.floor, 0, tt).astype(np.float32)
    shade = (1 + (tt > 0.42) + (tt > 0.72)).astype(np.uint8)
    # the boil only where there's light (and a `stray` share of it anywhere): black stays black,
    # cut out or not
    room = (P.stray + (1 - P.stray) * a * np.clip(tt * 10, 0, 1)).astype(np.float32)

    w_dash = (A['edge'] ** 2 * A['coh'] * (tt > 0.15)).astype(np.float32)
    w_field = (A['edge'] * A['coh'] * (tt <= 0.15) * (A['plain'] > 0.02) * (a > 0.3)).astype(np.float32)
    d = math.radians(P.rain_angle)
    dv = np.float32([math.cos(d), math.sin(d)])
    span = math.hypot(W, H) + 2 * P.rain_len[1]
    perp = np.float32([-dv[1], dv[0]])
    mid = np.float32([W / 2, H / 2])
    across = rng.uniform(-span / 2, span / 2, P.rain).astype(np.float32)
    rph = rng.random(P.rain).astype(np.float32)
    rk = rng.choice([1, 2], P.rain).astype(np.float32)
    rl = rng.integers(P.rain_len[0], P.rain_len[1] + 1, P.rain)

    out = np.zeros((P.frames, H, W), np.uint8)
    for f in range(P.frames):
        s = f / P.frames
        noisy = np.clip(tt + rng.normal(0, P.boil, (H, W)).astype(np.float32) * room, 0, 1)
        bits = np.asarray(Image.fromarray((noisy * 255).astype(np.uint8), 'L').convert('1'), bool)  # Floyd-Steinberg
        img = np.where(bits, shade, 0).astype(np.uint8)
        # dashes: placed afresh each frame along the shapes; bright on the lit figure, and in the
        # mid shade where they stand for texture the dither left black
        for w, n, (l0, l1), sh in ((w_field, P.field, P.field_len, 2), (w_dash, P.dashes, P.dash_len, 3)):
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
        along = (np.mod(rph - rk * s, 1) - 0.5) * span
        for j in range(P.rain_len[1]):
            on = (j < rl) & (rng.random(P.rain) > P.rain_gaps)
            plot(img, (mid + perp * across[:, None] + dv * (along - j)[:, None])[on], 3)
        out[f] = img
    return out, pal


@dataclass
class Dust(Streak):
    """streak's engine, sparse and grey, without rain: the figure as specks and dashes along its
    form, its dim parts dissolving."""
    long: int = 360
    scale: int = 3
    colour: str = '#b8b8b8'
    gain: float = 0.25
    gamma: float = 2.2
    black: float = 0.2
    floor: float = 0.06
    stray: float = 0.0
    dashes: int = 220
    field: int = 650
    rain: int = 0


STYLES = {'glow': (glow, Glow), 'streak': (streak, Streak), 'dust': (streak, Dust)}


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


COLOURS = {'red': '#ee1111', 'white': '#f2f2f2', 'grey': '#b8b8b8', 'green': '#4fe08a', 'blue': '#6f93e8', 'amber': '#ffb347',
           'rose': '#f07aa6'}


def params(style: str, frames: int | None = None, colour: str | None = None, angle: float | None = None):
    """A style's settings, with the given ones changed."""
    P = STYLES[style][1]()
    if frames:
        P.frames = frames
    if colour and isinstance(P, Streak):
        P.colour = colour if colour.startswith('#') else COLOURS.get(colour, P.colour)
    if angle is not None and isinstance(P, Glow):
        P.angle = angle
    return P


def render(rgba: np.ndarray, style: str, frames: int | None = None, colour: str | None = None,
           angle: float | None = None, seed: int = 1) -> Frames:
    """A framed picture (RGBA at the output size) as a clip in a style."""
    return STYLES[style][0](analyse(rgba), params(style, frames, colour, angle), seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('picture', type=Path)
    ap.add_argument('--style', choices=list(STYLES) + ['all'], default='all')
    ap.add_argument('--long', type=int, help='dots across the longer side (glow 540, streak 270)')
    ap.add_argument('--aspect', default='1x1', help='W x H of the frame (default 1x1, square)')
    ap.add_argument('--scale', type=int, help='each dot as NxN px in the files (glow 2, streak 4)')
    ap.add_argument('--frames', type=int)
    ap.add_argument('--fps', type=int)
    ap.add_argument('--cutout', action='store_true', help='cut the figure out (GrabCut) first')
    ap.add_argument('--colour', help='streak, dust: the colour (hex, or ' + ', '.join(COLOURS) + ')')
    ap.add_argument('--angle', type=float, help='glow: which way the streams go (degrees, 0 right, 90 down)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', type=Path, help='default out/<picture>/fx')
    ap.add_argument('--formats', default='gif,webp,png')
    a = ap.parse_args()
    aw, ah = (float(v) for v in a.aspect.lower().split('x'))
    out = a.out or Path('out') / a.picture.stem / 'fx'
    img = Image.open(a.picture)
    for name in (STYLES if a.style == 'all' else [a.style]):
        P = params(name, a.frames, a.colour, a.angle)
        long = a.long or P.long
        k = long / max(aw, ah)
        rgba = prepare(img, (max(8, round(aw * k)), max(8, round(ah * k))), a.cutout)
        fr = STYLES[name][0](analyse(rgba), P, a.seed)
        for p in save(fr, out, name, a.fps or P.fps, a.scale or P.scale, tuple(a.formats.split(','))):
            print(p)


if __name__ == '__main__':
    main()
