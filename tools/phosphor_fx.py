#!/usr/bin/env python3
"""Render a picture as a looping clip of dots on black, in one of three styles.

    phosphor_fx.py PICTURE [--style glow|streak|dust|all] [--long N] [--scale N] [--frames N] [--fps N]
                   [--cutout] [--colour C] [--angle DEG] [--seed N] [--out DIR] [--formats gif,webp,png]

  glow    seven colours on black, ordered-dithered: a sparse field of blue dots in the shade,
          green, orange and white in zones where the light gathers, contours in salmon, orange,
          blue and pink, pink on the dark side of edges, and a rim of thin parallel lines round
          the figure. The shape holds still while light and sparks run through it in slanted
          streaks, and bundles of lines grow out of its outline and draw back, along one line
          (--angle, degrees: 0 across, 90 up and down).
  streak  one colour (--colour, default red) on a coarse grid: the figure error-diffused afresh
          every frame so it boils, short dashes along its shapes (and over the texture of its
          shaded parts, which the dither leaves black), loose specks gathering towards the bottom,
          and slanted rain flashing up in new places every frame.
  dust    grey specks on an ordered dither, brighter in the light, the outline and strongest edges
          as bright strokes, short flecks along the shapes and a few glints; from its darker side
          along --angle the figure comes apart in lanes of dashes that drift off and fade.

These follow the looks of the references: glow measured at ~9% of dots lit, most of them blue,
~1.4% of the frame changing from one frame to the next; streak a coarse, boiling stipple on clean
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

# measured off the references: two blues carry two thirds of the lit dots
GLOW_PALETTE = ['#000000', '#3e72b4', '#2c75dc', '#59c077', '#e0904c', '#ae56a8', '#faf5f3', '#e88674']
BLUE, BLUE2, GREEN, ORANGE, MAGENTA, WHITE, SALMON = range(1, 8)


@dataclass
class Glow:
    frames: int = 72
    fps: int = 24
    long: int = 540             # dots across the longer side
    scale: int = 2
    density: float = 0.16       # dots lit in the figure, on average (0: no auto exposure)
    gain: float = 1.0           # more or less light than that
    solid: float = 0.2          # share of the figure lit near solid (picks the curve per picture; 0: use gamma)
    gamma: float = 3.0          # tone curve (higher: the shade empties out more)
    fill: float = 0.65          # the densest the dots get outside the brightest light
    halo: float = 0.2           # sparse dots over the mid-tones, so the shade fades out
    hot_fill: float = 0.9       # how solid the hot core is
    feat_fill: float = 0.5      # how densely the features (eyes, mouth, folds) are drawn where lit round them
    hot: float = 0.05           # share of the figure, its brightest broad light, coloured in calm bands
    core: float = 0.05          # share of the figure, its brightest, that may go near solid
    most: float = 0.95          # the densest the dots get: the brightest light near solid
    even: float = 0.25          # how much of the locally evened light (CLAHE) the tone takes
    sharpen: float = 1.0        # unsharp amount: the halo along edges
    radius: float = 1.6         # and its radius (dots)
    order: float = 0.55         # ordered (Bayer) against random dither: 1 all ordered
    hue: float = 0.1            # how much neighbouring dots' colours scatter
    zones: float = 10.0         # how broad the light is that sets the colour (dots)
    mix: float = 0.7            # how far neighbouring zones checker into each other, dot by dot
    edge_hue: float = 0.05      # how far fine edges push the colour up the ramp
    lines: float = 0.15         # how much of the contours is drawn as a line
    rim: float = 0.7            # how dense the rim lines are
    outline: float = 0.45       # how much of the unlit outline is drawn, sparse and cool
    rim_lines: int = 3          # parallel lines along a rim, a dot wide, a dot of black between
    rim_level: float = 0.985    # which edges get one (percentile of edge strength; the outline always)
    line_level: float = 0.99    # which edges count as contours (percentile of edge strength)
    ramp: tuple = (0.0, 0.3, 0.8, 0.95, 0.99)   # rank among lit dots where blue, blue 2, green, orange, white sit
    floor: float = 0.05         # the least light in the figure: its shade never quite goes black
    fringe: float = 0.8         # magenta on the dark side of bright edges
    warmth: float = 0.15        # how far the picture's own warm light leans to orange
    twinkle: float = 0.06       # share of dots given a fresh threshold each frame
    shimmer: float = 0.01       # share of dots that flicker over the loop
    flow: float = 0.35          # light running through the figure in streaks along the angle
    flow_speed: float = 6.0     # px per frame
    flow_len: float = 12.0      # how long a streak of it is (px, soft)
    flow_width: float = 0.8     # and how wide (px, soft): thin sweeps
    sparks: float = 0.04        # share of the scrolling field lit as sparks running through the light
    tilt: float = -20           # the flow's slant from the angle (degrees)
    streams: int = 20           # bundles of lines streaming out of the picture
    strands: tuple[int, int] = (1, 4)   # lines in a bundle, 2 dots apart
    stream_len: tuple[int, int] = (30, 120)  # dots in one, at its longest
    angle: float = 0            # the way they go (degrees: 0 right, 90 down)
    both: bool = True           # and the opposite way too
    palette: list = field(default_factory=lambda: list(GLOW_PALETTE))


def glow(A: dict, P: Glow | None = None, seed: int = 1) -> Frames:
    P = P or Glow()
    rng = np.random.default_rng(seed)
    a = A['alpha']
    t = (1 - P.even) * A['plain'] + P.even * A['tone']    # some of the evened-out light: dark pictures keep their shapes
    H, W = t.shape
    pal = np.array([hex_rgb(c) for c in P.palette], np.uint8)

    # the light, sharpened: the overshoot either side of an edge is the halo
    soft = cv2.GaussianBlur(t, (0, 0), P.radius)
    hp = t - soft
    ts = np.clip(t + P.sharpen * hp, 0, 1)
    # exposure: the gain that puts the figure at `density` dots, so dark and bright pictures alike
    # come out as sparse as the references (then `gain` on top)
    # And the curve's steepness that leaves `solid` of the figure near full light: busy pictures
    # still build solid masses of light, flat ones don't blow out
    body = a > 0.5
    k, gam = 1.0, P.gamma

    tsb = ts[body][::7]                  # a sample is plenty to set the exposure

    def expose(g):
        tb = tsb ** g
        lo, hi = 0.05, 50.0
        for _ in range(30):
            kk = math.sqrt(lo * hi)
            if np.clip(kk * tb, 0, P.most).mean() < P.density:
                lo = kk
            else:
                hi = kk
        return kk, (np.clip(kk * tb, 0, P.most) >= 0.7).mean()
    if P.density > 0 and body.any():
        k, sol = expose(gam)
        if P.solid > 0:
            best = None
            for g in np.linspace(1.5, 7.0, 12):
                kk, sl = expose(g)
                if best is None or abs(sl - P.solid) < abs(best[2] - P.solid):
                    best = (g, kk, sl)
            gam, k, _ = best
    p = np.clip(P.gain * k * ts ** gam, 0, P.most)
    if P.fill < P.most and body.any():   # solid only in the brightest light; elsewhere the dither keeps gaps
        top = ts > np.quantile(ts[body], 1 - P.core)
        p = np.where(top, p, np.minimum(p, P.fill))
    dark_side = np.clip(-hp * 7, 0, 1) * np.clip(soft * 3, 0, 1)   # just off something bright
    g0 = cv2.GaussianBlur(t, (0, 0), 2.0)
    g0 = np.hypot(cv2.Sobel(g0, cv2.CV_32F, 1, 0), cv2.Sobel(g0, cv2.CV_32F, 0, 1))
    big = g0 > (np.percentile(g0[a > 0.5], 95) if (a > 0.5).any() else g0.max())
    dark_side *= cv2.dilate(big.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # only by the strong edges, not over fine texture
    p = np.maximum(p, P.fringe * 0.7 * dark_side)
    shade_min = 0.04
    if body.mean() > 0.97:   # nothing cut out: the dim background isn't the figure's shade
        shade_min = max(shade_min, float(np.percentile(soft, 50)))
    p = np.maximum(p, P.floor * (a > 0.5) * (soft > shade_min))   # the dark of the figure keeps a sparse blue dither
    p = np.maximum(p, P.halo * np.clip((soft - 0.12) / 0.25, 0, 1) * (a > 0.5) * (soft > shade_min))   # the mid-tones fade out, not cut to black
    thr = P.order * bayer(H, W) + (1 - P.order) * rng.random((H, W), np.float32)

    # colour per dot: by the broad light, in zones along one ramp (blue, blue 2, green, orange,
    # white), each zone one colour or an ordered checker of its two neighbours on the ramp, never
    # colours far apart side by side. Where the ramp's steps fall is set by rank among the lit
    # dots, so every picture gets the references' mix (blue the most by far, white the least).
    zone = cv2.GaussianBlur(ts, (0, 0), P.zones)
    hot = zone > (np.quantile(zone[body], 1 - P.hot) if body.any() else 2)   # the hot core: calm, banded colour
    hot = cv2.GaussianBlur(hot.astype(np.float32), (0, 0), 2)
    lift = 0.0
    if body.any():   # the core near solid, if the light has one: evenly lit line art has none
        q50, q95 = np.quantile(zone[body], [0.5, 0.95])
        lift = float(np.clip(((q95 - q50) / (q50 + 0.05) - 0.3) / 0.25, 0, 1))
    # features (eyes, nostrils, mouth, folds): small spots darker than round them, not striations
    k = max(3, int(round(0.015 * max(H, W))) | 1)
    th = cv2.morphologyEx(cv2.GaussianBlur(t, (0, 0), 1.0), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    fl = np.percentile(th[body], 92) if body.any() else 1
    feat = (th > max(fl, 0.03)) & body & (A['coh'] < 0.7)
    sparse = (cv2.blur((thr < p).astype(np.float32), (15, 15)) < 0.4) & (A['coh'] < 0.4)   # not over busy texture
    p = np.where(feat & sparse & ~(hot > 0.5), np.maximum(p, P.feat_fill * np.clip(soft * 4, 0, 1)), p)   # drawn, not holes
    p = np.where((hot > 0.5) & body & ~feat, np.maximum(p, P.hot_fill * lift), p)
    if P.hot_fill * lift > 0.85:   # no holes in a solid core, but its features show through
        hot_solid = (hot > 0.6) & body & ~feat
    else:
        hot_solid = np.zeros_like(body)
    v = zone - 0.25 * feat + P.edge_hue * np.clip(hp * 5, 0, 1) + P.warmth * A['warm'] * zone + rng.normal(0, P.hue, (H, W)) * (1 - 0.75 * A['flat']) * (1 - 0.85 * hot)
    lit0 = thr < p
    anchors = np.quantile(v[lit0], P.ramp) if lit0.any() else np.linspace(0, 1, 5)
    r = np.interp(v, np.maximum.accumulate(anchors + np.arange(5) * 1e-6), np.arange(5))
    ramp = np.array([BLUE, BLUE2, GREEN, ORANGE, WHITE], np.uint8)
    col = ramp[np.clip(np.floor(r + P.mix * (bayer(H, W) - 0.5)), 0, 4).astype(int)]   # zones, checkered where they meet
    col[(col == BLUE2) & (r < 1.2 + P.mix) & (rng.random((H, W)) < 0.5)] = BLUE   # the two blues mingle
    col[(rng.random((H, W)) < P.fringe * dark_side)] = MAGENTA          # purple just off bright edges

    # contours: thin continuous lines, each line one colour
    l8 = (cv2.GaussianBlur(t, (0, 0), 2.0) * 255).astype(np.uint8)
    gm = np.hypot(cv2.Sobel(l8, cv2.CV_32F, 1, 0), cv2.Sobel(l8, cv2.CV_32F, 0, 1))
    hi = float(np.percentile(gm[a > 0.5], P.line_level * 100)) if (a > 0.5).any() else 100.0
    contour = cv2.Canny(l8, max(4, hi * 0.35), max(8, hi)) > 0
    n, lab = cv2.connectedComponents(contour.astype(np.uint8), connectivity=8)
    pick = rng.choice(np.array([ORANGE, SALMON, WHITE, BLUE2, MAGENTA], np.uint8), n, p=[0.06, 0.14, 0.06, 0.38, 0.36])
    contour &= rng.random((H, W)) < P.lines
    p = np.where(contour, 1.0, p)
    col[contour] = pick[lab[contour]]
    # the rim: where the light meets the dark hardest (and round the figure), parallel lines a dot
    # wide with a dot of black between, on the lit side: white and orange at the edge, then
    # salmon, then purple, fading
    strong = gm > np.percentile(gm[a > 0.5], P.rim_level * 100) if (a > 0.5).any() else np.zeros_like(contour)
    if not (a > 0.5).all():
        m8 = (a > 0.5).astype(np.uint8)
        lit_out = soft > np.percentile(soft[a > 0.5], 80)          # only where the outline is lit
        strong |= ((m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0) & lit_out
        strong[[0, -1], :] = strong[:, [0, -1]] = False
    d = cv2.distanceTransform((~strong).astype(np.uint8), cv2.DIST_L2, 3)
    lit_side = ((t - cv2.GaussianBlur(t, (0, 0), 3)) > -0.02) & (a > 0.5)
    yy, xx = np.mgrid[0:H, 0:W]
    bright = cv2.GaussianBlur(t, (0, 0), 4)
    bright = bright > (np.percentile(bright[a > 0.5], 55) if (a > 0.5).any() else 0.5)   # rims only where it's lit
    near = lit_side & bright & (d < 2 * P.rim_lines)
    runs = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 1.5)   # colour in runs of a few dots
    runs = np.argsort(np.argsort(runs, axis=None)).reshape(H, W) / runs.size
    first = np.array([ORANGE, WHITE, SALMON, MAGENTA], np.uint8)[np.digitize(runs, [0.3, 0.5, 0.72])]
    wide = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 15)
    wide = np.clip(0.5 + (wide - wide.mean()) / (wide.std() + 1e-6) * 0.35, 0.15, 1)   # the rim's weight changes along it
    rows = [(first, 1.0), (SALMON, 0.85), (MAGENTA, 0.6), (BLUE2, 0.4)]
    for k in range(P.rim_lines):
        line = near & (d >= 2 * k) & (d < 2 * k + 1)
        gap = near & (d >= 2 * k + 1) & (d < 2 * k + 2)
        c, dens = rows[min(k, 3)]
        on = line & (rng.random((H, W)) < P.rim * dens * (wide if k else 1.6 * wide))
        p = np.where(on, 1.0, np.where(line, 0.0, np.where(gap, 0.3 * p, p)))
        col[on] = c[on] if isinstance(c, np.ndarray) else c
    if P.outline > 0 and not (a > 0.5).all():   # the unlit rest of the outline: a sparse, cool line
        dark_out = ((m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0) & ~lit_out & ~near
        dark_out[[0, -1], :] = dark_out[:, [0, -1]] = False
        on = dark_out & (rng.random((H, W)) < P.outline * wide)
        p = np.where(on, 1.0, p)
        col[on] = np.where(runs[on] < 0.6, BLUE2, np.where(runs[on] < 0.85, MAGENTA, SALMON))
    lit0 = thr < p

    th0 = math.radians(P.angle + P.tilt)   # off the dot rows, or the flow bands along them
    flick = np.where(rng.random((H, W)) < P.shimmer, rng.choice([-1, 1], (H, W)), 0).astype(np.float32)

    # streams: bundles of parallel lines growing out of the figure from bright points, along the
    # angle and away from the figure, their colour changing along them in runs; each grows out,
    # holds, and draws back in once round the loop
    inner = (a > 0.5).astype(np.uint8)
    if 0 < inner.mean() < 0.97:       # a cut-out: from just inside its outline
        din = cv2.distanceTransform(np.pad(inner, 1, constant_values=1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        edge0 = (din < 5) & (inner > 0)
    else:                             # the whole picture: from its strongest edges
        edge0 = gm > np.percentile(gm, 97)
    seedw = (np.maximum(lit0, 0.3) * (0.15 + ts) ** 4 * edge0).astype(np.float32)   # from the brightest light
    srcs = seed_where(seedw, P.streams, rng)
    nb = len(srcs)
    th = math.radians(P.angle)
    d0 = np.float32([math.cos(th), math.sin(th)])
    ab = cv2.GaussianBlur(a, (0, 0), 6)
    look = lambda q: ab[np.clip(np.round(q[:, 1]).astype(int), 0, H - 1), np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)]
    out_side = look(srcs + d0 * 12) - look(srcs - d0 * 12)
    sign = np.where(np.abs(out_side) > 0.05, np.where(out_side > 0, -1, 1), rng.choice([-1, 1], nb) if P.both else 1)
    bdv = (d0[None] * sign[:, None]).astype(np.float32)
    nl = rng.integers(P.strands[0], P.strands[1] + 1, nb)
    bi = np.repeat(np.arange(nb), nl)
    k_in = np.concatenate([np.arange(k) for k in nl]) if nb else np.zeros(0, int)
    perp = np.stack([-bdv[:, 1], bdv[:, 0]], 1)
    lsrc = srcs[bi] + perp[bi] * (2.0 * (k_in - (nl[bi] - 1) / 2))[:, None] - bdv[bi] * rng.integers(0, 6, len(bi))[:, None]
    ldv = bdv[bi]
    L = (rng.integers(P.stream_len[0], P.stream_len[1] + 1, nb)[bi] * rng.uniform(0.6, 1, len(bi))).astype(np.float32)
    ph = (rng.random(nb)[bi] + rng.uniform(0, 0.04, len(bi))).astype(np.float32)
    # each line one colour, a quarter of its dots its neighbour on the ramp; most dotted every 5th dot
    nbr = np.zeros(9, np.uint8)
    for c0, c1 in ((BLUE, BLUE2), (BLUE2, GREEN), (GREEN, BLUE2), (ORANGE, SALMON), (SALMON, ORANGE),
                   (WHITE, ORANGE), (MAGENTA, SALMON)):
        nbr[c0] = c1
    sx, sy = np.clip(srcs[bi, 0].astype(int), 0, W - 1), np.clip(srcs[bi, 1].astype(int), 0, H - 1)
    base = col[sy, sx].copy()                                     # the colour of the light it leaves
    base[(base >= ORANGE) & (rng.random(len(bi)) < 0.8)] = BLUE2  # mostly cool all the same
    runs = np.where(rng.random((len(bi), P.stream_len[1] + 2)) < 0.25, nbr[base][:, None], base[:, None]).astype(np.uint8)
    step = rng.choice([2, 3], len(bi), p=[0.8, 0.2])   # dotted, never a solid bar
    # the flow: a streaky field, periodic along the angle so it scrolls round in one loop
    Lu, Lv = max(8, int(round(P.flow_speed * P.frames))), 256
    fld = cv2.GaussianBlur(np.tile(rng.random((Lv, Lu)).astype(np.float32), (3, 3)), (0, 0), sigmaX=P.flow_len,
                           sigmaY=P.flow_width)[Lv:2 * Lv, Lu:2 * Lu]   # tiled: it wraps round seamlessly
    fld = (np.argsort(np.argsort(fld, axis=None)).reshape(fld.shape) / fld.size).astype(np.float32)  # flat 0..1
    fld = np.clip((fld - 0.5) * 2.2 + 0.5, 0, 1)
    fu = (xx * math.cos(th0) + yy * math.sin(th0)).astype(np.float32)
    fv = (np.round(-xx * math.sin(th0) + yy * math.cos(th0)).astype(int)) % Lv
    held = contour | (near & (d < 2 * P.rim_lines))          # lines and rims hold still
    # it runs through texture (feathers, hair, folds), and only faintly over flat tone, where it
    # would read as scanlines
    tex = cv2.GaussianBlur(np.abs(hp), (0, 0), 4)
    tex = np.clip(tex / (np.percentile(tex[a > 0.5], 80) + 1e-6 if (a > 0.5).any() else 1), 0, 1)
    amp = (P.flow * 2 * (0.6 + 0.4 * tex)).astype(np.float32)
    # sparks: short fully lit dashes sliding along through the lit parts, on the same scroll
    f2 = cv2.GaussianBlur(np.tile(rng.random((Lv, Lu)).astype(np.float32), (3, 3)), (0, 0), sigmaX=7, sigmaY=0.5)[Lv:2 * Lv, Lu:2 * Lu]
    spark = f2 > np.quantile(f2, 1 - P.sparks)
    where_sp = (soft > (np.percentile(soft[a > 0.5], 85) if (a > 0.5).any() else 0.5)) & (a > 0.5)   # only the brightest

    out = np.zeros((P.frames, H, W), np.uint8)
    for f in range(P.frames):
        s = f / P.frames
        if P.flow > 0:
            m = 1 + amp * (fld[fv, np.round(fu - P.flow_speed * f).astype(int) % Lu] - 0.5)
            pf = np.where(held, p, p * m)
            if P.sparks > 0:
                pf = np.maximum(pf, 0.9 * (spark[fv, np.round(fu - P.flow_speed * f).astype(int) % Lu] & where_sp))
        else:
            pf = p
        th_f = np.mod(thr + flick * s, 1)
        if P.twinkle > 0:   # a few dots each frame take a fresh threshold: the light glitters
            tw = (rng.random((H, W)) < P.twinkle * (1 - hot)) & ~held & (p > 0.1)   # only where there is light
            th_f = np.where(tw, rng.random((H, W)), th_f)
        lit = (th_f < pf) | hot_solid
        img = np.where(lit, col, 0).astype(np.uint8)
        u = np.mod(ph + s, 1)
        grow = np.clip(np.minimum(u * 3, (1 - u) * 3), 0, 1) * L      # out, hold, back in
        for j in range(P.stream_len[1]):
            on = (j < grow) & (j % step == 0)
            plot(img, (lsrc + ldv * j)[on], runs[on, j])
        if not near.all():   # off the figure, no lone crumbs: only dashes and blobs
            n_, lab, st, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8), connectivity=8)
            crumb = (st[:, 4] < 3); crumb[0] = False
            img[crumb[lab] & ~near] = 0
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
    gain: float = 1.0           # more or less light
    gamma: float = 2.1          # tone curve: above 1, darker mid-tones and more contrast
    most: float = 0.92          # the densest the dots get: near solid in the highlights
    shades: int = 1             # 1: one ink, tone by how many dots only; 3: three shades of it
    black: float = 0.25         # light below this is black (no haze of stray dots in the shade)
    floor: float = 0.0          # and after the tone curve, light below this (no even sprinkle in the dim)
    peak: float = 0.06          # share of the figure, its brightest, dithered near solid
    features: float = 1.0       # how much darker small dark spots (eyes, mouth) go before the dither
    rim: float = 0.15           # light added along the inside of the silhouette, so it holds
    smooth: float = 0.7         # blur on the light before the dither (dots): clean worms, not blotches
    even: float = 0.5           # how much of the locally evened light (CLAHE) the tone takes; the rest plain
    jitter: float = 0.6         # threshold shake in the dither: maze-like chains, not stripes and checks
    hold: float = 0.5           # share of the figure's dots held from one dither all loop (less boil)
    boil: float = 0.05          # noise added before each frame's dither: how much it boils
    stray: float = 0.0          # how much of that noise is let into the background
    specks: float = 0.012       # loose dots over the background, more towards the bottom
    dashes: int = 70            # short straight dashes along the shapes, each frame
    dash_len: tuple[int, int] = (3, 7)
    field: int = 40             # dashes drawing the texture the dither leaves black (waves, hair, cloth in shade)
    field_len: tuple[int, int] = (2, 5)
    rain: int = 320             # bundles of streaks across the frame
    strands: tuple[int, int] = (1, 1)   # streaks in a bundle, side by side
    rain_len: tuple[int, int] = (20, 64)
    rain_angle: float = -24     # degrees from the right, upwards negative (screen y is down)
    rain_gaps: float = 0.35     # gaps in a streak (0: solid): dotted, broken runs


def diffuse(img: np.ndarray, jitter: float = 0.0, rng=None) -> np.ndarray:
    """Floyd-Steinberg error diffusion, serpentine (each row the other way), so the dots run no
    one way: plain left-to-right scanning spaces them out across and stacks them up and down.
    `jitter` shakes the threshold, which breaks up the stripes and checks it settles into at mid grey."""
    H, W = img.shape
    e = np.zeros((H + 1, W + 2), np.float32)
    e[:H, 1:W + 1] = img
    rows = e.tolist()
    th = (0.5 + jitter * (rng.random((H, W + 2)) - 0.5) if jitter > 0 else np.full((H, W + 2), 0.5)).tolist()
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


def streak(A: dict, P: Streak | None = None, seed: int = 1) -> Frames:
    P = P or Streak()
    rng = np.random.default_rng(seed)
    a = A['alpha']
    t = (1 - P.even) * A['plain'] + P.even * A['tone']   # some of the plain light: real shadows stay dark
    H, W = t.shape
    c = np.float32(hex_rgb(P.colour))
    pal = np.array([[0, 0, 0], c * 0.62, c * 0.8, c], np.uint8)       # three shades of the one colour
    tt = np.clip(P.gain * np.clip((t - P.black) / (1 - P.black), 0, 1) ** P.gamma, 0, 1)
    if (a > 0.5).any():   # the brightest of the figure at full light, whatever the picture
        tt = np.clip(tt / max(np.percentile(tt[a > 0.5], 99), 0.3), 0, 1)
    if P.peak > 0 and (a > 0.5).any():   # the brightest light near solid, as on a hat brim or a lit cheek
        q = np.percentile(tt[a > 0.5], 100 * (1 - P.peak))
        tt = np.where(tt > q, np.maximum(tt, np.clip(0.88 + (tt - q) / max(1 - q, 1e-3), 0, 1)), tt)
    tt = (np.where(tt < P.floor, 0, tt) * P.most).astype(np.float32)
    if P.smooth > 0:   # the source's fine grain out first, or the diffusion draws it as blotches
        tt = cv2.GaussianBlur(tt, (0, 0), P.smooth)
    if P.features > 0:   # eyes, nostrils, mouth: small dark spots go darker, so they read as holes
        k = max(3, int(round(0.03 * max(H, W))) | 1)
        th = cv2.morphologyEx(tt, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
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
    alt = (rng.random(nr) < 0.5)[:, None] & (jj % 2 == 1) & (rng.random((nr, 1 + 0 * jj.shape[1])) > 0.25)          # some every other dot
    gaps = rng.random((nr, P.rain_len[1])) < P.rain_gaps + 0.8 * np.abs(2 * jj / rl[:, None] - 1) ** 3   # thinning at both ends
    gaps |= alt
    yy = np.mgrid[0:H, 0:W][0]
    low = (0.2 + 1.8 * (yy / H) ** 2).astype(np.float32)          # rain and specks heavier lower down
    hold_m = rng.random((P.frames, H, W)) < P.hold if P.hold > 0 else None
    busy = np.clip(cv2.GaussianBlur(tt, (0, 0), 6) * 3, 0, 1)        # where the figure is lit and dense

    out = np.zeros((P.frames, H, W), np.uint8)
    for f in range(P.frames):
        s = f / P.frames
        noisy = np.clip(tt + rng.normal(0, P.boil, (H, W)).astype(np.float32) * room, 0, 1)
        o = f % 4                       # Floyd-Steinberg runs one way: turn it each frame so no direction shows
        nz = noisy.T if o & 2 else noisy
        nz = nz[:, ::-1] if o & 1 else nz
        bits = diffuse(np.ascontiguousarray(nz), P.jitter, rng)
        bits = bits[:, ::-1] if o & 1 else bits
        bits = bits.T if o & 2 else bits
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
    gamma: float = 2.8          # tone curve: the mid-tones thin out, so the features read
    black: float = 0.12         # light below this is black
    most: float = 0.55          # the densest the specks get, in the brightest light
    clump: float = 0.0          # how much the specks clump (0: even white noise)
    outline: float = 0.6        # how much of the silhouette and strongest edges is drawn as bright strokes
    shade_lev: float = 0.6      # how far each speck's grey follows the light (0: any grey)
    order: float = 0.85         # how much of the body is an ordered (Bayer) dither, not noise: crisper shading
    edges: float = 0.35         # specks gathering on the outline (and less, the edges inside)
    flecks: int = 5000          # short flecks, placed afresh each frame (counts are for 540x540)
    fleck_len: tuple[int, int] = (2, 3)
    angle: float = 0            # the way the dust drifts off (degrees: 0 right, 90 down)
    dissolve: float = 0.97      # how much of the far side's body is gone (to motes)
    spare_lit: bool = True      # drift the other way if the far side holds the most light
    side: float = 0.7           # how much of the figure, from its far side along the angle, dissolves
    motes: int = 3500           # particles drifting off it
    reach: int = 220            # how far they go in a loop, in dots
    trail: int = 16             # motes to a trail, following one another
    curl: float = 0.1           # how much they follow the shapes they come from, not the angle
    blobs: int = 60             # soft grey blots among them
    sparkles: int = 5           # four-point glints on the brightest light, each for part of the loop
    mote_len: tuple[int, int] = (3, 8)   # dots in a mote: a short dash along its way
    falloff: float = 1.5        # how fast motes thin out as they go
    swirl: float = 20.0         # how far the drift turns region by region (degrees, typical)
    bend: float = 0.2           # how far the lanes curve off the angle
    comb: float = 0.6           # how far the flecks lie along the drift, not the shapes, inside the figure


def dust(A: dict, P: Dust | None = None, seed: int = 1) -> Frames:
    """The figure as boiling specks of many greys, densest in the light, with short flecks along
    its shapes; from one side it comes apart into particles drifting off along the angle."""
    P = P or Dust()
    rng = np.random.default_rng(seed)
    t, a = A['tone'], A['alpha']
    H, W = t.shape
    c = np.float32(hex_rgb(P.colour))
    lv = np.float32([0, 0.2, 0.32, 0.45, 0.6, 0.8, 1.0])
    pal = (lv[:, None] * c[None]).astype(np.uint8)
    NL = len(lv) - 1
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
    if (a > 0.5).mean() > 0.97:   # nothing cut out: no silhouette to draw
        sil[:] = 0
    dens = (np.clip(tt * P.most + P.edges * (0.7 * sil + 0.4 * ed ** 1.5 * (0.25 + 0.75 * tt)), 0, 0.7) * (1 - P.dissolve * sw)).astype(np.float32)
    many = H * W / 540 ** 2                                  # the counts, for this size

    bay = bayer(H, W)
    k8 = np.ones((3, 3), np.float32); k8[1, 1] = 0
    strong_ed = (ed > np.percentile(ed[a > 0.5], 97)) & (sw < 0.5) if (a > 0.5).any() else np.zeros_like(a, bool)
    rim = ((sil > 0.5) | strong_ed) & (a > 0.3)   # the outline holds even where the figure comes apart

    def flat_noise():
        n = rng.random((H, W)).astype(np.float32)
        if P.clump > 0:
            n = (1 - P.clump) * n + P.clump * cv2.GaussianBlur(n, (0, 0), 0.8)
            n = (np.argsort(np.argsort(n, axis=None)).reshape(H, W) / n.size).astype(np.float32)
        return n

    # the motes: born on the figure (mostly its far side), drifting off along the angle, bending
    # with the shapes they leave, fading in and out; each lives once round the loop, so it loops
    wm = (tt + 0.15 * a + 0.5 * sil) * (0.01 + sw) * (a > 0.2) * (0.4 + 1.2 * yy / H)   # more of them lower down
    # they go in trails: a few share each starting point and path, at different times round it
    nm = int(P.motes * many)
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
    sfar = (P.reach * math.sqrt(many) * rng.uniform(0.4, 1, k) * np.where(rng.random(k) < 0.15, rng.uniform(1.5, 2.5, k), 1) * (0.4 + 0.6 * sw[iy, ix])).astype(np.float32)
    sbend = rng.normal(0, P.bend, k).astype(np.float32)     # a slow drift sideways too: the lanes curve
    who = rng.integers(0, k, nm) if len(src) else np.zeros(0, int)
    n = len(who)
    m0 = src[who] + rng.normal(0, 0.6, (n, 2)).astype(np.float32)
    swob = rng.random(k).astype(np.float32) * 6.283           # and a slow wave across the way they go
    mv, far, bend, wob = sv[who], sfar[who], sbend[who], swob[who]
    perp = np.float32([-dv[1], dv[0]])
    mph = rng.random(n).astype(np.float32)
    mlen = rng.choice(np.arange(P.mote_len[0], P.mote_len[1] + 1), n, p=None) if P.mote_len[1] > P.mote_len[0] else np.full(n, P.mote_len[0])
    mlen = np.minimum(mlen, rng.choice(np.arange(P.mote_len[0], P.mote_len[1] + 1), n))   # short ones the most
    mlev = rng.integers(NL - 2, NL + 1, n)   # the top greys, white the least often
    blob = rng.random(n) < P.blobs * many / max(n, 1)
    bdots = rng.uniform(-3.5, 3.5, (n, 12, 2)).astype(np.float32)   # a blob: a loose round cluster
    bdots[np.hypot(bdots[..., 0], bdots[..., 1]) > 3.5] = np.nan
    blev = rng.integers(2, 5, (n, 12)).astype(np.uint8)
    # flecks run mostly the way of the drift, bent a little to the shapes
    fdir = np.stack([A['tx'], A['ty']], -1)
    fdir *= np.where((fdir @ dv) < 0, -1, 1)[..., None]
    fdir = (P.comb + (1 - P.comb) * sw)[..., None] * dv + ((1 - P.comb) * (1 - sw))[..., None] * fdir   # combed by the drift where it comes apart
    fdir /= np.linalg.norm(fdir, axis=-1, keepdims=True) + 1e-6

    spk = seed_where((tt ** 4 * (a > 0.5) * (sw < 0.6)).astype(np.float32), max(0, int(round(P.sparkles * many))), rng)
    sph, sln = rng.random(len(spk)), rng.integers(2, 4, len(spk))
    inner = cv2.erode((a > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (sw < 0.3)
    near = cv2.dilate((a > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out = np.zeros((P.frames, H, W), np.uint8)
    for f in range(P.frames):
        s = f / P.frames
        # the body: specks where noise falls under the light, each its own grey
        lit = ((1 - P.order) * flat_noise() + P.order * bay) < dens
        lev = rng.integers(1, NL + 1, (H, W))
        if P.shade_lev > 0:   # brighter specks where it's lit: the greys follow the light, not chance
            lev = np.clip(np.round((1 - P.shade_lev) * lev + P.shade_lev * (1.5 + (NL - 1) * np.sqrt(tt))), 1, NL).astype(int)
        if P.outline > 0:   # the silhouette and strongest edges as bright strokes; stray dim specks go
            lit &= ~((lev <= 3) & (cv2.filter2D(lit.astype(np.uint8), -1, k8, borderType=cv2.BORDER_CONSTANT) == 0))
            stroke = rim & (rng.random((H, W)) < P.outline * (1 - 0.8 * sw))   # thinning where the figure comes apart
            lit |= stroke
            lev = np.where(stroke, NL - rng.integers(0, 2, (H, W)), lev)
        img = np.where(lit, lev, 0).astype(np.uint8)
        # flecks along the shapes, mostly in the lit and on edges
        wf = (A['edge'] * A['coh'] * tt + 0.3 * tt) * dens
        fs = seed_where(wf.astype(np.float32), int(P.flecks * many), rng)
        if len(fs):
            fx, fy = fs[:, 0].astype(int), fs[:, 1].astype(int)
            fv = fdir[fy, fx]
            ln = rng.integers(P.fleck_len[0], P.fleck_len[1] + 1, len(fs))
            fl = rng.integers(2, NL + 1, len(fs)).astype(np.uint8)
            wig = rng.normal(0, 0.15, (len(fs), 2)).astype(np.float32)
            for j in range(P.fleck_len[1]):
                on = j < ln
                plot(img, (fs + (fv + wig) * (j - ln[:, None] / 2))[on], fl[on])
        # the motes
        u = np.mod(mph + s, 1)
        pos = m0 + mv * (far * u)[:, None] + perp * (bend * far * u * u + 3 * np.sin(wob + 6.283 * u) - 3 * np.sin(wob))[:, None]
        fade = np.minimum(1, u * 8) * (1 - u) ** P.falloff   # thickest as they leave, thinning out as they go
        vel = mv * far[:, None] + perp * (2 * bend * far * u + 6 * np.pi * np.cos(wob + 6.283 * u))[:, None]
        vel /= np.linalg.norm(vel, axis=1, keepdims=True) + 1e-6   # each dash along the way it is going
        vis = rng.random(n) < fade ** 0.7                 # they fade by thinning out, each one staying crisp
        pi_ = np.clip(np.round(pos).astype(int), 0, [W - 1, H - 1])
        vis &= ~inner[pi_[:, 1], pi_[:, 0]]               # none over the whole of the figure: it stays clean
        ml = np.where(vis, np.clip(mlev, 1, NL), 0).astype(np.uint8)
        for j in range(P.mote_len[1]):
            on = vis & (j < mlen) & ~blob
            plot(img, (pos - vel * j)[on], ml[on])
        bi = np.nonzero(blob & vis)[0]
        if len(bi):
            pts = (pos[bi, None] + bdots[bi]).reshape(-1, 2)
            lvv = np.minimum(blev[bi], ml[bi, None]).reshape(-1)
            ok = ~np.isnan(pts[:, 0])
            plot(img, pts[ok], lvv[ok])
        # sparkles: a few four-point glints on the brightest light, each lit for part of the loop
        for (sx, sy), sp, sl in zip(spk, sph, sln):
            on = np.mod(s - sp, 1) < 0.25
            if on:
                arm = sl if np.mod(s - sp, 1) < 0.15 else max(1, sl - 1)
                pts = [(sx, sy)] + [(sx + d * j, sy) for d in (-1, 1) for j in range(1, arm + 1)] + [(sx, sy + d * j) for d in (-1, 1) for j in range(1, arm + 1)]
                lv8 = [NL] + [NL - 1] * (len(pts) - 1)
                plot(img, np.float32(pts), np.uint8(lv8))
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


def params(style: str, frames: int | None = None, colour: str | None = None, angle: float | None = None):
    """A style's settings, with the given ones changed."""
    P = STYLES[style][1]()
    if frames:
        P.frames = frames
    if colour and isinstance(P, (Streak, Dust)):
        P.colour = colour if colour.startswith('#') else COLOURS.get(colour, P.colour)
    if angle is not None and isinstance(P, (Glow, Dust)):
        P.angle = angle
    return P


def render(rgba: np.ndarray, style: str, frames: int | None = None, colour: str | None = None,
           angle: float | None = None, seed: int | None = None) -> Frames:
    """A framed picture (RGBA at the output size) as a clip in a style. Without a seed, one from
    the picture: the same picture comes out the same, and no two pictures share their rain."""
    if seed is None:
        seed = zlib.crc32(np.ascontiguousarray(rgba[::7, ::7]).tobytes())
    return STYLES[style][0](analyse(rgba), params(style, frames, colour, angle), seed)


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
    ap.add_argument('--angle', type=float, help='glow, dust: which way the streams or the dust go (degrees, 0 right, 90 down)')
    ap.add_argument('--seed', type=int, help='default: one from the picture')
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
        fr = STYLES[name][0](analyse(rgba), P, a.seed if a.seed is not None else zlib.crc32(np.ascontiguousarray(rgba[::7, ::7]).tobytes()))
        for p in save(fr, out, name, a.fps or P.fps, a.scale or P.scale, tuple(a.formats.split(','))):
            print(p)


if __name__ == '__main__':
    main()
