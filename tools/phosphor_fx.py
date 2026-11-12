#!/usr/bin/env python3
"""Render a picture as a looping clip of 1 px dots, in one of two styles.

    phosphor_fx.py PICTURE [--style glow|streak] [--size WxH] [--scale N] [--frames N] [--fps N]
                   [--cutout] [--colour C] [--seed N] [--out DIR] [--formats gif,webp,png]

  glow    many colours: dot colour follows the light (blue, cyan, green, orange, pink at the
          brightest), edges drawn as a bright rim in a warm colour, the shading hatched along the
          surface, the dots shimmering, dust drifting off the edges and drips falling under them.
  streak  one colour (--colour, default red): the figure as a fine stipple, short dashes flowing
          along its shapes, and rain streaking across the whole frame.

Both are drawn on black, at --size (default 540x540, one dot per px), and saved at --scale times
that (default 2, each dot as 2x2 px). The clip loops without a seam: everything that moves comes
back to where it started after --frames frames.

A picture with transparency is taken as cut out already; --cutout cuts the figure out of any other
(GrabCut, as phosphor_prep does). Otherwise the whole picture is used, framed to cover --size.

Writes DIR/<name>.gif, DIR/<name>.webp (animated, lossless: the same dots, a fraction of the size)
and DIR/<name>/0000.png... plus DIR/<name>.png (the first frame), where DIR defaults to
out/<picture name>/fx and <name> is the style. Lossy video (MP4, WebM) smears 1 px dots and comes
out bigger than the GIF, so there is none.

Also a module: analyse() reads a framed picture, glow() and streak() return the frames (as palette
indices plus the palette), save() writes them. Needs numpy, opencv-python(-headless) and pillow.
"""
from __future__ import annotations

import argparse
import io
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
    tone   0..1, the light (local contrast lifted, 0 outside the figure)
    alpha  0..1, the figure
    tx, ty the direction along the shapes at each px (unit; an axis, so its sign means nothing)
    coh    0..1, how strongly the shapes there run one way
    edge   0/1, a thin line where the light changes sharply, and round the figure
    rim    0/1, the figure's outline
    out    the outward direction at the outline (unit x, y), for things leaving it"""
    rgb = rgba[..., :3].astype(np.float32) / 255
    a = rgba[..., 3].astype(np.float32) / 255
    H, W = a.shape
    lum = (rgb @ np.float32([0.299, 0.587, 0.114]))
    # tone: equalised locally so the shading reads, mixed back with the plain light
    l8 = (lum * 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l8).astype(np.float32) / 255
    t = 0.55 * lum + 0.45 * eq
    inside = a > 0.5
    if inside.any():
        lo, hi = np.percentile(t[inside], [2, 99.5])
        t = np.clip((t - lo) / max(hi - lo, 1e-3), 0, 1)
    tone = t * a

    # structure tensor: the direction along the shapes
    g = cv2.GaussianBlur(lum, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    def tensor(s: float):
        return [cv2.GaussianBlur(v, (0, 0), s) for v in (gx * gx, gx * gy, gy * gy)]
    j1, j2 = tensor(2.5), tensor(14)
    jxx, jxy, jyy = (p + 0.35 * q for p, q in zip(j1, j2))
    jxx += 1e-6  # nothing there at all: lie horizontal
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2  # across the gradient
    tx, ty = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    lam = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = (lam / (jxx + jyy + 1e-6)).astype(np.float32)

    # edges: Canny on the light (thresholds from its own gradients), plus the outline
    med = float(np.median(np.hypot(gx, gy)[inside])) if inside.any() else 0.05
    lo_t = int(np.clip(med * 255 * 2.2, 8, 80))
    edge = cv2.Canny(cv2.GaussianBlur(l8, (0, 0), 1.2), lo_t, lo_t * 2.5) > 0
    m = inside.astype(np.uint8)
    rim = (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0 if not inside.all() else np.zeros_like(inside)
    rim[[0, -1], :] = rim[:, [0, -1]] = False  # where the picture is cut off is no edge
    edge = (edge & (a > 0.3)) | rim
    # outward direction: down the blurred figure's slope
    sa = cv2.GaussianBlur(a, (0, 0), 4)
    ox, oy = -cv2.Sobel(sa, cv2.CV_32F, 1, 0, ksize=3), -cv2.Sobel(sa, cv2.CV_32F, 0, 1, ksize=3)
    n = np.hypot(ox, oy) + 1e-6
    return dict(tone=tone.astype(np.float32), alpha=a, tx=tx, ty=ty, coh=coh, edge=edge, rim=rim,
                out=(ox / n, oy / n), size=(W, H))


def lic(A: dict, noise: np.ndarray, steps: int = 7) -> np.ndarray:
    """Noise smeared along the shapes (line integral convolution): streaks that follow them."""
    H, W = noise.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    acc, n = noise.copy(), 1
    for sgn in (1, -1):
        px, py = xs.copy(), ys.copy()
        dx, dy = A['tx'] * sgn, A['ty'] * sgn
        for _ in range(steps):
            vx = cv2.remap(A['tx'], px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            vy = cv2.remap(A['ty'], px, py, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            flip = np.where(vx * dx + vy * dy < 0, -1, 1).astype(np.float32)
            dx, dy = vx * flip, vy * flip
            px, py = px + dx, py + dy
            acc += cv2.remap(noise, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            n += 1
    return acc / n


def rank(v: np.ndarray) -> np.ndarray:
    """Values made uniform on 0..1, order kept (so as a dither threshold they keep the density right)."""
    r = np.empty(v.size, np.float32)
    r[np.argsort(v, axis=None, kind='stable')] = np.linspace(0, 1, v.size, endpoint=False, dtype=np.float32)
    return r.reshape(v.shape)


def flow_paths(A: dict, seeds: np.ndarray, steps: int, speed: float, wind: tuple[float, float],
               follow: float, rng: np.random.Generator) -> np.ndarray:
    """(N, steps, 2) positions: each seed carried along the shapes (`follow` of the way; the rest
    is `wind`), a step of `speed` px at a time, keeping its heading."""
    H, W = A['tone'].shape
    p = seeds.astype(np.float32).copy()
    out = np.empty((len(p), steps, 2), np.float32)
    wx, wy = wind
    d = np.tile(np.float32([wx, wy]), (len(p), 1))
    for k in range(steps):
        out[:, k] = p
        ix = np.clip(p[:, 0].astype(int), 0, W - 1)
        iy = np.clip(p[:, 1].astype(int), 0, H - 1)
        v = np.stack([A['tx'][iy, ix], A['ty'][iy, ix]], 1)
        v *= np.where((v * d).sum(1) < 0, -1, 1)[:, None]  # keep going the way it was
        f = follow * np.clip(A['coh'][iy, ix] * 1.6, 0, 1)[:, None]
        d = f * v + (1 - f) * np.float32([wx, wy])
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        p = p + d * speed + rng.normal(0, 0.12, p.shape).astype(np.float32)
    return out


def plot(img: np.ndarray, pts: np.ndarray, val) -> None:
    """Set the px at pts (M, 2) float, those inside, to val (scalar or (M,) array)."""
    H, W = img.shape
    x, y = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    img[y[ok], x[ok]] = val[ok] if isinstance(val, np.ndarray) else val


def seed_where(weight: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n positions (x, y), drawn with probability following weight (>= 0)."""
    w = weight.ravel().astype(np.float64)
    if w.sum() <= 0:
        return np.zeros((0, 2), np.float32)
    i = rng.choice(w.size, size=n, p=w / w.sum())
    H, W = weight.shape
    return np.stack([i % W + rng.random(n), i // W + rng.random(n)], 1).astype(np.float32)


def phase(n: int, rng: np.random.Generator, ks=(1,)) -> tuple[np.ndarray, np.ndarray]:
    """Per particle: where it starts in its cycle (0..1) and how many cycles it runs per loop."""
    return rng.random(n).astype(np.float32), rng.choice(np.array(ks), n).astype(np.float32)


# ---- colour

def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def ramp(stops: list[tuple[float, str]], n: int) -> np.ndarray:
    """n RGB colours along the stops (position 0..1, colour)."""
    xs = np.float32([s for s, _ in stops])
    cs = np.float32([hex_rgb(c) for _, c in stops])
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, xs, cs[:, i]) for i in range(3)], 1).round().astype(np.uint8)


# ---- glow

@dataclass
class Glow:
    frames: int = 48
    density: float = 1.15       # how much light makes a dot
    gamma: float = 1.25         # tone curve (higher: the dark thins out more)
    most: float = 0.8           # the densest the dots get (below 1, so flat light keeps a texture)
    hatch: float = 0.55         # how much the dots line up along the shapes (0: plain noise)
    shimmer: float = 0.18       # share of dots that flicker over the loop
    jitter: float = 0.09        # colour spread between neighbouring dots
    rim: float = 0.9            # how much of the outline is lit
    lines: float = 0.45         # how much of the inner edges are lit
    dust: int = 3200            # specks drifting off the edges
    dust_reach: float = 70      # how far they drift (px)
    drips: int = 160            # dotted lines falling under the lit parts
    drip_len: float = 90        # how far they fall (px)
    wind: tuple[float, float] = (0.85, -0.35)  # the way the dust drifts (x right, y down)
    palette: list = field(default_factory=lambda: [
        (0.00, '#1d2f9e'), (0.22, '#2c57d8'), (0.45, '#2a9ee0'), (0.60, '#22c7b8'),
        (0.74, '#62e07a'), (0.87, '#ffb347'), (1.00, '#ff8fb8')])
    rim_palette: list = field(default_factory=lambda: [
        (0.0, '#ff6fa8'), (0.6, '#ff9c7a'), (1.0, '#ffd2a0')])


def glow(A: dict, P: Glow | None = None, seed: int = 1) -> Frames:
    P = P or Glow()
    rng = np.random.default_rng(seed)
    H, W = A['tone'].shape
    N = P.frames
    K = 48                                     # colours along the body ramp; the rim's follow
    pal = np.concatenate([np.zeros((1, 3), np.uint8), ramp(P.palette, K), ramp(P.rim_palette, 16)])
    RIM0 = 1 + K

    t = A['tone']
    p = np.clip(P.density * t ** P.gamma, 0, P.most)       # chance of a dot
    noise = rng.random((H, W)).astype(np.float32)
    hw = P.hatch * np.clip(A['coh'] * 2.2 - 0.2, 0, 1)    # hatch only where the shapes run one way
    thr = rank((1 - hw) * noise + hw * rank(lic(A, noise)))
    flick = rng.random((H, W)) < P.shimmer
    kf = np.where(flick, rng.choice([-1, 1, 2], (H, W)), 0).astype(np.float32)
    # colour: by light, spread a little per dot, and per dot the same all loop
    cv = np.clip(t + rng.normal(0, P.jitter, (H, W)), 0, 1)
    body_c = (1 + np.round(cv * (K - 1))).astype(np.uint8)
    e = A['edge'] & ~A['rim']
    # the outline in the warm rim colours; inner edges in the body's, a step brighter
    rim_on = A['rim'] & (rng.random((H, W)) < P.rim)
    line_on = e & (rng.random((H, W)) < P.lines * (0.3 + t))
    rim_c = (RIM0 + np.round(np.clip(t + rng.normal(0, 0.15, (H, W)), 0, 1) * 15)).astype(np.uint8)
    line_c = (1 + np.round(np.clip(t + 0.3 + rng.normal(0, 0.08, (H, W)), 0, 1) * (K - 1))).astype(np.uint8)

    # dust: from the outline (mostly the side the wind blows off) and the edges, drifting away
    wx, wy = P.wind
    lee = np.clip(A['out'][0] * wx + A['out'][1] * wy, 0, 1) if A['rim'].any() else 1
    w_dust = (A['rim'] * (0.25 + lee) * 3 + e * 0.6).astype(np.float32)
    steps = max(8, int(P.dust_reach))
    ds = seed_where(w_dust, P.dust, rng)
    dpaths = flow_paths(A, ds, steps, 1.0, (wx, wy), 0.35, rng)
    dph, dk = phase(len(ds), rng)
    dcol = (1 + np.round(np.clip(t[np.clip(ds[:, 1].astype(int), 0, H - 1), np.clip(ds[:, 0].astype(int), 0, W - 1)]
                                  + 0.2, 0, 1) * (K - 1))).astype(np.uint8)
    # drips: from lit px with nothing lit right under them, falling
    lit = p > 0.35
    under = np.zeros_like(lit); under[:-6] = ~lit[6:]
    rs = seed_where((lit & under & (A['alpha'] > 0.5)).astype(np.float32) * (0.2 + t), P.drips, rng)
    rph, rk = phase(len(rs), rng)
    rlen = rng.uniform(0.3, 1, len(rs)) * P.drip_len
    rcol = (1 + np.round(np.clip(t[np.clip(rs[:, 1].astype(int), 0, H - 1), np.clip(rs[:, 0].astype(int), 0, W - 1)],
                                 0, 1) * (K - 1))).astype(np.uint8)

    out = np.zeros((N, H, W), np.uint8)
    for f in range(N):
        s = f / N
        u = np.mod(thr + kf * s, 1)
        img = np.where(u < p, body_c, 0).astype(np.uint8)
        img[line_on] = line_c[line_on]
        img[rim_on] = rim_c[rim_on]
        # dust: age 0..1 along its path, thinning out as it goes
        age = np.mod(dph + dk * s, 1)
        idx = np.minimum((age * steps).astype(int), steps - 1)
        keep = rng.random(len(ds)) > age ** 0.7
        plot(img, dpaths[np.arange(len(ds)), idx][keep], dcol[keep])
        # drips: a dotted run of up to 5 dots, falling and fading
        age = np.mod(rph + rk * s, 1)
        for j in range(5):
            y = rs[:, 1] + age * rlen - j * 3
            on = (y > rs[:, 1]) & (rng.random(len(rs)) > age * 0.9)
            plot(img, np.stack([rs[:, 0], y], 1)[on], rcol[on])
        out[f] = img
    return out, pal


# ---- streak

@dataclass
class Streak:
    frames: int = 48
    colour: str = '#ff2020'
    density: float = 1.05       # how much light makes a dot
    gamma: float = 1.4          # tone curve
    most: float = 0.82          # the densest the stipple gets
    shimmer: float = 0.3        # share of the stipple that flickers
    dashes: int = 2600          # dashes flowing along the shapes
    dash_len: tuple[int, int] = (3, 9)
    flow_speed: float = 1.2     # px per step along the shapes
    flow_reach: int = 60        # steps in a dash's life
    rain: int = 300             # streaks across the frame
    rain_len: tuple[int, int] = (8, 26)
    rain_angle: float = -22     # degrees from the right, upwards negative (screen y is down)
    rain_dash: float = 0.2      # gaps in a streak (0: solid)
    wind: tuple[float, float] = (1.0, -0.2)  # which way along the shapes the dashes flow


def streak(A: dict, P: Streak | None = None, seed: int = 1) -> Frames:
    P = P or Streak()
    rng = np.random.default_rng(seed)
    H, W = A['tone'].shape
    N = P.frames
    pal = np.array([[0, 0, 0], hex_rgb(P.colour)], np.uint8)
    t = A['tone']
    p = np.clip(P.density * t ** P.gamma, 0, P.most)
    thr = rng.random((H, W)).astype(np.float32)
    kf = np.where(rng.random((H, W)) < P.shimmer, rng.choice([-1, 1], (H, W)), 0).astype(np.float32)

    # dashes: seeded where the shapes run one way and there's light, carried along them
    w = (t * (0.25 + A['coh']) + A['edge'] * 0.4) * (A['alpha'] > 0.2)
    ds = seed_where(w.astype(np.float32), P.dashes, rng)
    steps = P.flow_reach + P.dash_len[1]
    paths = flow_paths(A, ds, steps, P.flow_speed, P.wind, 0.9, rng)
    dph, dk = phase(len(ds), rng, (1, 1, 2))
    dlen = rng.integers(P.dash_len[0], P.dash_len[1] + 1, len(ds))

    # rain: straight streaks crossing the frame, each wrapping round in its own time
    a = np.radians(P.rain_angle)
    dv = np.float32([np.cos(a), np.sin(a)])
    span = np.hypot(W, H) + P.rain_len[1] * 2
    perp = np.float32([-dv[1], dv[0]])
    c = np.float32([W / 2, H / 2])
    rp = rng.uniform(-span / 2, span / 2, P.rain)   # across
    rph, rk = phase(P.rain, rng, (1, 2, 3))
    rl = rng.integers(P.rain_len[0], P.rain_len[1] + 1, P.rain)

    out = np.zeros((N, H, W), np.uint8)
    for f in range(N):
        s = f / N
        u = np.mod(thr + kf * s, 1)
        img = (u < p).astype(np.uint8)
        # dashes
        age = np.mod(dph + dk * s, 1)
        head = (age * P.flow_reach).astype(int) + P.dash_len[1]
        fade = np.minimum(age, 1 - age) * 2                    # in and out over its life
        for j in range(P.dash_len[1]):
            on = (j < dlen) & (rng.random(len(ds)) < 0.25 + fade)
            plot(img, paths[np.arange(len(ds)), head - j][on], 1)
        # rain
        along = (np.mod(rph + rk * s, 1) - 0.5) * span
        for j in range(P.rain_len[1]):
            on = (j < rl) & (rng.random(P.rain) > P.rain_dash)
            pts = c + perp * rp[:, None] + dv * (along - j)[:, None]
            plot(img, pts[on], 1)
        out[f] = img
    return out, pal


STYLES = {'glow': (glow, Glow), 'streak': (streak, Streak)}


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


def encode(fr: Frames, fmt: str, fps: int = 16, scale: int = 2) -> bytes:
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


def save(fr: Frames, folder: Path, name: str, fps: int = 16, scale: int = 2,
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


COLOURS = {'red': '#ff2020', 'white': '#f2f2f2', 'green': '#4fe08a', 'blue': '#6f93e8', 'amber': '#ffb347',
           'rose': '#f07aa6'}


def render(rgba: np.ndarray, style: str, frames: int = 48, colour: str | None = None, seed: int = 1) -> Frames:
    """A framed picture (RGBA at the output size) as a clip in a style."""
    fn, cls = STYLES[style]
    P = cls(frames=frames)
    if colour and isinstance(P, Streak):
        P.colour = colour if colour.startswith('#') else COLOURS.get(colour, P.colour)
    return fn(analyse(rgba), P, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('picture', type=Path)
    ap.add_argument('--style', choices=list(STYLES) + ['all'], default='all')
    ap.add_argument('--size', default='540x540', help='WxH, one dot per px (default 540x540)')
    ap.add_argument('--scale', type=int, default=2, help='each dot as NxN px in the files (default 2)')
    ap.add_argument('--frames', type=int, default=48)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--cutout', action='store_true', help='cut the figure out (GrabCut) first')
    ap.add_argument('--colour', help='streak: the colour (hex, or ' + ', '.join(COLOURS) + ')')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', type=Path, help='default out/<picture>/fx')
    ap.add_argument('--formats', default='gif,webp,png')
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.lower().split('x'))
    rgba = prepare(Image.open(a.picture), (W, H), a.cutout)
    out = a.out or Path('out') / a.picture.stem / 'fx'
    for name in (STYLES if a.style == 'all' else [a.style]):
        fr = render(rgba, name, a.frames, a.colour, a.seed)
        for p in save(fr, out, name, a.fps, a.scale, tuple(a.formats.split(','))):
            print(p)


if __name__ == '__main__':
    main()
