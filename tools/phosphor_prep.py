#!/usr/bin/env python3
"""Get a picture ready for phosphor_map: cut the figure out, frame it, and paint a blink.

Everything here works on the source picture, in its own px (RGB uint8, HxWx3), so the steps can
be redone at any output size. phosphor_web.py drives it; it is also a module:

    alpha = cutout(rgb, rect=(x0, y0, x1, y1))           # 0..1, the figure
    view = auto_view(alpha, (340, 280))                  # the part of the picture the map shows
    shut = blink(rgb, [(x, y, r), (x, y, r)])            # the same picture, eyes shut
    idle = frame(rgb, alpha, view, (340, 280))           # RGBA at the map's size, for build()

Needs numpy, opencv-python(-headless) and pillow.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

View = tuple[float, float, float]   # left, top, width (source px); the height follows the map's aspect
Eye = tuple[float, float, float]    # centre x, y and half-width (source px)


def cutout(rgb: np.ndarray, rect: Sequence[float] | None = None,
           strokes: Iterable[tuple[float, float, float, bool]] = (), iters: int = 5) -> np.ndarray:
    """The figure's alpha (0..1, HxW), by GrabCut. rect: x0, y0, x1, y1 around the figure (default:
    the picture less 5% at the top and sides). strokes: brush dabs (x, y, radius, keep) that mark
    the figure (keep) or the background for sure. Worked at most 640 px across, then scaled back."""
    H, W = rgb.shape[:2]
    s = min(1.0, 640 / max(H, W))
    small = cv2.resize(rgb, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    h, w = small.shape[:2]
    x0, y0, x1, y1 = rect if rect is not None else (W * 0.05, H * 0.05, W * 0.95, H)
    x0, y0 = max(0, int(round(x0 * s))), max(0, int(round(y0 * s)))
    x1, y1 = min(w, int(round(x1 * s))), min(h, int(round(y1 * s)))
    mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    if not (mask == cv2.GC_BGD).any():  # the box is the whole picture: let its rim be maybe-background
        mask[:2], mask[-2:], mask[:, :2], mask[:, -2:] = (cv2.GC_PR_BGD,) * 4
    for x, y, r, keep in strokes:
        cv2.circle(mask, (int(x * s), int(y * s)), max(1, int(r * s)), cv2.GC_FGD if keep else cv2.GC_BGD, -1)
    if (mask == cv2.GC_FGD).any() or (mask == cv2.GC_PR_FGD).any():
        bm, fm = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(np.ascontiguousarray(small[..., ::-1]), mask, None, bm, fm, iters, cv2.GC_INIT_WITH_MASK)
    m = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # the largest piece, with its holes filled
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros((H, W), np.float32)
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    n, lab = cv2.connectedComponents(1 - m)
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        if ys.min() > 0 and xs.min() > 0 and xs.max() < w - 1 and ys.max() < h - 1:
            m[lab == i] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32)
    a = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) if s < 1 else m
    k = max(3, int(round(2 / s)) | 1)
    return np.clip(cv2.GaussianBlur(a, (k, k), 0), 0, 1)


def auto_view(alpha: np.ndarray, size: tuple[int, int]) -> View:
    """A view that fits the figure into a map of `size` (W, H): its full height with a little room
    on top (a figure cut off at the picture's bottom stays cut off there), centred across."""
    W, H = size
    ys, xs = np.nonzero(alpha > 0.5)
    if not len(ys):
        h, w = alpha.shape
        vw = min(w, h * W / H)
        return ((w - vw) / 2, (h - vw * H / W) / 2, vw)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    vh = (y1 - y0) * 1.06
    vw = vh * W / H
    if (x1 - x0) * 1.1 > vw:
        vw = (x1 - x0) * 1.1
        vh = vw * H / W
    cut = y1 >= alpha.shape[0] - 2
    top = y1 - vh if cut else y0 - (vh - (y1 - y0)) / 2
    return (float((x0 + x1) / 2 - vw / 2), float(top), float(vw))


def frame(rgb: np.ndarray, alpha: np.ndarray, view: View, size: tuple[int, int]) -> np.ndarray:
    """The view of the picture at the map's size (W, H): RGBA uint8, the figure's alpha in A.
    Outside the picture is transparent."""
    W, H = size
    x, y, vw = view
    vh = vw * H / W
    im = Image.fromarray(np.dstack([rgb, (np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)]), 'RGBA')
    ix, iy = int(np.floor(x)), int(np.floor(y))
    im = im.crop((ix, iy, int(np.ceil(x + vw)) + 1, int(np.ceil(y + vh)) + 1))  # pads past the edges
    return np.asarray(im.resize((W, H), Image.LANCZOS, box=(x - ix, y - iy, x - ix + vw, y - iy + vh)))


def blink(rgb: np.ndarray, eyes: Sequence[Eye]) -> np.ndarray:
    """The picture with its eyes shut: each eye (x, y, half-width r) painted over in the skin just
    below it, and a closed lid drawn across, sagging a little, with a flick at the outer corner
    (away from the other eye)."""
    img = rgb.copy()
    mid = np.mean([e[0] for e in eyes]) if len(eyes) > 1 else None
    for x, y, r in eyes:
        ry = r * 0.65
        sy, sx = int(min(img.shape[0] - 4, y + r * 1.15)), int(np.clip(x, 3, img.shape[1] - 4))
        skin = np.median(rgb[max(0, sy - 3):sy + 4, sx - 3:sx + 4].reshape(-1, 3), 0)
        lash = tuple(int(v) for v in np.percentile(rgb[int(max(0, y - ry)):int(y + ry), int(max(0, x - r)):int(x + r)]
                                                   .reshape(-1, 3), 2, axis=0))
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(m, (int(x), int(y)), (int(r), int(ry)), 0, 0, 360, 255, -1)
        k = max(3, int(r / 6) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)[..., None] / 255
        img = (img * (1 - m) + skin * m).astype(np.uint8)
        t = np.linspace(0, 1, 40)
        xs = x - r * 0.9 + r * 1.8 * t
        ys = y + ry * 0.2 + r * 0.22 * 4 * t * (1 - t)
        th = max(2, int(round(r / 8)))
        cv2.polylines(img, [np.stack([xs, ys], 1).round().astype(np.int32)], False, lash, th, cv2.LINE_AA)
        if mid is not None:
            d = 1 if x > mid else -1
            ox, oy = x + d * r * 0.9, y + ry * 0.2
            cv2.line(img, (int(ox), int(oy)), (int(ox + d * r * 0.17), int(oy + r * 0.12)), lash, max(1, th - 1), cv2.LINE_AA)
    return img


def guess_pose(eyes: Sequence[Eye], view: View, size: tuple[int, int]) -> dict | None:
    """Where the head is, from the eyes, in map px: `center` (the middle of the head) and `pivot`
    (the top of the neck, which it tilts about). None without eyes."""
    if not eyes:
        return None
    W, _ = size
    x, y, vw = view
    ex, ey = np.mean([e[0] for e in eyes]), np.mean([e[1] for e in eyes])
    d = abs(eyes[1][0] - eyes[0][0]) if len(eyes) > 1 else eyes[0][2] * 3.5
    k = W / vw
    to_map = lambda px, py: [round(float((px - x) * k), 1), round(float((py - y) * k), 1)]
    return {'center': to_map(ex, ey - 0.5 * d), 'pivot': to_map(ex, ey + 0.9 * d)}
