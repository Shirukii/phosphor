#!/usr/bin/env python3
"""Clean the benchmark inputs: frames, stands, plinths and cast shadows painted out to black.

  clean_inputs.py RAW_DIR OUT_DIR

Per picture, a list of steps on the 540 x 540 frame:
  ('keep', x0, y0, x1, y1, feather)        everything outside the box fades to black
  ('erase', [(x, y), ...], feather)        the polygon goes black
  ('pale', x0, y0, x1, y1, sat, val)       in the box, pale flat grey or white (saturation below sat,
                                           brightness above val: studio paper, shadow) goes black
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

STEPS = {
    'angel_wings': [('keep', 128, 62, 416, 478, 3), ('erase', [(240, 462), (300, 462), (300, 480), (240, 480)], 2)],
    'lion_durer': [('keep', 100, 50, 440, 490, 25)],
    'artemis_rocket': [('keep', 120, 40, 352, 268, 40, 'upper')],
    'burgonet': [('erase', [(290, 328), (343, 345), (343, 445), (290, 445)], 2), ('keep', 0, 0, 540, 428, 10)],
    'close_helmet': [('erase', [(135, 405), (183, 405), (183, 450), (135, 450)], 3)],
    'gauntlet': [('pale', 55, 255, 445, 435, 0.05, 0.08), ('pale', 215, 322, 345, 370, 0.14, 0.2)],
    'armor_suit': [('pale', 272, 375, 350, 485, 0.14, 0.35), ('pale', 296, 190, 335, 300, 0.16, 0.45),
                   ('pale', 225, 290, 275, 485, 0.08, 0.6)],
    'dragon_bronze': [('pale', 105, 335, 405, 485, 0.22, 0.5)],
    'helmet': [('erase', [(250, 392), (292, 392), (292, 495), (250, 495)], 2)],
    'helmet_dragon': [('pale', 0, 0, 540, 540, 0.13, 0.7), ('erase', [(299, 372), (317, 372), (317, 468), (299, 468)], 1.5)],
    'skull_met': [('pale', 150, 330, 480, 445, 0.04, 0.1)],
}


def clean(rgb: np.ndarray, steps: list) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    h, w = rgb.shape[:2]
    for s in steps:
        m = np.ones((h, w), np.float32)
        if s[0] == 'keep':
            x0, y0, x1, y1, f = s[1:6]
            m[:] = 0
            m[y0:y1, x0:x1] = 1
            if len(s) > 6:   # only this part of the picture is boxed: the rest is left alone
                m[y1:, :] = 1
            m = cv2.GaussianBlur(cv2.erode(m, np.ones((2 * f + 1, 2 * f + 1), np.uint8)), (0, 0), max(f / 2, 0.5))
            if len(s) > 6:
                m[y1 + f:, :] = 1
        elif s[0] == 'erase':
            cv2.fillPoly(m, [np.array(s[1], np.int32)], 0)
            m = cv2.GaussianBlur(m, (0, 0), s[2])
        else:
            x0, y0, x1, y1, sat, val = s[1:]
            hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255
            pale = (hsv[..., 1] < sat) & (hsv[..., 2] > val)
            box = np.zeros((h, w), bool)
            box[y0:y1, x0:x1] = True
            pale = cv2.morphologyEx((pale & box).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = 1 - cv2.GaussianBlur(cv2.dilate(pale, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 1.0)
        rgb = rgb * m[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(raw.glob('*.540.png')):
        n = p.name.split('.')[0]
        rgb = np.asarray(Image.open(p).convert('RGB'))
        if n in STEPS:
            rgb = clean(rgb, STEPS[n])
        Image.fromarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), 'RGBA').save(out / p.name)
    print('cleaned', len(STEPS), 'of', len(list(raw.glob('*.540.png'))))


if __name__ == '__main__':
    main()
