# Phosphor

A picture drawn as a grid of LED-like dots: an ordered (Bayer) dither in one WebGL1 fragment shader,
from a 4-bit brightness map bundled in your JS. No image files at runtime.

It is alive, not a still:

- a soft light field drifts across the dots, so they flip on and off locally
- the figure breathes: the head tilts back a little about the neck, the shoulders rise, the hair follows a moment later
- frames (a blink, a glance) crossfade through the dither, or swap at once
- particles drift off the back and top of the figure, shrink and flicker out
- a slowly turning burst of dotted rays sits behind it, kept out of the silhouette
- asleep, the head droops, breathing slows and everything calms
- at startup the grid resolves from giant blocks down to single dots
- moods (`idle`, `listening`, `thinking`, `speaking`) change how it moves, never how lit it is

It runs at 30 fps on a Mali-T720 (an Echo Show 5) at 340×280.

## Parts

| path | what |
|---|---|
| `tools/phosphor_map.py` | shaded image + line-art frames → `map.ts` (the brightness map) |
| `src/phosphor.ts` | the renderer: one class, no dependencies |

## 1. Make a map

You need:

- **a shaded picture**: the figure lit the way you want the dots lit, on black. It can be a
  drawing with grey tone painted in, or a render.
- optionally, **line-art frames** framed exactly like it, as RGBA PNGs on transparent: the frame
  the shaded picture shows, and others that differ in small areas (eyes shut, eyes moved). The
  others need no shaded version: their change is taken from their line art and shaded to match.
  The first frame's alpha is the silhouette the background stays out of.

```bash
pip install numpy pillow opencv-python-headless
```

```bash
python tools/phosphor_map.py shaded.png --frame idle=idle.png --frame blink=blink.png --out src/map.ts
```

Options: `--size WxH` (default 340x280, one value per display px), `--idle NAME`, `--eye
x0,x1,y0,y1` (where the shading is matched to the line art; automatic by default), `--export NAME`.
The tone settings (shadow lift, local contrast, adaptive equalisation in the deep shadows) are the
`Tone` dataclass at the top of the script. From Python, `build()` takes arrays and `to_ts()` writes
the module, for a pipeline that frames its own images.

## 2. Draw it

```ts
import { Phosphor } from 'phosphor';
import { LEDMAP } from './map';

const p = new Phosphor({
  map: LEDMAP,
  pivot: [205, 225],   // map px: top of the neck, which the head tilts about
  center: [205, 125],  // map px: middle of the head, where the rays come from
  colors: () => ['#24324d', '#6f93e8', '#eef3ff'], // dim, mid, bright
});
document.body.appendChild(p.canvas);
if (!p.load()) { /* no WebGL: show a fallback */ }

p.show('blink', 0);       // a frame, at once
p.show('idle');           // crossfade back (350 ms)
p.set('thinking');        // idle | listening | thinking | speaking
p.level(-30);             // mic level while listening, dBFS: more particles
p.sleep(true);            // drift off
p.palette();              // re-read colors() after a theme change
```

Options: `width`/`height` (display px; default the map's size), `pixelRatio` (device px per display
px), `facing` (`'left'` by default: particles leave from the right), `emitMaxY`, `look` (dot size,
drift, particles, breathing, sleep, burst; see `Look` and `DEFAULT_LOOK`), `motion` (per mood; see
`DEFAULT_MOTION`), `className`. `look` can also be changed live: `p.look.particles = 30`.

## Notes

- WebGL1 / GLSL ES 1.0, `highp`, no `sin()` with large arguments: Mali GPUs run `mediump` as fp16
  and `sin` inaccurately.
- The page should keep the canvas pixel-exact (`image-rendering: pixelated` if it scales).

## Licence

GPL-3.0-or-later.
