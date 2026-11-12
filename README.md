# Phosphor

A picture drawn as a grid of LED-like dots: an ordered (Bayer) dither in one WebGL1 fragment shader,
from a 4-bit brightness map bundled in your JS. No image files at runtime.

Out of the box it is alive, not a still, and draws nothing but the figure:

- the figure breathes: the head tilts back a little about the neck, the shoulders rise, the hair follows a moment later
- it blinks every few seconds, if the map has a `blink` frame
- a soft light field drifts across the dots, so they flip on and off locally
- frames (a blink, a glance) crossfade through the dither, or swap at once
- asleep, the eyes close, the head droops, breathing slows and the light calms
- at startup the grid resolves from giant blocks down to single dots

Opt-in, off unless you turn them on:

- `look.burst`: a slowly turning burst of dotted rays behind the figure, kept out of the silhouette
- `look.particles`: particles drift off the figure, shrink and flicker out; `particlesFrom` picks
  where from: `'right'` (default) or `'left'` (that side and the top), or `'all'` (the whole outline,
  drifting straight out)
- moods (`idle`, `listening`, `thinking`, `speaking`, via `set()`): change how it moves and how
  busy the extras are, never how lit it is

It runs at 30 fps on a Mali-T720 (an Echo Show 5) at 340×280.

## Parts

| path | what |
|---|---|
| `web/`, `tools/phosphor_web.py` | the web interface: picture in, map and PNGs out, live preview |
| `tools/phosphor_prep.py` | cut-out, framing, and painting the eyes shut (for the web interface, or your own scripts) |
| `tools/phosphor_map.py` | shaded image + line-art frames → `map.ts` (the brightness map) and PNGs |
| `tools/phosphor_fx.py` | a picture → a looping GIF / WebP in the *glow*, *streak* or *dust* style |
| `src/phosphor.ts` | the renderer: one class, no dependencies |
| `src/presets.ts` | the colour presets (`blue`, `green`, `amber`, `rose`, `grey`), shared by both |

## The web interface

The easy way: everything below, from a picture, in the browser.

```bash
pip install numpy pillow opencv-python-headless
```

```bash
npm install
```

```bash
npm run web
```

Then open http://localhost:8340, and drop a picture on it or pick one from `in/`.

1. **Cut out:** it guesses at once. Draw a box around the figure to redo it, and brush with
   *keep* / *remove* to fix what it got wrong. A picture with its own transparency is used as it
   is.
2. **Frame:** pick the map's size (340×280 for the Echo Show 5, or any up to 2048×2048), then drag
   and scroll with the *frame* tool to place the figure.
3. **Eyes:** click the middle of each eye. Phosphor paints them shut for the blink (and sleep) frame.
4. **Tone:** shadows, contrast, white point, and so on.
5. The live preview is the renderer itself: palettes, 1px / 2px dots, sleep, solid full light, rays,
   particles (left, right, all), blink rate, breathing. *Set neck* / *Set head* place the tilt
   pivot and the rays' centre.
6. **Export:** pick colours, backgrounds, dot sizes and PNG scale. You get a zip (`map.ts`, and a
   folder of PNGs per state, then per background), also saved to `out/<name>/`.
7. **Glow / Streak / Dust** (the switch above the preview): the same cut-out and frame as a looping clip
   in another style, rendered on the server. Pick its size (dots across), length, scale and, for
   glow and dust, the way the light streams or the dust drifts; then download a GIF (for X: up to 15 MB) or an animated WebP (lossless, a fraction of the size), saved
   to `out/<name>/fx/` too.

### Styled clips

`tools/phosphor_fx.py` draws a picture as a loop of dots on black, in one of three styles:

- **glow:** seven colours, ordered-dithered on a fine grid: sparse blue dots in the shade, green,
  orange and white in zones where the light gathers, contours in salmon, orange, blue and pink, pink
  on the dark side of edges, and a rim of thin parallel lines round the figure. The shape holds
  still; light and sparks run through it in slanted streaks, and bundles of lines grow out of its
  outline and draw back, along one line (`--angle`: 0 across, 90 up and down). 3 s at 24 fps.
- **streak:** one colour (red by default) on a coarse grid, error-diffused afresh each frame so the
  figure boils, short dashes along its shapes (and drawing the texture of its shaded parts, where
  the dither leaves black), clean black round it but for loose specks gathering towards the bottom, and slanted rain
  flashing up in new places every frame. 1 s at 24 fps.
- **dust:** grey specks of many shades, clumped and boiling, densest in the light and gathering on
  the edges, with short flecks along the shapes; from its far side (along `--angle`) the figure
  comes apart in trails of motes that drift off and fade. 2 s at 24 fps.

```bash
python tools/phosphor_fx.py in/picture.png --cutout
```

writes `out/picture/fx/glow.gif`, `glow.webp`, `glow/0000.png`… and the same for `streak` and `dust`. A
picture with transparency is taken as cut out already; `--cutout` cuts any other one out, and
without either the whole picture is used. `--style`, `--long` (dots across the longer side: glow
540, streak 270, dust 540), `--aspect` (`WxH`, default `1x1`), `--scale` (px per dot in the files:
glow 2, streak 4, dust 2), `--frames`, `--fps`, `--angle` (glow, dust), `--colour` (streak, dust),
`--seed` (by default one from the picture, so the same picture always comes out the same). All three
work best on detailed, lit pictures: the edges and the light are what they draw.

The server keeps pictures in memory while it runs, and only listens on localhost. It has no
dependencies past the three above (`--port`, `--host` to change where it listens).

### Hosting it

`--public` makes it fit for the internet: it reads and writes nothing of the host's (no `in/`,
no `out/`), refuses big uploads and decompression bombs, caps map size (1024) and PNG size,
keeps at most 16 pictures for an hour unused, and runs two heavy jobs at a time. It holds no
secrets and needs none.

`deploy/` has what runs it at phosphor.laika.fm: a hardened systemd unit (`phosphor.service`, its
own user, read-only), the Caddy site (`phosphor.caddy`: TLS, security headers, a 24 MB body limit),
and two scripts to run on the server, in a checkout of this repo:

```bash
deploy/setup.sh
```

```bash
deploy/deploy.sh
```

`setup.sh` runs once (it uses sudo): it creates the user, `/opt/phosphor`, the service and the
Caddy site. `deploy.sh` runs after each push: pull, build, copy, restart. The site's DNS has to
point at the server before Caddy can get its certificate.

The library doesn't need any of this: `src/` (the renderer) and `tools/phosphor_map.py` work on
their own, as in Echo Hermes.

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
python tools/phosphor_map.py shaded.png --frame idle=idle.png --frame blink=blink.png --out out/me/map.ts
```

It also writes PNGs of every frame, next to the map: a folder per frame, and in it one per
background (`transparent`, `black`, `white`, `grey`), holding every preset at each dot size
(`out/me/idle/transparent/blue.png`, `out/me/idle/white/rose_2px.png`, `out/me/blink/grey/amber.png`,
...). They show the dots as the renderer draws them at rest, with full light solid (`look.full = 1`;
by default the renderer leaves a sparse grid of pinholes there that the drifting light keeps
filling in). Each folder but `black` also has `black.png`: black dots like ink, where the picture
is dark (its light inverted inside the silhouette), for printing or light pages.

Options: `--size WxH` (default 340x280, one value per display px), `--idle NAME`, `--eye
x0,x1,y0,y1` (where the shading is matched to the line art; automatic by default), `--export NAME`,
`--png DIR` (where the frame folders go; default the folder of `--out`), `--dots 1,2` (dot sizes,
px; default both), `--backgrounds none,black,white,grey` (default all; `none` is transparent),
`--png-scale N` (each PNG px as N×N: bigger, no more detail; `--size` adds detail), `--no-png`.
The tone settings (shadow lift, local contrast, adaptive equalisation in the deep shadows) are the
`Tone` dataclass at the top of the script. From Python, `build()` takes arrays and `to_ts()` writes
the module, for a pipeline that frames its own images.

## 2. Draw it

```ts
import { Phosphor } from 'phosphor';
import { LEDMAP } from './out/me/map';

const p = new Phosphor({
  map: LEDMAP,
  pivot: [205, 225],   // map px: top of the neck, which the head tilts about
  center: [205, 125],  // map px: middle of the head, where the rays (if on) come from
  colors: 'blue',      // a preset, or () => ['#24324d', '#6f93e8', '#eef3ff'] (dim, mid, bright)
  dot: 2,              // optional: 2x2 px dots, same picture size (a coarser LED grid)
});
document.body.appendChild(p.canvas);
if (!p.load()) { /* no WebGL: show a fallback */ }
// that's it: it breathes and blinks

p.sleep(true);            // drift off (eyes shut)
p.show('glance');         // another frame, crossfading (350 ms); show('idle') to go back
p.palette();              // re-read colors() after a theme change

// opt-in extras
p.look.burst = 1;         // rays behind
p.look.particles = 16;    // particles per second
p.particlesFrom = 'all';  // 'left' | 'right' | 'all'
p.dot = 1;                // dot size can change live too
p.look.full = 1;          // full light solid, as in the PNGs (default: off)
p.set('thinking');        // idle | listening | thinking | speaking
p.level(-30);             // mic level while listening, dBFS: more particles
```

Options: `width`/`height` (display px; default the map's size), `pixelRatio` (device px per display
px), `dot` (dot size in display px at the same picture size; overrides `look.cell`),
`particlesFrom`, `emitMaxY`, `look` (dot size,
drift, breathing, blinking, sleep, and the extras; see `Look` and `DEFAULT_LOOK`), `blinkFrame`
(default `'blink'`), `motion` (per mood; see
`DEFAULT_MOTION`), `className`. `look` can also be changed live: `p.look.particles = 30`.

## Notes

- WebGL1 / GLSL ES 1.0, `highp`, no `sin()` with large arguments: Mali GPUs run `mediump` as fp16
  and `sin` inaccurately.
- The page should keep the canvas pixel-exact (`image-rendering: pixelated` if it scales).

## Licence

GPL-3.0-or-later.
