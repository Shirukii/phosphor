// Phosphor: a picture drawn as a grid of LED-like dots, in one WebGL1 fragment shader. Pure code:
// the picture comes from a 4-bit brightness map (written by tools/phosphor_map.py) bundled in the
// JS, not an image file. The map holds every frame (an idle frame, plus patches where the others
// differ); show() picks one, crossfading through the dither (or swapping at once, for blinks).
//
// Each dot compares the map's light with an 8x8 Bayer threshold, so the picture is an ordered 1-bit
// pattern: no outlines, nearly one colour, every dot the same brightness. (A `solid` level can draw
// the brightest values as solid lines; it is off by default.) A soft light field drifts across it,
// so dots flip on and off locally every frame; while thinking, the pattern itself also scrolls a
// few steps a second. Dots take three colours (dim, mid, bright), shading from dim toward mid (and
// near-white, if the ramp is set low enough) as the light rises. At startup the grid resolves from
// a few giant blocks down to single dots.
//
// Out of the box it breathes and blinks, and nothing else is drawn: on each breath in, the head
// tilts back a little about `pivot` (the top of the neck) and the shoulders rise; the far ends
// (hair) follow a moment later. If the map has a `blink` frame, it blinks every few seconds.
// Asleep (sleep()), over a few seconds: the eyes close, the head droops forward, breathing slows
// and deepens, and the light calms down.
//
// Opt-in extras (all off by default):
// - look.burst: behind the figure, a burst of thin dotted rays from `center` turns slowly, in a
//   dimmer version of the dim colour. It stays out of the figure's silhouette (`body` in the map).
// - look.particles: particles drift off the figure (a second, cheap draw of points): off its left
//   or right side and top, or off its whole outline (`particlesFrom`). They leave as larger dots,
//   shrink to single ones and flicker out.
// - moods (set()): idle, listening, thinking, speaking change how it moves (and the extras), never
//   how lit it is; a mic level (level()) adds particles while listening.
//
// Tuned for slow GPUs (a Mali-T720 holds 30 fps at 340x280): highp for coordinates, no sin() with
// large arguments, ~30 fps.

/** The brightness map tools/phosphor_map.py writes. */
import { PRESETS as presets } from './presets';

export interface PhosphorMap {
  w: number;
  h: number;
  idle: string;   // the frame `data` holds
  data: string;   // 4-bit values, base64, two per byte, row by row
  body?: string;  // silhouette, 1 bit per px (MSB first), base64; without it: wherever the map is lit
  frames: Record<string, { x: number; y: number; w: number; h: number; data: string }>; // patches over idle
}

export interface Look {
  cell: number;   // dot size, device px
  gap: number;    // gap, fraction of the dot (0 for a solid grid)
  gain: number;   // brightness scale before the threshold
  curve: number;  // gamma on the light (below 1 lifts the mid-tones)
  solid: number;  // brightness from which a dot is always lit (lines); below it, dithered
  fill: number;   // density of the dithered fills just below `solid` (0..1)
  full: number;   // map light (0..1) from which a dot is always lit, whatever the drift: 1 makes full
                  // light solid (as in the PNGs) instead of a flickering grid of pinholes; above 1: off
  rampA: number;  // brightness where the dim colour has turned into the mid colour
  rampB: number;  // brightness where the mid colour has turned into the bright one
  drift: number;  // strength of the drifting light field (0 = still)
  driftSize: number; // its feature size, display px
  particles: number; // particles drifting off the figure, per second (0 = none)
  particleSize: number; // their size when they leave, display px (they shrink to 1 px, then go out)
  breathTilt: number; // how far the head tilts back on a breath in, degrees (0 = still)
  breathRise: number; // how far the shoulders rise, display px (the head half that)
  breathS: number; // seconds per breath at rest
  sleepDroop: number; // asleep: how far the head droops forward, degrees
  sleepBreath: number; // asleep: breaths this much slower (and this much deeper), as a fraction
  sleepCalm: number; // asleep: the light drift, particles and burst drop by this fraction
  burst: number; // strength of the ray burst behind the figure (0 = none)
  blinkEvery: number; // mean seconds between blinks (0 = never); needs a `blink` frame in the map
  blinkMs: number; // how long the eyes stay shut
}

export const DEFAULT_LOOK: Look = { cell: 1, gap: 0, gain: 1, curve: 1, solid: 1.01, fill: 1, full: 1.01, rampA: 2, rampB: 3, drift: 0.2, driftSize: 28,
  particles: 0, particleSize: 2, breathTilt: 1.5, breathRise: 2.5, breathS: 4.8,
  sleepDroop: 3, sleepBreath: 0.45, sleepCalm: 0.8, burst: 0, blinkEvery: 4.5, blinkMs: 140 };

/** Dot colour presets (dim, mid, bright), shared with tools/phosphor_map.py's PNG export. */
export const PRESETS: Record<Preset, [string, string, string]> = presets;
export type Preset = keyof typeof presets;

export type ParticlesFrom = 'left' | 'right' | 'all';

export interface Motion { scroll: number; drift: number; reach: number; spin: number; emit: number; wind: number; breath: number }
export type Mode = 'idle' | 'listening' | 'thinking' | 'speaking';
export const DEFAULT_MOTION: Record<Mode, Motion> = {
  // scroll: pattern steps per second (0 = still; a moving pattern reads as a low frame rate);
  // drift: light field speed; reach: how far the ray burst reaches (1 = fading out just at the
  // edge, in every direction); spin: how fast it turns (rays per second); emit, wind: particle
  // rate and speed; breath: breathing rate; all but scroll, reach and spin as multiples of the look's
  idle:      { scroll: 0, reach: 1.0,  spin: 0.04, drift: 1,   emit: 1,   wind: 1,   breath: 1 },
  listening: { scroll: 0, reach: 1.25, spin: 0.08, drift: 1.8, emit: 1.5, wind: 1.2, breath: 1.1 },
  thinking:  { scroll: 6, reach: 1.15, spin: 0.25, drift: 2.5, emit: 2.5, wind: 1.8, breath: 1.2 },
  speaking:  { scroll: 0, reach: 1.4,  spin: 0.12, drift: 1.5, emit: 1.8, wind: 1.3, breath: 1.25 },
};

export interface PhosphorOptions {
  map: PhosphorMap;
  /** Display size, CSS px (default: the map's size). The canvas backing store is this × pixelRatio(). */
  width?: number;
  height?: number;
  /** Device px per display px, read when the canvas is sized and every frame (default: devicePixelRatio, 1..2). */
  pixelRatio?: () => number;
  /** Dot colours: a preset's name (default 'blue'), or dim, mid and bright as CSS colours (any the
   *  browser parses), read again by palette(). */
  colors?: Preset | (() => [string, string, string]);
  /** The map frame that shows the eyes shut, for blinking and sleep (default 'blink'). */
  blinkFrame?: string;
  /** Play the startup reveal (the grid resolving from giant blocks) on load (default true). */
  reveal?: boolean;
  /** Map px: the top of the neck, which the head tilts about (default: middle, 80% down). */
  pivot?: [number, number];
  /** Map px: the middle of the head, the ray burst's centre (default: middle, 45% down). */
  center?: [number, number];
  /** Dot size in display (CSS) px, at the same picture size: 2 draws 2x2 px dots, half as many each
   *  way, each lit from the map at its centre (for 2: the average of the 4 map px it covers); a
   *  coarser, chunkier LED grid. Overrides look.cell (device px). Default: look.cell. */
  dot?: number;
  /** Where particles leave from (when look.particles is on): the figure's right side and top
   *  ('right', the default: the back of a figure facing left), its left side and top, or its whole
   *  outline ('all'), drifting outward. */
  particlesFrom?: ParticlesFrom;
  /** Map px: particles leave only above this row (default: 78% down). */
  emitMaxY?: number;
  look?: Partial<Look>;
  motion?: Partial<Record<Mode, Partial<Motion>>>;
  /** Class name for the canvas (default 'phosphor'). */
  className?: string;
}

const MAX_PARTS = 256;
const FADE_MS = 350; // a change of frame crossfades over this by default
const LAG_S = 0.35;  // how far behind the far ends (hair) follow, seconds
const PART = 9;      // floats per particle: x, y, vx, vy, age, life, seed, and the wind's direction x, y
const REVEAL_FROM = 64, REVEAL_S = 3.2; // startup: dot size shrinks from this (device px), over this long
const RAYS = 48;     // rays in the burst

const VS = `
attribute vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FS = `
precision highp float; // Mali runs mediump as fp16: too coarse for the coordinates
uniform sampler2D u_map, u_prev; // the frame drawn, and the one it is fading from
uniform float u_mix;            // how far that fade is, 0..1 (1: u_map only)
uniform vec2 u_res, u_size;
uniform float u_pr, u_cell, u_gap, u_time, u_gain, u_curve, u_phase, u_drift, u_dsize, u_dt, u_solid, u_fill, u_full;
uniform vec2 u_breath;         // the breath, 0..1: now, and LAG_S ago (for the far ends)
uniform vec3 u_pose;           // head tilt at a full breath (radians), shoulder rise (px), head droop (radians)
uniform vec2 u_neck;           // the point the head tilts about, px
uniform vec2 u_ramp;          // brightness where the mid, then the bright colour, are reached
uniform vec3 u_pal[3];        // dim, mid, bright
uniform vec4 u_burst;         // ray burst: centre (px), reach (fraction of the way to the edge), turn (in rays)
uniform float u_bstr;         // its strength, 0..1

// 8x8 Bayer threshold from integer coordinates, 0..63/64
float bayer2(vec2 a) { a = floor(a); return fract(a.x / 2.0 + a.y * a.y * 0.75); }
float bayer4(vec2 a) { return bayer2(0.5 * a) * 0.25 + bayer2(a); }
float bayer8(vec2 a) { return bayer4(0.5 * a) * 0.25 + bayer2(a); }

// smooth value noise for the drifting light (sine-free hash: sin() is inaccurate on Mali)
float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

void main() {
  vec2 px = vec2(gl_FragCoord.x, u_res.y - gl_FragCoord.y);   // device px, y down
  vec2 c = floor(px / u_cell);
  vec2 f = fract(px / u_cell);
  if (f.x > 1.0 - u_gap || f.y > 1.0 - u_gap) { gl_FragColor = vec4(0.0); return; }
  vec2 sp = (c + 0.5) * u_cell / u_pr;                          // display px
  vec2 sp0 = sp;                                                // (before breathing: the background stays put)
  // breathing (inverse: where this dot's light comes from). The head tilts back about the neck, the
  // far ends a moment behind; below the neck the tilt gives way to the shoulders rising. Fractions
  // of a px, read through the map's linear filtering: the dots shift softly.
  vec2 d = sp - u_neck;
  float br = mix(u_breath.x, u_breath.y, smoothstep(60.0, 170.0, length(d)));
  float a = u_pose.x * br - u_pose.z;            // small: sin a ~ a, cos a ~ 1 - a^2/2
  vec2 head = u_neck + vec2(d.x * (1.0 - a * a * 0.5) - d.y * a, d.x * a + d.y * (1.0 - a * a * 0.5));
  float body = smoothstep(u_neck.y - 10.0, u_neck.y + 30.0, sp.y);
  sp = mix(head, sp, body);
  sp.y += u_pose.y * br * (0.5 + 0.5 * body);
  float b = 0.0, inside = 0.0;  // the light, and whether this is the figure (its silhouette)
  if (sp.x >= 0.0 && sp.x <= u_size.x) {
    vec2 t = texture2D(u_map, sp / u_size).ra;
    b = t.x; inside = t.y;
    if (u_mix < 1.0) b = mix(texture2D(u_prev, sp / u_size).r, b, u_mix); // crossfading frames
  }
  float light = vnoise(sp / u_dsize + vec2(u_dt * 0.30, -u_dt * 0.21)) * 0.6
              + vnoise(sp / (u_dsize * 0.45) + vec2(-u_dt * 0.45, u_dt * 0.33)) * 0.4;
  float v = pow(b, u_curve) * u_gain * (1.0 + u_breath.x * 0.03 + u_drift * (light - 0.5) * 2.0);

  float thr = bayer8(mod(c + u_phase, 8.0)) + 0.5 / 64.0;
  // lines are solid; fills below u_solid are dithered, sparser the fainter they are
  bool lit = b >= 0.02 && (b >= u_full - 0.001 || v >= u_solid || v / u_solid * u_fill > thr);
  if (!lit) {
    if (u_bstr <= 0.0) { gl_FragColor = vec4(0.0); return; } // no burst: nothing behind
    // behind: a burst of thin dotted rays, in a dimmer dim colour, starting a little out from its
    // centre. Each ray fades by how far along it is to the edge in its own direction, so the burst
    // fills the panel evenly although its centre is off centre.
    vec2 q = sp0 - u_burst.xy;
    float r = length(q);
    vec2 dir = q / max(r, 0.001);
    vec2 dsafe = vec2(dir.x >= 0.0 ? max(dir.x, 0.0001) : min(dir.x, -0.0001), dir.y >= 0.0 ? max(dir.y, 0.0001) : min(dir.y, -0.0001));
    vec2 edge = (step(0.0, dir) * u_size - u_burst.xy) / dsafe;  // distance to the edge along x, along y
    float s = r / min(edge.x, edge.y);                       // 0 at the centre, 1 at the edge
    float ray = step(0.42, abs(fract(atan(q.y, q.x) / 6.2831853 * ${RAYS.toFixed(1)} + u_burst.w) - 0.5));
    float bg = ray * u_bstr * clamp((r - 40.0) / 60.0, 0.0, 1.0) * (1.0 - smoothstep(u_burst.z - 0.3, u_burst.z, s));
    gl_FragColor = inside < 0.5 && bg * 0.55 > thr ? vec4(u_pal[0] * 0.65, 1.0) : vec4(0.0);
    return;
  }

  vec3 col = v < u_ramp.x ? mix(u_pal[0], u_pal[1], smoothstep(0.2, u_ramp.x, v))
                          : mix(u_pal[1], u_pal[2], smoothstep(u_ramp.x, u_ramp.y, v));
  gl_FragColor = vec4(col, 1.0);
}`;

// Particles: square points on the display px grid, one flat colour.
const PVS = `
attribute vec3 a_p;           // top-left x, y and size, display px
uniform vec2 u_size;
uniform float u_pr;
void main() {
  vec2 c = (a_p.xy + a_p.z * 0.5) / u_size * 2.0 - 1.0;
  gl_Position = vec4(c.x, -c.y, 0.0, 1.0);
  gl_PointSize = a_p.z * u_pr;
}`;
const PFS = `
precision mediump float;
uniform vec3 u_col;
void main() { gl_FragColor = vec4(u_col, 1.0); }`;

/** A CSS colour as 0..1 RGB, through a 1-px canvas (handles oklch and friends). */
function cssRgb(css: string): [number, number, number] {
  const c = document.createElement('canvas');
  c.width = c.height = 1;
  const g = c.getContext('2d')!;
  g.fillStyle = css;
  g.fillRect(0, 0, 1, 1);
  const d = g.getImageData(0, 0, 1, 1).data;
  return [d[0] / 255, d[1] / 255, d[2] / 255];
}

/** 4-bit values (base64, two per byte) as 0..255 bytes. */
function unpack(data: string, n: number): Uint8Array {
  const raw = atob(data);
  const out = new Uint8Array(n + 1);
  for (let i = 0; i < raw.length; i++) {
    const v = raw.charCodeAt(i);
    out[i * 2] = (v >> 4) * 17;
    out[i * 2 + 1] = (v & 15) * 17;
  }
  return out.subarray(0, n);
}

/** Every frame's full map: idle, and each other frame as idle with its patch applied. */
export function decodeMaps(M: PhosphorMap): Map<string, Uint8Array> {
  const W = M.w, base = unpack(M.data, W * M.h);
  const maps = new Map([[M.idle, base]]);
  for (const [name, p] of Object.entries(M.frames)) {
    const m = base.slice(), patch = unpack(p.data, p.w * p.h);
    for (let y = 0; y < p.h; y++) m.set(patch.subarray(y * p.w, (y + 1) * p.w), (p.y + y) * W + p.x);
    maps.set(name, m);
  }
  return maps;
}

/** A frame as texture data: its light (luminance) and the silhouette (alpha, 0 or 255) per px. */
function withBody(map: Uint8Array, body: string | undefined): Uint8Array {
  const raw = body ? atob(body) : '', out = new Uint8Array(map.length * 2);
  for (let i = 0; i < map.length; i++) {
    out[i * 2] = map[i];
    out[i * 2 + 1] = (body ? (raw.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1 : map[i]) ? 255 : 0;
  }
  return out;
}

/** Where particles leave, as x, y (map px), then the way they head (x, y) and a kind: 0 a side,
 *  1 the top, 2 the outline. All above maxY.
 *  'left' / 'right': the outermost lit px per row on that side, and the top (topmost per column),
 *  from the crown toward that side only, so particles drifting that way and up never cross the head.
 *  'all': every px of the silhouette's outline, heading straight out from it. */
function emitters(map: Uint8Array, body: Uint8Array, w: number, maxY: number, from: ParticlesFrom): number[] {
  const out: number[] = [];
  if (from === 'all') {
    const h = Math.min(maxY, body.length / w), inside = (x: number, y: number): boolean =>
      x < 0 || x >= w || y >= h ? true : y < 0 ? false : body[y * w + x] > 0; // past the sides and the bottom: still figure
    const R = 3;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!inside(x, y) || (inside(x - 1, y) && inside(x + 1, y) && inside(x, y - 1) && inside(x, y + 1))) continue;
      let nx = 0, ny = 0; // outward: toward the empty px around it
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (!inside(x + dx, y + dy)) { nx += dx; ny += dy; }
      const n = Math.hypot(nx, ny);
      if (n > 0) out.push(x, y, nx / n, ny / n, 2);
    }
    return out;
  }
  const back = from === 'right' ? 1 : -1;
  const xs = Array.from({ length: w }, (_, i) => back > 0 ? w - 1 - i : i); // outermost first
  for (let y = 0; y < maxY; y++) for (const x of xs) if (map[y * w + x]) { out.push(x, y, back, 0, 0); break; }
  const top: number[] = [];
  for (let x = 0; x < w; x++) {
    let t = -1;
    for (let y = 0; y < maxY; y++) if (map[y * w + x]) { t = y; break; }
    top.push(t);
  }
  let crown = -1;
  top.forEach((t, x) => { if (t >= 0 && (crown < 0 || t < top[crown])) crown = x; });
  if (crown < 0) return out;
  for (let x = crown; x >= 0 && x < w; x += back) if (top[x] >= 0) out.push(x, top[x], back, 0, 1);
  return out;
}

export class Phosphor {
  readonly canvas: HTMLCanvasElement;
  look: Look;
  motion: Record<Mode, Motion>;
  fps = 0;
  private M: PhosphorMap;
  private w: number;
  private h: number;
  private ratio: () => number;
  private colors: () => [string, string, string];
  private blinkFrame: string;
  private blinkAt = performance.now() + REVEAL_S * 1000; // the next blink (ms)
  private blinkEnd = 0;  // while blinking: when the eyes open again (ms)
  private pivot: [number, number];
  private center: [number, number];
  private from: ParticlesFrom;
  private dotPx: number | null;
  private emitMaxY: number;
  private gl: WebGLRenderingContext | null = null;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private mode: Mode = 'idle';
  private cur: Motion;
  private levelNow = 0;
  private levelTarget = 0;
  private sleepNow = 0;    // 0 awake .. 1 asleep, eased
  private sleepTarget = 0;
  private raf = 0;
  private last = 0;
  private t0 = performance.now();
  private pal = new Float32Array(9);
  private phase = 0;
  private driftT = 0;
  private breathT = 0;   // 0..1 through the current breath
  private burstTurn = 0; // the ray burst's turn, in rays
  private prog: WebGLProgram | null = null;
  private quad: WebGLBuffer | null = null;
  private pprog: WebGLProgram | null = null;
  private pbuf: WebGLBuffer | null = null;
  private pu: Record<string, WebGLUniformLocation | null> = {};
  private parts = new Float32Array(MAX_PARTS * PART);
  private verts = new Float32Array(MAX_PARTS * 3);
  private nParts = 0;
  private nVerts = 0;
  private emitAcc = 0;
  private emit: number[] = [];
  private map: Uint8Array | null = null;
  private maps = new Map<string, Uint8Array>();
  private shown: string; // the frame drawn
  private texs: WebGLTexture[] = [];
  private front = 0;            // the texture unit holding `shown`; the other holds the frame it fades from
  private fadeT0 = -Infinity;   // when the current crossfade started (ms)
  private fadeMs = FADE_MS;     // and how long it takes
  private frames = 0;
  private fpsT = 0;
  private onVisibility = (): void => { if (document.hidden) this.stop(); else this.start(); };

  constructor(o: PhosphorOptions) {
    const M = this.M = o.map;
    this.w = o.width ?? M.w;
    this.h = o.height ?? M.h;
    this.ratio = o.pixelRatio ?? ((): number => Math.min(2, Math.max(1, window.devicePixelRatio || 1)));
    const c = o.colors ?? 'blue';
    this.colors = typeof c === 'function' ? c : (): [string, string, string] => PRESETS[c] ?? PRESETS.blue;
    this.blinkFrame = o.blinkFrame ?? 'blink';
    if (o.reveal === false) { this.t0 -= REVEAL_S * 1000; this.blinkAt = performance.now() + 1500; }
    this.pivot = o.pivot ?? [M.w / 2, M.h * 0.8];
    this.center = o.center ?? [M.w / 2, M.h * 0.45];
    this.from = o.particlesFrom ?? 'right';
    this.dotPx = o.dot ?? null;
    this.emitMaxY = o.emitMaxY ?? Math.round(M.h * 0.78);
    this.look = { ...DEFAULT_LOOK, ...o.look };
    this.motion = { ...DEFAULT_MOTION };
    for (const k of Object.keys(DEFAULT_MOTION) as Mode[]) this.motion[k] = { ...DEFAULT_MOTION[k], ...o.motion?.[k] };
    this.cur = { ...this.motion.idle };
    this.shown = M.idle;
    this.canvas = document.createElement('canvas');
    this.canvas.className = o.className ?? 'phosphor';
    this.size();
    this.canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); this.stop(); this.gl = null; });
    this.canvas.addEventListener('webglcontextrestored', () => { if (this.init()) this.start(); });
    document.addEventListener('visibilitychange', this.onVisibility);
  }

  /** Set up WebGL and start drawing. False if WebGL isn't available (show a fallback instead). */
  load(): boolean {
    if (!this.init()) return false;
    this.start();
    return true;
  }

  /** Stop drawing and let go of the page listeners (the canvas stays where it is). */
  destroy(): void {
    this.stop();
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.gl?.getExtension('WEBGL_lose_context')?.loseContext();
    this.gl = null;
  }

  /** Play the startup reveal again. */
  reveal(): void { this.t0 = performance.now(); }

  private init(): boolean {
    this.size();
    const opts = { alpha: true, premultipliedAlpha: true, antialias: false, depth: false };
    const gl = this.canvas.getContext('webgl', opts) as WebGLRenderingContext | null;
    if (!gl) return false;
    const sh = (type: number, src: string): WebGLShader => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'shader');
      return s;
    };
    // both programs read one attribute: bind it to slot 0 in each, and keep that slot enabled
    const link = (vs: string, fs: string, attr: string): WebGLProgram => {
      const p = gl.createProgram()!;
      gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
      gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
      gl.bindAttribLocation(p, 0, attr);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'link');
      return p;
    };
    const p = this.prog = link(VS, FS, 'a_pos');
    for (const n of ['u_map', 'u_prev', 'u_mix', 'u_res', 'u_size', 'u_pr', 'u_cell', 'u_gap', 'u_time', 'u_gain', 'u_curve', 'u_phase',
      'u_drift', 'u_dsize', 'u_dt', 'u_solid', 'u_fill', 'u_full', 'u_ramp', 'u_pal', 'u_breath', 'u_pose', 'u_neck', 'u_burst', 'u_bstr']) this.u[n] = gl.getUniformLocation(p, n);
    const pp = this.pprog = link(PVS, PFS, 'a_p');
    for (const n of ['u_size', 'u_pr', 'u_col']) this.pu[n] = gl.getUniformLocation(pp, n);
    gl.enableVertexAttribArray(0);
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.pbuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pbuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.verts.byteLength, gl.DYNAMIC_DRAW);
    gl.useProgram(p);
    // two textures (units 0 and 1): the frame drawn, and the one it is fading from
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.texs = [0, 1].map(i => {
      const tex = gl.createTexture()!;
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return tex;
    });
    const plain = decodeMaps(this.M);
    const map = plain.get(this.M.idle)!;
    this.maps = new Map([...plain].map(([n, m]) => [n, withBody(m, this.M.body)]));
    this.map = map;
    this.emit = this.emitters();
    this.gl = gl;
    this.upload(0, this.shown);
    this.upload(1, this.shown);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.palette();
    return true;
  }

  /** Size the canvas: CSS px and its backing store. */
  private size(): void {
    const r = this.ratio();
    this.canvas.style.width = this.w + 'px';
    this.canvas.style.height = this.h + 'px';
    this.canvas.width = Math.round(this.w * r);
    this.canvas.height = Math.round(this.h * r);
    this.gl?.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  /** Dot size, display px (null: look.cell device px). Can be changed live. */
  get dot(): number | null { return this.dotPx; }
  set dot(d: number | null) { this.dotPx = d && d > 0 ? d : null; }

  /** Where particles leave from: 'left', 'right' or 'all' (the whole outline). Can be changed live. */
  get particlesFrom(): ParticlesFrom { return this.from; }
  set particlesFrom(f: ParticlesFrom) { this.from = f; if (this.map) this.emit = this.emitters(); }

  private emitters(): number[] { // from idle: frames differ only in small patches
    const lit = this.maps.get(this.M.idle)!, body = new Uint8Array(lit.length / 2);
    for (let i = 0; i < body.length; i++) body[i] = lit[i * 2 + 1];
    return emitters(this.map!, body, this.M.w, this.emitMaxY, this.from);
  }

  /** The frame names this map holds. */
  get names(): string[] { return [this.M.idle, ...Object.keys(this.M.frames)]; }

  /** Draw from another frame, crossfading over `fadeMs` (0: at once, for blinks). Unknown names show idle. */
  show(name: string, fadeMs = FADE_MS): void {
    const n = this.maps.has(name) ? name : this.M.idle;
    if (n === this.shown) return;
    this.shown = n;
    this.fadeMs = fadeMs;
    if (fadeMs <= 0) {
      this.upload(this.front, n);
      this.fadeT0 = -Infinity;
    } else {
      this.front ^= 1; // the old front keeps the frame it fades from
      this.upload(this.front, n);
      this.fadeT0 = performance.now();
    }
  }

  private upload(unit: number, name: string): void {
    const gl = this.gl, map = this.maps.get(name);
    if (!gl || !map) return;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, this.texs[unit]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE_ALPHA, this.M.w, this.M.h, 0, gl.LUMINANCE_ALPHA, gl.UNSIGNED_BYTE, map);
  }

  /** Read the dot colours again (call when your theme changes). */
  palette(): void {
    this.colors().forEach((c, i) => this.pal.set(cssRgb(c), i * 3));
  }

  /** How it moves: idle, or a part of a conversation. */
  set(mode: Mode): void {
    this.mode = mode in this.motion ? mode : 'idle';
    if (this.mode !== 'listening') this.levelTarget = 0;
  }

  /** Asleep or not: eases into it over a second or two (closing the eyes), and out of it faster. */
  sleep(on: boolean): void {
    this.sleepTarget = on ? 1 : 0;
    if (!this.maps.has(this.blinkFrame)) return;
    if (on && this.shown === this.M.idle) this.show(this.blinkFrame, 600);
    else if (!on && this.shown === this.blinkFrame) { this.blinkEnd = 0; this.show(this.M.idle, 400); }
  }

  /** Mic level while listening, dBFS (-55..-15 maps to 0..1): more particles. */
  level(db: number): void {
    if (this.mode === 'listening') this.levelTarget = Math.max(0, Math.min(1, (db + 55) / 40));
  }

  start(): void { if (!this.raf && this.gl && !document.hidden) this.raf = requestAnimationFrame(t => this.frame(t)); }
  stop(): void { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }

  private frame(now: number): void {
    this.raf = requestAnimationFrame(t => this.frame(t));
    if (now - this.last < 30) return; // ~30 fps: plenty for the drift, and half the CPU of 60
    const dt = Math.min(0.1, (now - (this.last || now)) / 1000);
    this.last = now;
    this.frames++;
    if (now - this.fpsT > 1000) { this.fps = Math.round(this.frames * 1000 / (now - this.fpsT)); this.frames = 0; this.fpsT = now; }
    const target = this.motion[this.mode], k = 1 - Math.exp(-dt * 3);
    for (const key of Object.keys(target) as (keyof Motion)[]) {
      const d = target[key] - this.cur[key];
      this.cur[key] = Math.abs(d) < 0.001 ? target[key] : this.cur[key] + d * k;
    }
    this.levelNow += (this.levelTarget - this.levelNow) * (1 - Math.exp(-dt * 12));
    this.phase = (this.phase + dt * this.cur.scroll) % 8; // integrated, so a speed change never jumps
    this.sleepNow += (this.sleepTarget - this.sleepNow) * (1 - Math.exp(-dt / (this.sleepTarget > this.sleepNow ? 1.2 : 0.6)));
    const z = this.sleepNow, L = this.look;
    this.driftT = (this.driftT + dt * this.cur.drift * (1 - L.sleepCalm * z)) % 1000;
    this.breathT = (this.breathT + dt * this.cur.breath * (1 - L.sleepBreath * z) / L.breathS) % 1;
    this.burstTurn = (this.burstTurn + dt * this.cur.spin * (1 - L.sleepCalm * z)) % 1;
    this.blink(now);
    const t = (now - this.t0) / 1000;
    this.particles(dt, t);
    this.draw(t);
  }

  /** Blink now and then: only from idle and awake, so a frame shown by show() is left alone. */
  private blink(now: number): void {
    const L = this.look, B = this.blinkFrame;
    if (this.blinkEnd) {
      if (now < this.blinkEnd) return;
      this.blinkEnd = 0;
      if (this.shown === B && !this.sleepTarget) this.show(this.M.idle, 60);
    }
    if (L.blinkEvery <= 0 || this.sleepTarget || !this.maps.has(B) || now < this.blinkAt) return;
    this.blinkAt = now + L.blinkMs + L.blinkEvery * 1000 * (0.4 + Math.random() * 1.2);
    if (this.shown !== this.M.idle) return;
    this.show(B, 0);
    this.blinkEnd = now + L.blinkMs;
  }

  /** Spawn, move and retire the particles, then write their points (shrinking, then flickering out). */
  private particles(dt: number, t: number): void {
    const P = this.parts, m = this.cur, L = this.look, E = this.emit, M = this.M;
    const sx = this.w / M.w, sy = this.h / M.h;
    if (t > REVEAL_S && E.length) this.emitAcc += L.particles * m.emit * (1 + this.levelNow * 2) * (1 - L.sleepCalm * this.sleepNow) * dt;
    while (this.emitAcc >= 1) {
      this.emitAcc -= 1;
      if (this.nParts >= MAX_PARTS) continue;
      const e = Math.floor(Math.random() * (E.length / 5)) * 5, o = this.nParts++ * PART;
      const nx = E[e + 2], ny = E[e + 3], kind = E[e + 4], speed = 5 + Math.random() * 10;
      P[o] = E[e] * sx; P[o + 1] = E[e + 1] * sy;
      if (kind === 2) {                                   // the outline: straight out, rising a little
        P[o + 2] = nx * speed; P[o + 3] = ny * speed - 2 + Math.random() * 2;
      } else {                                            // a side, away from the face; the top rises more
        P[o + 2] = nx * speed;
        P[o + 3] = kind === 1 ? -(3 + Math.random() * 5) : -3 + Math.random() * 4;
      }
      P[o + 4] = 0; P[o + 5] = 2.5 + Math.random() * 2.5; P[o + 6] = Math.random() * 6.283;
      P[o + 7] = nx; P[o + 8] = ny;                      // the wind carries them on that way
    }
    let n = 0;
    for (let i = 0; i < this.nParts; i++) {
      const o = i * PART;
      P[o + 4] += dt;
      P[o + 2] += P[o + 7] * 3 * dt;                                        // the wind picks them up
      P[o + 3] += P[o + 8] * 3 * dt;
      P[o] += P[o + 2] * m.wind * dt;
      P[o + 1] += (P[o + 3] + Math.sin(P[o + 4] * 1.7 + P[o + 6]) * 4) * m.wind * dt;
      if (P[o + 4] > P[o + 5] || P[o] > this.w || P[o] < -4 || P[o + 1] < -4 || P[o + 1] > this.h) { // gone: swap in the last one
        P.copyWithin(o, (this.nParts - 1) * PART, this.nParts * PART);
        this.nParts--; i--;
        continue;
      }
      const a = P[o + 4] / P[o + 5];
      if (a > 0.8 && Math.random() < (a - 0.8) * 5) continue;              // flickering out
      const mx = Math.floor(P[o] / sx), my = Math.floor(P[o + 1] / sy);     // never drawn over the figure
      if (this.map && mx >= 0 && mx < M.w && my >= 0 && my < M.h && this.map[my * M.w + mx]) continue;
      const size = a < 0.55 ? L.particleSize : Math.max(1, L.particleSize - 1);
      this.verts[n * 3] = Math.floor(P[o]); this.verts[n * 3 + 1] = Math.floor(P[o + 1]); this.verts[n * 3 + 2] = size;
      n++;
    }
    this.nVerts = n;
  }

  private draw(t: number): void {
    const gl = this.gl;
    if (!gl) return;
    const u = this.u, m = this.cur, L = this.look, M = this.M, pr = this.ratio();
    const base = this.dotPx ? Math.max(1, Math.round(this.dotPx * pr)) : L.cell; // device px per dot
    gl.useProgram(this.prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    // startup reveal: the dot size steps down from giant blocks, easing out
    const e = Math.min(1, t / REVEAL_S), ease = 1 - Math.pow(1 - e, 3);
    const cell = e >= 1 ? base : Math.max(base, Math.round(REVEAL_FROM + (base - REVEAL_FROM) * ease));
    gl.uniform1i(u.u_map, this.front);
    gl.uniform1i(u.u_prev, this.front ^ 1);
    gl.uniform1f(u.u_mix, Math.min(1, (performance.now() - this.fadeT0) / this.fadeMs));
    gl.uniform2f(u.u_res, this.canvas.width, this.canvas.height);
    gl.uniform2f(u.u_size, this.w, this.h);
    gl.uniform1f(u.u_pr, pr);
    gl.uniform1f(u.u_cell, cell);
    gl.uniform1f(u.u_gap, L.gap);
    gl.uniform1f(u.u_time, t);
    gl.uniform1f(u.u_gain, L.gain);
    gl.uniform1f(u.u_curve, L.curve);
    gl.uniform1f(u.u_phase, Math.floor(this.phase));
    gl.uniform1f(u.u_drift, L.drift);
    gl.uniform1f(u.u_solid, L.solid);
    gl.uniform1f(u.u_fill, L.fill);
    gl.uniform1f(u.u_full, L.full);
    gl.uniform1f(u.u_dsize, L.driftSize);
    gl.uniform1f(u.u_dt, this.driftT);
    gl.uniform2f(u.u_ramp, L.rampA, L.rampB);
    gl.uniform3fv(u.u_pal, this.pal);
    // a breath: in over the first 40%, out more slowly over the rest, eased at both ends
    const smooth = (x: number): number => x * x * (3 - 2 * x);
    const breath = (p: number): number => { p = ((p % 1) + 1) % 1; return p < 0.4 ? smooth(p / 0.4) : 1 - smooth((p - 0.4) / 0.6); };
    const lag = LAG_S * m.breath / L.breathS;
    gl.uniform2f(u.u_breath, breath(this.breathT), breath(this.breathT - lag));
    const deep = 1 + L.sleepBreath * this.sleepNow;
    gl.uniform3f(u.u_pose, L.breathTilt * deep * Math.PI / 180, L.breathRise * deep, L.sleepDroop * this.sleepNow * Math.PI / 180);
    const sx = this.w / M.w, sy = this.h / M.h;
    gl.uniform2f(u.u_neck, this.pivot[0] * sx, this.pivot[1] * sy);
    // the burst: shorter and fainter asleep, and it grows in with the startup reveal
    const zz = this.sleepNow, grow = Math.min(1, Math.max(0, (t - REVEAL_S * 0.5) / REVEAL_S));
    gl.uniform4f(u.u_burst, this.center[0] * sx, this.center[1] * sy, m.reach * (1 - 0.4 * zz) * grow, this.burstTurn);
    gl.uniform1f(u.u_bstr, L.burst * (1 - 0.6 * zz) * grow);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (!this.nVerts) return;
    // particles: one flat colour, the mid-tone (the ramp at 0.6)
    const k = Math.min(1, Math.max(0, (0.6 - 0.2) / (L.rampA - 0.2))), s = k * k * (3 - 2 * k), c = this.pal;
    gl.useProgram(this.pprog);
    gl.uniform2f(this.pu.u_size, this.w, this.h);
    gl.uniform1f(this.pu.u_pr, pr);
    gl.uniform3f(this.pu.u_col, c[0] + (c[3] - c[0]) * s, c[1] + (c[4] - c[1]) * s, c[2] + (c[5] - c[2]) * s);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pbuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.verts.subarray(0, this.nVerts * 3));
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, this.nVerts);
  }
}
