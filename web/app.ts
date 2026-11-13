// Phosphor's web interface: a picture in, a looping clip out (glow, streak, dust), or a live dot map.
// The server (tools/phosphor_web.py) does the image work: cut-out, framing, the clips, the map, PNGs.
// This page holds the choices (box, brush marks, view, shape, the style's settings), sends them, and
// shows the result; the map's preview is drawn here by the renderer itself, held still but for particles.

import { Phosphor, PRESETS, type ParticlesFrom, type PhosphorMap, type Preset } from '../src/phosphor';

type View = [number, number, number];     // left, top, width (picture px); height from the map's aspect
type Stroke = [number, number, number, boolean]; // x, y, radius (picture px), keep
type Tool = 'box' | 'keep' | 'remove' | 'frame';
type Pt = [number, number];

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const narrow = matchMedia('(max-width: 900px)');

async function post<T>(path: string, body: object): Promise<T> {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(await errorOf(r));
  return r.json() as Promise<T>;
}
async function errorOf(r: Response): Promise<string> {
  if (r.status === 413) return 'That picture is too big to upload.';
  try { return ((await r.json()) as { error?: string }).error || r.statusText; } catch { return r.statusText || 'The server is not answering.'; }
}
const loadImg = (src: string): Promise<HTMLImageElement> => new Promise((ok, no) => {
  const i = new Image();
  i.onload = () => ok(i);
  i.onerror = () => no(new Error('Could not show that picture.'));
  i.src = src;
});
const esc = (s: string): string => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);

// ---- status

let busyCount = 0;
function status(text: string, kind: 'idle' | 'ok' | 'bad' = 'ok'): void {
  $('status-text').textContent = text;
  const el = $('status');
  el.classList.toggle('bad', kind === 'bad');
  el.classList.toggle('ok', kind === 'ok');
  const m = $('mstatus');
  m.textContent = text;
  m.classList.toggle('bad', kind === 'bad');
  if (kind === 'bad') toast(text, true);
}
let toastTimer = 0;
/** A short note at the bottom of the page: errors, which the header may not have room for. */
function toast(text: string, bad = false): void {
  document.querySelector('.toast')?.remove();
  const t = document.createElement('div');
  t.className = 'toast' + (bad ? ' bad' : '');
  t.setAttribute('role', 'status');
  t.textContent = text;
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => t.remove(), bad ? 6000 : 2500);
}
async function task<T>(label: string, f: () => Promise<T>): Promise<T | undefined> {
  busyCount++;
  document.body.classList.add('busy');
  status(label, 'idle');
  try { return await f(); } catch (e) { status(e instanceof Error ? e.message : String(e), 'bad'); return undefined; } finally {
    if (--busyCount === 0) document.body.classList.remove('busy');
  }
}

// ---- state

const TONE: [keyof typeof toneDefaults, string, number, number, number][] = [
  ['lift', 'Shadows', 0.2, 1.2, 0.02],      // gamma: lower lifts more
  ['local', 'Local contrast', 0, 3, 0.05],
  ['gamma', 'Gamma', 0.4, 1.6, 0.02],
  ['white', 'White point', 90, 100, 0.1],
  ['amount', 'Shadow detail', 0, 1, 0.05],
  ['floor', 'Black floor', 0, 0.2, 0.005],
];
const toneDefaults = { lift: 0.6, local: 1.5, gamma: 0.8, white: 99.5, amount: 0.6, floor: 0.03 };

const st = {
  id: '', name: '', iw: 0, ih: 0,
  img: null as HTMLImageElement | null,
  mask: null as HTMLImageElement | null,
  rect: null as [number, number, number, number] | null,
  strokes: [] as Stroke[],
  marks: [] as number[],            // where each brush mark starts in strokes, for undo
  skip: false,
  hasAlpha: false,                // the picture came with its own transparency
  view: null as View | null,
  framed: false,                  // the view was moved by hand: a new cut-out keeps it
  size: [512, 512] as Pt,
  tone: { ...toneDefaults },
  tool: 'box' as Tool,
  brush: 20,                      // screen px
  map: null as PhosphorMap | null,
  frames: 1,
};
const look = {
  preset: 'blue' as Preset, dot: 1, full: false, particles: 'off' as 'off' | ParticlesFrom,
};
let server = { public: false, maxSize: 2048, maxOut: 0 };

// ---- small controls

function pressed(el: Element, on: boolean): void { el.setAttribute('aria-pressed', String(on)); }

/** One-of-several buttons inside `id`: clicking one presses it, and calls on(value). */
function seg(id: string, on: (v: string) => void): void {
  const el = $(id);
  el.addEventListener('click', e => {
    const b = (e.target as HTMLElement).closest('button');
    if (!b || !el.contains(b) || b.dataset.v === undefined) return;
    el.querySelectorAll('button[data-v]').forEach(x => pressed(x, x === b));
    on(b.dataset.v);
  });
}
function setSeg(id: string, v: string): void { $(id).querySelectorAll<HTMLElement>('button[data-v]').forEach(x => pressed(x, x.dataset.v === v)); }
function toggle(id: string, on: (v: boolean) => void): void {
  const b = $(id);
  b.addEventListener('click', () => { const v = b.getAttribute('aria-pressed') !== 'true'; pressed(b, v); on(v); });
}
/** A range: keeps its track filled up to the thumb, and its output in step. */
function range(id: string, fmt: (v: number) => string, on: (v: number) => void): HTMLInputElement {
  const el = $<HTMLInputElement>(id), out = $(id + '-out');
  const show = (): void => { paintRange(el); out.textContent = fmt(Number(el.value)); };
  el.addEventListener('input', () => { show(); on(Number(el.value)); });
  show();
  return el;
}
function paintRange(el: HTMLInputElement): void {
  const min = Number(el.min || 0), max = Number(el.max || 100);
  el.style.setProperty('--fill', `${((Number(el.value) - min) / (max - min || 1)) * 100}%`);
}
function setRange(id: string, v: number, text?: string): void {
  const el = $<HTMLInputElement>(id);
  el.value = String(v);
  paintRange(el);
  if (text !== undefined) $(id + '-out').textContent = text;
}
/** Multi-select chips: [value, label, swatch css or '', on]. */
function chips(id: string, items: [string, string, string, boolean][], onChange: () => void): void {
  const el = $(id);
  el.innerHTML = items.map(([v, label, sw, on]) =>
    `<button class="chip" data-v="${esc(v)}" aria-pressed="${on}">${sw ? `<i class="${sw.startsWith('.') ? sw.slice(1) : ''}" style="${sw.startsWith('.') ? '' : `background:${esc(sw)}`}"></i>` : ''}${esc(label)}</button>`).join('');
  el.addEventListener('click', e => {
    const b = (e.target as HTMLElement).closest('button');
    if (!b) return;
    pressed(b, b.getAttribute('aria-pressed') !== 'true');
    onChange();
  });
}
const picked = (id: string): string[] => [...$(id).querySelectorAll<HTMLElement>('[aria-pressed=true]')].map(b => b.dataset.v!);

// ---- the picture editor

const cv = $<HTMLCanvasElement>('src');
const g = cv.getContext('2d')!;
let fit = { s: 1, ox: 0, oy: 0 }; // picture px -> canvas css px
let dim: HTMLCanvasElement | null = null; // the background, darkened (from the mask)

function layout(): void {
  const r = $('editor-box').getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  cv.style.width = r.width + 'px';
  cv.style.height = r.height + 'px';
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (st.img) {
    const padB = narrow.matches ? 120 : 128; // room for the tool palette and its options
    // fit the picture and the view together, so the frame never runs off the pane
    let [x0, y0, x1, y1] = [0, 0, st.iw, st.ih];
    if (st.view) {
      const [vx, vy, vw] = st.view, vh = vw * st.size[1] / st.size[0];
      x0 = Math.min(x0, vx); y0 = Math.min(y0, vy); x1 = Math.max(x1, vx + vw); y1 = Math.max(y1, vy + vh);
    }
    const bw = x1 - x0, bh = y1 - y0, top = 28; // room for the size label
    const s = Math.min((r.width - 40) / bw, (r.height - top - 12 - padB) / bh);
    fit = { s, ox: (r.width - bw * s) / 2 - x0 * s, oy: top + (r.height - top - 12 - padB - bh * s) / 2 - y0 * s };
  }
  makeDim();
  draw();
}

function makeDim(): void {
  dim = null;
  if (!st.img || !st.mask || st.skip) return;
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(st.iw * fit.s));
  c.height = Math.max(1, Math.round(st.ih * fit.s));
  const x = c.getContext('2d')!;
  x.fillStyle = 'rgba(6, 7, 10, 0.8)';
  x.fillRect(0, 0, c.width, c.height);
  x.globalCompositeOperation = 'destination-out';
  x.drawImage(st.mask, 0, 0, c.width, c.height);
  dim = c;
}

const toScreen = (x: number, y: number): Pt => [fit.ox + x * fit.s, fit.oy + y * fit.s];
const toPic = (x: number, y: number): Pt => [(x - fit.ox) / fit.s, (y - fit.oy) / fit.s];
const css = (name: string): string => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function draw(): void {
  const w = cv.width, h = cv.height;
  g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, w, h); g.restore();
  $('drop').hidden = !!st.img;
  document.body.classList.toggle('has-pic', !!st.img);
  $('tools').classList.toggle('off', !st.img);
  if (!st.img) return;
  const { s, ox, oy } = fit, acc = css('--accent');
  g.imageSmoothingQuality = 'high';
  g.drawImage(st.img, ox, oy, st.iw * s, st.ih * s);
  if (dim) g.drawImage(dim, ox, oy, st.iw * s, st.ih * s);
  // the view: dim outside it, a hairline round it, corner ticks
  if (st.view) {
    const [vx, vy, vw] = st.view, vh = vw * st.size[1] / st.size[0];
    const [x0, y0] = toScreen(vx, vy), W = vw * s, H = vh * s;
    g.save();
    g.beginPath(); g.rect(0, 0, w, h); g.rect(x0, y0, W, H);
    g.fillStyle = st.tool === 'frame' ? 'rgba(6, 7, 10, 0.55)' : 'rgba(6, 7, 10, 0.3)';
    g.fill('evenodd');
    g.strokeStyle = st.tool === 'frame' ? acc : 'rgba(255,255,255,0.35)'; g.lineWidth = 1;
    g.strokeRect(Math.round(x0) + 0.5, Math.round(y0) + 0.5, Math.round(W), Math.round(H));
    g.strokeStyle = acc; g.lineWidth = 2;
    const t = Math.min(14, W / 4, H / 4);
    for (const [cx, cy, dx, dy] of [[x0, y0, 1, 1], [x0 + W, y0, -1, 1], [x0, y0 + H, 1, -1], [x0 + W, y0 + H, -1, -1]]) {
      g.beginPath(); g.moveTo(cx + dx * t, cy); g.lineTo(cx, cy); g.lineTo(cx, cy + dy * t); g.stroke();
    }
    g.fillStyle = 'rgba(255,255,255,0.6)'; g.font = `10px ${css('--mono')}`;
    const label = `${st.size[0]} × ${st.size[1]}`;
    g.fillText(label, x0 + 6, y0 > 16 ? y0 - 6 : y0 + 14);
    g.restore();
  }
  // brush marks
  for (const [x, y, r, keep] of st.strokes) {
    const [sx, sy] = toScreen(x, y);
    g.fillStyle = keep ? 'rgba(95, 216, 144, 0.32)' : 'rgba(240, 122, 140, 0.32)';
    g.beginPath(); g.arc(sx, sy, r * s, 0, Math.PI * 2); g.fill();
  }
  // the box
  if (st.rect && !st.skip) {
    const [x0, y0] = toScreen(st.rect[0], st.rect[1]), [x1, y1] = toScreen(st.rect[2], st.rect[3]);
    g.setLineDash([5, 4]); g.strokeStyle = '#f2c76b'; g.lineWidth = 1.25;
    g.strokeRect(x0, y0, x1 - x0, y1 - y0); g.setLineDash([]);
  }
  // brush cursor
  if (hover && !drag?.touch && (st.tool === 'keep' || st.tool === 'remove')) {
    g.strokeStyle = st.tool === 'keep' ? '#5fd890' : '#f07a8c'; g.lineWidth = 1;
    g.beginPath(); g.arc(hover[0], hover[1], st.brush, 0, Math.PI * 2); g.stroke();
  }
}

let hover: Pt | null = null;
let drag: { kind: 'box' | 'paint' | 'view'; start: Pt; last: Pt; view?: View; touch: boolean } | null = null;

const at = (e: PointerEvent | WheelEvent | MouseEvent): Pt => {
  const r = cv.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
};
cv.addEventListener('pointerdown', e => {
  if (!st.img || e.button !== 0) return;
  cv.setPointerCapture(e.pointerId);
  const sp = at(e), p = toPic(...sp), touch = e.pointerType !== 'mouse';
  if (st.tool === 'box') {
    if (st.skip) setSkip(false);
    drag = { kind: 'box', start: p, last: p, touch };
  } else if (st.tool === 'keep' || st.tool === 'remove') {
    if (st.skip) setSkip(false);
    drag = { kind: 'paint', start: p, last: p, touch };
    st.marks.push(st.strokes.length);
    dab(p);
  } else if (st.tool === 'frame' && st.view) {
    drag = { kind: 'view', start: p, last: p, view: [...st.view] as View, touch };
  }
  draw();
});

cv.addEventListener('pointermove', e => {
  hover = at(e);
  if (!drag) { if (st.tool === 'keep' || st.tool === 'remove') draw(); return; }
  const p = toPic(...hover);
  if (drag.kind === 'box') {
    const [a, b] = [drag.start, p];
    st.rect = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
  } else if (drag.kind === 'paint') {
    const step = st.brush / fit.s / 3;
    const d = Math.hypot(p[0] - drag.last[0], p[1] - drag.last[1]);
    for (let k = 1; k * step <= d; k++) {
      const t = k * step / d;
      dab([drag.last[0] + (p[0] - drag.last[0]) * t, drag.last[1] + (p[1] - drag.last[1]) * t]);
    }
    if (d >= step) drag.last = p;
  } else if (drag.kind === 'view' && drag.view && st.view) {
    st.view = [drag.view[0] - (p[0] - drag.start[0]), drag.view[1] - (p[1] - drag.start[1]), drag.view[2]];
    st.framed = true;
  }
  draw();
});

const endDrag = (): void => {
  const d = drag;
  drag = null;
  if (!d) return;
  if (d.kind === 'box') {
    if (st.rect && (st.rect[2] - st.rect[0]) * fit.s > 8 && (st.rect[3] - st.rect[1]) * fit.s > 8) void cutout();
    else { st.rect = null; draw(); }
  } else if (d.kind === 'paint') void cutout();
  else { if (d.kind === 'view') layout(); rebuild(0); }
};
cv.addEventListener('pointerup', endDrag);
cv.addEventListener('pointercancel', endDrag);
cv.addEventListener('pointerleave', () => { hover = null; if (!drag) draw(); });

cv.addEventListener('wheel', e => {
  if (!st.img) return;
  const p = toPic(...at(e));
  const f = Math.pow(1.0015, e.deltaY);
  if (st.tool === 'frame' && st.view) {
    e.preventDefault();
    zoomView(st.view[2] * f, p);
  }
}, { passive: false });

function dab(p: Pt): void { st.strokes.push([p[0], p[1], st.brush / fit.s, st.tool === 'keep']); }

/** Zoom the view to width `w`, keeping `about` (picture px; default its middle) where it is. */
function zoomView(w: number, about?: Pt): void {
  if (!st.view) return;
  const [x, y, w0] = st.view, k = st.size[1] / st.size[0];
  w = Math.max(16, Math.min(w, Math.max(st.iw, st.ih) * 4));
  const p = about ?? [x + w0 / 2, y + w0 * k / 2];
  st.view = [p[0] - (p[0] - x) * w / w0, p[1] - (p[1] - y) * w / w0, w];
  st.framed = true;
  syncZoom();
  draw(); rebuild();
}
/** The zoom slider: log2 of how much narrower than the picture the view is. */
function syncZoom(): void {
  if (!st.view || !st.iw) return;
  const z = Math.log2(st.iw / st.view[2]);
  setRange('zoom', z, `${Math.round(st.iw / st.view[2] * 100)}%`);
}

function setTool(t: Tool): void {
  st.tool = t;
  $('editor-box').dataset.tool = t;
  document.querySelectorAll<HTMLButtonElement>('#tools button').forEach(b => pressed(b, b.dataset.tool === t));
  cv.style.cursor = t === 'frame' ? 'grab' : t === 'box' ? 'crosshair' : 'none';
  draw();
}

function setSkip(on: boolean): void {
  st.skip = on;
  $<HTMLInputElement>('skip').checked = on;
}

// ---- server calls

async function openPicture(body: { name: string; data?: string }, src: string): Promise<void> {
  const r = await task('Opening…', () => post<{ id: string; w: number; h: number; hasAlpha: boolean; name: string }>('/api/image', body));
  if (!r) return;
  const img = await loadImg(src).catch(() => null);
  if (!img) { status('Could not show that picture.', 'bad'); return; }
  Object.assign(st, {
    id: r.id, name: r.name, iw: r.w, ih: r.h, img, mask: null, rect: null, strokes: [], marks: [], hasAlpha: r.hasAlpha, view: null, framed: false,
    map: null, frames: 1,
  });
  $<HTMLInputElement>('name').value = r.name;
  fxs.seq++; fxs.w = 0; $('fx-img').hidden = true; // the last picture's clip
  clearThumbs(); fresh = true;
  setSkip(r.hasAlpha); // a picture with its own transparency is cut out already
  setTool('box');
  $('file-meta').innerHTML = `<strong>${esc(r.name)}</strong><span>${r.w} × ${r.h}</span>`;
  $('src-meta').textContent = `${r.w} × ${r.h}${r.hasAlpha ? ' · transparent' : ''}`;
  layout();
  await cutout();
}

// one cut-out at a time: the server keeps the last one to finish, so an older, slower request must
// not land after a newer one
let cutBusy = false, cutAgain = false;
async function cutout(): Promise<void> {
  if (cutBusy) { cutAgain = true; return; }
  cutBusy = true;
  try { do { cutAgain = false; await cutoutOnce(); } while (cutAgain); } finally { cutBusy = false; }
}
async function cutoutOnce(): Promise<void> {
  if (!st.id) return;
  const r = await task(st.skip ? 'Using the whole picture…' : 'Finding the figure…', () => post<{ mask: string; view: View; empty: boolean }>('/api/cutout', {
    id: st.id, rect: st.rect, strokes: st.strokes, skip: st.skip, size: st.size,
  }));
  if (!r) return;
  st.mask = await loadImg(r.mask);
  if (!st.framed || !st.view) { st.view = r.view; syncZoom(); }
  layout();
  if (r.empty) { status('Nothing found: draw a box around the figure, or brush over it with Keep.', 'bad'); return; }
  await build();
}

let buildSeq = 0, buildTimer = 0;
function rebuild(ms = 300): void {
  clearTimeout(buildTimer);
  buildTimer = window.setTimeout(() => { void build(); }, ms);
}

const params = (): object => ({ id: st.id, view: st.view, size: st.size, tone: st.tone });

async function build(): Promise<void> {
  if (!st.id || !st.view) return;
  const seq = ++buildSeq;
  const r = await task('Building…', () => post<{ map: PhosphorMap; stats: { lit: number } }>('/api/build', params()));
  if (!r || seq !== buildSeq) return;
  st.map = r.map;
  st.frames = 1 + Object.keys(r.map.frames).length;
  showPreview();
  $('live-meta').textContent = `${r.map.w} × ${r.map.h} · ${Math.round(r.stats.lit * 100)}% lit`;
  status('Ready', 'ok');
  $<HTMLButtonElement>('export').disabled = false;
  exportSummary();
  if (fxs.mode !== 'live') renderFx(); else void renderThumbs();
}

// ---- the live preview

let ph: Phosphor | null = null;
let zoom = 1;

function showPreview(): void {
  if (!st.map) return;
  const first = !ph;
  if (ph) { ph.destroy(); ph.canvas.remove(); }
  ph = new Phosphor({
    map: st.map, colors: () => PRESETS[look.preset], pixelRatio: () => 1,
    reveal: first, particlesFrom: look.particles === 'off' ? 'right' : look.particles,
  });
  applyLook();
  $('stage').prepend(ph.canvas);
  if (!ph.load()) { status('No WebGL in this browser: no live preview, but export still works.', 'bad'); return; }
  $('prev-empty').hidden = true;
  setMode(fxs.mode);
}

function applyLook(): void {
  if (!ph) return;
  Object.assign(ph.look, {
    full: look.full ? 1 : 1.01, burst: 0,
    particles: look.particles === 'off' ? 0 : look.particles === 'all' ? 40 : 16,
    blinkEvery: 0, breathTilt: 0, breathRise: 0,   // held still: no blink, no breathing
  });
  ph.dot = look.dot === 2 ? 2 : null;
  if (look.particles !== 'off') ph.particlesFrom = look.particles;
  ph.palette();
}

function fitPreview(): void {
  if (fxs.mode !== 'live') { fitFx(); return; }
  if (!st.map || !ph) return;
  const r = $('preview-box').getBoundingClientRect(), W = st.map.w, H = st.map.h;
  const bar = $('live-bar').getBoundingClientRect().height + 28;
  const room = Math.min((r.width - 32) / W, (r.height - bar - 32) / H);
  zoom = room >= 1 ? Math.floor(room) : Math.max(0.1, room); // whole steps keep the dots crisp
  ph.canvas.style.transform = `scale(${zoom})`;
  const stage = $('stage');
  stage.style.width = W * zoom + 'px';
  stage.style.height = H * zoom + 'px';
  stage.style.marginBottom = bar + 'px';
}

/** The accent follows the palette: the page takes the preview's colour. */
function setAccent(p: Preset): void {
  tint(PRESETS[p][1]);
}
/** The page takes the colour of what it is making: glow's blue, streak's or dust's own, the map's palette. */
function tint(colour?: string): void {
  const m = fxs.mode;
  const c = colour ?? (m === 'live' ? PRESETS[look.preset][1] : m === 'glow' ? '#5b93f0'
    : FX_COLOURS.find(([v]) => v === fxSets[m].colour)?.[1] ?? '#5b93f0');
  document.documentElement.style.setProperty('--accent', c);
  document.querySelectorAll<HTMLInputElement>('input[type=range]').forEach(paintRange);
  draw();
}

// ---- controls

document.querySelectorAll<HTMLButtonElement>('#tools button').forEach(b => b.addEventListener('click', () => setTool(b.dataset.tool as Tool)));

// palettes, as swatches
$('presets').innerHTML = Object.entries(PRESETS).map(([k, c]) =>
  `<button data-v="${k}" aria-pressed="${k === look.preset}" title="${k}" aria-label="${k}"><span class="sw" style="background:${c[1]};color:${c[1]}"></span></button>`).join('');
seg('presets', v => { look.preset = v as Preset; ph?.palette(); setAccent(look.preset); });
seg('dot', v => { look.dot = Number(v); applyLook(); });
seg('parts', v => { look.particles = v as typeof look.particles; applyLook(); });
toggle('full', v => { look.full = v; applyLook(); });

range('brush', v => `${v} px`, v => { st.brush = v; draw(); });
$<HTMLInputElement>('skip').addEventListener('change', e => { st.skip = (e.target as HTMLInputElement).checked; void cutout(); });
$('redo-cut').addEventListener('click', () => { st.rect = null; st.strokes = []; st.marks = []; setSkip(st.hasAlpha); void cutout(); });   // as it opened: no box, no marks
$('clear-strokes').addEventListener('click', () => { if (st.strokes.length) { st.strokes = []; st.marks = []; untouched(); void cutout(); } });
function undoStroke(): void {
  const from = st.marks.pop();
  if (from === undefined) return;
  st.strokes.length = from;
  untouched();
  draw(); void cutout();
}
/** With no box and no marks left, a picture with its own transparency goes back to using it. */
function untouched(): void { if (!st.strokes.length && !st.rect && st.hasAlpha) setSkip(true); }
$('undo-stroke').addEventListener('click', undoStroke);

range('zoom', () => st.view ? `${Math.round(st.iw / st.view[2] * 100)}%` : '–', v => { if (st.iw) zoomView(st.iw / Math.pow(2, v)); });

// size
const sizes = { clip: [512, 512] as Pt, map: [512, 512] as Pt };   // the clip's shape, the map's size
const SHAPES = ['512x512', '512x640', '480x640', '640x360', '360x640'];
function setSize(W: number, H: number): void {
  W = Math.max(8, Math.min(server.maxSize, Math.round(W) || 8));
  H = Math.max(8, Math.min(server.maxSize, Math.round(H) || 8));
  if (fxs.mode !== 'live' && !SHAPES.includes(`${W}x${H}`)) [W, H] = [512, 512];
  if (st.view) { // keep the middle and width; the height follows
    const [x, y, w] = st.view, cy = y + w * st.size[1] / st.size[0] / 2;
    st.view = [x, cy - w * H / W / 2, w];
  }
  st.size = sizes[fxs.mode === 'live' ? 'map' : 'clip'] = [W, H];
  $<HTMLInputElement>('w').value = String(W);
  $<HTMLInputElement>('h').value = String(H);
  document.querySelectorAll<HTMLElement>('#sizes .chip, #shapes .chip').forEach(c => pressed(c, c.dataset.v === `${W}x${H}`));
  remember();
  exportSummary();
  if (st.id && !st.framed) void autoFrame();   // not placed by hand: fit the figure to the new shape
  else { layout(); rebuild(0); }
}
async function autoFrame(): Promise<void> {
  if (!st.id) return;
  const r = await task('Framing…', () => post<{ view: View }>('/api/view', { id: st.id, size: st.size }));
  if (!r) return;
  st.view = r.view; st.framed = false;
  syncZoom(); layout(); rebuild(0);
}
for (const id of ['sizes', 'shapes']) $(id).addEventListener('click', e => {
  const b = (e.target as HTMLElement).closest<HTMLElement>('.chip');
  if (!b?.dataset.v) return;
  const [W, H] = b.dataset.v.split('x').map(Number);
  setSize(W, H);
});
for (const id of ['w', 'h']) $(id).addEventListener('change', () => setSize(Number($<HTMLInputElement>('w').value), Number($<HTMLInputElement>('h').value)));
$('auto-frame').addEventListener('click', () => void autoFrame());

// tone
const toneEl = $('tone');
toneEl.innerHTML = TONE.map(([k, label, min, max, step]) =>
  `<div class="range"><label for="t-${k}">${label}</label><output id="t-${k}-out"></output><input type="range" id="t-${k}" min="${min}" max="${max}" step="${step}" value="${toneDefaults[k]}"></div>`).join('');
const fmtTone = (v: number): string => String(Math.round(v * 1000) / 1000);
for (const [k] of TONE) range(`t-${k}`, fmtTone, v => { st.tone[k] = v; rebuild(); });
$('reset-tone').addEventListener('click', () => {
  st.tone = { ...toneDefaults };
  for (const [k] of TONE) setRange(`t-${k}`, toneDefaults[k], fmtTone(toneDefaults[k]));
  rebuild(0);
});

// export
chips('ex-colours', [...Object.entries(PRESETS).map(([k, c]) => [k, k, c[1], true] as [string, string, string, boolean]), ['black', 'black ink', '#000', true]], exportSummary);
chips('ex-bgs', [['none', 'transparent', '.transparent', true], ['black', 'black', '#000', true], ['white', 'white', '#fff', true], ['grey', 'grey', '#808080', true]], exportSummary);
chips('ex-dots', [['1', '1 px', '', true], ['2', '2 px', '', true]], exportSummary);
let scale = 1;
seg('scale', v => { scale = Number(v); exportSummary(); });

/** How many files an export makes, and how big each PNG is. */
function exportSummary(): void {
  const cs = picked('ex-colours'), bgs = picked('ex-bgs'), ds = picked('ex-dots');
  const per = cs.reduce((n, c) => n + bgs.filter(b => !(c === 'black' && b === 'black')).length, 0) * ds.length;
  const n = per * st.frames;
  const [W, H] = st.size, big = server.maxOut > 0 && W * H * scale * scale > server.maxOut;
  $('ex-count').textContent = n ? `${n} PNG${n === 1 ? '' : 's'} + map.ts` : 'map.ts only';
  $('ex-size').textContent = big ? 'too big here' : `${W * scale} × ${H * scale}`;
  $('ex-size').style.color = big ? 'var(--bad)' : '';
  $<HTMLButtonElement>('export').disabled = !st.map || big;
}

$('export').addEventListener('click', async () => {
  if (!st.id) return;
  const name = ($<HTMLInputElement>('name').value.trim() || st.name).replace(/[^A-Za-z0-9_.-]+/g, '-');
  const body = {
    ...params(), name, colours: picked('ex-colours'), backgrounds: picked('ex-bgs'),
    dots: picked('ex-dots').map(Number), scale, save: !server.public && $<HTMLInputElement>('save').checked,
  };
  await task('Exporting…', async () => {
    const r = await fetch('/api/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await errorOf(r));
    const blob = await r.blob(), a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.zip`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    const saved = r.headers.get('X-Saved');
    status(`Downloaded ${name}.zip${saved ? ` · saved to ${saved}/` : ''}`, 'ok');
  });
});

// ---- styled clips: glow, streak and dust, rendered on the server as a looping WebP (preview) or GIF

type Mode = 'live' | 'glow' | 'streak' | 'dust';
type Style = Exclude<Mode, 'live'>;
const FX_COLOURS: [string, string][] = [['red', '#ee1111'], ['white', '#f2f2f2'], ['grey', '#d8d8d8'], ['green', '#4fe08a'], ['blue', '#6f93e8'], ['amber', '#ffb347'], ['rose', '#f07aa6']];
// each style's choices (value, label), and what it starts at: glow is a slow 3 s loop at 24 fps on a
// fine grid, streak a fast 1 s boil on a coarse one, dust a 2 s drift on a fine one
// hand controls: sliders run -100..100 around the style's own settings (0), a factor of 2^(v/63) (about ⅓× to 3×)
type FxSet = { long: number, frames: number, scale: number, angle: number | null, colour: string, streaks?: boolean,
  exposure: number, highlights: number, warmth: number, focus: [number, number] | null };
const TUNE0 = { exposure: 0, highlights: 0, warmth: 0, focus: null };
const FX: Record<Style, { long: number[], len: [number, string][], scale: number[], fps: number, d: FxSet }> = {
  glow: { long: [360, 540, 720], len: [[48, '2 s'], [72, '3 s'], [96, '4 s']], scale: [1, 2], fps: 24,
    d: { long: 540, frames: 72, scale: 2, angle: 0, colour: '', streaks: true, ...TUNE0 } },
  streak: { long: [180, 270, 360], len: [[12, '½ s'], [24, '1 s'], [48, '2 s']], scale: [2, 4], fps: 24,
    d: { long: 270, frames: 24, scale: 4, angle: 0, colour: 'red', ...TUNE0 } },
  dust: { long: [360, 540, 720], len: [[24, '1 s'], [48, '2 s'], [72, '3 s']], scale: [1, 2], fps: 24,
    d: { long: 540, frames: 48, scale: 2, angle: null, colour: 'white', ...TUNE0 } },
};
// which way the dust drifts: a dial (degrees, 0 right, 90 down); auto drifts across, away from the lit side
const fxSets: Record<Style, FxSet> = { glow: { ...FX.glow.d }, streak: { ...FX.streak.d }, dust: { ...FX.dust.d } };
const fxs = { mode: 'glow' as Mode, url: '', seq: 0, w: 0, h: 0, bytes: 0 };

// what was chosen last time: the style, each style's settings, the shape
const KEEP = 'phosphor.ui.1';
function remember(): void {
  try { localStorage.setItem(KEEP, JSON.stringify({ mode: fxs.mode, sets: fxSets, size: sizes.clip, mapSize: sizes.map })); } catch { /* private window: nothing kept */ }
}
function recall(): { mode?: Mode, size?: Pt, mapSize?: Pt } {
  try {
    const k = JSON.parse(localStorage.getItem(KEEP) || '{}') as { mode?: Mode, sets?: Partial<Record<Style, Partial<FxSet>>>, size?: Pt, mapSize?: Pt };
    for (const s of ['glow', 'streak', 'dust'] as Style[]) {
      const got = k.sets?.[s];
      if (!got) continue;
      const o = FX[s], d = fxSets[s];
      if (o.long.includes(got.long!)) d.long = got.long!;
      if (o.len.some(([v]) => v === got.frames)) d.frames = got.frames!;
      if (o.scale.includes(got.scale!)) d.scale = got.scale!;
      if (typeof got.colour === 'string' && (s === 'glow' || FX_COLOURS.some(([v]) => v === got.colour))) d.colour = got.colour;
      if (got.angle === null || typeof got.angle === 'number') d.angle = got.angle;
      if (typeof got.streaks === 'boolean') d.streaks = got.streaks;
      for (const t of ['exposure', 'highlights', 'warmth'] as const) if (typeof got[t] === 'number' && Math.abs(got[t]!) <= 100) d[t] = got[t]!;
    }
    const mode = ['glow', 'streak', 'dust', 'live'].includes(k.mode!) ? k.mode : undefined;
    const pair = (v?: Pt): Pt | undefined => Array.isArray(v) && v.length === 2 && v.every(n => Number.isFinite(n)) ? v : undefined;
    return { mode, size: pair(k.size), mapSize: pair(k.mapSize) };
  } catch { return {}; }
}
const fxSet = (): FxSet => fxSets[fxs.mode === 'live' ? 'glow' : fxs.mode];
let fxTimer = 0;

function fxButtons(): void {
  if (fxs.mode === 'live') return;
  const o = FX[fxs.mode], c = fxSet();
  const b = (v: number, label: string, on: boolean, title = '') =>
    `<button data-v="${v}" aria-pressed="${on}"${title ? ` title="${title}"` : ''}>${label}</button>`;
  $('fx-long').innerHTML = o.long.map(v => b(v, String(v), v === c.long)).join('');
  $('fx-len').innerHTML = o.len.map(([v, l]) => b(v, l, v === c.frames)).join('');
  $('fx-scale').innerHTML = o.scale.map(v => b(v, v + '×', v === c.scale)).join('');
  dial(c.angle);
  for (const k of TUNE_KEYS) { const el = $<HTMLInputElement>('tu-' + k); el.value = String(c[k]); paintRange(el); $('tu-' + k + '-out').textContent = tuneText(c[k]); }
  placeFocus();
  setSeg('fx-colours', c.colour);
  $<HTMLInputElement>('fx-streaks').checked = c.streaks !== false;
  fileSize();
}
/** What the downloaded clip will measure. */
function fileSize(): void {
  const c = fxSet(), [W, H] = st.size, k = c.long / Math.max(W, H);
  const rnd = (x: number): number => { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f; };
  const w = Math.max(8, rnd(W * k)) * c.scale, h = Math.max(8, rnd(H * k)) * c.scale;
  $('fx-size').textContent = `${w} × ${h} px · ${+(c.frames / 24).toFixed(1)} s`;
}

$('fx-colours').innerHTML = FX_COLOURS.map(([v, c]) =>
  `<button data-v="${v}" aria-pressed="${v === 'red'}" title="${v}"><span class="sw" style="background:${c}"></span></button>`).join('');
seg('fx-colours', v => { fxSet().colour = v; tint(); renderFx(0); });
seg('fx-long', v => { fxSet().long = Number(v); renderFx(0); });
seg('fx-len', v => { fxSet().frames = Number(v); renderFx(0); });
seg('fx-scale', v => { fxSet().scale = Number(v); fileSize(); remember(); });
const TUNE_KEYS = ['exposure', 'highlights', 'warmth'] as const;
const tuneF = (v: number): number => +Math.pow(2, v / 63).toFixed(3);
const tuneText = (v: number): string => v === 0 ? '1×' : tuneF(v).toFixed(2) + '×';
for (const k of TUNE_KEYS) range('tu-' + k, tuneText, v => { fxSet()[k] = v; renderFx(); });
$('tu-reset').addEventListener('click', () => {
  if (fxs.mode === 'live') return;
  const d = FX[fxs.mode].d;
  Object.assign(fxSet(), TUNE0, { streaks: d.streaks, angle: d.angle }); setPickFocus(false); fxButtons(); renderFx(0); });
$('tu-focus').addEventListener('click', () => setPickFocus(!$('fx-img').classList.contains('picking')));
function setPickFocus(on: boolean): void {
  $('fx-img').classList.toggle('picking', on);
  pressed($('tu-focus'), on || fxSet().focus !== null);
  $('tu-focus').textContent = on ? 'Click the clip…' : fxSet().focus ? 'Move light source' : 'Set light source';
}
$('fx-img').addEventListener('click', e => {
  const img = $<HTMLImageElement>('fx-img');
  if (!img.classList.contains('picking')) return;
  const b = img.getBoundingClientRect();
  fxSet().focus = [Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)), Math.min(1, Math.max(0, (e.clientY - b.top) / b.height))];
  setPickFocus(false); placeFocus(); renderFx(0);
});
/** The focus marker over the clip, where the light gathers (glow only). */
function placeFocus(): void {
  const m = $('fx-focus'), img = $<HTMLImageElement>('fx-img'), f = fxSet().focus;
  if (!f || img.hidden || fxs.mode !== 'glow') { m.hidden = true; return; }
  const b = img.getBoundingClientRect(), p = $('preview-box').getBoundingClientRect();
  m.style.left = (b.left - p.left + f[0] * b.width) + 'px'; m.style.top = (b.top - p.top + f[1] * b.height) + 'px';
  m.hidden = false;
}
seg('fx-angle', () => { fxSet().angle = null; dial(null); renderFx(0); });
// the dial: its hand points the way the dust goes; drag or click round it, or use the arrow keys
function dial(deg: number | null): void {
  const d = $('fx-dial'), r = ((deg ?? 0) * Math.PI) / 180;
  const hand = d.querySelector('line')!, knob = d.querySelector('circle.knob')!;
  hand.setAttribute('x2', String(12 + 8 * Math.cos(r))); hand.setAttribute('y2', String(12 + 8 * Math.sin(r)));
  knob.setAttribute('cx', String(12 + 8 * Math.cos(r))); knob.setAttribute('cy', String(12 + 8 * Math.sin(r)));
  d.classList.toggle('auto', deg === null);
  d.setAttribute('aria-valuenow', String(deg ?? 0));
  d.setAttribute('aria-valuetext', deg === null ? 'auto' : `${Math.round(deg)}°`);
  d.title = deg === null ? 'Drift direction: auto. Drag round the dial to set it' : `Drift direction: ${Math.round(deg)}°`;
  pressed($('fx-angle').querySelector('button[data-v=auto]')!, deg === null);
}
function dialAt(e: PointerEvent): number {
  const b = $('fx-dial').getBoundingClientRect();
  const deg = (Math.atan2(e.clientY - (b.top + b.height / 2), e.clientX - (b.left + b.width / 2)) * 180) / Math.PI;
  return ((Math.round(deg / 5) * 5) % 360 + 360) % 360;   // to the nearest 5°
}
function dialSet(deg: number): void { fxSet().angle = deg; dial(deg); renderFx(0); }
{
  const d = $('fx-dial');
  let drag = false;
  d.addEventListener('pointerdown', e => { drag = true; d.setPointerCapture(e.pointerId); dial(dialAt(e)); e.preventDefault(); });
  d.addEventListener('pointermove', e => { if (drag) dial(dialAt(e)); });   // only the hand moves while dragging
  d.addEventListener('pointerup', e => { if (!drag) return; drag = false; dialSet(dialAt(e)); });
  d.addEventListener('pointercancel', () => { drag = false; dial(fxSet().angle); });
  d.addEventListener('keydown', e => {
    const step = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 15 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -15 : 0;
    if (!step) return;
    e.preventDefault();
    dialSet((((fxSet().angle ?? 0) + step) % 360 + 360) % 360);
  });
}
$<HTMLInputElement>('fx-streaks').addEventListener('change', e => { fxSet().streaks = (e.target as HTMLInputElement).checked; renderFx(0); });
seg('mode', v => setMode(v as Mode));

function setMode(m: Mode): void {
  fxs.mode = m;
  const want = sizes[m === 'live' ? 'map' : 'clip'];
  if (want[0] !== st.size[0] || want[1] !== st.size[1]) setSize(want[0], want[1]);
  setSeg('mode', m);
  document.body.dataset.mode = m;
  const live = m === 'live';
  if (ph) ph.canvas.hidden = !live;
  $('live-bar').classList.toggle('off', !live || !st.map);
  $('fx-colours-row').hidden = live || m === 'glow';
  $('fx-angle').hidden = m !== 'dust';   // only dust has a direction
  $('fx-streaks-row').hidden = m !== 'glow';
  $('tu-warmth-l').hidden = m !== 'glow';
  $('tu-focus').hidden = m !== 'glow';
  setPickFocus(false);
  fxButtons();
  $('fx-wait').hidden = true;
  $('fx-img').hidden = true;   // the last clip was another style, or another picture
  $<HTMLButtonElement>('fx-gif').disabled = $<HTMLButtonElement>('fx-webp').disabled = true;
  tint();
  remember();
  if (live) {
    fxs.seq++; // a clip still on its way is no longer wanted
    if (st.map) { $('live-meta').textContent = `${st.map.w} × ${st.map.h} · live`; status('Ready', 'ok'); }
    $('prev-empty').hidden = !!st.map;
    requestAnimationFrame(() => { layout(); fitPreview(); });
  } else {
    $('prev-empty').hidden = !!st.id;
    requestAnimationFrame(layout);
    renderFx(0);
  }
}

function fxBody(format: string, scale: number, save: boolean): object {
  const name = ($<HTMLInputElement>('name').value.trim() || st.name).replace(/[^A-Za-z0-9_.-]+/g, '-');
  return { ...params(), style: fxs.mode, long: fxSet().long, frames: fxSet().frames, angle: fxs.mode === 'dust' ? fxSet().angle ?? undefined : undefined,
    colour: fxSet().colour || undefined,
    streaks: fxs.mode === 'glow' ? fxSet().streaks !== false : undefined, tune: tuneBody(), format, scale, name,
    save: save && !server.public && $<HTMLInputElement>('save').checked };
}
function tuneBody(): object {
  const c = fxSet(), glow = fxs.mode === 'glow';
  return { exposure: tuneF(c.exposure), highlights: tuneF(c.highlights), warmth: glow ? tuneF(c.warmth) : undefined,
    focus: glow && c.focus ? c.focus : undefined };
}
async function fxFetch(format: string, scale: number, save = false): Promise<Response> {
  const r = await fetch('/api/fx', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fxBody(format, scale, save)) });
  if (!r.ok) throw new Error(await errorOf(r));
  return r;
}

function renderFx(ms = 600): void {
  clearTimeout(fxTimer);
  fileSize(); remember();
  if (fxs.mode === 'live' || !st.id || !st.view) return;
  fxTimer = window.setTimeout(() => { void drawFx(); }, ms);
}
async function drawFx(): Promise<void> {
  const seq = ++fxs.seq, mode = fxs.mode;
  const wait = $('fx-wait');
  wait.classList.toggle('over', !$('fx-img').hidden);
  wait.hidden = false;
  const blob = await task(`Rendering ${mode}…`, async () => (await fxFetch('webp', 1)).blob());
  if (seq !== fxs.seq) return;
  wait.hidden = true;
  if (!blob) return;
  const url = URL.createObjectURL(blob), img = $<HTMLImageElement>('fx-img');
  await new Promise<void>(ok => { img.onload = () => ok(); img.onerror = () => ok(); img.src = url; });
  if (fxs.url) URL.revokeObjectURL(fxs.url);
  fxs.url = url;
  fxs.w = img.naturalWidth; fxs.h = img.naturalHeight; fxs.bytes = blob.size;
  img.hidden = fxs.mode === 'live';
  $('prev-empty').hidden = true;
  $<HTMLButtonElement>('fx-gif').disabled = $<HTMLButtonElement>('fx-webp').disabled = false;
  $('live-meta').textContent = `${fxs.w} × ${fxs.h} dots · ${fxSet().frames} frames`;
  status('Ready', 'ok');
  if (fresh && narrow.matches) showView('live');   // on a phone, a new picture's first clip comes to the front
  fresh = false;
  fitFx();
  void renderThumbs();
}

// ---- the style tiles: the open picture in each style, small

let thumbKey = '', thumbSeq = 0, fresh = false;
function clearThumbs(): void {
  thumbKey = ''; thumbSeq++;
  for (const s of ['glow', 'streak', 'dust']) $('th-' + s).hidden = true;
  const c = $<HTMLCanvasElement>('th-live');
  c.getContext('2d')!.clearRect(0, 0, c.width, c.height);
}
async function renderThumbs(): Promise<void> {
  if (!st.id || !st.view) return;
  const key = JSON.stringify([st.id, st.view, st.size, fxSets.streak.colour, fxSets.dust.colour, look.preset]);
  if (key === thumbKey) return;
  thumbKey = key;
  const seq = ++thumbSeq;
  mapThumb();
  for (const s of ['glow', 'streak', 'dust'] as Style[]) {
    try {
      const r = await fetch('/api/fx', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        ...params(), style: s, long: s === 'streak' ? 96 : 128, frames: 12, colour: fxSets[s].colour || undefined, format: 'webp', scale: 1, thumb: true }) });
      if (seq !== thumbSeq) return;
      if (!r.ok) continue;
      const img = $<HTMLImageElement>('th-' + s), old = img.src;
      img.src = URL.createObjectURL(await r.blob());
      img.hidden = false;
      if (old.startsWith('blob:')) URL.revokeObjectURL(old);
    } catch { /* a tile stays empty */ }
  }
}
/** The map's tile: the framed picture as a coarse grid of dots in the palette's colour. */
function mapThumb(): void {
  const c = $<HTMLCanvasElement>('th-live'), g2 = c.getContext('2d')!;
  g2.clearRect(0, 0, c.width, c.height);
  if (!st.img || !st.view) return;
  const cols = c.width / 2, rows = c.height / 2, t = document.createElement('canvas');
  t.width = cols; t.height = rows;
  const [vx, vy, vw] = st.view, vh = vw * st.size[1] / st.size[0], k = Math.max(cols / vw, rows / vh);
  const tg = t.getContext('2d')!;
  tg.drawImage(st.img, (cols - vw * k) / 2 - vx * k, (rows - vh * k) / 2 - vy * k, st.iw * k, st.ih * k);
  if (st.mask && !st.skip) { tg.globalCompositeOperation = 'destination-in'; tg.drawImage(st.mask, (cols - vw * k) / 2 - vx * k, (rows - vh * k) / 2 - vy * k, st.iw * k, st.ih * k); }
  let d: Uint8ClampedArray;
  try { d = tg.getImageData(0, 0, cols, rows).data; } catch { return; }
  g2.fillStyle = PRESETS[look.preset][1];
  const bayer = [0.1, 0.6, 0.85, 0.35];
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const i = (y * cols + x) * 4, l = (0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2]) / 255 * (d[i + 3] / 255);
    if (l * 1.3 > bayer[(y & 1) * 2 + (x & 1)]) g2.fillRect(x * 2, y * 2, 1.4, 1.4);
  }
}
function fitFx(): void {
  if (!fxs.w) return;
  const r = $('preview-box').getBoundingClientRect(), img = $<HTMLImageElement>('fx-img');
  const room = Math.min((r.width - 32) / fxs.w, (r.height - 32) / fxs.h);
  const z = room >= 1 ? Math.floor(room * 2) / 2 : Math.max(0.1, room);   // half steps: a coarse clip still fills the pane
  img.style.width = fxs.w * z + 'px';
  img.style.height = fxs.h * z + 'px';
  placeFocus();
}
async function downloadFx(format: 'gif' | 'webp'): Promise<void> {
  if (!st.id || fxs.mode === 'live') return;
  await task(`Making the ${format.toUpperCase()}…`, async () => {
    const r = await fxFetch(format, fxSet().scale, true);
    const blob = await r.blob(), a = document.createElement('a');
    const name = /filename="([^"]+)"/.exec(r.headers.get('Content-Disposition') || '')?.[1] || `clip.${format}`;
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    const saved = r.headers.get('X-Saved');
    const mb = blob.size / 1048576;
    status(`Downloaded ${name} (${mb < 1 ? Math.round(blob.size / 1024) + ' KB' : mb.toFixed(1) + ' MB'})${saved ? ` · saved to ${saved}/` : ''}`, 'ok');
    toast(`${name} · ${mb < 1 ? Math.round(blob.size / 1024) + ' KB' : mb.toFixed(1) + ' MB'}`);
  });
}
$('fx-gif').addEventListener('click', () => void downloadFx('gif'));
$('fx-webp').addEventListener('click', () => void downloadFx('webp'));

// ---- opening pictures

async function listInputs(): Promise<void> {
  try {
    const names = await (await fetch('/api/inputs')).json() as string[];
    const sel = $<HTMLSelectElement>('inputs');
    sel.hidden = !names.length;
    sel.innerHTML = '<option value="">From in/</option>' + names.map(n => `<option>${esc(n)}</option>`).join('');
  } catch { /* the first call will say the server is down */ }
}
$<HTMLSelectElement>('inputs').addEventListener('change', e => {
  const sel = e.target as HTMLSelectElement, n = sel.value;
  sel.value = '';
  if (n) void openPicture({ name: n }, '/in/' + encodeURIComponent(n));
});
const pick = (): void => $<HTMLInputElement>('file').click();
$('open').addEventListener('click', pick);
$('open-2').addEventListener('click', pick);
function openFile(f: File | undefined): void {
  if (!f) return;
  if (!/^image\/(png|jpeg|webp|gif|bmp)$/.test(f.type)) { status('That isn’t a PNG, JPEG, WebP, GIF or BMP.', 'bad'); return; }
  if (f.size > 16 << 20) { status('That picture is over 16 MB.', 'bad'); return; }
  const rd = new FileReader();
  rd.onload = () => { const data = String(rd.result); void openPicture({ name: f.name, data }, data); };
  rd.readAsDataURL(f);
}
$<HTMLInputElement>('file').addEventListener('change', e => { const i = e.target as HTMLInputElement; openFile(i.files?.[0]); i.value = ''; });
let dragDepth = 0;
addEventListener('dragenter', e => { if (e.dataTransfer?.types.includes('Files')) { dragDepth++; document.body.classList.add('drag'); } });
addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('drag'); } });
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => { e.preventDefault(); dragDepth = 0; document.body.classList.remove('drag'); openFile(e.dataTransfer?.files[0]); });
addEventListener('paste', e => {
  const f = [...(e.clipboardData?.files ?? [])].find(x => x.type.startsWith('image/'));
  if (f) openFile(f);
});

// ---- what the stage shows: the result; while editing the cut-out, the picture beside it (on a phone, instead of it)

type Stage = 'picture' | 'split' | 'live';
const editing = (): boolean => $('stage-box').dataset.view !== 'live';
function showView(v: Stage): void {
  if (narrow.matches && v === 'split') v = 'picture';
  $('stage-box').dataset.view = v;
  const b = $('edit-btn');
  pressed(b, v !== 'live');
  b.textContent = v === 'live' ? 'Crop & cut-out' : 'Done';
  requestAnimationFrame(() => { layout(); fitPreview(); });
}
$('edit-btn').addEventListener('click', () => showView(editing() ? 'live' : 'split'));

// ---- the preview full screen, the shortcuts sheet, sliders back to their defaults

function fullScreen(): void {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void $('preview-box').requestFullscreen?.().catch(() => { /* not allowed here */ });
}
$('full-btn').addEventListener('click', fullScreen);
document.addEventListener('fullscreenchange', () => requestAnimationFrame(fitPreview));
const help = $<HTMLDialogElement>('help');
$('help-btn').addEventListener('click', () => help.showModal());
help.addEventListener('click', e => { if (e.target === help) help.close(); });   // a click on the backdrop
document.addEventListener('dblclick', e => {
  const r = (e.target as HTMLElement).closest<HTMLInputElement>('input[type=range]');
  if (!r || r.disabled || r.value === r.defaultValue) return;
  r.value = r.defaultValue;
  r.dispatchEvent(new Event('input', { bubbles: true }));
});

// ---- keys: 1-4 for the styles, B K R E F for the tools, [ ] for the brush, O to open, G W to download

addEventListener('keydown', e => {
  const t = e.target instanceof Element ? e.target : document.body;
  if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'z' && !t.closest('input[type=text], input[type=number], textarea')) { undoStroke(); e.preventDefault(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey || help.open || t.closest('input[type=text], input[type=number], select, textarea')) return;
  const modes: Record<string, Mode> = { 1: 'glow', 2: 'streak', 3: 'dust', 4: 'live' };
  if (modes[e.key]) { setMode(modes[e.key]); e.preventDefault(); return; }
  if (e.key === '?') { help.showModal(); e.preventDefault(); return; }
  if (e.key.toLowerCase() === 'v' && !e.shiftKey) {
    if (st.img) showView(editing() ? 'live' : 'split');
    e.preventDefault(); return;
  }
  if (e.shiftKey && e.key.toLowerCase() === 'f') { fullScreen(); e.preventDefault(); return; }
  if (!e.shiftKey && fxs.mode !== 'live' && (e.key === 'g' || e.key === 'w') && !$<HTMLButtonElement>('fx-gif').disabled) { void downloadFx(e.key === 'g' ? 'gif' : 'webp'); e.preventDefault(); return; }
  const tools: Record<string, Tool> = { b: 'box', k: 'keep', r: 'remove', f: 'frame' };
  const k = e.key.toLowerCase();
  if (tools[k] && st.img) { if (!editing()) showView('split'); setTool(tools[k]); e.preventDefault(); }
  else if (k === 'o') { pick(); e.preventDefault(); }
  else if (k === 'escape') {   // out of whatever is in hand: picking the light source, then the editor
    if ($('fx-img').classList.contains('picking')) setPickFocus(false);
    else if (editing()) showView('live');
  }
  else if ((k === '[' || k === ']') && st.img) {
    st.brush = Math.max(4, Math.min(80, st.brush + (k === ']' ? 4 : -4)));
    setRange('brush', st.brush, `${st.brush} px`);
    draw();
  }
});

// ---- start

/** What this server allows: on the public site there is no in/ or out/, and smaller limits. */
async function configure(): Promise<void> {
  try {
    server = await (await fetch('/api/config')).json() as typeof server;
  } catch { status('The server is not answering.', 'bad'); return; }
  if (server.public) {
    $('privacy').textContent = 'or drop one anywhere, or paste · up to 16 MB, nothing is stored';
  } else {
    $('save-row').hidden = false;
    void listInputs();
  }
  for (const id of ['w', 'h']) $<HTMLInputElement>(id).max = String(server.maxSize);
  document.querySelectorAll<HTMLElement>('#sizes .chip, #shapes .chip').forEach(c => {
    const [W, H] = c.dataset.v!.split('x').map(Number);
    c.hidden = W > server.maxSize || H > server.maxSize;
  });
  exportSummary();
}

new ResizeObserver(() => { layout(); fitPreview(); }).observe($('main'));
narrow.addEventListener('change', () => showView(editing() ? 'split' : 'live'));
// ---- lettering in dots: the wordmark, the styles' names, and the empty stage

const GLYPHS: Record<string, string> = {   // 5 x 7, rows top to bottom, a bit per dot
  A: '0e 11 11 1f 11 11 11', C: '0e 11 10 10 10 11 0e', D: '1e 11 11 11 11 11 1e', E: '1f 10 10 1e 10 10 1f', G: '0e 11 10 17 11 11 0f',
  H: '11 11 11 1f 11 11 11', I: '0e 04 04 04 04 04 0e', K: '11 12 14 18 14 12 11', L: '10 10 10 10 10 10 1f', M: '11 1b 15 15 11 11 11',
  O: '0e 11 11 11 11 11 0e', P: '1e 11 11 1e 10 10 10', R: '1e 11 11 1e 14 12 11', S: '0f 10 10 0e 01 01 1e', T: '1f 04 04 04 04 04 04',
  U: '11 11 11 11 11 11 0e', W: '11 11 11 15 15 1b 11', ' ': '00 00 00 00 00 00 00',
};
/** Where a line of text lights dots: [column, row] pairs, and how many columns it takes. */
function dotsOf(text: string): { at: Pt[], cols: number } {
  const at: Pt[] = [];
  let x0 = 0;
  for (const ch of text) {
    const rows = (GLYPHS[ch] ?? GLYPHS[' ']).split(' ').map(h => parseInt(h, 16));
    rows.forEach((bits, y) => { for (let x = 0; x < 5; x++) if (bits & (16 >> x)) at.push([x0 + x, y]); });
    x0 += ch === ' ' ? 4 : 6;
  }
  return { at, cols: Math.max(0, x0 - 1) };
}
function dm(text: string): string {
  const { at, cols } = dotsOf(text);
  return `<svg class="dm" viewBox="0 0 ${cols} 7" role="img" aria-label="${esc(text)}">${at.map(([x, y]) => `<circle cx="${x + 0.5}" cy="${y + 0.5}" r=".42"/>`).join('')}</svg>`;
}
$('brand').innerHTML = dm('PHOSPHOR');
document.querySelectorAll<HTMLElement>('[data-dm]').forEach(el => { el.innerHTML = dm(el.dataset.dm!); });

/** The empty stage: a field of dots spelling out what to do, shimmering as glow does. */
{
  const c = $<HTMLCanvasElement>('drop-dots'), x = c.getContext('2d')!, box = $('drop');
  const INKS = ['#2c72de', '#2c72de', '#2c72de', '#2c72de', '#2c72de', '#2c72de', '#b549c3', '#b549c3', '#ed8d37', '#ed8d37', '#4fc852', '#ffffff'];
  const hash = (a: number, b: number, t: number): number => { const v = Math.sin(a * 127.1 + b * 311.7 + t * 74.7) * 43758.5453; return v - Math.floor(v); };
  let tick = 0;
  const paint = (): void => {
    if (box.hidden) return;
    const r = box.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (r.width < 2) return;
    if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); }
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.clearRect(0, 0, r.width, r.height);
    const words = document.body.classList.contains('drag') ? ['LET GO'] : ['DROP A PICTURE'];
    let lines = words.map(dotsOf);
    let p = Math.min(13, Math.floor(r.width / (lines[0].cols + 10)));
    if (p < 6 && words[0] === 'DROP A PICTURE') { lines = ['DROP A', 'PICTURE'].map(dotsOf); p = Math.min(13, Math.floor(r.width / (lines[1].cols + 6))); }
    p = Math.max(4, p);
    const cols = Math.ceil(r.width / p), rows = Math.ceil(r.height / p), rad = p * 0.34;
    const ox = Math.round((r.width - cols * p) / 2) + p / 2, oy = p / 2;
    const dot = (cx: number, cy: number, ink: string, a: number): void => { x.globalAlpha = a; x.fillStyle = ink; x.beginPath(); x.arc(ox + cx * p, oy + cy * p, rad, 0, 6.2832); x.fill(); };
    // the dark: a few faint dots, a few of them blinking
    for (let cy = 0; cy < rows; cy++) for (let cx = 0; cx < cols; cx++) {
      const h = hash(cx, cy, 0);
      if (h < 0.035) dot(cx, cy, '#2c72de', hash(cx, cy, tick >> 2) < 0.3 ? 0.5 : 0.16);
    }
    // the words: mostly blue, warm here and there, each dot now and then another colour or out
    const top = Math.round(rows * 0.42 - (lines.length * 10 - 3) / 2);
    lines.forEach((ln, i) => {
      const left = Math.round((cols - ln.cols) / 2);
      for (const [gx, gy] of ln.at) {
        const cx = left + gx, cy = top + i * 10 + gy, turn = Math.floor((tick + hash(cx, cy, 1) * 40) / 5);
        if (hash(cx, cy, turn + 0.5) < 0.05) continue;
        dot(cx, cy, INKS[Math.floor(hash(cx, cy, turn) * INKS.length)], 1);
      }
    });
    x.globalAlpha = 1;
  };
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  setInterval(() => { if (!box.hidden && !document.hidden) { if (!still.matches) tick++; paint(); } }, 110);
  paint();
  box.addEventListener('click', e => { if (!(e.target as HTMLElement).closest('button')) pick(); });
  box.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { pick(); e.preventDefault(); } });
}

const kept = recall();
if (kept.mapSize) sizes.map = kept.mapSize;
setSize(...(kept.size ?? [512, 512]));
setTool('box');
setMode(kept.mode ?? 'glow');
showView('live');
status('Ready', 'idle');
void configure();
