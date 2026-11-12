// Phosphor's web interface: a picture in, a map and PNGs out, with a live preview.
// The server (tools/phosphor_web.py) does the image work: cut-out, framing, blink, the map, PNGs.
// This page holds the choices (box, brush marks, eyes, view, size, tone), sends them, and draws the
// live preview with the renderer itself.

import { Phosphor, PRESETS, type ParticlesFrom, type PhosphorMap, type Preset } from '../src/phosphor';

type Eye = [number, number, number];      // x, y, half-width (picture px)
type View = [number, number, number];     // left, top, width (picture px); height from the map's aspect
type Stroke = [number, number, number, boolean]; // x, y, radius (picture px), keep
type Tool = 'box' | 'keep' | 'remove' | 'eyes' | 'frame';
type Pt = [number, number];
interface Pose { pivot?: Pt; center?: Pt }

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const narrow = matchMedia('(max-width: 820px)');

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
  skip: false,
  view: null as View | null,
  framed: false,                  // the view was moved by hand: a new cut-out keeps it
  size: [340, 280] as Pt,
  eyes: [] as Eye[],
  eyeSel: -1,
  tone: { ...toneDefaults },
  tool: 'box' as Tool,
  brush: 20,                      // screen px
  pose: null as Pose | null,      // guessed from the eyes
  poseSet: {} as Pose,            // clicked
  map: null as PhosphorMap | null,
  frames: 1,
};
const look = {
  preset: 'blue' as Preset, dot: 1, full: false, rays: false, particles: 'off' as 'off' | ParticlesFrom,
  blinkEvery: 4.5, breath: 1, sleep: false,
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
    const padB = narrow.matches ? 76 : 84; // room for the tool palette
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
  // eyes
  st.eyes.forEach(([x, y, r], i) => {
    const [sx, sy] = toScreen(x, y), sel = i === st.eyeSel;
    g.strokeStyle = sel ? '#ffffff' : acc; g.lineWidth = sel ? 1.75 : 1.25;
    g.beginPath(); g.ellipse(sx, sy, r * s, r * 0.65 * s, 0, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(sx - 4, sy); g.lineTo(sx + 4, sy); g.moveTo(sx, sy - 4); g.lineTo(sx, sy + 4); g.stroke();
  });
  // brush cursor
  if (hover && !drag?.touch && (st.tool === 'keep' || st.tool === 'remove')) {
    g.strokeStyle = st.tool === 'keep' ? '#5fd890' : '#f07a8c'; g.lineWidth = 1;
    g.beginPath(); g.arc(hover[0], hover[1], st.brush, 0, Math.PI * 2); g.stroke();
  }
}

let hover: Pt | null = null;
let drag: { kind: 'box' | 'paint' | 'eye' | 'view'; start: Pt; last: Pt; eye?: number; view?: View; touch: boolean } | null = null;

const at = (e: PointerEvent | WheelEvent | MouseEvent): Pt => {
  const r = cv.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
};
const eyeAt = (p: Pt): number => st.eyes.findIndex(([x, y, r]) => Math.hypot(p[0] - x, (p[1] - y) / 0.65) <= r + 8 / fit.s);

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
    dab(p);
  } else if (st.tool === 'eyes') {
    let i = eyeAt(p);
    if (i < 0) {
      const r = st.eyes.length ? st.eyes[st.eyes.length - 1][2] : Math.round(Math.min(st.iw, st.ih) * 0.04);
      if (st.eyes.length >= 2) { // a third tap moves the nearer one
        i = Math.hypot(p[0] - st.eyes[0][0], p[1] - st.eyes[0][1]) < Math.hypot(p[0] - st.eyes[1][0], p[1] - st.eyes[1][1]) ? 0 : 1;
        st.eyes[i] = [p[0], p[1], st.eyes[i][2]];
      } else {
        st.eyes.push([p[0], p[1], r]);
        i = st.eyes.length - 1;
      }
    }
    selectEye(i);
    drag = { kind: 'eye', start: p, last: p, eye: i, touch };
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
  } else if (drag.kind === 'eye' && drag.eye !== undefined) {
    const e0 = st.eyes[drag.eye];
    st.eyes[drag.eye] = [e0[0] + p[0] - drag.last[0], e0[1] + p[1] - drag.last[1], e0[2]];
    drag.last = p;
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
  } else if (st.tool === 'eyes') {
    const i = eyeAt(p);
    if (i < 0) return;
    e.preventDefault();
    st.eyes[i] = [st.eyes[i][0], st.eyes[i][1], Math.max(2, st.eyes[i][2] / f)];
    selectEye(i); draw(); rebuild();
  }
}, { passive: false });

cv.addEventListener('contextmenu', e => {
  if (st.tool !== 'eyes') return;
  const i = eyeAt(toPic(...at(e)));
  if (i < 0) return;
  e.preventDefault();
  st.eyes.splice(i, 1);
  selectEye(st.eyes.length - 1);
  draw(); rebuild(0);
});

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

function selectEye(i: number): void {
  st.eyeSel = i;
  const sl = $<HTMLInputElement>('eye-r');
  if (i >= 0 && st.eyes[i]) {
    sl.disabled = false;
    sl.max = String(Math.max(10, Math.round(Math.min(st.iw, st.ih) * 0.2)));
    setRange('eye-r', Math.round(st.eyes[i][2]), `${Math.round(st.eyes[i][2])} px`);
  } else {
    sl.disabled = true;
    $('eye-r-out').textContent = '–';
  }
}

function setTool(t: Tool): void {
  st.tool = t;
  document.querySelectorAll<HTMLButtonElement>('#tools button').forEach(b => pressed(b, b.dataset.tool === t));
  cv.style.cursor = t === 'frame' ? 'grab' : t === 'eyes' ? 'crosshair' : t === 'box' ? 'crosshair' : 'none';
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
    id: r.id, name: r.name, iw: r.w, ih: r.h, img, mask: null, rect: null, strokes: [], view: null, framed: false,
    eyes: [], eyeSel: -1, pose: null, poseSet: {}, map: null, frames: 1,
  });
  $<HTMLInputElement>('name').value = r.name;
  fxs.seq++; fxs.w = 0; $('fx-img').hidden = true; // the last picture's clip
  setSkip(r.hasAlpha); // a picture with its own transparency is cut out already
  selectEye(-1);
  setTool('box');
  $('file-meta').innerHTML = `<strong>${esc(r.name)}</strong><span>${r.w} × ${r.h}</span>`;
  $('src-meta').textContent = `${r.w} × ${r.h}${r.hasAlpha ? ' · transparent' : ''}`;
  layout();
  await cutout();
}

async function cutout(): Promise<void> {
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

const params = (): object => ({ id: st.id, view: st.view, size: st.size, eyes: st.eyes, tone: st.tone });

async function build(): Promise<void> {
  if (!st.id || !st.view) return;
  const seq = ++buildSeq;
  const r = await task('Building…', () => post<{ map: PhosphorMap; stats: { lit: number }; pose: Pose | null }>('/api/build', params()));
  if (!r || seq !== buildSeq) return;
  st.map = r.map;
  st.pose = r.pose;
  st.frames = 1 + Object.keys(r.map.frames).length;
  showPreview();
  const blinks = st.frames > 1;
  $('live-meta').textContent = `${r.map.w} × ${r.map.h} · ${Math.round(r.stats.lit * 100)}% lit · ${blinks ? 'blinks' : 'no blink yet'}`;
  status(blinks ? 'Ready' : 'Ready · mark the eyes to make it blink', 'ok');
  $<HTMLButtonElement>('export').disabled = false;
  exportSummary();
  if (fxs.mode !== 'live') renderFx();
}

// ---- the live preview

let ph: Phosphor | null = null;
let zoom = 1;

function pose(): Pose { return { ...(st.pose ?? {}), ...st.poseSet }; }

function showPreview(): void {
  if (!st.map) return;
  const first = !ph;
  if (ph) { ph.destroy(); ph.canvas.remove(); }
  const p = pose();
  ph = new Phosphor({
    map: st.map, pivot: p.pivot, center: p.center, colors: () => PRESETS[look.preset], pixelRatio: () => 1,
    reveal: first, particlesFrom: look.particles === 'off' ? 'right' : look.particles,
  });
  applyLook();
  $('stage').prepend(ph.canvas);
  if (!ph.load()) { status('No WebGL in this browser: no live preview, but export still works.', 'bad'); return; }
  $('prev-empty').hidden = true;
  $('mode-bar').classList.remove('off');
  setMode(fxs.mode);
}

function applyLook(): void {
  if (!ph) return;
  Object.assign(ph.look, {
    full: look.full ? 1 : 1.01, burst: look.rays ? 1 : 0,
    particles: look.particles === 'off' ? 0 : look.particles === 'all' ? 40 : 16,
    blinkEvery: look.blinkEvery, breathTilt: 1.5 * look.breath, breathRise: 2.5 * look.breath,
  });
  ph.dot = look.dot === 2 ? 2 : null;
  if (look.particles !== 'off') ph.particlesFrom = look.particles;
  ph.sleep(look.sleep);
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
  placeMarkers();
}

function placeMarkers(): void {
  const p = pose(), M = st.map;
  if (!M) return;
  for (const k of ['pivot', 'center'] as const) {
    const el = $(k === 'pivot' ? 'm-pivot' : 'm-center');
    const v = p[k] ?? (k === 'pivot' ? [M.w / 2, M.h * 0.8] : [M.w / 2, M.h * 0.45]);
    el.style.left = v[0] * zoom + 'px';
    el.style.top = v[1] * zoom + 'px';
  }
}

let picking: 'pivot' | 'center' | null = null;
function setPicking(k: 'pivot' | 'center' | null): void {
  picking = k;
  setSeg('pick', k ?? '');
  $('preview-box').classList.toggle('picking', !!k);
  $('pick-note').textContent = k === 'pivot' ? 'Click the top of the neck' : 'Click the middle of the head';
  placeMarkers();
}
$('stage').addEventListener('click', e => {
  if (!picking || !st.map) return;
  const r = $('stage').getBoundingClientRect();
  st.poseSet[picking] = [Math.round((e.clientX - r.left) / zoom), Math.round((e.clientY - r.top) / zoom)];
  setPicking(null);
  showPreview();
});

/** The accent follows the palette: the page takes the preview's colour. */
function setAccent(p: Preset): void {
  document.documentElement.style.setProperty('--accent', PRESETS[p][1]);
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
$('pick').addEventListener('click', e => {
  const b = (e.target as HTMLElement).closest('button');
  if (!b?.dataset.v) return;
  setPicking(picking === b.dataset.v ? null : b.dataset.v as 'pivot' | 'center');
});
toggle('sleep', v => { look.sleep = v; applyLook(); });
toggle('full', v => { look.full = v; applyLook(); });
toggle('rays', v => { look.rays = v; applyLook(); });
$('reveal').addEventListener('click', () => ph?.reveal());
range('blink-every', v => v ? `${v} s` : 'never', v => { look.blinkEvery = v; applyLook(); });
range('breath', v => `${v.toFixed(1)}×`, v => { look.breath = v; applyLook(); });

range('brush', v => `${v} px`, v => { st.brush = v; draw(); });
$<HTMLInputElement>('skip').addEventListener('change', e => { st.skip = (e.target as HTMLInputElement).checked; void cutout(); });
$('redo-cut').addEventListener('click', () => { void cutout(); });
$('clear-strokes').addEventListener('click', () => { st.strokes = []; void cutout(); });

range('eye-r', v => `${v} px`, v => {
  const i = st.eyeSel;
  if (i < 0 || !st.eyes[i]) return;
  st.eyes[i] = [st.eyes[i][0], st.eyes[i][1], v];
  draw(); rebuild();
});
$('clear-eyes').addEventListener('click', () => { st.eyes = []; selectEye(-1); draw(); rebuild(0); });
range('zoom', () => st.view ? `${Math.round(st.iw / st.view[2] * 100)}%` : '–', v => { if (st.iw) zoomView(st.iw / Math.pow(2, v)); });

// size
function setSize(W: number, H: number): void {
  W = Math.max(8, Math.min(server.maxSize, Math.round(W) || 8));
  H = Math.max(8, Math.min(server.maxSize, Math.round(H) || 8));
  if (st.view) { // keep the middle and width; the height follows
    const [x, y, w] = st.view, cy = y + w * st.size[1] / st.size[0] / 2;
    st.view = [x, cy - w * H / W / 2, w];
  }
  st.size = [W, H];
  $<HTMLInputElement>('w').value = String(W);
  $<HTMLInputElement>('h').value = String(H);
  $('sizes').querySelectorAll<HTMLElement>('.chip').forEach(c => pressed(c, c.dataset.v === `${W}x${H}`));
  st.poseSet = {}; // in map px: no longer where they were
  exportSummary();
  layout(); rebuild(0);
}
$('sizes').addEventListener('click', e => {
  const b = (e.target as HTMLElement).closest<HTMLElement>('.chip');
  if (!b?.dataset.v) return;
  const [W, H] = b.dataset.v.split('x').map(Number);
  setSize(W, H);
});
for (const id of ['w', 'h']) $(id).addEventListener('change', () => setSize(Number($<HTMLInputElement>('w').value), Number($<HTMLInputElement>('h').value)));
$('auto-frame').addEventListener('click', async () => {
  if (!st.id) return;
  const r = await task('Framing…', () => post<{ view: View }>('/api/view', { id: st.id, size: st.size }));
  if (!r) return;
  st.view = r.view; st.framed = false;
  syncZoom(); layout(); rebuild(0);
});

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

// ---- styled clips: glow and streak, rendered on the server as a looping WebP (preview) or GIF

type Mode = 'live' | 'glow' | 'streak' | 'dust';
type Style = Exclude<Mode, 'live'>;
const FX_COLOURS: [string, string][] = [['red', '#ee1111'], ['white', '#f2f2f2'], ['grey', '#d8d8d8'], ['green', '#4fe08a'], ['blue', '#6f93e8'], ['amber', '#ffb347'], ['rose', '#f07aa6']];
// each style's choices (value, label), and what it starts at: glow is a slow 3 s loop at 24 fps on a
// fine grid, streak a fast 1 s boil on a coarse one, dust a 2 s drift on a fine one
type FxSet = { long: number, frames: number, scale: number, angle: number, colour: string, flat?: boolean };
const FX: Record<Style, { long: number[], len: [number, string][], scale: number[], fps: number, d: FxSet }> = {
  glow: { long: [360, 540, 720], len: [[48, '2 s'], [72, '3 s'], [96, '4 s']], scale: [1, 2], fps: 24,
    d: { long: 540, frames: 72, scale: 2, angle: 0, colour: '' } },
  streak: { long: [180, 270, 360], len: [[12, '½ s'], [24, '1 s'], [48, '2 s']], scale: [2, 4], fps: 24,
    d: { long: 270, frames: 24, scale: 4, angle: 0, colour: 'red' } },
  dust: { long: [360, 540, 720], len: [[24, '1 s'], [48, '2 s'], [72, '3 s']], scale: [1, 2], fps: 24,
    d: { long: 540, frames: 48, scale: 2, angle: 0, colour: 'white', flat: false } },
};
const FX_ANGLES: [number, string, string][] = [[0, '↔', 'Across'], [90, '↕', 'Up and down'], [45, '⤡', 'Diagonal, falling'], [315, '⤢', 'Diagonal, rising']];
const fxSets: Record<Style, FxSet> = { glow: { ...FX.glow.d }, streak: { ...FX.streak.d }, dust: { ...FX.dust.d } };
const fxs = { mode: 'live' as Mode, url: '', seq: 0, w: 0, h: 0 };
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
  $('fx-angle').innerHTML = FX_ANGLES.map(([v, l, t]) => b(v, l, v === c.angle, t)).join('');
  setSeg('fx-colours', c.colour);
  setSeg('fx-tone', c.flat ? '1' : '0');
}

$('fx-colours').innerHTML = FX_COLOURS.map(([v, c]) =>
  `<button data-v="${v}" aria-pressed="${v === 'red'}" title="${v}"><span class="sw" style="background:${c}"></span></button>`).join('');
seg('fx-colours', v => { fxSet().colour = v; renderFx(0); });
seg('fx-long', v => { fxSet().long = Number(v); renderFx(0); });
seg('fx-len', v => { fxSet().frames = Number(v); renderFx(0); });
seg('fx-scale', v => { fxSet().scale = Number(v); });
seg('fx-angle', v => { fxSet().angle = Number(v); renderFx(0); });
seg('fx-tone', v => { fxSet().flat = v === '1'; renderFx(0); });
seg('mode', v => setMode(v as Mode));

function setMode(m: Mode): void {
  fxs.mode = m;
  setSeg('mode', m);
  const live = m === 'live';
  if (ph) ph.canvas.hidden = !live;
  $('live-bar').classList.toggle('off', !live || !st.map);
  $('fx-bar').classList.toggle('off', live || !st.map);
  $('fx-colours').hidden = m === 'live' || m === 'glow';
  $('fx-angle').hidden = m !== 'glow' && m !== 'dust';
  $('fx-tone').hidden = m !== 'dust';
  fxButtons();
  $('fx-img').hidden = live || !fxs.url;
  $('fx-wait').hidden = true;
  if (live) {
    if (picking) setPicking(null);
    fxs.seq++; // a clip still on its way is no longer wanted
    if (st.map) $('live-meta').textContent = `${st.map.w} × ${st.map.h} · live`;
    fitPreview();
  } else {
    $('fx-img').hidden = true; // the last clip was another style, or another picture
    renderFx(0);
  }
}

function fxBody(format: string, scale: number, save: boolean): object {
  const name = ($<HTMLInputElement>('name').value.trim() || st.name).replace(/[^A-Za-z0-9_.-]+/g, '-');
  return { ...params(), style: fxs.mode, long: fxSet().long, frames: fxSet().frames, angle: fxs.mode === 'streak' ? undefined : fxSet().angle,
    colour: fxSet().colour || undefined, flat: fxs.mode === 'dust' ? fxSet().flat : undefined, format, scale, name,
    save: save && !server.public && $<HTMLInputElement>('save').checked };
}
async function fxFetch(format: string, scale: number, save = false): Promise<Response> {
  const r = await fetch('/api/fx', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fxBody(format, scale, save)) });
  if (!r.ok) throw new Error(await errorOf(r));
  return r;
}

function renderFx(ms = 600): void {
  clearTimeout(fxTimer);
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
  fxs.w = img.naturalWidth; fxs.h = img.naturalHeight;
  img.hidden = fxs.mode === 'live';
  $('live-meta').textContent = `${fxs.w} × ${fxs.h} · ${fxSet().frames} frames · ${+(fxSet().frames / FX[mode as Style].fps).toFixed(2)} s loop`;
  status('Ready', 'ok');
  fitFx();
}
function fitFx(): void {
  if (!fxs.w) return;
  const r = $('preview-box').getBoundingClientRect(), img = $<HTMLImageElement>('fx-img');
  const room = Math.min((r.width - 32) / fxs.w, (r.height - 2 * ($('fx-bar').getBoundingClientRect().height + 28)) / fxs.h);
  const z = room >= 1 ? Math.floor(room) : Math.max(0.1, room);
  img.style.width = fxs.w * z + 'px';
  img.style.height = fxs.h * z + 'px';
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
    status(`Downloaded ${name} (${(blob.size / 1048576).toFixed(1)} MB)${saved ? ` · saved to ${saved}/` : ''}`, 'ok');
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
  showView('picture');
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

// ---- phone: Picture / Live tabs

function showView(v: 'picture' | 'live'): void {
  $('main').classList.toggle('show-live', v === 'live');
  setSeg('view-tabs', v);
  requestAnimationFrame(() => { layout(); fitPreview(); });
}
seg('view-tabs', v => showView(v as 'picture' | 'live'));

// ---- keys: B K R E F for the tools, [ ] for the brush, O to open

addEventListener('keydown', e => {
  const t = e.target as HTMLElement;
  if (e.metaKey || e.ctrlKey || e.altKey || t.closest('input, select, textarea')) return;
  const tools: Record<string, Tool> = { b: 'box', k: 'keep', r: 'remove', e: 'eyes', f: 'frame' };
  const k = e.key.toLowerCase();
  if (tools[k] && st.img) { setTool(tools[k]); e.preventDefault(); }
  else if (k === 'o') { pick(); e.preventDefault(); }
  else if ((k === '[' || k === ']') && st.img) {
    st.brush = Math.max(4, Math.min(80, st.brush + (k === ']' ? 4 : -4)));
    setRange('brush', st.brush, `${st.brush} px`);
    draw();
  } else if (k === 'escape' && picking) setPicking(null);
});

// ---- start

/** The empty preview: a little field of dots, some lit. */
function ghost(): void {
  const dots: string[] = [];
  for (let y = 0; y < 8; y++) for (let x = 0; x < 12; x++) {
    const d = Math.hypot(x - 5.5, (y - 3.5) * 1.4);
    if (d < 5.2) dots.push(`<circle cx="${x * 10 + 5}" cy="${y * 10 + 5}" r="2" opacity="${d < 3 ? 0.9 : 0.35}"/>`);
  }
  $('ghost-dots').innerHTML = dots.join('');
}

/** What this server allows: on the public site there is no in/ or out/, and smaller limits. */
async function configure(): Promise<void> {
  try {
    server = await (await fetch('/api/config')).json() as typeof server;
  } catch { status('The server is not answering.', 'bad'); return; }
  if (server.public) {
    $('privacy').textContent = 'PNG, JPEG, WebP, GIF or BMP, up to 16 MB. Your picture stays on the server only while you work, and is never stored.';
  } else {
    $('save-row').hidden = false;
    void listInputs();
  }
  for (const id of ['w', 'h']) $<HTMLInputElement>(id).max = String(server.maxSize);
  $('sizes').querySelectorAll<HTMLElement>('.chip').forEach(c => {
    const [W, H] = c.dataset.v!.split('x').map(Number);
    c.hidden = W > server.maxSize || H > server.maxSize;
  });
  exportSummary();
}

new ResizeObserver(() => { layout(); fitPreview(); }).observe($('main'));
narrow.addEventListener('change', () => showView('picture'));
setSize(340, 280);
setTool('box');
ghost();
status('Ready', 'idle');
void configure();
