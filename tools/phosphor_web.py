#!/usr/bin/env python3
"""Phosphor in the browser: from a picture to a map and PNGs, with a live preview, or a styled clip.

    npm run web            (builds the page, then runs this)
    python tools/phosphor_web.py [--port 8340] [--host 127.0.0.1] [--public]

then open http://localhost:8340. Pictures can be dropped on the page or picked from in/; an export
downloads a zip and also saves it to out/<name>/ (map.ts, and a folder of PNGs per frame).

--public is for a server on the internet (behind a reverse proxy, as at phosphor.laika.fm): no in/
or out/ (nothing of the host's is read or written), smaller limits on uploads, map and PNG sizes,
fewer pictures kept and for less long, and a few heavy jobs at a time.

A small JSON API over phosphor_prep (cut-out, framing, blink), phosphor_map (the map, PNGs) and
phosphor_fx (the glow and streak clips).
Pictures are kept in memory, by a random id, until they go unused for a while (or the server
restarts). Only the standard library besides those two's numpy, opencv and pillow. No secrets.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import secrets
import sys
import threading
import time
import traceback
import warnings
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phosphor_fx as fx  # noqa: E402
import phosphor_map as pm  # noqa: E402
import phosphor_prep as pp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB, IN, OUT = ROOT / 'web', ROOT / 'in', ROOT / 'out'
PICS = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')
TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css',
         '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
         '.gif': 'image/gif', '.bmp': 'image/bmp', '.map': 'application/json'}
TONE = {f for f in pm.Tone.__dataclass_fields__}
FORMATS = ['PNG', 'JPEG', 'WEBP', 'GIF', 'BMP']


@dataclass
class Limits:
    public: bool = False
    max_size: int = 2048           # map side, px
    max_body: int = 64 << 20       # request body, bytes (a picture comes as base64: 4/3 its size)
    max_pixels: int = 64_000_000   # a picture's px (Pillow refuses bigger: decompression bombs)
    max_side: int = 0              # pictures are scaled down to this longest side (0: kept)
    max_out: int = 0               # one PNG's px, after scale (0: no limit)
    max_fx: int = 1080             # a styled clip's longest side, in dots
    max_fx_out: int = 2160         # and in px, after scale
    max_fx_frames: int = 96
    keep: int = 64                 # pictures kept
    ttl: float = 12 * 3600         # seconds unused before one is dropped
    slots: int = 4                 # heavy jobs (cut-out, build, export) at once


LOCAL = Limits()
PUBLIC = Limits(public=True, max_size=1024, max_body=24 << 20, max_pixels=40_000_000, max_side=2048,
                max_out=16_000_000, max_fx=720, max_fx_out=1440, max_fx_frames=72,
                keep=16, ttl=3600, slots=2)  # ~50 MB a picture: under 1 GB held
CFG = LOCAL
PICTURES: dict[str, dict] = {}  # id -> rgb, alpha (or None), given (its own alpha), name, used
LOCK = threading.Lock()
WORK = threading.BoundedSemaphore(CFG.slots)


class Bad(Exception):
    """A request we can't do: answered 400 with this message."""


class Busy(Exception):
    """Too many heavy jobs at once: answered 503."""


def load(raw: bytes, name: str) -> dict:
    Image.MAX_IMAGE_PIXELS = CFG.max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(raw), formats=FORMATS)
            if im.width * im.height > CFG.max_pixels:
                raise Bad('That picture is too big.')
            im.seek(0)
            im.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise Bad('Could not read that picture (PNG, JPEG, WebP, GIF or BMP).')
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise Bad('That picture is too big.')
    im = im.convert('RGBA')
    if CFG.max_side and max(im.size) > CFG.max_side:
        k = CFG.max_side / max(im.size)
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    rgba = np.asarray(im)
    a = rgba[..., 3].astype(np.float32) / 255
    own = a if a.min() < 0.98 else None  # the picture's own transparency, if it has any
    pid = secrets.token_urlsafe(16)
    nm = re.sub(r'[^A-Za-z0-9_.-]+', '-', Path(name).stem).strip('.-')[:60] or 'picture'
    with LOCK:
        forget()
        while len(PICTURES) >= CFG.keep:  # the least recently used goes
            del PICTURES[min(PICTURES, key=lambda k: PICTURES[k]['used'])]
        PICTURES[pid] = {'rgb': np.ascontiguousarray(rgba[..., :3]), 'alpha': own, 'given': own, 'name': nm,
                         'used': time.monotonic()}
    return {'id': pid, 'w': rgba.shape[1], 'h': rgba.shape[0], 'hasAlpha': own is not None, 'name': nm}


def forget() -> None:
    """Drop pictures unused for CFG.ttl (call with LOCK held)."""
    now = time.monotonic()
    for k in [k for k, p in PICTURES.items() if now - p['used'] > CFG.ttl]:
        del PICTURES[k]


def picture(q: dict) -> dict:
    with LOCK:
        forget()
        p = PICTURES.get(str(q.get('id')))
        if not p:
            raise Bad('That picture is gone (unused for a while, or the server restarted). Open it again.')
        p['used'] = time.monotonic()
    return p


def size_of(q: dict) -> tuple[int, int]:
    try:
        W, H = (int(v) for v in q.get('size', (340, 280)))
    except (TypeError, ValueError):
        raise Bad('size: two whole numbers, W and H')
    if not (8 <= W <= CFG.max_size and 8 <= H <= CFG.max_size):
        raise Bad(f'size: each side 8 to {CFG.max_size} px')
    return W, H


def mask_png(a: np.ndarray) -> str:
    buf = io.BytesIO()
    v = (a * 255).round().astype(np.uint8)  # white, the figure opaque: a mask the page can draw with
    Image.fromarray(np.dstack([np.full_like(v, 255), v]), 'LA').save(buf, 'PNG', compress_level=1)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def make(q: dict) -> tuple[dict, dict | None]:
    """The map for a request: picture, cut-out, view, size, eyes, tone. Also a guessed pose."""
    p = picture(q)
    W, H = size_of(q)
    rgb = p['rgb']
    alpha = p['alpha'] if p['alpha'] is not None else np.ones(rgb.shape[:2], np.float32)
    try:
        v = q['view']
        view = (float(v[0]), float(v[1]), float(v[2]))
    except (KeyError, TypeError, ValueError, IndexError):
        view = pp.auto_view(alpha, (W, H))
    if view[2] <= 1:
        raise Bad('view: too small')
    eyes = [(float(e[0]), float(e[1]), max(2.0, float(e[2]))) for e in q.get('eyes') or []][:2]
    T = pm.Tone(**{k: type(getattr(pm.Tone(), k))(v) for k, v in (q.get('tone') or {}).items() if k in TONE})
    shots = {'idle': pp.frame(rgb, alpha, view, (W, H))}
    if eyes:
        shots['blink'] = pp.frame(pp.blink(rgb, eyes), alpha, view, (W, H))
    idle = shots['idle'].astype(np.float32) / 255
    a0 = idle[..., 3]
    if a0.max() < 0.5:
        raise Bad('Nothing of the figure is in view: move the frame onto it, or redo the cut-out.')
    lum = lambda im: np.asarray(Image.fromarray(im[..., :3]).convert('L'), np.float32) / 255
    shaded = lum(shots['idle']) * a0
    frames = {n: (lum(im) * (im[..., 3] / 255), im[..., 3].astype(np.float32) / 255) for n, im in shots.items()}
    m = pm.build(shaded, frames, 'idle', None, T)
    return m, pp.guess_pose(eyes, view, (W, H))


def styled(q: dict) -> tuple[fx.Frames, str]:
    """A styled clip for a request: picture, cut-out, view, the map's size (for the aspect), style,
    long (the clip's longest side, in dots), frames, colour. The last one is kept with the picture,
    so a download right after its preview isn't drawn again."""
    p = picture(q)
    style = str(q.get('style'))
    if style not in fx.STYLES:
        raise Bad('style: ' + ' or '.join(fx.STYLES))
    W0, H0 = size_of(q)
    try:
        long = int(q.get('long') or 540)
        frames = int(q.get('frames') or 48)
    except (TypeError, ValueError):
        raise Bad('long, frames: whole numbers')
    if not 64 <= long <= CFG.max_fx:
        raise Bad(f'long: 64 to {CFG.max_fx} dots')
    if not 8 <= frames <= CFG.max_fx_frames:
        raise Bad(f'frames: 8 to {CFG.max_fx_frames}')
    colour = str(q.get('colour') or '')
    if colour and colour not in fx.COLOURS and not re.fullmatch(r'#[0-9a-fA-F]{6}', colour):
        raise Bad('colour: #rrggbb or ' + ', '.join(fx.COLOURS))
    k = long / max(W0, H0)
    W, H = max(8, round(W0 * k)), max(8, round(H0 * k))
    rgb = p['rgb']
    alpha = p['alpha'] if p['alpha'] is not None else np.ones(rgb.shape[:2], np.float32)
    try:
        v = q['view']
        view = (float(v[0]), float(v[1]), float(v[2]))
    except (KeyError, TypeError, ValueError, IndexError):
        view = pp.auto_view(alpha, (W0, H0))
    if view[2] <= 1:
        raise Bad('view: too small')
    key = (style, W, H, view, frames, colour, id(p['alpha']))
    if p.get('fx', (None,))[0] == key:
        return p['fx'][1], style
    rgba = pp.frame(rgb, alpha, view, (W, H))
    if rgba[..., 3].max() < 128:
        raise Bad('Nothing of the figure is in view: move the frame onto it, or redo the cut-out.')
    fr = fx.render(rgba, style, frames, colour or None)
    p['fx'] = (key, fr)
    return fr, style


HEAVY = {'/api/cutout', '/api/build', '/api/export', '/api/fx'}


def api(path: str, q: dict) -> tuple[str, bytes, dict]:
    if path in HEAVY:
        if not WORK.acquire(timeout=30):
            raise Busy()
        try:
            return route(path, q)
        finally:
            WORK.release()
    return route(path, q)


def route(path: str, q: dict) -> tuple[str, bytes, dict]:
    if path == '/api/config':
        return reply({'public': CFG.public, 'maxSize': CFG.max_size, 'maxOut': CFG.max_out,
                      'maxFx': CFG.max_fx, 'maxFxOut': CFG.max_fx_out})
    if path == '/api/inputs':
        names = [] if CFG.public or not IN.is_dir() else sorted(f.name for f in IN.glob('*') if f.suffix.lower() in PICS)
        return 'application/json', json.dumps(names).encode(), {}
    if path == '/api/image':
        if q.get('data'):
            try:
                raw = base64.b64decode(str(q['data']).split(',', 1)[-1], validate=False)
            except ValueError:
                raise Bad('Could not read that picture.')
            return reply(load(raw, str(q.get('name') or 'picture')))
        if CFG.public:
            raise Bad('Upload a picture.')
        f = (IN / str(q.get('name', ''))).resolve()
        if f.parent != IN.resolve() or not f.is_file():
            raise Bad('No such picture in in/')
        return reply(load(f.read_bytes(), f.name))
    if path == '/api/cutout':
        p = picture(q)
        rgb = p['rgb']
        if q.get('skip'):
            a = p['given'] if p['given'] is not None else np.ones(rgb.shape[:2], np.float32)
        else:
            strokes = [(float(s[0]), float(s[1]), float(s[2]), bool(s[3])) for s in q.get('strokes') or []]
            a = pp.cutout(rgb, q.get('rect'), strokes)
        p['alpha'] = a
        p.pop('fx', None)
        return reply({'mask': mask_png(a), 'view': pp.auto_view(a, size_of(q)), 'empty': bool(a.max() < 0.5)})
    if path == '/api/view':
        p = picture(q)
        a = p['alpha'] if p['alpha'] is not None else np.ones(p['rgb'].shape[:2], np.float32)
        return reply({'view': pp.auto_view(a, size_of(q))})
    if path == '/api/build':
        m, pose = make(q)
        keep = {k: m[k] for k in ('w', 'h', 'idle', 'data', 'body', 'frames')}
        return reply({'map': keep, 'stats': m['stats'], 'pose': pose})
    if path == '/api/export':
        m, _ = make(q)
        name = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(q.get('name') or picture(q)['name'])).strip('.-')[:60] or 'picture'
        dots = sorted({int(d) for d in q.get('dots') or [1] if int(d) in (1, 2, 3, 4)}) or [1]
        bgs = [b for b in q.get('backgrounds') or ['none'] if b in pm.BACKGROUNDS] or ['none']
        scale = max(1, min(16, int(q.get('scale') or 1)))
        W, H = size_of(q)
        if CFG.max_out and W * H * scale * scale > CFG.max_out:
            raise Bad(f'PNG scale {scale}x makes {W * scale}x{H * scale} px PNGs: too big here. Pick a smaller scale.')
        files = [('map.ts', pm.to_ts(m, 'LEDMAP', pm.HEADER).encode())]
        for rel, img in pm.pngs(m, dots, bgs, q.get('colours'), scale):
            buf = io.BytesIO()
            img.save(buf, 'PNG')
            files.append((rel, buf.getvalue()))
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
            for rel, data in files:
                z.writestr(f'{name}/{rel}', data)
        saved = ''
        if q.get('save') and not CFG.public:
            dest = OUT / name
            for rel, data in files:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                (dest / rel).write_bytes(data)
            saved = str(dest.relative_to(ROOT))
        return 'application/zip', zbuf.getvalue(), {
            'Content-Disposition': f'attachment; filename="{name}.zip"',
            'X-Files': str(len(files)), 'X-Saved': saved}
    if path == '/api/fx':
        fr, style = styled(q)
        fmt = str(q.get('format') or 'gif')
        if fmt not in ('gif', 'webp', 'png'):
            raise Bad('format: gif, webp or png')
        try:
            scale = max(1, min(8, int(q.get('scale') or 1)))
        except (TypeError, ValueError):
            raise Bad('scale: a whole number')
        H, W = fr[0].shape[1:]
        if max(W, H) * scale > CFG.max_fx_out:
            raise Bad(f'{scale}x makes {W * scale}x{H * scale} px: too big here. Pick a smaller scale or size.')
        data = fx.encode(fr, fmt, 16, scale)
        name = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(q.get('name') or picture(q)['name'])).strip('.-')[:60] or 'picture'
        saved = ''
        if q.get('save') and not CFG.public:
            dest = OUT / name / 'fx'
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f'{style}.{fmt}').write_bytes(data)
            saved = str(dest.relative_to(ROOT))
        return {'gif': 'image/gif', 'webp': 'image/webp', 'png': 'image/png'}[fmt], data, {
            'Content-Disposition': f'attachment; filename="{name}-{style}.{fmt}"', 'X-Saved': saved}
    raise Bad(f'No such call: {path}')


def reply(obj) -> tuple[str, bytes, dict]:
    # numpy scalars (bools, floats) as plain ones
    return 'application/json', json.dumps(obj, default=lambda o: o.item() if hasattr(o, 'item') else str(o)).encode(), {}


class Handler(BaseHTTPRequestHandler):
    server_version = 'phosphor'

    def send(self, code: int, kind: str, body: bytes, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def fail(self, code: int, msg: str) -> None:
        self.send(code, 'application/json', json.dumps({'error': msg}).encode())

    def do_GET(self) -> None:
        path = self.path.split('?', 1)[0]
        if path.startswith('/api/'):
            return self.call(path, {})
        if path.startswith('/in/'):
            if CFG.public:
                return self.fail(404, 'not found')
            base, rel = IN, path[4:]
        else:
            base, rel = WEB, path.lstrip('/') or 'index.html'
        f = (base / rel).resolve()
        if base.resolve() not in f.parents or not f.is_file():
            return self.fail(404, 'not found')
        self.send(200, TYPES.get(f.suffix.lower(), 'application/octet-stream'), f.read_bytes())

    def do_POST(self) -> None:
        try:
            n = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            return self.fail(400, 'bad Content-Length')
        if n > CFG.max_body:
            self.close_connection = True
            return self.fail(413, f'That is too big (over {CFG.max_body >> 20} MB).')
        try:
            q = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            return self.fail(400, 'bad JSON')
        self.call(self.path.split('?', 1)[0], q if isinstance(q, dict) else {})

    def call(self, path: str, q: dict) -> None:
        try:
            self.send(200, *api(path, q))
        except Bad as e:
            self.fail(400, str(e))
        except Busy:
            self.fail(503, 'Busy right now: try again in a moment.')
        except MemoryError:
            self.fail(503, 'Out of memory for that: try a smaller picture or size.')
        except Exception as e:  # noqa: BLE001 - the trace goes to the log; the page gets a summary
            traceback.print_exc()
            self.fail(500, 'Something went wrong.' if CFG.public else f'{type(e).__name__}: {e}')

    def log_message(self, fmt: str, *args) -> None:
        if not self.path.startswith('/api/'):
            return
        sys.stderr.write('%s %s\n' % (self.log_date_time_string(), fmt % args))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--port', type=int, default=8340)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--public', action='store_true', help='internet-facing: no in/ or out/, tighter limits')
    args = ap.parse_args()
    global CFG, WORK
    CFG = PUBLIC if args.public else LOCAL
    WORK = threading.BoundedSemaphore(CFG.slots)
    if not (WEB / 'dist' / 'app.js').is_file():
        sys.exit('web/dist/app.js is missing: run `npm run web` (or `npm run build:web` first).')
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f'Phosphor{" (public)" if CFG.public else ""}: '
          f'http://{"localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host}:{args.port}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
