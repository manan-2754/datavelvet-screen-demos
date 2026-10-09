"""
Screen-demo engine: renders a realistic Windows 11 desktop with a photo-editing app and performs a scripted,
narrated demo on it - the cursor moves and clicks, menus open, sliders drag, a brush paints a real mask -
while a smooth "screen-recording" camera zooms into whatever is being used.

All UI is vector-drawn (Skia) in virtual desktop coordinates (2560x1440) and rendered through the camera,
so zoomed-in shots stay razor sharp. Image edits are real pixel operations (photo_ops).
"""
import json
import math
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import skia

import photo_ops as po

BASE = Path(__file__).parent
FONT_DIR = BASE / "assets" / "fonts"
PHOTO_DIR = BASE / "assets" / "photos"
POOL = json.loads((PHOTO_DIR / "pool.json").read_text(encoding="utf-8")) if (PHOTO_DIR / "pool.json").exists() else []

VW, VH = 2560, 1440                 # virtual desktop
DW, DH = 1080, 1920                 # output design space
CARD_X, CARD_W, CARD_CY = 40, 1000, 930
HOOK_END = 2.3

SANS, SANS_REG, SANS_BOLD, MONO = "Inter-Medium", "Inter-Regular", "Inter-SemiBold", "JetBrainsMono-Regular"

C_WIN = (32, 32, 35)
C_PANEL = (40, 40, 44)
C_PANEL_HDR = (48, 48, 53)
C_LINE = (22, 22, 24)
C_TEXT = (222, 222, 228)
C_DIM = (150, 150, 160)
C_CANVAS = (26, 26, 28)
C_ACCENT = (0, 120, 212)            # Windows blue
C_ACCENT2 = (76, 194, 255)
BRAND = (56, 148, 255)              # overlay accent


def ffmpeg_exe():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_back(x):
    x = clamp(x)
    c1 = 1.70158
    return 1 + (c1 + 1) * (x - 1) ** 3 + c1 * (x - 1) ** 2


@lru_cache(maxsize=None)
def typeface(name):
    return skia.Typeface.MakeFromFile(str(FONT_DIR / f"{name}.ttf"))


def col(rgb, a=1.0):
    return skia.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(255 * clamp(a)))


def fillp(rgb, a=1.0):
    return skia.Paint(AntiAlias=True, Color=col(rgb, a))


def strokep(rgb, a=1.0, w=1.0, dash=None):
    p = skia.Paint(AntiAlias=True, Color=col(rgb, a), Style=skia.Paint.kStroke_Style, StrokeWidth=w,
                   StrokeCap=skia.Paint.kRound_Cap, StrokeJoin=skia.Paint.kRound_Join)
    if dash:
        p.setPathEffect(skia.DashPathEffect.Make(list(dash), 0))
    return p


class Text:
    """Text rasterised once per (string, size, colour, zoom bucket) and blitted - fast and crisp at any zoom."""

    def __init__(self):
        self.cache = {}

    def width(self, s, font, size):
        return skia.Font(typeface(font), size).measureText(s)

    def image(self, s, font, size, rgb, q):
        key = (s, font, size, rgb, q)
        hit = self.cache.get(key)
        if hit is None:
            f = skia.Font(typeface(font), size * q)
            f.setSubpixel(True)
            m = f.getMetrics()
            w = max(1, int(math.ceil(f.measureText(s))) + 4)
            h = max(1, int(math.ceil(m.fDescent - m.fAscent)) + 4)
            surf = skia.Surface(w, h)
            cv = surf.getCanvas()
            cv.clear(skia.ColorTRANSPARENT)
            cv.drawString(s, 2, 2 - m.fAscent, f, skia.Paint(AntiAlias=True, Color=col(rgb)))
            hit = (surf.makeImageSnapshot(), -m.fAscent + 2)
            if len(self.cache) > 6000:
                self.cache.clear()
            self.cache[key] = hit
        return hit

    def draw(self, c, s, x, y, size, rgb=C_TEXT, font=SANS_REG, align="left", a=1.0, q=1.0):
        if not s or a <= 0.003:
            return
        img, asc = self.image(s, font, size, tuple(int(v) for v in rgb), q)
        w = img.width() / q
        if align == "center":
            x -= w / 2
        elif align == "right":
            x -= w
        c.save()
        c.translate(x, y)
        c.scale(1 / q, 1 / q)
        c.drawImage(img, 0, -asc, skia.SamplingOptions(skia.FilterMode.kLinear), skia.Paint(Alphaf=clamp(a)))
        c.restore()


def rr(c, x, y, w, h, r, rgb, a=1.0):
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r), fillp(rgb, a))


def rr_stroke(c, x, y, w, h, r, rgb, a=1.0, lw=1.0):
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r), strokep(rgb, a, lw))


def rect(c, x, y, w, h, rgb, a=1.0):
    c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), fillp(rgb, a))


def line(c, x0, y0, x1, y1, rgb, a=1.0, w=1.0):
    c.drawLine(x0, y0, x1, y1, strokep(rgb, a, w))


def soft_shadow(c, x, y, w, h, r, spread, a=0.5, steps=7):
    """Cheap soft shadow: a few expanding translucent round-rects (a blur MaskFilter is far too slow at 4K)."""
    for i in range(steps):
        g = spread * (i + 1) / steps
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x - g, y - g, w + 2 * g, h + 2 * g), r + g, r + g),
                    fillp((0, 0, 0), a / steps * (1 - i / steps) * 1.6))


def native_image(arr):
    """numpy RGBA -> Skia-native raster image (drawing numpy-backed images scaled is ~27x slower)."""
    img = skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kOpaque_AlphaType)
    surf = skia.Surface(img.width(), img.height())
    surf.getCanvas().drawImage(img, 0, 0)
    return surf.makeImageSnapshot()


def path(pts, close=False):
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if close:
        p.close()
    return p


# ===================================================================== layout
class Layout:
    def __init__(self, photo_w, photo_h):
        self.win = (24, 14, 2512, 1368)
        x0, y0, w, h = self.win
        self.title_h, self.menu_h, self.opt_h, self.status_h = 38, 32, 46, 26
        self.menu_y = y0 + self.title_h
        self.opt_y = self.menu_y + self.menu_h
        self.work_y = self.opt_y + self.opt_h
        self.bottom = y0 + h - self.status_h
        self.tool_w, self.panel_w = 58, 440
        self.panel_x = x0 + w - self.panel_w
        self.canvas = (x0 + self.tool_w, self.work_y, w - self.tool_w - self.panel_w, self.bottom - self.work_y)
        self.hist = (self.panel_x, self.work_y, self.panel_w, 190)
        self.props = (self.panel_x, self.work_y + 194, self.panel_w, 470)
        ly = self.props[1] + self.props[3] + 4
        self.layers = (self.panel_x, ly, self.panel_w, self.bottom - ly)
        cx, cy, cw, ch = self.canvas
        tab_h = 34
        s = min((cw - 90) / photo_w, (ch - tab_h - 70) / photo_h)
        self.zoom = s
        pw, ph = photo_w * s, photo_h * s
        self.photo = (cx + (cw - pw) / 2, cy + tab_h + (ch - tab_h - ph) / 2, pw, ph)
        self.menus = ["File", "Edit", "Image", "Layer", "Type", "Select", "Filter", "3D", "View", "Window", "Help"]
        self.tools = ["move", "marquee", "lasso", "wand", "crop", "eyedropper", "heal", "brush", "stamp", "eraser",
                      "gradient", "dodge", "pen", "type", "path", "shape", "hand", "zoom"]

    def tool_pos(self, name):
        i = self.tools.index(name)
        return (self.win[0] + self.tool_w / 2, self.work_y + 26 + i * 46)

    def swatch_pos(self):
        return (self.win[0] + self.tool_w / 2 - 6, self.work_y + 26 + len(self.tools) * 46 + 30)

    def layer_row(self, i, n):
        lx, ly, lw, lh = self.layers
        top = ly + 118
        return (lx, top + (n - 1 - i) * 66, lw, 66)

    def footer_icons(self):
        lx, ly, lw, lh = self.layers
        names = ["link", "fx", "mask", "adjust", "group", "new", "trash"]
        y = ly + lh - 22
        x_start = lx + lw - 30 - (len(names) - 1) * 46
        return {n: (x_start + i * 46, y) for i, n in enumerate(names)}

    def slider(self, name, tab, kind="Exposure"):
        px, py, pw, ph = self.props
        order = [sp[0] for sp in (MASK_SPECS if tab == "mask" else ADJ_SPECS.get(kind, []))]
        i = order.index(name) if name in order else 0
        return (px + 28, py + 156 + i * 88, pw - 56)          # track x, y, width

    def named_rect(self, name):
        x0, y0, w, h = self.win
        px, py, pw, ph = self.photo
        if name == "overview":
            return (0, 0, VW, VH)
        if name == "canvas":
            return (px - 30, py - 50, pw + 60, ph + 80)
        if name == "sky":
            return (px - 10, py - 30, pw + 20, ph * 0.58)
        if name == "layers":
            lx, ly, lw, lh = self.layers
            return (lx - 10, ly - 10, lw + 20, lh + 20)
        if name == "props":
            return (self.props[0] - 10, self.props[1] - 10, self.props[2] + 20, self.props[3] + 20)
        if name == "panels":
            return (self.panel_x - 10, self.work_y - 10, self.panel_w + 20, self.bottom - self.work_y + 20)
        if name == "work":     # right half of the photo + the properties panel
            return (px + pw * 0.3, py - 40, (self.panel_x + self.panel_w) - (px + pw * 0.3) + 10, ph + 60)
        if name == "work_layers":
            return (px + pw * 0.35, self.props[1] - 20, (self.panel_x + self.panel_w) - (px + pw * 0.35) + 10,
                    self.bottom - self.props[1] + 30)
        if name == "toolbar":
            return (x0 - 10, self.opt_y - 10, 900, 1000)
        return (0, 0, VW, VH)


# ===================================================================== state
ADJ_MENU = ["Solid Color...", "Gradient...", "Pattern...", "-", "Brightness/Contrast...", "Levels...", "Curves...",
            "Exposure...", "-", "Vibrance...", "Hue/Saturation...", "Color Balance...", "Black & White...",
            "Photo Filter...", "-", "Invert", "Posterize...", "Threshold...", "Gradient Map..."]


ADJ_SPECS = {   # name, label, min, max, default, format
    "Exposure": [("exposure", "Exposure:", -5, 5, 0.0, "{:+.2f}"), ("offset", "Offset:", -0.5, 0.5, 0.0, "{:.4f}"),
                 ("gamma", "Gamma Correction:", 0.01, 9.99, 1.0, "{:.2f}")],
    "Brightness/Contrast": [("brightness", "Brightness:", -150, 150, 0, "{:+.0f}"), ("contrast", "Contrast:", -50, 100, 0, "{:+.0f}")],
    "Vibrance": [("vibrance", "Vibrance:", -100, 100, 0, "{:+.0f}"), ("saturation", "Saturation:", -100, 100, 0, "{:+.0f}")],
    "Hue/Saturation": [("hue", "Hue:", -180, 180, 0, "{:+.0f}"), ("saturation", "Saturation:", -100, 100, 0, "{:+.0f}"),
                       ("lightness", "Lightness:", -100, 100, 0, "{:+.0f}")],
    "Black & White": [("reds", "Reds:", -200, 300, 40, "{:.0f}%"), ("blues", "Blues:", -200, 300, 20, "{:.0f}%")],
    "Photo Filter": [("warmth", "Warmth:", -100, 100, 0, "{:+.0f}"), ("filter_density", "Density:", 0, 100, 25, "{:.0f}%")],
}
MASK_SPECS = [("density", "Density:", 0, 100, 100, "{:.0f}%"), ("feather", "Feather:", 0, 250, 0, "{:.1f} px")]
REGIONS = ("sky", "foreground", "subject", "background", "left", "right", "edges", "all")


def spec_for(name, kind):
    for sp in (ADJ_SPECS.get(kind, []) + MASK_SPECS):
        if sp[0] == name:
            return sp
    return None


class State:
    def __init__(self):
        self.layers = [{"name": "Background", "kind": "image"}]
        self.sel = 0
        self.mask_selected = False
        self.adj_type = None
        self.adj = {}
        self.mask = None
        self.feather = 0.0
        self.density = 100.0
        self.props_tab = "doc"
        self.tool = "move"
        self.fg, self.bg = (0, 0, 0), (255, 255, 255)
        self.view_mask = False
        self.compare = None          # divider position (fraction of width showing BEFORE), None = off
        self.popup = None
        self.pressed = None
        self.brush_size = 300
        self.version = 0             # bumps whenever the picture changes
        self.done = set()            # ids of finished events


class Event:
    _next = 0

    def __init__(self, t0, dur, apply=None, update=None):
        self.t0, self.dur = t0, dur
        self._apply, self._update = apply, update
        Event._next += 1
        self.id = Event._next

    def run(self, st, t):
        if self._update:
            self._update(st, clamp((t - self.t0) / self.dur) if self.dur > 0 else 1.0)
        if self._apply and t >= self.t0 + self.dur:
            self._apply(st)


class Demo:
    """Compiles a demo script (steps + actions on spoken cues) into events, camera and cursor keyframes."""

    def __init__(self, script, durations, word_times):
        self.script = script
        self.base = po.load(PHOTO_DIR / script["photo"], exposure=float(script.get("photo_exposure", 0.0)))
        if script.get("look"):
            self.base = po.apply_look(self.base, script["look"])
        self.meta = next((m for m in POOL if m["file"] == script["photo"]), {"sky": 0.3, "subject": [0.5, 0.55, 0.2, 0.25]})
        self._adj_type = None
        self.H, self.W = self.base.shape[:2]
        self.L = Layout(self.W, self.H)
        self.steps, self.events, self.cam_keys, self.cur_keys, self.overlays, self.sfx = [], [], [], [], [], []
        self._slider_vals = {}
        self.cursor0 = (self.L.photo[0] + self.L.photo[2] * 0.62, self.L.photo[1] + self.L.photo[3] * 0.66)
        self.cur_keys.append((0.0, 0.0, self.cursor0, self.cursor0))
        self.cam_keys.append((0.0, "canvas"))
        t = 0.0
        for step, dur, words in zip(script["steps"], durations, word_times):
            self.steps.append({"start": t, "dur": dur, "title": step.get("title", ""), "chunks": self._chunks(words)})
            ptr = 0
            for a in step.get("actions", []):
                idx = self._match(a.get("cue", ""), words, ptr)
                at = t + (words[idx][1] - 0.15 if idx is not None else dur * 0.3)
                if idx is not None:
                    ptr = idx
                self._compile(a, max(t, at), t + dur)
            t += dur
        self.total = t
        self.events.sort(key=lambda e: e.t0)

    @staticmethod
    def _norm(w):
        return "".join(ch for ch in w.lower() if ch.isalnum())

    def _match(self, cue, words, start):
        toks = [self._norm(w) for w in cue.split() if self._norm(w)]
        ws = [self._norm(w[0]) for w in words]
        if not toks:
            return None
        for b in (start, 0):
            for i in range(b, len(ws) - len(toks) + 1):
                if ws[i:i + len(toks)] == toks:
                    return i
        return None

    @staticmethod
    def _chunks(words):
        out, cur = [], []
        for w in words:
            if cur and (len(cur) >= 5 or w[1] - cur[-1][2] > 0.25):
                out.append(cur)
                cur = []
            cur.append(w)
        if cur:
            out.append(cur)
        return out

    def _move(self, t, dur, target):
        self.cur_keys.append((t, dur, self.cur_keys[-1][3], target))
        return t + dur

    def _cam(self, t, name):
        self.cam_keys.append((t, name))

    def _sfx(self, t, kind, gain=1.0):
        self.sfx.append((t, kind, gain))

    def target_pos(self, name):
        L = self.L
        if name.startswith("tool."):
            return L.tool_pos(name.split(".", 1)[1])
        if name in ("layers.adjust", "layers.mask_btn", "layers.new"):
            return L.footer_icons()[{"layers.adjust": "adjust", "layers.mask_btn": "mask", "layers.new": "new"}[name]]
        if name == "layer.mask":
            x, y, w, h = L.layer_row(1, 2)
            return (x + 144 + 38, y + h / 2)
        if name == "layer.thumb":
            x, y, w, h = L.layer_row(1, 2)
            return (x + 102, y + h / 2)
        if name == "props.tab.mask":
            return (L.props[0] + 110, L.props[1] + 70)
        if name == "props.tab.adjust":
            return (L.props[0] + 50, L.props[1] + 70)
        if name == "swatch.fg":
            return L.swatch_pos()
        px, py, pw, ph = L.photo
        return (px + pw / 2, py + ph / 2)

    def slider_knob(self, name, value, tab, kind=None):
        kind = kind or self._adj_type or "Exposure"
        tx, ty, tw = self.L.slider(name, tab, kind)
        sp = spec_for(name, kind) or (name, name, 0, 1, 0, "{}")
        lo, hi = sp[2], sp[3]
        return (tx + tw * clamp((value - lo) / (hi - lo)), ty)

    def region_path(self, region):
        """A natural zig-zag brush path over an image region, in image pixel coords."""
        W, H = self.W, self.H
        def zig(y0, y1, x0=0.03, x1=0.97, rows=4):
            pts = []
            for k in range(rows):
                ry_ = y0 + (y1 - y0) * (k + 0.5) / rows
                xs = (x0, x1) if k % 2 == 0 else (x1, x0)
                for j in range(9):
                    u = j / 8
                    pts.append(((xs[0] + (xs[1] - xs[0]) * u) * W, (ry_ + 0.012 * math.sin(u * 9 + k)) * H))
            return pts
        sky = max(0.12, float(self.meta.get("sky") or 0.3))
        cx, cy, rx, ry = self.meta.get("subject") or [0.5, 0.55, 0.2, 0.25]
        if region == "sky":
            return zig(0.02, sky, rows=max(2, int(sky * 12)))
        if region == "foreground":
            return zig(min(0.9, sky + 0.08), 0.97, rows=4)
        if region == "subject":
            pts = []
            for k, f in enumerate((0.85, 0.6, 0.35)):
                for a in np.linspace(0, 6.28, 22):
                    pts.append((W * (cx + rx * f * math.cos(a + k)), H * (cy + ry * f * math.sin(a + k))))
            return pts
        if region == "background":
            pts = zig(0.03, 0.97, rows=6)
            out = [q for q in pts if ((q[0] / W - cx) / (rx * 1.3)) ** 2 + ((q[1] / H - cy) / (ry * 1.3)) ** 2 > 1]
            return out if len(out) > 4 else pts
        if region == "left":
            return zig(0.04, 0.96, 0.02, 0.48, rows=5)
        if region == "right":
            return zig(0.04, 0.96, 0.52, 0.98, rows=5)
        if region == "edges":
            pts = []
            for m in (0.03, 0.09):
                pts += [(W * m, H * m), (W * (1 - m), H * m), (W * (1 - m), H * (1 - m)), (W * m, H * (1 - m)), (W * m, H * m)]
            return pts
        return zig(0.03, 0.97, rows=6)

    def img_to_virtual(self, x, y):
        px, py, pw, ph = self.L.photo
        return (px + x / self.W * pw, py + y / self.H * ph)

    def _compile(self, a, t, step_end):
        do = a.get("do")
        if do == "focus":
            self._cam(t, a.get("target", "canvas"))
        elif do == "note":
            self.overlays.append({"kind": "note", "t0": t, "t1": min(step_end - 0.1, t + float(a.get("hold", 3.2))),
                                  "text": a.get("text", "")})
        elif do == "click":
            target = a["target"]
            default_focus = ("layers" if target.startswith("layer") else "props" if target.startswith("props")
                             else "toolbar" if target.startswith(("tool", "swatch")) else "canvas")
            self._cam(t, a.get("focus") or default_focus)
            tc = self._move(t, 0.5, self.target_pos(target))
            self._sfx(tc, "click")
            self.overlays.append({"kind": "ripple", "t0": tc, "pos": self.target_pos(target)})
            if a.get("mod"):
                self.overlays.append({"kind": "keys", "t0": t + 0.15, "t1": tc + 0.9, "keys": f'{a["mod"]} + Click'})
            fn = None
            if target == "layer.mask" and a.get("mod") == "Alt":
                def fn(st):
                    st.view_mask = not st.view_mask
                    st.version += 1
                self._cam(tc + 0.35, a.get("after", "canvas"))
            elif target == "layer.mask":
                def fn(st):
                    st.mask_selected, st.sel = True, 1
            elif target == "layer.thumb":
                def fn(st):
                    st.mask_selected = False
            elif target.startswith("props.tab."):
                tab = target.split(".")[-1]

                def fn(st, tab=tab):
                    st.props_tab = tab
            elif target.startswith("tool."):
                tool = target.split(".")[1]

                def fn(st, tool=tool):
                    st.tool = tool
            if fn:
                self.events.append(Event(tc, 0, apply=fn))
        elif do == "pick":
            anchor = self.target_pos(a["target"])
            self._cam(t, "layers")
            tc = self._move(t, 0.5, anchor)
            self._sfx(tc, "click")
            self.overlays.append({"kind": "ripple", "t0": tc, "pos": anchor})
            item = a.get("item", "Exposure...")
            idx = ADJ_MENU.index(item) if item in ADJ_MENU else 7
            menu_x, menu_y = anchor[0] - 300, anchor[1] - 14 - (len(ADJ_MENU) * 34 + 16)
            item_pos = (menu_x + 130, menu_y + 8 + idx * 34 + 17)

            def open_popup(st):
                st.popup = {"x": menu_x, "y": menu_y, "hover": -1}
            self.events.append(Event(tc, 0, apply=open_popup))
            self._sfx(tc + 0.03, "pop", 0.6)
            tp = self._move(tc + 0.2, 0.6, item_pos)

            def hover(st, u, idx=idx):
                if st.popup:
                    st.popup["hover"] = idx if u > 0.55 else -1
            self.events.append(Event(tc + 0.2, 0.6, update=hover))
            self._sfx(tp + 0.12, "click")
            self.overlays.append({"kind": "ripple", "t0": tp + 0.12, "pos": item_pos})
            name = item.replace("...", "")
            self._adj_type = name if name in ADJ_SPECS else "Exposure"
            H, W = self.H, self.W

            def add_layer(st, name=self._adj_type):
                st.popup = None
                st.adj_type = name
                st.adj = {sp[0]: sp[4] for sp in ADJ_SPECS[name]}
                st.layers.append({"name": f"{name} 1", "kind": "adjust"})
                st.sel, st.mask_selected = 1, False
                st.mask = np.ones((H, W), np.float32)
                st.props_tab = "adjust"
                st.version += 1
            self.events.append(Event(tp + 0.14, 0, apply=add_layer))
            self._cam(tp + 0.45, a.get("after", "work"))
        elif do == "slider":
            name, to = a["target"], float(a["to"])
            tab = "mask" if name in ("feather", "density") else "adjust"
            kind = self._adj_type or "Exposure"
            sp = spec_for(name, kind)
            if sp is None:
                return
            to = max(sp[2], min(sp[3], to))
            self._cam(t, a.get("focus", "work"))
            v0 = self._slider_vals.get((kind, name), sp[4])
            self._slider_vals[(kind, name)] = to
            knob0, knob1 = self.slider_knob(name, v0, tab, kind), self.slider_knob(name, to, tab, kind)
            tc = self._move(t, 0.5, knob0)
            self._sfx(tc, "click", 0.7)
            drag = float(a.get("dur", 1.3))
            self.cur_keys.append((tc + 0.08, drag, knob0, knob1))

            def upd(st, u, name=name, v0=v0, to=to):
                v = v0 + (to - v0) * ease_io(u)
                st.pressed = name if u < 1 else None
                if name == "feather":
                    st.feather = v
                elif name == "density":
                    st.density = v
                else:
                    st.adj[name] = v
                st.version += 1
            self.events.append(Event(tc + 0.08, drag, update=upd))
            for k in range(10):
                self._sfx(tc + 0.1 + k * drag / 10, "tick", 0.3)
        elif do == "key":
            keys = a["keys"]
            self.overlays.append({"kind": "keys", "t0": t, "t1": t + 1.4, "keys": keys})
            self._sfx(t + 0.05, "key")
            if "+" in keys:
                self._sfx(t + 0.15, "key")
            k = keys.lower().replace(" ", "")
            fn = None
            if k == "ctrl+i":
                def fn(st):
                    if st.mask is not None:
                        st.mask = 1 - st.mask
                        st.version += 1
                self._cam(t + 0.25, a.get("focus", "work_layers"))
            elif k == "b":
                def fn(st):
                    st.tool = "brush"
                self._cam(t, a.get("focus", "toolbar"))
            elif k == "d":
                def fn(st):
                    st.fg, st.bg = ((255, 255, 255), (0, 0, 0)) if st.mask is not None else ((0, 0, 0), (255, 255, 255))
            elif k == "x":
                def fn(st):
                    st.fg, st.bg = st.bg, st.fg
            elif k in ("]", "["):
                def fn(st, k=k):
                    st.brush_size = st.brush_size * (1.25 if k == "]" else 0.8)
            elif k == "\\":
                def fn(st):
                    st.view_mask = not st.view_mask
                    st.version += 1
            if fn:
                self.events.append(Event(t + 0.15, 0, apply=fn))
        elif do == "paint":
            pts = self.region_path(a.get("region", "sky"))
            value = float(a.get("value", 1.0))
            self._cam(t, a.get("focus", "canvas"))
            tc = self._move(t, 0.5, self.img_to_virtual(*pts[0]))
            stroke_dur = float(a.get("dur", 3.0))
            vpts = [self.img_to_virtual(*p) for p in pts]
            seg = stroke_dur / (len(vpts) - 1)
            for i in range(len(vpts) - 1):
                self.cur_keys.append((tc + i * seg, seg, vpts[i], vpts[i + 1], "lin"))
            radius = float(a.get("size", 300)) / 2
            hardness = float(a.get("hardness", 0.0))
            holder = {}

            def upd(st, u, pts=pts, value=value, radius=radius, hardness=hardness, holder=holder):
                if st.mask is None:
                    return
                if holder.get("st") is not st:
                    holder["st"], holder["done"] = st, 0
                n = 1 + int(u * (len(pts) - 1))
                if n > holder["done"]:
                    lo = max(0, holder["done"] - 1)
                    po.stamp_stroke(st.mask, pts[lo:n + 1], radius, value, hardness)
                    holder["done"] = n
                    st.version += 1
            self.events.append(Event(tc, stroke_dur, update=upd))
            self.overlays.append({"kind": "painting", "t0": tc, "t1": tc + stroke_dur})
            self._sfx(tc, "brush:%.2f" % stroke_dur, 1.0)
        elif do == "compare":
            dur = float(a.get("dur", 2.4))
            self._cam(t, "canvas")

            def upd(st, u):
                st.compare = 1 - 0.5 * ease_io(min(1.0, u * 1.4))   # just slides between two finished images

            def end(st):
                st.compare = None
            self.events.append(Event(t, dur, update=upd))
            self.events.append(Event(step_end - 0.05, 0, apply=end))
            self._sfx(t, "whoosh", 0.8)

    def camera_rect(self, t):
        keys = sorted(self.cam_keys, key=lambda k: k[0])
        val = target = self.L.named_rect(keys[0][1])
        prev_t = keys[0][0]
        for kt, name in keys[1:]:
            if kt > t:
                break
            u = ease_io((kt - prev_t) / 0.8)
            val = tuple(val[i] + (target[i] - val[i]) * u for i in range(4))
            target = self.L.named_rect(name)
            prev_t = kt
        u = ease_io((t - prev_t) / 0.8)
        return tuple(val[i] + (target[i] - val[i]) * u for i in range(4))

    def cursor_pos(self, t):
        pos = self.cursor0
        for key in self.cur_keys:
            kt, dur, p0, p1 = key[:4]
            if t < kt:
                break
            u = 1.0 if dur <= 0 else clamp((t - kt) / dur)
            if len(key) > 4:
                pos = (p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u)
            else:
                e = ease_io(u)
                arc = math.sin(math.pi * e) * 0.06 * math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                pos = (p0[0] + (p1[0] - p0[0]) * e, p0[1] + (p1[1] - p0[1]) * e - arc)
        return pos

    def step_at(self, t):
        si = 0
        for i, s in enumerate(self.steps):
            if t >= s["start"]:
                si = i
        return si


# ===================================================================== renderer
class ScreenRenderer:
    def __init__(self, demo, width=2160, height=3840, fps=30, handle="", app_name="Lumen Photo"):
        self.d = demo
        self.w, self.h, self.fps = width, height, fps
        self.k = width / DW
        self.handle = handle
        self.app = app_name
        info = skia.ImageInfo.Make(width, height, skia.kBGRA_8888_ColorType, skia.kPremul_AlphaType)
        self.buf = np.zeros((height, width, 4), np.uint8)
        self.surface = skia.Surface.MakeRasterDirect(info, self.buf)
        self.c = self.surface.getCanvas()
        self.text = Text()
        self.wall = self._wallpaper()
        self.bg = self._background()
        self.state = None
        self.img_cache = {}
        self.adj_cache = (None, None)
        self.final_img = None
        self.cur_array = demo.base
        self.before_img = native_image(po.to_rgba8(demo.base))
        self._thumb_img = native_image(po.to_rgba8(po.thumb(demo.base, 152)))
        self._q = 1.0
        self.card_px_w = int(CARD_W * self.k)
        self.card_surf = skia.Surface(self.card_px_w, int(1220 * self.k))   # desktop is drawn here, then clipped once
        self.card_shadow = self._shadow_image()

    def _wallpaper(self):
        s = 2
        surf = skia.Surface(VW * s, VH * s)
        c = surf.getCanvas()
        c.scale(s, s)
        c.drawPaint(skia.Paint(Shader=skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(VW, VH)], [col((10, 32, 74)), col((22, 70, 150)), col((8, 22, 54))])))
        for (x, y, r, rgb, a) in ((1500, 820, 900, (70, 150, 255), 0.55), (1150, 700, 620, (130, 90, 255), 0.35),
                                  (1800, 980, 520, (40, 200, 255), 0.35)):
            c.drawPaint(skia.Paint(Shader=skia.GradientShader.MakeRadial(skia.Point(x, y), r, [col(rgb, a), col(rgb, 0)])))
        for i in range(5):
            p = skia.Path()
            p.addOval(skia.Rect.MakeXYWH(-520, -150, 1040, 300))
            m = skia.Matrix()
            m.setRotate(math.degrees(-0.6 + i * 0.32))
            m.postTranslate(1450, 760)
            p.transform(m)
            c.drawPath(p, skia.Paint(AntiAlias=True, Color=col((120, 190, 255), 0.10 + 0.03 * i)))
        return surf.makeImageSnapshot()

    def backdrop(self, card_h, card_y):
        """Background with the card's soft shadow baked in; re-rendered only when the card size changes."""
        key = (card_h, round(card_y))
        if getattr(self, "_backdrop_key", None) != key:
            surf = skia.Surface(self.w, self.h)
            c = surf.getCanvas()
            k = self.k
            c.drawImage(self.bg, 0, 0)
            c.drawImageRect(self.card_shadow, skia.Rect.MakeXYWH((CARD_X - 70) * k, (card_y - 40) * k, (CARD_W + 140) * k,
                                                                 (card_h + 160) * k), skia.SamplingOptions(skia.FilterMode.kLinear))
            self._backdrop, self._backdrop_key = surf.makeImageSnapshot(), key
        return self._backdrop

    def _shadow_image(self):
        """Blurred card shadow rendered once at low resolution, stretched to the card size every frame."""
        w, h = 280, 340
        surf = skia.Surface(w, h)
        c = surf.getCanvas()
        c.clear(skia.ColorTRANSPARENT)
        p = skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.75))
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 16))
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(40, 50, w - 80, h - 80), 18, 18), p)
        return surf.makeImageSnapshot()

    def _background(self):
        surf = skia.Surface(self.w, self.h)
        c = surf.getCanvas()
        k = self.k
        c.drawPaint(skia.Paint(Shader=skia.GradientShader.MakeLinear(
            [skia.Point(0, 0), skia.Point(0, self.h)], [col((14, 16, 24)), col((20, 26, 44)), col((10, 12, 18))])))
        for (x, y, r, rgb, a) in ((200, 500, 700, (40, 110, 255), 0.35), (900, 1500, 800, (120, 70, 255), 0.25)):
            c.drawPaint(skia.Paint(Shader=skia.GradientShader.MakeRadial(
                skia.Point(x * k, y * k), r * k, [col(rgb, a), col(rgb, 0)])))
        return surf.makeImageSnapshot()

    # ----------------------------------------------------------- state replay
    def state_at(self, t):
        if self.state is None or t < getattr(self, "_state_t", -1):
            self.state = State()
        st = self.state
        for e in self.d.events:
            if e.t0 > t:
                break
            if e.id in st.done:
                continue
            e.run(st, t)
            if t >= e.t0 + e.dur:
                st.done.add(e.id)
        self._state_t = t
        return st

    def compute_final(self):
        st = State()
        for e in self.d.events:
            e.run(st, self.d.total + 1)
        st.compare, st.view_mask = None, False
        self.final_img = self.composite(st)
        self.img_cache = {}

    def composite(self, st):
        key = (st.version, round(st.feather, 1), st.view_mask, len(st.layers))
        if self.img_cache.get("key") == key:
            return self.img_cache["img"]
        if st.view_mask and st.mask is not None:
            arr = po.mask_rgba8(po.feather(st.mask, st.feather))
        else:
            if len(st.layers) > 1:
                adj_key = (st.adj_type, tuple(sorted((k_, round(v, 3)) for k_, v in st.adj.items())))
                if self.adj_cache[0] != adj_key:
                    self.adj_cache = (adj_key, po.adjust_layer(self.d.base, st.adj_type, st.adj))
                m = po.feather(st.mask, st.feather) * (st.density / 100) if st.mask is not None else None
                out = po.composite(self.d.base, self.adj_cache[1], m)
            else:
                out = self.d.base
            self.cur_array = out
            arr = po.to_rgba8(out)
        img = native_image(arr)
        self.img_cache = {"key": key, "img": img}
        return img

    # ----------------------------------------------------------- desktop drawing (virtual coords)
    def T(self, c, s, x, y, size, rgb=C_TEXT, font=SANS_REG, align="left", a=1.0):
        self.text.draw(c, s, x, y, size, rgb, font, align, a, self._q)

    def draw_desktop(self, c, st, t):
        L = self.d.L
        c.drawImageRect(self.wall, skia.Rect.MakeWH(VW, VH), skia.SamplingOptions(skia.FilterMode.kLinear))
        x0, y0, w, h = L.win
        c.save()
        c.clipRect(skia.Rect.MakeXYWH(x0, y0, w, h))   # plain clip: rounded AA clips make every fill slow at 4K
        rect(c, x0, y0, w, h, C_WIN)
        self.draw_titlebar(c)
        self.draw_menubar(c)
        self.draw_options(c, st)
        self.draw_toolbar(c, st)
        self.draw_canvas(c, st)
        self.draw_histogram(c, st)
        self.draw_props(c, st)
        self.draw_layers(c, st)
        self.draw_status(c)
        c.restore()
        rr_stroke(c, x0, y0, w, h, 9, (70, 70, 78), 0.8, 1)
        self.draw_taskbar(c)
        if st.popup:
            self.draw_popup(c, st)

    def draw_titlebar(self, c):
        L = self.d.L
        x0, y0, w, h = L.win
        rect(c, x0, y0, w, L.title_h, (24, 24, 27))
        rr(c, x0 + 14, y0 + 9, 20, 20, 5, (49, 168, 255))
        self.T(c, "Lu", x0 + 24, y0 + 24, 11, (8, 30, 60), SANS_BOLD, "center")
        self.T(c, f"{self.app}  —  {self.d.script.get('file_name', 'photo.jpg')}", x0 + 44, y0 + 25, 13, (200, 200, 206))
        for i, sym in enumerate(("min", "max", "close")):
            cx = x0 + w - 23 - (2 - i) * 46
            g = (210, 210, 215)
            if sym == "min":
                line(c, cx - 5, y0 + 19, cx + 5, y0 + 19, g, 1, 1)
            elif sym == "max":
                c.drawRect(skia.Rect.MakeXYWH(cx - 5, y0 + 14, 10, 10), strokep(g, 1, 1))
            else:
                line(c, cx - 5, y0 + 14, cx + 5, y0 + 24, g, 1, 1)
                line(c, cx + 5, y0 + 14, cx - 5, y0 + 24, g, 1, 1)

    def draw_menubar(self, c):
        L = self.d.L
        x0, y0, w, h = L.win
        rect(c, x0, L.menu_y, w, L.menu_h, (36, 36, 40))
        x = x0 + 16
        for m in L.menus:
            self.T(c, m, x, L.menu_y + 21, 13, (215, 215, 220))
            x += self.text.width(m, SANS_REG, 13) + 26
        line(c, x0, L.menu_y + L.menu_h, x0 + w, L.menu_y + L.menu_h, C_LINE, 1, 1)

    def draw_options(self, c, st):
        L = self.d.L
        x0, y0, w, h = L.win
        rect(c, x0, L.opt_y, w, L.opt_h, (42, 42, 46))
        y = L.opt_y + L.opt_h / 2
        self.draw_tool_icon(c, st.tool, x0 + 30, y, (230, 230, 235), 1.0)
        line(c, x0 + 58, L.opt_y + 8, x0 + 58, L.opt_y + L.opt_h - 8, (70, 70, 76), 1, 1)
        x = x0 + 76
        if st.tool == "brush":
            c.drawCircle(x + 14, y, 11, strokep((220, 220, 225), 1, 1.2))
            self.T(c, f"{int(st.brush_size)}", x + 34, y + 5, 13, C_TEXT)
            x += 90
            for label, val in (("Mode:", "Normal"), ("Opacity:", "100%"), ("Flow:", "100%"), ("Smoothing:", "10%")):
                self.T(c, label, x, y + 5, 13, C_DIM)
                x += self.text.width(label, SANS_REG, 13) + 8
                rr(c, x, y - 12, self.text.width(val, SANS_REG, 13) + 26, 24, 4, (30, 30, 33))
                self.T(c, val, x + 9, y + 5, 13, C_TEXT)
                x += self.text.width(val, SANS_REG, 13) + 46
        else:
            for label in ("Auto-Select:", "Layer", "Show Transform Controls"):
                self.T(c, label, x, y + 5, 13, C_DIM if label.endswith(":") else C_TEXT)
                x += self.text.width(label, SANS_REG, 13) + 22
        line(c, x0, L.opt_y + L.opt_h, x0 + w, L.opt_y + L.opt_h, C_LINE, 1, 1)

    def draw_tool_icon(self, c, tool, x, y, rgb, a):
        p = strokep(rgb, a, 1.6)
        if tool == "move":
            c.drawPath(path([(x - 6, y - 8), (x - 6, y + 7), (x - 2, y + 3), (x + 2, y + 10), (x + 4, y + 9), (x, y + 2),
                             (x + 5, y + 2)], True), p)
        elif tool == "marquee":
            c.drawRect(skia.Rect.MakeXYWH(x - 8, y - 7, 16, 14), strokep(rgb, a, 1.4, dash=(3, 2.5)))
        elif tool == "lasso":
            c.drawOval(skia.Rect.MakeXYWH(x - 8, y - 7, 16, 11), p)
            line(c, x - 4, y + 3, x - 6, y + 9, rgb, a, 1.6)
        elif tool == "wand":
            line(c, x - 7, y + 7, x + 3, y - 3, rgb, a, 2)
            for dx, dy in ((5, -7), (8, -2), (1, -9)):
                c.drawCircle(x + dx, y + dy, 1.2, fillp(rgb, a))
        elif tool == "crop":
            c.drawPath(path([(x - 4, y - 9), (x - 4, y + 5), (x + 9, y + 5)]), p)
            c.drawPath(path([(x - 9, y - 4), (x + 5, y - 4), (x + 5, y + 9)]), p)
        elif tool == "eyedropper":
            line(c, x - 7, y + 7, x + 4, y - 4, rgb, a, 2.2)
            c.drawCircle(x + 6, y - 6, 3, fillp(rgb, a))
        elif tool == "heal":
            rr_stroke(c, x - 9, y - 4, 18, 8, 4, rgb, a, 1.5)
            line(c, x, y - 4, x, y + 4, rgb, a, 1)
        elif tool == "brush":
            line(c, x + 7, y - 8, x - 1, y + 2, rgb, a, 2.4)
            c.drawPath(path([(x - 1, y + 1), (x - 6, y + 4), (x - 8, y + 9), (x - 3, y + 7), (x + 1, y + 3)], True), fillp(rgb, a))
        elif tool == "stamp":
            rr_stroke(c, x - 7, y + 1, 14, 6, 2, rgb, a, 1.5)
            line(c, x, y - 8, x, y + 1, rgb, a, 2.4)
        elif tool == "eraser":
            c.drawPath(path([(x - 8, y + 4), (x, y - 6), (x + 8, y + 1), (x + 1, y + 9)], True), p)
        elif tool == "gradient":
            c.drawRect(skia.Rect.MakeXYWH(x - 8, y - 7, 16, 14), skia.Paint(Shader=skia.GradientShader.MakeLinear(
                [skia.Point(x - 8, y), skia.Point(x + 8, y)], [col(rgb, a), col(rgb, 0.05)])))
        elif tool == "dodge":
            c.drawCircle(x - 2, y - 2, 5, p)
            line(c, x + 2, y + 2, x + 7, y + 8, rgb, a, 2)
        elif tool == "pen":
            c.drawPath(path([(x, y - 9), (x + 6, y + 3), (x, y + 8), (x - 6, y + 3)], True), p)
        elif tool == "type":
            self.T(c, "T", x, y + 6, 17, rgb, SANS_BOLD, "center", a)
        elif tool == "path":
            c.drawPath(path([(x - 6, y - 8), (x - 6, y + 7), (x - 2, y + 3), (x + 4, y + 6)], True), fillp(rgb, a))
        elif tool == "shape":
            c.drawRect(skia.Rect.MakeXYWH(x - 7, y - 7, 14, 14), fillp(rgb, a * 0.9))
        elif tool == "hand":
            rr_stroke(c, x - 6, y - 5, 12, 13, 4, rgb, a, 1.5)
            for dx in (-4, -1, 2, 5):
                line(c, x + dx, y - 9, x + dx, y - 4, rgb, a, 1.4)
        elif tool == "zoom":
            c.drawCircle(x - 2, y - 2, 6, p)
            line(c, x + 2, y + 2, x + 8, y + 8, rgb, a, 2)

    def draw_toolbar(self, c, st):
        L = self.d.L
        x0 = L.win[0]
        rect(c, x0, L.work_y, L.tool_w, L.bottom - L.work_y, (38, 38, 42))
        line(c, x0 + L.tool_w, L.work_y, x0 + L.tool_w, L.bottom, C_LINE, 1, 1)
        for name in L.tools:
            tx, ty = L.tool_pos(name)
            if name == st.tool:
                rr(c, tx - 19, ty - 18, 38, 36, 5, (70, 70, 78))
            self.draw_tool_icon(c, name, tx, ty, (225, 225, 230) if name == st.tool else (185, 185, 192), 1.0)
        sx, sy = L.swatch_pos()
        rect(c, sx + 9, sy + 9, 22, 22, st.bg)
        c.drawRect(skia.Rect.MakeXYWH(sx + 9, sy + 9, 22, 22), strokep((120, 120, 128), 1, 1))
        rect(c, sx - 4, sy - 4, 22, 22, st.fg)
        c.drawRect(skia.Rect.MakeXYWH(sx - 4, sy - 4, 22, 22), strokep((160, 160, 168), 1, 1))

    def draw_canvas(self, c, st):
        L = self.d.L
        cx, cy, cw, ch = L.canvas
        rect(c, cx, cy, cw, ch, C_CANVAS)
        rect(c, cx, cy, cw, 34, (36, 36, 40))
        layer = st.layers[st.sel]["name"] if st.layers else "Background"
        tab = f"{self.d.script.get('file_name', 'photo.jpg')} @ {L.zoom * 100:.1f}% ({layer}{', Layer Mask' if st.mask_selected else ''}/8)   ×"
        tw = self.text.width(tab, SANS_REG, 12.5) + 32
        rect(c, cx, cy, tw, 34, (52, 52, 57))
        rect(c, cx, cy, tw, 2, C_ACCENT)
        self.T(c, tab, cx + 14, cy + 22, 12.5, (225, 225, 230))
        px, py, pw, ph = L.photo
        rect(c, px + 2, py + 5, pw, ph, (14, 14, 16))
        img = self.composite(st)
        samp = skia.SamplingOptions(skia.FilterMode.kLinear)
        dst = skia.Rect.MakeXYWH(px, py, pw, ph)
        if st.compare is not None and self.final_img is not None:
            split = px + pw * st.compare
            c.save()
            c.clipRect(skia.Rect.MakeLTRB(px, py, split, py + ph))
            c.drawImageRect(self.before_img, dst, samp)
            c.restore()
            c.save()
            c.clipRect(skia.Rect.MakeLTRB(split, py, px + pw, py + ph))
            c.drawImageRect(self.final_img, dst, samp)
            c.restore()
            line(c, split, py, split, py + ph, (255, 255, 255), 1, 3)
            cy2 = py + ph / 2
            c.drawCircle(split, cy2, 18, fillp((255, 255, 255)))
            c.drawPath(path([(split - 4, cy2 - 6), (split - 10, cy2), (split - 4, cy2 + 6)]), strokep((40, 40, 45), 1, 2))
            c.drawPath(path([(split + 4, cy2 - 6), (split + 10, cy2), (split + 4, cy2 + 6)]), strokep((40, 40, 45), 1, 2))
            for label, lx, al in (("BEFORE", px + 18, "left"), ("AFTER", px + pw - 18, "right")):
                w = self.text.width(label, SANS_BOLD, 15) + 24
                bx = lx if al == "left" else lx - w
                rr(c, bx, py + 16, w, 30, 6, (0, 0, 0), 0.6)
                self.T(c, label, bx + 12, py + 36, 15, (255, 255, 255), SANS_BOLD)
        else:
            c.drawImageRect(img, dst, samp)

    def draw_panel_header(self, c, x, y, w, tabs, active=0):
        rect(c, x, y, w, 30, C_PANEL_HDR)
        tx = x + 10
        for i, tname in enumerate(tabs):
            tw = self.text.width(tname, SANS, 12.5) + 22
            if i == active:
                rect(c, tx, y, tw, 30, C_PANEL)
                self.T(c, tname, tx + 11, y + 20, 12.5, (235, 235, 240), SANS)
            else:
                self.T(c, tname, tx + 11, y + 20, 12.5, C_DIM, SANS)
            tx += tw
        for i in range(3):
            line(c, x + w - 24, y + 11 + i * 4, x + w - 12, y + 11 + i * 4, C_DIM, 1, 1)

    def draw_histogram(self, c, st):
        x, y, w, h = self.d.L.hist
        rect(c, x, y, w, h, C_PANEL)
        self.draw_panel_header(c, x, y, w, ["Histogram", "Navigator"])
        arr = self.cur_array
        if getattr(self, "_hist_v", None) != st.version:
            chans = [np.clip(arr[::8, ::8, i], 0, 1) for i in range(3)]
            hs = [np.histogram(ch_, bins=96, range=(0, 1))[0].astype(np.float32) for ch_ in chans]
            mx = max(1.0, max(float(hh[2:-2].max()) for hh in hs))
            self._hist = [np.clip(hh / mx, 0, 1) for hh in hs]
            self._hist_v = st.version
        gx, gy, gw, gh = x + 16, y + 44, w - 32, h - 60
        rect(c, gx, gy, gw, gh, (30, 30, 33))
        for hh, cc in zip(self._hist, [(255, 70, 70), (70, 230, 110), (80, 140, 255)]):
            pts = [(gx, gy + gh)] + [(gx + gw * i / (len(hh) - 1), gy + gh - gh * 0.95 * float(v)) for i, v in enumerate(hh)] + [(gx + gw, gy + gh)]
            c.drawPath(path(pts, True), skia.Paint(AntiAlias=True, Color=col(cc, 0.42), BlendMode=skia.BlendMode.kPlus))

    def draw_slider(self, c, name, label, value, fmt, tab, st):
        tx, ty, tw = self.d.L.slider(name, tab, st.adj_type or "Exposure")
        self.T(c, label, tx, ty - 22, 13, C_TEXT)
        rr(c, tx + tw - 78, ty - 40, 78, 26, 4, (30, 30, 33))
        self.T(c, fmt.format(value), tx + tw - 10, ty - 22, 13, C_TEXT, SANS_REG, "right")
        rr(c, tx, ty - 2, tw, 4, 2, (78, 78, 86))
        kx, ky = self.d.slider_knob(name, value, tab, st.adj_type)
        sp = spec_for(name, st.adj_type) or (name, "", 0, 1, 0, "")
        zx = tx + tw * clamp((max(sp[2], min(sp[3], 0)) - sp[2]) / (sp[3] - sp[2]))
        rr(c, min(zx, kx), ty - 2, abs(kx - zx), 4, 2, C_ACCENT2 if st.pressed == name else (150, 150, 160))
        c.drawPath(path([(kx, ty - 9), (kx + 8, ty + 6), (kx - 8, ty + 6)], True),
                   fillp((255, 255, 255) if st.pressed == name else (225, 225, 230)))

    def draw_props(self, c, st):
        x, y, w, h = self.d.L.props
        rect(c, x, y, w, h, C_PANEL)
        self.draw_panel_header(c, x, y, w, ["Properties", "Adjustments", "Libraries"])
        if st.props_tab == "doc" or len(st.layers) < 2:
            self.T(c, "Document", x + 20, y + 62, 14, C_TEXT, SANS)
            for i, (k_, v) in enumerate((("W", f"{self.d.W} px"), ("H", f"{self.d.H} px"), ("Resolution", "300 ppi"),
                                         ("Mode", "RGB Color  8 bit"))):
                self.T(c, k_, x + 20, y + 104 + i * 34, 13, C_DIM)
                self.T(c, v, x + 150, y + 104 + i * 34, 13, C_TEXT)
            return
        ay = y + 70
        for tab, cx_ in (("adjust", x + 50), ("mask", x + 110)):
            if st.props_tab == tab:
                rr(c, cx_ - 22, ay - 18, 44, 36, 5, (66, 66, 74))
            g = (220, 220, 225)
            if tab == "adjust":
                c.drawCircle(cx_, ay, 10, strokep(g, 1, 1.5))
                c.drawArc(skia.Rect.MakeXYWH(cx_ - 10, ay - 10, 20, 20), 90, 180, True, fillp(g))
            else:
                rr_stroke(c, cx_ - 11, ay - 9, 22, 18, 2, g, 1, 1.5)
                c.drawCircle(cx_, ay, 5, fillp(g))
        self.T(c, (st.adj_type or "Adjustment") if st.props_tab == "adjust" else "Layer Mask", x + 150, ay + 5, 14, C_TEXT, SANS)
        line(c, x + 14, y + 100, x + w - 14, y + 100, (60, 60, 66), 1, 1)
        if st.props_tab == "adjust":
            for sp in ADJ_SPECS.get(st.adj_type, []):
                self.draw_slider(c, sp[0], sp[1], st.adj.get(sp[0], sp[4]), sp[5], "adjust", st)
        else:
            self.draw_slider(c, "density", "Density:", st.density, "{:.0f}%", "mask", st)
            self.draw_slider(c, "feather", "Feather:", st.feather, "{:.1f} px", "mask", st)
            bx = x + 28
            for b in ("Select and Mask...", "Color Range...", "Invert"):
                bw = self.text.width(b, SANS_REG, 12.5) + 28
                rr(c, bx, y + 330, bw, 30, 4, (58, 58, 64))
                self.T(c, b, bx + 14, y + 350, 12.5, C_TEXT)
                bx += bw + 12

    def draw_layers(self, c, st):
        L = self.d.L
        x, y, w, h = L.layers
        rect(c, x, y, w, h, C_PANEL)
        self.draw_panel_header(c, x, y, w, ["Layers", "Channels", "Paths"])
        self.T(c, "Normal", x + 18, y + 60, 13, C_TEXT)
        self.T(c, "Opacity:  100%", x + w - 150, y + 60, 13, C_TEXT)
        self.T(c, "Lock:", x + 18, y + 96, 13, C_DIM)
        self.T(c, "Fill:  100%", x + w - 150, y + 96, 13, C_TEXT)
        line(c, x, y + 114, x + w, y + 114, (58, 58, 64), 1, 1)
        n = len(st.layers)
        for i, lay in enumerate(st.layers):
            rx, ry, rw, rh = L.layer_row(i, 2) if n > 1 else L.layer_row(1, 2)
            selected = i == st.sel
            rect(c, rx, ry, rw, rh, (62, 82, 112) if selected else C_PANEL)
            line(c, rx, ry + rh, rx + rw, ry + rh, (34, 34, 38), 1, 1)
            cy_ = ry + rh / 2
            c.drawOval(skia.Rect.MakeXYWH(rx + 18, cy_ - 6, 22, 12), strokep((210, 210, 215), 1, 1.4))
            c.drawCircle(rx + 29, cy_, 3, fillp((210, 210, 215)))
            if lay["kind"] == "image":
                tx_ = rx + 64
                c.drawImageRect(self._thumb_img, skia.Rect.MakeXYWH(tx_, cy_ - 25, 76, 51), skia.SamplingOptions(skia.FilterMode.kLinear))
                c.drawRect(skia.Rect.MakeXYWH(tx_, cy_ - 25, 76, 51), strokep((20, 20, 22), 1, 1))
                self.T(c, lay["name"], tx_ + 92, cy_ + 5, 13.5, C_TEXT)
                rr_stroke(c, rx + rw - 34, cy_ - 4, 12, 10, 2, C_DIM, 1, 1.2)
                c.drawArc(skia.Rect.MakeXYWH(rx + rw - 32, cy_ - 10, 8, 10), 180, 180, False, strokep(C_DIM, 1, 1.2))
            else:
                ix = rx + 64
                rr(c, ix, cy_ - 22, 44, 44, 4, (230, 230, 235))
                c.drawCircle(ix + 22, cy_, 13, strokep((40, 40, 45), 1, 1.6))
                c.drawArc(skia.Rect.MakeXYWH(ix + 9, cy_ - 13, 26, 26), 90, 180, True, fillp((40, 40, 45)))
                line(c, ix + 54, cy_ - 8, ix + 54, cy_ + 8, (150, 150, 160), 1, 1.2)
                mx = ix + 64
                if st.mask is not None:
                    if getattr(self, "_mthumb_v", None) != st.version:
                        small = st.mask[::max(1, self.d.H // 51), ::max(1, self.d.W // 76)]
                        self._mthumb = native_image(po.mask_rgba8(small))
                        self._mthumb_v = st.version
                    c.drawImageRect(self._mthumb, skia.Rect.MakeXYWH(mx, cy_ - 25, 76, 51), skia.SamplingOptions(skia.FilterMode.kLinear))
                    c.drawRect(skia.Rect.MakeXYWH(mx, cy_ - 25, 76, 51), strokep((20, 20, 22), 1, 1))
                if st.mask_selected and selected:
                    c.drawRect(skia.Rect.MakeXYWH(mx - 3, cy_ - 28, 82, 57), strokep((255, 255, 255), 1, 2))
                self.T(c, lay["name"], mx + 92, cy_ + 5, 13.5, C_TEXT)
        for name, (fx, fy) in L.footer_icons().items():
            g = (190, 190, 198)
            if name == "mask":
                rr_stroke(c, fx - 9, fy - 7, 18, 14, 2, g, 1, 1.3)
                c.drawCircle(fx, fy, 4, fillp(g))
            elif name == "adjust":
                c.drawCircle(fx, fy, 8, strokep(g, 1, 1.3))
                c.drawArc(skia.Rect.MakeXYWH(fx - 8, fy - 8, 16, 16), 90, 180, True, fillp(g))
            elif name == "new":
                rr_stroke(c, fx - 8, fy - 8, 16, 16, 2, g, 1, 1.3)
                line(c, fx, fy - 4, fx, fy + 4, g, 1, 1.3)
                line(c, fx - 4, fy, fx + 4, fy, g, 1, 1.3)
            elif name == "trash":
                rr_stroke(c, fx - 6, fy - 5, 12, 13, 2, g, 1, 1.3)
                line(c, fx - 8, fy - 7, fx + 8, fy - 7, g, 1, 1.3)
            elif name == "fx":
                self.T(c, "fx", fx, fy + 5, 13, g, SANS_BOLD, "center")
            elif name == "group":
                c.drawPath(path([(fx - 9, fy - 6), (fx - 3, fy - 6), (fx - 1, fy - 3), (fx + 9, fy - 3), (fx + 9, fy + 7), (fx - 9, fy + 7)], True), strokep(g, 1, 1.3))
            elif name == "link":
                c.drawOval(skia.Rect.MakeXYWH(fx - 9, fy - 4, 10, 8), strokep(g, 1, 1.3))
                c.drawOval(skia.Rect.MakeXYWH(fx - 1, fy - 4, 10, 8), strokep(g, 1, 1.3))

    def draw_popup(self, c, st):
        x, y = st.popup["x"], st.popup["y"]
        w, row = 300, 34
        hgt = len(ADJ_MENU) * row + 16
        sh = skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.5))
        sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 12))
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + 3, y + 6, w, hgt), 8, 8), sh)
        rr(c, x, y, w, hgt, 8, (44, 44, 49))
        rr_stroke(c, x, y, w, hgt, 8, (75, 75, 82), 1, 1)
        for i, item in enumerate(ADJ_MENU):
            iy = y + 8 + i * row
            if item == "-":
                line(c, x + 10, iy + row / 2, x + w - 10, iy + row / 2, (70, 70, 76), 1, 1)
                continue
            if st.popup.get("hover") == i:
                rr(c, x + 6, iy + 2, w - 12, row - 4, 5, C_ACCENT)
            self.T(c, item, x + 22, iy + 22, 13.5, (240, 240, 245))

    def draw_status(self, c):
        L = self.d.L
        x0, y0, w, h = L.win
        rect(c, x0, L.bottom, w, L.status_h, (36, 36, 40))
        self.T(c, f"{L.zoom * 100:.2f}%", x0 + 70, L.bottom + 17, 12, C_DIM)
        self.T(c, f"{self.d.W} px x {self.d.H} px (300 ppi)", x0 + 150, L.bottom + 17, 12, C_DIM)

    def draw_taskbar(self, c):
        y = VH - 50
        c.drawRect(skia.Rect.MakeXYWH(0, y, VW, 50), fillp((24, 30, 46)))
        line(c, 0, y, VW, y, (70, 76, 92), 0.6, 1)
        icons = ["start", "search", "files", "browser", "app", "store", "mail"]
        x = VW / 2 - (len(icons) - 1) * 28
        for name in icons:
            cy = y + 25
            if name == "app":
                rr(c, x - 19, cy - 19, 38, 38, 6, (255, 255, 255), 0.10)
                rr(c, x - 6, y + 45, 12, 3, 1.5, C_ACCENT2)
            if name == "start":
                for dx, dy in ((-7, -7), (1, -7), (-7, 1), (1, 1)):
                    rr(c, x + dx, cy + dy, 6.5, 6.5, 1, (90, 170, 255))
            elif name == "search":
                c.drawCircle(x - 2, cy - 2, 6, strokep((230, 230, 235), 1, 1.8))
                line(c, x + 2, cy + 2, x + 7, cy + 7, (230, 230, 235), 1, 1.8)
            elif name == "files":
                c.drawPath(path([(x - 10, cy - 7), (x - 3, cy - 7), (x - 1, cy - 4), (x + 10, cy - 4), (x + 10, cy + 8), (x - 10, cy + 8)], True), fillp((255, 196, 60)))
            elif name == "browser":
                c.drawCircle(x, cy, 10, fillp((40, 140, 230)))
                c.drawCircle(x, cy, 4.5, fillp((140, 220, 255)))
            elif name == "app":
                rr(c, x - 10, cy - 10, 20, 20, 5, (49, 168, 255))
                self.T(c, "Lu", x, cy + 4.5, 10, (8, 30, 60), SANS_BOLD, "center")
            elif name == "store":
                rr(c, x - 9, cy - 6, 18, 15, 2, (90, 120, 230))
            elif name == "mail":
                rr(c, x - 10, cy - 7, 20, 14, 2, (60, 140, 255))
            x += 56
        self.T(c, "9:41 PM", VW - 70, y + 21, 12.5, (235, 235, 240), SANS_REG, "center")
        self.T(c, "10/8/2026", VW - 70, y + 39, 12.5, (235, 235, 240), SANS_REG, "center")
        g = (230, 230, 235)
        c.drawArc(skia.Rect.MakeXYWH(VW - 208, y + 18, 16, 16), 225, 90, False, strokep(g, 1, 1.6))
        c.drawPath(path([(VW - 176, y + 22), (VW - 172, y + 22), (VW - 167, y + 18), (VW - 167, y + 32), (VW - 172, y + 28), (VW - 176, y + 28)], True), fillp(g))
        rr_stroke(c, VW - 149, y + 20, 16, 10, 2, g, 1, 1.3)
        rect(c, VW - 147, y + 22, 9, 6, g)

    def draw_cursor(self, c, st, t):
        d = self.d
        x, y = d.cursor_pos(t)
        px, py, pw, ph = d.L.photo
        painting = any(o["kind"] == "painting" and o["t0"] - 0.6 <= t <= o["t1"] + 0.3 for o in d.overlays)
        if painting or (st.tool == "brush" and px <= x <= px + pw and py <= y <= py + ph):
            r = st.brush_size / 2 * pw / d.W
            c.drawCircle(x, y, r, strokep((255, 255, 255), 0.95, 1.4))
            c.drawCircle(x, y, r + 1.2, strokep((0, 0, 0), 0.45, 0.8))
            line(c, x - 5, y, x + 5, y, (255, 255, 255), 0.9, 1)
            line(c, x, y - 5, x, y + 5, (255, 255, 255), 0.9, 1)
            return
        pts = [(x, y), (x, y + 21), (x + 5, y + 16.5), (x + 9, y + 25), (x + 12.5, y + 23.5), (x + 8.7, y + 15.2), (x + 15, y + 15)]
        sh = skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.35))
        sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 2))
        c.save()
        c.translate(1.5, 2)
        c.drawPath(path(pts, True), sh)
        c.restore()
        c.drawPath(path(pts, True), fillp((255, 255, 255)))
        c.drawPath(path(pts, True), strokep((0, 0, 0), 1, 1.1))

    def draw_ripples(self, c, t):
        for o in self.d.overlays:
            if o["kind"] == "ripple" and 0 <= t - o["t0"] < 0.55:
                u = (t - o["t0"]) / 0.55
                x, y = o["pos"]
                c.drawCircle(x, y, 6 + 26 * ease(u), strokep(C_ACCENT2, 0.9 * (1 - u), 3))
                c.drawCircle(x, y, 10 * (1 - u), fillp(C_ACCENT2, 0.35 * (1 - u)))

    # ----------------------------------------------------------- overlays (design coords)
    def O(self, s, x, y, size, rgb=(255, 255, 255), font=SANS_BOLD, align="left", a=1.0):
        k = self.k
        self.c.save()
        self.c.scale(k, k)
        self.text.draw(self.c, s, x, y, size, rgb, font, align, a, k)
        self.c.restore()

    def orr(self, x, y, w, h, r, rgb, a=1.0):
        k = self.k
        self.c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x * k, y * k, w * k, h * k), r * k, r * k), fillp(rgb, a))

    def frame(self, t):
        d, c, k = self.d, self.c, self.k
        st = self.state_at(t)
        si = d.step_at(t)
        step = d.steps[si]
        tl = t - step["start"]

        fx, fy, fw, fh = d.camera_rect(t)
        card_h = clamp(CARD_W / (fw / fh), 860, 1200)
        crop_w = max(fw, fh * CARD_W / card_h)
        crop_h = crop_w * card_h / CARD_W
        if crop_w > VW:
            crop_w, crop_h = VW, VW * card_h / CARD_W
        if crop_h > VH:
            crop_h = VH
            crop_w = crop_h * CARD_W / card_h
        ccx = clamp(fx + fw / 2, crop_w / 2, VW - crop_w / 2)
        ccy = clamp(fy + fh / 2, crop_h / 2, VH - crop_h / 2)
        card_y = CARD_CY - card_h / 2 + 30
        s = CARD_W * k / crop_w
        self._q = 2 ** (round(math.log2(max(0.25, s)) * 6) / 6)

        card = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(CARD_X * k, card_y * k, CARD_W * k, card_h * k), 30 * k, 30 * k)
        c.drawImage(self.backdrop(round(card_h / 3) * 3, card_y), 0, 0)   # background + card shadow, cached
        cs = self.card_surf.getCanvas()
        cs.save()
        cs.clear(col((0, 0, 0)))
        cs.scale(s, s)
        cs.translate(-(ccx - crop_w / 2), -(ccy - crop_h / 2))
        self.draw_desktop(cs, st, t)
        self.draw_ripples(cs, t)
        self.draw_cursor(cs, st, t)
        cs.restore()
        snap = self.card_surf.makeImageSnapshot()
        c.save()
        c.clipRRect(card, True)
        c.drawImageRect(snap, skia.Rect.MakeWH(self.card_px_w, card_h * k),
                        skia.Rect.MakeXYWH(CARD_X * k, card_y * k, CARD_W * k, card_h * k), skia.SamplingOptions())
        c.restore()
        c.drawRRect(card, strokep((255, 255, 255), 0.14, 2 * k))

        self.draw_header(t, si, tl)
        self.draw_keys(t, card_y + card_h)
        self.draw_note(t, card_y + card_h)
        self.draw_captions(step, tl, t)
        self.orr(0, 1908, DW * clamp(t / d.total), 12, 0, BRAND, 0.9)
        if self.handle:
            self.O(self.handle, 990, 92, 20, (255, 255, 255), SANS, "right", 0.55)
        self.draw_hook(t)
        self.draw_end(t)
        fade = min(clamp(t / 0.3), clamp((d.total - t) / 0.4))
        if fade < 1:
            c.drawRect(skia.Rect.MakeWH(self.w, self.h), fillp((8, 9, 14), 1 - fade))
        return self.buf

    def draw_header(self, t, si, tl):
        d = self.d
        n = len(d.steps)
        dot_w = 900 / n
        for i in range(n):
            self.orr(90 + i * dot_w + 3, 120, dot_w - 6, 6, 3, BRAND if i <= si else (70, 76, 96), 0.95 if i <= si else 0.7)
        if d.script.get("hook") and t < HOOK_END:
            return
        e = ease(tl / 0.45) if t >= HOOK_END + 0.01 or si > 0 else ease((t - HOOK_END + 0.4) / 0.45)
        title = d.steps[si]["title"]
        label = "THE PROBLEM" if si == 0 else ("THE RESULT" if si == n - 1 else f"STEP {si}")
        lw = self.text.width(label, SANS_BOLD, 22) + 30
        self.orr(90, 168 + 20 * (1 - e), lw, 40, 20, BRAND, e)
        self.O(label, 105, 196 + 20 * (1 - e), 22, (255, 255, 255), SANS_BOLD, a=e)
        size = 54
        while self.text.width(title, SANS_BOLD, size) > 900 and size > 34:
            size -= 2
        self.O(title, 90, 284 + 24 * (1 - e), size, (255, 255, 255), SANS_BOLD, a=e)

    def draw_keys(self, t, card_bottom):
        for o in self.d.overlays:
            if o["kind"] != "keys" or not (o["t0"] <= t < o["t1"]):
                continue
            u_in, a = ease_back((t - o["t0"]) / 0.3), min(1.0, ease((o["t1"] - t) / 0.25))
            parts = [p for p in o["keys"].replace("+", " + ").split()]
            widths = [max(70, self.text.width(p, SANS_BOLD, 34) + 44) if p != "+" else 30 for p in parts]
            total = sum(widths) + 14 * (len(parts) - 1)
            x = DW / 2 - total / 2
            y = card_bottom - 104 - 20 * (1 - u_in)
            for p, w in zip(parts, widths):
                if p == "+":
                    self.O("+", x + w / 2, y + 48, 34, (200, 205, 220), SANS_BOLD, "center", a)
                else:
                    self.orr(x, y + 7, w, 72, 14, (0, 0, 0), 0.5 * a)
                    self.orr(x, y, w, 72, 14, (245, 246, 250), a)
                    self.O(p, x + w / 2, y + 48, 34, (20, 22, 30), SANS_BOLD, "center", a)
                x += w + 14

    def draw_note(self, t, card_bottom):
        for o in self.d.overlays:
            if o["kind"] != "note" or not (o["t0"] <= t < o["t1"] + 0.3):
                continue
            a = ease((t - o["t0"]) / 0.3) * (1 - ease((t - o["t1"]) / 0.3))
            size = 34
            while self.text.width(o["text"], SANS_BOLD, size) + 100 > 980 and size > 22:
                size -= 1
            w = self.text.width(o["text"], SANS_BOLD, size) + 100
            x, y = DW / 2 - w / 2, card_bottom - 220
            self.orr(x, y + 8, w, 86, 22, (0, 0, 0), 0.35 * a)
            self.orr(x, y, w, 86, 22, (255, 255, 255), a)
            self.orr(x + 24, y + 30, 28, 28, 14, BRAND, a)
            self.O("!", x + 38, y + 52, 20, (255, 255, 255), SANS_BOLD, "center", a)
            self.O(o["text"], x + 66, y + 55, size, (20, 22, 30), SANS_BOLD, a=a)

    def draw_captions(self, step, tl, t):
        if self.d.script.get("hook") and t < HOOK_END:
            return
        cur, ch = None, step["chunks"]
        for i, chunk in enumerate(ch):
            end = ch[i + 1][0][1] if i + 1 < len(ch) else chunk[-1][2] + 0.5
            if chunk[0][1] - 0.05 <= tl < end:
                cur = chunk
                break
        if not cur:
            return
        size, space, y, k = 44, 14, 1735, self.k
        widths = [self.text.width(w[0], SANS_BOLD, size) for w in cur]
        total = sum(widths) + space * (len(cur) - 1)
        scale = min(1.0, 900 / total)
        self.c.save()
        self.c.translate(DW / 2 * k, y * k)
        self.c.scale(scale, scale)
        self.c.translate(-DW / 2 * k, -y * k)
        self.orr(DW / 2 - total / 2 - 30, y - 62, total + 60, 86, 43, (255, 255, 255))
        x = DW / 2 - total / 2
        for wi, ((wd, s0, s1), w) in enumerate(zip(cur, widths)):
            nxt = cur[wi + 1][1] if wi + 1 < len(cur) else s1 + 0.3
            active = s0 - 0.03 <= tl < nxt
            self.O(wd, x, y - 4, size, BRAND if active else ((20, 22, 30) if tl >= s0 else (150, 155, 170)), SANS_BOLD)
            x += w + space
        self.c.restore()

    def draw_hook(self, t):
        hook = self.d.script.get("hook")
        if not hook or t >= HOOK_END:
            return
        a = 1 - ease((t - (HOOK_END - 0.4)) / 0.4)
        parts, bold = [], False
        for chunk in hook.split("**"):
            for w in chunk.split():
                parts.append((w, bold))
            bold = not bold
        size = 72
        lines, cur, cw = [], [], 0
        for w, b in parts:
            ww = self.text.width(w, SANS_BOLD, size)
            if cur and cw + ww + 18 > 920:
                lines.append(cur)
                cur, cw = [], 0
            cur.append((w, b, ww))
            cw += ww + 18
        lines.append(cur)
        y0 = 210
        n = 0
        for li, ln in enumerate(lines):
            x = 90
            for (w, b, ww) in ln:
                u = ease_back((t - 0.08 - n * 0.07) / 0.35)
                if b:
                    self.orr(x - 10, y0 + li * 88 - 66, ww + 20, 84, 12, BRAND, a * clamp(u))
                self.O(w, x, y0 + li * 88 + 20 * (1 - u), size, (255, 255, 255), SANS_BOLD, a=a * clamp(u))
                x += ww + 18
                n += 1

    def draw_end(self, t):
        d = self.d
        u = (t - (d.total - 2.6)) / 0.5
        if u <= 0:
            return
        a = ease(u)
        self.c.drawRect(skia.Rect.MakeWH(self.w, self.h), fillp((10, 12, 20), 0.82 * a))
        self.orr(240, 700, 600, 560, 60, (255, 255, 255), 0.97 * a)
        self.orr(460, 770, 160, 160, 40, BRAND, a)
        self.O((self.handle.lstrip("@")[:1] or "L").upper(), 540, 885, 96, (255, 255, 255), SANS_BOLD, "center", a)
        self.O(self.handle or "", 540, 1010, 46, (20, 22, 30), SANS_BOLD, "center", ease(u - 0.2))
        self.O("Pro editing tricks, every day", 540, 1062, 28, (90, 96, 115), SANS_REG, "center", ease(u - 0.3))
        b = ease_back(u - 0.45)
        if b > 0:
            w = 320 * b
            self.orr(540 - w / 2, 1110, w, 92, 46, BRAND, a)
            self.O("Follow", 540, 1170, 38, (255, 255, 255), SANS_BOLD, "center", a * clamp(b))


def render(demo, audio_path, out_path, width=2160, height=3840, fps=30, handle="", log=print):
    r = ScreenRenderer(demo, width, height, fps, handle)
    r.compute_final()
    n = int(math.ceil(demo.total * fps))
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-", "-i", str(audio_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-rc-lookahead", "20", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-movflags", "+faststart", "-shortest", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for f in range(n):
            proc.stdin.write(r.frame(f / fps).data)
            if f % (fps * 15) == 0:
                log(f"  frame {f}/{n} ({time.time() - t0:.0f}s)")
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError("ffmpeg failed")
    log(f"  rendered {n} frames in {time.time() - t0:.0f}s")


def stills(demo, times, pattern, width=1080, height=1920, handle=""):
    r = ScreenRenderer(demo, width, height, 30, handle)
    r.compute_final()
    for i, t in enumerate(times):
        r.frame(t)
        r.surface.makeImageSnapshot().save(pattern % i, skia.kPNG)
