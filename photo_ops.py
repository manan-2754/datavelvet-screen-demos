"""
Real image processing for the on-screen photo editor - every slider and brush stroke in the demo
actually changes the pixels, the same way a real editor would.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

WORK_W = 2000   # working resolution of the photo


def load(path, width=WORK_W, exposure=0.0):
    im = Image.open(path).convert("RGB")
    h = int(im.height * width / im.width)
    a = np.asarray(im.resize((width, h), Image.LANCZOS), dtype=np.float32) / 255.0
    if exposure:
        a = a * (2.0 ** exposure)      # keep values above 1.0 - like RAW headroom, so darkening recovers detail
    return a


def luma(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def adjust(img, p):
    """Apply an adjustment layer's parameters (exposure, contrast, highlights, shadows, saturation, temperature)."""
    out = img
    if p.get("exposure"):
        out = out * (2.0 ** p["exposure"])
    if p.get("highlights") or p.get("shadows"):
        L = luma(np.clip(out, 0, 1))[..., None]
        hi = np.clip((L - 0.5) * 2, 0, 1)
        sh = np.clip((0.5 - L) * 2, 0, 1)
        out = out + out * hi * p.get("highlights", 0) * 0.5 + (1 - out) * sh * p.get("shadows", 0) * 0.5
    if p.get("contrast"):
        out = (out - 0.5) * (1 + p["contrast"]) + 0.5
    if p.get("saturation"):
        L = luma(np.clip(out, 0, 1))[..., None]
        out = L + (out - L) * (1 + p["saturation"])
    if p.get("temperature"):
        t = p["temperature"]
        out = out * np.array([1 + 0.12 * t, 1.0, 1 - 0.12 * t], np.float32)
    return np.clip(out, 0, 1)


def _rgb_to_hsv(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(-1), img.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    s = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    out = np.zeros(h.shape + (3,), np.float32)
    for k, (a, b, c) in enumerate(((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))):
        m = i == k
        out[m] = np.stack([a[m], b[m], c[m]], -1)
    return out


def adjust_layer(img, kind, p):
    """One adjustment layer of a given kind (matches the editor's Adjustments menu)."""
    x = np.clip(img, 0, None)
    if kind == "Exposure":
        out = x * (2.0 ** p.get("exposure", 0)) + p.get("offset", 0)
        g = max(0.01, p.get("gamma", 1.0))
        out = np.clip(out, 0, None) ** (1 / g)
    elif kind == "Brightness/Contrast":
        out = x + p.get("brightness", 0) / 300.0
        out = (out - 0.5) * (1 + p.get("contrast", 0) / 100.0) + 0.5
    elif kind == "Vibrance":
        L = luma(np.clip(x, 0, 1))[..., None]
        sat = np.clip(x.max(-1, keepdims=True) - x.min(-1, keepdims=True), 0, 1)
        vib = p.get("vibrance", 0) / 100.0 * (1 - sat)          # boosts muted colours more than vivid ones
        out = L + (x - L) * (1 + vib + p.get("saturation", 0) / 100.0)
    elif kind == "Hue/Saturation":
        h, s, v = _rgb_to_hsv(np.clip(x, 0, 1))
        h = (h + p.get("hue", 0) / 360.0) % 1.0
        s = np.clip(s * (1 + p.get("saturation", 0) / 100.0), 0, 1)
        out = _hsv_to_rgb(h, s, v)
        lt = p.get("lightness", 0) / 100.0
        out = out + (1 - out) * lt if lt > 0 else out * (1 + lt)
    elif kind == "Black & White":
        wr = 0.30 * (1 + (p.get("reds", 40) - 40) / 100.0)
        wb = 0.11 * (1 + (p.get("blues", 20) - 20) / 100.0)
        wg = 0.59
        L = (x[..., 0] * wr + x[..., 1] * wg + x[..., 2] * wb) / max(0.2, wr + wg + wb) * 1.0
        out = np.repeat(L[..., None], 3, -1)
    elif kind == "Photo Filter":
        d = p.get("warmth", 0) / 100.0 * p.get("filter_density", 25) / 25.0 * 0.35
        tinted = x * np.array([1 + d, 1 + 0.15 * d, 1 - d], np.float32)
        L0, L1 = luma(np.clip(x, 0, 1)), luma(np.clip(tinted, 0, 1))
        out = tinted * (L0 / np.maximum(L1, 1e-4))[..., None]       # preserve luminosity like the real filter
    else:
        out = x
    return np.clip(out, 0, 1).astype(np.float32)


def apply_look(img, look):
    """Simulate the 'problem' photo a tutorial starts from (too dark, flat, washed out, cold...)."""
    out = img
    if look.get("exposure"):
        out = out * (2.0 ** look["exposure"])        # keep >1.0 headroom so darkening recovers detail
    if look.get("contrast"):
        out = (out - 0.5) * (1 + look["contrast"]) + 0.5
    if look.get("saturation"):
        L = luma(np.clip(out, 0, 1))[..., None]
        out = L + (out - L) * (1 + look["saturation"])
    if look.get("warmth"):
        out = out * np.array([1 + 0.12 * look["warmth"], 1.0, 1 - 0.12 * look["warmth"]], np.float32)
    return np.clip(out, 0, None).astype(np.float32)


def stamp_stroke(mask, pts, radius, value=1.0, hardness=0.0, flow=1.0):
    """Paint soft round brush stamps along a polyline (image pixel coords) into mask (float32 HxW, 0..1)."""
    h, w = mask.shape
    spacing = max(2.0, radius * 0.18)
    seq = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        d = max(1e-6, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        n = max(1, int(d / spacing))
        seq += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    if pts:
        seq.append(pts[-1])
    r = int(radius) + 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    dist = np.sqrt(xx ** 2 + yy ** 2) / radius
    brush = np.clip((1 - dist) / max(1e-3, 1 - hardness), 0, 1)
    brush = (brush * brush * (3 - 2 * brush)) * flow          # smoothstep falloff
    for (cx, cy) in seq:
        cx, cy = int(round(cx)), int(round(cy))
        x0, x1, y0, y1 = cx - r, cx + r + 1, cy - r, cy + r + 1
        bx0, by0 = max(0, -x0), max(0, -y0)
        bx1, by1 = brush.shape[1] - max(0, x1 - w), brush.shape[0] - max(0, y1 - h)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        b = brush[by0:by1, bx0:bx1]
        region = mask[y0:y1, x0:x1]
        region += (value - region) * b
    return mask


def feather(mask, radius_px):
    return gaussian_filter(mask, sigma=max(0.01, radius_px / 2.0)) if radius_px > 0.5 else mask


def composite(base, adjusted, mask):
    if mask is None:
        return adjusted
    m = mask[..., None]
    return base * (1 - m) + adjusted * m


def to_rgba8(img):
    a = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(np.dstack([a, np.full(a.shape[:2], 255, np.uint8)]))


def mask_rgba8(mask):
    g = (np.clip(mask, 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(np.dstack([g, g, g, np.full(g.shape, 255, np.uint8)]))


def thumb(img, w=96):
    h = int(img.shape[0] * w / img.shape[1])
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def histogram(img, bins=64):
    L = luma(img[::6, ::6])
    hist, _ = np.histogram(L, bins=bins, range=(0, 1))
    return hist / max(1, hist.max())
