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
