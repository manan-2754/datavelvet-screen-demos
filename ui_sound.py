"""
Sound design for screen demos: real-sounding mouse clicks, keyboard presses, slider ticks and brush swishes,
placed exactly where they happen on screen, over a soft music bed that ducks under the narrator.
"""
import wave

import numpy as np

from sound import SR, _decay, _noise, music_bed, read_wav


def click():
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * 1800 * t) * _decay(n, 0.004) * 0.35
    thump = np.sin(2 * np.pi * 180 * t) * _decay(n, 0.012) * 0.4
    return (body + thump + _noise(n, 3, 1) * _decay(n, 0.003) * 0.6).astype(np.float32)


def key(seed=0):
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    clack = _noise(n, 2, seed) * _decay(n, 0.006) * 1.2
    low = np.sin(2 * np.pi * (240 + 30 * (seed % 3)) * t) * _decay(n, 0.02) * 0.35
    return (clack + low).astype(np.float32)


def tick():
    n = int(0.025 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 3200 * t) * _decay(n, 0.003) * 0.25).astype(np.float32)


def pop():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    f = 500 + 500 * (1 - np.exp(-t * 50))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.03) * 0.3).astype(np.float32)


def brush(seconds):
    n = int(seconds * SR)
    x = _noise(n, 40, 5)
    t = np.arange(n) / SR
    wobble = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 1.6 * t))       # strokes back and forth
    env = np.clip(np.minimum(t / 0.15, (seconds - t) / 0.25), 0, 1)
    return (x * wobble * env * 0.35 / (np.abs(x).max() + 1e-9)).astype(np.float32)


def whoosh(seconds=0.7, seed=2):
    n = int(seconds * SR)
    x = _noise(n, 16, seed) * np.sin(np.linspace(0, np.pi, n)) ** 2
    return (x * 0.3 / (np.abs(x).max() + 1e-9)).astype(np.float32)


def impact():
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    f = 80 * np.exp(-t * 3) + 40
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.3)
    return ((boom + _noise(n, 4, 9) * _decay(n, 0.05) * 2) * 0.6).astype(np.float32)


def mix_screen(narration_wav, sfx, total, out_path, seed=1, hook=True):
    voice = read_wav(narration_wav)
    n = max(len(voice), int(total * SR))
    voice = np.pad(voice, (0, n - len(voice)))
    win = int(0.25 * SR)
    env = np.convolve(np.abs(voice), np.ones(win, np.float32) / win, mode="same")
    duck = 1 - 0.65 * np.clip(env / 0.05, 0, 1)
    music = music_bed(n / SR, seed) * duck[:, None] * 0.75
    fx = np.zeros(n, np.float32)

    def place(sound, t, gain):
        s = int(max(0.0, t) * SR)
        if s < n:
            e = min(n, s + len(sound))
            fx[s:e] += sound[: e - s] * gain

    c, tk, pp = click(), tick(), pop()
    for i, (t, kind, gain) in enumerate(sfx):
        if kind == "click":
            place(c, t, 0.9 * gain)
        elif kind == "key":
            place(key(i), t, 0.9 * gain)
        elif kind == "tick":
            place(tk, t, gain)
        elif kind == "pop":
            place(pp, t, gain)
        elif kind.startswith("brush"):
            place(brush(float(kind.split(":")[1]) if ":" in kind else 2.5), t, gain)
        elif kind == "whoosh":
            place(whoosh(), t, gain)
    if hook:
        place(impact(), 0.0, 0.7)
    stereo = np.stack([voice + music[:, 0] + fx * 0.8, voice + music[:, 1] + fx * 0.8], axis=1)
    stereo = np.tanh(stereo * 1.1) / np.tanh(1.1)
    stereo *= 0.95 / max(1e-9, float(np.abs(stereo).max()))
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((stereo * 32767).astype(np.int16).tobytes())
    return out_path
