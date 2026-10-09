"""
Sound design: a soft synthesized music bed that ducks under the narrator, plus sound effects
placed on the exact frames where the animations happen (pops, whooshes, chimes, ticks, thuds).

Everything is generated with numpy - no music or sample files, so no licensing issues.
"""
import wave

import numpy as np

SR = 48000


def _env(n, attack, release):
    e = np.ones(n, np.float32)
    a, r = min(n, int(attack * SR)), min(n, int(release * SR))
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r:
        e[-r:] *= np.linspace(1, 0, r)
    return e


def _decay(n, tau):
    return np.exp(-np.arange(n) / (tau * SR)).astype(np.float32)


def _noise(n, smooth=24, seed=0):
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    k = np.hanning(max(3, smooth) + 2)[1:-1].astype(np.float32)   # drop the zero end-points
    return np.convolve(x, k / k.sum(), mode="same")


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# ---------------------------------------------------------------- sound effects
def sfx_pop():
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    f = 420 + 700 * (1 - np.exp(-t * 40))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.045) * 0.5).astype(np.float32)


def sfx_whoosh(seconds=0.55, seed=1):
    n = int(seconds * SR)
    x = _noise(n, 18, seed) * np.sin(np.linspace(0, np.pi, n)) ** 2
    return (x * 0.35 / (np.abs(x).max() + 1e-9)).astype(np.float32)


def sfx_chime():
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((1568, 0.5), (2349, 0.3), (3136, 0.12)))
    return (x * _decay(n, 0.22) * _env(n, 0.004, 0) * 0.42).astype(np.float32)


def sfx_tick():
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 2200 * t) * _decay(n, 0.008) * 0.45).astype(np.float32)


def sfx_thud():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    f = 120 * np.exp(-t * 6) + 45
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.16)
    crunch = _noise(n, 6, 3) * _decay(n, 0.04) * 3
    return ((body + crunch) * 0.6).astype(np.float32)


def sfx_shimmer():
    n = int(0.8 * SR)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in (880, 1320, 1760)) / 3
    return (x * np.sin(np.pi * t / t[-1]) ** 2 * (0.7 + 0.3 * np.sin(2 * np.pi * 12 * t)) * 0.16).astype(np.float32)


def sfx_impact():
    """Cinematic hit for the opening hook: sub boom + bright noise burst."""
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    f = 90 * np.exp(-t * 3) + 38
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * _decay(n, 0.35)
    burst = _noise(n, 4, 9) * _decay(n, 0.06) * 2.5
    return ((boom + burst) * 0.7).astype(np.float32)


# ---------------------------------------------------------------- music bed
def music_bed(seconds, seed=0):
    """Warm minor-key pad + soft arpeggio + sub bass, ~88 BPM. Returns stereo (n, 2)."""
    n = int(seconds * SR)
    out = np.zeros((n, 2), np.float32)
    beat = 60 / 88
    t_chord = 8 * beat
    progs = [[57, 53, 48, 55], [57, 52, 53, 55], [50, 53, 57, 52]]
    prog = progs[seed % len(progs)]
    minor = {57, 52, 50}
    i, t0 = 0, 0.0
    while t0 < seconds:
        root = prog[i % len(prog)]
        notes = [root, root + (3 if root in minor else 4), root + 7]
        s0, s1 = int(t0 * SR), min(n, int((t0 + t_chord + 1.0) * SR))
        m = s1 - s0
        t = np.arange(m) / SR
        env = _env(m, 0.9, 1.0)
        for nt in notes:  # pad with slight stereo detune
            for ch, det in ((0, -0.08), (1, 0.08)):
                f = hz(nt + det)
                x = sum(np.sin(2 * np.pi * f * h * t) / h ** 1.6 for h in range(1, 6))
                out[s0:s1, ch] += (x * env * 0.032).astype(np.float32)
        sub = (np.sin(2 * np.pi * hz(root - 24) * t) * env * 0.07).astype(np.float32)
        out[s0:s1, 0] += sub
        out[s0:s1, 1] += sub
        arp = notes + [notes[1] + 12]
        for k in range(int(t_chord / (beat / 2))):
            st = s0 + int(k * beat / 2 * SR)
            if st >= n:
                break
            ln = min(int(0.45 * SR), n - st)
            tt = np.arange(ln) / SR
            f = hz(arp[(k * 3) % len(arp)] + 12)
            note = ((np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * _decay(ln, 0.12) * 0.035).astype(np.float32)
            pan = 0.35 if k % 2 else -0.35
            out[st:st + ln, 0] += note * (1 - pan)
            out[st:st + ln, 1] += note * (1 + pan)
            echo = st + int(0.75 * beat * SR)
            if echo < n:
                el = min(ln, n - echo)
                out[echo:echo + el, 0] += note[:el] * 0.3 * (1 + pan)
                out[echo:echo + el, 1] += note[:el] * 0.3 * (1 - pan)
        for k in range(int(t_chord / beat)):
            st = s0 + int((k + 0.5) * beat * SR)
            if st >= n:
                break
            ln = min(int(0.05 * SR), n - st)
            h = np.diff(_noise(ln + 1, 2, seed + k)) * _decay(ln, 0.012) * 0.05
            out[st:st + ln] += h[:, None]
        t0 += t_chord
        i += 1
    fade = np.ones(n, np.float32)
    fi, fo = min(n, int(1.5 * SR)), min(n, int(2.5 * SR))
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] *= np.linspace(1, 0, fo)
    return out * fade[:, None]


# ---------------------------------------------------------------- mix
def read_wav(path):
    with wave.open(str(path)) as wf:
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
        if wf.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x


def mix_audio(narration_wav, compiled_scenes, total, out_path, seed=0, hook=True):
    voice = read_wav(narration_wav)
    n = max(len(voice), int(total * SR))
    voice = np.pad(voice, (0, n - len(voice)))

    # music ducks under the voice (smoothed voice envelope)
    win = int(0.25 * SR)
    env = np.convolve(np.abs(voice), np.ones(win, np.float32) / win, mode="same")
    duck = 1 - 0.62 * np.clip(env / 0.05, 0, 1)
    music = music_bed(n / SR, seed) * duck[:, None] * 0.9

    fx = np.zeros(n, np.float32)

    def place(sound, t, gain=1.0):
        s = int(max(0.0, t) * SR)
        if s >= n:
            return
        e = min(n, s + len(sound))
        fx[s:e] += sound[: e - s] * gain

    pop, chime, tick, thud, shimmer = sfx_pop(), sfx_chime(), sfx_tick(), sfx_thud(), sfx_shimmer()
    for sc in compiled_scenes:
        for o in sc["objs"].values():
            if not o["carried"]:
                place(pop, sc["start"] + o["appear"], 0.55)
        for b in sc["beats"]:
            t = sc["start"] + b["t"]
            if b["do"] == "send":
                place(sfx_whoosh(0.9, seed=int(t * 10)), t, 0.5)
                place(chime, t + 0.95, 0.5)
            elif b["do"] == "flow":
                place(sfx_whoosh(1.2, seed=int(t * 10)), t, 0.35)
            elif b["do"] == "connect":
                place(sfx_whoosh(0.5, seed=int(t * 10)), t, 0.35)
            elif b["do"] == "text":
                place(tick, t, 0.6)
                place(tick, t + 0.07, 0.4)
            elif b["do"] == "break":
                place(thud, t, 0.75)
            elif b["do"] == "highlight":
                place(shimmer, t, 0.7)
            elif b["do"] == "count":
                for j in range(4):
                    place(pop, t + j * 0.09, 0.35)
            elif b["do"] == "focus":
                place(sfx_whoosh(0.8, seed=int(t * 10)), t, 0.3)
    place(chime, total - 2.6, 0.5)  # end card
    if hook:
        place(sfx_impact(), 0.0, 0.8)

    stereo = np.stack([voice + music[:, 0] + fx * 0.7, voice + music[:, 1] + fx * 0.7], axis=1)
    stereo = np.tanh(stereo * 1.1) / np.tanh(1.1)          # gentle limiter
    stereo *= 0.95 / max(1e-9, float(np.abs(stereo).max()))
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return out_path
