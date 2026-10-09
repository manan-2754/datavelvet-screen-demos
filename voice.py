"""
Narration: one natural-sounding English voice clip per scene, stitched into a single track.

Each scene is synthesised separately so we know exactly how long it lasts -
the video timeline is built from these durations, keeping visuals in sync with the voice.
"""
import asyncio
import random
import subprocess
import wave
from pathlib import Path

import numpy as np

from screen_engine import ffmpeg_exe

SR = 48000
LEAD_IN = 0.35      # silence before the first word
SCENE_GAP = 0.45    # breath between scenes

# Warm, conversational US English voices (Microsoft neural, free via edge-tts)
VOICES = ["en-US-AvaMultilingualNeural", "en-US-EmmaMultilingualNeural"]   # different voice from the 3D channel


async def _edge(text, path, voice, rate):
    """Save the clip and return word timings [(word, start_s, end_s)]."""
    import edge_tts
    words = []
    with open(path, "wb") as f:
        async for chunk in edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary").stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                words.append((chunk["text"], start, start + chunk["duration"] / 1e7))
    return words


def synth_clip(text, path, voice, rate="+4%"):
    """Synthesise one clip with retries; falls back to the other voice, then gTTS.
    Returns (voice_used, word_timings or None)."""
    for attempt, v in enumerate([voice, voice, *[x for x in VOICES if x != voice]]):
        try:
            words = asyncio.run(_edge(text, path, v, rate))
            if path.exists() and path.stat().st_size > 1000:
                return v, words or None
        except Exception as e:
            print(f"  edge-tts attempt {attempt + 1} ({v}) failed: {e}")
    from gtts import gTTS
    gTTS(text=text, lang="en").save(str(path))
    return "gtts", None


def decode(path):
    """Decode any audio file to mono float32 at SR. Returns (audio, seconds trimmed from the start)."""
    out = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-f", "f32le",
                          "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(out, dtype=np.float32).copy()
    # trim encoder padding / leading silence so scene timing stays tight
    nz = np.nonzero(np.abs(audio) > 0.004)[0]
    if not len(nz):
        return audio, 0.0
    first = max(0, nz[0] - int(0.03 * SR))
    return audio[first: nz[-1] + int(0.08 * SR)], first / SR


def estimate_words(text, seconds):
    """Fallback word timings spread by character length (used if the TTS gave none)."""
    toks = text.split()
    total = sum(len(t) + 1 for t in toks) or 1
    out, t = [], 0.0
    for tok in toks:
        d = seconds * (len(tok) + 1) / total
        out.append((tok, t, t + d * 0.9))
        t += d
    return out


def whoosh(seconds=0.35, gain=0.035, seed=0):
    """Soft filtered-noise swoosh used under scene transitions."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    kernel = np.hanning(64).astype(np.float32)
    noise = np.convolve(noise, kernel / kernel.sum(), mode="same")
    env = np.sin(np.linspace(0, np.pi, n)) ** 2
    return noise * env * gain / (np.abs(noise).max() + 1e-9)


def build_narration(scenes, work_dir, voice=None):
    """Returns (wav_path, per-scene durations, voice used, per-scene word timings relative to scene start)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    voice = voice or random.choice(VOICES)
    clips, timings, used = [], [], voice
    for i, sc in enumerate(scenes):
        mp3 = work_dir / f"scene_{i:02d}.mp3"
        used, words = synth_clip(sc["narration"], mp3, voice)
        audio, trimmed = decode(mp3)
        clips.append(audio)
        if not words:
            words = estimate_words(sc["narration"], len(audio) / SR)
            trimmed = 0.0
        timings.append([(w, s0 - trimmed, s1 - trimmed) for w, s0, s1 in words])
        print(f"  voice scene {i + 1}/{len(scenes)}: {len(audio) / SR:.1f}s, {len(words)} words timed")

    durations, parts, word_times = [], [], []
    for i, clip in enumerate(clips):
        pad_before = LEAD_IN if i == 0 else 0.0
        seg = np.concatenate([np.zeros(int(pad_before * SR), np.float32), clip,
                              np.zeros(int(SCENE_GAP * SR), np.float32)])
        durations.append(len(seg) / SR)
        parts.append(seg)
        word_times.append([(w, s0 + pad_before, s1 + pad_before) for w, s0, s1 in timings[i]])
    tail = np.zeros(int(3.0 * SR), np.float32)   # room for the end card
    track = np.concatenate(parts + [tail])
    durations[-1] += len(tail) / SR

    # transition swooshes, centred on each scene change
    t = 0.0
    for i, d in enumerate(durations[:-1]):
        t += d
        w = whoosh(seed=i)
        start = max(0, int((t - 0.15) * SR))
        track[start:start + len(w)] += w[: len(track) - start]

    peak = np.abs(track).max()
    if peak > 0:
        track = track * (0.89 / peak)
    pcm = (np.clip(track, -1, 1) * 32767).astype(np.int16)
    wav_path = work_dir / "narration.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return wav_path, durations, used, word_times
