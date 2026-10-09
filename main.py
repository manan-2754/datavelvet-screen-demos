"""
Screen Demo Bot - narrated "live editing on a Windows PC" tutorials (photo editing first; colour grading,
VFX and game-design apps plug into the same engine).

  python main.py --script scripts/sky_mask.json            # render a demo script in 4K
  python main.py --script scripts/sky_mask.json --preview  # 1080p, faster
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env", override=True)
except ImportError:
    pass

import screen_engine as S  # noqa: E402
from ui_sound import mix_screen  # noqa: E402
from voice import build_narration  # noqa: E402

OUTPUT = BASE / "output"
HANDLE = os.getenv("CHANNEL_HANDLE", "@datavelvet")


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50] or "demo"


def make_video(script, out_dir, preview=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    print("🔊 Narration")
    wav, durations, voice, words = build_narration(script["steps"], out_dir / "audio")
    print(f"  voice: {voice}, total {sum(durations):.1f}s")
    demo = S.Demo(script, durations, words)
    print("🎵 Sound design")
    mixed = mix_screen(wav, demo.sfx, demo.total, out_dir / "audio" / "mix.wav", seed=len(script["topic"]),
                       hook=bool(script.get("hook")))
    w, h = (1080, 1920) if preview else (2160, 3840)
    print(f"🎬 Rendering {w}x{h}")
    video = out_dir / "video.mp4"
    S.render(demo, mixed, video, w, h, 30, HANDLE)
    meta = {**script.get("youtube", {}), "instagram_caption": script.get("instagram_caption", ""),
            "topic": script["topic"], "duration": round(demo.total, 1), "voice": voice}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(out_dir / "audio", ignore_errors=True)
    print(f"✅ {video} ({video.stat().st_size / 1e6:.1f} MB, {demo.total:.0f}s)")
    return video, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="demo script JSON")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    script = json.loads(Path(args.script).read_text(encoding="utf-8"))
    out_dir = OUTPUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{slugify(script['topic'])}"
    make_video(script, out_dir, args.preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
