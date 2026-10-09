# DataVelvet Screen Demos

Narrated "live editing on a Windows PC" tutorials, rendered in 4K vertical video.

A realistic Windows 11 desktop with a photo editor ("Lumen Photo", a fictional app) performs a scripted demo:
the cursor moves and clicks, menus open, sliders drag and a brush paints a real mask - every edit is real pixel
processing on a real photo. A screen-recording style camera zooms into whatever is being used, with keystroke
overlays, step titles, word-by-word captions, a female AI voice and synced UI sound effects.

| File | What it does |
|---|---|
| `main.py` | `python main.py --script scripts/sky_mask.json` renders a demo (`--preview` for 1080p) |
| `screen_engine.py` | Desktop + editor UI, camera, cursor, timeline, overlays, renderer |
| `photo_ops.py` | Real image processing: exposure, brush strokes on masks, feathering, compositing |
| `voice.py` | Narration + per-word timings (edge-tts) |
| `ui_sound.py`, `sound.py` | Click / key / slider / brush sounds + music bed |
| `scripts/` | Demo scripts: steps with narration and actions cued to spoken words |
| `assets/photos/` | Free-to-use Unsplash photos (via Lorem Picsum) |

Action vocabulary: `focus`, `note`, `click`, `pick`, `slider`, `key`, `paint`, `compare`.

Roadmap: Gemini-written demo scripts, more apps (colour grading, VFX compositor, game-engine editor), daily schedule + auto-posting.
