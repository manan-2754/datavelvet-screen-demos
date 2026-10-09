"""
Script writer for screen demos: Gemini (with OpenRouter -> Groq -> OpenCode fallbacks) writes a live-editing demo
using ONLY actions the on-screen editor can really perform, then a second model pass fact-checks it.
"""
import json
import os
import re
import time
from pathlib import Path

import requests

from screen_engine import ADJ_MENU, ADJ_SPECS, MASK_SPECS, POOL, REGIONS

BASE = Path(__file__).parent
MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"]
LITE_MODELS = ["gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
FALLBACKS = [
    ("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
     ["nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3.5-lightning:free"]),
    ("Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]),
    ("OpenCode", "https://opencode.ai/zen/v1", "OPENCODE_API_KEY", ["gemini-3.8-flash", "claude-sonnet-5-5", "gpt-5.4-mini"]),
]
KEYS = {"ctrl+i", "b", "d", "x", "[", "]", "\\"}
CLICKS = {"layer.mask", "layer.thumb", "props.tab.mask", "props.tab.adjust", "tool.brush", "tool.move"}


def toolkit_text():
    adj = "\n".join(f"   - {k}...: sliders " + ", ".join(f"{n} ({lo}..{hi}, default {d})" for n, _, lo, hi, d, _ in v)
                    for k, v in ADJ_SPECS.items())
    photos = "\n".join(f"   - {p['file']}: {p['desc']}" + (" (has sky)" if p.get("sky", 0) >= 0.2 else "") for p in POOL)
    return adj, photos


PROMPT = """You write scripts for a faceless YouTube Shorts / Instagram Reels channel that teaches photo editing with
LIVE SCREEN DEMOS: viewers watch a Photoshop-style editor on a Windows PC while a friendly female creator narrates.
The editor really performs every action you write, so you may ONLY use the actions and settings below.

Topic: "{topic}"

PHOTOS (pick one that suits the topic):
{photos}

"look" simulates the problem photo before editing (all optional): exposure -1.5..1.5 (stops), contrast -0.5..0.3,
saturation -0.8..0.3, warmth -1..1 (negative = cold/blue). Choose a look that makes the topic's problem obvious.

ADJUSTMENT LAYERS (Layers panel > adjustment icon > menu item). Exactly ONE per video:
{adjustments}
Every new adjustment layer comes with a WHITE mask = the effect applies everywhere.
Mask: click "layer.mask" to select it; "Ctrl + I" inverts it (black = effect hidden everywhere).
Painting: press "B" for the brush, press "D" for default colours (with a mask selected that makes WHITE the foreground),
"X" swaps black/white. Painting white reveals the effect, black hides it. Paint regions: {regions}
Mask properties: click "props.tab.mask", then slider "feather" (0..250 px, softens edges) or "density" (0..100).
Alt + click the mask ("layer.mask" with "mod": "Alt") shows the mask itself; do it again to go back.

ACTIONS (each fires when its "cue" words are spoken - cue = 1-4 words copied EXACTLY from that step's narration):
  {{"cue": "...", "do": "compare"}}                                   before/after split view
  {{"cue": "...", "do": "focus", "target": "canvas"|"sky"|"layers"|"props"|"work"}}   camera only
  {{"cue": "...", "do": "note", "text": "max 32 chars"}}               on-screen tip card
  {{"cue": "...", "do": "pick", "target": "layers.adjust", "item": one of {items}}}
  {{"cue": "...", "do": "slider", "target": slider name, "to": number}}
  {{"cue": "...", "do": "click", "target": "layer.mask"|"layer.thumb"|"props.tab.mask"|"props.tab.adjust", "mod": optional "Alt"}}
  {{"cue": "...", "do": "key", "keys": "Ctrl + I"|"B"|"D"|"X"|"["|"]"}}
  {{"cue": "...", "do": "paint", "region": region, "value": 1 or 0, "size": 150-450}}

RULES
- 6 to 9 steps, total narration 140-250 words (about 60-95 seconds). Each step: "title" (max 36 chars, what we do now),
  "narration" (1-3 short sentences), "actions".
- Step 1 states the problem and MUST contain a "compare" action (shows the final result as the hook).
  The last step shows the result, MUST contain a "compare", and ends with "Follow for more editing tricks."
- The edit must genuinely solve the problem, in a correct order (pick before its sliders; select the mask before Ctrl + I;
  B before painting). Narrate exactly what happens on screen; speak shortcuts naturally ("Control I").
- Accurate Photoshop knowledge only. Say it works in Photoshop and most layer-based editors when useful.
- "hook": max 6 words for giant on-screen text in the first second; wrap ONE word in **double asterisks**.
- Casual, warm, confident; short sentences; no emojis in narration.
- "youtube": {{"title": max 70 chars ending " #shorts", "description": 2 sentences + 5-6 hashtags, "tags": 6-8}},
  "instagram_caption": 2-3 short lines + "Save this for your next edit." + 6-8 hashtags.

Return ONLY JSON: {{"topic", "photo", "look", "file_name" (a realistic .jpg name), "hook", "steps": [...], "youtube", "instagram_caption"}}
{feedback}"""

REVIEW = """You are a senior photo-editing instructor reviewing a short screen-demo script before it is published.
Topic: "{topic}"

SCRIPT:
{script}

Check: (1) any statement or step that is WRONG for Photoshop (shortcuts, what white/black masks do, what a slider does,
order of operations), (2) does the demo actually solve the stated problem, (3) is it clear for a beginner.
Simplifications are fine. Return ONLY JSON:
{{"score": 1-10, "critical_errors": ["what is wrong + correct fact", ...], "minor_issues": ["...", ...]}}"""


def _json_text(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
    a = text.find("{")
    if a == -1:
        return text
    try:
        _, end = json.JSONDecoder().raw_decode(text[a:])
        return text[a:a + end]
    except ValueError:
        b = text.rfind("}")
        return text[a:b + 1] if b > a else text


def _gemini_call(model, prompt, key):
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      headers={"x-goog-api-key": key}, timeout=240,
                      json={"contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.8}})
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:120]}")
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _openai_compatible(name, base, key, model, prompt):
    r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, timeout=300, json={
        "model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7,
        "max_tokens": 5000 if name == "Groq" else 12000, "response_format": {"type": "json_object"}})
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or "choices" not in data:
        raise RuntimeError(f"{r.status_code}: {r.text[:160]}")
    text = data["choices"][0]["message"].get("content") or ""
    json.loads(_json_text(text))
    return text


def llm(prompt, label="script"):
    """Gemini first; then OpenRouter -> Groq -> OpenCode; Gemini Lite as the very last resort."""
    key, last = os.getenv("GEMINI_API_KEY", "").strip(), None
    if key:
        for attempt in range(2):
            for model in MODELS:
                try:
                    text = _gemini_call(model, prompt, key)
                    print(f"  {label} by {model}")
                    return _json_text(text)
                except Exception as e:
                    last = f"{model}: {e}"
                    print(f"  {last}")
            if attempt == 0:
                time.sleep(15)
    print("  Gemini unavailable - trying fallback models")
    for name, base, env, models in FALLBACKS:
        fkey = os.getenv(env, "").strip()
        for model in (models if fkey else []):
            try:
                text = _openai_compatible(name, base, fkey, model, prompt)
                print(f"  {label} by {name} / {model}")
                return _json_text(text)
            except Exception as e:
                last = f"{name}/{model}: {e}"
                print(f"  {last}")
    for model in (LITE_MODELS if key else []):
        try:
            text = _gemini_call(model, prompt, key)
            print(f"  {label} by {model} (last resort)")
            return _json_text(text)
        except Exception as e:
            last = f"{model}: {e}"
    raise RuntimeError(f"All models failed: {last}")


def _norm(w):
    return "".join(ch for ch in w.lower() if ch.isalnum())


def _cue_ok(cue, narration):
    toks = [_norm(w) for w in str(cue).split() if _norm(w)]
    ws = [_norm(w) for w in narration.split() if _norm(w)]
    return bool(toks) and any(ws[i:i + len(toks)] == toks for i in range(len(ws) - len(toks) + 1))


def validate(sc):
    """Keep only actions the editor can perform, in a possible order; reject scripts that can't make a good demo."""
    photos = {p["file"] for p in POOL}
    if sc.get("photo") not in photos:
        raise ValueError(f"unknown photo {sc.get('photo')}")
    steps = sc.get("steps") or []
    if not 5 <= len(steps) <= 10:
        raise ValueError(f"bad step count {len(steps)}")
    raw_look = sc.get("look") if isinstance(sc.get("look"), dict) else {}
    look = {k: max(lo, min(hi, float(raw_look.get(k, 0) or 0)))
            for k, lo, hi in (("exposure", -1.5, 1.5), ("contrast", -0.5, 0.3), ("saturation", -0.8, 0.3), ("warmth", -1, 1))}
    kind, has_mask, mask_sel, brush, picks = None, False, False, False, 0
    fg, mask_col = "black", "white"          # Photoshop defaults; tracked so painting really does something
    clean = []
    for st in steps:
        narration = re.sub(r"\s+", " ", str(st.get("narration", ""))).strip()
        if len(narration.split()) < 4:
            raise ValueError("empty narration")
        acts = []
        for a in st.get("actions") or []:
            if not isinstance(a, dict) or not _cue_ok(a.get("cue", ""), narration):
                continue
            do = a.get("do")
            if do in ("compare", "focus", "note"):
                acts.append({k: a[k] for k in ("cue", "do", "target", "text") if k in a})
            elif do == "pick" and a.get("item") in ADJ_MENU and a["item"].replace("...", "") in ADJ_SPECS and picks == 0:
                kind, has_mask, picks = a["item"].replace("...", ""), True, 1
                acts.append({"cue": a["cue"], "do": "pick", "target": "layers.adjust", "item": a["item"]})
            elif do == "slider" and kind:
                names = [sp[0] for sp in ADJ_SPECS[kind]] + [sp[0] for sp in MASK_SPECS]
                if a.get("target") in names:
                    acts.append({"cue": a["cue"], "do": "slider", "target": a["target"], "to": float(a.get("to", 0))})
            elif do == "click" and a.get("target") in CLICKS:
                if a["target"] == "layer.mask" and not has_mask:
                    continue
                if a["target"] == "layer.mask":
                    mask_sel = True
                if a["target"] == "tool.brush":
                    brush = True
                acts.append({k: a[k] for k in ("cue", "do", "target", "mod") if k in a})
            elif do == "key" and str(a.get("keys", "")).lower().replace(" ", "") in KEYS:
                k = str(a["keys"]).lower().replace(" ", "")
                if k == "ctrl+i" and not (has_mask and mask_sel):
                    continue
                if k == "ctrl+i":
                    mask_col = "black" if mask_col == "white" else "white"
                if k == "d":
                    fg = "white" if mask_sel else "black"
                if k == "x":
                    fg = "white" if fg == "black" else "black"
                if k == "b":
                    brush = True
                acts.append({"cue": a["cue"], "do": "key", "keys": a["keys"]})
            elif do == "paint" and has_mask and brush and a.get("region") in REGIONS:
                if fg == mask_col:
                    raise ValueError(f"painting {fg} on a {mask_col} mask does nothing - set the colour with D / X first "
                                     f"(D = white foreground when the mask is selected, X swaps)")
                acts.append({"cue": a["cue"], "do": "paint", "region": a["region"],
                             "value": 1.0 if fg == "white" else 0.0,
                             "size": max(150, min(450, float(a.get("size", 320)))), "dur": 3.0})
        clean.append({"title": str(st.get("title", ""))[:40], "narration": narration, "actions": acts})
    if picks != 1:
        raise ValueError("no adjustment layer picked")
    if not any(a["do"] == "slider" for s in clean for a in s["actions"]):
        raise ValueError("no slider adjustment in the demo")
    for idx in (0, -1):
        if not any(a["do"] == "compare" for a in clean[idx]["actions"]):
            clean[idx]["actions"].insert(0, {"cue": " ".join(clean[idx]["narration"].split()[:2]), "do": "compare"})
    words = sum(len(s["narration"].split()) for s in clean)
    if not 110 <= words <= 290:
        raise ValueError(f"narration length {words} words")
    yt = sc.get("youtube") or {}
    title = str(yt.get("title") or sc.get("topic", "Photo editing trick"))[:95]
    if "#shorts" not in title.lower():
        title = title[:86] + " #shorts"
    return {"topic": sc.get("topic", ""), "app": "photo", "photo": sc["photo"], "look": look,
            "file_name": re.sub(r"[^\w.\-]", "_", str(sc.get("file_name") or "photo.jpg"))[:40],
            "hook": str(sc.get("hook", ""))[:48], "steps": clean,
            "youtube": {"title": title, "description": str(yt.get("description", ""))[:4500],
                        "tags": [str(t)[:30] for t in (yt.get("tags") or [])][:10]},
            "instagram_caption": str(sc.get("instagram_caption", ""))[:2100]}


def review(sc):
    lines = []
    for i, st in enumerate(sc["steps"]):
        acts = "; ".join(f"{a['do']} {a.get('item') or a.get('target') or a.get('keys') or a.get('region') or ''} {a.get('to', '')}".strip()
                         for a in st["actions"])
        lines.append(f"{i + 1}. [{st['title']}] {st['narration']}  (on screen: {acts})")
    data = json.loads(llm(REVIEW.format(topic=sc["topic"], script="\n".join(lines)), label="quality gate review"))
    return {"score": int(data.get("score", 0)), "critical": [str(x) for x in data.get("critical_errors") or []],
            "minor": [str(x) for x in data.get("minor_issues") or []]}


def write_script(topic, tries=4):
    adj, photos = toolkit_text()
    feedback, err, best = "", None, None
    for i in range(tries):
        prompt = PROMPT.format(topic=topic, photos=photos, adjustments=adj, regions=", ".join(REGIONS),
                               items=[f"{k}..." for k in ADJ_SPECS], feedback=feedback)
        try:
            sc = validate(json.loads(llm(prompt)))
        except Exception as e:
            err = e
            print(f"  draft {i + 1} rejected: {e}")
            feedback = f"\nYour previous draft was rejected by the validator: {e}. Fix it.\n"
            continue
        try:
            rv = review(sc)
        except Exception as e:
            print(f"  quality gate unavailable ({e}) - accepting the validated draft")
            return sc
        print(f"  quality gate: score {rv['score']}/10, {len(rv['critical'])} real error(s)")
        sc["review"] = rv
        if not rv["critical"] and rv["score"] >= 7:
            return sc
        if not rv["critical"] and (best is None or rv["score"] > best["review"]["score"]):
            best = sc
        err = "; ".join(rv["critical"] + rv["minor"])[:500]
        feedback = "\nA senior instructor REJECTED the previous draft. Fix every point:\n- " + "\n- ".join(rv["critical"] + rv["minor"]) + "\n"
    if best:
        return best
    raise RuntimeError(f"No usable script: {err}")
