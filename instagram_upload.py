"""
Instagram Reels uploader (Instagram API with Instagram Login - no Facebook Page needed).

Instagram downloads the video from a public URL. In the GitHub Actions job the 1080x1920 copy is attached
to a release ("ig-media") in this repo, Instagram fetches it, and the asset is deleted after publishing.
The repo must be public so Instagram can reach the file. Instagram caps Reels at 1920px wide.

One-time setup:
  1. Switch your Instagram account to Professional (Creator or Business)
  2. developers.facebook.com -> Create app -> add "Instagram" product -> "API setup with Instagram login"
  3. Add your Instagram account as a tester / generate a token with instagram_business_basic +
     instagram_business_content_publish -> this is a long-lived token (60 days)
  4. GitHub secrets: IG_ACCESS_TOKEN, IG_USER_ID (python instagram_upload.py --whoami prints the id)
  The cloud job refreshes the token automatically (needs secret GH_PAT so it can save the new one).
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

HOST = os.getenv("IG_GRAPH_HOST", "graph.instagram.com")
VERSION = os.getenv("IG_API_VERSION", "v23.0")


def _token():
    tok = os.getenv("IG_ACCESS_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("IG_ACCESS_TOKEN is not set")
    return tok


def _user_id():
    uid = os.getenv("IG_USER_ID", "").strip()
    return uid or whoami()["id"]


def _check(r):
    if r.status_code >= 400:
        raise RuntimeError(f"Instagram API {r.status_code}: {r.text[:300]}")
    return r.json()


def whoami():
    return _check(requests.get(f"https://{HOST}/{VERSION}/me", params={"fields": "id,username", "access_token": _token()}, timeout=30))


def to_1080(video_path):
    from screen_engine import ffmpeg_exe
    out = Path(tempfile.gettempdir()) / f"ig_{Path(video_path).parent.name}.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_path), "-vf", "scale=1080:1920:flags=lanczos",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)], check=True)
    return out


def _gh(method, url, **kw):
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}
    headers.update(kw.pop("headers", {}))
    r = requests.request(method, url, headers=headers, timeout=kw.pop("timeout", 60), **kw)
    if r.status_code >= 400 and r.status_code != 404:
        raise RuntimeError(f"GitHub API {r.status_code}: {r.text[:200]}")
    return r


def host_on_github(path):
    """Attach the file to the 'ig-media' release of this repo and return (public_url, asset_api_url)."""
    repo = os.getenv("GITHUB_REPOSITORY")
    if not repo or not os.getenv("GITHUB_TOKEN"):
        raise RuntimeError("Instagram needs a public video URL - run inside GitHub Actions (GITHUB_TOKEN/GITHUB_REPOSITORY)")
    api = f"https://api.github.com/repos/{repo}/releases"
    r = _gh("GET", f"{api}/tags/ig-media")
    if r.status_code == 404:
        r = _gh("POST", api, json={"tag_name": "ig-media", "name": "Instagram media (temporary)",
                                   "body": "Temporary video files fetched by Instagram. Deleted after posting.",
                                   "prerelease": True})
    release = r.json()
    name = f"{int(time.time())}_{Path(path).name}"
    with open(path, "rb") as f:
        asset = _gh("POST", f"https://uploads.github.com/repos/{repo}/releases/{release['id']}/assets",
                    params={"name": name}, data=f, headers={"Content-Type": "video/mp4"}, timeout=600).json()
    return asset["browser_download_url"], asset["url"]


def upload_reel(video_path, caption):
    token, uid = _token(), _user_id()
    small = to_1080(video_path)
    url, asset_api = host_on_github(small)
    print(f"  Instagram: video hosted at {url}")
    try:
        fields = {"media_type": "REELS", "video_url": url, "caption": caption, "share_to_feed": "true",
                  "is_ai_generated": "true",   # Instagram's "AI info" self-disclosure label
                  "access_token": token}
        r = requests.post(f"https://{HOST}/{VERSION}/{uid}/media", data=fields, timeout=60)
        if r.status_code >= 400 and "is_ai_generated" in r.text:
            print(f"  ⚠️ Instagram rejected is_ai_generated ({r.text[:160]}) - posting with an AI note in the caption instead")
            fields.pop("is_ai_generated")
            fields["caption"] = caption + "\n\n(Made with AI)"
            r = requests.post(f"https://{HOST}/{VERSION}/{uid}/media", data=fields, timeout=60)
        container = _check(r)
        cid = container["id"]
        print("  Instagram: processing...")
        for _ in range(60):
            st = _check(requests.get(f"https://{HOST}/{VERSION}/{cid}",
                                     params={"fields": "status_code,status", "access_token": token}, timeout=30))
            if st.get("status_code") == "FINISHED":
                break
            if st.get("status_code") in ("ERROR", "EXPIRED"):
                raise RuntimeError(f"Instagram processing failed: {st}")
            time.sleep(10)
        else:
            raise RuntimeError("Instagram processing timed out")

        pub = _check(requests.post(f"https://{HOST}/{VERSION}/{uid}/media_publish",
                                   data={"creation_id": cid, "access_token": token}, timeout=60))
    finally:
        _gh("DELETE", asset_api)
        small.unlink(missing_ok=True)
    print(f"✅ Instagram: posted media {pub['id']}")
    return pub["id"]


def refresh_token():
    """Extend the 60-day token. If it changed and GH_PAT is set, save it back to the repo secret."""
    old = _token()
    r = requests.get(f"https://{HOST}/refresh_access_token",
                     params={"grant_type": "ig_refresh_token", "access_token": old}, timeout=30)
    if r.status_code != 200:
        print(f"Token refresh skipped: {r.status_code} {r.text[:200]}")
        return
    new = r.json()["access_token"]
    days = r.json().get("expires_in", 0) // 86400
    print(f"Instagram token refreshed - valid for {days} more days")
    if new != old and os.getenv("GH_PAT") and os.getenv("GITHUB_REPOSITORY"):
        print(f"::add-mask::{new}")
        subprocess.run(["gh", "secret", "set", "IG_ACCESS_TOKEN", "--repo", os.environ["GITHUB_REPOSITORY"], "--body", new],
                       check=True, env={**os.environ, "GH_TOKEN": os.environ["GH_PAT"]})
        print("Saved refreshed token to the IG_ACCESS_TOKEN secret")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
    if "--whoami" in sys.argv:
        print(whoami())
    elif "--refresh" in sys.argv:
        refresh_token()
    else:
        print(__doc__)
