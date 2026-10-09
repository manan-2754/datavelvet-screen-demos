"""
YouTube Shorts uploader.

One-time setup on your PC (creates token.json, which holds a refresh token):
  1. console.cloud.google.com -> new project -> enable "YouTube Data API v3"
  2. OAuth consent screen -> External -> add yourself as a test user -> then click "Publish app"
     (apps left in "Testing" get refresh tokens that expire after 7 days)
  3. Credentials -> Create OAuth client ID -> Desktop app -> download JSON as client_secret.json here
  4. python youtube_upload.py --auth      (browser opens - pick the channel to post to)
  5. Put the contents of token.json in the GitHub secret YT_TOKEN_JSON
"""
import json
import os
import sys
import time
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

BASE = Path(__file__).parent
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]
TOKEN_FILE = BASE / "token.json"
CLIENT_SECRET = BASE / "client_secret.json"


def get_credentials(interactive=False):
    creds = None
    env = os.getenv("YT_TOKEN_JSON", "").strip()
    if env:
        creds = Credentials.from_authorized_user_info(json.loads(env), SCOPES)
    elif TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds and not creds.valid and creds.refresh_token:
        creds.refresh(Request())
    if creds and creds.valid:
        return creds
    if not interactive:
        raise RuntimeError("No valid YouTube credentials - run `python youtube_upload.py --auth` once on your PC")
    if not CLIENT_SECRET.exists():
        raise RuntimeError("client_secret.json not found - see the setup steps at the top of this file")
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    TOKEN_FILE.write_text(creds.to_json())
    return creds


def channel_name(youtube):
    items = youtube.channels().list(part="snippet", mine=True).execute().get("items", [])
    return items[0]["snippet"]["title"] if items else "(no channel)"


def upload_video(video_path, meta, privacy=None):
    youtube = build("youtube", "v3", credentials=get_credentials(), cache_discovery=False)
    body = {
        "snippet": {
            "title": meta["title"][:100],
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "categoryId": "27",  # Education
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus": privacy or os.getenv("YT_PRIVACY") or "public",
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True,   # YouTube's "Altered or synthetic content" (AI) disclosure
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=16 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response, retries = None, 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"  YouTube upload {int(status.progress() * 100)}%")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 5:
                retries += 1
                time.sleep(2 ** retries)
                continue
            raise
    vid = response["id"]
    print(f"✅ YouTube: https://youtube.com/shorts/{vid}")
    return vid


if __name__ == "__main__":
    if "--auth" in sys.argv:
        creds = get_credentials(interactive=True)
        yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
        print(f"✅ Authorised for channel: {channel_name(yt)}")
        print(f"token saved to {TOKEN_FILE} - paste its contents into the GitHub secret YT_TOKEN_JSON")
    elif len(sys.argv) >= 2:
        folder = Path(sys.argv[1])
        upload_video(folder / "video.mp4", json.loads((folder / "meta.json").read_text(encoding="utf-8")))
    else:
        print(__doc__)
