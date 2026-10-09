"""
Failure alerts by email - you only hear from the bot when a video did NOT get posted.

  python notify.py --failed run.log    # called by the posting job when it fails (sends the log tail)
  python notify.py --watchdog          # called a while after each posting slot: alerts if nothing was posted
  python notify.py --test              # send a test email

Sends through Gmail SMTP from your own account to yourself:
  secret   SMTP_PASSWORD  = a Gmail App Password (myaccount.google.com/apppasswords)
  variable SMTP_USER      = the Gmail address that sends
  variable ALERT_EMAIL    = where alerts go (defaults to SMTP_USER)
"""
import json
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

BASE = Path(__file__).parent


def run_url():
    if os.getenv("GITHUB_RUN_ID"):
        return f'{os.getenv("GITHUB_SERVER_URL", "https://github.com")}/{os.getenv("GITHUB_REPOSITORY")}/actions/runs/{os.getenv("GITHUB_RUN_ID")}'
    return "(local run)"


def send_email(subject, body):
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    to = os.getenv("ALERT_EMAIL", "").strip() or user
    if not (user and password and to):
        print("Email not configured (SMTP_USER / SMTP_PASSWORD / ALERT_EMAIL) - alert not sent:")
        print(subject, body, sep="\n")
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, f"DataVelvet Screen Demos <{user}>", to
    msg.set_content(body)
    with smtplib.SMTP_SSL(os.getenv("SMTP_HOST", "smtp.gmail.com"), int(os.getenv("SMTP_PORT", "465")), timeout=30) as s:
        s.login(user, password)
        s.send_message(msg)
    print(f"Alert emailed to {to}")
    return True


def last_post():
    state_file = BASE / "state.json"
    if not state_file.exists():
        return None
    posts = json.loads(state_file.read_text(encoding="utf-8")).get("posts", [])
    return posts[-1] if posts else None


def failed(log_path):
    log = Path(log_path).read_text(encoding="utf-8", errors="replace") if Path(log_path).exists() else "(no log captured)"
    lines = [l for l in log.splitlines() if l.strip()]
    problems = [l for l in lines if any(k in l for k in ("❌", "⚠️", "Error", "error", "failed", "Traceback"))]
    p = last_post() or {}
    posted = []
    if p.get("youtube_id"):
        posted.append(f"YouTube: https://youtube.com/shorts/{p['youtube_id']}")
    if p.get("instagram_id"):
        posted.append(f"Instagram: posted (media {p['instagram_id']})")
    body = (
        "A scheduled DataVelvet screen-demo video was NOT posted everywhere.\n\n"
        f"Run: {run_url()}\n"
        f"Topic: {p.get('topic', 'unknown')}\n\n"
        "What did get posted:\n  " + ("\n  ".join(posted) if posted else "nothing") + "\n\n"
        "Problems found in the log:\n  " + ("\n  ".join(problems[-15:]) if problems else "(see full log)") + "\n\n"
        "Last 40 log lines:\n" + "\n".join(lines[-40:])
    )
    send_email("⚠️ DataVelvet: video not posted", body)


def watchdog(hours=3.0):
    p = last_post()
    now = datetime.now(timezone.utc)
    if p:
        when = datetime.fromisoformat(p["time"])
        if now - when <= timedelta(hours=hours) and p.get("youtube_id") and p.get("instagram_id"):
            print(f"OK - last full post {p.get('topic')!r} at {p['time']}")
            return
    detail = "no posts recorded yet" if not p else (
        f"last post: {p.get('topic')!r} at {p['time']} "
        f"(YouTube {'ok' if p.get('youtube_id') else 'MISSING'}, Instagram {'ok' if p.get('instagram_id') else 'MISSING'})")
    body = (
        f"No video was posted to both YouTube and Instagram in the last {hours:.0f} hours.\n\n"
        f"{detail}\n\n"
        "This usually means GitHub skipped the scheduled run, the run is still stuck, or it failed.\n"
        f"Check the Actions tab: https://github.com/{os.getenv('GITHUB_REPOSITORY', '')}/actions\n"
    )
    send_email("⚠️ DataVelvet: no video posted recently", body)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--failed":
        failed(sys.argv[2] if len(sys.argv) > 2 else "run.log")
    elif len(sys.argv) >= 2 and sys.argv[1] == "--watchdog":
        watchdog()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--test":
        send_email("✅ DataVelvet alerts are working", "This is a test. You'll only get emails like this when a video fails to post.")
    else:
        print(__doc__)
