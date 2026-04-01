#!/usr/bin/env python3
"""
Idea watcher — polls the idea queue and launches pipeline runs.

Designed to run alongside the webhook server. Checks the idea queue
every --interval seconds, picks up the next pending idea, launches a
consortium pipeline run, and sends WhatsApp notifications on start
and completion.

Usage:
    python scripts/idea_watcher.py --interval 30
    python scripts/idea_watcher.py --interval 30 --queue-path /path/to/ideas.json
    python scripts/idea_watcher.py --once  # process one idea and exit

Environment variables (for WhatsApp notifications):
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
    WHATSAPP_FROM  — e.g. whatsapp:+14155238886
    WHATSAPP_TO    — e.g. whatsapp:+447123456789
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time as _time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)

from consortium.interaction.idea_queue import (
    next_pending, mark_running, mark_completed, mark_failed,
)
from publish_research import publish as publish_research


def _send_whatsapp(message: str) -> None:
    """Send a WhatsApp message using Twilio. Fail-safe."""
    import requests

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("WHATSAPP_FROM")
    to_number = os.environ.get("WHATSAPP_TO")

    if not all([account_sid, auth_token, from_number, to_number]):
        print(f"[idea_watcher] WhatsApp not configured, printing only: {message}")
        return

    try:
        resp = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            auth=(account_sid, auth_token),
            data={"From": from_number, "To": to_number, "Body": message[:1600]},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[idea_watcher] WhatsApp delivery failed: {e}")


def _parse_workspace_from_output(stdout: str) -> str | None:
    """Extract workspace path from pipeline stdout.

    The runner prints 'Created workspace: results/consortium_{timestamp}'
    on a new run, or 'Resuming from: ...' on a resume.
    """
    import re
    for pattern in (r"Created workspace:\s*(.+)", r"Resuming from:\s*(.+)"):
        m = re.search(pattern, stdout)
        if m:
            return m.group(1).strip()
    return None


def _launch_pipeline(idea_text: str, extra_args: list[str] | None = None) -> tuple[int, str | None]:
    """Launch a consortium pipeline run. Returns (exit_code, workspace_path).

    Streams output line-by-line so logs are visible in real time, while
    also capturing workspace path from the output.
    """
    cmd = [
        sys.executable, "-u",  # unbuffered so lines appear immediately
        os.path.join(_REPO_ROOT, "launch_multiagent.py"),
        "--task", idea_text,
        "--no-steering",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[idea_watcher] Launching: {' '.join(cmd[:6])}...")
    workspace = None

    proc = subprocess.Popen(
        cmd, cwd=_REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,  # line-buffered
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        # Parse workspace path on the fly
        if workspace is None:
            ws = _parse_workspace_from_output(line)
            if ws:
                workspace = ws

    proc.wait()

    # Resolve relative path from the pipeline's CWD
    if workspace and not os.path.isabs(workspace):
        workspace = os.path.join(_REPO_ROOT, workspace)

    return proc.returncode, workspace


def process_one(queue_path: str | None, extra_args: list[str] | None = None) -> bool:
    """Process the next pending idea. Returns True if an idea was processed."""
    idea = next_pending(queue_path)
    if idea is None:
        return False

    idea_id = idea["id"]
    idea_text = idea["idea"]
    sender = idea["sender"]
    short_id = idea_id[:8]

    # Notify: starting
    start_msg = (
        f"[consortium] Starting analysis of your idea (ID: {short_id}):\n"
        f"\"{idea_text[:200]}{'...' if len(idea_text) > 200 else ''}\"\n"
        f"I'll let you know when results are ready."
    )
    _send_whatsapp(start_msg)
    print(f"[idea_watcher] Processing idea {short_id}: {idea_text[:80]}...")

    mark_running(idea_id, "(launching...)", queue_path)

    # Run the pipeline — it creates its own workspace under results/
    exit_code, workspace = _launch_pipeline(idea_text, extra_args)

    # Update queue entry with actual workspace path
    if workspace:
        mark_running(idea_id, workspace, queue_path)

    if exit_code == 0:
        mark_completed(idea_id, queue_path)

        # Publish to website
        url_note = ""
        if workspace:
            website_repo = os.environ.get(
                "WEBSITE_REPO_PATH",
                os.path.expanduser("~/le-big-mac.github.io"),
            )
            try:
                url_path = publish_research(
                    workspace=workspace,
                    website_repo=website_repo,
                    idea_id=short_id,
                )
                if url_path:
                    site_domain = os.environ.get("WEBSITE_DOMAIN", "le-big-mac.github.io")
                    url_note = f"\nRead it: https://{site_domain}{url_path}"
            except Exception as e:
                print(f"[idea_watcher] Publish failed (non-fatal): {e}")

        done_msg = (
            f"[consortium] ✓ Analysis complete for idea {short_id}!\n"
            f"\"{idea_text[:120]}{'...' if len(idea_text) > 120 else ''}\""
            f"{url_note}"
        )
        _send_whatsapp(done_msg)
        print(f"[idea_watcher] ✓ Idea {short_id} completed successfully.")
    else:
        mark_failed(idea_id, queue_path)
        ws_note = f"\nCheck logs in {workspace}" if workspace else ""
        fail_msg = (
            f"[consortium] ✗ Analysis failed for idea {short_id} (exit code {exit_code}).\n"
            f"\"{idea_text[:120]}{'...' if len(idea_text) > 120 else ''}\""
            f"{ws_note}"
        )
        _send_whatsapp(fail_msg)
        print(f"[idea_watcher] ✗ Idea {short_id} failed with exit code {exit_code}.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Idea queue watcher — polls and launches pipeline runs.")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
    parser.add_argument("--queue-path", default=None, help="Path to ideas.json")
    parser.add_argument("--once", action="store_true", help="Process one idea and exit")
    parser.add_argument(
        "extra_args", nargs="*",
        help="Extra args passed through to launch_multiagent.py (e.g. --enable-math-agents)",
    )
    args = parser.parse_args()

    if args.once:
        found = process_one(args.queue_path, args.extra_args)
        sys.exit(0 if found else 1)

    print(f"[idea_watcher] Polling every {args.interval}s. Queue: {args.queue_path or '(default)'}")
    print(f"[idea_watcher] Extra pipeline args: {args.extra_args or '(none)'}")

    while True:
        try:
            processed = process_one(args.queue_path, args.extra_args)
            if not processed:
                _time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[idea_watcher] Shutting down.")
            break
        except Exception as e:
            print(f"[idea_watcher] Error: {e}")
            _time.sleep(args.interval)


if __name__ == "__main__":
    main()
