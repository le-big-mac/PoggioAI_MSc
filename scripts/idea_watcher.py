#!/usr/bin/env python3
"""
Idea watcher — polls GitHub Issues for research ideas and slash commands.

Watches a GitHub repo for:
  1. New issues labeled "idea" → runs quick-pass pipeline
  2. Comments with slash commands on existing issues:
     /plan [feedback]       — re-run quick pass with feedback
     /full [feedback]       — run full pipeline (experiments + paper)
     /theory [feedback]     — run full pipeline with math agents
     /experiment [feedback] — run full pipeline, experiment track only
     /close                 — close the issue

Usage:
    python scripts/idea_watcher.py --repo le-big-mac/le-big-mac.github.io --interval 30

Environment variables:
    GITHUB_TOKEN            — GitHub PAT (needs issues:write on the repo)
    WEBSITE_REPO_PATH       — local clone of the GitHub Pages repo
    WEBSITE_DOMAIN          — e.g. le-big-mac.github.io
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time as _time
from datetime import datetime, timezone

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)

from pathlib import Path
from publish_results import build_post

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

_API = "https://api.github.com"


def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _gh_get(path: str) -> list | dict:
    resp = requests.get(f"{_API}{path}", headers=_gh_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


def _gh_post(path: str, data: dict) -> dict:
    resp = requests.post(f"{_API}{path}", headers=_gh_headers(), json=data, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _gh_patch(path: str, data: dict) -> dict:
    resp = requests.patch(f"{_API}{path}", headers=_gh_headers(), json=data, timeout=15)
    resp.raise_for_status()
    return resp.json()


def list_idea_issues(repo: str) -> list[dict]:
    """List open issues with the 'idea' label."""
    return _gh_get(f"/repos/{repo}/issues?labels=idea&state=open&per_page=50")


def get_issue_comments(repo: str, issue_number: int) -> list[dict]:
    return _gh_get(f"/repos/{repo}/issues/{issue_number}/comments?per_page=100")


def add_comment(repo: str, issue_number: int, body: str) -> dict:
    return _gh_post(f"/repos/{repo}/issues/{issue_number}/comments", {"body": body})


def add_label(repo: str, issue_number: int, label: str) -> None:
    try:
        _gh_post(f"/repos/{repo}/issues/{issue_number}/labels", {"labels": [label]})
    except Exception:
        pass  # label may already exist


def remove_label(repo: str, issue_number: int, label: str) -> None:
    try:
        requests.delete(
            f"{_API}/repos/{repo}/issues/{issue_number}/labels/{label}",
            headers=_gh_headers(), timeout=10,
        )
    except Exception:
        pass


def close_issue(repo: str, issue_number: int) -> None:
    _gh_patch(f"/repos/{repo}/issues/{issue_number}", {"state": "closed"})


# ---------------------------------------------------------------------------
# Slash command parsing
# ---------------------------------------------------------------------------

COMMANDS = {"plan", "full", "theory", "experiment", "close"}


def parse_command(text: str) -> tuple[str | None, str]:
    """Parse a slash command from comment text. Returns (command, feedback) or (None, "")."""
    text = text.strip()
    m = re.match(r"^/(\w+)\s*(.*)", text, re.DOTALL)
    if m and m.group(1).lower() in COMMANDS:
        return m.group(1).lower(), m.group(2).strip()
    return None, ""


# ---------------------------------------------------------------------------
# Pipeline launching
# ---------------------------------------------------------------------------

def _parse_workspace_from_output(line: str) -> str | None:
    for pattern in (r"Created workspace:\s*(.+)", r"Resuming from:\s*(.+)"):
        m = re.search(pattern, line)
        if m:
            return m.group(1).strip()
    return None


def launch_pipeline(
    task: str,
    extra_args: list[str] | None = None,
) -> tuple[int, str | None]:
    """Launch a consortium pipeline run. Returns (exit_code, workspace_path)."""
    cmd = [
        sys.executable, "-u",
        os.path.join(_REPO_ROOT, "launch_multiagent.py"),
        "--task", task,
        "--no-steering",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[watcher] Launching: {' '.join(cmd[:6])}...")
    workspace = None

    proc = subprocess.Popen(
        cmd, cwd=_REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        if workspace is None:
            ws = _parse_workspace_from_output(line)
            if ws:
                workspace = ws
    proc.wait()

    if workspace and not os.path.isabs(workspace):
        workspace = os.path.join(_REPO_ROOT, workspace)

    return proc.returncode, workspace


def build_task_with_feedback(idea: str, feedback: str, prior_plan_path: str | None = None) -> str:
    """Combine the original idea with user feedback into a task prompt."""
    parts = [idea]

    if prior_plan_path and os.path.isfile(prior_plan_path):
        with open(prior_plan_path) as f:
            prior = f.read()
        parts.append(f"\n\n--- Prior research plan ---\n{prior}\n---")

    if feedback:
        parts.append(f"\n\n--- User feedback ---\n{feedback}\n---")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _pipeline_args_for_command(command: str) -> list[str]:
    if command == "plan":
        return ["--quick-pass"]
    elif command == "theory":
        return ["--enable-math-agents"]
    elif command == "experiment":
        return []
    elif command == "full":
        return []
    return []


def _publish_and_comment(
    repo: str,
    issue_number: int,
    workspace: str | None,
    command: str,
    exit_code: int,
) -> str | None:
    """Publish results and add a comment to the issue. Returns the URL path or None."""
    website_repo = os.environ.get(
        "WEBSITE_REPO_PATH",
        os.path.expanduser("~/le-big-mac.github.io"),
    )
    domain = os.environ.get("WEBSITE_DOMAIN", "le-big-mac.github.io")

    url_path = None
    if workspace and exit_code == 0:
        try:
            site_repo = Path(website_repo)
            research_dir = site_repo / "_research"
            research_dir.mkdir(exist_ok=True)

            filename, content = build_post(Path(workspace), site_repo, issue_number=issue_number)
            post_path = research_dir / filename
            with open(post_path, "w") as f:
                f.write(content)

            # Git commit and push
            subprocess.run(["git", "add", "-A"], cwd=site_repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Add research: {filename}"],
                cwd=site_repo, check=True, capture_output=True,
            )
            subprocess.run(["git", "push"], cwd=site_repo, check=True, capture_output=True, timeout=30)

            # Derive URL from filename
            slug = filename.replace(".md", "").split("-", 3)[-1] if "-" in filename else filename.replace(".md", "")
            url_path = f"/research/{slug}/"
            print(f"[watcher] Published: {post_path}")
        except Exception as e:
            print(f"[watcher] Publish failed: {e}")

    if exit_code == 0:
        mode = {"plan": "Quick assessment", "full": "Full analysis",
                "theory": "Theory analysis", "experiment": "Experiment analysis"}.get(command, "Analysis")
        body = f"**{mode} complete.**\n\n"
        if url_path:
            body += f"Read the results: [https://{domain}{url_path}](https://{domain}{url_path})\n\n"
        if workspace:
            body += f"Workspace: `{workspace}`\n"
        body += "\nCommands: `/plan`, `/full`, `/theory`, `/experiment`, `/close`"
        add_comment(repo, issue_number, body)
    else:
        body = f"**Run failed** (exit code {exit_code}).\n\n"
        if workspace:
            body += f"Check logs in `{workspace}`\n"
        body += "\nYou can retry with `/plan`, `/full`, etc."
        add_comment(repo, issue_number, body)

    return url_path


def handle_new_issue(repo: str, issue: dict) -> None:
    """Handle a new issue labeled 'idea': run quick pass."""
    number = issue["number"]
    idea = f"{issue['title']}\n\n{issue.get('body', '') or ''}".strip()

    print(f"[watcher] New idea: #{number} — {issue['title'][:80]}")
    add_label(repo, number, "running")
    add_comment(repo, number, "Starting quick assessment... I'll comment here when results are ready.")

    exit_code, workspace = launch_pipeline(idea, ["--quick-pass"])
    remove_label(repo, number, "running")

    if exit_code == 0:
        add_label(repo, number, "assessed")
    else:
        add_label(repo, number, "failed")

    _publish_and_comment(repo, number, workspace, "plan", exit_code)


def handle_command(repo: str, issue: dict, command: str, feedback: str) -> None:
    """Handle a slash command on an existing issue."""
    number = issue["number"]
    idea = f"{issue['title']}\n\n{issue.get('body', '') or ''}".strip()

    if command == "close":
        add_comment(repo, number, "Closing this investigation. Thanks!")
        close_issue(repo, number)
        return

    # Find prior workspace for context (if re-running plan)
    prior_plan = None
    if command == "plan":
        # Look for existing final_paper.md in the most recent workspace for this issue
        results_dir = os.path.join(_REPO_ROOT, "results")
        if os.path.isdir(results_dir):
            for d in sorted(os.listdir(results_dir), reverse=True):
                fp = os.path.join(results_dir, d, "final_paper.md")
                if os.path.isfile(fp):
                    prior_plan = fp
                    break

    task = build_task_with_feedback(idea, feedback, prior_plan)
    pipeline_args = _pipeline_args_for_command(command)

    mode = {"plan": "quick assessment", "full": "full analysis",
            "theory": "theory analysis", "experiment": "experiment analysis"}[command]
    msg = f"Starting {mode}..."
    if feedback:
        msg += f"\n\nFeedback noted:\n> {feedback[:500]}"
    add_comment(repo, number, msg)
    add_label(repo, number, "running")

    exit_code, workspace = launch_pipeline(task, pipeline_args)
    remove_label(repo, number, "running")

    if exit_code == 0 and command != "plan":
        add_label(repo, number, "completed")

    _publish_and_comment(repo, number, workspace, command, exit_code)


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------

def _load_seen(state_path: str) -> dict:
    """Load seen issues/comments state."""
    if os.path.isfile(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"issues": [], "comments": {}}


def _save_seen(state_path: str, seen: dict) -> None:
    with open(state_path, "w") as f:
        json.dump(seen, f, indent=2)


def poll_once(repo: str, state_path: str) -> bool:
    """Poll for new issues and commands. Returns True if work was done."""
    seen = _load_seen(state_path)
    seen_issues = set(seen.get("issues", []))
    seen_comments = seen.get("comments", {})  # {issue_number_str: last_comment_id}

    issues = list_idea_issues(repo)
    did_work = False

    for issue in issues:
        number = issue["number"]
        num_str = str(number)

        # New issue?
        if number not in seen_issues:
            seen_issues.add(number)
            seen["issues"] = list(seen_issues)
            _save_seen(state_path, seen)

            handle_new_issue(repo, issue)
            did_work = True
            # Refresh seen state after handling (in case of crash/restart)
            seen = _load_seen(state_path)
            continue

        # Check for new slash commands in comments
        try:
            comments = get_issue_comments(repo, number)
        except Exception as e:
            print(f"[watcher] Failed to fetch comments for #{number}: {e}")
            continue

        last_seen_id = seen_comments.get(num_str, 0)
        new_comments = [c for c in comments if c["id"] > last_seen_id]

        if new_comments:
            # Update last seen to latest comment
            seen_comments[num_str] = max(c["id"] for c in comments)
            seen["comments"] = seen_comments
            _save_seen(state_path, seen)

        for comment in new_comments:
            command, feedback = parse_command(comment.get("body", ""))
            if command:
                handle_command(repo, issue, command, feedback)
                did_work = True
                # Only handle one command per poll to avoid overload
                return True

    return did_work


def main():
    parser = argparse.ArgumentParser(description="GitHub Issues idea watcher")
    parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    parser.add_argument(
        "--state-file",
        default=os.path.join(_REPO_ROOT, ".idea_watcher_state.json"),
        help="Path to state file tracking seen issues/comments",
    )
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    if not os.environ.get("GITHUB_TOKEN"):
        print("[watcher] WARNING: GITHUB_TOKEN not set. API rate limits will be very low.")

    if args.once:
        did_work = poll_once(args.repo, args.state_file)
        sys.exit(0 if did_work else 1)

    print(f"[watcher] Polling {args.repo} every {args.interval}s")
    print(f"[watcher] State file: {args.state_file}")

    while True:
        try:
            did_work = poll_once(args.repo, args.state_file)
            if not did_work:
                _time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[watcher] Shutting down.")
            break
        except Exception as e:
            print(f"[watcher] Error: {e}")
            _time.sleep(args.interval)


if __name__ == "__main__":
    main()
