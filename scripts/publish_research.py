#!/usr/bin/env python3
"""
Publish completed research to the GitHub Pages website.

Takes a pipeline workspace directory, extracts the final paper (markdown),
wraps it in Jekyll frontmatter for the _research collection, commits it
to the website repo, and pushes.

Usage:
    python scripts/publish_research.py --workspace results/consortium_20260401_120000 \
        --website-repo ~/le-big-mac.github.io

    # With explicit metadata:
    python scripts/publish_research.py --workspace results/consortium_20260401_120000 \
        --website-repo ~/le-big-mac.github.io \
        --title "My Research Title" \
        --idea-id abc12345

Environment variables:
    WEBSITE_REPO_PATH  — path to the GitHub Pages repo (default: ~/le-big-mac.github.io)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text[:80].strip('-')


def _find_paper(workspace: str) -> str | None:
    """Find the final paper markdown in a workspace."""
    candidates = [
        os.path.join(workspace, "final_paper.md"),
        os.path.join(workspace, "paper_workspace", "final_paper.md"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _extract_title_from_markdown(content: str) -> str | None:
    """Extract the first H1 heading from markdown content."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _extract_summary(content: str, max_chars: int = 200) -> str:
    """Extract a short summary from the first paragraph of content."""
    in_frontmatter = False
    for line in content.splitlines():
        line = line.strip()
        if line == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if line.startswith("#"):
            continue
        if line:
            if len(line) > max_chars:
                return line[:max_chars].rsplit(' ', 1)[0] + "..."
            return line
    return ""


def _strip_existing_frontmatter(content: str) -> str:
    """Remove any existing YAML frontmatter from markdown content."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].lstrip("\n")
    return content


def publish(
    workspace: str,
    website_repo: str,
    title: str | None = None,
    idea_id: str | None = None,
    author: str = "Consortium AI",
) -> str | None:
    """
    Publish research from a workspace to the website repo.

    Returns the URL path of the published page, or None on failure.
    """
    # Find the paper
    paper_path = _find_paper(workspace)
    if not paper_path:
        print(f"[publish] No final_paper.md found in {workspace}")
        return None

    with open(paper_path) as f:
        content = f.read()

    content = _strip_existing_frontmatter(content)

    # Determine title
    if not title:
        title = _extract_title_from_markdown(content)
    if not title:
        # Try run_summary.json
        summary_path = os.path.join(workspace, "run_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                try:
                    summary = json.load(f)
                    title = summary.get("task", "")[:100]
                except json.JSONDecodeError:
                    pass
    if not title:
        title = "Untitled Research"

    summary_text = _extract_summary(content)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = _slugify(title)
    filename = f"{date_str}-{slug}.md"

    # Build Jekyll frontmatter
    frontmatter_lines = [
        "---",
        "layout: research-post",
        f"title: \"{title}\"",
        f"date: {date_str}",
        f"author: \"{author}\"",
        "usemathjax: true",
    ]
    if idea_id:
        frontmatter_lines.append(f"idea_id: \"{idea_id}\"")
    if summary_text:
        # Escape quotes in summary for YAML
        safe_summary = summary_text.replace('"', '\\"')
        frontmatter_lines.append(f"summary: \"{safe_summary}\"")
    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)

    # Write the file
    research_dir = os.path.join(website_repo, "_research")
    os.makedirs(research_dir, exist_ok=True)
    output_path = os.path.join(research_dir, filename)

    with open(output_path, "w") as f:
        f.write(frontmatter)
        f.write("\n\n")
        f.write(content)

    print(f"[publish] Written: {output_path}")

    # Git commit and push
    try:
        subprocess.run(
            ["git", "add", output_path],
            cwd=website_repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Add research: {title[:60]}"],
            cwd=website_repo, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=website_repo, check=True, capture_output=True, timeout=30,
        )
        print(f"[publish] Pushed to website repo.")
    except subprocess.CalledProcessError as e:
        print(f"[publish] Git operation failed: {e.stderr.decode()[:200] if e.stderr else e}")
        return None
    except subprocess.TimeoutExpired:
        print("[publish] Git push timed out (30s). Push manually.")
        return None

    url_path = f"/research/{slug}/"
    print(f"[publish] Published at: {url_path}")
    return url_path


def main():
    parser = argparse.ArgumentParser(description="Publish research to GitHub Pages website.")
    parser.add_argument("--workspace", required=True, help="Pipeline workspace directory")
    parser.add_argument(
        "--website-repo",
        default=os.environ.get("WEBSITE_REPO_PATH", os.path.expanduser("~/le-big-mac.github.io")),
        help="Path to the GitHub Pages repo",
    )
    parser.add_argument("--title", default=None, help="Override paper title")
    parser.add_argument("--idea-id", default=None, help="Idea queue ID")
    parser.add_argument("--author", default="Consortium AI", help="Author attribution")
    args = parser.parse_args()

    url = publish(
        workspace=args.workspace,
        website_repo=args.website_repo,
        title=args.title,
        idea_id=args.idea_id,
        author=args.author,
    )
    sys.exit(0 if url else 1)


if __name__ == "__main__":
    main()
