#!/usr/bin/env python3
"""
Publish research results to a Jekyll site as a hidden research post.

Converts a completed consortium workspace into a Jekyll-compatible markdown
file with MathJax-enabled LaTeX, and optionally pushes to deploy.

Usage:
    python scripts/publish_results.py \\
        --workspace results/consortium_20260330_143022 \\
        --site-repo ~/Documents/Oxford_DPhil/le-big-mac.github.io \\
        --push
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].strip("-")


def _read_file(path: str, max_chars: int = 50000) -> str:
    """Read a file, truncating if too large."""
    try:
        with open(path) as f:
            content = f.read(max_chars)
        if len(content) == max_chars:
            content += "\n\n... (truncated)"
        return content
    except Exception:
        return ""


def _read_json(path: str) -> dict:
    """Read a JSON file, returning empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _extract_title(workspace: Path, metadata: dict) -> str:
    """Extract a title from the workspace metadata or task."""
    task = metadata.get("task_preview", "")
    if task:
        # Use first sentence or first 100 chars
        first_sentence = task.split(".")[0].strip()
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."
        return first_sentence
    return f"Research Run {workspace.name}"


def _extract_proposal(workspace: Path) -> str:
    """Extract the research proposal if it exists."""
    for candidate in [
        "paper_workspace/research_proposal.md",
        "research_proposal.md",
    ]:
        path = workspace / candidate
        if path.exists():
            return _read_file(str(path))
    return ""


def _extract_paper(workspace: Path) -> str:
    """Extract the final paper content (LaTeX or markdown)."""
    # Prefer markdown if available
    for candidate in ["final_paper.md", "paper_workspace/final_paper.md"]:
        path = workspace / candidate
        if path.exists():
            return _read_file(str(path))

    # Fall back to LaTeX — wrap in display blocks for MathJax
    for candidate in ["final_paper.tex", "paper_workspace/final_paper.tex"]:
        path = workspace / candidate
        if path.exists():
            tex = _read_file(str(path))
            # Strip LaTeX preamble/document wrappers, keep content
            tex = re.sub(
                r"\\documentclass.*?\\begin\{document\}",
                "", tex, flags=re.DOTALL,
            )
            tex = tex.replace("\\end{document}", "")
            tex = tex.strip()
            if tex:
                return f"```latex\n{tex}\n```"
    return ""


def _extract_claims(workspace: Path) -> str:
    """Extract claim graph summary."""
    cg_path = workspace / "math_workspace" / "claim_graph.json"
    if not cg_path.exists():
        return ""

    cg = _read_json(str(cg_path))
    claims = cg.get("claims", [])
    if not claims:
        return ""

    lines = ["## Claim Graph\n"]
    for c in claims:
        status = c.get("status", "unknown")
        emoji = {"accepted": "+", "proven": "+", "verified": "+",
                 "rejected": "x", "failed": "x"}.get(status, "-")
        lines.append(
            f"- [{emoji}] **{c.get('id', '?')}** ({status}): "
            f"{c.get('statement', 'N/A')[:200]}"
        )
    return "\n".join(lines)


def _extract_experiments(workspace: Path) -> str:
    """Extract experiment results summary."""
    exp_dir = workspace / "experiment_workspace"
    if not exp_dir.exists():
        return ""

    results_path = exp_dir / "all_results.json"
    if results_path.exists():
        results = _read_json(str(results_path))
        return (
            "## Experiment Results\n\n"
            f"```json\n{json.dumps(results, indent=2, default=str)[:5000]}\n```"
        )
    return ""


def _extract_budget(workspace: Path) -> str:
    """Extract budget/invocation summary."""
    budget = _read_json(str(workspace / "cli_budget_state.json"))
    summary = _read_json(str(workspace / "run_summary.json"))

    parts = []
    if budget:
        parts.append(f"- Invocations: {budget.get('invocation_count', '?')}")
        secs = budget.get("total_seconds", 0)
        parts.append(f"- Wall-clock time: {secs / 60:.1f} min")
    if summary:
        duration = summary.get("duration_seconds", 0)
        parts.append(f"- Total duration: {duration / 60:.1f} min")
        stages = summary.get("stages_completed", [])
        if stages:
            parts.append(f"- Stages completed: {len(stages)}")

    if parts:
        return "## Run Metadata\n\n" + "\n".join(parts)
    return ""


def _copy_images(workspace: Path, site_repo: Path, run_id: str) -> str:
    """Copy images from workspace to site assets. Returns asset base path."""
    asset_dir = site_repo / "assets" / "images" / "research" / run_id
    source_dirs = [
        workspace / "paper_workspace",
        workspace / "experiment_workspace",
        workspace / "math_workspace",
    ]

    copied = 0
    for src_dir in source_dirs:
        if not src_dir.exists():
            continue
        for f in src_dir.rglob("*"):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf"):
                rel = f.relative_to(workspace)
                dst = asset_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                copied += 1

    if copied:
        print(f"Copied {copied} images to {asset_dir}")
    return f"/assets/images/research/{run_id}"


def build_post(workspace: Path, site_repo: Path) -> tuple[str, str]:
    """Build a Jekyll research post from a workspace. Returns (filename, content)."""
    run_id = workspace.name
    metadata = _read_json(str(workspace / "experiment_metadata.json"))
    summary = _read_json(str(workspace / "run_summary.json"))

    title = _extract_title(workspace, metadata)
    date = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(title)
    filename = f"{date}-{slug}.md"

    # Copy images
    _copy_images(workspace, site_repo, run_id)

    # Build sections
    sections = []

    proposal = _extract_proposal(workspace)
    if proposal:
        sections.append(f"## Research Proposal\n\n{proposal}")

    claims = _extract_claims(workspace)
    if claims:
        sections.append(claims)

    experiments = _extract_experiments(workspace)
    if experiments:
        sections.append(experiments)

    paper = _extract_paper(workspace)
    if paper:
        sections.append(f"## Paper\n\n{paper}")

    budget = _extract_budget(workspace)
    if budget:
        sections.append(budget)

    # Compose
    task = metadata.get("task_preview", summary.get("task", ""))
    status = "complete" if (workspace / "STATUS.txt").exists() else "in_progress"

    body = "\n\n---\n\n".join(sections) if sections else "*No results found in workspace.*"

    content = f"""---
layout: research
title: "{title}"
date: {date}
author: "PoggioAI Consortium"
usemathjax: true
status: "{status}"
pipeline_run: "{run_id}"
---

> **Task:** {task[:500]}

{body}
"""
    return filename, content


def main():
    parser = argparse.ArgumentParser(description="Publish research results to Jekyll site")
    parser.add_argument("--workspace", required=True, help="Path to consortium workspace")
    parser.add_argument(
        "--site-repo",
        default=os.environ.get("RESEARCH_SITE_REPO"),
        help="Path to Jekyll site repo (default: $RESEARCH_SITE_REPO)",
    )
    parser.add_argument("--push", action="store_true", help="Git add, commit, and push")
    args = parser.parse_args()

    if not args.site_repo:
        parser.error("--site-repo is required (or set RESEARCH_SITE_REPO env var)")

    workspace = Path(args.workspace).resolve()
    site_repo = Path(args.site_repo).resolve()

    if not workspace.exists():
        parser.error(f"Workspace not found: {workspace}")
    if not site_repo.exists():
        parser.error(f"Site repo not found: {site_repo}")

    research_dir = site_repo / "_research"
    research_dir.mkdir(exist_ok=True)

    filename, content = build_post(workspace, site_repo)
    post_path = research_dir / filename

    with open(post_path, "w") as f:
        f.write(content)
    print(f"Published: {post_path}")

    if args.push:
        subprocess.run(["git", "add", "-A"], cwd=site_repo, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Add research: {filename}"],
            cwd=site_repo, check=True,
        )
        subprocess.run(["git", "push"], cwd=site_repo, check=True)
        print("Pushed to remote.")


if __name__ == "__main__":
    main()
