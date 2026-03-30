#!/usr/bin/env python3
"""
Quick-submit a research idea as a campaign.

Takes a text idea, generates a campaign.yaml + task.txt, and initializes it.
The heartbeat service then picks it up and runs stages automatically.

Usage:
    python scripts/quick_research.py "Does training on paraphrased responses reduce hallucinations?"

    # With custom name:
    python scripts/quick_research.py --name "paraphrase-hallucination" "Does training on..."

    # Then run heartbeat to start:
    python scripts/campaign_heartbeat.py --campaign campaigns/active/campaign.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:50].strip("-")


def main():
    parser = argparse.ArgumentParser(description="Quick-submit a research idea")
    parser.add_argument("idea", help="Research idea (text)")
    parser.add_argument("--name", help="Campaign name (default: derived from idea)")
    parser.add_argument("--init", action="store_true", default=True,
                        help="Initialize campaign immediately (default: true)")
    parser.add_argument("--no-init", action="store_true",
                        help="Don't initialize — just create files")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(args.idea[:50])
    campaign_name = args.name or slug
    campaign_id = f"quick_{slug}_{timestamp}"

    # Create campaign directory
    campaign_dir = Path("campaigns/active")
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # Write task file
    task_path = campaign_dir / "task.txt"
    task_path.write_text(args.idea + "\n")
    print(f"Task written: {task_path}")

    # Read and fill template
    template_path = Path("campaigns/templates/quick_research.yaml")
    if not template_path.exists():
        print(f"Error: template not found at {template_path}", file=sys.stderr)
        return 1

    template = template_path.read_text()
    config = template.replace("{{CAMPAIGN_NAME}}", campaign_name)
    config = config.replace("{{CAMPAIGN_ID}}", campaign_id)
    config = config.replace("{{TASK_FILE}}", str(task_path))

    config_path = campaign_dir / "campaign.yaml"
    config_path.write_text(config)
    print(f"Campaign config: {config_path}")

    # Initialize campaign
    if args.init and not args.no_init:
        heartbeat = Path("scripts/campaign_heartbeat.py")
        if heartbeat.exists():
            result = subprocess.run(
                [sys.executable, str(heartbeat),
                 "--campaign", str(config_path), "--init"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print("Campaign initialized.")
                print(f"\nTo start: python scripts/campaign_heartbeat.py --campaign {config_path}")
            else:
                print(f"Init warning: {result.stderr[:500]}", file=sys.stderr)
        else:
            print(f"Heartbeat script not found at {heartbeat}. Campaign files created but not initialized.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
