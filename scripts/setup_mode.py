#!/usr/bin/env python3
"""
PoggioAI/MSc quick setup — detects environment and prepares configuration.

Usage:
    ./poggioaimsc setup              # auto-detect and configure
    ./poggioaimsc setup --mode local # force a specific mode
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_mode() -> str:
    """Auto-detect the best deployment mode for this environment."""
    # Check SLURM
    if shutil.which("sbatch"):
        return "hpc"
    if os.environ.get("CONSORTIUM_SLURM_ENABLED", "0") in ("1", "true"):
        return "hpc"

    # Check Tinker
    if os.environ.get("TINKER_API_KEY"):
        return "tinker"

    return "local"


def setup_env_file(mode: str) -> None:
    """Create .env from .env.example if it doesn't exist, with mode hints."""
    env_path = REPO_ROOT / ".env"
    env_example = REPO_ROOT / ".env.example"

    if env_path.exists():
        print(f"  .env already exists — skipping creation")
        return

    lines = []
    if env_example.exists():
        lines = env_example.read_text().splitlines()
    else:
        lines = [
            "# PoggioAI/MSc Environment Configuration",
            "# CLI-agent mode: no LLM API keys needed.",
            "# Agents run via local CLI tools (claude, codex, gemini).",
            "",
            "# Optional: search and GPU training keys",
            "# SERPER_API_KEY=...",
            "# TINKER_API_KEY=...",
            "",
        ]

    # Add mode-specific hints
    lines.append("")
    lines.append(f"# --- Deployment mode: {mode} ---")
    lines.append(f"CONSORTIUM_MODE={mode}")

    if mode == "tinker":
        lines.append("")
        lines.append("# Tinker API key (https://auth.thinkingmachines.ai/sign-up)")
        lines.append("# TINKER_API_KEY=your-key-here")

    if mode == "hpc":
        lines.append("")
        lines.append("# HPC/SLURM settings")
        lines.append("CONSORTIUM_SLURM_ENABLED=1")
        lines.append("# CONDA_INIT_SCRIPT=/path/to/conda.sh")
        lines.append("# CONDA_ENV_PREFIX=/path/to/conda_envs/consortium")

    env_path.write_text("\n".join(lines) + "\n")
    print(f"  Created .env with {mode} mode defaults")


def setup_llm_config() -> None:
    """Copy .llm_config.yaml.example to .llm_config.yaml if needed."""
    config = REPO_ROOT / ".llm_config.yaml"
    example = REPO_ROOT / ".llm_config.yaml.example"

    if config.exists():
        print(f"  .llm_config.yaml already exists — skipping")
        return

    if example.exists():
        shutil.copy2(example, config)
        print(f"  Created .llm_config.yaml from example (customize as needed)")
    else:
        print(f"  WARNING: .llm_config.yaml.example not found")


def run_preflight(mode: str) -> bool:
    """Run preflight checks for the detected mode."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "preflight_check.py")]
    cmd.extend(["--mode", mode])
    if mode == "hpc":
        cmd.append("--with-experiment")
    print(f"\n  Running preflight checks...")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="PoggioAI/MSc quick setup")
    parser.add_argument(
        "--mode",
        choices=["local", "tinker", "hpc"],
        default=None,
        help="Force a specific deployment mode (default: auto-detect)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PoggioAI/MSc Setup")
    print("=" * 60)

    # Detect or use explicit mode
    if args.mode:
        mode = args.mode
        print(f"\n  Mode: {mode} (explicit)")
    else:
        mode = detect_mode()
        print(f"\n  Detected mode: {mode}")

    # Print mode description
    descriptions = {
        "local": "Laptop-only — CPU experiments, no external compute needed",
        "tinker": "Local orchestrator + Tinker API (thinkingmachines.ai) for GPU training",
        "hpc": "HPC cluster with SLURM job scheduler for GPU experiments",
    }
    print(f"  {descriptions[mode]}")

    # Setup steps
    print(f"\n--- Configuration ---")
    setup_env_file(mode)
    setup_llm_config()

    # Preflight
    print(f"\n--- Validation ---")
    preflight_ok = run_preflight(mode)

    # Summary
    print(f"\n--- Quick Start ---")
    print(f"  # Dry run (validates setup, no API cost):")
    print(f"  ./poggioaimsc run --mode {mode} --task \"test\" --dry-run")
    print()
    print(f"  # Real run:")
    print(f"  ./poggioaimsc run --mode {mode} --task \"Your research question here\"")

    if mode == "local":
        print()
        print(f"  Note: Local mode runs experiments on CPU (small scale).")
        print(f"  Output defaults to markdown (no LaTeX needed).")

    if mode == "tinker":
        print()
        print(f"  Note: Set TINKER_API_KEY in .env before running.")
        print(f"  Sign up: https://auth.thinkingmachines.ai/sign-up")

    if mode == "hpc":
        print()
        print(f"  Note: Edit engaging_config.yaml for your cluster's partitions.")
        print(f"  Set CONDA_INIT_SCRIPT and CONDA_ENV_PREFIX in .env.")

    print()
    if preflight_ok:
        print("  Setup complete.")
    else:
        print("  Setup complete (with warnings — review preflight output above).")


if __name__ == "__main__":
    main()
