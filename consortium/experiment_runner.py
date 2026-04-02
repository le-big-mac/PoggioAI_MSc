"""
Lightweight experiment stage runner — CLI agent replacement for AI-Scientist.

Runs experiments through 4 forced stages, each handled by a full CLI agent
that can write code, execute it, debug errors, and iterate. The Python
structure ensures all stages execute and context passes forward.

Replaces the heavyweight AI-Scientist-v2 subprocess pipeline while keeping
the enforced multi-stage structure that prevents agents from giving up early.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

STAGES = [
    {
        "name": "baseline",
        "description": "Initial implementation — write a basic working baseline experiment",
        "instructions": (
            "Write and run the initial baseline experiment code. "
            "Implement the core method described in the idea. "
            "Focus on getting a working version with reasonable defaults. "
            "Run the code and verify it produces results. Fix any errors. "
            "Save results to results.json and any plots to the plots/ directory."
        ),
    },
    {
        "name": "tuning",
        "description": "Baseline tuning — optimize hyperparameters and fix issues",
        "instructions": (
            "Review the baseline results from the previous stage. "
            "Tune hyperparameters (learning rate, batch size, epochs, etc.) "
            "to improve performance. Fix any issues found in the baseline. "
            "Run experiments with different configurations. "
            "Save the best configuration and updated results to results.json."
        ),
    },
    {
        "name": "creative",
        "description": "Creative research — try novel approaches and compare to baseline",
        "instructions": (
            "Review baseline and tuning results. Now try creative improvements: "
            "alternative architectures, loss functions, training strategies, "
            "data augmentation, or other novel approaches. "
            "Compare each approach against the tuned baseline. "
            "Document what works and what doesn't. "
            "Save comparative results to results.json."
        ),
    },
    {
        "name": "ablation",
        "description": "Ablation studies — systematically vary components",
        "instructions": (
            "Design and run ablation studies to understand which components "
            "of the best approach actually matter. Systematically remove or "
            "vary one component at a time while keeping others fixed. "
            "Create clear comparison tables/plots. "
            "Save ablation results to results.json and summary to ablation_summary.md."
        ),
    },
]


def _run_cli_agent(
    prompt: str,
    workspace_dir: str,
    backend: str = "claude",
    model: Optional[str] = None,
    timeout: int = 1800,
    session_id: Optional[str] = None,
    resume: bool = False,
    metadata: Optional[dict] = None,
) -> str:
    """Run a full CLI agent with tool use in the given workspace.

    Unlike cli_completion() (single-turn, no tools), this gives the agent
    file I/O, bash, and code execution capabilities so it can write code,
    run it, see errors, fix them, and iterate.

    Supports session resume to keep context across sequential stages.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    if backend == "claude":
        cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "100"]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(["--allowedTools",
                     "Edit,Read,Write,Bash,Glob,Grep"])
        if resume and session_id:
            cmd.extend(["--resume", session_id])
        elif session_id:
            cmd.extend(["--session-id", session_id])
    elif backend == "codex":
        if resume and session_id:
            cmd = ["codex", "exec", "resume", session_id,
                   "--dangerously-bypass-approvals-and-sandbox"]
        else:
            cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
        if model:
            cmd.extend(["-m", model])
    elif backend == "gemini":
        cmd = ["gemini", "--approval-mode", "yolo"]
        if model:
            cmd.extend(["--model", model])
        if resume:
            cmd.extend(["--resume", "latest"])
    else:
        raise RuntimeError(f"Unknown backend {backend!r}")

    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            cwd=workspace_dir, timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Experiment stage timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError(f"'{backend}' CLI tool not found on PATH")

    # Extract session ID from codex stderr
    if backend == "codex" and metadata is not None and result.stderr:
        from .cli_completion import extract_session_id
        sid = extract_session_id(result.stderr)
        if sid:
            metadata["session_id"] = sid

    if result.returncode != 0:
        stderr = (result.stderr or "")[:1000]
        raise RuntimeError(f"Experiment stage failed (rc={result.returncode}): {stderr}")

    return result.stdout.strip()


def _read_results(workspace_dir: str) -> str:
    """Read results.json from a stage workspace, or return empty string."""
    results_path = os.path.join(workspace_dir, "results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                return f.read()
        except Exception:
            pass

    # Fallback: look for any JSON file with "result" in the name
    for fname in os.listdir(workspace_dir):
        if "result" in fname.lower() and fname.endswith(".json"):
            try:
                with open(os.path.join(workspace_dir, fname)) as f:
                    return f.read()
            except Exception:
                continue

    return "(no results.json found)"


def _collect_all_results(workspace_dir: str) -> dict:
    """Collect results from all stage directories."""
    all_results = {}
    for stage in STAGES:
        stage_dir = os.path.join(workspace_dir, f"experiment_{stage['name']}")
        if os.path.isdir(stage_dir):
            results_text = _read_results(stage_dir)
            try:
                all_results[stage["name"]] = json.loads(results_text)
            except (json.JSONDecodeError, TypeError):
                all_results[stage["name"]] = {"raw_output": results_text[:2000]}
    return all_results


def run_experiment_stages(
    idea_spec: str,
    workspace_dir: str,
    backend: str = "claude",
    model: Optional[str] = None,
    end_stage: int = 4,
    timeout_per_stage: int = 1800,
    stage_root: Optional[str] = None,
) -> str:
    """Run an experiment through forced sequential stages.

    Each stage gets a full CLI agent session (with tools) that can write code,
    run it, debug errors, and iterate. The Python structure ensures all stages
    execute and context passes forward between stages.

    Args:
        idea_spec: JSON string describing the research idea.
        workspace_dir: Root directory for experiment files.
        backend: CLI backend ("claude", "codex", "gemini").
        model: Optional model override.
        end_stage: Last stage to run (1-4). Default: 4 (all stages).
        timeout_per_stage: Max seconds per stage. Default: 30 min.

    Returns:
        JSON string with aggregated results from all stages.
    """
    import uuid as _uuid

    experiment_dir = stage_root or os.path.join(workspace_dir, "experiment_workspace")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the idea spec for reference
    idea_path = os.path.join(experiment_dir, "idea_spec.json")
    with open(idea_path, "w") as f:
        f.write(idea_spec)

    session_id = str(_uuid.uuid4())
    stages_to_run = STAGES[:end_stage]

    for i, stage in enumerate(stages_to_run, 1):
        stage_dir = os.path.join(experiment_dir, f"experiment_{stage['name']}")
        os.makedirs(stage_dir, exist_ok=True)

        is_first = (i == 1)

        prompt = f"# Experiment Stage {i}/{len(stages_to_run)}: {stage['description']}\n\n"
        if is_first:
            prompt += f"## Research Idea\n{idea_spec}\n\n"
        prompt += f"## Stage Instructions\n{stage['instructions']}\n\n"
        prompt += (
            f"Work in the `experiment_{stage['name']}/` subdirectory. "
            "Write your code, run it, fix any errors, and save final results "
            f"to `experiment_{stage['name']}/results.json`. "
            "If you create plots, save them to a `plots/` subdirectory within."
        )
        if not is_first:
            prompt += (
                "\n\nYou have full context from previous stages in this session. "
                "Build on your prior work — reference results and code from earlier stages."
            )

        logger.info(
            "[ExperimentRunner] Stage %d/%d: %s — backend=%s resume=%s",
            i, len(stages_to_run), stage["name"], backend, not is_first,
        )
        t0 = time.time()
        meta: dict = {}
        output = _run_cli_agent(
            prompt, experiment_dir, backend, model, timeout_per_stage,
            session_id=session_id, resume=not is_first, metadata=meta,
        )
        # Codex may assign a different session ID
        if "session_id" in meta:
            session_id = meta["session_id"]

        elapsed = time.time() - t0
        logger.info(
            "[ExperimentRunner] Stage %s completed in %.1fs — output_len=%d",
            stage["name"], elapsed, len(output),
        )

        # Save the agent's full output for debugging
        with open(os.path.join(stage_dir, "agent_output.txt"), "w") as f:
            f.write(output)

        # Record to CLI budget tracker
        try:
            from .cli_budget import get_global_cli_tracker
            tracker = get_global_cli_tracker()
            if tracker is not None:
                tracker.record_invocation(
                    agent_name=f"experiment_{stage['name']}",
                    backend=backend,
                    model=model or "default",
                    resumed=not is_first,
                    duration_seconds=elapsed,
                    prompt_chars=len(prompt),
                    output_chars=len(output),
                )
        except Exception:
            pass

    # Collect and return all results
    all_results = _collect_all_results(experiment_dir)
    summary = json.dumps(all_results, indent=2, default=str)

    # Save combined results
    with open(os.path.join(experiment_dir, "all_results.json"), "w") as f:
        f.write(summary)

    return summary
