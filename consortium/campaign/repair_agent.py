"""
Autonomous repair agent — deploys Claude Code to diagnose and fix stage failures.

When a campaign stage fails (process dies with missing artifacts), instead of
immediately escalating to a human, OpenClaw can spawn a Claude Code coding agent
that:

  1. **Diagnoses** the failure by reading logs, workspace state, and error traces.
  2. **Repairs** the issue — fixing code bugs, config errors, missing files, etc.
  3. **Retries** the stage if the fix looks viable.

The repair agent runs in non-interactive mode (`claude -p`) with a carefully
constructed prompt that includes all failure context. It operates directly on
the workspace filesystem, so its fixes persist for the retry.

Two launcher modes:
  - **local** (default): runs `claude -p` as a subprocess, blocking until done.
    Good for fast repairs or when the heartbeat runs on a node with API access.
  - **slurm**: submits the repair as a SLURM batch job on the Engaging cluster.
    The heartbeat polls for completion on subsequent ticks instead of blocking.
    Good for longer repairs or when the heartbeat runs on a login node.

Usage (called automatically by campaign_heartbeat.py):

    from consortium.campaign.repair_agent import attempt_repair
    result = attempt_repair(stage, spec, status, campaign_dir)
    if result.success:
        # re-launch the stage
    else:
        # escalate to human

    # For SLURM mode:
    from consortium.campaign.repair_agent import submit_slurm_repair, poll_slurm_repair
    job_id = submit_slurm_repair(stage, spec, status, campaign_dir)
    # ... on next heartbeat tick:
    result = poll_slurm_repair(stage, campaign_dir)

Configuration (campaign.yaml):

    repair:
      enabled: true
      max_attempts: 2
      launcher: slurm              # "local" or "slurm"
      claude_binary: auto          # auto-detect from conda env PATH
      model: claude-opus-4-6       # always use opus for strongest reasoning
      effort: max                  # max thinking effort
      budget_usd: 10.0            # per-repair-attempt budget cap
      two_phase: true             # plan-then-execute flow
      plan_model: claude-opus-4-6
      plan_effort: max
      plan_budget_usd: 5.0
      review_model: claude-opus-4-6
      min_review_score: 7
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .spec import CampaignSpec, RepairConfig, Stage
from .status import CampaignStatus


@dataclass
class RepairResult:
    """Result of a single repair attempt."""
    success: bool
    diagnosis: str
    actions_taken: List[str]
    duration_seconds: float
    agent_output: str
    error: Optional[str] = None


# ------------------------------------------------------------------
# Claude Code binary discovery
# ------------------------------------------------------------------

_CLAUDE_SEARCH_PATHS = [
    # Conda environment bin (preferred — works in SLURM without IDE)
    os.environ.get("CONDA_PREFIX", "") + "/bin" if os.environ.get("CONDA_PREFIX") else None,
    # Standalone installs (work without IDE session)
    os.path.expanduser("~/.local/bin"),
    "/usr/local/bin",
    os.path.expanduser("~/.npm-global/bin"),
    # IDE extension fallbacks (only available during active IDE sessions)
    os.path.expanduser("~/.cursor-server/extensions"),
    os.path.expanduser("~/.vscode-server/extensions"),
    os.path.expanduser("~/.vscode/extensions"),
]
# Filter out None entries
_CLAUDE_SEARCH_PATHS = [p for p in _CLAUDE_SEARCH_PATHS if p]


def find_claude_binary(explicit_path: Optional[str] = None) -> Optional[str]:
    """Locate the claude CLI binary.

    Args:
        explicit_path: If provided and valid, use this path directly.

    Returns:
        Absolute path to the claude binary, or None if not found.
    """
    if explicit_path and explicit_path != "auto":
        if os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK):
            return explicit_path
        return None

    # Check PATH first
    result = shutil.which("claude")
    if result:
        return result

    # Search common locations
    for base in _CLAUDE_SEARCH_PATHS:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            if "claude" in files:
                candidate = os.path.join(root, "claude")
                if os.access(candidate, os.X_OK):
                    return candidate
            # Limit search depth
            depth = root.replace(base, "").count(os.sep)
            if depth >= 5:
                dirs.clear()

    return None


# ------------------------------------------------------------------
# Context collection
# ------------------------------------------------------------------

def _read_tail(path: str, max_lines: int = 200) -> str:
    """Read the last N lines of a file, returning empty string if missing."""
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


def _list_workspace(workspace: str, max_depth: int = 3) -> str:
    """Get a tree-like listing of the workspace."""
    lines = []
    workspace = os.path.abspath(workspace)
    for root, dirs, files in os.walk(workspace):
        depth = root.replace(workspace, "").count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        indent = "  " * depth
        dirname = os.path.basename(root)
        lines.append(f"{indent}{dirname}/")
        for f in sorted(files)[:50]:  # cap files per dir
            lines.append(f"{indent}  {f}")
        if len(files) > 50:
            lines.append(f"{indent}  ... and {len(files) - 50} more files")
    return "\n".join(lines[:500])  # cap total output


def _collect_failure_context(
    stage: Stage,
    status: CampaignStatus,
    campaign_dir: str,
) -> dict:
    """Gather all relevant context about a stage failure."""
    sid = stage.id
    workspace = status.stage_workspace(sid) or ""
    stage_data = status.raw()["stages"].get(sid, {})

    # Read logs
    log_dir = os.path.join(campaign_dir, "logs")
    stdout_log = _read_tail(os.path.join(log_dir, f"{sid}_stdout.log"), 300)
    stderr_log = _read_tail(os.path.join(log_dir, f"{sid}_stderr.log"), 300)

    # Read SLURM logs if present
    slurm_log_dir = os.path.join(campaign_dir, "slurm_logs")
    slurm_stdout = ""
    slurm_stderr = ""
    if os.path.isdir(slurm_log_dir):
        for fname in sorted(os.listdir(slurm_log_dir), reverse=True):
            if fname.startswith(sid) and fname.endswith(".out"):
                slurm_stdout = _read_tail(os.path.join(slurm_log_dir, fname), 200)
                break
        for fname in sorted(os.listdir(slurm_log_dir), reverse=True):
            if fname.startswith(sid) and fname.endswith(".err"):
                slurm_stderr = _read_tail(os.path.join(slurm_log_dir, fname), 200)
                break

    # Read the task prompt that was used
    task_prompt = ""
    task_file = os.path.join(campaign_dir, "task_prompts", f"{sid}_task.txt")
    if os.path.isfile(task_file):
        try:
            with open(task_file) as f:
                task_prompt = f.read()[:3000]  # cap length
        except Exception:
            pass

    # Workspace listing
    ws_listing = _list_workspace(workspace) if workspace and os.path.isdir(workspace) else ""

    # Read STATUS.txt if present
    status_txt = ""
    if workspace:
        status_txt_path = os.path.join(workspace, "STATUS.txt")
        if os.path.isfile(status_txt_path):
            try:
                with open(status_txt_path) as f:
                    status_txt = f.read()[:2000]
            except Exception:
                pass

    return {
        "stage_id": sid,
        "workspace": workspace,
        "fail_reason": stage_data.get("fail_reason", ""),
        "missing_artifacts": stage_data.get("missing_artifacts", []),
        "required_artifacts": stage.success_artifacts.get("required", []),
        "stdout_log_tail": stdout_log,
        "stderr_log_tail": stderr_log,
        "slurm_stdout_tail": slurm_stdout,
        "slurm_stderr_tail": slurm_stderr,
        "task_prompt_excerpt": task_prompt,
        "workspace_listing": ws_listing,
        "status_txt": status_txt,
        "stage_args": stage.args,
    }


# ------------------------------------------------------------------
# Prompt construction — shared context block
# ------------------------------------------------------------------

def _context_block(context: dict) -> str:
    """Render the failure context block shared by both phases."""
    missing = ", ".join(context["missing_artifacts"]) or "(none listed)"
    required = ", ".join(context["required_artifacts"]) or "(none listed)"

    return f"""## SITUATION
Stage '{context["stage_id"]}' of the campaign has FAILED.
- **Failure reason**: {context["fail_reason"]}
- **Missing artifacts**: {missing}
- **Required artifacts**: {required}
- **Workspace**: {context["workspace"]}

## CONTEXT

### Workspace file listing
```
{context["workspace_listing"][:3000]}
```

### STATUS.txt (pipeline status)
```
{context["status_txt"][:2000]}
```

### stdout log (last 300 lines)
```
{context["stdout_log_tail"][-8000:]}
```

### stderr log (last 300 lines)
```
{context["stderr_log_tail"][-8000:]}
```

### SLURM stdout (if applicable)
```
{context["slurm_stdout_tail"][-4000:]}
```

### SLURM stderr (if applicable)
```
{context["slurm_stderr_tail"][-4000:]}
```

### Task prompt excerpt (what the stage was asked to do)
```
{context["task_prompt_excerpt"][:2000]}
```

### Stage CLI args
```
{context["stage_args"]}
```"""


# ------------------------------------------------------------------
# Phase 1: Plan prompt (read-only diagnosis + structured plan)
# ------------------------------------------------------------------

def _build_plan_prompt(context: dict, repair_config: RepairConfig) -> str:
    """Build the Phase 1 prompt: diagnose and plan only, NO edits."""
    allowed = ", ".join(repair_config.allowed_actions) if repair_config.allowed_actions else "all"

    return f"""You are an autonomous repair agent for a research pipeline campaign.
You are in PLANNING MODE — you must ONLY read files to diagnose the issue and
produce a detailed repair plan. Do NOT edit or create any files.

{_context_block(context)}

## YOUR TASK
1. **Read** the relevant source files, logs, configs, and error traces in the workspace.
2. **Diagnose** the root cause of the failure.
3. **Produce a detailed repair plan** — exactly what files to change and how.

## ALLOWED REPAIR ACTIONS (for your plan): {allowed}

## CONSTRAINTS
- You are in PLAN MODE. Do NOT edit, write, or create any files.
- Only use Read, Glob, Grep, and Bash (for `ls`, `find`, `cat`, `head`, `wc` etc.) tools.
- Scope your plan ONLY to the workspace directory: {context["workspace"]}
- Be surgical — plan the minimum changes needed to fix the issue.

## OUTPUT FORMAT
Output your analysis in this exact structured format:

<repair_plan>
DIAGNOSIS: <one-paragraph description of the root cause>

ROOT_CAUSE_FILE: <path to the file containing the root cause, relative to workspace>
ROOT_CAUSE_LINE: <approximate line number, or "N/A">

PLAN:
- STEP 1: <file_path> — <what to change and why>
- STEP 2: <file_path> — <what to change and why>
- ...

RISK_ASSESSMENT: <what could go wrong with this plan>
CONFIDENCE: <high|medium|low>
NEEDS_RETRY: <true|false — whether the stage should be retried after repair>
</repair_plan>
"""


# ------------------------------------------------------------------
# Phase 2: Execute prompt (apply the approved plan)
# ------------------------------------------------------------------

def _build_execute_prompt(
    context: dict,
    plan_output: str,
    review_feedback: str,
    repair_config: RepairConfig,
) -> str:
    """Build the Phase 2 prompt: execute the approved plan."""
    allowed = ", ".join(repair_config.allowed_actions) if repair_config.allowed_actions else "all"

    return f"""You are an autonomous repair agent for a research pipeline campaign.
An earlier planning phase diagnosed the failure and produced a repair plan that
has been APPROVED by the review system. Your job is to EXECUTE the plan precisely.

{_context_block(context)}

## APPROVED REPAIR PLAN
The following plan was produced in the diagnosis phase and approved by the reviewer:

```
{plan_output[-6000:]}
```

## REVIEWER FEEDBACK
{review_feedback}

## YOUR TASK
1. Execute the repair plan step by step.
2. After each step, verify the change is correct.
3. When done, produce a summary of what you did.

## ALLOWED ACTIONS: {allowed}

## CONSTRAINTS
- Work ONLY within the workspace directory: {context["workspace"]}
- Do NOT delete existing data or results — only add or fix.
- Do NOT modify files outside the workspace.
- Follow the plan. If you discover the plan is wrong mid-execution, stop and
  explain why in your report rather than improvising a different fix.
- Be surgical — make the minimum changes needed.

## OUTPUT FORMAT
After executing the plan, output a structured summary in this exact format:

<repair_report>
DIAGNOSIS: <one-paragraph description of the root cause>
ACTIONS: <bulleted list of what you changed>
CONFIDENCE: <high|medium|low — your confidence the fix will resolve the failure>
NEEDS_RETRY: <true|false — whether the stage should be retried>
</repair_report>
"""


# ------------------------------------------------------------------
# Single-phase prompt (fallback when two_phase=False)
# ------------------------------------------------------------------

def _build_repair_prompt(context: dict, repair_config: RepairConfig) -> str:
    """Build single-phase prompt (diagnose + fix in one shot)."""
    allowed = ", ".join(repair_config.allowed_actions) if repair_config.allowed_actions else "all"

    return f"""You are an autonomous repair agent for a research pipeline campaign.

{_context_block(context)}

## YOUR TASK
1. **Diagnose** the root cause of the failure by examining the logs and workspace.
2. **Fix** the issue so the stage can be retried successfully.
3. **Report** what you found and what you did.

## ALLOWED ACTIONS: {allowed}

## CONSTRAINTS
- Work ONLY within the workspace directory: {context["workspace"]}
- Do NOT delete existing data or results — only add or fix.
- Do NOT modify files outside the workspace.
- Be surgical — make the minimum changes needed to fix the issue.
- If the fix requires installing a package, note it but do not run pip install globally.

## OUTPUT FORMAT
After you have made your fixes, output a structured summary in this exact format:

<repair_report>
DIAGNOSIS: <one-paragraph description of the root cause>
ACTIONS: <bulleted list of what you changed>
CONFIDENCE: <high|medium|low — your confidence the fix will resolve the failure>
NEEDS_RETRY: <true|false — whether the stage should be retried>
</repair_report>
"""


# ------------------------------------------------------------------
# Agent execution
# ------------------------------------------------------------------

def _run_claude_agent(
    prompt: str,
    workspace: str,
    claude_binary: str,
    model: Optional[str] = None,
    effort: Optional[str] = None,
    budget_usd: float = 5.0,
    timeout_seconds: int = 600,
    permission_mode: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
    max_turns: int = 30,
    session_id: Optional[str] = None,
    resume: bool = False,
) -> tuple[str, int]:
    """Spawn a Claude Code agent in non-interactive mode.

    Args:
        prompt: The repair prompt.
        workspace: Working directory for the agent.
        claude_binary: Path to the claude binary.
        model: Model to use (e.g. "claude-opus-4-6").
        effort: Effort/thinking level ("low", "medium", "high", "max").
        budget_usd: Maximum spend in USD.
        timeout_seconds: Hard timeout for the agent process.
        permission_mode: Claude Code permission mode ("plan", "default", etc.).
        allowed_tools: Restrict to these tools only (e.g. ["Read", "Glob", "Grep"]).
        max_turns: Maximum agent turns.
        session_id: Session UUID for multi-turn (plan → execute).
        resume: If True, resume the session instead of starting a new one.

    Returns:
        (agent_output, return_code)
    """
    cmd = [
        claude_binary,
        "-p",  # non-interactive / print mode
        "--output-format", "text",
    ]

    if model:
        cmd.extend(["--model", model])

    if effort:
        cmd.extend(["--effort", effort])

    if permission_mode:
        cmd.extend(["--permission-mode", permission_mode])

    if allowed_tools:
        cmd.extend(["--tools", ",".join(allowed_tools)])

    cmd.extend(["--max-turns", str(max_turns)])

    if resume and session_id:
        cmd.extend(["--resume", session_id])
    elif session_id:
        cmd.extend(["--session-id", session_id])

    env = {**os.environ}
    env["CLAUDE_CODE_MAX_COST_CENTS"] = str(int(budget_usd * 100))

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=workspace,
            env=env,
            timeout=timeout_seconds,
        )
        output = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        return output, proc.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT: Repair agent exceeded time limit.", 1
    except Exception as e:
        return f"ERROR spawning repair agent: {e}", 1


# ------------------------------------------------------------------
# Plan parsing
# ------------------------------------------------------------------

def _parse_repair_plan(output: str) -> dict:
    """Extract the structured repair plan from Phase 1 output."""
    plan = {
        "diagnosis": "",
        "root_cause_file": "",
        "root_cause_line": "",
        "steps": [],
        "risk_assessment": "",
        "confidence": "low",
        "needs_retry": False,
        "raw": "",
    }

    match = re.search(r"<repair_plan>(.*?)</repair_plan>", output, re.DOTALL)
    if not match:
        plan["diagnosis"] = "Planning agent did not produce a structured plan."
        return plan

    block = match.group(1)
    plan["raw"] = block.strip()

    diag = re.search(r"DIAGNOSIS:\s*(.+?)(?=\nROOT_CAUSE|\nPLAN:|\Z)", block, re.DOTALL)
    if diag:
        plan["diagnosis"] = diag.group(1).strip()

    rcf = re.search(r"ROOT_CAUSE_FILE:\s*(.+)", block)
    if rcf:
        plan["root_cause_file"] = rcf.group(1).strip()

    rcl = re.search(r"ROOT_CAUSE_LINE:\s*(.+)", block)
    if rcl:
        plan["root_cause_line"] = rcl.group(1).strip()

    steps_match = re.search(r"PLAN:\s*\n(.+?)(?=\nRISK_ASSESSMENT:|\nCONFIDENCE:|\Z)", block, re.DOTALL)
    if steps_match:
        plan["steps"] = [
            line.strip().lstrip("- •*")
            for line in steps_match.group(1).strip().splitlines()
            if line.strip() and re.match(r"[-•*]?\s*STEP\s+\d+", line.strip(), re.IGNORECASE)
        ]
        # If no STEP prefix, just grab all bullet lines
        if not plan["steps"]:
            plan["steps"] = [
                line.strip().lstrip("- •*")
                for line in steps_match.group(1).strip().splitlines()
                if line.strip()
            ]

    risk = re.search(r"RISK_ASSESSMENT:\s*(.+?)(?=\nCONFIDENCE:|\Z)", block, re.DOTALL)
    if risk:
        plan["risk_assessment"] = risk.group(1).strip()

    conf = re.search(r"CONFIDENCE:\s*(\w+)", block)
    if conf:
        plan["confidence"] = conf.group(1).strip().lower()

    retry = re.search(r"NEEDS_RETRY:\s*(\w+)", block)
    if retry:
        plan["needs_retry"] = retry.group(1).strip().lower() == "true"

    return plan


def _parse_repair_report(output: str) -> dict:
    """Extract the structured repair report from Phase 2 / single-phase output."""
    report = {
        "diagnosis": "",
        "actions": [],
        "confidence": "low",
        "needs_retry": False,
    }

    match = re.search(r"<repair_report>(.*?)</repair_report>", output, re.DOTALL)
    if not match:
        report["diagnosis"] = "Agent did not produce structured report. Raw output available in logs."
        return report

    block = match.group(1)

    diag = re.search(r"DIAGNOSIS:\s*(.+?)(?=\nACTIONS:|\Z)", block, re.DOTALL)
    if diag:
        report["diagnosis"] = diag.group(1).strip()

    actions = re.search(r"ACTIONS:\s*(.+?)(?=\nCONFIDENCE:|\Z)", block, re.DOTALL)
    if actions:
        report["actions"] = [
            line.strip().lstrip("- •*")
            for line in actions.group(1).strip().splitlines()
            if line.strip()
        ]

    conf = re.search(r"CONFIDENCE:\s*(\w+)", block)
    if conf:
        report["confidence"] = conf.group(1).strip().lower()

    retry = re.search(r"NEEDS_RETRY:\s*(\w+)", block)
    if retry:
        report["needs_retry"] = retry.group(1).strip().lower() == "true"

    return report


# ------------------------------------------------------------------
# OpenClaw plan review — LLM judge via cli_completion
# ------------------------------------------------------------------

@dataclass
class PlanReview:
    """Result of OpenClaw reviewing a repair plan."""
    approved: bool
    score: int              # 1-10
    reasoning: str
    suggestions: str        # feedback passed to Phase 2 if approved

    def __str__(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        return f"[{status} score={self.score}/10] {self.reasoning}"


def _review_plan(
    plan: dict,
    context: dict,
    repair_config: RepairConfig,
) -> PlanReview:
    """
    Have OpenClaw (via cli_completion) review the repair plan before execution.

    Uses a separate LLM call (not Claude Code) to judge whether the plan
    is safe, correct, and likely to fix the issue. This is the gate between
    Phase 1 (plan) and Phase 2 (execute).
    """
    from ..cli_completion import cli_completion

    plan_text = plan["raw"] or f"Diagnosis: {plan['diagnosis']}\nSteps: {plan['steps']}"

    review_prompt = f"""You are OpenClaw, an autonomous research pipeline orchestrator.
A Claude Code repair agent has diagnosed a failure and proposed a repair plan.
Your job is to REVIEW the plan and decide whether to approve it for execution.

## FAILURE CONTEXT
- Stage: {context["stage_id"]}
- Failure reason: {context["fail_reason"]}
- Missing artifacts: {", ".join(context["missing_artifacts"]) or "none listed"}
- Required artifacts: {", ".join(context["required_artifacts"]) or "none listed"}

## PROPOSED REPAIR PLAN
{plan_text}

## REVIEW CRITERIA
Score the plan from 1 to 10 on each dimension, then give an overall score:

1. **Correctness**: Does the diagnosis accurately identify the root cause?
2. **Completeness**: Does the plan address all aspects of the failure?
3. **Safety**: Will the changes avoid breaking other parts of the pipeline?
4. **Minimality**: Are the proposed changes surgical and focused?
5. **Feasibility**: Can the changes actually be made (files exist, etc.)?

## RESPONSE FORMAT (respond in EXACTLY this format)
<plan_review>
CORRECTNESS: <1-10>
COMPLETENESS: <1-10>
SAFETY: <1-10>
MINIMALITY: <1-10>
FEASIBILITY: <1-10>
OVERALL_SCORE: <1-10>
REASONING: <2-3 sentence explanation of your overall assessment>
SUGGESTIONS: <specific improvements or cautions for the execution phase, or "none">
APPROVED: <true|false>
</plan_review>
"""

    try:
        def _model_to_backend(mid: str) -> str:
            if "claude" in mid or "anthropic" in mid: return "claude"
            if "gpt" in mid or mid.startswith(("o1-","o3-","o4-")): return "codex"
            if "gemini" in mid: return "gemini"
            return "claude"

        review_text = cli_completion(
            review_prompt,
            backend=_model_to_backend(repair_config.review_model),
        ) or ""
    except Exception as e:
        # If the review LLM fails, track consecutive failures.
        # After max_review_failures (default 3), reject instead of auto-approving
        # to prevent unreviewed repairs from executing repeatedly.
        print(f"[repair:review] LLM review failed: {e}")

        # Track consecutive review failures via module-level counter
        _review_plan._consecutive_failures = getattr(_review_plan, "_consecutive_failures", 0) + 1
        max_failures = getattr(repair_config, "max_review_failures", 3)

        if _review_plan._consecutive_failures >= max_failures:
            print(
                f"[repair:review] {_review_plan._consecutive_failures} consecutive review "
                f"LLM failures (limit={max_failures}). Rejecting plan."
            )
            return PlanReview(
                approved=False,
                score=0,
                reasoning=f"Review LLM has failed {_review_plan._consecutive_failures} times consecutively. Rejecting for safety.",
                suggestions="Review LLM is unreliable. Fix API access before retrying.",
            )

        # Only auto-approve on "high" confidence — "medium" is too risky without review
        auto_approve = plan["confidence"] == "high"
        if auto_approve:
            print(
                f"[repair:review] WARNING: Review LLM failed ({e}). "
                f"Auto-approving high-confidence plan."
            )
        else:
            print(
                f"[repair:review] Review LLM failed ({e}). "
                f"Rejecting plan with confidence={plan['confidence']} (only 'high' auto-approved)."
            )
        return PlanReview(
            approved=auto_approve,
            score=7 if auto_approve else 3,
            reasoning=f"Review LLM unavailable ({e}). Auto-{'approved' if auto_approve else 'rejected'} based on plan confidence={plan['confidence']}. Only 'high' confidence plans are auto-approved.",
            suggestions="Review was skipped due to LLM error." if auto_approve else "Escalate to human — review LLM unavailable and plan confidence too low for auto-approval.",
        )

    # Parse the review
    match = re.search(r"<plan_review>(.*?)</plan_review>", review_text, re.DOTALL)
    if not match:
        return PlanReview(
            approved=False,
            score=0,
            reasoning="Reviewer did not produce structured output.",
            suggestions="",
        )

    block = match.group(1)

    score_match = re.search(r"OVERALL_SCORE:\s*(\d+)", block)
    score = int(score_match.group(1)) if score_match else 0

    reasoning_match = re.search(r"REASONING:\s*(.+?)(?=\nSUGGESTIONS:|\nAPPROVED:|\Z)", block, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    suggestions_match = re.search(r"SUGGESTIONS:\s*(.+?)(?=\nAPPROVED:|\Z)", block, re.DOTALL)
    suggestions = suggestions_match.group(1).strip() if suggestions_match else ""

    approved_match = re.search(r"APPROVED:\s*(\w+)", block)
    explicit_approved = approved_match.group(1).strip().lower() == "true" if approved_match else False

    # Final approval: reviewer says approved AND score meets threshold
    approved = explicit_approved and score >= repair_config.min_review_score

    return PlanReview(
        approved=approved,
        score=score,
        reasoning=reasoning,
        suggestions=suggestions,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def attempt_repair(
    stage: Stage,
    spec: CampaignSpec,
    status: CampaignStatus,
    campaign_dir: str,
) -> RepairResult:
    """
    Attempt to repair a failed stage by deploying a Claude Code agent.

    Two-phase flow (when repair.two_phase=True):
      Phase 1: Claude Code in plan mode (read-only) diagnoses and proposes a plan.
      Review:  OpenClaw judges the plan via a cli_completion call.
      Phase 2: If approved, Claude Code executes the plan with full edit access.

    Single-phase flow (when repair.two_phase=False):
      Claude Code diagnoses and fixes in one shot (original behavior).

    Args:
        stage:        The failed stage definition.
        spec:         Campaign spec (contains repair config).
        status:       Current campaign status.
        campaign_dir: Campaign directory.

    Returns:
        RepairResult with diagnosis, actions taken, and success/failure.
    """
    repair_config = spec.repair
    t0 = time.time()

    # Find claude binary
    claude_bin = find_claude_binary(repair_config.claude_binary)
    if not claude_bin:
        return RepairResult(
            success=False,
            diagnosis="Could not locate claude CLI binary.",
            actions_taken=[],
            duration_seconds=time.time() - t0,
            agent_output="",
            error="Claude binary not found. Set repair.claude_binary in campaign.yaml.",
        )

    # Collect failure context
    context = _collect_failure_context(stage, status, campaign_dir)
    workspace = context["workspace"]

    if not workspace or not os.path.isdir(workspace):
        return RepairResult(
            success=False,
            diagnosis="Stage workspace does not exist.",
            actions_taken=[],
            duration_seconds=time.time() - t0,
            agent_output="",
            error=f"Workspace not found: {workspace}",
        )

    repair_log_dir = os.path.join(campaign_dir, "repair_logs")
    os.makedirs(repair_log_dir, exist_ok=True)
    attempt_num = status.repair_attempt_count(stage.id) + 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log_prefix = f"{stage.id}_attempt{attempt_num}_{ts}"

    # ==================================================================
    # Two-phase flow: Plan → Review → Execute
    # ==================================================================
    if repair_config.two_phase:
        return _attempt_repair_two_phase(
            context=context,
            workspace=workspace,
            claude_bin=claude_bin,
            repair_config=repair_config,
            repair_log_dir=repair_log_dir,
            log_prefix=log_prefix,
            attempt_num=attempt_num,
            t0=t0,
        )

    # ==================================================================
    # Single-phase flow (original behavior)
    # ==================================================================
    return _attempt_repair_single_phase(
        context=context,
        workspace=workspace,
        claude_bin=claude_bin,
        repair_config=repair_config,
        repair_log_dir=repair_log_dir,
        log_prefix=log_prefix,
        attempt_num=attempt_num,
        t0=t0,
    )


def _attempt_repair_two_phase(
    context: dict,
    workspace: str,
    claude_bin: str,
    repair_config: RepairConfig,
    repair_log_dir: str,
    log_prefix: str,
    attempt_num: int,
    t0: float,
) -> RepairResult:
    """Two-phase repair: plan (read-only) → review → execute.

    Uses session resume so Phase 2 retains full context from Phase 1's
    diagnosis — no need to re-read logs or re-diagnose.
    """
    import uuid as _uuid

    sid = context["stage_id"]
    repair_session = str(_uuid.uuid4())

    # ------------------------------------------------------------------
    # PHASE 1: Plan (read-only, plan permission mode)
    # ------------------------------------------------------------------
    plan_prompt = _build_plan_prompt(context, repair_config)

    # Save plan prompt
    plan_prompt_file = os.path.join(repair_log_dir, f"{log_prefix}_phase1_prompt.txt")
    with open(plan_prompt_file, "w") as f:
        f.write(plan_prompt)

    plan_model = repair_config.plan_model or repair_config.model
    print(f"[repair:phase1] Planning repair for '{sid}' (attempt {attempt_num})...")
    print(f"[repair:phase1] Model: {plan_model or 'default'}, Budget: ${repair_config.plan_budget_usd:.2f}")
    print(f"[repair:phase1] Permission mode: plan (read-only)")

    plan_effort = repair_config.plan_effort or repair_config.effort
    plan_output, plan_rc = _run_claude_agent(
        prompt=plan_prompt,
        workspace=workspace,
        claude_binary=claude_bin,
        model=plan_model,
        effort=plan_effort,
        budget_usd=repair_config.plan_budget_usd,
        timeout_seconds=repair_config.plan_timeout_seconds,
        permission_mode="plan",
        allowed_tools=["Read", "Glob", "Grep", "Bash"],
        max_turns=20,
        session_id=repair_session,
    )

    # Save plan output
    plan_output_file = os.path.join(repair_log_dir, f"{log_prefix}_phase1_output.txt")
    with open(plan_output_file, "w") as f:
        f.write(plan_output)

    plan = _parse_repair_plan(plan_output)
    print(f"[repair:phase1] Diagnosis: {plan['diagnosis'][:200]}")
    print(f"[repair:phase1] Steps: {len(plan['steps'])}")
    print(f"[repair:phase1] Confidence: {plan['confidence']}")

    if plan_rc != 0 or not plan["steps"]:
        return RepairResult(
            success=False,
            diagnosis=plan["diagnosis"] or "Planning phase failed to produce a plan.",
            actions_taken=[],
            duration_seconds=time.time() - t0,
            agent_output=plan_output[:10000],
            error=f"Phase 1 failed (exit={plan_rc}, steps={len(plan['steps'])})",
        )

    # ------------------------------------------------------------------
    # REVIEW: OpenClaw judges the plan via cli_completion
    # ------------------------------------------------------------------
    print(f"[repair:review] Reviewing plan with {repair_config.review_model}...")

    review = _review_plan(plan, context, repair_config)

    # Save review
    review_file = os.path.join(repair_log_dir, f"{log_prefix}_review.txt")
    with open(review_file, "w") as f:
        f.write(str(review))
        f.write(f"\n\nScore: {review.score}/{repair_config.min_review_score} (min)\n")
        f.write(f"Approved: {review.approved}\n")
        f.write(f"Suggestions: {review.suggestions}\n")

    print(f"[repair:review] {review}")

    if not review.approved:
        return RepairResult(
            success=False,
            diagnosis=plan["diagnosis"],
            actions_taken=[f"Plan rejected by reviewer (score={review.score}/10): {review.reasoning}"],
            duration_seconds=time.time() - t0,
            agent_output=plan_output[:5000] + "\n\n--- REVIEW ---\n" + str(review),
            error=f"Plan rejected by OpenClaw reviewer (score={review.score}, min={repair_config.min_review_score})",
        )

    # ------------------------------------------------------------------
    # PHASE 2: Execute the approved plan
    # ------------------------------------------------------------------
    exec_prompt = _build_execute_prompt(context, plan_output, review.suggestions, repair_config)

    exec_prompt_file = os.path.join(repair_log_dir, f"{log_prefix}_phase2_prompt.txt")
    with open(exec_prompt_file, "w") as f:
        f.write(exec_prompt)

    print(f"[repair:phase2] Plan approved (score={review.score}/10). Executing...")
    print(f"[repair:phase2] Model: {repair_config.model or 'default'}, Budget: ${repair_config.budget_usd:.2f}")

    exec_output, exec_rc = _run_claude_agent(
        prompt=exec_prompt,
        workspace=workspace,
        claude_binary=claude_bin,
        model=repair_config.model,
        effort=repair_config.effort,
        budget_usd=repair_config.budget_usd,
        timeout_seconds=repair_config.timeout_seconds,
        permission_mode="bypassPermissions",
        max_turns=30,
        session_id=repair_session,
        resume=True,
    )

    duration = time.time() - t0

    # Save execution output
    exec_output_file = os.path.join(repair_log_dir, f"{log_prefix}_phase2_output.txt")
    with open(exec_output_file, "w") as f:
        f.write(exec_output)

    report = _parse_repair_report(exec_output)

    print(f"[repair:phase2] Execution finished in {duration:.1f}s (exit={exec_rc})")
    print(f"[repair:phase2] Confidence: {report['confidence']}")
    if report["actions"]:
        print(f"[repair:phase2] Actions taken:")
        for action in report["actions"]:
            print(f"[repair:phase2]   - {action}")

    success = (
        exec_rc == 0
        and report["confidence"] in ("high", "medium")
        and report["needs_retry"]
    )

    # Combine all outputs for the log
    combined_output = (
        f"=== PHASE 1: PLAN ===\n{plan_output[:4000]}\n\n"
        f"=== REVIEW ===\n{review}\n\n"
        f"=== PHASE 2: EXECUTE ===\n{exec_output[:5000]}"
    )

    return RepairResult(
        success=success,
        diagnosis=report["diagnosis"] or plan["diagnosis"],
        actions_taken=report["actions"],
        duration_seconds=duration,
        agent_output=combined_output[:10000],
        error=None if exec_rc == 0 else f"Phase 2 exited with code {exec_rc}",
    )


def _attempt_repair_single_phase(
    context: dict,
    workspace: str,
    claude_bin: str,
    repair_config: RepairConfig,
    repair_log_dir: str,
    log_prefix: str,
    attempt_num: int,
    t0: float,
) -> RepairResult:
    """Single-phase repair: diagnose + fix in one shot."""
    prompt = _build_repair_prompt(context, repair_config)

    prompt_file = os.path.join(repair_log_dir, f"{log_prefix}_prompt.txt")
    with open(prompt_file, "w") as f:
        f.write(prompt)

    print(f"[repair] Deploying Claude Code agent for stage '{context['stage_id']}' (attempt {attempt_num})...")
    print(f"[repair] Claude binary: {claude_bin}")
    print(f"[repair] Model: {repair_config.model or 'default'}")
    print(f"[repair] Budget: ${repair_config.budget_usd:.2f}")

    agent_output, return_code = _run_claude_agent(
        prompt=prompt,
        workspace=workspace,
        claude_binary=claude_bin,
        model=repair_config.model,
        effort=repair_config.effort,
        budget_usd=repair_config.budget_usd,
        timeout_seconds=repair_config.timeout_seconds,
    )

    duration = time.time() - t0

    output_file = os.path.join(repair_log_dir, f"{log_prefix}_output.txt")
    with open(output_file, "w") as f:
        f.write(agent_output)

    report = _parse_repair_report(agent_output)

    print(f"[repair] Agent finished in {duration:.1f}s (exit={return_code})")
    print(f"[repair] Diagnosis: {report['diagnosis'][:200]}")
    print(f"[repair] Confidence: {report['confidence']}")
    print(f"[repair] Needs retry: {report['needs_retry']}")
    if report["actions"]:
        print(f"[repair] Actions taken:")
        for action in report["actions"]:
            print(f"[repair]   - {action}")

    success = (
        return_code == 0
        and report["confidence"] in ("high", "medium")
        and report["needs_retry"]
    )

    return RepairResult(
        success=success,
        diagnosis=report["diagnosis"],
        actions_taken=report["actions"],
        duration_seconds=duration,
        agent_output=agent_output[:10000],
        error=None if return_code == 0 else f"Agent exited with code {return_code}",
    )


# ------------------------------------------------------------------
# SLURM-based repair (non-blocking, for Engaging cluster)
# ------------------------------------------------------------------

def _load_engaging_config(repo_root: Optional[str] = None) -> dict:
    """Load engaging_config.yaml for SLURM settings."""
    import yaml

    if repo_root is None:
        # Walk up from this file to find repo root
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(4):
            here = os.path.dirname(here)
            if os.path.exists(os.path.join(here, "engaging_config.yaml")):
                repo_root = here
                break

    config_path = os.environ.get(
        "ENGAGING_CONFIG",
        os.path.join(repo_root or ".", "engaging_config.yaml"),
    )
    if os.path.exists(config_path):
        from ..workflow_utils import expand_env_vars
        with open(config_path) as f:
            raw = f.read()
        # Expand ${VAR:-default} patterns before parsing YAML
        raw = expand_env_vars(raw)
        return yaml.safe_load(raw) or {}
    return {}


def _repair_sentinel_path(campaign_dir: str, stage_id: str) -> str:
    """Path to the sentinel file written by a SLURM repair job on completion."""
    return os.path.join(campaign_dir, "repair_logs", f"{stage_id}_slurm_result.json")


def submit_slurm_repair(
    stage: Stage,
    spec: CampaignSpec,
    status: CampaignStatus,
    campaign_dir: str,
) -> Optional[int]:
    """
    Submit a repair job to SLURM that runs the full two-phase repair flow
    (plan → review → execute) on a compute node.

    Instead of invoking `claude -p` directly, the SLURM job calls
    `attempt_repair()` via Python — inheriting the two-phase logic, plan
    review, and all configuration from campaign.yaml.

    Returns:
        SLURM job ID, or None if submission failed.
    """
    repair_config = spec.repair
    attempt_num = status.repair_attempt_count(stage.id) + 1

    # Quick pre-checks
    claude_bin = find_claude_binary(repair_config.claude_binary)
    if not claude_bin:
        print("[repair:slurm] Cannot find claude binary. Skipping SLURM repair.")
        return None

    context = _collect_failure_context(stage, status, campaign_dir)
    workspace = context["workspace"]
    if not workspace or not os.path.isdir(workspace):
        print(f"[repair:slurm] Workspace not found: {workspace}")
        return None

    # Load cluster config
    eng_config = _load_engaging_config()
    cluster = eng_config.get("cluster", {})
    repair_cluster = cluster.get("repair", cluster.get("orchestrator", {}))
    conda_init = cluster.get("conda_init_script") or os.environ.get("CONDA_INIT_SCRIPT", "")
    conda_env = cluster.get("conda_env_prefix") or os.environ.get("CONDA_PREFIX", "")
    conda_module = cluster.get("modules", {}).get("conda", "miniforge/25.11.0-0")
    # Derive repo_root from this file's location if not configured
    _default_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repo_root = cluster.get("repo_root", _default_repo_root)

    partition = repair_cluster.get("partition", "pi_tpoggio")
    wall_time = repair_cluster.get("time", "01:00:00")
    cpus = repair_cluster.get("cpus", 2)
    mem = repair_cluster.get("mem", "8G")

    # Sentinel file — the SLURM job writes its result here
    sentinel = _repair_sentinel_path(campaign_dir, stage.id)
    if os.path.exists(sentinel):
        os.remove(sentinel)

    repair_log_dir = os.path.join(campaign_dir, "repair_logs")
    os.makedirs(repair_log_dir, exist_ok=True)
    slurm_log_dir = os.path.join(campaign_dir, "slurm_logs")
    os.makedirs(slurm_log_dir, exist_ok=True)

    # The SLURM script calls attempt_repair() via Python, so it inherits
    # the full two-phase plan→review→execute flow automatically.
    campaign_yaml = spec.spec_file
    script_content = f"""#!/bin/bash
#SBATCH --job-name=repair_{stage.id}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={wall_time}
#SBATCH --mem={mem}
#SBATCH --output={slurm_log_dir}/repair_{stage.id}_%j.out
#SBATCH --error={slurm_log_dir}/repair_{stage.id}_%j.err

set -eo pipefail
# Fix PS1 unbound variable in non-interactive batch shells
export PS1="${{PS1:-}}"

echo "========================================"
echo "Repair Agent — Stage: {stage.id}"
echo "Attempt: {attempt_num}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Time:   $(date)"
echo "Two-phase: {repair_config.two_phase}"
echo "========================================"

module load {conda_module}
source {conda_init}

# Clear all inherited conda envs to avoid PATH collision
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate {conda_env}

# Validate python resolves to expected env
EXPECTED_PYTHON="{conda_env}/bin/python"
ACTUAL_PYTHON="$(which python)"
if [ "$ACTUAL_PYTHON" != "$EXPECTED_PYTHON" ]; then
    echo "WARNING: python resolves to $ACTUAL_PYTHON (expected $EXPECTED_PYTHON)"
    export PATH="{conda_env}/bin:$PATH"
fi

cd "{repo_root}"

# Run the full repair flow (plan→review→execute) via Python.
# This calls attempt_repair() which handles both two-phase and single-phase
# modes, plan review via cli_completion, and writes the sentinel file.
python3 -c "
import json, sys, time
sys.path.insert(0, '.')
from consortium.campaign.spec import load_spec
from consortium.campaign.status import read_status
from consortium.campaign.repair_agent import attempt_repair, RepairResult

spec = load_spec('{campaign_yaml}')
status = read_status('{campaign_dir}')
campaign_dir = '{campaign_dir}'

# Find the stage object
stage = None
for s in spec.stages:
    if s.id == '{stage.id}':
        stage = s
        break

if stage is None:
    print('ERROR: Stage {stage.id} not found in spec')
    sys.exit(1)

print(f'Running attempt_repair for stage {{stage.id!r}}...')
result = attempt_repair(stage, spec, status, campaign_dir)

# Write sentinel file for the heartbeat to pick up
sentinel = {{
    'claude_exit_code': 0 if result.error is None else 1,
    'duration_seconds': result.duration_seconds,
    'report': {{
        'diagnosis': result.diagnosis[:500],
        'actions': result.actions_taken[:20],
        'confidence': 'high' if result.success else 'low',
        'needs_retry': result.success,
    }},
    'success': result.success,
}}

sentinel_path = '{sentinel}'
with open(sentinel_path, 'w') as f:
    json.dump(sentinel, f, indent=2)

print(f'Sentinel written: {{sentinel_path}}')
print(f'Success: {{result.success}}')
print(f'Diagnosis: {{result.diagnosis[:200]}}')
"

echo "Repair job complete at $(date)"
"""

    # Write and submit the SLURM script
    script_dir = os.path.join(campaign_dir, "slurm_scripts")
    os.makedirs(script_dir, exist_ok=True)
    script_path = os.path.join(script_dir, f"repair_{stage.id}.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[repair:slurm] sbatch failed: {result.stderr}")
            return None

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            print(f"[repair:slurm] Could not parse job ID from: {result.stdout}")
            return None

        job_id = int(match.group(1))
        print(
            f"[repair:slurm] Repair job submitted for '{stage.id}' "
            f"(SLURM job {job_id}, attempt {attempt_num})"
        )
        return job_id

    except FileNotFoundError:
        print("[repair:slurm] sbatch not found — not on a SLURM cluster?")
        return None
    except Exception as e:
        print(f"[repair:slurm] Error submitting repair job: {e}")
        return None


def poll_slurm_repair(
    stage: Stage,
    status: CampaignStatus,
    campaign_dir: str,
) -> Optional[RepairResult]:
    """
    Check if a SLURM repair job has completed by looking for the sentinel file.

    Returns:
        RepairResult if the job is done, None if still running or no sentinel.
    """
    sentinel = _repair_sentinel_path(campaign_dir, stage.id)

    # Atomic read: skip exists() check to avoid TOCTOU race
    try:
        with open(sentinel) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None  # still running or never submitted
    except Exception as e:
        print(f"[repair:slurm] Error reading sentinel: {e}")
        return None

    report = data.get("report", {})
    success = data.get("success", False)
    duration = data.get("duration_seconds", 0)

    # Read the full agent output for the log
    attempt_num = status.repair_attempt_count(stage.id) + 1
    repair_log_dir = os.path.join(campaign_dir, "repair_logs")
    agent_output = ""
    # Find the most recent output file for this stage
    if os.path.isdir(repair_log_dir):
        output_files = sorted(
            [f for f in os.listdir(repair_log_dir)
             if f.startswith(f"{stage.id}_attempt") and f.endswith("_output.txt")],
            reverse=True,
        )
        if output_files:
            try:
                with open(os.path.join(repair_log_dir, output_files[0])) as f:
                    agent_output = f.read()[:10000]
            except Exception:
                pass

    print(f"[repair:slurm] Repair job for '{stage.id}' completed.")
    print(f"[repair:slurm] Diagnosis: {report.get('diagnosis', '')[:200]}")
    print(f"[repair:slurm] Confidence: {report.get('confidence', 'unknown')}")
    print(f"[repair:slurm] Success: {success}")

    # Mark sentinel as processed (rename instead of delete to avoid race
    # where a concurrent heartbeat tick misses the result).
    try:
        os.rename(sentinel, sentinel + ".processed")
    except OSError:
        # If rename fails (e.g., already processed by another tick), that's OK
        pass

    error = None
    exit_code = data.get("claude_exit_code", -1)
    if exit_code != 0:
        error = f"Claude Code exited with code {exit_code}"

    return RepairResult(
        success=success,
        diagnosis=report.get("diagnosis", ""),
        actions_taken=report.get("actions", []),
        duration_seconds=duration,
        agent_output=agent_output,
        error=error,
    )
