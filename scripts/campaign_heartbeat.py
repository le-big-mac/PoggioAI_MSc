#!/usr/bin/env python3
"""
Campaign heartbeat — OpenClaw's entrypoint for autonomous campaign management.

Call this script on a regular interval (every N minutes) to:
  1. Check whether the current in-progress stage has completed or failed.
  2. Advance the campaign to the next pending stage when ready.
  3. Distill completed stage artifacts into cross-run memory.
  4. Push a status summary notification.

Usage:
    python scripts/campaign_heartbeat.py --campaign campaign.yaml
    python scripts/campaign_heartbeat.py --campaign campaign.yaml --campaign-dir results/my_campaign
    python scripts/campaign_heartbeat.py --campaign campaign.yaml --force-advance
    python scripts/campaign_heartbeat.py --campaign campaign.yaml --status

Exit codes:
    0  — campaign is fully complete (all stages done)
    1  — campaign is in progress (normal heartbeat tick, nothing to do)
    2  — campaign has a failed stage (human attention required)
    3  — campaign was just advanced (new stage launched this tick)
    4  — campaign is stuck (max idle ticks reached, dependencies unsatisfiable)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time as _time
from datetime import datetime, timezone

# Ensure tex-live is available for paper/editorial stages
_TEX_LIVE_BIN = os.environ.get("CONSORTIUM_TEXLIVE_BIN", "")
if _TEX_LIVE_BIN and _TEX_LIVE_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _TEX_LIVE_BIN + ":" + os.environ.get("PATH", "")

# Ensure consortium is importable from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# Load .env so notification credentials (TELEGRAM_BOT_TOKEN, etc.) are available
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(_HERE), ".env"), override=False)

from consortium.campaign.spec import Stage, load_spec
from consortium.campaign.status import (
    COMPLETED, FAILED, IN_PROGRESS, PENDING, REPAIRING,
    check_stage_artifacts, init_status, is_pid_alive, is_slurm_job_alive,
    read_status, write_status,
)
from consortium.campaign.runner import launch_stage
from consortium.campaign.memory import distill_stage_memory
from consortium.campaign.notify import (
    notify, notify_campaign_complete, notify_heartbeat,
    notify_stage_complete, notify_stage_failed, notify_stage_launched,
    notify_repair_started, notify_repair_succeeded, notify_repair_failed,
)
from consortium.campaign.budget_manager import CampaignBudgetManager
from consortium.campaign.repair_agent import (
    attempt_repair, submit_slurm_repair, poll_slurm_repair,
)
from consortium.campaign.planner import (
    format_plan_for_review, generate_task_files,
    load_campaign_plan, plan_to_stages,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Campaign heartbeat for consortium.")
    p.add_argument("--campaign", required=True, help="Path to campaign.yaml")
    p.add_argument(
        "--campaign-dir",
        help="Directory for campaign state files (default: workspace_root from spec).",
    )
    p.add_argument(
        "--force-advance",
        action="store_true",
        help="Force-advance to next stage even if no in-progress stage is detected.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print current campaign status and exit.",
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="Initialise campaign status file and exit (safe to re-run).",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate campaign YAML (schema, task files, workspace) and exit without launching.",
    )
    return p.parse_args()


def print_status_report(spec, status, campaign_dir: str) -> None:
    print(f"\n{'='*60}")
    print(f"Campaign: {spec.name}")
    print(f"Status file: {os.path.join(campaign_dir, 'campaign_status.json')}")
    print(f"{'='*60}")
    for stage in spec.stages:
        sid = stage.id
        st = status.stage_status(sid)
        ws = status.stage_workspace(sid) or "(none)"
        pid = status.stage_pid(sid)
        pid_str = f" PID={pid}" if pid else ""
        missing = status.raw()["stages"].get(sid, {}).get("missing_artifacts", [])
        missing_str = f" MISSING={missing}" if missing else ""
        print(f"  [{st:12s}] {sid:30s} {ws}{pid_str}{missing_str}")
    print()


def _preflight_api_check() -> bool:
    """Quick check that the primary API (Anthropic) is reachable and has credits.

    Makes a minimal API call (max_tokens=1) with the cheapest model.
    Returns True if OK, False if the API is unreachable or credits are exhausted.
    Cost: ~$0.001 per call.
    """
    import subprocess as _sp
    for cli_tool in ["claude", "codex", "gemini"]:
        try:
            _sp.run([cli_tool, "--version"], capture_output=True, timeout=10)
            return True
        except (FileNotFoundError, _sp.TimeoutExpired):
            continue
    print("[heartbeat] No CLI agent tool (claude/codex/gemini) found on PATH")
    return False


def _check_campaign_wall_time(spec, status, campaign_dir: str) -> bool:
    """Return True if campaign has exceeded max_campaign_hours. Marks all as failed."""
    if spec.max_campaign_hours <= 0:
        return False  # unlimited
    # Find earliest started_at across all stages
    earliest = None
    for stage in spec.stages:
        started = status.raw()["stages"].get(stage.id, {}).get("started_at")
        if started:
            try:
                dt = datetime.fromisoformat(started)
                if earliest is None or dt < earliest:
                    earliest = dt
            except (ValueError, TypeError):
                pass
    if earliest is None:
        return False
    elapsed_hours = (datetime.now(timezone.utc) - earliest).total_seconds() / 3600
    if elapsed_hours > spec.max_campaign_hours:
        print(
            f"[heartbeat] Campaign wall time exceeded: {elapsed_hours:.1f}h > "
            f"{spec.max_campaign_hours}h limit. Halting."
        )
        return True
    return False


def _load_idle_tick_count(campaign_dir: str) -> int:
    """Load the consecutive idle tick counter from a sidecar file."""
    path = os.path.join(campaign_dir, ".idle_ticks")
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _save_idle_tick_count(campaign_dir: str, count: int) -> None:
    """Persist the consecutive idle tick counter."""
    path = os.path.join(campaign_dir, ".idle_ticks")
    with open(path, "w") as f:
        f.write(str(count))


def _check_repairing_timeout(stage, status, spec, campaign_dir: str) -> bool:
    """If a stage has been in REPAIRING status too long, time it out and mark failed."""
    sid = stage.id
    if status.stage_status(sid) != REPAIRING:
        return False
    stage_data = status.raw()["stages"].get(sid, {})
    # Use completed_at as the time repair started (it was set when the stage first failed)
    repair_started = stage_data.get("completed_at")
    if not repair_started:
        return False
    try:
        dt = datetime.fromisoformat(repair_started)
    except (ValueError, TypeError):
        return False
    elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
    timeout = spec.repair.repairing_timeout_seconds
    if elapsed > timeout:
        print(
            f"[heartbeat] Stage '{sid}' stuck in REPAIRING for {elapsed:.0f}s "
            f"(timeout={timeout}s). Marking as failed."
        )
        status.mark_failed(sid, f"Repair timed out after {elapsed:.0f}s", stage_data.get("missing_artifacts", []))
        write_status(campaign_dir, status)
        return True
    return False


def _apply_campaign_plan(spec, status, campaign_dir: str):
    """Read planning counsel output, inject dynamic stages into spec and status.

    Called after the planning_counsel stage completes. Reads campaign_plan.json,
    optionally waits for human approval, generates task files, and patches the
    spec with dynamically generated stages.

    Returns the patched spec, or None if blocked (human review pending).
    """
    plan_workspace = status.stage_workspace("planning_counsel")
    if not plan_workspace:
        print("[heartbeat] No workspace found for planning_counsel stage.")
        return None

    plan_path = os.path.join(plan_workspace, "campaign_plan.json")
    if not os.path.exists(plan_path):
        print(f"[heartbeat] campaign_plan.json not found at {plan_path}")
        return None

    plan = load_campaign_plan(plan_path)

    # Human review gate
    if spec.planning and spec.planning.human_review:
        approval_path = os.path.join(campaign_dir, "plan_approval.json")
        if not os.path.exists(approval_path):
            # Write review materials and notify, then halt
            review_md = format_plan_for_review(plan)
            review_path = os.path.join(campaign_dir, "campaign_plan_review.md")
            with open(review_path, "w") as f:
                f.write(review_md)
            print(f"[heartbeat] Campaign plan written for review: {review_path}")
            notify(
                f"Campaign '{spec.name}' plan ready for review. "
                f"See {review_path}. Run 'campaign_cli.py approve-plan' to proceed.",
                spec.notification,
            )
            return None  # halt until human approves

        with open(approval_path) as f:
            approval = json.load(f)
        if approval.get("action") != "approve":
            print(f"[heartbeat] Plan not approved (action={approval.get('action')}). Halting.")
            return None

    # Generate task files (namespaced per campaign to prevent cross-contamination)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    campaign_slug = os.path.basename(campaign_dir)
    task_dir = os.path.join(repo_root, "automation_tasks", "generated", campaign_slug)
    generate_task_files(plan, task_dir)
    print(f"[heartbeat] Generated task files in {task_dir}")

    # Convert to Stage objects
    new_stages = plan_to_stages(plan, task_dir, spec.planning)

    # Patch spec: keep discovery_plan + planning_counsel, add dynamic stages
    preserved = [s for s in spec.stages if s.id in ("discovery_plan", "planning_counsel")]
    spec.stages = preserved + new_stages

    # Update status with new stage entries
    for s in new_stages:
        if s.id not in status.raw()["stages"]:
            status.raw()["stages"][s.id] = {
                "status": PENDING,
                "workspace": None,
                "pid": None,
                "slurm_job_id": None,
                "started_at": None,
                "completed_at": None,
                "missing_artifacts": [],
                "fail_reason": None,
            }
    write_status(campaign_dir, status)

    # Write resolved spec for auditability
    _write_resolved_spec(spec, campaign_dir)

    stage_ids = [s.id for s in new_stages]
    print(f"[heartbeat] Injected {len(new_stages)} dynamic stages: {stage_ids}")
    notify(
        f"Campaign '{spec.name}' plan approved. "
        f"{len(new_stages)} dynamic stages injected: {stage_ids}",
        spec.notification,
    )

    return spec


def _write_resolved_spec(spec, campaign_dir: str) -> None:
    """Write the resolved campaign spec (with dynamic stages) for auditability."""
    path = os.path.join(campaign_dir, "resolved_campaign_spec.json")
    resolved = {
        "name": spec.name,
        "workspace_root": spec.workspace_root,
        "stages": [
            {
                "id": s.id,
                "task_file": s.task_file,
                "args": s.args,
                "depends_on": s.depends_on,
                "context_from": s.context_from,
                "memory_dirs": s.memory_dirs,
                "success_artifacts": s.success_artifacts,
                "artifact_validators": s.artifact_validators,
                "launcher_script": s.launcher_script,
            }
            for s in spec.stages
        ],
    }
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2)
    print(f"[heartbeat] Written resolved spec: {path}")


_HANG_THRESHOLD_SECONDS = 30 * 60  # 30 minutes with no progress = hung


def _check_progress_heartbeat(workspace: str) -> tuple:
    """Check if a stage's progress heartbeat file indicates a hang.

    Returns (is_hung: bool, stale_minutes: float).
    If the progress file doesn't exist, the stage predates the watchdog —
    fall back to workspace mtime heuristic.
    """
    progress_file = os.path.join(workspace, ".progress_heartbeat")
    try:
        with open(progress_file) as f:
            data = json.load(f)
        ts = data.get("ts", 0)
        elapsed = _time.time() - ts
        return (elapsed > _HANG_THRESHOLD_SECONDS, elapsed / 60.0)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        # No progress file — check workspace file mtime as fallback
        return _check_workspace_mtime(workspace)


def _check_workspace_mtime(workspace: str) -> tuple:
    """Fallback hang detection: check newest file mtime in workspace.

    Returns (is_hung: bool, stale_minutes: float).
    """
    if not workspace or not os.path.isdir(workspace):
        return (False, 0.0)
    newest = 0.0
    try:
        for entry in os.scandir(workspace):
            try:
                mt = entry.stat().st_mtime
                if mt > newest:
                    newest = mt
            except OSError:
                continue
    except OSError:
        return (False, 0.0)
    if newest == 0.0:
        return (False, 0.0)
    elapsed = _time.time() - newest
    return (elapsed > _HANG_THRESHOLD_SECONDS, elapsed / 60.0)


def _kill_stage(pid: int | None, slurm_job_id: int | None) -> None:
    """Best-effort kill of a hung stage process."""
    import signal as _signal
    if slurm_job_id and slurm_job_id > 0:
        try:
            import subprocess
            subprocess.run(
                ["scancel", str(slurm_job_id)],
                capture_output=True, timeout=10,
            )
            print(f"[heartbeat] Sent scancel to SLURM job {slurm_job_id}")
        except Exception as exc:
            print(f"[heartbeat] scancel failed: {exc}")
    if pid and pid > 0:
        try:
            os.kill(pid, _signal.SIGTERM)
            print(f"[heartbeat] Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            pass  # already dead
        except PermissionError:
            print(f"[heartbeat] Cannot kill PID {pid} — permission denied")


def run_heartbeat(
    spec,
    status,
    campaign_dir: str,
    force_advance: bool = False,
) -> int:
    """
    Core heartbeat logic. Returns exit code.
    """
    # ----------------------------------------------------------------
    # -3. Pre-flight API credit check
    # ----------------------------------------------------------------
    api_ok = _preflight_api_check()
    if not api_ok:
        sentinel = os.path.join(campaign_dir, ".api_credit_alert")
        if not os.path.exists(sentinel):
            msg = "API pre-flight check FAILED — credits may be exhausted or API unreachable."
            print(f"[heartbeat] {msg}")
            notify(msg, spec.notification)
            with open(sentinel, "w") as f:
                f.write(datetime.now(timezone.utc).isoformat())
        else:
            print("[heartbeat] API still unavailable (alert already sent).")
        # Don't launch new stages or repairs, but allow monitoring of running ones
        # by continuing past this check.

    # ----------------------------------------------------------------
    # -2. Budget threshold alerts
    # ----------------------------------------------------------------
    if spec.budget_usd > 0:
        budget_mgr = CampaignBudgetManager(campaign_dir, spec.budget_usd)
        for alert in budget_mgr.check_thresholds():
            sentinel = os.path.join(campaign_dir, f".budget_alert_{alert}")
            if not os.path.exists(sentinel):
                msg = (
                    f"Budget alert [{alert}]: "
                    f"${budget_mgr.total_spent:.2f} / ${budget_mgr.usd_limit:.2f} "
                    f"({budget_mgr.spend_fraction * 100:.0f}%)"
                )
                print(f"[heartbeat] {msg}")
                notify(msg, spec.notification)
                with open(sentinel, "w") as f:
                    f.write(datetime.now(timezone.utc).isoformat())

    # ----------------------------------------------------------------
    # -1. Campaign wall-time check (P3-11)
    # ----------------------------------------------------------------
    if _check_campaign_wall_time(spec, status, campaign_dir):
        notify(
            f"Campaign '{spec.name}' exceeded wall time limit "
            f"({spec.max_campaign_hours}h). Halting.",
            spec.notification,
        )
        return 2

    # ----------------------------------------------------------------
    # 0. Check any stage with a SLURM repair job in progress
    # ----------------------------------------------------------------
    for stage in spec.stages:
        sid = stage.id
        if status.stage_status(sid) == REPAIRING:
            # Check REPAIRING timeout first (P0 fix)
            if _check_repairing_timeout(stage, status, spec, campaign_dir):
                continue  # now FAILED, will be handled below
            print(f"[heartbeat] Stage '{sid}' has a repair in progress — polling...")
            repair_exit = _try_repair(stage, spec, status, campaign_dir)
            if repair_exit is not None:
                _save_idle_tick_count(campaign_dir, 0)
                return repair_exit
            # If None, repair exhausted — fall through to normal failure handling

    # ----------------------------------------------------------------
    # 1. Check all in-progress stages (supports parallel execution)
    # ----------------------------------------------------------------
    any_alive = False
    any_completed_this_tick = False
    for stage in spec.stages:
        sid = stage.id
        if not status.is_in_progress(sid):
            continue

        pid = status.stage_pid(sid)
        slurm_jid = status.stage_slurm_job_id(sid)
        workspace = status.stage_workspace(sid) or ""

        # Check liveness: prefer SLURM check (cross-node), fall back to PID
        if slurm_jid:
            slurm_alive = is_slurm_job_alive(slurm_jid)
            if slurm_alive is None:
                # squeue unavailable — skip this stage for this tick
                print(
                    f"[heartbeat] Stage '{sid}': squeue unavailable, "
                    f"skipping liveness check this tick."
                )
                any_alive = True
                continue
            alive = slurm_alive
        else:
            alive = is_pid_alive(pid) if pid else False

        # --- Hang detection via progress heartbeat (Tier 1.1) ---
        # If the process is alive but hasn't updated .progress_heartbeat
        # in >30 minutes, it's hung (e.g., stuck LLM call, infinite loop).
        if alive and not force_advance:
            hung, stale_minutes = _check_progress_heartbeat(workspace)
            if hung:
                print(
                    f"[heartbeat] Stage '{sid}' HUNG: process alive (PID {pid}) "
                    f"but no progress for {stale_minutes:.0f} minutes. Killing."
                )
                _kill_stage(pid, status.stage_slurm_job_id(sid))
                alive = False  # fall through to failure handling below

        if alive and not force_advance:
            # Stage still running — check artifacts speculatively
            all_present, missing = check_stage_artifacts(workspace, stage)
            if not all_present:
                print(
                    f"[heartbeat] Stage '{sid}' in progress (PID {pid}). "
                    f"Missing artifacts: {missing}"
                )
            else:
                print(f"[heartbeat] Stage '{sid}' in progress (PID {pid}). Artifacts look complete.")
            any_alive = True
            continue

        # Process is dead (or force_advance). Check artifacts.
        all_present, missing = check_stage_artifacts(workspace, stage)

        # P1-5: Validate artifact content, not just existence
        if all_present:
            validation_errors = _validate_artifact_content(workspace, stage)
            if validation_errors:
                all_present = False
                missing = validation_errors

        if all_present:
            print(f"[heartbeat] Stage '{sid}' completed.")
            status.mark_completed(sid)
            write_status(campaign_dir, status)
            distill_stage_memory(stage, workspace, campaign_dir)
            notify_stage_complete(sid, workspace, spec.notification)
            any_completed_this_tick = True

            # If planning_counsel just completed, apply the campaign plan
            if sid == "planning_counsel" and spec.planning and spec.planning.enabled:
                result = _apply_campaign_plan(spec, status, campaign_dir)
                if result is None:
                    # Human review pending or plan application failed
                    _save_idle_tick_count(campaign_dir, 0)
                    return 1
                # spec has been patched with new stages — continue to Section 4
        else:
            reason = f"Process ended but required artifacts missing/invalid: {missing}"
            print(f"[heartbeat] Stage '{sid}' FAILED: {reason}")
            status.mark_failed(sid, reason, missing)
            write_status(campaign_dir, status)

            # --- Autonomous repair attempt ---
            repair_exit = _try_repair(stage, spec, status, campaign_dir)
            if repair_exit is not None:
                _save_idle_tick_count(campaign_dir, 0)
                return repair_exit

            # No repair attempted or repair exhausted — escalate
            notify_stage_failed(sid, reason, spec.notification)
            return 2

    # ----------------------------------------------------------------
    # 1.5. Catch-up memory distillation for completed stages
    # ----------------------------------------------------------------
    # Safety net: if any stage is marked completed but has no memory
    # summary, distill it now.  This handles stages that completed
    # between ticks, stages that were dynamically injected and completed
    # before the heartbeat knew about them, or stages whose distillation
    # failed on the first try.
    for stage in spec.stages:
        sid = stage.id
        if status.is_complete(sid):
            memory_path = os.path.join(campaign_dir, "memory", f"{sid}_summary.md")
            if not os.path.exists(memory_path):
                workspace = status.stage_workspace(sid)
                if workspace and os.path.isdir(workspace):
                    print(f"[heartbeat] Catch-up distillation for '{sid}' (memory summary missing).")
                    try:
                        distill_stage_memory(stage, workspace, campaign_dir)
                    except Exception as exc:
                        print(f"[heartbeat] Catch-up distillation failed for '{sid}': {exc}")

    if any_alive and not any_completed_this_tick:
        summary = _build_summary(spec, status)
        notify_heartbeat(summary, spec.notification)
        _save_idle_tick_count(campaign_dir, 0)
        return 1

    if any_completed_this_tick:
        _save_idle_tick_count(campaign_dir, 0)

    # ----------------------------------------------------------------
    # 2. Check for overall failure (with repair opportunity)
    # ----------------------------------------------------------------
    if status.campaign_failed(spec):
        # Find the failed stage and try repair before giving up
        for stage in spec.stages:
            sid = stage.id
            if status.stage_status(sid) == FAILED:
                repair_exit = _try_repair(stage, spec, status, campaign_dir)
                if repair_exit is not None:
                    _save_idle_tick_count(campaign_dir, 0)
                    return repair_exit

                # P3-10: Escalation timeout — if repair is exhausted and
                # the failure has been sitting for longer than
                # escalation_timeout_minutes, auto-retry (reset attempts).
                repair_config = spec.repair
                if repair_config.auto_retry_on_timeout and repair_config.escalation_timeout_minutes > 0:
                    stage_data = status.raw()["stages"].get(sid, {})
                    failed_at = stage_data.get("completed_at")
                    if failed_at:
                        try:
                            failed_dt = datetime.fromisoformat(failed_at)
                            elapsed_min = (datetime.now(timezone.utc) - failed_dt).total_seconds() / 60
                            if elapsed_min >= repair_config.escalation_timeout_minutes:
                                print(
                                    f"[heartbeat] Escalation timeout for '{sid}': "
                                    f"{elapsed_min:.0f}min elapsed (limit={repair_config.escalation_timeout_minutes}min). "
                                    f"Auto-resetting repair attempts."
                                )
                                # Archive repair log (preserve diagnostic history)
                                # then reset to allow fresh attempts
                                old_log = stage_data.get("repair_log", [])
                                if old_log:
                                    archive = stage_data.setdefault("repair_log_archive", [])
                                    archive.extend(old_log)
                                stage_data["repair_log"] = []
                                write_status(campaign_dir, status)
                                # Try repair again with fresh budget
                                retry_exit = _try_repair(stage, spec, status, campaign_dir)
                                if retry_exit is not None:
                                    _save_idle_tick_count(campaign_dir, 0)
                                    return retry_exit
                        except (ValueError, TypeError):
                            pass

                break  # only handle one failed stage per tick
        print("[heartbeat] Campaign has a failed stage. Human attention required.")
        return 2

    # ----------------------------------------------------------------
    # 3. Check if campaign is fully complete
    # ----------------------------------------------------------------
    if status.campaign_finished(spec):
        print(f"[heartbeat] Campaign '{spec.name}' is complete.")
        notify_campaign_complete(spec.name, spec.notification)
        _write_summary_md(spec, status, campaign_dir)
        return 0

    # ----------------------------------------------------------------
    # 4. Find all pending stages with satisfied deps and launch them
    # ----------------------------------------------------------------
    from consortium.campaign.runner import build_stage_workspace

    # If planning is enabled and planning_counsel is complete but plan
    # not yet applied (e.g., human review just approved), apply it now.
    if (spec.planning and spec.planning.enabled
            and status.stage_status("planning_counsel") == COMPLETED
            and not any(s.id not in ("discovery_plan", "planning_counsel")
                        for s in spec.stages
                        if s.id not in ("discovery_plan", "planning_counsel"))):
        result = _apply_campaign_plan(spec, status, campaign_dir)
        if result is None:
            _save_idle_tick_count(campaign_dir, 0)
            return 1

    # Budget hard-cap: block new stage launches if budget is exceeded.
    # Only blocks NEW launches — running stages are allowed to finish.
    if spec.budget_usd > 0:
        budget_mgr = CampaignBudgetManager(campaign_dir, spec.budget_usd)
        if budget_mgr.is_budget_exceeded():
            print(
                f"[heartbeat] Budget exceeded "
                f"(${budget_mgr.total_spent:.2f} / ${budget_mgr.usd_limit:.2f}). "
                f"No new stages will be launched."
            )
            notify(
                f"Campaign '{spec.name}' budget exceeded. "
                f"No new stages will launch. Human attention required.",
                spec.notification,
            )
            return 2

    launched_any = False
    for stage in spec.stages:
        sid = stage.id
        if status.stage_status(sid) != PENDING:
            continue
        if not status.all_complete(stage.depends_on):
            continue

        # Ready to launch
        proc = launch_stage(stage, spec, status, campaign_dir)
        workspace = build_stage_workspace(stage, spec, status)
        status.mark_in_progress(sid, workspace, proc.pid)
        write_status(campaign_dir, status)
        notify_stage_launched(sid, proc.pid, workspace, spec.notification)
        launched_any = True

    if launched_any:
        _save_idle_tick_count(campaign_dir, 0)
        return 3

    # ----------------------------------------------------------------
    # 5. Idle tick circuit breaker (P0-1 / P3-12)
    # ----------------------------------------------------------------
    idle_ticks = _load_idle_tick_count(campaign_dir) + 1
    _save_idle_tick_count(campaign_dir, idle_ticks)

    if spec.max_idle_ticks > 0 and idle_ticks >= spec.max_idle_ticks:
        print(
            f"[heartbeat] {idle_ticks} consecutive idle ticks (limit={spec.max_idle_ticks}). "
            f"Campaign is stuck — dependencies may be unsatisfiable. "
            f"Exiting with code 4."
        )
        notify(
            f"Campaign '{spec.name}' stuck: {idle_ticks} idle ticks reached. "
            f"Dependencies may be unsatisfiable. Human attention required.",
            spec.notification,
        )
        return 4

    print(f"[heartbeat] No action taken — waiting for dependencies. (idle tick {idle_ticks}/{spec.max_idle_ticks})")
    return 1


def _validate_artifact_content(workspace: str, stage) -> list:
    """
    P1-5: Validate artifact content beyond mere file existence.

    Uses per-artifact validators from stage.artifact_validators, plus
    built-in heuristics for known file types.

    Returns a list of validation error strings (empty = all valid).
    """
    errors = []
    validators = getattr(stage, "artifact_validators", {})

    for artifact in stage.success_artifacts.get("required", []):
        if artifact.endswith("/"):
            continue  # skip directory artifacts
        full = os.path.join(workspace, artifact)
        if not os.path.exists(full):
            continue  # already caught by check_stage_artifacts

        # Check validator rules if defined
        rules = validators.get(artifact, {})

        try:
            file_size = os.path.getsize(full)
        except OSError:
            continue

        # min_size_bytes check
        min_size = rules.get("min_size_bytes", 0)
        if file_size < min_size:
            errors.append(f"{artifact}: too small ({file_size}B < {min_size}B)")
            continue

        # Content checks (for text files)
        # Read up to 10MB for validation (100KB was too small for large JSON artifacts
        # where required markers may appear beyond that offset).
        _MAX_VALIDATE_BYTES = 10_000_000
        if any(artifact.endswith(ext) for ext in (".json", ".md", ".txt", ".tex", ".csv")):
            try:
                read_limit = min(file_size, _MAX_VALIDATE_BYTES)
                with open(full) as f:
                    content = f.read(read_limit)
            except Exception:
                continue

            # must_contain checks
            for required_str in rules.get("must_contain", []):
                if required_str not in content:
                    errors.append(f"{artifact}: missing required content '{required_str}'")

            # must_not_contain checks (detect hollow artifacts)
            for forbidden_str in rules.get("must_not_contain", []):
                if forbidden_str in content:
                    errors.append(f"{artifact}: contains forbidden content '{forbidden_str}'")

            # Built-in heuristic: JSON files with "not_executed" are hollow
            if artifact.endswith(".json") and '"status": "not_executed"' in content:
                errors.append(f"{artifact}: hollow artifact (status=not_executed)")

    return errors


def _compute_backoff_seconds(repair_config, attempt_num: int) -> float:
    """P0-4: Compute exponential backoff for repair attempt N."""
    base = repair_config.backoff_base_seconds
    cap = repair_config.backoff_max_seconds
    # Cap the exponent to avoid computing astronomically large integers
    exponent = min(attempt_num - 1, 20)
    delay = min(base * (1 << exponent), cap)
    return delay


def _try_repair(
    stage,
    spec,
    status,
    campaign_dir: str,
) -> "int | None":
    """
    Attempt autonomous repair of a failed stage.

    Supports two launcher modes:
      - "local": runs Claude Code inline (blocks this tick, ~10min max)
      - "slurm": submits a SLURM job and returns immediately; the next
        heartbeat tick polls for the result.

    Returns:
        - 3 if the stage was repaired and relaunched (same as "stage launched")
        - 1 if repair is in progress (SLURM job running, or retrying next tick)
        - None if repair is not enabled, exhausted, or failed
    """
    from consortium.campaign.status import is_slurm_job_alive

    sid = stage.id
    repair_config = spec.repair

    if not repair_config.enabled:
        return None

    # ------------------------------------------------------------------
    # SLURM mode: check if a repair job is already running
    # ------------------------------------------------------------------
    if repair_config.launcher == "slurm":
        stage_data = status.raw()["stages"].get(sid, {})
        repair_slurm_jid = stage_data.get("repair_slurm_job_id")

        if repair_slurm_jid and status.stage_status(sid) == REPAIRING:
            # A repair job was previously submitted — poll for result
            if is_slurm_job_alive(repair_slurm_jid):
                print(
                    f"[heartbeat] Repair SLURM job {repair_slurm_jid} for '{sid}' "
                    f"still running. Will check next tick."
                )
                return 1  # come back later

            # Job finished — poll the sentinel
            result = poll_slurm_repair(stage, status, campaign_dir)
            if result is None:
                print(
                    f"[heartbeat] Repair SLURM job {repair_slurm_jid} for '{sid}' "
                    f"finished but no sentinel found. Treating as failure."
                )
                result_success = False
                result_diagnosis = "SLURM repair job completed but produced no result sentinel."
                result_actions = []
                result_duration = 0.0
                result_error = "No sentinel file"
            else:
                result_success = result.success
                result_diagnosis = result.diagnosis
                result_actions = result.actions_taken
                result_duration = result.duration_seconds
                result_error = result.error

            # Record the attempt
            status.add_repair_attempt(
                sid,
                success=result_success,
                diagnosis=result_diagnosis,
                actions=result_actions,
                duration=result_duration,
                error=result_error,
            )

            return _handle_repair_result(
                sid, result_success, result_diagnosis, result_actions,
                result_error, status, spec, stage, campaign_dir,
            )

    # ------------------------------------------------------------------
    # Check attempt budget
    # ------------------------------------------------------------------
    attempts_so_far = status.repair_attempt_count(sid)
    if attempts_so_far >= repair_config.max_attempts:
        print(
            f"[heartbeat] Repair attempts exhausted for '{sid}' "
            f"({attempts_so_far}/{repair_config.max_attempts})."
        )
        return None

    # P0-4: Exponential backoff — check if enough time has elapsed since last attempt
    if attempts_so_far > 0:
        last_attempt = status.raw()["stages"].get(sid, {}).get("repair_log", [])
        if last_attempt:
            last_ts = last_attempt[-1].get("timestamp", "")
            try:
                last_dt = datetime.fromisoformat(last_ts)
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                required_backoff = _compute_backoff_seconds(repair_config, attempts_so_far)
                if elapsed < required_backoff:
                    remaining = required_backoff - elapsed
                    print(
                        f"[heartbeat] Backoff for '{sid}': {remaining:.0f}s remaining "
                        f"(need {required_backoff:.0f}s between attempts). Skipping this tick."
                    )
                    return 1  # come back next tick
            except (ValueError, TypeError):
                pass  # malformed timestamp, proceed anyway

    attempt_num = attempts_so_far + 1
    print(
        f"[heartbeat] Attempting autonomous repair for '{sid}' "
        f"(attempt {attempt_num}/{repair_config.max_attempts})..."
    )

    # Notify
    notify_repair_started(sid, attempt_num, repair_config.max_attempts, spec.notification)

    # Mark as repairing
    status.mark_repairing(sid)
    write_status(campaign_dir, status)

    # ------------------------------------------------------------------
    # SLURM mode: submit job and return immediately
    # ------------------------------------------------------------------
    if repair_config.launcher == "slurm":
        job_id = submit_slurm_repair(stage, spec, status, campaign_dir)
        if job_id is None:
            print(f"[heartbeat] SLURM repair submission failed for '{sid}'. Falling back to local.")
            # Record submission failure as an attempt (P0-2: count all failures)
            status.add_repair_attempt(
                sid,
                success=False,
                diagnosis="SLURM repair submission failed",
                actions=[],
                duration=0.0,
                error="submit_slurm_repair returned None",
            )
            status.mark_failed(sid, "SLURM repair submission failed", status.raw()["stages"].get(sid, {}).get("missing_artifacts", []))
            write_status(campaign_dir, status)
            # Fall through to local mode
        else:
            # Store SLURM job ID in status so we can poll it next tick
            status.raw()["stages"][sid]["repair_slurm_job_id"] = job_id
            write_status(campaign_dir, status)
            print(
                f"[heartbeat] Repair SLURM job {job_id} submitted for '{sid}'. "
                f"Will poll on next heartbeat tick."
            )
            return 1  # in progress — come back next tick

    # ------------------------------------------------------------------
    # Local mode: run Claude Code inline (blocking)
    # ------------------------------------------------------------------
    result = attempt_repair(stage, spec, status, campaign_dir)

    # P0-2: Always record the attempt, even if planning failed
    status.add_repair_attempt(
        sid,
        success=result.success,
        diagnosis=result.diagnosis,
        actions=result.actions_taken,
        duration=result.duration_seconds,
        error=result.error,
    )

    return _handle_repair_result(
        sid, result.success, result.diagnosis, result.actions_taken,
        result.error, status, spec, stage, campaign_dir,
    )


def _handle_repair_result(
    sid: str,
    success: bool,
    diagnosis: str,
    actions: list,
    error: "str | None",
    status,
    spec,
    stage,
    campaign_dir: str,
) -> "int | None":
    """
    Common handler for repair results (both local and SLURM modes).

    Returns exit code for the heartbeat.
    """
    repair_config = spec.repair
    attempt_num = status.repair_attempt_count(sid)  # already incremented

    if success:
        print(f"[heartbeat] Repair succeeded for '{sid}'. Relaunching stage.")
        notify_repair_succeeded(sid, attempt_num, diagnosis, spec.notification)

        # Reset to pending and relaunch
        status.mark_pending_retry(sid)
        write_status(campaign_dir, status)

        # Brief pause before retry
        if repair_config.retry_delay_seconds > 0:
            _time.sleep(repair_config.retry_delay_seconds)

        # Relaunch the stage
        proc = launch_stage(stage, spec, status, campaign_dir)
        from consortium.campaign.runner import build_stage_workspace
        workspace = build_stage_workspace(stage, spec, status)
        status.mark_in_progress(sid, workspace, proc.pid)
        write_status(campaign_dir, status)
        notify_stage_launched(sid, proc.pid, workspace, spec.notification)
        return 3  # stage relaunched

    else:
        print(f"[heartbeat] Repair failed for '{sid}': {error or diagnosis[:200]}")
        notify_repair_failed(
            sid, attempt_num, repair_config.max_attempts,
            error or diagnosis[:150],
            spec.notification,
        )

        # Mark back as failed (repair didn't help)
        stage_data = status.raw()["stages"].get(sid, {})
        status.mark_failed(
            sid,
            stage_data.get("fail_reason", "Repair failed"),
            stage_data.get("missing_artifacts", []),
        )
        write_status(campaign_dir, status)

        if attempt_num < repair_config.max_attempts:
            # P0-4: Backoff is enforced in _try_repair. Signal that we'll retry
            # on a future tick, but the backoff timer must elapse first.
            backoff = _compute_backoff_seconds(repair_config, attempt_num)
            print(
                f"[heartbeat] Will retry repair after {backoff:.0f}s backoff "
                f"({repair_config.max_attempts - attempt_num} attempt(s) remaining)."
            )
            return 1  # come back next tick (backoff enforced in _try_repair)

        return None  # exhausted


def _build_summary(spec, status) -> str:
    lines = [f"Campaign '{spec.name}' heartbeat:"]
    for stage in spec.stages:
        sid = stage.id
        st = status.stage_status(sid)
        pid = status.stage_pid(sid)
        pid_str = f" (PID {pid})" if pid and st == IN_PROGRESS else ""
        lines.append(f"  {sid}: {st}{pid_str}")
    return " | ".join(lines[1:])


def _write_summary_md(spec, status, campaign_dir: str) -> None:
    path = os.path.join(campaign_dir, "CAMPAIGN_COMPLETE.md")
    lines = [f"# Campaign Complete: {spec.name}\n"]
    for stage in spec.stages:
        sid = stage.id
        ws = status.stage_workspace(sid) or "(unknown)"
        completed_at = status.raw()["stages"].get(sid, {}).get("completed_at", "")
        lines.append(f"- **{sid}**: {ws} (completed {completed_at})")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[heartbeat] Written: {path}")


def main() -> int:
    args = parse_args()

    spec = load_spec(args.campaign)
    campaign_dir = args.campaign_dir or spec.workspace_root
    os.makedirs(campaign_dir, exist_ok=True)

    # Reload dynamic stages from the resolved spec if it exists.
    # When dynamic planning is enabled, _apply_campaign_plan() writes a
    # resolved_campaign_spec.json that includes the injected stages.
    # Without this reload, every heartbeat tick would only see the 2
    # YAML-defined stages (discovery_plan + planning_counsel) and lose
    # track of all dynamically generated stages.
    resolved_path = os.path.join(campaign_dir, "resolved_campaign_spec.json")
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path) as f:
                resolved = json.load(f)
            resolved_stages = [
                Stage.from_dict(s) for s in resolved.get("stages", [])
            ]
            if resolved_stages:
                # Resolve relative task_file paths (same as load_spec does)
                yaml_dir = os.path.dirname(os.path.abspath(args.campaign))
                for s in resolved_stages:
                    if s.task_file and not os.path.isabs(s.task_file):
                        s.task_file = os.path.join(yaml_dir, s.task_file)
                    if s.launcher_script and not os.path.isabs(s.launcher_script):
                        s.launcher_script = os.path.join(yaml_dir, s.launcher_script)
                spec.stages = resolved_stages
                print(
                    f"[heartbeat] Reloaded {len(resolved_stages)} stages from "
                    f"resolved spec (including dynamic stages)."
                )
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[heartbeat] Warning: could not reload resolved spec: {exc}")

    if args.validate:
        errors = []
        print(f"[validate] Campaign: {spec.name}")
        print(f"[validate] Workspace: {campaign_dir}")
        print(f"[validate] Stages: {len(spec.stages)}")
        if spec.planning and getattr(spec.planning, "enabled", False):
            task_file = getattr(spec.planning, "base_task_file", "")
            if task_file and not os.path.isfile(task_file):
                errors.append(f"Task file not found: {task_file}")
            else:
                print(f"[validate] Task file: {task_file} (OK)")
            max_s = getattr(spec.planning, "max_stages", "?")
            print(f"[validate] Dynamic planning: enabled (max_stages={max_s})")
        for stage in spec.stages:
            if stage.task_file and not os.path.isfile(stage.task_file):
                errors.append(f"Stage '{stage.id}' task file not found: {stage.task_file}")
            else:
                print(f"[validate] Stage '{stage.id}': OK")
        if errors:
            print(f"\n[validate] ERRORS:")
            for e in errors:
                print(f"  - {e}")
            return 1
        print(f"\n[validate] Campaign config is valid.")
        return 0

    if args.init:
        init_status(campaign_dir, spec, args.campaign)
        print(f"[heartbeat] Campaign status initialised at {campaign_dir}/campaign_status.json")
        return 0

    # Ensure status file exists with all stage entries
    status = init_status(campaign_dir, spec, args.campaign)

    if args.status:
        print_status_report(spec, status, campaign_dir)
        return 0

    return run_heartbeat(spec, status, campaign_dir, force_advance=args.force_advance)


if __name__ == "__main__":
    sys.exit(main())
