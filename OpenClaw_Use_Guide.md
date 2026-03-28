# OpenClaw Use Guide: Deploying the consortium Research Pipeline

This guide covers everything you need to run `consortium` as an autonomous, multi-stage research pipeline under OpenClaw's control.

OpenClaw interacts with this system via two surfaces:

- **Campaign manager** — OpenClaw calls `scripts/campaign_heartbeat.py` on a schedule to advance a multi-stage campaign (theory → experiments → paper) from start to finish without human intervention.
- **HTTP steering API** — While a stage is running, OpenClaw can pause it and inject mid-run instructions via a REST API on port 5002.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Mental Model](#2-mental-model)
3. [campaign.yaml Anatomy](#3-campaignyaml-anatomy)
4. [Writing Task Files](#4-writing-task-files)
5. [Running a Campaign](#5-running-a-campaign)
6. [Scheduling Heartbeats](#6-scheduling-heartbeats)
7. [Understanding Campaign Outputs](#7-understanding-campaign-outputs)
8. [Live Steering via HTTP](#8-live-steering-via-http)
9. [Notifications Setup](#9-notifications-setup)
10. [Failure Recovery](#10-failure-recovery)
11. [Adapting to Your Own Research Topic](#11-adapting-to-your-own-research-topic)
12. [Reference](#12-reference)

---

## 1. Prerequisites

### Environment

The pipeline must be installed and working before adding campaign orchestration on top of it. If you have not done this yet:

```bash
./scripts/bootstrap.sh researchlab full
conda activate researchlab
python scripts/preflight_check.py --with-docs --with-web --with-experiment --with-latex
```

Verify a single manual run works end-to-end before using OpenClaw.

### CLI Agent Tools

At minimum, have one CLI agent tool installed. For the default campaign config, which uses `claude` as the main CLI tool, verify it is on PATH:

```bash
which claude
```

If using Model Counsel (`--enable-counsel` in stage `args`), all three CLI tools are required:

```bash
which claude    # Anthropic Claude CLI
which codex     # OpenAI Codex CLI
which gemini    # Google Gemini CLI
```

No API keys are needed — CLI tools authenticate through their own subscriptions (Claude Max, ChatGPT Pro, Gemini Advanced, etc.).

For notifications, optionally set in `.env`:

```
SLACK_WEBHOOK_URL=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

---

## 2. Mental Model

```
OpenClaw
   |
   | calls every N minutes
   v
campaign_heartbeat.py  ←→  campaign_status.json   (state file)
   |
   | subprocess.Popen()
   v
launch_multiagent.py   (one running stage at a time)
   |
   | artifacts written to workspace
   v
results/campaign/<stage_id>/
```

OpenClaw is the **scheduler**. The heartbeat script is the **state machine**. Each stage is an independent `launch_multiagent.py` subprocess. Stages are chained through workspace reuse (`--resume`) and context injection (prior-stage memory summaries prepended to the task prompt).

The heartbeat script is **idempotent** — calling it when nothing has changed (stage still running) is a no-op with exit code 1.

---

## 3. `campaign.yaml` Anatomy

The repo ships with `campaign.yaml` at the root as a working three-stage example. Here is a complete field-by-field breakdown:

```yaml
name: "My Research Campaign"          # Human-readable label; used in notifications

workspace_root: "results/my_campaign" # Parent directory for all stage workspaces.
                                       # Relative to repo root.

heartbeat_interval_minutes: 30        # Informational — tells OpenClaw how often to tick.
                                       # The heartbeat script itself does not sleep; OpenClaw
                                       # is responsible for calling it on this interval.

notification:
  slack_webhook: "${SLACK_WEBHOOK_URL}"       # Env var reference; omit or null to disable.
  telegram_bot_token: "${TELEGRAM_BOT_TOKEN}" # Optional.
  telegram_chat_id: "${TELEGRAM_CHAT_ID}"     # Required if telegram_bot_token is set.
  on_stage_complete: true    # Notify when a stage finishes successfully.
  on_failure: true           # Notify when a stage fails.
  on_heartbeat: false        # Set true for a progress ping every heartbeat tick.

stages:
  - id: theory                            # Unique stage identifier. Used in file names,
                                          # dependency references, and log files.

    task_file: automation_tasks/run1_theory_task_stable.txt
                                          # Path to the task prompt text file.
                                          # Relative to repo root.

    args:                                 # Extra CLI flags passed to launch_multiagent.py.
      - "--enable-math-agents"

    success_artifacts:                    # What must exist in the workspace for this
                                          # stage to be considered complete.
      required:
        - math_workspace/claim_graph.json # File path relative to workspace root.
      optional:                           # Optional artifacts — checked but not blocking.
        - math_workspace/proofs/          # Trailing / means a directory.
        - math_workspace/checks/

    memory_dirs:                          # Directories to walk during memory distillation.
      - math_workspace/                   # Up to 20 files, 1500 chars each, included in
                                          # the summary prepended to downstream stages.

  - id: experiments
    task_file: automation_tasks/run2_experiment_task_stable.txt
    depends_on: theory                    # This stage will not launch until 'theory' is
                                          # COMPLETED. String or list of strings.
    context_from: theory                  # Resume the theory workspace so this stage can
                                          # read its artifacts directly, AND prepend the
                                          # theory memory summary to the task prompt.
    args: []
    success_artifacts:
      required:
        - experiment_results.json
        - experiment_analysis.md
      optional:
        - fairness_protocol.json
        - experiment_plan.md

  - id: paper
    task_file: automation_tasks/run3_paper_task_stable.txt
    depends_on:
      - theory
      - experiments
    context_from:
      - experiments    # First listed context_from is the workspace that gets resumed.
      - theory         # Additional entries only contribute their memory summaries.
    args:
      - "--require-pdf"
    success_artifacts:
      required:
        - final_paper.tex
        - final_paper.pdf
      optional:
        - analysis_connection.md
        - artifacts_index.md
```

Older campaign files may still include `--pipeline-mode` entries in `args`; current launcher behavior accepts them but ignores them.

### `depends_on` vs `context_from`

These are distinct and serve different purposes:

| Field | What it does |
|---|---|
| `depends_on` | Ordering gate only — this stage will not launch until all listed stages are `completed`. Does not affect the workspace or task prompt. |
| `context_from` | Workspace resume + memory injection. The runner resumes the first listed stage's workspace (so the new stage inherits its files), and prepends a distilled memory summary from each listed stage to the task prompt. |

A stage can `depends_on` a stage without `context_from` if it does not need its artifacts directly (e.g., a parallel stage that just needs to sequence after another).

---

## 4. Writing Task Files

Each stage's `task_file` is a plain `.txt` file containing the task prompt that will be passed to `launch_multiagent.py --task`. When the stage has `context_from` entries, the runner automatically prepends a block like:

```
--- Context from 'theory' stage ---
# Stage Summary: theory

## math_workspace/claim_graph.json

```json
{
  "claims": [...]
}
```

## Budget

Invocations used: 47

## Wall-Clock Time

Elapsed: 2h 15m
---

[your task file text here]
```

The agent receives this combined prompt. This means:

- **Your task file does not need to repeat prior-stage results.** The memory summary handles that.
- **Reference artifact names explicitly.** If stage 2 should build on `math_workspace/claim_graph.json`, name it in the task file so the agent knows to look there.
- **Keep scope tight.** Each stage task should have one primary deliverable. The more focused the task, the less likely an agent is to drift or repeat prior work.

### Effective task file structure

```
[One-sentence goal]

Objectives:
1) ...
2) ...
3) ...

Required outputs:
- artifact_name.ext (what it should contain)
- ...

Execution constraints:
- [tool restrictions, sub-agent restrictions, network access policy]
```

See `automation_tasks/run1_theory_task_stable.txt`, `run2_experiment_task_stable.txt`, and `run3_paper_task_stable.txt` for concrete examples.

---

## 5. Running a Campaign

### Step 1: Initialise

Creates `campaign_status.json` with all stages set to `pending`. Safe to re-run.

```bash
python scripts/campaign_heartbeat.py --campaign campaign.yaml --init
```

Optionally specify a different campaign directory:

```bash
python scripts/campaign_heartbeat.py \
  --campaign campaign.yaml \
  --campaign-dir results/my_run_001 \
  --init
```

### Step 2: Check initial status

```bash
python scripts/campaign_heartbeat.py --campaign campaign.yaml --status
```

Output:

```
============================================================
Campaign: Muon Implicit Regularization
Status file: results/muon_campaign/campaign_status.json
============================================================
  [pending      ] theory                         (none)
  [pending      ] experiments                    (none)
  [pending      ] paper                          (none)
```

### Step 3: Tick the heartbeat

```bash
python scripts/campaign_heartbeat.py --campaign campaign.yaml
```

On the first tick with all stages pending and no dependencies blocking, the first stage launches. Exit code `3` is returned to signal that a new stage was just started.

On subsequent ticks while a stage is running, exit code `1` is returned (in-progress, nothing to do). When the stage finishes, the next tick detects completion (PID dead + required artifacts present), distills memory, and launches the next stage.

### Step 4: Monitor progress

```bash
# Human-readable status table
python scripts/campaign_heartbeat.py --campaign campaign.yaml --status

# Live stage logs
tail -f results/muon_campaign/logs/theory_stdout.log
tail -f results/muon_campaign/logs/theory_stderr.log

# Machine-readable state file
cat results/muon_campaign/campaign_status.json
```

---

## 6. Scheduling Heartbeats

The heartbeat script does not loop internally — OpenClaw calls it and acts on the exit code. The three most common scheduling approaches are:

### cron

```cron
*/30 * * * * cd /path/to/consortium && conda run -n researchlab \
  python scripts/campaign_heartbeat.py --campaign campaign.yaml >> \
  results/muon_campaign/logs/heartbeat.log 2>&1
```

Exit codes are not directly visible in cron. Check `heartbeat.log` and `campaign_status.json` for state.

### Simple shell polling loop

Useful during development or when OpenClaw manages the scheduling externally:

```bash
while true; do
  python scripts/campaign_heartbeat.py --campaign campaign.yaml
  CODE=$?
  if [ $CODE -eq 0 ]; then
    echo "Campaign complete."
    break
  elif [ $CODE -eq 2 ]; then
    echo "Campaign failed. Human attention required."
    break
  fi
  sleep 1800  # 30 minutes
done
```

### SLURM / HPC

Submit `scripts/launch_multiagent_slurm.sh` as a template for each stage, or use the campaign manager from a login node with a cron entry pointing at the shared filesystem.

### OpenClaw native scheduling

OpenClaw should call the heartbeat script on its configured interval and act on the exit code:

| Exit code | Meaning | Recommended action |
|---|---|---|
| `0` | Campaign fully complete | Stop scheduling; notify operator |
| `1` | In progress or waiting | Do nothing; tick again next interval |
| `2` | Stage failed | Pause scheduling; alert operator |
| `3` | New stage just launched | Tick again sooner (e.g. in 5 minutes) to confirm launch |

---

## 7. Understanding Campaign Outputs

### Directory layout

```
results/muon_campaign/           ← campaign_dir
  campaign_status.json           ← machine-readable stage state
  CAMPAIGN_COMPLETE.md           ← written when all stages finish
  memory/
    theory_summary.md            ← distilled after theory completes
    experiments_summary.md       ← distilled after experiments completes
  logs/
    theory_stdout.log
    theory_stderr.log
    experiments_stdout.log
    experiments_stderr.log
    heartbeat.log                ← if you redirect heartbeat output
  pids/
    theory.pid
    experiments.pid
    paper.pid
  task_prompts/
    theory_task.txt              ← enriched prompt actually sent to the agent
    experiments_task.txt
    paper_task.txt

results/muon_campaign/theory/    ← stage workspace (fresh stage)
  math_workspace/
  final_paper.tex                ← if produced
  run_invocation_usage.json
  cli_budget_state.json
  ...

results/muon_campaign/experiments/  ← stage workspace (resumed from theory)
  math_workspace/                   ← inherited from theory
  experiment_results.json
  ...
```

### `campaign_status.json` schema

```json
{
  "campaign_name": "Muon Implicit Regularization",
  "spec_file": "/abs/path/to/campaign.yaml",
  "stages": {
    "theory": {
      "status": "completed",
      "workspace": "/abs/path/to/results/muon_campaign/theory",
      "pid": null,
      "started_at": "2026-02-28T09:23:07+00:00",
      "completed_at": "2026-02-28T11:47:22+00:00",
      "missing_artifacts": [],
      "fail_reason": null
    },
    "experiments": {
      "status": "in_progress",
      "workspace": "/abs/path/to/results/muon_campaign/theory",
      "pid": 48291,
      "started_at": "2026-02-28T11:47:45+00:00",
      "completed_at": null,
      "missing_artifacts": [],
      "fail_reason": null
    },
    "paper": {
      "status": "pending",
      "workspace": null,
      "pid": null,
      ...
    }
  }
}
```

### Memory summaries

After each stage completes, `distill_stage_memory()` writes `memory/<stage_id>_summary.md`. It includes:

- Excerpts (up to 4000 chars) of each required artifact
- Short excerpts from each file in `memory_dirs` (up to 1500 chars, up to 20 files)
- Budget total from `cli_budget_state.json`
- Invocation counts from `run_invocation_usage.json`
- Pipeline status from `STATUS.txt`

This summary is what gets prepended to the next stage's task prompt. Inspect it to verify that the downstream agent receives the information it needs:

```bash
cat results/muon_campaign/memory/theory_summary.md
```

### Enriched task prompts

The exact prompt sent to each stage's agent is saved for auditability:

```bash
cat results/muon_campaign/task_prompts/experiments_task.txt
```

This is the `context_from` memory blocks + the original task file content, concatenated.

---

## 8. Live Steering via HTTP

While any stage is running, `launch_multiagent.py` starts an HTTP REST server at `callback_port + 1` (default: port 5002). OpenClaw can use this to pause and redirect a running stage without killing it.

### Endpoints

| Method | Path | Body | Effect |
|---|---|---|---|
| `POST` | `/interrupt` | none | Enqueues "interrupt" — pauses the agent at the next tool boundary |
| `POST` | `/instruction` | `{"text": "...", "type": "m"\|"n"}` | Injects a steering instruction after a pause |
| `GET` | `/status` | none | Returns `{"paused": bool, "queue_depth": int}` |

### Typical flow

```bash
# 1. Check the agent is running and not already paused
curl -s http://127.0.0.1:5002/status
# {"paused": false, "queue_depth": 0}

# 2. Pause it
curl -s -X POST http://127.0.0.1:5002/interrupt
# {"ok": true}

# 3. Inject a modification instruction (type "m" = modify current task)
curl -s -X POST http://127.0.0.1:5002/instruction \
     -H "Content-Type: application/json" \
     -d '{"text": "Focus only on the linear case. Skip the nonlinear extension.", "type": "m"}'
# {"ok": true}

# 4. Confirm the queue drained (agent has received the instruction)
curl -s http://127.0.0.1:5002/status
# {"paused": false, "queue_depth": 0}
```

### `type` values

| Value | Meaning |
|---|---|
| `"m"` | Modify the current task — the agent continues from where it paused with the new instruction layered on top |
| `"n"` | New task — replaces the current task entirely |

### Port when using non-default callback port

If you start the pipeline with `--callback_port 5010`, the HTTP server listens on `5011`. Always HTTP port = TCP port + 1.

### Safety note

Check `/status` before sending `/instruction`. If `queue_depth` is already non-zero, a prior instruction has not yet been consumed. Wait for it to drain before injecting another.

---

## 9. Notifications Setup

### Slack

1. In your Slack workspace, create an Incoming Webhook app and copy the URL.
2. Add to `.env`:
   ```
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../...
   ```
3. In `campaign.yaml`:
   ```yaml
   notification:
     slack_webhook: "${SLACK_WEBHOOK_URL}"
     on_stage_complete: true
     on_failure: true
     on_heartbeat: false
   ```

You will receive a message for each stage launch, each stage completion, each failure, and (if `on_heartbeat: true`) each heartbeat tick while a stage is running.

### Telegram

1. Create a bot via `@BotFather` and note the token.
2. Start a conversation with your bot or add it to a group, then get the `chat_id` via the Telegram API.
3. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456789:AAF...
   TELEGRAM_CHAT_ID=987654321
   ```
4. In `campaign.yaml`:
   ```yaml
   notification:
     telegram_bot_token: "${TELEGRAM_BOT_TOKEN}"
     telegram_chat_id: "${TELEGRAM_CHAT_ID}"
   ```

Both Slack and Telegram can be active simultaneously. Notification failures are silently swallowed — they will never crash the campaign.

### `on_heartbeat: true`

Useful for long-running stages when you want progress pings. Every heartbeat tick while a stage is running sends a message like:

```
[consortium] theory: in_progress (PID 48123) | experiments: pending | paper: pending
```

For quiet operation (only alerts on events), keep `on_heartbeat: false`.

---

## 10. Failure Recovery

When a stage fails (exit code 2 from the heartbeat), OpenClaw should pause scheduling and alert the operator. The failure is recorded in `campaign_status.json`:

```json
"theory": {
  "status": "failed",
  "fail_reason": "Process ended but required artifacts missing: ['math_workspace/claim_graph.json']",
  "missing_artifacts": ["math_workspace/claim_graph.json"]
}
```

### Diagnosis steps

```bash
# 1. Check what is missing
python scripts/campaign_heartbeat.py --campaign campaign.yaml --status

# 2. Read the stage's stderr log for the root cause
tail -100 results/muon_campaign/logs/theory_stderr.log

# 3. Inspect what did get produced
ls results/muon_campaign/theory/math_workspace/

# 4. Check invocation and budget state
cat results/muon_campaign/theory/run_invocation_usage.json
cat results/muon_campaign/theory/cli_budget_state.json
```

### Option A: Fix and retry from scratch

Edit the campaign status file to reset the failed stage to `pending`:

```bash
# Open and manually set "status": "pending" for the failed stage
# Also clear workspace, pid, fail_reason, missing_artifacts
$EDITOR results/muon_campaign/campaign_status.json
```

Then tick the heartbeat again. The stage will re-launch into a fresh workspace.

### Option B: Force-advance past a failed stage

If the stage produced enough artifacts to proceed (even without meeting the full `success_artifacts` list), use `--force-advance`:

```bash
python scripts/campaign_heartbeat.py \
  --campaign campaign.yaml \
  --force-advance
```

This marks the in-progress (or dead) stage as completed based on whatever artifacts are present, then immediately advances to the next pending stage.

### Option C: Resume the stage workspace manually

If the stage made significant progress, resume it with a refined task:

```bash
python launch_multiagent.py \
  --resume results/muon_campaign/theory \
  --task "Continue from existing artifacts. The claim_graph.json is missing — produce it now from the proofs already in math_workspace/proofs/."
```

Once the required artifacts exist, tick the heartbeat — it will detect completion and advance normally.

### Budget exhaustion

If `budget.lock` exists in a stage workspace, the stage stopped due to the invocation or wall-clock budget cap. Raise the limits in `.llm_config.yaml`, remove the lock file, and resume.

---

## 11. Adapting to Your Own Research Topic

### Checklist

1. **Copy and rename** `campaign.yaml` — e.g., `campaign_attention_heads.yaml`.
2. **Create task files** in `automation_tasks/` — one `.txt` per stage. See [section 4](#4-writing-task-files).
3. **Set `success_artifacts`** for each stage. Choose artifacts that are unambiguous signals that the stage succeeded (a specific JSON file, a `.pdf`, etc.). Avoid checking directories unless you know they will always be populated.
4. **Set `memory_dirs`** on stages whose outputs need to flow into downstream task prompts. For theory stages this is typically `math_workspace/`. For experiment stages, list the directory containing results files.
5. **Wire `depends_on` and `context_from`** correctly:
   - Use `depends_on` for all upstream stages.
   - Use `context_from` only for the stage(s) whose workspace or memory the new stage needs directly.
6. **Set `workspace_root`** to a new directory so runs do not collide.
7. **Run `--init` and `--status`** to validate the YAML before scheduling.

### Adding a fourth stage

Add a new entry at the bottom of `stages:`. It can depend on any prior stages and `context_from` any combination:

```yaml
  - id: rebuttal
    task_file: automation_tasks/rebuttal_task.txt
    depends_on:
      - paper
    context_from:
      - paper
    args: []
    success_artifacts:
      required:
        - rebuttal_response.md
```

### Parallel stages

Two stages with no shared `depends_on` relationship can run concurrently. The heartbeat script processes one in-progress stage per tick — it will not launch a second stage until the first completes. True parallelism requires running two campaigns with separate `campaign.yaml` files and merging their outputs manually.

### Budget guidance

All LLM costs are **included in your CLI tool subscription** (Claude Max, ChatGPT Pro, Gemini Advanced, etc.). Budget tracking in consortium monitors invocation counts and wall-clock time rather than dollar costs.

| Stage type | Recommended invocation limit |
|---|---|
| Theory only (`full_research` + math agents) | 200–500 invocations |
| Experiments only (`full_research`) | 100–300 invocations |
| Paper synthesis (`full_research` + `--require-pdf`) | 150–400 invocations |
| Full three-stage campaign | 500–1000 invocations |
| Full campaign with `--enable-counsel` | 2000+ invocations |

Set invocation and wall-clock limits in `.llm_config.yaml` before starting a campaign. Each stage has its own budget ledger in its workspace.

---

## 12. Reference

### `campaign.yaml` field reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | no | directory name | Campaign label used in notifications and status output |
| `workspace_root` | string | no | `results/campaign` | Parent directory for all stage workspaces |
| `heartbeat_interval_minutes` | int | no | `30` | Informational; OpenClaw uses this to configure its scheduler |
| `notification.slack_webhook` | string | no | null | Slack incoming webhook URL or `${ENV_VAR}` reference |
| `notification.telegram_bot_token` | string | no | null | Telegram bot token or `${ENV_VAR}` reference |
| `notification.telegram_chat_id` | string | no | null | Telegram chat ID or `${ENV_VAR}` reference |
| `notification.on_stage_complete` | bool | no | `true` | Send notification on stage completion |
| `notification.on_failure` | bool | no | `true` | Send notification on stage failure |
| `notification.on_heartbeat` | bool | no | `false` | Send notification on every heartbeat tick |
| `stages[].id` | string | yes | — | Unique stage identifier |
| `stages[].task_file` | string | yes | — | Path to task prompt `.txt` file (relative to repo root) |
| `stages[].args` | list[string] | no | `[]` | Extra CLI args for `launch_multiagent.py` |
| `stages[].depends_on` | string or list[string] | no | `[]` | Stage IDs that must be `completed` before this stage can launch |
| `stages[].context_from` | string or list[string] | no | `[]` | Stage IDs to resume workspace from and inject memory summaries for |
| `stages[].memory_dirs` | list[string] | no | `[]` | Workspace subdirectories to include in memory distillation |
| `stages[].success_artifacts.required` | list[string] | no | `[]` | Paths (relative to workspace) that must exist for stage to be marked complete |
| `stages[].success_artifacts.optional` | list[string] | no | `[]` | Paths checked but not blocking completion |

### Heartbeat exit codes

| Code | Meaning | OpenClaw action |
|---|---|---|
| `0` | Campaign fully complete | Stop scheduling |
| `1` | In progress or no action taken | Tick again at next interval |
| `2` | A stage has failed | Pause, alert operator |
| `3` | A new stage was just launched | Optionally tick sooner to confirm |

### HTTP REST API reference

Base URL: `http://<callback_host>:<callback_port + 1>` (default: `http://127.0.0.1:5002`)

| Method | Path | Request body | Response | Effect |
|---|---|---|---|---|
| `POST` | `/interrupt` | none | `{"ok": true}` | Enqueues "interrupt"; sets paused=true |
| `POST` | `/instruction` | `{"text": string, "type": "m"\|"n"}` | `{"ok": true}` or `{"error": "..."}` | Enqueues instruction lines + double-enter + type choice |
| `GET` | `/status` | none | `{"paused": bool, "queue_depth": int}` | Returns current pause state |

### Key file paths cheat sheet

| Path | Description |
|---|---|
| `campaign.yaml` | Campaign definition (repo root) |
| `<campaign_dir>/campaign_status.json` | Live stage state machine |
| `<campaign_dir>/CAMPAIGN_COMPLETE.md` | Written when all stages finish |
| `<campaign_dir>/memory/<stage_id>_summary.md` | Distilled memory from a completed stage |
| `<campaign_dir>/task_prompts/<stage_id>_task.txt` | Actual enriched prompt sent to each stage |
| `<campaign_dir>/logs/<stage_id>_stdout.log` | Stage stdout |
| `<campaign_dir>/logs/<stage_id>_stderr.log` | Stage stderr — first place to look on failure |
| `<campaign_dir>/pids/<stage_id>.pid` | PID of the running stage subprocess |
| `<workspace>/cli_budget_state.json` | Invocation budget for this stage |
| `<workspace>/budget.lock` | Present if budget cap was hit |
| `<workspace>/run_invocation_usage.json` | Invocation totals for this stage |
| `<workspace>/checkpoints.db` | LangGraph SQLite checkpoint — enables `--resume` |
