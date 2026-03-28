# Campaign Quick-Start Guide

How to go from a research idea to an autonomous campaign producing a conference-grade paper.

## Prerequisites

| Requirement | How to verify |
|------------|---------------|
| Python environment | `conda activate <your-env> && python -c "import consortium"` |
| CLI agent tool installed | At minimum `which claude` (or `which codex` / `which gemini`) |
| `pdflatex` on PATH | `which pdflatex` (install via `./scripts/bootstrap.sh <env> latex`) |
| **Optional**: Telegram bot | `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| **Optional**: OpenClaw | `which openclaw` — only needed for Telegram-based monitoring |
| **Optional**: Claude CLI | `which claude` — only needed for autonomous repair agent |

## Step 1: Write Your Research Proposal

Create a task file describing the research question. This is the seed for the entire campaign.

```bash
cat > automation_tasks/my_discovery_task.txt << 'EOF'
Investigate whether [your research question here].

Key aspects to explore:
1. ...
2. ...
3. ...

Target venue: NeurIPS 2026
EOF
```

See `examples/quickstart/task.txt` for a concrete example.

## Step 2: Create the Campaign YAML

Copy the template and customize:

```bash
cp campaign_template.yaml my_campaign.yaml
```

Edit `my_campaign.yaml` — the critical fields to change:

```yaml
name: "Your Campaign Name"
workspace_root: "results/your_campaign_01"      # MUST be unique per campaign

planning:
  enabled: true
  base_task_file: automation_tasks/my_discovery_task.txt   # your task from Step 1
  max_stages: 6
  max_parallel: 2
  human_review: true          # set false for fully autonomous (auto-approve plan)

stages: []                    # empty = dynamic planning generates stages automatically

notification:
  telegram_bot_token: "${TELEGRAM_BOT_TOKEN}"
  telegram_chat_id: "${TELEGRAM_CHAT_ID}"
  on_stage_complete: true
  on_failure: true
  on_heartbeat: true
```

### Campaign Independence Rules

- **Every campaign MUST have a unique `workspace_root`** — never reuse a directory from a prior campaign.
- **Never reference prior campaign results** in `context_from` or `--resume` flags.
- All stage artifacts, logs, and status files live under `workspace_root/`.

## Step 3: Initialize the Campaign

```bash
cd <your-clone-of-OpenPI>
conda activate <your-env>

python scripts/campaign_heartbeat.py --campaign my_campaign.yaml --init
```

This creates:
- `results/your_campaign_01/campaign_status.json` (tracks stage states)
- `results/your_campaign_01/campaign_status.lock` (concurrency lock)

## Step 4: Launch the First Stage

```bash
python scripts/campaign_cli.py --campaign my_campaign.yaml launch discovery_plan
```

The heartbeat will auto-advance subsequent stages as dependencies are satisfied.

---

## Running Campaigns Locally (No SLURM/OpenClaw)

If you don't have SLURM or OpenClaw, you can run campaigns manually:

```bash
# Initialize
python scripts/campaign_heartbeat.py --campaign my_campaign.yaml --init

# Launch first stage
python scripts/campaign_cli.py --campaign my_campaign.yaml launch discovery_plan

# Run heartbeat ticks manually (or set up a system cron)
python scripts/campaign_heartbeat.py --campaign my_campaign.yaml

# Check status between ticks
python scripts/campaign_cli.py --campaign my_campaign.yaml status
```

To automate on Linux/macOS without OpenClaw, add a crontab entry:
```bash
# Run heartbeat every 15 minutes
*/15 * * * * cd /path/to/OpenPI && /path/to/conda/envs/bin/python scripts/campaign_heartbeat.py --campaign my_campaign.yaml >> logs/heartbeat.log 2>&1
```

---

## HPC Only: OpenClaw Gateway + SLURM

> Skip this section if running locally.

### Step 5: Start the OpenClaw Gateway

If the gateway is not already running:

```bash
sbatch scripts/launch_openclaw_gateway.sh
```

The gateway:
- Runs on your configured SLURM partition (default: 12-hour wall time)
- Self-resubmits before wall time expires
- Hosts the OpenClaw agent on port 18789
- Only one instance needed (shared across campaigns)

### Step 6: Set Up Cron Monitoring

Via OpenClaw CLI or Telegram, register the heartbeat cron:

```
Campaign heartbeat: every 15 minutes
  python scripts/campaign_heartbeat.py --campaign my_campaign.yaml

Log monitor: every 5 minutes (lightweight liveness checks)
```

**Important**: OpenClaw cron has TWO timeouts that must both be set:
- `--timeout <ms>` (gateway-level)
- `--timeout-seconds <n>` (agent payload)

Recommended: 900s for heartbeat, 180s for log monitor.

---

## Step 7: Monitor Progress

### Via Telegram (if configured)
- **Every 15 min**: Heartbeat status (stages in progress, invocations used, artifacts found)
- **On stage complete**: Summary + artifacts
- **On failure**: Error diagnosis + repair attempt status

### Via CLI

```bash
# Full status
python scripts/campaign_cli.py --campaign my_campaign.yaml status

# Stage logs (last 100 lines)
python scripts/campaign_cli.py --campaign my_campaign.yaml stage-logs <stage_id> --tail 100

# Budget summary
python scripts/campaign_cli.py --campaign my_campaign.yaml budget

# Check CLI tool availability before launch
python scripts/campaign_cli.py --campaign my_campaign.yaml check-credits

# Approve the dynamic plan (if human_review: true)
python scripts/campaign_cli.py --campaign my_campaign.yaml approve-plan

# List stages ready to launch
python scripts/campaign_cli.py --campaign my_campaign.yaml launchable

# Override a stage status (emergency)
python scripts/campaign_cli.py --campaign my_campaign.yaml set-stage-status <stage_id> <status>
```

## Campaign Lifecycle

```
1. discovery_plan     — Literature review + research plan (auto-launched)
2. planning_counsel   — 4-model debate generates stage DAG (auto-launched)
   [HUMAN REVIEW]     — Approve/reject plan via CLI or Telegram
3. theory1/theory2    — Parallel theory tracks (auto-launched after approval)
4. experiment1        — Empirical validation (auto-launched when theory deps complete)
5. paper1             — Publication-quality paper (auto-launched when experiments done)
```

Exit codes from heartbeat:
- `0`: Campaign complete (all stages done)
- `1`: In progress (normal tick)
- `2`: Failed stage (human attention needed)
- `3`: New stage just launched

## Budget

All LLM costs are **included in your CLI tool subscription** (Claude Max, ChatGPT Pro, Gemini Advanced, etc.). Budget tracking monitors invocation counts and wall-clock time. Configure limits in `.llm_config.yaml`.

Monitor usage:
```bash
python scripts/campaign_cli.py --campaign my_campaign.yaml budget
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Heartbeat shows 0 invocations | Normal during early ticks — budget updates on stage completion |
| CLI tool rate limit hit | Add fallback CLI backends in `.llm_config.yaml` |
| Stage died silently | Heartbeat detects + triggers repair agent (2 attempts max) |
| Plan not approved | Run `campaign_cli.py approve-plan` or set `human_review: false` |
| Hollow experiment artifacts | Agents lack GPU access — configure SLURM in `engaging_config.yaml` |
| Telegram not receiving updates | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` |
| Gateway Telegram 409 conflict | Only one gateway instance should run — check with `squeue -u $USER` |

## Files Reference

| File | Purpose |
|------|---------|
| `campaign_template.yaml` | Campaign template to copy and customize |
| `my_campaign.yaml` | Your campaign configuration |
| `engaging_config.yaml` | HPC cluster settings (optional, SLURM only) |
| `.llm_config.yaml` | Model configuration (main model, counsel models, budget) |
| `.env` | Optional service keys and notification credentials |
| `scripts/campaign_heartbeat.py` | Heartbeat orchestrator (called by cron) |
| `scripts/campaign_cli.py` | CLI for status, launch, repair, approve-plan |
| `scripts/launch_openclaw_gateway.sh` | SLURM launcher for OpenClaw gateway (HPC only) |
| `results/your_campaign_01/campaign_status.json` | Campaign state (auto-managed) |
