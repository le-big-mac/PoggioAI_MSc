# Consortium on SLURM/HPC Clusters — Setup Guide

## Prerequisites

- Access to a SLURM cluster
- Conda installed (Miniconda or Miniforge)
- At least one CLI agent tool installed (`claude`, `codex`, or `gemini`)

## Quick Start

```bash
# 1. Clone the repo and enter it
git clone <repo-url> OpenPI && cd OpenPI

# 2. Bootstrap the environment (creates conda env + installs deps)
./scripts/bootstrap.sh consortium full

# 3. Verify CLI tools are available
which claude || which codex || which gemini
# No API keys needed — CLI tools authenticate via their own subscriptions
# (Claude Max, ChatGPT Pro, Gemini Advanced, etc.)

# 4. Configure cluster paths (if using SLURM)
# Edit engaging_config.yaml with your cluster-specific paths:
#   - conda_init_script, conda_env_prefix, repo_root, slurm_output_dir
# Or set env vars: CONDA_INIT_SCRIPT, CONDA_ENV_PREFIX, REPO_ROOT, SLURM_OUTPUT_DIR

# 5. Test configuration
conda activate consortium
python launch_multiagent.py --dry-run

# 6. Submit the orchestrator to SLURM
RESEARCH_TASK="Your research prompt here..." ./scripts/submit_orchestrator.sh --no-counsel
```

## Architecture: Two-Tier SLURM Model

Consortium uses a **two-tier execution model** on HPC clusters:

### Tier 1: Orchestrator (CPU)
- Runs on a CPU partition (e.g., 12hr limit)
- Invokes CLI agent subprocesses (claude, codex, gemini) for LLM calls
- Coordinates 23+ specialist agents via LangGraph
- Does NOT need GPU

### Tier 2: Experiment Jobs (GPU)
- Submitted by the orchestrator via `sbatch` when experiments need GPU
- Partition configured in `engaging_config.yaml`
- Runs AI-Scientist-v2 experiment execution

Set `CONSORTIUM_SLURM_ENABLED=1` to enable automatic GPU job submission.

## Configuration

### engaging_config.yaml
All cluster-specific settings are centralized here:
- Partition names (GPU and CPU)
- Conda paths and module names
- Resource limits (CPUs, memory, time)

**Important**: Set these paths for your cluster either via env vars or by editing the file directly:
- `CONDA_INIT_SCRIPT` — path to your conda `conda.sh` init script
- `CONDA_ENV_PREFIX` — path to your conda environment
- `REPO_ROOT` — path to the OpenPI clone
- `SLURM_OUTPUT_DIR` — where to write SLURM logs

### .llm_config.yaml
LLM model selection, budget limits, counsel mode settings.

### .env
Optional service keys (search tools, Tinker, notifications). CLI agent tools do not require API keys.

## Running a Campaign (Multi-Stage)

```bash
# Initialize campaign
python scripts/campaign_heartbeat.py --campaign campaign.yaml --init

# Run heartbeat (advances stages)
python scripts/campaign_heartbeat.py --campaign campaign.yaml

# Or submit heartbeat as a SLURM job that advances stages via sbatch
```

Campaign stages can be launched as SLURM jobs using `launch_stage_slurm()` from the campaign runner.

## Monitoring

```bash
# Check SLURM jobs
squeue -u $USER

# Check job output
cat slurm_outputs/orch_<job_id>.out

# List past runs
python launch_multiagent.py --list-runs

# Check experiment GPU job logs
cat results/consortium_*/experiment_runs/*/slurm_logs/exp_*.out
```

## Troubleshooting

### "conda not found"
Set the `CONDA_INIT_SCRIPT` env var or edit `engaging_config.yaml`:
```bash
export CONDA_INIT_SCRIPT=/path/to/miniforge3/etc/profile.d/conda.sh
source "$CONDA_INIT_SCRIPT"
```

### "module not found"
Load the appropriate modules for your cluster:
```bash
module load miniforge    # or your cluster's conda module
module load cuda         # for GPU experiments
```

### GPU experiment job fails
Check the SLURM logs in the experiment run directory:
```bash
cat results/consortium_*/experiment_runs/*/slurm_logs/exp_*.out
cat results/consortium_*/experiment_runs/*/slurm_logs/exp_*.err
```

### CLI tool not found from compute node
The orchestrator needs CLI agent tools (claude/codex/gemini) on PATH. If compute nodes have a different environment, ensure the tools are installed or run the orchestrator on a login node instead:
```bash
conda activate <your-env>
nohup python launch_multiagent.py --task "..." --no-counsel &
```

### LaTeX/PDF compilation fails
```bash
# Install TeX toolchain in conda env
./scripts/bootstrap.sh <your-env> latex
# Or set the path to your system TeX installation
export CONSORTIUM_PDFLATEX_PATH=/path/to/pdflatex
```
