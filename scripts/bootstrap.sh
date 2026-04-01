#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-consortium}"
PROFILE_RAW="${2:-full}"
PROFILE="${PROFILE_RAW// /}"

if [[ -z "$PROFILE" ]]; then
  PROFILE="full"
fi

IFS=',' read -r -a PROFILE_ITEMS <<< "$PROFILE"

has_capability() {
  local cap="$1"
  local item=""
  for item in "${PROFILE_ITEMS[@]}"; do
    if [[ "$item" == "full" || "$item" == "$cap" ]]; then
      return 0
    fi
  done
  return 1
}

if ! has_capability minimal && ! has_capability docs && ! has_capability web && ! has_capability experiment && ! has_capability latex; then
  echo "Error: profile '$PROFILE_RAW' is invalid."
  echo "Use one of: minimal, docs, web, experiment, latex, full"
  echo "You can combine capabilities with commas (example: minimal,web)."
  exit 1
fi

# --- Engaging cluster: auto-source conda if not on PATH ---
if ! command -v conda >/dev/null 2>&1; then
  # Try to read conda init path from engaging_config.yaml if present
  ENGAGING_CONDA_INIT=""
  _ENGAGING_CFG="$REPO_ROOT/engaging_config.yaml"
  if [[ -f "$_ENGAGING_CFG" ]]; then
    ENGAGING_CONDA_INIT=$(grep 'conda_init_script:' "$_ENGAGING_CFG" 2>/dev/null | head -1 | sed 's/.*conda_init_script:\s*//' | sed 's/\s*#.*//' | tr -d '[:space:]' | sed 's/\${[^}]*:-\(.*\)}/\1/')
  fi
  # Override with env var if set
  ENGAGING_CONDA_INIT="${CONDA_INIT_SCRIPT:-$ENGAGING_CONDA_INIT}"
  if [[ -n "$ENGAGING_CONDA_INIT" && -f "$ENGAGING_CONDA_INIT" ]]; then
    echo "Sourcing conda init: $ENGAGING_CONDA_INIT"
    # shellcheck disable=SC1090
    source "$ENGAGING_CONDA_INIT"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- Detect if we should use --prefix (Engaging/HPC) or named env ---
# If CONSORTIUM_ENV_PREFIX is set, use prefix-based env (avoids home quota issues).
# Otherwise, check if a prefix path is defined in engaging_config.yaml.
ENGAGING_CONFIG="$REPO_ROOT/engaging_config.yaml"
USE_PREFIX=""
if [[ -n "${CONSORTIUM_ENV_PREFIX:-}" ]]; then
  USE_PREFIX="$CONSORTIUM_ENV_PREFIX"
elif [[ -f "$ENGAGING_CONFIG" ]]; then
  # Simple yaml extraction — no dependencies required
  _prefix=$(grep 'conda_env_prefix:' "$ENGAGING_CONFIG" 2>/dev/null | head -1 | sed 's/.*conda_env_prefix:\s*//' | tr -d '[:space:]')
  if [[ -n "$_prefix" ]]; then
    USE_PREFIX="$_prefix"
  fi
fi

if [[ -n "$USE_PREFIX" ]]; then
  # Prefix-based conda environment
  if [[ -d "$USE_PREFIX" ]]; then
    echo "Using existing prefix environment: $USE_PREFIX"
  else
    echo "Creating prefix-based conda environment at: $USE_PREFIX"
    conda create --prefix "$USE_PREFIX" python=3.11 -y
  fi
  conda activate "$USE_PREFIX"
else
  # Standard named environment
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Using existing conda environment: $ENV_NAME"
  else
    echo "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.11 -y
  fi
  conda activate "$ENV_NAME"
fi

python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements-minimal.txt"

if has_capability observability || has_capability full; then
  python -m pip install -r "$REPO_ROOT/requirements-observability.txt"
fi

if has_capability docs; then
  python -m pip install -r "$REPO_ROOT/requirements-docs.txt"
fi

if has_capability web; then
  python -m pip install -r "$REPO_ROOT/requirements-web.txt"
  # Required by crawl4ai at runtime.
  python -m playwright install chromium
fi

if has_capability experiment; then
  # Load CUDA modules on HPC clusters (best-effort, not fatal if unavailable)
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.4.0 2>/dev/null || true
    module load cudnn/9.8.0.87-cuda12 2>/dev/null || true
  fi
  python -m pip install -r "$REPO_ROOT/requirements-experiment.txt"
fi

if has_capability latex; then
  # Install tectonic (single-binary LaTeX compiler, no texlive needed)
  if ! command -v tectonic >/dev/null 2>&1; then
    echo "Installing tectonic LaTeX compiler..."
    mkdir -p "$HOME/.local/bin"
    curl -L "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz" \
      | tar xz -C "$HOME/.local/bin/" tectonic
    chmod +x "$HOME/.local/bin/tectonic"
    export PATH="$HOME/.local/bin:$PATH"
    echo "tectonic installed: $(tectonic --version)"
  else
    echo "tectonic already installed: $(tectonic --version)"
  fi
fi

python -m pip check

PREFLIGHT_ARGS=()
if has_capability docs; then
  PREFLIGHT_ARGS+=("--with-docs")
fi
if has_capability web; then
  PREFLIGHT_ARGS+=("--with-web")
fi
if has_capability experiment; then
  PREFLIGHT_ARGS+=("--with-experiment")
fi
if has_capability latex; then
  PREFLIGHT_ARGS+=("--with-latex")
fi
python "$REPO_ROOT/scripts/preflight_check.py" "${PREFLIGHT_ARGS[@]}"

# Copy config templates if they don't exist yet
if [[ ! -f "$REPO_ROOT/.llm_config.yaml" && -f "$REPO_ROOT/.llm_config.yaml.example" ]]; then
  cp "$REPO_ROOT/.llm_config.yaml.example" "$REPO_ROOT/.llm_config.yaml"
  echo "Created .llm_config.yaml from .llm_config.yaml.example"
fi

echo "Setup complete."
echo "Installed profile: $PROFILE_RAW"
echo "Next steps:"
echo "  1. Copy and edit .env:  cp .env.example .env"
echo "  2. Add your API key(s) in .env"
echo "  3. Validate:  python launch_multiagent.py --task 'test' --dry-run"
