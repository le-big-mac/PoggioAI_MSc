# consortium Dockerfile — CLI agent mode
# Agents run as local CLI tool subprocesses (claude, codex, gemini).
# Auth handled by mounting volumes (~/.claude, ~/.codex, ~/.gemini)
# set up manually before first run.
#
# Build:
#   docker build -t consortium .
#   docker build --target minimal -t consortium:minimal .
#
# First-time auth (interactive, once per tool):
#   docker run -it -v claude-auth:/root/.claude consortium claude setup-token
#   docker run -it -v codex-auth:/root/.codex consortium codex auth login
#   docker run -it -v gemini-auth:/root/.gemini consortium gemini auth login
#
# Run:
#   docker compose run consortium --task "Your research task"

# ── Stage 1: system dependencies ──────────────────────────────────────────────
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    # LaTeX toolchain
    texlive-latex-base \
    texlive-latex-extra \
    texlive-bibtex-extra \
    bibtex2html \
    latexmk \
    # Git (for git-commit metadata + publish script)
    git \
    # Build tools for some Python packages
    build-essential \
    # curl for Node.js install
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 (for claude, codex, gemini CLIs)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install CLI agent tools
RUN npm install -g \
    @anthropic-ai/claude-code \
    @openai/codex \
    @google/gemini-cli \
    && npm cache clean --force

WORKDIR /app

# ── Stage 2: minimal install (no web crawl) ──────────────────────────────────
FROM base AS minimal

COPY pyproject.toml .
COPY requirements-minimal.txt .
COPY requirements-docs.txt .
COPY consortium/ consortium/
COPY launch_multiagent.py .
COPY config/ config/
COPY .env.example .

RUN pip install --no-cache-dir -e ".[docs]"

ENV CONSORTIUM_LOG_TO_FILES=0
ENV CONSORTIUM_DOCKER=1
ENTRYPOINT ["python", "launch_multiagent.py", "--no-steering"]

# ── Stage 3: full install (all optional extras) ──────────────────────────────
FROM base AS full

COPY pyproject.toml .
COPY requirements-minimal.txt .
COPY requirements-docs.txt .
COPY requirements-web.txt .
COPY requirements-observability.txt .
COPY consortium/ consortium/
COPY launch_multiagent.py .
COPY scripts/ scripts/
COPY config/ config/
COPY .env.example .
COPY automation_tasks/ automation_tasks/
COPY examples/ examples/
COPY campaigns/ campaigns/

# Install all extras
RUN pip install --no-cache-dir -e ".[docs,web,observability]"

# Install Playwright browsers
RUN python -m playwright install chromium --with-deps 2>/dev/null || true

ENV CONSORTIUM_LOG_TO_FILES=0
ENV CONSORTIUM_DOCKER=1
ENTRYPOINT ["python", "launch_multiagent.py"]

# ── Stage 4: idea-receiver (GitHub Issues watcher) ──────────────────────────
FROM full AS idea-receiver

ENTRYPOINT []
CMD ["python", "scripts/idea_watcher.py", "--repo", "le-big-mac/le-big-mac.github.io", "--interval", "30"]

# Default target is full
FROM full
