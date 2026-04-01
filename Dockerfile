# consortium Dockerfile
# Multi-stage build: installs system deps, then Python deps, then the package.
#
# Build targets:
#   docker build -t consortium .                              # full (CPU)
#   docker build --target minimal -t consortium:minimal .     # minimal (CPU)
#   docker build --target gpu -t consortium:gpu .             # full + GPU (CUDA)
#   docker build --target idea-receiver -t consortium:idea .  # webhook + watcher + GPU
#
# Run (GPU):
#   docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=3 --env-file .env \
#     -v $(pwd)/results:/app/results consortium:gpu \
#     python launch_multiagent.py --task "Your research task"

# ── Stage 1: system dependencies (CPU) ───────────────────────────────────────
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    # LaTeX toolchain
    texlive-latex-base \
    texlive-latex-extra \
    texlive-bibtex-extra \
    bibtex2html \
    latexmk \
    # Git (for git-commit metadata)
    git \
    # Build tools for some Python packages
    build-essential \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 1b: system dependencies (GPU / CUDA) ──────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS base-gpu

# Install Python 3.11 + system deps on the CUDA base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    # LaTeX toolchain
    texlive-latex-base \
    texlive-latex-extra \
    texlive-bibtex-extra \
    bibtex2html \
    latexmk \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# ── Stage 2: minimal install (no web crawl, no experiment tool) ──────────────
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
ENTRYPOINT ["python", "launch_multiagent.py", "--no-steering"]

# ── Stage 3: full install (CPU, all optional extras) ─────────────────────────
FROM base AS full

COPY pyproject.toml .
COPY requirements-minimal.txt .
COPY requirements-docs.txt .
COPY requirements-web.txt .
COPY requirements-observability.txt .
COPY requirements-experiment.txt .
COPY consortium/ consortium/
COPY external_tools/ external_tools/
COPY launch_multiagent.py .
COPY scripts/ scripts/
COPY config/ config/
COPY .env.example .
COPY automation_tasks/ automation_tasks/
COPY examples/ examples/

# Install all extras
RUN pip install --no-cache-dir -e ".[docs,web,observability]"

# Install Playwright browsers
RUN python -m playwright install chromium --with-deps 2>/dev/null || true

ENV CONSORTIUM_LOG_TO_FILES=0
EXPOSE 5003
ENTRYPOINT ["python", "launch_multiagent.py"]

# ── Stage 4: GPU install (CUDA base + all extras + torch GPU) ────────────────
FROM base-gpu AS gpu

COPY pyproject.toml .
COPY requirements-minimal.txt .
COPY requirements-docs.txt .
COPY requirements-web.txt .
COPY requirements-observability.txt .
COPY requirements-experiment.txt .
COPY consortium/ consortium/
COPY external_tools/ external_tools/
COPY launch_multiagent.py .
COPY scripts/ scripts/
COPY config/ config/
COPY .env.example .
COPY automation_tasks/ automation_tasks/
COPY examples/ examples/

# Install core + docs + web + observability first
RUN pip install --no-cache-dir --break-system-packages -e ".[docs,web,observability]"

# Install experiment deps (torch GPU from PyTorch index, rest from PyPI)
RUN pip install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir --break-system-packages \
    -r requirements-experiment.txt

# Install Playwright browsers
RUN python -m playwright install chromium --with-deps 2>/dev/null || true

# Ensure agents find the right Python
ENV CONSORTIUM_LOG_TO_FILES=0
ENV CONSORTIUM_PYTHON=/usr/bin/python
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 5003
ENTRYPOINT ["python", "launch_multiagent.py"]

# ── Stage 5: idea-receiver (webhook + watcher + GPU) ────────────────────────
FROM gpu AS idea-receiver

EXPOSE 5003
ENTRYPOINT []
CMD ["sh", "-c", "python -m consortium.interaction.webhook_server --port 5003 & python scripts/idea_watcher.py --interval 30"]

# Default target is full (CPU)
FROM full
