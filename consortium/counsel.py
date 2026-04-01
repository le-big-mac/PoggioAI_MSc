"""
Model counsel — multi-model debate and synthesis for each pipeline stage.

CLI-agent-only version: sandbox agents run as CLI tool subprocesses (Claude
Code, Codex, Gemini CLI) instead of LangChain ReAct agents. Debate and
synthesis use full CLI agent subprocess calls with tool access, so debate
critics can examine workspace files, search for information, and verify claims.

For each stage, spawns one CLI agent per counsel backend, each working in
its own sandbox copy of the workspace. Their outputs feed a text-based
debate across N rounds, then a synthesis call produces the final consensus.
Sandbox artifacts are merged back into the main workspace.

Usage (via graph.py/build_node):
    from .counsel import create_counsel_node
    node = create_counsel_node(system_prompt, tools, "ideation_agent",
                               workspace_dir, counsel_specs, max_debate_rounds=3)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

import logging

from .cli_completion import cli_completion  # kept for potential fallback use

logger = logging.getLogger(__name__)


def _model_to_backend(model_id: str) -> str:
    """Map a model ID to its CLI backend name."""
    if "claude" in model_id or "anthropic" in model_id:
        return "claude"
    if "gpt" in model_id or model_id.startswith(("o1-", "o3-", "o4-")):
        return "codex"
    if "gemini" in model_id:
        return "gemini"
    return "claude"


# Default counsel specs — one per CLI backend for diverse perspectives.
DEFAULT_COUNSEL_SPECS = [
    {"model": "claude-opus-4-6", "backend": "claude"},
    {"model": "gpt-5.4", "backend": "codex"},
    {"model": "gemini-3-pro-preview", "backend": "gemini"},
]

SYNTHESIS_BACKEND = "claude"
SYNTHESIS_MODEL = "claude-opus-4-6"

# Per-model timeout for sandbox and debate phases.
DEFAULT_MODEL_TIMEOUT_SECONDS: int = int(
    os.environ.get("COUNSEL_MODEL_TIMEOUT_SECONDS", "3600")
)


def set_counsel_timeout(seconds: int) -> None:
    """Set the global per-model counsel timeout."""
    global DEFAULT_MODEL_TIMEOUT_SECONDS
    DEFAULT_MODEL_TIMEOUT_SECONDS = seconds


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------

def _populate_sandbox(workspace_dir: str, sandbox_dir: str) -> None:
    """Copy current workspace into a fresh sandbox, skipping sandbox subtrees and DBs."""
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            shutil.copytree(
                workspace_dir,
                sandbox_dir,
                ignore=shutil.ignore_patterns(
                    "counsel_sandboxes", "_test_sandboxes", "*_sandboxes",
                    "*.db", "*.lock", "__pycache__",
                ),
            )
            return
        except (InterruptedError, OSError) as exc:
            if attempt == max_attempts - 1:
                print(f"[counsel] sandbox copy failed after {max_attempts} attempts: {exc}")
                raise
            print(
                f"[counsel] sandbox copy attempt {attempt + 1} failed ({exc}), "
                f"retrying in {1 << attempt}s..."
            )
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir, ignore_errors=True)
            time.sleep(1 << attempt)


def _merge_sandbox(sandbox_dir: str, workspace_dir: str) -> None:
    """Copy all files from sandbox into workspace (last sandbox wins on conflicts)."""
    for root, dirs, files in os.walk(sandbox_dir):
        dirs[:] = [d for d in dirs if d != "counsel_sandboxes"]
        for fname in files:
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, sandbox_dir)
            dst = os.path.join(workspace_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception as exc:
                print(f"[counsel] merge WARNING: failed to copy {rel}: {exc}")


def _run_sandbox_cli_agent(
    prompt: str,
    sandbox_dir: str,
    backend: str,
    model: Optional[str],
    timeout: int,
) -> str:
    """Run a full CLI agent in a sandbox directory."""
    if backend == "claude":
        cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "100"]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(["--allowedTools",
                     "Edit,Read,Write,WebFetch,WebSearch,Bash,Glob,Grep"])
    elif backend == "codex":
        cmd = ["codex", "--approval-mode", "full-auto", "--quiet"]
        if model:
            cmd.extend(["--model", model])
    elif backend == "gemini":
        cmd = ["gemini"]
        if model:
            cmd.extend(["--model", model])
    else:
        return f"[unknown backend {backend!r}]"

    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            cwd=sandbox_dir, timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    except FileNotFoundError:
        return f"['{backend}' CLI tool not found on PATH]"

    if result.returncode != 0:
        stderr = (result.stderr or "")[:500]
        return f"[failed (rc={result.returncode}): {stderr}]"

    return result.stdout.strip()


# ---------------------------------------------------------------------------
# CLI agent completion with tool access (for debate & synthesis)
# ---------------------------------------------------------------------------

def _cli_agent_completion(
    prompt: str,
    backend: str,
    model: Optional[str],
    workspace_dir: str,
    timeout: int = 600,
) -> str:
    """Run a CLI agent with full tool access for debate/synthesis phases.

    Unlike ``cli_completion()`` which runs with no tools (single prompt-in,
    text-out), this function gives the agent access to Read, Bash, Grep, Glob,
    etc. so it can examine workspace files, verify claims, and search for
    information during debate critiques and synthesis.

    Args:
        prompt: The debate/synthesis prompt.
        backend: CLI backend — "claude", "codex", or "gemini".
        model: Optional model override.
        workspace_dir: Working directory for the agent (gives file access).
        timeout: Max seconds for the subprocess.

    Returns:
        Response text. On error/timeout returns a descriptive error string.
    """
    t0 = time.time()

    if backend == "claude":
        cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "20"]
        if model:
            cmd.extend(["--model", model])
        cmd.extend([
            "--allowedTools",
            "Read,Write,WebFetch,WebSearch,Bash(cat*),Bash(ls*),Bash(grep*),Bash(find*),Bash(python*),Bash(curl*),Glob,Grep",
        ])
    elif backend == "codex":
        cmd = ["codex", "--approval-mode", "full-auto", "--quiet"]
        if model:
            cmd.extend(["--model", model])
    elif backend == "gemini":
        cmd = ["gemini"]
        if model:
            cmd.extend(["--model", model])
    else:
        return f"[_cli_agent_completion error: unknown backend {backend!r}]"

    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            cwd=workspace_dir, timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        msg = f"[_cli_agent_completion timed out after {elapsed:.0f}s — backend={backend}, model={model}]"
        logger.warning(msg)
        return msg
    except FileNotFoundError:
        msg = f"[_cli_agent_completion error: '{backend}' CLI tool not found on PATH]"
        logger.error(msg)
        return msg

    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr_snippet = (result.stderr or "")[:500]
        msg = f"[_cli_agent_completion error (rc={result.returncode}): {stderr_snippet}]"
        logger.warning(msg)
        return msg

    output = result.stdout.strip()

    # Record invocation for budget tracking
    try:
        from .cli_budget import get_global_cli_tracker
        tracker = get_global_cli_tracker()
        if tracker is not None:
            tracker.record_invocation(
                agent_name="counsel_debate",
                backend=backend,
                model=model or "default",
                duration_seconds=elapsed,
                prompt_chars=len(prompt),
                output_chars=len(output),
            )
    except Exception:
        pass  # Never break callers for tracking errors

    logger.debug(
        "[_cli_agent_completion] backend=%s model=%s elapsed=%.1fs output_len=%d",
        backend, model, elapsed, len(output),
    )
    return output


# ---------------------------------------------------------------------------
# Core counsel stage runner
# ---------------------------------------------------------------------------

def run_counsel_stage(
    task: str,
    system_prompt: str,
    workspace_dir: str,
    agent_name: str,
    max_debate_rounds: int = 3,
    counsel_specs: Optional[List[dict]] = None,
    model_timeout_seconds: int = 600,
) -> str:
    """
    Run multi-model counsel for one pipeline stage.

    1. Sandbox phase  — each CLI agent runs independently in its own workspace copy
                        (all N agents run in parallel via ThreadPoolExecutor).
    2. Debate phase   — models critique each other's solutions via full CLI agent
                        calls with tool access (Read, Bash, Grep, Glob);
                        all N critiques per round run in parallel.
    3. Synthesis      — full CLI agent call with tool access produces the final
                        consensus output.
    4. Promotion      — sandbox artifacts are merged back to the main workspace.

    Returns the consensus output string.
    """
    specs = counsel_specs or DEFAULT_COUNSEL_SPECS
    sandbox_base = os.path.join(workspace_dir, "counsel_sandboxes", agent_name)
    os.makedirs(sandbox_base, exist_ok=True)

    # Pre-build sandbox dir paths
    sandbox_dirs: List[str] = []
    for i, spec in enumerate(specs):
        label = spec.get("model", f"model_{i}")
        safe_label = label.replace("/", "_").replace(":", "_")
        sandbox_dirs.append(os.path.join(sandbox_base, f"model_{i}_{safe_label}"))

    # ------------------------------------------------------------------
    # 1. Sandbox phase — all CLI agents run in parallel
    # ------------------------------------------------------------------

    def _run_one_sandbox(idx: int) -> tuple[int, str]:
        spec = specs[idx]
        label = spec.get("model", f"model_{idx}")
        backend = spec.get("backend") or _model_to_backend(label)
        model = spec.get("model")
        sandbox_dir = sandbox_dirs[idx]

        _populate_sandbox(workspace_dir, sandbox_dir)

        prompt = f"{system_prompt}\n\n{task}"

        try:
            output = _run_sandbox_cli_agent(
                prompt, sandbox_dir, backend, model, model_timeout_seconds
            )
            # Fallback: if no text output, summarize workspace files
            if not output or output.startswith("["):
                new_files = []
                for root, dirs, files in os.walk(sandbox_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        if not any(x in fp for x in ["__pycache__", "token_usage", "budget_", "checkpoint"]):
                            new_files.append(os.path.relpath(fp, sandbox_dir))
                if new_files and (not output or output.startswith("[")):
                    output = (output or "") + f"\nAgent completed. Files in sandbox: {len(new_files)} files."
        except Exception as e:
            output = f"[{label} failed: {e}]"

        print(f"[counsel:{agent_name}] model_{idx} ({label}/{backend}) complete.")
        return idx, output

    sandbox_outputs: List[str] = [""] * len(specs)
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(_run_one_sandbox, i): i for i in range(len(specs))}
        try:
            for future in as_completed(futures, timeout=model_timeout_seconds + 60):
                try:
                    idx, output = future.result(timeout=model_timeout_seconds)
                    sandbox_outputs[idx] = output
                except TimeoutError:
                    for f, i in futures.items():
                        if f is future:
                            label = specs[i].get("model", f"model_{i}")
                            sandbox_outputs[i] = f"[{label} timed out after {model_timeout_seconds}s]"
                            print(f"[counsel:{agent_name}] model_{i} ({label}) TIMED OUT")
                            break
                except Exception as e:
                    for f, i in futures.items():
                        if f is future:
                            label = specs[i].get("model", f"model_{i}")
                            sandbox_outputs[i] = f"[{label} error: {e}]"
                            print(f"[counsel:{agent_name}] model_{i} ({label}) error: {e}")
                            break
        except TimeoutError:
            for f, i in futures.items():
                if not f.done():
                    label = specs[i].get("model", f"model_{i}")
                    sandbox_outputs[i] = f"[{label} timed out after {model_timeout_seconds}s]"
                    print(f"[counsel:{agent_name}] model_{i} ({label}) TIMED OUT (outer)")
                    f.cancel()

    # ------------------------------------------------------------------
    # 1b. Quorum check — skip debate if too few models succeeded
    # ------------------------------------------------------------------
    _MIN_QUORUM = 2
    valid_outputs = [
        i for i, out in enumerate(sandbox_outputs)
        if out and not out.startswith("[")
    ]
    for i, out in enumerate(sandbox_outputs):
        is_valid = i in valid_outputs
        print(f"[counsel:QUORUM] model_{i}: valid={is_valid} len={len(out)} first60={repr(out[:60])}")
    if len(valid_outputs) < _MIN_QUORUM and len(specs) >= _MIN_QUORUM:
        print(
            f"[counsel:{agent_name}] Only {len(valid_outputs)}/{len(specs)} models "
            f"produced valid output (quorum={_MIN_QUORUM}). Skipping debate."
        )
        max_debate_rounds = 0

    # ------------------------------------------------------------------
    # 2. Debate phase — all critiques per round run in parallel
    # ------------------------------------------------------------------
    formatted = "\n\n".join(
        f"=== Solution {i} ({specs[i].get('model', i)}) ===\n{out}"
        for i, out in enumerate(sandbox_outputs)
    )
    debate_history: List[str] = []

    for rnd in range(max_debate_rounds):
        base_prompt = (
            f"Task: {task}\n\n"
            f"Here are {len(sandbox_outputs)} independent solutions:\n\n{formatted}\n\n"
        )
        if debate_history:
            base_prompt += "Prior debate:\n" + "\n---\n".join(debate_history) + "\n\n"
        base_prompt += (
            "Identify: (1) strongest elements of each solution, "
            "(2) weaknesses or errors, (3) a synthesized approach capturing the best of all. "
            "Be specific and concise."
        )

        def _one_critique(i: int) -> tuple[int, str]:
            spec = specs[i]
            backend = spec.get("backend") or _model_to_backend(spec.get("model", ""))
            model = spec.get("model")
            try:
                critique = _cli_agent_completion(
                    base_prompt,
                    backend=backend,
                    model=model,
                    workspace_dir=workspace_dir,
                    timeout=model_timeout_seconds,
                )
            except Exception as e:
                critique = f"[{model} debate error: {e}]"
            return i, f"Model {i} ({model}):\n{critique}"

        critiques: List[str] = [""] * len(specs)
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {pool.submit(_one_critique, i): i for i in range(len(specs))}
            try:
                for future in as_completed(futures, timeout=model_timeout_seconds + 60):
                    try:
                        i, text = future.result(timeout=model_timeout_seconds)
                        critiques[i] = text
                    except TimeoutError:
                        for f, idx in futures.items():
                            if f is future:
                                label = specs[idx].get("model", f"model_{idx}")
                                critiques[idx] = f"Model {idx} ({label}):\n[debate timed out]"
                                break
                    except Exception as e:
                        for f, idx in futures.items():
                            if f is future:
                                critiques[idx] = f"Model {idx}:\n[debate error: {e}]"
                                break
            except TimeoutError:
                for f, idx in futures.items():
                    if not f.done():
                        label = specs[idx].get("model", f"model_{idx}")
                        critiques[idx] = f"Model {idx} ({label}):\n[debate timed out]"
                        f.cancel()

        debate_history.append(f"[Round {rnd + 1}]\n" + "\n\n".join(critiques))

        # Circuit breaker
        failed_count = sum(1 for c in critiques if "error:" in c or "timed out" in c)
        if failed_count > len(specs) / 2:
            print(
                f"[counsel:{agent_name}] CIRCUIT BREAKER: {failed_count}/{len(specs)} "
                f"debate models failed in round {rnd + 1}. Skipping remaining rounds."
            )
            break
        print(f"[counsel:{agent_name}] debate round {rnd + 1} complete.")

    # ------------------------------------------------------------------
    # 3. Pre-synthesis circuit breaker
    # ------------------------------------------------------------------
    sandbox_failures = sum(1 for o in sandbox_outputs if o.startswith("["))
    if sandbox_failures == len(sandbox_outputs):
        print(
            f"[counsel:{agent_name}] ALL {len(sandbox_outputs)} sandbox models failed. "
            f"Returning first sandbox error."
        )
        return sandbox_outputs[0]

    # ------------------------------------------------------------------
    # 4. Synthesis
    # ------------------------------------------------------------------
    synthesis_prompt = (
        f"Task: {task}\n\n"
        f"You are the synthesis model. Review {len(sandbox_outputs)} independent solutions "
        f"and {len(debate_history)} rounds of debate, then produce the final authoritative "
        f"output for this pipeline stage.\n\n"
        f"Solutions:\n{formatted}\n\n"
        f"Debate:\n" + "\n---\n".join(debate_history) + "\n\n"
        "Incorporate the strongest elements from all solutions. Be comprehensive and precise."
    )

    try:
        final_output = _cli_agent_completion(
            synthesis_prompt,
            backend=SYNTHESIS_BACKEND,
            model=SYNTHESIS_MODEL,
            workspace_dir=workspace_dir,
            timeout=model_timeout_seconds,
        )
        if not final_output:
            final_output = sandbox_outputs[0] if sandbox_outputs else ""
    except Exception as e:
        print(f"[counsel:{agent_name}] synthesis failed ({e}), using first sandbox output.")
        final_output = sandbox_outputs[0] if sandbox_outputs else ""

    # ------------------------------------------------------------------
    # 5. Artifact promotion (earlier sandboxes first, later sandboxes win)
    # ------------------------------------------------------------------
    for sandbox_dir in sandbox_dirs:
        _merge_sandbox(sandbox_dir, workspace_dir)

    print(f"[counsel:{agent_name}] counsel complete.")
    return final_output


# ---------------------------------------------------------------------------
# LangGraph node factory
# ---------------------------------------------------------------------------

def create_counsel_node(
    system_prompt: str,
    tools: Any,  # Accepted for signature compat but ignored in CLI mode
    agent_name: str,
    workspace_dir: str,
    counsel_specs: Optional[List[dict]] = None,
    max_debate_rounds: int = 3,
    model_specs: Optional[List[dict]] = None,
    model_timeout_seconds: Optional[int] = None,
) -> Any:
    """
    Return a LangGraph node callable that wraps run_counsel_stage.

    Drop-in replacement for create_specialist_agent when counsel mode is enabled.
    """
    _timeout = model_timeout_seconds if model_timeout_seconds is not None else DEFAULT_MODEL_TIMEOUT_SECONDS
    _specs = counsel_specs or model_specs

    def counsel_node(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")
        output = run_counsel_stage(
            task=task,
            system_prompt=system_prompt,
            workspace_dir=workspace_dir,
            agent_name=agent_name,
            max_debate_rounds=max_debate_rounds,
            counsel_specs=_specs,
            model_timeout_seconds=_timeout,
        )
        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
            "agent_task": None,
        }

    counsel_node.__name__ = agent_name
    return counsel_node
