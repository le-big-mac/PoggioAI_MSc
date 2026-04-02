"""
Completion-based agent node — for agents that only need to reason about
inputs and produce structured output (no tools, no file exploration).

The orchestrator reads input files, passes them to cli_completion, parses
the structured output into files, writes them, and checks mandatory artifacts.

Used for: brainstorm, formalize_goals, math_proposer, math_prover,
math_rigorous_verifier, experiment_design, formalize_results.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from ..cli_completion import cli_completion

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 50_000


def _read_workspace_file(workspace_dir: str, rel_path: str) -> str:
    """Read a file from the workspace, returning empty string if not found."""
    full_path = os.path.join(workspace_dir, rel_path)
    if not os.path.isfile(full_path):
        return ""
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read(_MAX_INPUT_CHARS)
    if os.path.getsize(full_path) > _MAX_INPUT_CHARS:
        content += "\n... [truncated]"
    return content


def _parse_output_files(output: str) -> Dict[str, str]:
    """Parse === FILE: path === delimited output into {path: content} dict."""
    parts = re.split(r'^=== FILE:\s*(.+?)\s*===\s*$', output, flags=re.MULTILINE)
    files: Dict[str, str] = {}
    # parts[0] is preamble, then alternating [path, content, ...]
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            path = parts[i].strip()
            content = parts[i + 1].strip()
            # Strip markdown code fences wrapping the whole content
            content = re.sub(r'^```(?:json|markdown|md|tex|text|bibtex)?\s*\n', '', content)
            content = re.sub(r'\n```\s*$', '', content)
            files[path] = content
    return files


def _strip_instructions(system_prompt: str) -> str:
    """Extract domain instructions from a system prompt, stripping boilerplate."""
    from .cli_agent import _strip_react_boilerplate
    return _strip_react_boilerplate(system_prompt)


def create_completion_node(
    system_prompt: str,
    agent_name: str,
    workspace_dir: str,
    input_files: List[str],
    output_files: List[str],
    mandatory_artifacts: List[str],
    backend: str = "claude",
    model: Optional[str] = None,
    timeout: int = 600,
) -> Callable:
    """Create a LangGraph node that runs via cli_completion.

    Instead of spawning a full CLI agent subprocess, the orchestrator:
    1. Reads input files from workspace
    2. Calls cli_completion with instructions + file contents
    3. Parses structured output into files (=== FILE: path === delimiters)
    4. Writes files to workspace
    5. Checks mandatory artifacts exist

    Args:
        system_prompt: Full system prompt (ReAct boilerplate auto-stripped).
        agent_name: Node name for state updates.
        workspace_dir: Root workspace directory.
        input_files: Relative paths to read and include in prompt.
        output_files: Relative paths the model should produce.
        mandatory_artifacts: Files that MUST be produced (raises if missing).
        backend: CLI backend for completion.
        model: Optional model override.
        timeout: Max seconds for the completion call.
    """
    instructions = _strip_instructions(system_prompt)

    def node_fn(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")

        # Read input files
        file_sections = []
        for rel_path in input_files:
            content = _read_workspace_file(workspace_dir, rel_path)
            if content:
                file_sections.append(f"=== INPUT FILE: {rel_path} ===\n{content}")

        file_context = "\n\n".join(file_sections) if file_sections else "(no input files found)"

        output_list = "\n".join(f"  - {f}" for f in output_files)
        mandatory_list = ", ".join(mandatory_artifacts)

        prompt = f"""Here are the input files from the workspace:

{file_context}

<task>
{task}
</task>

Produce your output. For EACH output file, use this exact delimiter format on its own line:
=== FILE: relative/path/filename.ext ===
[file contents — raw content, no markdown fences]

You MUST produce these files: {mandatory_list}

Expected output files:
{output_list}"""

        logger.info("[completion] %s starting — backend=%s model=%s", agent_name, backend, model)
        t0 = time.time()

        output = cli_completion(
            prompt,
            system_prompt=instructions,
            backend=backend,
            model=model,
            timeout=timeout,
        )

        elapsed = time.time() - t0
        logger.info("[completion] %s completed in %.1fs — output_len=%d", agent_name, elapsed, len(output))

        # Parse and write output files
        parsed_files = _parse_output_files(output)

        for rel_path, content in parsed_files.items():
            full_path = os.path.join(workspace_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Save raw completion output for debugging
        debug_path = os.path.join(workspace_dir, f"{agent_name}_completion_output.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(output)

        # Check mandatory artifacts
        missing = []
        for artifact in mandatory_artifacts:
            if not os.path.isfile(os.path.join(workspace_dir, artifact)):
                missing.append(artifact)

        if missing:
            raise RuntimeError(
                f"[{agent_name}] Missing mandatory artifacts: {missing}. "
                f"Agent produced files: {list(parsed_files.keys())}"
            )

        # Budget tracking
        try:
            from ..cli_budget import get_global_cli_tracker
            tracker = get_global_cli_tracker()
            if tracker is not None:
                tracker.record_invocation(
                    agent_name=agent_name,
                    backend=backend,
                    model=model or "default",
                    duration_seconds=elapsed,
                    prompt_chars=len(prompt),
                    output_chars=len(output),
                )
        except Exception:
            pass

        return {
            "agent_outputs": {**state.get("agent_outputs", {}), agent_name: output},
            "agent_task": None,
        }

    node_fn.__name__ = agent_name
    return node_fn
