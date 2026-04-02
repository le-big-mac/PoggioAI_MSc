"""
Persona council — 3-persona debate for research proposal synthesis, plus
dual-lens evaluation of formalized results.

The persona council runs a structured multi-phase debate:
  1. Independent evaluation — each persona assesses the task from its unique lens.
  2. Debate rounds       — personas critique each other's evaluations (parallel per round).
  3. Synthesis            — a synthesis model integrates the debate into a 1-2 page proposal.

The duality check performs two parallel evaluations of formalized results from
complementary analytical lenses (Check A and Check B), returning structured
pass/fail verdicts with scores and suggestions.

Both integrate with BudgetManager for cost tracking and use ThreadPoolExecutor
for parallel execution, matching the patterns in counsel.py.

Usage (via graph.py):
    from .persona_council import create_persona_council_node, create_duality_check_node
    council_node = create_persona_council_node(workspace_dir, ...)
    duality_node = create_duality_check_node(workspace_dir, ...)
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cli_completion import cli_completion
from .prompts.persona_instructions import (
    PERSONA_POST_SYNTHESIS_VOTE_PROMPT,
    PERSONA_SYSTEM_PROMPTS,
    PERSONA_SYNTHESIS_PROMPT,
)
from .prompts.duality_check_instructions import (
    DUALITY_CHECK_A_PROMPT,
    DUALITY_CHECK_B_PROMPT,
)


# ---------------------------------------------------------------------------
# Default persona model specs
# ---------------------------------------------------------------------------

DEFAULT_PERSONA_MODEL_SPECS: List[Dict[str, Any]] = [
    {"persona": "practical_compass",   "model": "claude-opus-4-6",      "reasoning_effort": "high"},
    {"persona": "rigor_novelty",       "model": "gpt-5.4",              "reasoning_effort": "high"},
    {"persona": "narrative_architect",  "model": "gemini-3.1-pro-preview", "thinking_budget": 32768},
]

DEFAULT_SYNTHESIS_MODEL = "claude-opus-4-6"
DEFAULT_DUALITY_CHECK_MODEL = "claude-opus-4-6"

# Max chars to read from each workspace file for duality check context.
_DUALITY_FILE_TRUNCATE = 8000

# False-positive patterns stripped before scanning for ACCEPT/REJECT verdicts.
_FALSE_POSITIVE_PATTERNS = [
    r"REJECT THE (?:PREMISE|CLAIM|ASSUMPTION|FRAMING)",
    r"WOULD REJECT THE",
    r"CANNOT ACCEPT",
    r"NOT ACCEPT",
    r"REFUSE TO ACCEPT",
]


def _model_to_backend(model_id: str) -> str:
    """Map a model identifier to a cli_completion backend name."""
    m = model_id.lower()
    if m.startswith("gpt-") or m.startswith("o3-") or m.startswith("o4-"):
        return "codex"
    if m.startswith("gemini-"):
        return "gemini"
    # Default covers "claude-*", "anthropic*", and anything else
    return "claude"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_verdict(text: str) -> str:
    """Extract ACCEPT/REJECT verdict from persona output.

    Uses a two-pass approach:
    1. Look for a structured ``VERDICT: ACCEPT/REJECT`` marker near the end
       of the text (preferred — reliable and unambiguous).
    2. Fall back to a full-text scan with false-positive filtering.

    Returns ``"ACCEPT"``, ``"REJECT"``, or ``"UNKNOWN"`` if neither is found.
    """
    if not text:
        return "UNKNOWN"

    # Pass 1: structured marker in the last 500 chars
    tail = text[-500:].upper()
    structured = re.search(r"(?:FINAL\s+)?VERDICT\s*:\s*(ACCEPT|REJECT)", tail)
    if structured:
        return structured.group(1)

    # Pass 2: full-text scan with false-positive pattern removal
    scan_text = text.upper()
    for pattern in _FALSE_POSITIVE_PATTERNS:
        scan_text = re.sub(pattern, "", scan_text)

    if re.search(r"\bACCEPT\b", scan_text):
        return "ACCEPT"
    if re.search(r"\bREJECT\b", scan_text):
        return "REJECT"
    return "UNKNOWN"


def _read_file_truncated(path: str, max_chars: int = _DUALITY_FILE_TRUNCATE) -> str:
    """Read a file and truncate to *max_chars*.  Returns empty string on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(max_chars)
        if os.path.getsize(path) > max_chars:
            content += "\n... [truncated]"
        return content
    except Exception:
        return ""


def _parse_json_response(text: str) -> Optional[dict]:
    """Parse a JSON object from an LLM response, handling markdown code fences."""
    # Try to extract from code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


## _record_budget removed — LLM calls now go through cli_completion().


# ---------------------------------------------------------------------------
# Core: run_persona_council
# ---------------------------------------------------------------------------

def run_persona_council(
    task: str,
    persona_specs: Optional[List[Dict[str, Any]]] = None,
    max_debate_rounds: int = 3,
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL,
    budget_manager: Optional[Any] = None,
    timeout_seconds: int = 600,
    max_post_vote_retries: int = 1,
    synthesis_prompt_override: Optional[str] = None,
    council_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Run a 3-persona debate to synthesize a research proposal.

    Each persona runs as a single long-lived CLI agent that:
    1. Writes its evaluation to a shared directory
    2. Polls for other personas' evaluations (file-based coordination)
    3. Conducts all debate rounds in the same session
    4. Writes a final verdict

    This uses 3 + 1 = 4 CLI invocations total (3 personas + 1 synthesis),
    regardless of debate rounds.

    Returns (proposal_text, verdicts) where verdicts maps persona name to
    "ACCEPT" | "REJECT" | "UNKNOWN".
    """
    import subprocess as _sp
    import tempfile

    specs = persona_specs or DEFAULT_PERSONA_MODEL_SPECS

    # Create coordination directory for file-based communication
    if council_dir:
        coord_dir = os.path.join(council_dir, "persona_council")
    else:
        coord_dir = tempfile.mkdtemp(prefix="persona_council_")
    os.makedirs(coord_dir, exist_ok=True)

    other_names = {spec["persona"] for spec in specs}

    # ------------------------------------------------------------------
    # Phase 1+2: Spawn all personas in parallel (each does eval + debate)
    # ------------------------------------------------------------------

    def _run_persona(spec: Dict[str, Any], task_override: Optional[str] = None,
                     coord_dir_override: Optional[str] = None) -> Tuple[str, str, str]:
        """Run one persona through eval + wait + debate + verdict in a single CLI session."""
        effective_task = task_override or task
        effective_coord_dir = coord_dir_override or coord_dir
        persona_name = spec["persona"]
        model_id = spec["model"]
        backend = _model_to_backend(model_id)
        system_prompt = PERSONA_SYSTEM_PROMPTS.get(persona_name, "")

        peers = [s["persona"] for s in specs if s["persona"] != persona_name]

        cd = effective_coord_dir

        # Build wait commands for initial eval barrier
        wait_for_evals = " && ".join(
            f'while [ ! -f "{cd}/{p}.md" ]; do sleep 3; done'
            for p in peers
        )

        peer_files_str = " ".join(f"{cd}/{p}.md" for p in peers)

        # Example wait command for round N (used in the prompt as a template)
        wait_example = " && ".join(
            f'while [ ! -f "{cd}/{p}_round{{N}}.done" ]; do sleep 3; done'
            for p in peers
        )

        prompt = f"""{system_prompt}

You are participating in a multi-persona research council debate.
Your persona is: {persona_name}

INSTRUCTIONS — complete ALL steps in order within this single session:

STEP 1: EVALUATE
Read and evaluate the following research proposal from your persona's lens.
Write your evaluation (assessment, strengths, gaps, initial verdict) to:
  {cd}/{persona_name}.md

STEP 2: WAIT FOR PEER EVALUATIONS
Run this command to wait for the other personas to finish their evaluations:
  {wait_for_evals}
Then read their evaluations from: {peer_files_str}

STEP 3: DEBATE ({max_debate_rounds} synchronized rounds)
For each round N from 1 to {max_debate_rounds}:
  a) If N > 1, wait for peers to finish round N-1 (replace {{N}} with the actual round):
       {wait_example}
     Then re-read peer files: {peer_files_str}
     (Round 1 needs no wait — you already read peers in step 2.)
  b) Append your critique to {cd}/{persona_name}.md under "## Debate Round N".
     Focus on the single strongest reason the proposal should be REJECTED from your
     lens. Be a harsh critic. Only concede if evidence from another persona is overwhelming.
  c) Signal completion: touch {cd}/{persona_name}_roundN.done

STEP 4: FINAL VERDICT
Append to {cd}/{persona_name}.md:
## Final Verdict
VERDICT: ACCEPT or REJECT
One-sentence justification.

STEP 5: DONE
Output "PERSONA COMPLETE" as your last line.

THE PROPOSAL TO EVALUATE:
{effective_task}
"""
        if backend == "claude":
            cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "40"]
            if model_id:
                cmd.extend(["--model", model_id])
            cmd.extend(["--allowedTools",
                         "Read,Write,Edit,WebFetch,WebSearch,Bash(sleep*),Bash(cat*),Bash(ls*),Bash(touch*),Bash(while*),Bash(test*),Bash([*),Glob,Grep"])
        elif backend == "codex":
            cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
            if model_id:
                cmd.extend(["-m", model_id])
        elif backend == "gemini":
            cmd = ["gemini", "--approval-mode", "yolo"]
            if model_id:
                cmd.extend(["--model", model_id])
        else:
            return persona_name, f"[unknown backend: {backend}]", "UNKNOWN"

        print(f"[persona_council] Spawning {persona_name} ({backend}/{model_id})...")
        try:
            result = _sp.run(
                cmd, input=prompt, capture_output=True, text=True,
                cwd=cd, timeout=timeout_seconds,
                env=os.environ.copy(),
            )
            stdout = result.stdout.strip()
        except _sp.TimeoutExpired:
            stdout = f"[{persona_name} timed out after {timeout_seconds}s]"
            print(f"[persona_council] {persona_name} TIMED OUT.")
        except FileNotFoundError:
            stdout = f"[{persona_name} error: {backend} CLI not found]"
            print(f"[persona_council] {persona_name} error: {backend} not found.")

        # Read the persona's output file
        eval_path = os.path.join(cd, f"{persona_name}.md")
        if os.path.isfile(eval_path):
            with open(eval_path) as f:
                eval_text = f.read()
        else:
            raise RuntimeError(
                f"Persona {persona_name} ({backend}/{model_id}) did not write "
                f"{eval_path}. Stdout ({len(stdout)} chars): {stdout[:500]}"
            )

        verdict = _extract_verdict(eval_text)
        print(f"[persona_council] {persona_name} complete — verdict: {verdict}")

        # Budget tracking
        try:
            from .cli_budget import get_global_cli_tracker
            tracker = get_global_cli_tracker()
            if tracker:
                tracker.record_invocation(
                    agent_name=f"persona_{persona_name}",
                    backend=backend,
                    model=model_id or "default",
                    duration_seconds=0,
                    prompt_chars=len(prompt),
                    output_chars=len(eval_text),
                )
        except Exception:
            pass

        return persona_name, eval_text, verdict

    # Run all personas in parallel
    evaluations: Dict[str, str] = {}
    verdicts: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(_run_persona, spec): spec["persona"] for spec in specs}
        try:
            for future in as_completed(futures, timeout=timeout_seconds + 120):
                try:
                    name, eval_text, verdict = future.result(timeout=timeout_seconds + 60)
                    evaluations[name] = eval_text
                    verdicts[name] = verdict
                except Exception as e:
                    for f, pname in futures.items():
                        if f is future:
                            evaluations[pname] = f"[{pname} error: {e}]"
                            verdicts[pname] = "UNKNOWN"
                            break
        except TimeoutError:
            for f, pname in futures.items():
                if not f.done():
                    evaluations[pname] = f"[{pname} timed out]"
                    verdicts[pname] = "UNKNOWN"
                    f.cancel()

    # ------------------------------------------------------------------
    # Phase 3 — Synthesis (1 CLI call)
    # ------------------------------------------------------------------

    verdict_summary = ", ".join(f"{k}={v}" for k, v in verdicts.items())
    accept_count = sum(1 for v in verdicts.values() if v == "ACCEPT")
    reject_count = sum(1 for v in verdicts.values() if v == "REJECT")

    formatted_evals = "\n\n".join(
        f"=== {name} (verdict: {verdicts.get(name, 'UNKNOWN')}) ===\n{text}"
        for name, text in evaluations.items()
    )

    synthesis_input = (
        f"VERDICT SUMMARY: {verdict_summary} "
        f"({accept_count} ACCEPT, {reject_count} REJECT)\n\n"
        f"Original task:\n{task}\n\n"
        f"Persona evaluations and debate:\n\n{formatted_evals}"
    )

    try:
        proposal_text = cli_completion(
            synthesis_input,
            system_prompt=synthesis_prompt_override or PERSONA_SYNTHESIS_PROMPT,
            backend=_model_to_backend(synthesis_model),
        ) or ""
    except Exception as e:
        print(f"[persona_council] Synthesis failed ({e}), using first evaluation as fallback.")
        first_eval = next(iter(evaluations.values()), f"[synthesis error: {e}]")
        proposal_text = first_eval

    # ------------------------------------------------------------------
    # Phase 4 — Handle rejections
    #   3/3 reject → UNVIABLE immediately (no point retrying)
    #   2/3 reject → synthesize fix, re-run council, 2+ reject again → UNVIABLE
    #   0-1 reject → proceed with synthesized proposal
    # ------------------------------------------------------------------

    if reject_count == 3:
        # Unanimous reject — UNVIABLE, no retry
        rejection_reasons = "\n\n".join(
            f"**{name}**:\n{evaluations[name]}"
            for name in evaluations
        )
        proposal_text = (
            "## Verdict: UNVIABLE\n\n"
            "All three personas unanimously rejected this research direction.\n\n"
            "## Rejection Reasons\n\n"
            f"{rejection_reasons}"
        )
        print("[persona_council] UNVIABLE — unanimous rejection.")

    elif reject_count == 2:
        print("[persona_council] 2/3 rejected — re-running council on synthesized fix...")

        # Clean coordination dir for second round
        import shutil as _shutil
        coord_dir_retry = coord_dir + "_retry"
        if os.path.exists(coord_dir_retry):
            _shutil.rmtree(coord_dir_retry)
        os.makedirs(coord_dir_retry, exist_ok=True)

        # Re-run all personas on the synthesized proposal
        retry_evaluations: Dict[str, str] = {}
        retry_verdicts: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {
                pool.submit(_run_persona, spec, task_override=proposal_text,
                            coord_dir_override=coord_dir_retry): spec["persona"]
                for spec in specs
            }
            try:
                for future in as_completed(futures, timeout=timeout_seconds + 120):
                    try:
                        name, eval_text, verdict = future.result(timeout=timeout_seconds + 60)
                        retry_evaluations[name] = eval_text
                        retry_verdicts[name] = verdict
                    except Exception as e:
                        for f, pname in futures.items():
                            if f is future:
                                retry_evaluations[pname] = f"[{pname} error: {e}]"
                                retry_verdicts[pname] = "UNKNOWN"
                                break
            except TimeoutError:
                for f, pname in futures.items():
                    if not f.done():
                        retry_evaluations[pname] = f"[{pname} timed out]"
                        retry_verdicts[pname] = "UNKNOWN"
                        f.cancel()

        retry_reject_count = sum(1 for v in retry_verdicts.values() if v == "REJECT")
        print(f"[persona_council] Retry verdicts: {retry_verdicts}")

        if retry_reject_count >= 2:
            rejection_reasons = "\n\n".join(
                f"**{name}** ({retry_verdicts[name]}):\n{retry_evaluations[name]}"
                for name in retry_evaluations
                if retry_verdicts.get(name) == "REJECT"
            )
            proposal_text = (
                "## Verdict: UNVIABLE\n\n"
                "This research direction was rejected after two rounds of evaluation. "
                "The synthesis agent attempted to address the initial concerns but "
                "the revised proposal was still rejected.\n\n"
                "## Rejection Reasons\n\n"
                f"{rejection_reasons}"
            )
            verdicts = retry_verdicts
            print("[persona_council] UNVIABLE — rejected on retry.")
        else:
            verdicts = retry_verdicts
            print("[persona_council] Retry passed — proposal accepted after revision.")

    # Warn about UNKNOWN verdicts
    unknown_personas = [name for name, v in verdicts.items() if v == "UNKNOWN"]
    if unknown_personas:
        print(
            f"[persona_council] WARNING: {len(unknown_personas)} persona(s) "
            f"returned UNKNOWN verdict: {unknown_personas}"
        )

    print(f"[persona_council] Complete. Final verdicts: {verdicts}")
    return proposal_text, verdicts


# ---------------------------------------------------------------------------
# Core: run_duality_check
# ---------------------------------------------------------------------------

def run_duality_check(
    workspace_dir: str,
    check_model: str = DEFAULT_DUALITY_CHECK_MODEL,
    budget_manager: Optional[Any] = None,
    timeout_seconds: int = 600,
) -> Dict[str, Any]:
    """
    Run parallel dual-lens evaluation of formalized results.

    Reads key workspace artifacts, then runs Check A and Check B in parallel.
    Each check produces a JSON verdict with ``passed``, ``reasoning``,
    ``score`` (1-10), and ``suggestions``.

    Parameters
    ----------
    workspace_dir : str
        Root workspace directory containing math_workspace / paper_workspace.
    check_model : str
        Model used for both checks.
    budget_manager : BudgetManager or None
        If provided, token usage is recorded.
    timeout_seconds : int
        Per-call timeout (default 600).

    Returns
    -------
    dict
        ``{both_passed: bool, check_a: {...}, check_b: {...}}``
    """
    # ------------------------------------------------------------------
    # Gather workspace context
    # ------------------------------------------------------------------
    context_files = {
        "research_proposal.md": os.path.join(workspace_dir, "paper_workspace", "research_proposal.md"),
        "formalized_results.md": os.path.join(workspace_dir, "math_workspace", "formalized_results.md"),
        "formalized_results.json": os.path.join(workspace_dir, "math_workspace", "formalized_results.json"),
        "claim_graph.json": os.path.join(workspace_dir, "math_workspace", "claim_graph.json"),
        "experiment_results.json": next(
            (p for p in [
                os.path.join(workspace_dir, "paper_workspace", "experiment_results.json"),
                os.path.join(workspace_dir, "experiment_workspace", "results_summary.json"),
                os.path.join(workspace_dir, "writeup_agent", "experiment_results.json"),
                os.path.join(workspace_dir, "experiment_results.json"),
            ] if os.path.isfile(p)),
            os.path.join(workspace_dir, "experiment_results.json"),  # fallback (may not exist)
        ),
    }

    context_parts: List[str] = []
    for label, path in context_files.items():
        content = _read_file_truncated(path, _DUALITY_FILE_TRUNCATE)
        if content:
            context_parts.append(f"=== {label} ===\n{content}")

    workspace_context = "\n\n".join(context_parts) if context_parts else "[no workspace artifacts found]"

    # ------------------------------------------------------------------
    # Run Check A and Check B in parallel
    # ------------------------------------------------------------------
    default_fail = {"passed": False, "reasoning": "Check did not complete.", "score": 0, "suggestions": []}

    def _run_check(prompt_template: str, check_label: str) -> Tuple[str, dict]:
        user_content = f"{prompt_template}\n\n--- WORKSPACE ARTIFACTS ---\n\n{workspace_context}"
        try:
            raw = cli_completion(
                user_content,
                backend=_model_to_backend(check_model),
            ) or ""

            parsed = _parse_json_response(raw)
            if parsed is None:
                print(f"[duality_check] {check_label}: failed to parse JSON from response.")
                return check_label, {
                    "passed": False,
                    "reasoning": f"Failed to parse JSON response. Raw: {raw[:500]}",
                    "score": 0,
                    "suggestions": [],
                }
            # Normalize keys
            result = {
                "passed": bool(parsed.get("passed", False)),
                "reasoning": str(parsed.get("reasoning", "")),
                "score": int(parsed.get("score", 0)),
                "suggestions": list(parsed.get("suggestions", [])),
            }
            return check_label, result
        except Exception as e:
            print(f"[duality_check] {check_label} error: {e}")
            return check_label, {
                "passed": False,
                "reasoning": f"Check error: {e}",
                "score": 0,
                "suggestions": [],
            }

    results: Dict[str, dict] = {"check_a": dict(default_fail), "check_b": dict(default_fail)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_run_check, DUALITY_CHECK_A_PROMPT, "check_a")
        future_b = pool.submit(_run_check, DUALITY_CHECK_B_PROMPT, "check_b")
        try:
            for future in as_completed([future_a, future_b], timeout=timeout_seconds + 60):
                try:
                    label, result = future.result(timeout=timeout_seconds)
                    results[label] = result
                except TimeoutError:
                    check_label = "check_a" if future is future_a else "check_b"
                    results[check_label] = {
                        "passed": False,
                        "reasoning": f"Timed out after {timeout_seconds}s",
                        "score": 0,
                        "suggestions": [],
                    }
                    print(f"[duality_check] {check_label} TIMED OUT.")
                except Exception as e:
                    check_label = "check_a" if future is future_a else "check_b"
                    results[check_label] = {
                        "passed": False,
                        "reasoning": f"Execution error: {e}",
                        "score": 0,
                        "suggestions": [],
                    }
        except TimeoutError:
            print(f"[duality_check] outer timeout — one or both checks did not complete within {timeout_seconds + 60}s")
            for future, check_label in [(future_a, "check_a"), (future_b, "check_b")]:
                if not future.done():
                    results[check_label] = {
                        "passed": False,
                        "reasoning": f"Timed out after {timeout_seconds}s",
                        "score": 0,
                        "suggestions": [],
                    }
                    future.cancel()

    both_passed = results["check_a"]["passed"] and results["check_b"]["passed"]
    final = {"both_passed": both_passed, "check_a": results["check_a"], "check_b": results["check_b"]}

    print(
        f"[duality_check] Complete. "
        f"A: {'PASS' if results['check_a']['passed'] else 'FAIL'} (score {results['check_a']['score']}), "
        f"B: {'PASS' if results['check_b']['passed'] else 'FAIL'} (score {results['check_b']['score']}). "
        f"Both passed: {both_passed}"
    )
    return final


# ---------------------------------------------------------------------------
# LangGraph node factories
# ---------------------------------------------------------------------------

def create_persona_council_node(
    workspace_dir: str,
    persona_specs: Optional[List[Dict[str, Any]]] = None,
    max_debate_rounds: int = 3,
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL,
    budget_manager: Optional[Any] = None,
    timeout_seconds: int = 600,
    max_post_vote_retries: int = 1,
    synthesis_prompt_override: Optional[str] = None,
) -> Callable:
    """
    Return a LangGraph node callable that runs the persona council.

    The node reads ``state["task"]``, invokes :func:`run_persona_council`,
    writes the proposal and verdicts to ``paper_workspace/``, and returns
    a state-update dict.
    """

    def persona_council_node(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")

        proposal, verdicts = run_persona_council(
            task=task,
            persona_specs=persona_specs,
            max_debate_rounds=max_debate_rounds,
            synthesis_model=synthesis_model,
            budget_manager=budget_manager,
            timeout_seconds=timeout_seconds,
            max_post_vote_retries=max_post_vote_retries,
            synthesis_prompt_override=synthesis_prompt_override,
            council_dir=workspace_dir,
        )

        # Write artifacts to paper_workspace
        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        os.makedirs(paper_ws, exist_ok=True)

        proposal_path = os.path.join(paper_ws, "research_proposal.md")
        try:
            with open(proposal_path, "w", encoding="utf-8") as f:
                f.write(proposal)
        except Exception as e:
            print(f"[persona_council_node] Failed to write proposal: {e}")

        verdicts_path = os.path.join(paper_ws, "persona_verdicts.json")
        try:
            with open(verdicts_path, "w", encoding="utf-8") as f:
                json.dump(verdicts, f, indent=2)
        except Exception as e:
            print(f"[persona_council_node] Failed to write verdicts: {e}")

        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "persona_council": proposal,
            },
            "research_proposal": proposal,
            "artifacts": {
                **state.get("artifacts", {}),
                "research_proposal": proposal_path,
                "persona_verdicts": verdicts_path,
            },
        }

    persona_council_node.__name__ = "persona_council"
    return persona_council_node


def create_duality_check_node(
    workspace_dir: str,
    check_model: str = DEFAULT_DUALITY_CHECK_MODEL,
    budget_manager: Optional[Any] = None,
    timeout_seconds: int = 600,
) -> Callable:
    """
    Return a LangGraph node callable that runs the duality check.

    Invokes :func:`run_duality_check`, writes the result to
    ``paper_workspace/duality_check.json``, and returns a state-update dict.
    """

    def duality_check_node(state: dict) -> dict:
        results = run_duality_check(
            workspace_dir=workspace_dir,
            check_model=check_model,
            budget_manager=budget_manager,
            timeout_seconds=timeout_seconds,
        )

        # Write artifact
        paper_ws = os.path.join(workspace_dir, "paper_workspace")
        os.makedirs(paper_ws, exist_ok=True)

        check_path = os.path.join(paper_ws, "duality_check.json")
        try:
            with open(check_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"[duality_check_node] Failed to write duality check: {e}")

        # Build human-readable summary for agent_outputs
        summary_parts = []
        for label in ("check_a", "check_b"):
            c = results.get(label, {})
            status = "PASS" if c.get("passed") else "FAIL"
            summary_parts.append(f"{label}: {status} (score {c.get('score', '?')}/10)")
        summary = f"Duality check: {' | '.join(summary_parts)}. Both passed: {results.get('both_passed', False)}"

        return {
            "duality_check_result": results,
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "duality_check": summary,
            },
            "artifacts": {
                **state.get("artifacts", {}),
                "duality_check": check_path,
            },
        }

    duality_check_node.__name__ = "duality_check"
    return duality_check_node
