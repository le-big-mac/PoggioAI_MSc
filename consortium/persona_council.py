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
    timeout_seconds: int = 300,
    max_post_vote_retries: int = 1,
    synthesis_prompt_override: Optional[str] = None,
    council_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Run a 3-persona debate to synthesize a research proposal.

    Uses session-resume to keep each persona alive across eval, debate rounds,
    and retry. The orchestrator controls timing in Python — no bash polling.

    Flow:
      1. 3 parallel evals (new sessions)
      2. Orchestrator waits for all eval files
      3. N debate rounds: resume each persona, wait for all to finish each round
      4. Read final verdicts
      5. Synthesis (completion)
      6. If 2/3 reject: resume personas with synthesis, re-evaluate
         If 3/3 reject: UNVIABLE
    """
    import subprocess as _sp
    import tempfile
    import time as _time
    import uuid as _uuid

    specs = persona_specs or DEFAULT_PERSONA_MODEL_SPECS

    if council_dir:
        coord_dir = os.path.join(council_dir, "persona_council")
    else:
        coord_dir = tempfile.mkdtemp(prefix="persona_council_")
    os.makedirs(coord_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------

    def _cli_call(backend: str, model: str, prompt: str, session_id: str,
                  resume: bool = False) -> str:
        """Run a CLI call, optionally resuming a session. Returns stdout."""
        if backend == "claude":
            cmd = ["claude", "-p", "--output-format", "text", "--max-turns", "20"]
            if model:
                cmd.extend(["--model", model])
            cmd.extend(["--allowedTools",
                         "Read,Write,Edit,WebFetch,WebSearch,Bash(cat*),Bash(ls*),Glob,Grep"])
            if resume:
                cmd.extend(["--resume", session_id])
            else:
                cmd.extend(["--session-id", session_id])
        elif backend == "codex":
            if resume:
                cmd = ["codex", "exec", "resume", session_id,
                       "--dangerously-bypass-approvals-and-sandbox"]
            else:
                cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
            if model:
                cmd.extend(["-m", model])
        elif backend == "gemini":
            cmd = ["gemini", "--approval-mode", "yolo"]
            if model:
                cmd.extend(["--model", model])
            if resume:
                cmd.extend(["--resume", session_id])
        else:
            raise ValueError(f"Unknown backend: {backend}")

        result = _sp.run(
            cmd, input=prompt, capture_output=True, text=True,
            cwd=coord_dir, timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        return result.stdout.strip()

    def _extract_session_id(stdout: str, backend: str) -> Optional[str]:
        """Extract session ID from CLI output."""
        for line in stdout.splitlines():
            if "session id:" in line.lower() or "session_id:" in line.lower():
                return line.split(":")[-1].strip()
        return None

    def _wait_for_files(paths: List[str], poll_interval: float = 3.0,
                        max_wait: float = 600.0) -> None:
        """Block until all paths exist."""
        start = _time.time()
        while True:
            if all(os.path.isfile(p) for p in paths):
                return
            if _time.time() - start > max_wait:
                missing = [p for p in paths if not os.path.isfile(p)]
                raise TimeoutError(f"Timed out waiting for: {missing}")
            _time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Setup: assign session IDs and backends
    # ------------------------------------------------------------------

    personas: List[Dict[str, Any]] = []
    for spec in specs:
        name = spec["persona"]
        model_id = spec["model"]
        backend = _model_to_backend(model_id)
        session_id = str(_uuid.uuid4())
        personas.append({
            "name": name,
            "model": model_id,
            "backend": backend,
            "session_id": session_id,
            "system_prompt": PERSONA_SYSTEM_PROMPTS.get(name, ""),
        })

    peer_names = {p["name"] for p in personas}

    # ------------------------------------------------------------------
    # Phase 1: Parallel evaluations (new sessions)
    # ------------------------------------------------------------------

    def _eval_persona(p: Dict) -> Tuple[str, str]:
        peers = [x["name"] for x in personas if x["name"] != p["name"]]
        peer_files = ", ".join(f"{coord_dir}/{peer}.md" for peer in peers)

        prompt = f"""{p['system_prompt']}

You are {p['name']} in a multi-persona research council.

Evaluate this research proposal from your lens. Write your evaluation
(assessment, strengths, critical gaps, and verdict ACCEPT or REJECT) to:
  {coord_dir}/{p['name']}.md

End your evaluation with:
VERDICT: ACCEPT or REJECT

THE PROPOSAL:
{task}
"""
        stdout = _cli_call(p["backend"], p["model"], prompt, p["session_id"], resume=False)
        # For codex, extract the real session ID from output
        if p["backend"] == "codex":
            real_sid = _extract_session_id(stdout, p["backend"])
            if real_sid:
                p["session_id"] = real_sid
        return p["name"], stdout

    print("[persona_council] Phase 1 — evaluations...")
    with ThreadPoolExecutor(max_workers=len(personas)) as pool:
        futures = {pool.submit(_eval_persona, p): p["name"] for p in personas}
        for future in as_completed(futures, timeout=timeout_seconds + 60):
            name = futures[future]
            try:
                future.result(timeout=timeout_seconds)
                print(f"[persona_council] {name} eval complete.")
            except Exception as e:
                print(f"[persona_council] {name} eval failed: {e}")

    # Verify all eval files exist
    eval_files = [os.path.join(coord_dir, f"{p['name']}.md") for p in personas]
    for ef in eval_files:
        if not os.path.isfile(ef):
            raise RuntimeError(f"Persona did not write eval: {ef}")

    # ------------------------------------------------------------------
    # Phase 2: Debate rounds (resume sessions, orchestrator controls timing)
    # ------------------------------------------------------------------

    for rnd in range(1, max_debate_rounds + 1):
        print(f"[persona_council] Phase 2 — debate round {rnd}/{max_debate_rounds}...")

        def _debate_round(p: Dict, round_num: int) -> str:
            peers = [x["name"] for x in personas if x["name"] != p["name"]]
            peer_files = " ".join(f"{coord_dir}/{peer}.md" for peer in peers)

            prompt = (
                f"DEBATE ROUND {round_num}.\n\n"
                f"Read the other personas' latest evaluations from: {peer_files}\n\n"
                f"Write your round {round_num} critique — append it to "
                f"{coord_dir}/{p['name']}.md under '## Debate Round {round_num}'.\n"
                f"Focus on the single strongest reason the proposal should be REJECTED "
                f"from your lens. Be a harsh critic. Only concede if evidence from "
                f"another persona is overwhelming.\n\n"
                f"After writing your critique, update your verdict at the end of your "
                f"file under '## Final Verdict' with VERDICT: ACCEPT or REJECT."
            )
            return _cli_call(p["backend"], p["model"], prompt, p["session_id"], resume=True)

        with ThreadPoolExecutor(max_workers=len(personas)) as pool:
            futures = {pool.submit(_debate_round, p, rnd): p["name"] for p in personas}
            for future in as_completed(futures, timeout=timeout_seconds + 60):
                name = futures[future]
                try:
                    future.result(timeout=timeout_seconds)
                except Exception as e:
                    print(f"[persona_council] {name} debate round {rnd} failed: {e}")

    # ------------------------------------------------------------------
    # Read final evaluations and verdicts
    # ------------------------------------------------------------------

    evaluations: Dict[str, str] = {}
    verdicts: Dict[str, str] = {}
    for p in personas:
        eval_path = os.path.join(coord_dir, f"{p['name']}.md")
        with open(eval_path) as f:
            evaluations[p["name"]] = f.read()
        verdicts[p["name"]] = _extract_verdict(evaluations[p["name"]])
        print(f"[persona_council] {p['name']} final verdict: {verdicts[p['name']]}")

    # ------------------------------------------------------------------
    # Phase 3: Synthesis (completion, no tools)
    # ------------------------------------------------------------------

    accept_count = sum(1 for v in verdicts.values() if v == "ACCEPT")
    reject_count = sum(1 for v in verdicts.values() if v == "REJECT")
    verdict_summary = ", ".join(f"{k}={v}" for k, v in verdicts.items())

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
        print(f"[persona_council] Synthesis failed: {e}")
        proposal_text = next(iter(evaluations.values()), "")

    # ------------------------------------------------------------------
    # Phase 4: Handle rejections (same sessions for retry)
    # ------------------------------------------------------------------

    if reject_count >= 2:
        print("[persona_council] 2/3 rejected — resuming personas with synthesized fix...")

        # Resume each persona with the synthesis, ask to re-evaluate
        def _retry_persona(p: Dict) -> Tuple[str, str, str]:
            retry_path = os.path.join(coord_dir, f"{p['name']}_retry.md")
            prompt = (
                f"The synthesis coordinator has revised the proposal based on all "
                f"three personas' feedback. Here is the revised proposal:\n\n"
                f"{proposal_text}\n\n"
                f"Re-evaluate this revised proposal from your persona's lens.\n"
                f"Write your re-evaluation to: {retry_path}\n"
                f"End with VERDICT: ACCEPT or REJECT"
            )
            _cli_call(p["backend"], p["model"], prompt, p["session_id"], resume=True)
            if os.path.isfile(retry_path):
                with open(retry_path) as f:
                    text = f.read()
            else:
                text = f"[{p['name']} did not write retry evaluation]"
            return p["name"], text, _extract_verdict(text)

        retry_verdicts: Dict[str, str] = {}
        retry_evaluations: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=len(personas)) as pool:
            futures = {pool.submit(_retry_persona, p): p["name"] for p in personas}
            for future in as_completed(futures, timeout=timeout_seconds + 60):
                try:
                    name, text, verdict = future.result(timeout=timeout_seconds)
                    retry_evaluations[name] = text
                    retry_verdicts[name] = verdict
                    print(f"[persona_council] Retry {name}: {verdict}")
                except Exception as e:
                    fname = futures[future]
                    retry_evaluations[fname] = f"[error: {e}]"
                    retry_verdicts[fname] = "UNKNOWN"

        retry_reject_count = sum(1 for v in retry_verdicts.values() if v == "REJECT")

        if retry_reject_count >= 2:
            rejection_reasons = "\n\n".join(
                f"**{name}** ({retry_verdicts[name]}):\n{retry_evaluations[name]}"
                for name in retry_evaluations
                if retry_verdicts.get(name) == "REJECT"
            )
            proposal_text = (
                "## Verdict: UNVIABLE\n\n"
                "This research direction was rejected after two rounds of evaluation.\n\n"
                "## Rejection Reasons\n\n"
                f"{rejection_reasons}"
            )
            verdicts = retry_verdicts
            print("[persona_council] UNVIABLE — rejected on retry.")
        else:
            verdicts = retry_verdicts
            retry_formatted = "\n\n".join(
                f"=== {name} ({retry_verdicts.get(name, 'UNKNOWN')}) ===\n{text}"
                for name, text in retry_evaluations.items()
            )
            retry_synthesis_input = (
                f"VERDICT SUMMARY (RETRY): {sum(1 for v in retry_verdicts.values() if v == 'ACCEPT')} ACCEPT, "
                f"{retry_reject_count} REJECT\n\n"
                f"First synthesis:\n{proposal_text}\n\n"
                f"Retry evaluations:\n\n{retry_formatted}"
            )
            try:
                proposal_text = cli_completion(
                    retry_synthesis_input,
                    system_prompt=synthesis_prompt_override or PERSONA_SYNTHESIS_PROMPT,
                    backend=_model_to_backend(synthesis_model),
                ) or proposal_text
            except Exception as e:
                print(f"[persona_council] Retry synthesis failed: {e}")
            print("[persona_council] Retry passed — re-synthesized.")

    unknown = [n for n, v in verdicts.items() if v == "UNKNOWN"]
    if unknown:
        print(f"[persona_council] WARNING: UNKNOWN verdicts: {unknown}")

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
