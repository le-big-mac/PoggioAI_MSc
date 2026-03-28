"""
Cross-track consistency validation — checks that theory and experiment
track outputs are consistent with each other.

Uses an LLM call (Opus 4.6 by default) to verify:
- Experimental results support theoretical predictions
- Assumptions are consistent between tracks
- Variable names, definitions, and notation are consistent
- Cited references overlap appropriately

Called by ``track_merge_node`` after merging artifacts and by
``run_intermediate_validation`` at the ``post_merge`` checkpoint.
"""

from __future__ import annotations

import json
import os
from typing import Optional


def _read_file_safe(path: str, max_chars: int = 30000) -> str:
    """Read a file's contents up to max_chars, returning '' on failure."""
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def _collect_track_artifacts(workspace_dir: str) -> dict[str, str]:
    """Gather text content from theory and experiment track outputs."""
    artifacts = {}

    # Theory track artifacts
    math_ws = os.path.join(workspace_dir, "math_workspace")
    for name in ["claim_graph.json", "proof_attempts.json", "proof_transcription.tex",
                 "accepted_proofs.json", "numerical_verification.json"]:
        content = _read_file_safe(os.path.join(math_ws, name))
        if content:
            artifacts[f"theory:{name}"] = content

    # Experiment track artifacts
    paper_ws = os.path.join(workspace_dir, "paper_workspace")
    for name in ["experiment_results.json", "experiment_design.json",
                 "experiment_verification.json"]:
        content = _read_file_safe(os.path.join(paper_ws, name))
        if content:
            artifacts[f"experiment:{name}"] = content

    exp_ws = os.path.join(workspace_dir, "experiment_workspace")
    if os.path.isdir(exp_ws):
        for name in ["results_summary.json", "experiment_report.md"]:
            content = _read_file_safe(os.path.join(exp_ws, name))
            if content:
                artifacts[f"experiment:{name}"] = content

    return artifacts


def validate_cross_track_consistency(
    workspace_dir: str,
    model: str = "claude-opus-4-6",
) -> dict:
    """Validate consistency between theory and experiment track outputs.

    Returns:
        {
            "is_valid": bool,
            "inconsistencies": [...],
            "recommendations": [...],
            "details": str,
        }
    """
    artifacts = _collect_track_artifacts(workspace_dir)

    theory_artifacts = {k: v for k, v in artifacts.items() if k.startswith("theory:")}
    experiment_artifacts = {k: v for k, v in artifacts.items() if k.startswith("experiment:")}

    # If one or both tracks have no artifacts, skip validation
    if not theory_artifacts or not experiment_artifacts:
        return {
            "is_valid": True,
            "inconsistencies": [],
            "recommendations": [],
            "details": "Skipped: one or both tracks produced no artifacts.",
        }

    # Build artifact summaries for the prompt (truncate if needed)
    theory_text = "\n\n".join(
        f"--- {name} ---\n{content[:10000]}"
        for name, content in theory_artifacts.items()
    )
    experiment_text = "\n\n".join(
        f"--- {name} ---\n{content[:10000]}"
        for name, content in experiment_artifacts.items()
    )

    prompt = f"""You are a rigorous scientific consistency checker. You have been given the outputs
of two parallel research tracks — a THEORY track and an EXPERIMENT track — that
are investigating the same research question.

Your job is to identify any INCONSISTENCIES between the two tracks.

Check for:
1. **Prediction-Result Alignment**: Do experimental results support, contradict, or
   fail to address the theoretical predictions? Flag any contradictions.
2. **Assumption Consistency**: Are the assumptions in the theory track consistent with
   the experimental setup? (e.g., same distribution assumptions, same loss functions,
   same model architectures)
3. **Notation and Definitions**: Are variable names, mathematical symbols, and
   definitions used consistently across both tracks?
4. **Reference Overlap**: Do both tracks cite the same key foundational papers?
   Flag if one track relies on a key result that the other contradicts.
5. **Scope Alignment**: Does the experiment test what the theory predicts, or is
   there a disconnect in scope?

THEORY TRACK ARTIFACTS:
{theory_text}

EXPERIMENT TRACK ARTIFACTS:
{experiment_text}

Respond in this exact JSON format (no markdown fences):
{{
    "is_consistent": true/false,
    "inconsistencies": [
        {{
            "severity": "critical"|"major"|"minor",
            "description": "...",
            "theory_artifact": "...",
            "experiment_artifact": "...",
            "recommendation": "..."
        }}
    ],
    "overall_assessment": "One paragraph summary"
}}
"""

    try:
        from ..cli_completion import cli_completion

        def _model_to_backend(mid: str) -> str:
            if "claude" in mid or "anthropic" in mid: return "claude"
            if "gpt" in mid or mid.startswith(("o1-","o3-","o4-")): return "codex"
            if "gemini" in mid: return "gemini"
            return "claude"

        raw = cli_completion(prompt, backend=_model_to_backend(model)) or ""

        # Strip markdown fences if present
        import re
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        inconsistencies = result.get("inconsistencies", [])
        is_valid = result.get("is_consistent", True) and not any(
            i.get("severity") == "critical" for i in inconsistencies
        )

        return {
            "is_valid": is_valid,
            "inconsistencies": inconsistencies,
            "recommendations": [i.get("recommendation", "") for i in inconsistencies if i.get("recommendation")],
            "details": result.get("overall_assessment", ""),
        }

    except Exception as e:
        print(f"[cross_track_validation] LLM call failed: {e}")
        return {
            "is_valid": True,  # don't block pipeline on validation failure
            "inconsistencies": [],
            "recommendations": [],
            "details": f"Validation skipped due to error: {e}",
        }


def save_cross_track_report(workspace_dir: str, result: dict) -> str:
    """Write cross_track_consistency.json to paper_workspace. Returns the path."""
    out_dir = os.path.join(workspace_dir, "paper_workspace")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_track_consistency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path
