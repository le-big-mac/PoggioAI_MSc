"""
Campaign planner — dynamically generates campaign stages via multi-model counsel debate.

Analyzes the output of a completed discovery phase (research plan + track decomposition)
and produces a CampaignPlan specifying how many theory/experiment stages are needed,
their dependency ordering, task prompts, and artifact requirements.

The debate follows the same pattern as consortium/counsel.py but simplified: all phases
use cli_completion() directly (no ReAct agents/tools needed since planning is pure
reasoning, not tool-using).
"""

from __future__ import annotations

import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..cli_completion import cli_completion

from .planner_prompt import (
    CAMPAIGN_PLANNING_SYSTEM_PROMPT,
    STAGE_ARTIFACT_TEMPLATES,
    STAGE_ARG_TEMPLATES,
    build_planning_task,
)
from .spec import PlanningConfig, Stage


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlannedStage:
    """A single stage proposed by the planning counsel."""
    id: str
    stage_type: str                  # "theory" | "experiment" | "paper"
    description: str
    task_prompt: str
    depends_on: List[str] = field(default_factory=list)
    context_from: List[str] = field(default_factory=list)
    research_questions: List[str] = field(default_factory=list)
    estimated_budget_usd: float = 0.0


@dataclass
class CampaignPlan:
    """Structured output of the planning counsel."""
    stages: List[PlannedStage]
    rationale: str
    total_estimated_budget_usd: float = 0.0
    research_questions: List[str] = field(default_factory=list)


# Counsel defaults (same models as counsel.py)
_DEFAULT_PLANNING_MODEL_SPECS = [
    {"model": "claude-opus-4-6", "reasoning_effort": "high"},
    {"model": "claude-sonnet-4-6", "reasoning_effort": "high"},
    {"model": "gpt-5.4", "reasoning_effort": "high", "verbosity": "high"},
    {"model": "gemini-3-pro-preview", "thinking_budget": 65536},
]

_SYNTHESIS_MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_campaign_plan(plan: CampaignPlan, max_stages: int = 0) -> List[str]:
    """Validate a campaign plan for structural correctness.

    Returns a list of error strings (empty = valid).
    """
    errors: List[str] = []

    if not plan.stages:
        errors.append("Plan has no stages.")
        return errors

    # Unique IDs
    ids = [s.id for s in plan.stages]
    seen = set()
    for sid in ids:
        if sid in seen:
            errors.append(f"Duplicate stage ID: '{sid}'")
        seen.add(sid)

    id_set = set(ids)

    # Reference integrity
    for s in plan.stages:
        for dep in s.depends_on:
            if dep not in id_set:
                errors.append(f"Stage '{s.id}' depends_on unknown stage '{dep}'")
        for ctx in s.context_from:
            if ctx not in id_set:
                errors.append(f"Stage '{s.id}' context_from unknown stage '{ctx}'")

    # Valid stage types
    valid_types = {"theory", "experiment", "paper"}
    for s in plan.stages:
        if s.stage_type not in valid_types:
            errors.append(f"Stage '{s.id}' has invalid type '{s.stage_type}'")

    # Paper stage must exist and be terminal
    paper_stages = [s for s in plan.stages if s.stage_type == "paper"]
    if not paper_stages:
        errors.append("No paper stage found (must have exactly one)")
    elif len(paper_stages) > 1:
        errors.append(f"Multiple paper stages found: {[s.id for s in paper_stages]}")
    else:
        paper = paper_stages[0]
        # No other stage should depend on the paper stage
        for s in plan.stages:
            if paper.id in s.depends_on and s.id != paper.id:
                errors.append(f"Stage '{s.id}' depends on paper stage '{paper.id}'")

    # At least one non-paper stage
    non_paper = [s for s in plan.stages if s.stage_type != "paper"]
    if not non_paper:
        errors.append("Must have at least one non-paper stage")

    # Max stages constraint
    if max_stages > 0 and len(non_paper) > max_stages:
        errors.append(
            f"Too many non-paper stages: {len(non_paper)} > max {max_stages}"
        )

    # DAG acyclicity check (Kahn's algorithm)
    if not errors:  # only check if no reference errors
        in_degree: Dict[str, int] = {s.id: 0 for s in plan.stages}
        adj: Dict[str, List[str]] = {s.id: [] for s in plan.stages}
        for s in plan.stages:
            for dep in s.depends_on:
                if dep in adj:
                    adj[dep].append(s.id)
                    in_degree[s.id] += 1

        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for child in adj[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if visited != len(plan.stages):
            errors.append("Dependency graph contains a cycle")

    return errors


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_plan_json(text: str) -> dict:
    """Extract and parse JSON from model output, stripping markdown fences."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])

    return json.loads(text)


def _dict_to_plan(d: dict) -> CampaignPlan:
    """Convert a parsed JSON dict to a CampaignPlan."""
    stages = []
    for s in d.get("stages", []):
        stages.append(PlannedStage(
            id=s["id"],
            stage_type=s["stage_type"],
            description=s.get("description", ""),
            task_prompt=s.get("task_prompt", ""),
            depends_on=s.get("depends_on", []),
            context_from=s.get("context_from", []),
            research_questions=s.get("research_questions", []),
            estimated_budget_usd=float(s.get("estimated_budget_usd", 0)),
        ))
    return CampaignPlan(
        stages=stages,
        rationale=d.get("rationale", ""),
        total_estimated_budget_usd=float(d.get("total_estimated_budget_usd", 0)),
        research_questions=d.get("research_questions", []),
    )


def load_campaign_plan(path: str) -> CampaignPlan:
    """Load a CampaignPlan from a campaign_plan.json file."""
    with open(path) as f:
        d = json.load(f)
    return _dict_to_plan(d)


# ---------------------------------------------------------------------------
# Stage conversion
# ---------------------------------------------------------------------------

def plan_to_stages(
    plan: CampaignPlan,
    task_dir: str,
    planning_config: Optional[PlanningConfig] = None,
) -> List[Stage]:
    """Convert a CampaignPlan into Stage objects compatible with CampaignSpec.

    Args:
        plan: The validated campaign plan.
        task_dir: Directory where generated task files are written.
        planning_config: Optional PlanningConfig for stage type constraints.

    Returns:
        List of Stage objects ready for heartbeat orchestration.
    """
    result: List[Stage] = []
    custom_constraints = (planning_config.stage_type_constraints
                          if planning_config else {})

    for ps in plan.stages:
        # Determine args: custom constraints override defaults
        if ps.stage_type in custom_constraints:
            args = list(custom_constraints[ps.stage_type].get(
                "args", STAGE_ARG_TEMPLATES.get(ps.stage_type, [])
            ))
        else:
            args = list(STAGE_ARG_TEMPLATES.get(ps.stage_type, []))

        # Determine artifacts, validators, and memory dirs from templates
        tmpl = STAGE_ARTIFACT_TEMPLATES.get(ps.stage_type, {})
        success_artifacts = dict(tmpl.get("success_artifacts", {
            "required": [], "optional": [],
        }))
        artifact_validators = dict(tmpl.get("artifact_validators", {}))
        memory_dirs = list(tmpl.get("memory_dirs", []))

        # context_from defaults to depends_on if not specified
        context_from = ps.context_from if ps.context_from else list(ps.depends_on)

        # Task file path
        task_file = os.path.join(task_dir, f"{ps.id}_task.txt")

        result.append(Stage(
            id=ps.id,
            task_file=task_file,
            args=args,
            depends_on=list(ps.depends_on),
            context_from=context_from,
            memory_dirs=memory_dirs,
            success_artifacts=success_artifacts,
            artifact_validators=artifact_validators,
        ))

    return result


def generate_task_files(plan: CampaignPlan, task_dir: str) -> Dict[str, str]:
    """Write task prompt files for each planned stage.

    Args:
        plan: The campaign plan with task prompts.
        task_dir: Directory to write task files into.

    Returns:
        Mapping of stage_id -> absolute task file path.
    """
    os.makedirs(task_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for ps in plan.stages:
        path = os.path.join(task_dir, f"{ps.id}_task.txt")
        with open(path, "w") as f:
            f.write(ps.task_prompt)
        paths[ps.id] = os.path.abspath(path)
    return paths


# ---------------------------------------------------------------------------
# Planning counsel runner
# ---------------------------------------------------------------------------

def _model_to_backend(model_id: str) -> str:
    if "claude" in model_id or "anthropic" in model_id: return "claude"
    if "gpt" in model_id or model_id.startswith(("o1-","o3-","o4-")): return "codex"
    if "gemini" in model_id: return "gemini"
    return "claude"


def run_campaign_planning_counsel(
    research_plan_text: str,
    track_decomposition: Optional[dict] = None,
    planning_config: Optional[PlanningConfig] = None,
    model_specs: Optional[List[dict]] = None,
    max_debate_rounds: int = 3,
    model_timeout_seconds: int = 600,
) -> CampaignPlan:
    """Run multi-model counsel to produce a campaign plan.

    Follows the same debate pattern as counsel.py but simplified:
    all phases use cli_completion() directly (no tools needed).

    1. Proposal phase — each model independently proposes a campaign structure
    2. Debate phase   — models critique each other's proposals
    3. Synthesis      — Opus produces the final consensus plan

    Args:
        research_plan_text: Full text of the research plan from discovery phase.
        track_decomposition: The track_decomposition.json dict (if available).
        planning_config: PlanningConfig with constraints.
        model_specs: Override model specs (default: 4 frontier models).
        max_debate_rounds: Number of debate rounds.
        model_timeout_seconds: Per-model timeout.

    Returns:
        Validated CampaignPlan.

    Raises:
        ValueError: If the counsel fails to produce a valid plan after retries.
    """
    specs = model_specs or (
        planning_config.counsel_models if planning_config and planning_config.counsel_models
        else _DEFAULT_PLANNING_MODEL_SPECS
    )

    constraints = {}
    if planning_config:
        constraints["max_stages"] = planning_config.max_stages
        constraints["max_parallel"] = planning_config.max_parallel

    task = build_planning_task(
        research_plan_text=research_plan_text,
        track_decomposition=track_decomposition,
        constraints=constraints if constraints else None,
    )

    # ------------------------------------------------------------------
    # 1. Proposal phase — each model proposes a campaign structure
    # ------------------------------------------------------------------
    print("[planner] Starting proposal phase...")

    def _one_proposal(i: int) -> tuple:
        spec = specs[i]
        try:
            output = cli_completion(
                task,
                system_prompt=CAMPAIGN_PLANNING_SYSTEM_PROMPT,
                backend=_model_to_backend(spec["model"]),
            ) or ""
        except Exception as e:
            output = f"[{spec['model']} failed: {e}]"
        print(f"[planner] model_{i} ({spec['model']}) proposal complete.")
        return i, output

    proposals: List[str] = [""] * len(specs)
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(_one_proposal, i): i for i in range(len(specs))}
        for future in as_completed(futures, timeout=model_timeout_seconds + 60):
            try:
                idx, output = future.result(timeout=model_timeout_seconds)
                proposals[idx] = output
            except Exception as e:
                for f, i in futures.items():
                    if f is future:
                        label = specs[i]["model"] if i < len(specs) else f"model_{i}"
                        proposals[i] = f"[{label} error: {e}]"
                        print(f"[planner] model_{i} error: {e}")
                        break

    # ------------------------------------------------------------------
    # 2. Debate phase
    # ------------------------------------------------------------------
    formatted = "\n\n".join(
        f"=== Proposal {i} ({specs[i]['model'] if i < len(specs) else i}) ===\n{out}"
        for i, out in enumerate(proposals)
    )
    debate_history: List[str] = []

    for rnd in range(max_debate_rounds):
        print(f"[planner] Starting debate round {rnd + 1}/{max_debate_rounds}...")
        base_prompt = (
            f"You are evaluating campaign structure proposals for a research campaign.\n\n"
            f"Original research plan task:\n{task}\n\n"
            f"Here are {len(proposals)} independent proposals:\n\n{formatted}\n\n"
        )
        if debate_history:
            base_prompt += "Prior debate:\n" + "\n---\n".join(debate_history) + "\n\n"
        base_prompt += (
            "Evaluate each proposal and identify:\n"
            "1. Which proposal has the best stage decomposition and why\n"
            "2. Weaknesses in each proposal (missing stages, wrong dependencies, "
            "unnecessary stages)\n"
            "3. The optimal synthesis incorporating the best elements\n"
            "Be specific about dependency ordering, parallelism opportunities, "
            "and whether the task prompts are detailed enough."
        )

        def _one_critique(i: int) -> tuple:
            spec = specs[i]
            try:
                critique = cli_completion(
                    base_prompt,
                    backend=_model_to_backend(spec["model"]),
                ) or ""
            except Exception as e:
                critique = f"[{spec['model']} debate error: {e}]"
            return i, f"Model {i} ({spec['model']}):\n{critique}"

        critiques: List[str] = [""] * len(specs)
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {pool.submit(_one_critique, i): i for i in range(len(specs))}
            for future in as_completed(futures, timeout=model_timeout_seconds + 60):
                try:
                    i, text = future.result(timeout=model_timeout_seconds)
                    critiques[i] = text
                except Exception as e:
                    for f, idx in futures.items():
                        if f is future:
                            critiques[idx] = f"Model {idx}:\n[debate error: {e}]"
                            break

        debate_history.append(f"[Round {rnd + 1}]\n" + "\n\n".join(critiques))
        print(f"[planner] Debate round {rnd + 1} complete.")

    # ------------------------------------------------------------------
    # 3. Synthesis — Opus produces the final plan as JSON
    # ------------------------------------------------------------------
    print("[planner] Starting synthesis...")
    synthesis_prompt = (
        f"You are the synthesis model for campaign planning.\n\n"
        f"Original task:\n{task}\n\n"
        f"Review {len(proposals)} independent proposals and "
        f"{len(debate_history)} rounds of debate, then produce the FINAL "
        f"campaign plan.\n\n"
        f"Proposals:\n{formatted}\n\n"
        f"Debate:\n" + "\n---\n".join(debate_history) + "\n\n"
        "Produce the final campaign plan as a single JSON object matching "
        "the schema from the system prompt. Incorporate the strongest elements "
        "from all proposals based on the debate consensus. "
        "Output ONLY valid JSON, no other text."
    )

    # Retry loop for synthesis (JSON parsing may fail)
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            user_prompt = synthesis_prompt
            if attempt > 0 and last_error:
                user_prompt += (
                    f"\n\nYour previous output had errors: {last_error}\n"
                    "Please fix and output valid JSON only."
                )

            raw_output = cli_completion(
                user_prompt,
                system_prompt=CAMPAIGN_PLANNING_SYSTEM_PROMPT,
                backend=_model_to_backend(_SYNTHESIS_MODEL),
            ) or ""

            # Parse JSON
            plan_dict = _parse_plan_json(raw_output)
            plan = _dict_to_plan(plan_dict)

            # Validate
            max_stages = planning_config.max_stages if planning_config else 0
            validation_errors = validate_campaign_plan(plan, max_stages=max_stages)
            if validation_errors:
                last_error = "; ".join(validation_errors)
                print(f"[planner] Synthesis attempt {attempt + 1} validation failed: {last_error}")
                continue

            print(f"[planner] Synthesis complete. {len(plan.stages)} stages planned.")
            return plan

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            print(f"[planner] Synthesis attempt {attempt + 1} JSON parse failed: {e}")
        except Exception as e:
            last_error = str(e)
            print(f"[planner] Synthesis attempt {attempt + 1} error: {e}")

    raise ValueError(
        f"Planning counsel failed to produce a valid plan after {max_retries} "
        f"attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Human-readable plan review
# ---------------------------------------------------------------------------

def format_plan_for_review(plan: CampaignPlan) -> str:
    """Format a CampaignPlan as a human-readable markdown document."""
    lines = ["# Campaign Plan Review\n"]

    lines.append(f"**Rationale:** {plan.rationale}\n")

    if plan.research_questions:
        lines.append("## Research Questions\n")
        for q in plan.research_questions:
            lines.append(f"- {q}")
        lines.append("")

    lines.append("## Stage Summary\n")
    lines.append("| Stage | Type | Dependencies | Description |")
    lines.append("|-------|------|-------------|-------------|")
    for s in plan.stages:
        deps = ", ".join(s.depends_on) if s.depends_on else "(none)"
        lines.append(f"| {s.id} | {s.stage_type} | {deps} | {s.description} |")
    lines.append("")

    # Dependency graph (text-based)
    lines.append("## Dependency Graph\n")
    lines.append("```")
    for s in plan.stages:
        if not s.depends_on:
            lines.append(f"  {s.id} ({s.stage_type})")
        else:
            for dep in s.depends_on:
                lines.append(f"  {dep} --> {s.id} ({s.stage_type})")
    lines.append("```\n")

    # Detailed stage descriptions
    lines.append("## Stage Details\n")
    for s in plan.stages:
        lines.append(f"### {s.id} ({s.stage_type})\n")
        lines.append(f"**Description:** {s.description}\n")
        if s.depends_on:
            lines.append(f"**Depends on:** {', '.join(s.depends_on)}")
        if s.context_from:
            lines.append(f"**Context from:** {', '.join(s.context_from)}")
        if s.research_questions:
            lines.append("\n**Research questions:**")
            for q in s.research_questions:
                lines.append(f"- {q}")
        lines.append(f"\n**Task prompt:**\n")
        lines.append(f">{s.task_prompt}\n")

    if plan.total_estimated_budget_usd > 0:
        lines.append(f"\n**Estimated total budget:** ${plan.total_estimated_budget_usd:.2f}\n")

    return "\n".join(lines)
