"""
Shared graph state schema for the LangGraph research pipeline.

ResearchState is the single source of truth passed between all nodes.
The `messages` field uses add_messages reducer so each node can append
without overwriting prior conversation history.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge dict state produced by parallel branches."""
    return {**(left or {}), **(right or {})}


class ResearchState(TypedDict):
    # -----------------------------------------------------------------
    # Core conversation -- uses add_messages reducer (append-only)
    # -----------------------------------------------------------------
    messages: Annotated[list, add_messages]

    # -----------------------------------------------------------------
    # Run configuration (set once at graph entry, read-only after that)
    # -----------------------------------------------------------------
    task: str
    workspace_dir: str
    pipeline_mode: str          # default | full_research | quick
    execution_scope: str        # all | theory | experiment
    math_enabled: bool
    enforce_paper_artifacts: bool
    enforce_editorial_artifacts: bool
    require_pdf: bool
    require_experiment_plan: bool
    min_review_score: int
    followup_max_iterations: int
    manager_max_steps: int
    pipeline_stages: list[str]   # deterministic stage order for this run

    # -----------------------------------------------------------------
    # Dynamic routing state
    # -----------------------------------------------------------------
    pipeline_stage_index: int             # next stage index in pipeline_stages
    current_agent: Optional[str]          # name of next specialist to invoke
    agent_task: Optional[str]             # task string sent to that specialist

    # -----------------------------------------------------------------
    # Agent outputs -- keyed by agent name
    # -----------------------------------------------------------------
    agent_outputs: Annotated[dict, _merge_dicts]  # agent_name -> last output string

    # -----------------------------------------------------------------
    # Artifact tracking
    # -----------------------------------------------------------------
    artifacts: dict                       # artifact_name -> absolute path

    # -----------------------------------------------------------------
    # Iteration counters
    # -----------------------------------------------------------------
    iteration_count: int                  # total manager steps taken
    followup_iteration: int               # Step 6/6.2 loop counter
    research_cycle: int                   # current plan-execute-analyze cycle
    max_research_cycles: int              # hard cap on follow-up replanning cycles
    novelty_check_attempts: int           # novelty gate retry counter (max 3)
    rebuttal_iteration: int               # rebuttal loop counter (Phase 2)
    max_rebuttal_iterations: int          # hard cap on rebuttal loops (default 2)
    validation_retry_count: int           # validation loop retry counter
    max_validation_retries: int           # hard cap on validation retries (default 3)

    # -----------------------------------------------------------------
    # Validation results
    # -----------------------------------------------------------------
    validation_results: dict              # gate_name -> {is_valid, errors}

    # -----------------------------------------------------------------
    # Live-steering interrupt
    # -----------------------------------------------------------------
    interrupt_instruction: Optional[str]  # set by socket listener thread

    # -----------------------------------------------------------------
    # Theory / experiment track state (populated after Step 4.5)
    # -----------------------------------------------------------------
    theory_track_status: Optional[str]       # pending | in_progress | completed
    experiment_track_status: Optional[str]   # pending | in_progress | completed
    track_decomposition: Optional[dict]      # {"empirical_questions": [...], "theory_questions": [...]}
    theory_track_summary: Optional[dict]     # structured summary from proof_transcription_agent
    theory_repair_count: int                 # intra-track retry counter (max 2)

    # -----------------------------------------------------------------
    # Agentic tree search state
    # -----------------------------------------------------------------
    tree_search_enabled: bool                # whether tree search is active for this run
    tree_state_path: Optional[str]           # path to tree_search_state.json on disk
    active_branch_id: Optional[str]          # currently executing branch node ID

    # -----------------------------------------------------------------
    # Milestone reports & human-in-the-loop gates
    # -----------------------------------------------------------------
    milestone_reports: Annotated[list[str], operator.add]  # paths to generated milestone report PDFs
    human_feedback_history: Annotated[list[dict], operator.add]  # [{phase, action, feedback, timestamp}]
    enable_milestone_gates: bool             # pause at milestones for human input
    milestone_timeout: int                   # seconds to wait for human (default 3600)

    # -----------------------------------------------------------------
    # Intermediate validation checkpoints
    # -----------------------------------------------------------------
    intermediate_validation_log: Annotated[list[dict], operator.add]  # [{checkpoint, timestamp, results}]

    # -----------------------------------------------------------------
    # V2 pipeline — persona council output
    # -----------------------------------------------------------------
    research_proposal: Optional[str]           # 1-2 page synthesized proposal from persona council
    autonomous_mode: bool                      # True = no human-in-the-loop gates

    # -----------------------------------------------------------------
    # V2 pipeline — brainstorm & goals
    # -----------------------------------------------------------------
    brainstorm_output: Optional[str]           # structured practical approaches
    brainstorm_history: list[str]              # accumulated brainstorm outputs across cycles
    research_goals: Optional[dict]             # {goals: [{id, description, success_criteria, track}], total_goals: int}
    formalized_results: Optional[str]          # structured findings from execution

    # -----------------------------------------------------------------
    # V2 pipeline — gate results
    # -----------------------------------------------------------------
    lit_review_feasibility: Optional[dict]     # {feasible: bool, reason: str}
    verify_completion_result: Optional[dict]   # {goals_met, goals_total, ratio, verdict, goal_verdicts}
    verify_completion_history: list[dict]      # previous verify_completion_results for progress vetting
    duality_check_result: Optional[dict]       # {both_passed, check_a: {passed, reasoning, score, suggestions}, check_b: {...}}

    # -----------------------------------------------------------------
    # V2 pipeline — loop counters
    # -----------------------------------------------------------------
    lit_review_attempts: int                   # lit_review→council loop (max 2)
    brainstorm_cycle: int                      # brainstorm re-entry from verify or duality (max 3)
    verify_rework_attempts: int                # verify→formalize_goals loop (max 3)
    duality_rework_attempts: int               # duality→followup_lit→brainstorm loop (max 2)

    # -----------------------------------------------------------------
    # Terminal flag
    # -----------------------------------------------------------------
    finished: bool
