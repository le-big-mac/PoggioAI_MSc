# Data Format Reference

This document describes the JSON/JSONL file formats produced and consumed by consortium.

---

## `experiment_metadata.json`

Written at workspace creation. Records the environment that produced a run.

```json
{
  "timestamp": "2026-03-07T12:00:00.000000",
  "platform": "macOS-15.0-arm64",
  "python_version": "3.11.5",
  "git_commit": "abc1234def5678...",
  "git_dirty": false,
  "model": "claude-opus-4-6",
  "task_preview": "Investigate whether batch normalization...",
  "cli_args": {
    "enable_math_agents": false,
    "enable_counsel": false,
    "output_format": "latex",
    "enforce_paper_artifacts": false,
    "min_review_score": 8
  }
}
```

---

## `run_summary.json`

Written at run completion. Provides a quick overview of what the run produced.

```json
{
  "task": "Investigate whether batch normalization...",
  "model": "claude-opus-4-6",
  "started_at": "2026-03-07T12:00:00",
  "duration_seconds": 1842.3,
  "stages_completed": [
    "ideation_agent",
    "literature_review_agent",
    "research_planner_agent",
    "experimentation_agent",
    "results_analysis_agent",
    "resource_preparation_agent",
    "writeup_agent",
    "proofreading_agent",
    "reviewer_agent"
  ],
  "total_invocations": 47,
  "total_wall_clock_seconds": 1842.3,
  "final_paper": "final_paper.pdf",
  "workspace": "results/consortium_20260307_120000"
}
```

---

## `cli_budget_state.json`

Written after every CLI agent invocation. Tracks cumulative usage via the CLIBudgetTracker.

```json
{
  "invocation_limit": 500,
  "wall_clock_limit_seconds": 14400,
  "total_invocations": 47,
  "total_wall_clock_seconds": 1842.3,
  "by_backend": {
    "claude": 35,
    "codex": 12
  },
  "last_updated": "2026-03-07T14:30:00Z"
}
```

---

## `cli_budget_ledger.jsonl`

Append-only log of every CLI agent invocation. One JSON object per line.

```jsonl
{"call_id": "uuid-1234", "timestamp": "2026-03-07T12:05:00Z", "backend": "claude", "model": "claude-opus-4-6", "wall_clock_seconds": 45.2, "invocation_count": 1, "total_invocations": 1, "invocation_limit": 500}
{"call_id": "uuid-5678", "timestamp": "2026-03-07T12:10:00Z", "backend": "claude", "model": "claude-opus-4-6", "wall_clock_seconds": 38.7, "invocation_count": 1, "total_invocations": 2, "invocation_limit": 500}
```

---

## `math_workspace/claim_graph.json`

Stores the directed acyclic graph of mathematical claims. Produced by MathProposerAgent.

```json
{
  "nodes": {
    "D1": {
      "type": "Definition",
      "id": "D1",
      "label": "L-smooth function",
      "statement": "A function f: R^n -> R is L-smooth if...",
      "status": "accepted",
      "depends_on": []
    },
    "L1": {
      "type": "Lemma",
      "id": "L1",
      "label": "Gradient descent descent lemma",
      "statement": "For L-smooth f, gradient step satisfies...",
      "status": "proven",
      "depends_on": ["D1"]
    },
    "T1": {
      "type": "Theorem",
      "id": "T1",
      "label": "Main convergence theorem",
      "statement": "Under conditions D1 and L1...",
      "status": "proven",
      "depends_on": ["D1", "L1"]
    }
  },
  "edges": [
    {"from": "D1", "to": "L1", "type": "used_by"},
    {"from": "L1", "to": "T1", "type": "used_by"}
  ]
}
```

**Node types**: `Definition`, `Lemma`, `Theorem`, `Corollary`, `Assumption`

**Status values**: `pending`, `draft`, `proven`, `verified`, `accepted`, `rejected`

**Edge types**: `depends_on`, `used_by`, `contradicts`

---

## `paper_workspace/track_decomposition.json`

Written by ResearchPlannerAgent to decompose the task into theory and empirical tracks.

```json
{
  "empirical_questions": [
    "Does batch normalization reduce spectral norm in practice?",
    "How does the effect vary with learning rate?"
  ],
  "theory_questions": [
    "Can we prove an upper bound on spectral norm growth under BN?",
    "What conditions are required for the bound to hold?"
  ],
  "recommended_track": "both",
  "rationale": "The question has both a theoretical component (bounding spectral norm) and an empirical component (measuring it in real training)."
}
```

---

## `paper_workspace/followup_decision.json`

Written by ManagerAgent after the reviewer stage to decide whether to revise or accept.

```json
{
  "decision": "revise",
  "review_score": 6,
  "min_required_score": 8,
  "revision_instructions": "Strengthen the proof in Section 3. Add ablation experiment on learning rate.",
  "iteration": 1
}
```

**Decision values**: `accept`, `revise`, `reject`

---

## `paper_workspace/review_verdict.json`

Written by ReviewerAgent with scoring and actionable feedback.

```json
{
  "overall_score": 7,
  "novelty_score": 8,
  "clarity_score": 6,
  "rigor_score": 7,
  "verdict": "weak_accept",
  "strengths": [
    "Clear hypothesis",
    "Good literature coverage"
  ],
  "weaknesses": [
    "Proof in Section 3 has a gap at step 4",
    "Missing comparison to prior work on spectral norms"
  ],
  "actionable_feedback": "Address the proof gap and add one comparison experiment."
}
```

---

## `agent_llm_calls.jsonl`

Full log of every CLI agent interaction. One JSON object per line. Large file (~20–100 MB per run).

```jsonl
{"timestamp": "2026-03-07T12:05:00Z", "agent": "ideation_agent", "backend": "claude", "model": "claude-opus-4-6", "wall_clock_seconds": 45.2, "message_preview": "Based on the research task, I will..."}
```

---

## `STATUS.txt`

Plain text file written at run end. Contains one of:

- `COMPLETE` — all stages ran and all required artifacts are present
- `INCOMPLETE: missing [final_paper.tex, ...]` — some required artifacts are absent
- `ERROR: <message>` — pipeline raised an unhandled exception

---

## `campaign_status.json`

Written and locked by campaign_heartbeat.py. Tracks multi-stage campaign state.

```json
{
  "campaign_name": "muon_research",
  "spec_file": "/abs/path/to/campaign.yaml",
  "stages": {
    "theory": {
      "status": "completed",
      "workspace": "results/muon_theory_20260307/",
      "pid": null,
      "started_at": "2026-03-07T08:00:00Z",
      "completed_at": "2026-03-07T10:30:00Z",
      "missing_artifacts": [],
      "fail_reason": null
    },
    "experiments": {
      "status": "in_progress",
      "workspace": "results/muon_experiments_20260307/",
      "pid": 12345,
      "started_at": "2026-03-07T10:31:00Z",
      "completed_at": null,
      "missing_artifacts": [],
      "fail_reason": null
    }
  }
}
```

**Stage status values**: `pending`, `in_progress`, `completed`, `failed`
