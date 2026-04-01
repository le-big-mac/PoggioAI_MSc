"""
Campaign spec loader — parses and validates campaign.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class Stage:
    id: str
    task_file: str
    args: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    context_from: List[str] = field(default_factory=list)
    memory_dirs: List[str] = field(default_factory=list)
    # success_artifacts: {"required": [...], "optional": [...]}
    success_artifacts: dict = field(default_factory=lambda: {"required": [], "optional": []})
    # artifact_validators: {"filename": {"min_size_bytes": N, "must_contain": ["str"], "must_not_contain": ["str"]}}
    artifact_validators: Dict[str, dict] = field(default_factory=dict)
    # Optional override: run this script instead of launch_multiagent.py
    launcher_script: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Stage":
        depends_on = d.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        context_from = d.get("context_from", [])
        if isinstance(context_from, str):
            context_from = [context_from]

        artifacts = d.get("success_artifacts", {})
        if isinstance(artifacts, list):
            # shorthand: just a list of required files
            artifacts = {"required": artifacts, "optional": []}
        else:
            artifacts.setdefault("required", [])
            artifacts.setdefault("optional", [])

        validators = d.get("artifact_validators", {})

        return cls(
            id=d["id"],
            task_file=d.get("task_file", ""),
            args=d.get("args", []),
            depends_on=depends_on,
            context_from=context_from,
            memory_dirs=d.get("memory_dirs", []),
            success_artifacts=artifacts,
            artifact_validators=validators,
            launcher_script=d.get("launcher_script"),
        )


@dataclass
class NotificationConfig:
    slack_webhook: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    ntfy_topic: Optional[str] = None
    ntfy_server: Optional[str] = None  # defaults to https://ntfy.sh
    on_stage_complete: bool = True
    on_failure: bool = True
    on_heartbeat: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "NotificationConfig":
        return cls(
            ntfy_topic=_expand_env(d.get("ntfy_topic")),
            ntfy_server=_expand_env(d.get("ntfy_server")),
            slack_webhook=_expand_env(d.get("slack_webhook")),
            telegram_bot_token=_expand_env(d.get("telegram_bot_token")),
            telegram_chat_id=_expand_env(d.get("telegram_chat_id")),
            on_stage_complete=d.get("on_stage_complete", True),
            on_failure=d.get("on_failure", True),
            on_heartbeat=d.get("on_heartbeat", False),
        )


@dataclass
class RepairConfig:
    """Configuration for autonomous failure repair via Claude Code agents."""
    enabled: bool = False
    max_attempts: int = 2
    launcher: str = "local"             # "local" (blocking) or "slurm" (async)
    claude_binary: str = "auto"         # "auto" to search, or explicit path
    model: Optional[str] = "claude-opus-4-6"  # strongest model for repair
    effort: str = "max"                  # Claude Code effort level (low/medium/high/max)
    budget_usd: float = 10.0            # per-attempt budget cap
    timeout_seconds: int = 600           # hard timeout per repair attempt
    allowed_actions: List[str] = field(default_factory=lambda: [
        "edit_code",
        "fix_config",
        "generate_missing_artifacts",
        "install_dependencies",
    ])
    retry_delay_seconds: int = 10       # pause before retrying after repair
    # --- Two-phase plan-then-execute settings ---
    two_phase: bool = True              # enable plan→review→execute flow
    plan_model: Optional[str] = None    # model for planning (defaults to self.model)
    plan_effort: Optional[str] = None   # effort for planning (defaults to self.effort)
    plan_budget_usd: float = 5.0        # budget for the read-only planning phase
    plan_timeout_seconds: int = 300     # timeout for planning phase
    review_model: str = "claude-opus-4-6"  # model OpenClaw uses to judge the plan
    review_temperature: float = 0.2     # low temp for deterministic review
    min_review_score: int = 7           # plan must score >= this (1-10) to proceed
    # --- Backoff settings ---
    backoff_base_seconds: int = 60      # initial backoff after failed repair
    backoff_max_seconds: int = 900      # max backoff (15 minutes)
    # --- REPAIRING state timeout ---
    repairing_timeout_seconds: int = 3600  # 1 hour max in REPAIRING state
    # --- Review fallback ---
    max_review_failures: int = 3        # after N review LLM failures, reject (don't auto-approve)
    # --- Escalation ---
    escalation_timeout_minutes: int = 60  # auto-retry transient failures after N min with no human response
    auto_retry_on_timeout: bool = True    # when escalation times out, retry (True) or halt (False)

    @classmethod
    def from_dict(cls, d: dict) -> "RepairConfig":
        return cls(
            enabled=d.get("enabled", False),
            max_attempts=int(d.get("max_attempts", 2)),
            launcher=d.get("launcher", "local"),
            claude_binary=d.get("claude_binary", "auto"),
            model=d.get("model", "claude-opus-4-6"),
            effort=d.get("effort", "max"),
            budget_usd=float(d.get("budget_usd", 10.0)),
            timeout_seconds=int(d.get("timeout_seconds", 600)),
            allowed_actions=d.get("allowed_actions", [
                "edit_code", "fix_config",
                "generate_missing_artifacts", "install_dependencies",
            ]),
            retry_delay_seconds=int(d.get("retry_delay_seconds", 10)),
            two_phase=d.get("two_phase", True),
            plan_model=d.get("plan_model"),
            plan_effort=d.get("plan_effort"),
            plan_budget_usd=float(d.get("plan_budget_usd", 5.0)),
            plan_timeout_seconds=int(d.get("plan_timeout_seconds", 300)),
            review_model=d.get("review_model", "claude-opus-4-6"),
            review_temperature=float(d.get("review_temperature", 0.2)),
            min_review_score=int(d.get("min_review_score", 7)),
            backoff_base_seconds=int(d.get("backoff_base_seconds", 60)),
            backoff_max_seconds=int(d.get("backoff_max_seconds", 900)),
            repairing_timeout_seconds=int(d.get("repairing_timeout_seconds", 3600)),
            max_review_failures=int(d.get("max_review_failures", 3)),
            escalation_timeout_minutes=int(d.get("escalation_timeout_minutes", 60)),
            auto_retry_on_timeout=d.get("auto_retry_on_timeout", True),
        )


@dataclass
class PlanningConfig:
    """Configuration for dynamic campaign planning via multi-model counsel."""
    enabled: bool = False
    base_task_file: str = ""           # task file for the discovery stage
    max_stages: int = 6                # hard cap on non-paper stages
    max_parallel: int = 2              # max simultaneously-running stages
    counsel_models: Optional[List[dict]] = None  # override default counsel models
    human_review: bool = True          # pause for human approval before executing
    planning_budget_usd: float = 5.0   # budget for the planning counsel debate
    planning_timeout_seconds: int = 600
    stage_type_constraints: dict = field(default_factory=dict)  # override args per type

    @classmethod
    def from_dict(cls, d: dict) -> "PlanningConfig":
        return cls(
            enabled=d.get("enabled", False),
            base_task_file=d.get("base_task_file", ""),
            max_stages=int(d.get("max_stages", 6)),
            max_parallel=int(d.get("max_parallel", 2)),
            counsel_models=d.get("counsel_models"),
            human_review=d.get("human_review", True),
            planning_budget_usd=float(d.get("planning_budget_usd", 5.0)),
            planning_timeout_seconds=int(d.get("planning_timeout_seconds", 600)),
            stage_type_constraints=d.get("stage_type_constraints", {}),
        )


@dataclass
class CampaignSpec:
    name: str
    workspace_root: str
    stages: List[Stage]
    notification: NotificationConfig
    repair: RepairConfig = field(default_factory=RepairConfig)
    heartbeat_interval_minutes: int = 30
    # --- Circuit breakers ---
    max_idle_ticks: int = 6             # auto-exit after N consecutive no-op ticks
    max_campaign_hours: float = 0       # 0 = unlimited; hard wall-time for entire campaign
    counsel_model_timeout_seconds: int = 600  # per-model timeout for counsel sandbox agents
    planning: Optional[PlanningConfig] = None  # dynamic campaign planning config
    spec_file: str = ""  # absolute path to the campaign YAML that created this spec
    budget_usd: float = 0.0  # 0 = unlimited; campaign-wide budget cap in USD

    def stage(self, stage_id: str) -> Optional[Stage]:
        for s in self.stages:
            if s.id == stage_id:
                return s
        return None

    def stage_ids(self) -> List[str]:
        return [s.id for s in self.stages]


def load_spec(path: str) -> CampaignSpec:
    """Load and validate a campaign.yaml file. Raises ValueError on bad config."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    abs_path = os.path.abspath(path)
    yaml_dir = os.path.dirname(abs_path)
    name = raw.get("name") or os.path.basename(os.path.dirname(abs_path))
    workspace_root = raw.get("workspace_root", "results/campaign")
    # Resolve relative workspace_root relative to the YAML file's directory
    if not os.path.isabs(workspace_root):
        workspace_root = os.path.join(yaml_dir, workspace_root)

    # Parse planning config (if present)
    planning_raw = raw.get("planning", {})
    planning = PlanningConfig.from_dict(planning_raw) if planning_raw else None

    stages_raw = raw.get("stages", [])

    # When dynamic planning is enabled, stages can be empty (they'll be generated)
    if not stages_raw and not (planning and planning.enabled):
        raise ValueError(f"campaign.yaml at {path} has no stages defined.")

    stages = [Stage.from_dict(s) for s in stages_raw]

    # Resolve relative task_file paths relative to the YAML directory
    for s in stages:
        if s.task_file and not os.path.isabs(s.task_file):
            s.task_file = os.path.join(yaml_dir, s.task_file)
        if s.launcher_script and not os.path.isabs(s.launcher_script):
            s.launcher_script = os.path.join(yaml_dir, s.launcher_script)

    # Auto-inject discovery_plan and planning_counsel stages when planning is enabled
    if planning and planning.enabled:
        if not planning.base_task_file:
            raise ValueError(
                "planning.base_task_file is required when planning.enabled is true."
            )
        existing_ids = {s.id for s in stages}

        if "discovery_plan" not in existing_ids:
            discovery_stage = Stage(
                id="discovery_plan",
                task_file=planning.base_task_file,
                args=[
                    "--pipeline-mode", "full_research",
                    "--enable-math-agents", "--enable-counsel",
                    "--enable-tree-search",
                ],
                depends_on=[],
                context_from=[],
                memory_dirs=["paper_workspace/"],
                success_artifacts={
                    "required": [
                        "paper_workspace/research_plan.tex",
                        "paper_workspace/track_decomposition.json",
                    ],
                    "optional": [
                        "paper_workspace/research_plan_tasks.json",
                    ],
                },
            )
            stages.insert(0, discovery_stage)

        if "planning_counsel" not in existing_ids:
            planning_counsel_stage = Stage(
                id="planning_counsel",
                task_file=planning.base_task_file,
                args=[],
                depends_on=["discovery_plan"],
                context_from=["discovery_plan"],
                memory_dirs=[],
                success_artifacts={
                    "required": ["campaign_plan.json"],
                    "optional": ["campaign_plan_review.md"],
                },
                launcher_script="scripts/run_campaign_planner.py",
            )
            # Insert after discovery_plan
            discovery_idx = next(
                (i for i, s in enumerate(stages) if s.id == "discovery_plan"), 0
            )
            stages.insert(discovery_idx + 1, planning_counsel_stage)

    # Validate dependency references
    ids = {s.id for s in stages}
    for s in stages:
        for dep in s.depends_on:
            if dep not in ids:
                raise ValueError(f"Stage '{s.id}' depends_on unknown stage '{dep}'.")
        for ctx in s.context_from:
            if ctx not in ids:
                raise ValueError(f"Stage '{s.id}' context_from unknown stage '{ctx}'.")

    notification = NotificationConfig.from_dict(raw.get("notification") or {})
    repair = RepairConfig.from_dict(raw.get("repair") or {})

    return CampaignSpec(
        name=name,
        workspace_root=workspace_root,
        stages=stages,
        notification=notification,
        repair=repair,
        heartbeat_interval_minutes=int(raw.get("heartbeat_interval_minutes", 30)),
        max_idle_ticks=int(raw.get("max_idle_ticks", 6)),
        max_campaign_hours=float(raw.get("max_campaign_hours", 0)),
        counsel_model_timeout_seconds=int(raw.get("counsel_model_timeout_seconds", 600)),
        planning=planning,
        spec_file=abs_path,
        budget_usd=float(raw.get("budget_usd", 0)),
    )


def _expand_env(value: Optional[str]) -> Optional[str]:
    """Expand ${VAR_NAME} environment variable references in config strings."""
    if not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        return os.environ.get(var)
    return value
