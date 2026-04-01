"""Configuration dataclasses for build_research_graph_v2.

Bundles graph-construction parameters into a serializable config object.
CLI-agent-only branch: agents run via local CLI tools, not API calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class ArtifactEnforcementConfig:
    """Controls which artefact / quality gates are active."""

    enforce_paper_artifacts: bool = False
    enforce_editorial_artifacts: bool = False
    require_pdf: bool = False
    require_experiment_plan: bool = False
    lit_review_max_attempts: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enforce_paper_artifacts": self.enforce_paper_artifacts,
            "enforce_editorial_artifacts": self.enforce_editorial_artifacts,
            "require_pdf": self.require_pdf,
            "require_experiment_plan": self.require_experiment_plan,
            "lit_review_max_attempts": self.lit_review_max_attempts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactEnforcementConfig":
        return cls(
            enforce_paper_artifacts=data.get("enforce_paper_artifacts", False),
            enforce_editorial_artifacts=data.get("enforce_editorial_artifacts", False),
            require_pdf=data.get("require_pdf", False),
            require_experiment_plan=data.get("require_experiment_plan", False),
            lit_review_max_attempts=data.get("lit_review_max_attempts", 2),
        )


@dataclass
class PersonaCouncilConfig:
    """Persona-council debate settings."""

    specs: Optional[List[dict]] = None
    debate_rounds: int = 3
    synthesis_model: str = "claude-opus-4-6"
    max_post_vote_retries: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specs": self.specs,
            "debate_rounds": self.debate_rounds,
            "synthesis_model": self.synthesis_model,
            "max_post_vote_retries": self.max_post_vote_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaCouncilConfig":
        return cls(
            specs=data.get("specs"),
            debate_rounds=data.get("debate_rounds", 3),
            synthesis_model=data.get("synthesis_model", "claude-opus-4-6"),
            max_post_vote_retries=data.get("max_post_vote_retries", 1),
        )


@dataclass
class DualityCheckConfig:
    """Duality-check gate settings."""

    enabled: bool = True
    model: str = "claude-opus-4-6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DualityCheckConfig":
        return cls(
            enabled=data.get("enabled", True),
            model=data.get("model", "claude-opus-4-6"),
        )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

# Fields that hold runtime objects and cannot roundtrip through JSON.
_RUNTIME_FIELDS = frozenset(
    {"cli_backend_registry", "checkpointer"}
)


@dataclass
class ResearchGraphConfig:
    """All knobs for ``build_research_graph_v2``.

    CLI-agent-only: agents run via local CLI tools (Claude Code, Codex,
    Gemini CLI) as subprocesses.  The ``cli_backend_registry`` maps agent
    names to CLI backend specs.
    """

    # -- required ----------------------------------------------------------
    cli_backend_registry: Any = field(repr=False)
    workspace_dir: str = ""

    # -- pipeline flags ----------------------------------------------------
    pipeline_mode: str = "default"
    enable_math_agents: bool = False
    enable_milestone_gates: bool = False
    adversarial_verification: bool = False

    # -- validation / iteration limits -------------------------------------
    min_review_score: int = 8
    followup_max_iterations: int = 3
    manager_max_steps: int = 50

    # -- execution ---------------------------------------------------------
    authorized_imports: Optional[List[str]] = None

    # -- model configuration -----------------------------------------------
    summary_model_id: Optional[str] = "claude-opus-4-6"

    # -- runtime objects (excluded from serialization) ---------------------
    checkpointer: Any = field(default=None, repr=False)

    # -- composed sub-configs ----------------------------------------------
    tree_search: Any = None  # Optional[TreeSearchConfig]
    persona_council: PersonaCouncilConfig = field(
        default_factory=PersonaCouncilConfig
    )
    duality_check: DualityCheckConfig = field(default_factory=DualityCheckConfig)
    artifacts: ArtifactEnforcementConfig = field(
        default_factory=ArtifactEnforcementConfig
    )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict (runtime objects excluded)."""
        d: Dict[str, Any] = {
            "workspace_dir": self.workspace_dir,
            "pipeline_mode": self.pipeline_mode,
            "enable_math_agents": self.enable_math_agents,
            "enable_milestone_gates": self.enable_milestone_gates,
            "adversarial_verification": self.adversarial_verification,
            "min_review_score": self.min_review_score,
            "followup_max_iterations": self.followup_max_iterations,
            "manager_max_steps": self.manager_max_steps,
            "authorized_imports": self.authorized_imports,
            "summary_model_id": self.summary_model_id,
            "persona_council": self.persona_council.to_dict(),
            "duality_check": self.duality_check.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }
        if self.tree_search is not None:
            d["tree_search"] = self.tree_search.to_dict()
        return d

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        cli_backend_registry: Any = None,
        checkpointer: Any = None,
    ) -> "ResearchGraphConfig":
        """Reconstruct from a serialized dict plus runtime objects."""
        tree_data = data.get("tree_search")
        tree_cfg = None
        if tree_data is not None:
            # Lazy import to avoid pulling in tree_search when unused.
            from consortium.tree_search.tree_state import TreeSearchConfig
            tree_cfg = TreeSearchConfig.from_dict(tree_data)

        return cls(
            cli_backend_registry=cli_backend_registry,
            workspace_dir=data.get("workspace_dir", ""),
            pipeline_mode=data.get("pipeline_mode", "default"),
            enable_math_agents=data.get("enable_math_agents", False),
            enable_milestone_gates=data.get("enable_milestone_gates", False),
            adversarial_verification=data.get("adversarial_verification", False),
            min_review_score=data.get("min_review_score", 8),
            followup_max_iterations=data.get("followup_max_iterations", 3),
            manager_max_steps=data.get("manager_max_steps", 50),
            authorized_imports=data.get("authorized_imports"),
            summary_model_id=data.get("summary_model_id", "claude-opus-4-6"),
            checkpointer=checkpointer,
            tree_search=tree_cfg,
            persona_council=PersonaCouncilConfig.from_dict(
                data.get("persona_council", {})
            ),
            duality_check=DualityCheckConfig.from_dict(
                data.get("duality_check", {})
            ),
            artifacts=ArtifactEnforcementConfig.from_dict(
                data.get("artifacts", {})
            ),
        )
