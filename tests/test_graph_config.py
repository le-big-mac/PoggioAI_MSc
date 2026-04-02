"""Tests for consortium.graph_config in CLI-agent mode."""

import json

from consortium.graph_config import (
    ArtifactEnforcementConfig,
    DualityCheckConfig,
    PersonaCouncilConfig,
    ResearchGraphConfig,
)
from consortium.utils import CLIBackendRegistry, CLIBackendSpec


def _registry():
    return CLIBackendRegistry(
        default=CLIBackendSpec(backend="claude", model="claude-opus-4-6")
    )


def _minimal_config(**overrides):
    kwargs = {"cli_backend_registry": _registry(), "workspace_dir": "/tmp/ws"}
    kwargs.update(overrides)
    return ResearchGraphConfig(**kwargs)


class TestDefaults:
    def test_top_level_defaults(self):
        cfg = _minimal_config()
        assert cfg.pipeline_mode == "default"
        assert cfg.enable_math_agents is False
        assert cfg.enable_milestone_gates is False
        assert cfg.adversarial_verification is False
        assert cfg.min_review_score == 8
        assert cfg.followup_max_iterations == 3
        assert cfg.manager_max_steps == 50
        assert cfg.authorized_imports is None
        assert cfg.summary_model_id == "claude-opus-4-6"
        assert cfg.checkpointer is None
        assert cfg.tree_search is None

    def test_artifact_defaults(self):
        a = _minimal_config().artifacts
        assert a.enforce_paper_artifacts is False
        assert a.enforce_editorial_artifacts is False
        assert a.require_pdf is False
        assert a.require_experiment_plan is False
        assert a.lit_review_max_attempts == 2

    def test_persona_council_defaults(self):
        pc = _minimal_config().persona_council
        assert pc.specs is None
        assert pc.debate_rounds == 3
        assert pc.synthesis_model == "claude-opus-4-6"
        assert pc.max_post_vote_retries == 1

    def test_duality_check_defaults(self):
        dc = _minimal_config().duality_check
        assert dc.enabled is True
        assert dc.model == "claude-opus-4-6"


class TestSerialization:
    def test_roundtrip_minimal(self):
        cfg = _minimal_config()
        d = cfg.to_dict()
        json.dumps(d)

        restored = ResearchGraphConfig.from_dict(
            d, cli_backend_registry=_registry()
        )
        assert restored.workspace_dir == cfg.workspace_dir
        assert restored.pipeline_mode == cfg.pipeline_mode
        assert restored.min_review_score == cfg.min_review_score

    def test_roundtrip_full(self):
        cfg = ResearchGraphConfig(
            cli_backend_registry=_registry(),
            workspace_dir="/tmp/full",
            pipeline_mode="full_research",
            enable_math_agents=True,
            enable_milestone_gates=True,
            adversarial_verification=True,
            min_review_score=6,
            followup_max_iterations=5,
            manager_max_steps=100,
            authorized_imports=["numpy", "scipy"],
            summary_model_id="gpt-5.4",
            persona_council=PersonaCouncilConfig(
                specs=[{"role": "mathematician"}],
                debate_rounds=5,
                synthesis_model="gpt-5.4",
                max_post_vote_retries=2,
            ),
            duality_check=DualityCheckConfig(enabled=False, model="gpt-5.4"),
            artifacts=ArtifactEnforcementConfig(
                enforce_paper_artifacts=True,
                enforce_editorial_artifacts=True,
                require_pdf=True,
                require_experiment_plan=True,
                lit_review_max_attempts=4,
            ),
        )
        d = cfg.to_dict()
        json.dumps(d)

        restored = ResearchGraphConfig.from_dict(
            d, cli_backend_registry=_registry(), checkpointer="ckpt"
        )
        assert restored.workspace_dir == "/tmp/full"
        assert restored.pipeline_mode == "full_research"
        assert restored.enable_math_agents is True
        assert restored.min_review_score == 6
        assert restored.authorized_imports == ["numpy", "scipy"]
        assert restored.summary_model_id == "gpt-5.4"
        assert restored.checkpointer == "ckpt"
        assert restored.persona_council.debate_rounds == 5
        assert restored.persona_council.specs == [{"role": "mathematician"}]
        assert restored.duality_check.enabled is False
        assert restored.artifacts.require_pdf is True
        assert restored.artifacts.lit_review_max_attempts == 4


class TestRuntimeExclusion:
    def test_runtime_fields_absent_from_dict(self):
        cfg = ResearchGraphConfig(
            cli_backend_registry=_registry(),
            workspace_dir="/tmp/ws",
            checkpointer="ckpt",
        )
        d = cfg.to_dict()
        for key in ("cli_backend_registry", "checkpointer"):
            assert key not in d


class TestNestedStructure:
    def test_sub_configs_are_nested_dicts(self):
        d = _minimal_config().to_dict()
        assert isinstance(d["persona_council"], dict)
        assert isinstance(d["duality_check"], dict)
        assert isinstance(d["artifacts"], dict)
        assert "tree_search" not in d

    def test_tree_search_included_when_set(self):
        class FakeTreeConfig:
            def to_dict(self):
                return {"enabled": True, "max_depth": 4}

        d = _minimal_config(tree_search=FakeTreeConfig()).to_dict()
        assert d["tree_search"] == {"enabled": True, "max_depth": 4}
