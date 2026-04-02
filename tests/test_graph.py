"""
Tests for consortium/graph.py — pipeline stage construction and routing.
"""

import pytest


class TestBuildPipelineStagesV2:
    def test_quick_pipeline_starts_with_quick_develop(self):
        from consortium.graph import build_pipeline_stages_quick
        stages = build_pipeline_stages_quick()
        assert stages[0] == "quick_develop"

    def test_base_pipeline_stage_count(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=False)
        # PRE_TRACK(6) + EXPERIMENT(5) + POST_TRACK(5) = 16
        assert len(stages) == 16

    def test_base_pipeline_starts_with_persona_council(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=False)
        assert stages[0] == "persona_council"

    def test_base_pipeline_ends_with_reviewer(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=False)
        assert stages[-1] == "reviewer_agent"

    def test_math_pipeline_has_twenty_stages(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=True)
        # PRE_TRACK(6) + MATH(6) + EXPERIMENT(5) + POST_TRACK(5) = 22
        assert len(stages) == 22

    def test_math_stages_before_experiment_stages(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=True)
        math_lit_idx = stages.index("math_literature_agent")
        exp_lit_idx = stages.index("experiment_literature_agent")
        assert math_lit_idx < exp_lit_idx, "Math stages should precede experiment track"

    def test_math_stages_after_pre_track(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=True)
        goals_idx = stages.index("formalize_goals_agent")
        math_lit_idx = stages.index("math_literature_agent")
        assert math_lit_idx > goals_idx, "Math stages should follow pre-track stages"

    def test_math_stages_before_post_track(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=True)
        rp_idx = stages.index("resource_preparation_agent")
        assert stages.index("proof_transcription_agent") < rp_idx

    def test_all_base_stages_present_in_math_pipeline(self):
        from consortium.graph import build_pipeline_stages_v2
        base = build_pipeline_stages_v2(enable_math_agents=False)
        math = build_pipeline_stages_v2(enable_math_agents=True)
        for stage in base:
            assert stage in math, f"Base stage '{stage}' missing from math pipeline"

    def test_no_duplicate_stages(self):
        from consortium.graph import build_pipeline_stages_v2
        for enable_math in [False, True]:
            stages = build_pipeline_stages_v2(enable_math)
            assert len(stages) == len(set(stages)), \
                f"Duplicate stages found in pipeline (math={enable_math}): {stages}"

    def test_pre_track_stage_order(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=False)
        assert stages[0] == "persona_council"
        assert stages[1] == "literature_review_agent"
        assert stages[2] == "brainstorm_agent"
        assert stages[3] == "formalize_goals_entry"
        assert stages[4] == "formalize_goals_agent"
        assert stages[5] == "research_plan_writeup_agent"

    def test_theory_scope_excludes_experiment_stages(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=True, execution_scope="theory")
        assert "math_literature_agent" in stages
        assert "experiment_literature_agent" not in stages

    def test_experiment_scope_excludes_math_stages(self):
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=True, execution_scope="experiment")
        assert "experiment_literature_agent" in stages
        assert "math_literature_agent" not in stages


class TestStageAliases:
    def test_canonical_stage_name_council(self):
        from consortium.runner import _STAGE_ALIASES
        assert _STAGE_ALIASES.get("council") == "persona_council"

    def test_canonical_stage_name_brainstorm(self):
        from consortium.runner import _STAGE_ALIASES
        assert _STAGE_ALIASES.get("brainstorm") == "brainstorm_agent"

    def test_canonical_stage_name_writeup(self):
        from consortium.runner import _STAGE_ALIASES
        assert _STAGE_ALIASES.get("writeup") == "writeup_agent"

    def test_canonical_stage_name_math_prover(self):
        from consortium.runner import _STAGE_ALIASES
        assert _STAGE_ALIASES.get("math_prover") == "math_prover_agent"

    def test_resolve_start_stage_index_valid(self):
        from consortium.runner import _resolve_start_stage_index
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=False)
        idx = _resolve_start_stage_index("writeup", stages)
        assert stages[idx] == "writeup_agent"

    def test_resolve_start_stage_index_invalid_raises(self):
        from consortium.runner import _resolve_start_stage_index
        from consortium.graph import build_pipeline_stages_v2
        stages = build_pipeline_stages_v2(enable_math_agents=False)
        with pytest.raises(ValueError, match="Unknown --start-from-stage"):
            _resolve_start_stage_index("nonexistent_stage", stages)


class TestScopedRouting:
    def test_track_router_theory_scope_only_sends_theory(self):
        from consortium.graph import track_router

        sends = track_router(
            {
                "execution_scope": "theory",
                "math_enabled": True,
                "track_decomposition": {
                    "recommended_track": "both",
                    "theory_questions": ["Q1"],
                    "empirical_questions": ["E1"],
                },
            }
        )

        assert len(sends) == 1
        assert sends[0].node == "theory_track"

    def test_track_router_experiment_scope_only_sends_experiment(self):
        from consortium.graph import track_router

        sends = track_router(
            {
                "execution_scope": "experiment",
                "math_enabled": True,
                "track_decomposition": {
                    "recommended_track": "both",
                    "theory_questions": ["Q1"],
                    "empirical_questions": ["E1"],
                },
            }
        )

        assert len(sends) == 1
        assert sends[0].node == "experiment_track"

    def test_validation_router_respects_theory_scope(self, monkeypatch):
        from consortium.graph import validation_router

        monkeypatch.setattr("consortium.graph.classify_review_fixes", lambda workspace: "experiment")
        route = validation_router(
            {
                "execution_scope": "theory",
                "finished": False,
                "workspace_dir": ".",
                "validation_retry_count": 0,
                "max_validation_retries": 3,
            }
        )

        assert route == "writeup_agent"


class TestResumeEntrySelection:
    def test_choose_quick_entry_stage_uses_requested_stage(self):
        from consortium.graph import _choose_quick_entry_stage

        assert _choose_quick_entry_stage("literature_review_agent") == "literature_review_agent"

    def test_choose_quick_entry_stage_defaults_to_quick_develop(self):
        from consortium.graph import _choose_quick_entry_stage

        assert _choose_quick_entry_stage(None) == "quick_develop"

    def test_choose_v2_entry_stage_routes_theory_track_stages_via_track_node(self):
        from consortium.graph import _choose_v2_entry_stage

        assert _choose_v2_entry_stage("math_prover_agent", enable_math_agents=True) == "theory_track"

    def test_choose_v2_entry_stage_routes_experiment_track_stages_via_track_node(self):
        from consortium.graph import _choose_v2_entry_stage

        assert _choose_v2_entry_stage("experiment_verification_agent", enable_math_agents=True) == "experiment_track"

    def test_choose_v2_entry_stage_uses_top_level_stage_directly(self):
        from consortium.graph import _choose_v2_entry_stage

        assert _choose_v2_entry_stage("writeup_agent", enable_math_agents=True) == "writeup_agent"
