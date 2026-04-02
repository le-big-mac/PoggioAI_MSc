"""
Core run logic for the consortium multi-agent system.

CLI-agent-only branch: all agents run as local CLI tool subprocesses
(Claude Code, Codex, Gemini CLI) instead of making API calls.
"""

import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

from dotenv import load_dotenv

from .args import parse_arguments
from .config import load_llm_config
from .interaction.callback_tools import setup_user_input_socket
from .interaction.http_steering import add_http_steering
# Legacy prereqs import removed — tectonic check is inline in main()
from .supervision import sanitize_result_payload
from .graph import build_pipeline_stages_v2, get_default_checkpointer
from .utils import create_cli_backend_registry

load_dotenv(override=True)


_DEFAULT_TASK = (
    "Investigate whether training small language models with multiple paraphrased responses "
    "reduces hallucinations and improves response quality. Generate research ideas exploring: "
    "(1) Fine-tuning GPT-2 Small (124M) on instruction-following datasets with single vs multiple "
    "response variants, using small subsets (5K samples) for rapid experimentation, "
    "(2) Comparing response diversity and factual accuracy between single-response and multi-response "
    "training regimes, (3) Testing whether exposure to diverse correct answers during training acts "
    "as implicit regularization against repetitive or factually incorrect outputs, "
    "(4) Measuring trade-offs between response quality, diversity, and training efficiency with "
    "automated metrics. Use small-scale datasets like Alpaca-5K with automated paraphrase generation. "
    "Focus on fast automated evaluation including response diversity via Self-BLEU, factual consistency "
    "via rule-based checks, and response quality via ROUGE scores. Target achieving measurable "
    "improvements in response diversity while maintaining quality, with each experimental run "
    "completing in under 1 hour."
)

_CONTINUATION_TASK = (
    "Continue working on a previous research task. Please meticulously analyze the existing "
    "files and then plan how to call the relevant agents to further progress the research task "
    "and deliver better research outputs."
)

_STAGE_ALIASES = {
    "literature": "literature_review_agent",
    "litreview": "literature_review_agent",
    "literature_review": "literature_review_agent",
    "literature_review_agent": "literature_review_agent",
    "experiment": "experimentation_agent",
    "experimentation": "experimentation_agent",
    "experimentation_agent": "experimentation_agent",
    "analysis": "results_analysis_agent",
    "results_analysis": "results_analysis_agent",
    "results_analysis_agent": "results_analysis_agent",
    "math_literature": "math_literature_agent",
    "math_literature_agent": "math_literature_agent",
    "math_proposer": "math_proposer_agent",
    "math_proposer_agent": "math_proposer_agent",
    "math_prover": "math_prover_agent",
    "math_prover_agent": "math_prover_agent",
    "math_rigorous_verifier": "math_rigorous_verifier_agent",
    "math_rigorous_verifier_agent": "math_rigorous_verifier_agent",
    "math_empirical_verifier": "math_empirical_verifier_agent",
    "math_empirical_verifier_agent": "math_empirical_verifier_agent",
    "proof_transcription": "proof_transcription_agent",
    "proof_transcription_agent": "proof_transcription_agent",
    "resources": "resource_preparation_agent",
    "resource_preparation": "resource_preparation_agent",
    "resource_preparation_agent": "resource_preparation_agent",
    "writeup": "writeup_agent",
    "writeup_agent": "writeup_agent",
    "proofread": "proofreading_agent",
    "proofreading": "proofreading_agent",
    "proofreading_agent": "proofreading_agent",
    "review": "reviewer_agent",
    "reviewer": "reviewer_agent",
    "reviewer_agent": "reviewer_agent",
    "persona_council": "persona_council",
    "council": "persona_council",
    "brainstorm": "brainstorm_agent",
    "brainstorm_agent": "brainstorm_agent",
    "formalize_goals": "formalize_goals_agent",
    "formalize_goals_agent": "formalize_goals_agent",
    "goals": "formalize_goals_agent",
    "research_plan_writeup": "research_plan_writeup_agent",
    "research_plan_writeup_agent": "research_plan_writeup_agent",
    "plan_writeup": "research_plan_writeup_agent",
    "formalize_results": "formalize_results_agent",
    "formalize_results_agent": "formalize_results_agent",
}


def _setup_optional_tracing():
    enabled = os.getenv("CONSORTIUM_ENABLE_TRACING", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        print("Tracing disabled (set CONSORTIUM_ENABLE_TRACING=1 to enable).")
        return
    if os.getenv("LANGCHAIN_TRACING_V2"):
        print("LangSmith tracing enabled.")
    else:
        print("Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable LangSmith tracing.")


def _filter_installed_imports(import_names):
    available, missing = [], []
    for name in import_names:
        root = name.split(".")[0]
        if importlib.util.find_spec(root) is not None:
            available.append(name)
        else:
            missing.append(name)
    if missing:
        print("Skipping unavailable authorized imports: " + ", ".join(sorted(set(missing))))
    return available


def _build_required_artifacts(args, enforce_paper_artifacts, enforce_editorial_artifacts,
                               require_experiment_plan):
    artifacts = []
    if enforce_paper_artifacts:
        artifacts.append("final_paper.tex")
        if require_experiment_plan:
            artifacts.append("experiments_to_run_later.md")
        if args.require_pdf:
            artifacts.append("final_paper.pdf")
        if enforce_editorial_artifacts:
            artifacts.extend([
                "paper_workspace/author_style_guide.md",
                "paper_workspace/intro_skeleton.tex",
                "paper_workspace/style_macros.tex",
                "paper_workspace/reader_contract.json",
                "paper_workspace/editorial_contract.md",
                "paper_workspace/theorem_map.json",
                "paper_workspace/revision_log.md",
                "paper_workspace/copyedit_report.tex",
                "paper_workspace/review_report.tex",
                "paper_workspace/review_verdict.json",
            ])
            if args.enable_math_agents:
                artifacts.append("paper_workspace/claim_traceability.json")
    return artifacts


def _canonical_stage_name(stage_name: str) -> str:
    normalized = stage_name.strip().lower().replace("-", "_")
    return _STAGE_ALIASES.get(normalized, normalized)


def _resolve_start_stage_index(stage_name: str, pipeline_stages: list[str]) -> int:
    canonical = _canonical_stage_name(stage_name)
    if canonical not in pipeline_stages:
        valid = ", ".join(pipeline_stages)
        raise ValueError(
            f"Unknown --start-from-stage '{stage_name}' (resolved: '{canonical}'). "
            f"Valid stages for this run: {valid}"
        )
    return pipeline_stages.index(canonical)


def _list_runs(results_dir: str = "results") -> None:
    """Print a summary table of past runs in the results directory."""
    if not os.path.isdir(results_dir):
        print(f"No results directory found at '{results_dir}'.")
        return

    entries = sorted(
        (d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))),
        reverse=True,
    )
    if not entries:
        print("No past runs found in results/.")
        return

    header = f"{'Workspace':<45} {'Task (truncated)':<45} {'Time':>8}  Status"
    print(header)
    print("-" * len(header))

    for name in entries:
        ws = os.path.join(results_dir, name)
        # Budget / time
        time_str = "?"
        budget_path = os.path.join(ws, "cli_budget_state.json")
        if os.path.exists(budget_path):
            try:
                with open(budget_path) as f:
                    bd = json.load(f)
                total = bd.get("total_seconds")
                if total is not None:
                    time_str = f"{float(total):.0f}s"
            except Exception:
                pass
        # Status
        status_str = "unknown"
        status_path = os.path.join(ws, "STATUS.txt")
        if os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    status_str = f.read().strip()[:20]
            except Exception:
                pass
        # Task
        task_str = ""
        summary_path = os.path.join(ws, "run_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    sm = json.load(f)
                task_str = sm.get("task", "")[:43]
            except Exception:
                pass
        print(f"{name:<45} {task_str:<45} {time_str:>8}  {status_str}")


def _write_experiment_metadata(workspace_dir: str, args, task: str,
                                cli_backend: str, cli_model: str) -> None:
    """Write experiment_metadata.json to the workspace at run start."""
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass

    git_dirty = False
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        git_dirty = bool(result.strip())
    except Exception:
        pass

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "mode": "cli_agent",
        "cli_backend": cli_backend,
        "cli_model": cli_model,
        "task_preview": task,
        "cli_args": {
            "enable_math_agents": getattr(args, "enable_math_agents", False),
            "output_format": getattr(args, "output_format", "latex"),
            "enforce_paper_artifacts": getattr(args, "enforce_paper_artifacts", False),
            "min_review_score": getattr(args, "min_review_score", 8),
        },
    }
    out_path = os.path.join(workspace_dir, "experiment_metadata.json")
    try:
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass


def _write_run_summary(workspace_dir: str, task: str, cli_backend: str,
                        cli_model: str, start_time: datetime,
                        stages_completed: list[str]) -> None:
    """Write run_summary.json to the workspace at run end."""
    duration_s = (datetime.now() - start_time).total_seconds()

    # CLI budget summary
    cli_budget = None
    budget_path = os.path.join(workspace_dir, "cli_budget_state.json")
    if os.path.exists(budget_path):
        try:
            with open(budget_path) as f:
                cli_budget = json.load(f)
        except Exception:
            pass

    # Detect final paper path
    paper_path = None
    for candidate in ["final_paper.pdf", "final_paper.tex", "final_paper.md"]:
        p = os.path.join(workspace_dir, candidate)
        if os.path.exists(p):
            paper_path = candidate
            break

    summary = {
        "task": task,
        "mode": "cli_agent",
        "cli_backend": cli_backend,
        "cli_model": cli_model,
        "started_at": start_time.isoformat(),
        "duration_seconds": round(duration_s, 1),
        "stages_completed": stages_completed,
        "cli_budget": cli_budget,
        "final_paper": paper_path,
        "workspace": workspace_dir,
    }
    out_path = os.path.join(workspace_dir, "run_summary.json")
    try:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Run summary written: {out_path}")
    except Exception:
        pass


def _validate_cli_tools(backend: str) -> list[str]:
    """Check that the required CLI tool is installed."""
    errors = []
    try:
        subprocess.run([backend, "--version"], capture_output=True, timeout=10)
    except FileNotFoundError:
        errors.append(
            f"CLI tool '{backend}' is not installed or not on PATH.\n"
            f"  Install it before running in CLI agent mode."
        )
    except Exception:
        pass  # --version might not be supported but the binary exists
    return errors


def _collect_required_cli_backends(cli_registry) -> list[str]:
    """Return the sorted set of CLI backends referenced by the current config."""
    backends = {cli_registry.default_spec.backend}
    for spec in getattr(cli_registry, "_overrides", {}).values():
        backends.add(spec.backend)
    return sorted(b for b in backends if b)


def main():
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_start_time = datetime.now()
    quick_pass = getattr(args, "quick_pass", False)
    effective_pipeline_mode = "quick" if quick_pass else "full_research"

    # --list-runs: print past workspaces and exit
    if getattr(args, "list_runs", False):
        _list_runs()
        return 0

    # --- Logging setup ---
    env_log_default = os.getenv("CONSORTIUM_LOG_TO_FILES", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    enable_file_logs = env_log_default if args.log_to_files is None else args.log_to_files
    if enable_file_logs and not getattr(args, "dry_run", False):
        os.makedirs("logs", exist_ok=True)
        out_path = f"logs/consortium_{timestamp}.out"
        err_path = f"logs/consortium_{timestamp}.err"
        print(f"Redirecting stdout/stderr to: {out_path}, {err_path}")
        sys.stdout = open(out_path, "w", buffering=1)
        sys.stderr = open(err_path, "w", buffering=1)

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    _setup_optional_tracing()

    # --- Deployment mode resolution ---
    from .mode import resolve_mode, load_mode_config, apply_mode_defaults
    mode = resolve_mode(args)
    mode_config = load_mode_config(mode)
    apply_mode_defaults(args, mode_config)
    print(f"[PoggioAI] Running in {mode} mode — {mode_config.get('description', '')}")

    llm_config = load_llm_config() or {}

    if args.pipeline_mode is not None:
        print(
            "Warning: --pipeline-mode is deprecated and ignored. "
            "Running fixed-stage full pipeline mode."
        )

    if args.start_from_stage and not args.resume:
        print("Error: --start-from-stage requires --resume <workspace_dir>.")
        return 1

    # --- CLI backend registry ---
    cli_registry = create_cli_backend_registry(llm_config)
    default_spec = cli_registry.default_spec
    print(f"CLI agent mode — backend: {default_spec.backend}, model: {default_spec.model}")

    # Validate all configured CLI tools are installed
    cli_errors = []
    for backend in _collect_required_cli_backends(cli_registry):
        cli_errors.extend(_validate_cli_tools(backend))
    if cli_errors:
        for err in cli_errors:
            print(f"[ERROR] {err}")
        return 1

    # --- Dry-run mode ---
    if getattr(args, "dry_run", False):
        budget_cfg = llm_config.get("budget", {})
        print("\n[dry-run] Environment checks passed:")
        print(f"  cli_backend     : {default_spec.backend}")
        print(f"  cli_model       : {default_spec.model}")
        print(f"  timeout         : {default_spec.timeout_seconds}s")
        print(f"  max invocations : {budget_cfg.get('max_invocations', 100)}")
        print(f"  math agents     : {getattr(args, 'enable_math_agents', False)}")
        print(f"  output format   : {getattr(args, 'output_format', 'latex')}")
        print("\n[dry-run] All checks passed. Remove --dry-run to start the real run.")
        return 0

    # Set up interrupt socket
    input_queue = None
    if not getattr(args, "no_steering", False):
        input_queue = setup_user_input_socket(args.callback_host, args.callback_port)
        print(f"Interruption port available at {args.callback_host}:{args.callback_port}")
        add_http_steering(input_queue, host=args.callback_host, port=args.callback_port + 1)
    else:
        from queue import Queue
        input_queue = Queue()
        print("[PoggioAI] Steering sockets disabled (--no-steering)")

    print(f"\nCLI Agent Research System Initialized — {default_spec.backend} / {default_spec.model}")

    # --- Workspace setup ---
    if args.resume:
        results_base_dir = os.path.abspath(args.resume)
        if not os.path.exists(results_base_dir):
            print(f"Workspace directory does not exist: {results_base_dir}")
            return 1
        if not os.path.isdir(results_base_dir):
            print(f"Path is not a directory: {results_base_dir}")
            return 1
        task = args.task or _CONTINUATION_TASK
        print(f"Resuming from: {results_base_dir}")
    else:
        results_base_dir = os.path.join("results", f"consortium_{timestamp}")
        os.makedirs(results_base_dir, exist_ok=True)
        task = args.task or _DEFAULT_TASK
        print(f"Created workspace: {results_base_dir}")
        _write_experiment_metadata(
            results_base_dir, args, task,
            default_spec.backend, default_spec.model or "default",
        )

    os.environ["RESULTS_BASE_DIR"] = results_base_dir
    os.environ["CONSORTIUM_OUTPUT_FORMAT"] = getattr(args, "output_format", "latex")

    # --- CLI budget tracker ---
    from .cli_budget import CLIBudgetTracker, set_global_cli_tracker
    budget_cfg = llm_config.get("budget", {})
    cli_tracker = CLIBudgetTracker(
        max_invocations=int(budget_cfg.get("max_invocations", 100)),
        max_wall_clock_seconds=float(budget_cfg.get("max_wall_clock_seconds", 14400)),
        state_dir=results_base_dir,
    )
    set_global_cli_tracker(cli_tracker)

    print(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")

    if quick_pass:
        args.enable_math_agents = False
        args.no_duality_check = True
        args.enforce_paper_artifacts = False
        args.enforce_editorial_artifacts = False
        args.require_pdf = False
        args.require_experiment_plan = False
        from .graph import build_pipeline_stages_quick
        pipeline_stages = build_pipeline_stages_quick()
        print("Pipeline mode: quick-pass (no experiments, proposal artifact only)")
    else:
        print("Pipeline mode: full_research (fixed-stage)")
        if args.enable_math_agents:
            print("Math agent workflow enabled.")
        pipeline_stages = build_pipeline_stages_v2(
            args.enable_math_agents,
            execution_scope=getattr(args, "execution_scope", "all"),
        )
        if getattr(args, "execution_scope", "all") != "all":
            print(f"Execution scope: {args.execution_scope}")
        print("Pipeline version: v2 (persona-council-driven)")
    try:
        start_stage_index = (
            _resolve_start_stage_index(args.start_from_stage, pipeline_stages)
            if args.start_from_stage
            else 0
        )
    except ValueError as e:
        print(str(e))
        return 1

    if args.start_from_stage:
        resolved_stage = pipeline_stages[start_stage_index]
        print(
            f"Stage-based resume requested: start from '{resolved_stage}' "
            f"(index {start_stage_index})."
        )

    # --- Artifact gate setup ---
    auto_enforce = "final_paper" in task.lower() or "experiments_to_run_later" in task.lower()
    enforce_paper_artifacts = args.enforce_paper_artifacts or auto_enforce
    enforce_editorial_artifacts = args.enforce_editorial_artifacts and enforce_paper_artifacts
    require_experiment_plan = args.require_experiment_plan or (
        "experiments_to_run_later" in task.lower()
    )
    required_paper_artifacts = _build_required_artifacts(
        args, enforce_paper_artifacts, enforce_editorial_artifacts, require_experiment_plan
    )

    os.environ["CONSORTIUM_REQUIRE_PDF"] = "1" if args.require_pdf else "0"
    os.environ["CONSORTIUM_ENFORCE_PAPER_ARTIFACTS"] = "1" if enforce_paper_artifacts else "0"
    os.environ["CONSORTIUM_REQUIRE_EXPERIMENT_PLAN"] = "1" if require_experiment_plan else "0"
    os.environ["CONSORTIUM_ENFORCE_EDITORIAL_ARTIFACTS"] = (
        "1" if enforce_editorial_artifacts else "0"
    )

    if enforce_paper_artifacts:
        print("Paper artifact gate: " + ", ".join(required_paper_artifacts))

    # --- LaTeX prereq check (tectonic) ---
    require_latex = enforce_paper_artifacts or args.require_pdf or enforce_editorial_artifacts
    if require_latex:
        import shutil as _shutil
        tectonic_path = _shutil.which("tectonic")
        if not tectonic_path:
            print("Missing LaTeX prerequisite: 'tectonic' not found on PATH.\n"
                  "Install: curl -LsSf https://drop-sh.fullyjustified.net | sh")
            return 1
        os.environ["CONSORTIUM_TECTONIC_PATH"] = tectonic_path
        print(f"LaTeX toolchain: tectonic={tectonic_path}")

    essential_imports = _filter_installed_imports([
        "json", "os", "posixpath", "ntpath", "sys", "datetime", "uuid", "typing",
        "pathlib", "shutil", "textwrap", "functools", "copy", "pickle", "logging",
        "warnings", "gc", "argparse", "configparser", "yaml", "toml", "requests",
        "urllib", "datasets", "transformers", "huggingface_hub", "tokenizers",
        "wandb", "tensorboard", "tqdm", "zipfile", "tarfile",
    ])

    # --- Build tree search config if enabled ---
    tree_search_config = None
    if getattr(args, "enable_tree_search", False):
        from consortium.tree_search.tree_state import TreeSearchConfig
        tree_search_config = TreeSearchConfig(
            enabled=True,
            max_breadth=getattr(args, "tree_max_breadth", 3),
            max_depth=getattr(args, "tree_max_depth", 4),
            max_parallel=getattr(args, "tree_max_parallel", 6),
            pruning_threshold=getattr(args, "tree_pruning_threshold", 0.2),
        )

    try:
        autonomous_mode = getattr(args, "autonomous_mode", True)
        enable_milestone_gates = getattr(args, "enable_milestone_gates", False)
        if autonomous_mode and enable_milestone_gates:
            print("Autonomous mode: milestone gates disabled.")
            enable_milestone_gates = False
        adversarial_verification = getattr(args, "adversarial_verification", False)

        # --- Persona council / duality config ---
        pc_cfg = llm_config.get("persona_council", {})
        dc_cfg = llm_config.get("duality_check", {})

        persona_debate_rounds = getattr(args, "persona_debate_rounds", None)
        if persona_debate_rounds is None:
            persona_debate_rounds = int(pc_cfg.get("max_debate_rounds", 3))

        persona_council_specs = pc_cfg.get("personas") or None
        persona_synthesis_model = pc_cfg.get("synthesis_model", "claude-opus-4-6")
        duality_check_model = dc_cfg.get("model", "claude-opus-4-6")
        enable_duality_check = not getattr(args, "no_duality_check", False)

        from .graph import build_research_graph_v2, build_research_graph_quick
        from .graph_config import (
            ResearchGraphConfig,
            PersonaCouncilConfig,
            DualityCheckConfig,
            ArtifactEnforcementConfig,
        )
        checkpointer = get_default_checkpointer(results_base_dir)
        graph_config = ResearchGraphConfig(
            cli_backend_registry=cli_registry,
            workspace_dir=results_base_dir,
            pipeline_mode=effective_pipeline_mode,
            execution_scope=getattr(args, "execution_scope", "all"),
            start_stage=_canonical_stage_name(args.start_from_stage) if args.start_from_stage else None,
            enable_math_agents=args.enable_math_agents,
            artifacts=ArtifactEnforcementConfig(
                enforce_paper_artifacts=enforce_paper_artifacts,
                enforce_editorial_artifacts=enforce_editorial_artifacts,
                require_pdf=args.require_pdf,
                require_experiment_plan=require_experiment_plan,
            ),
            min_review_score=args.min_review_score,
            followup_max_iterations=args.followup_max_iterations,
            manager_max_steps=args.manager_max_steps or 50,
            authorized_imports=essential_imports,
            checkpointer=checkpointer,
            tree_search=tree_search_config,
            enable_milestone_gates=enable_milestone_gates,
            adversarial_verification=adversarial_verification,
            persona_council=PersonaCouncilConfig(
                specs=persona_council_specs,
                debate_rounds=persona_debate_rounds,
                synthesis_model=persona_synthesis_model,
            ),
            duality_check=DualityCheckConfig(
                enabled=enable_duality_check,
                model=duality_check_model,
            ),
        )
        if quick_pass:
            graph = build_research_graph_quick(graph_config)
            print(f"Pipeline: quick-pass, {persona_debate_rounds} persona debate rounds")
        else:
            graph = build_research_graph_v2(graph_config)
            print(f"Pipeline: {persona_debate_rounds} persona debate rounds, "
                  f"duality_check={'enabled' if enable_duality_check else 'disabled'}")

        if adversarial_verification:
            print("Adversarial verification enabled.")

        if enable_milestone_gates:
            milestone_timeout = getattr(args, "milestone_timeout", 3600)
            print(f"Milestone gates enabled (timeout={milestone_timeout}s).")

        # Build initial state
        initial_state = {
            "messages": [],
            "task": task,
            "workspace_dir": results_base_dir,
            "pipeline_mode": effective_pipeline_mode,
            "execution_scope": getattr(args, "execution_scope", "all"),
            "math_enabled": args.enable_math_agents,
            "enforce_paper_artifacts": enforce_paper_artifacts,
            "enforce_editorial_artifacts": enforce_editorial_artifacts,
            "require_pdf": args.require_pdf,
            "require_experiment_plan": require_experiment_plan,
            "min_review_score": args.min_review_score,
            "followup_max_iterations": args.followup_max_iterations,
            "manager_max_steps": args.manager_max_steps if args.manager_max_steps else 50,
            "pipeline_stages": pipeline_stages,
            "pipeline_stage_index": start_stage_index,
            "current_agent": None,
            "agent_task": None,
            "agent_outputs": {},
            "artifacts": {},
            "iteration_count": 0,
            "followup_iteration": 0,
            "research_cycle": 0,
            "max_research_cycles": args.followup_max_iterations,
            "novelty_check_attempts": 0,
            "rebuttal_iteration": 0,
            "max_rebuttal_iterations": args.max_rebuttal_iterations,
            "validation_results": {},
            "interrupt_instruction": None,
            "theory_track_status": None,
            "experiment_track_status": None,
            "track_decomposition": None,
            "tree_search_enabled": getattr(args, "enable_tree_search", False),
            "tree_state_path": None,
            "active_branch_id": None,
            "milestone_reports": [],
            "human_feedback_history": [],
            "enable_milestone_gates": enable_milestone_gates,
            "milestone_timeout": getattr(args, "milestone_timeout", 3600),
            "intermediate_validation_log": [],
            "finished": False,
            "autonomous_mode": autonomous_mode,
            "research_proposal": None,
            "brainstorm_output": None,
            "brainstorm_history": [],
            "research_goals": None,
            "formalized_results": None,
            "lit_review_feasibility": None,
            "verify_completion_result": None,
            "verify_completion_history": [],
            "duality_check_result": None,
            "lit_review_attempts": 0,
            "brainstorm_cycle": 0,
            "verify_rework_attempts": 0,
            "duality_rework_attempts": 0,
        }

        # Use workspace_dir as thread_id for default resumability.
        if args.start_from_stage:
            canonical_stage = _canonical_stage_name(args.start_from_stage)
            thread_id = f"{results_base_dir}::stage_resume::{canonical_stage}::{timestamp}"
        else:
            thread_id = results_base_dir
        run_config = {"configurable": {"thread_id": thread_id}}

        print(f"\n{'='*50}\nRunning CLI agent research pipeline...\nTask: {task}\n{'='*50}")

        # --- Progress heartbeat watchdog ---
        progress_file = os.path.join(results_base_dir, ".progress_heartbeat")
        _graph_done = threading.Event()

        def _watchdog_writer():
            while not _graph_done.is_set():
                try:
                    tmp = progress_file + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump({"ts": time.time(), "pid": os.getpid()}, f)
                    os.replace(tmp, progress_file)
                except OSError:
                    pass
                _graph_done.wait(120)

        watchdog_thread = threading.Thread(target=_watchdog_writer, daemon=True)
        watchdog_thread.start()

        # --- Hard timeout via SIGALRM ---
        max_run_seconds = getattr(args, "max_run_seconds", None)
        _old_alarm_handler = None
        if max_run_seconds and hasattr(signal, "SIGALRM"):
            def _timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Pipeline exceeded --max-run-seconds={max_run_seconds}. "
                    f"Killing run to prevent infinite hang."
                )
            _old_alarm_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max_run_seconds)
            print(f"Hard timeout set: {max_run_seconds}s")

        try:
            final_state = graph.invoke(initial_state, config=run_config)
        finally:
            _graph_done.set()
            if max_run_seconds and hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if _old_alarm_handler is not None:
                    signal.signal(signal.SIGALRM, _old_alarm_handler)
            try:
                os.remove(progress_file)
            except OSError:
                pass

        result = final_state.get("agent_outputs", {})
        result = sanitize_result_payload(
            result=str(result),
            workspace_dir=results_base_dir,
            required_artifacts=required_paper_artifacts,
        )
        if isinstance(result, dict) and result.get("status") == "incomplete":
            print(
                "Run marked incomplete — missing: "
                + ", ".join(result.get("missing_required_artifacts", []))
            )

        stages_done = []
        if isinstance(final_state, dict):
            idx = final_state.get("pipeline_stage_index", 0)
            stages_done = pipeline_stages[:idx]

        _write_run_summary(
            workspace_dir=results_base_dir,
            task=task,
            cli_backend=default_spec.backend,
            cli_model=default_spec.model or "default",
            start_time=run_start_time,
            stages_completed=stages_done,
        )

        # Print CLI budget summary
        print(f"\nCLI Budget: {cli_tracker.summary}")
        print(f"\n{'='*50}\nTask finished.\n{'='*50}")

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0
