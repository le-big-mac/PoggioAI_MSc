"""LangGraph integration for experiment track tree search.

Mirrors the theory track pattern (graph_integration.py) for experiments.
Instead of proof strategies, generates alternative experiment designs.
Instead of prover → rigorous verifier → empirical verifier, runs
experiment_design → experimentation → experiment_verification.

Provides:
- ``generate_experiment_strategies``: LLM-based experiment design generation
- ``build_experiment_tree_controller``: LangGraph node for experiment branching
- ``build_tree_search_experiment_track``: replacement experiment track subgraph
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from ..cli_completion import cli_completion

from langgraph.graph import END, StateGraph

from consortium.state import ResearchState
from consortium.tree_search.graph_integration import _AgentDescriptor, _run_agent_stage
from consortium.tree_search.node_evaluator import score_node
from consortium.tree_search.tree_persistence import ensure_tree_state, save_tree_state
from consortium.tree_search.tree_state import (
    NodeStatus,
    NodeType,
    TreeNode,
    TreeSearchConfig,
    TreeSearchState,
)
from consortium.tree_search.workspace_fork import fork_workspace, promote_branch


# ---------------------------------------------------------------------------
# Experiment strategy data class
# ---------------------------------------------------------------------------

@dataclass
class ExperimentStrategy:
    """A candidate experimental approach."""

    name: str
    description: str
    prompt_directive: str  # injected into experiment_design_agent's task
    estimated_cost: str  # "low" | "medium" | "high"
    information_gain: str  # "low" | "medium" | "high"
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "prompt_directive": self.prompt_directive,
            "estimated_cost": self.estimated_cost,
            "information_gain": self.information_gain,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# LLM-based experiment strategy generation
# ---------------------------------------------------------------------------

_EXPERIMENT_STRATEGY_SYSTEM = """\
You are an experimental research strategist.  Given a research question and any
prior experimental results, propose exactly {n} distinct experimental designs.

Each design must be genuinely different — not just cosmetic parameter changes.
Good dimensions of variation include:

- Different experimental methodologies (simulation vs analytical vs numerical)
- Different datasets or problem instances
- Different baselines for comparison
- Ablation studies isolating different factors
- Scaling experiments (varying key parameters over ranges)
- Robustness checks (perturbations, noise, edge cases)
- Different metrics or evaluation criteria

Respond with a JSON array of exactly {n} objects, each with fields:
  name               — short slug (e.g. "scaling_hidden_dim")
  description        — 2-3 sentence summary of the experiment
  prompt_directive   — a precise instruction paragraph for an experiment_design
                       agent specifying exactly what to design, run, and measure.
                       Include which variables to vary, which to hold fixed,
                       what baselines to compare against, and what metrics to report.
  estimated_cost     — one of "low", "medium", "high" (computational cost)
  information_gain   — one of "low", "medium", "high" (expected insight value)
  rationale          — why this experiment would strengthen the research

Return ONLY the JSON array, no markdown fences or commentary.
"""


def generate_experiment_strategies(
    research_question: str,
    *,
    n: int = 3,
    prior_results: Optional[str] = None,
    theory_predictions: Optional[str] = None,
    failure_memory: Optional[Any] = None,
    model: str = "claude-sonnet-4-6",
) -> list[ExperimentStrategy]:
    """Ask an LLM to propose *n* distinct experiment designs.

    Parameters
    ----------
    research_question : str
        The research question or hypothesis to test experimentally.
    n : int
        Number of strategies to generate.
    prior_results : str, optional
        Summary of any previous experimental results.
    theory_predictions : str, optional
        Theoretical predictions that experiments should validate.
    failure_memory : FailureMemory, optional
        Cross-branch failure records to avoid repeating.
    model : str
        LLM model ID for the strategy generation call.
    """
    user_parts = [f"## Research Question\n{research_question}"]

    if theory_predictions:
        user_parts.append(f"## Theoretical Predictions to Validate\n{theory_predictions}")
    if prior_results:
        user_parts.append(f"## Prior Experimental Results\n{prior_results}")

    if failure_memory is not None:
        failure_text = failure_memory.format_for_strategy_prompt("experiment")
        if failure_text:
            user_parts.append(f"## FAILED EXPERIMENTS — DO NOT REPEAT\n{failure_text}")

    user_msg = "\n\n".join(user_parts)

    raw = cli_completion(
        user_msg,
        system_prompt=_EXPERIMENT_STRATEGY_SYSTEM.format(n=n),
        backend="claude",
    ).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]

    strategies_data = json.loads(raw)
    return [
        ExperimentStrategy(
            name=s["name"],
            description=s["description"],
            prompt_directive=s["prompt_directive"],
            estimated_cost=s.get("estimated_cost", "medium"),
            information_gain=s.get("information_gain", "medium"),
            rationale=s.get("rationale", ""),
        )
        for s in strategies_data[:n]
    ]


# ---------------------------------------------------------------------------
# Context loading helpers
# ---------------------------------------------------------------------------

def _load_research_question(workspace_dir: str) -> str:
    """Extract the research question from the workspace."""
    # Try research_plan.json first
    plan_path = os.path.join(workspace_dir, "paper_workspace", "research_plan.json")
    if os.path.exists(plan_path):
        try:
            with open(plan_path) as f:
                plan = json.load(f)
            rq = plan.get("research_question", "")
            if rq:
                return rq
        except Exception:
            pass

    # Fall back to agent_task from state (will be passed in)
    return ""


def _load_theory_predictions(workspace_dir: str) -> Optional[str]:
    """Load theory track results that experiments should validate."""
    claim_graph_path = os.path.join(workspace_dir, "math_workspace", "claim_graph.json")
    if not os.path.exists(claim_graph_path):
        return None
    try:
        with open(claim_graph_path) as f:
            cg = json.load(f)
        predictions = []
        for claim in cg.get("claims", []):
            if claim.get("status") in ("accepted", "verified_symbolic", "verified_numeric"):
                predictions.append(
                    f"- {claim['id']}: {claim.get('statement', '')[:200]}"
                )
        if predictions:
            return "Verified theoretical claims:\n" + "\n".join(predictions)
    except Exception:
        pass
    return None


def _load_prior_experiment_results(workspace_dir: str) -> Optional[str]:
    """Load summary of any prior experimental results."""
    results_dir = os.path.join(workspace_dir, "paper_workspace", "results")
    if not os.path.isdir(results_dir):
        return None
    summaries = []
    for fname in sorted(os.listdir(results_dir)):
        fpath = os.path.join(results_dir, fname)
        if os.path.isfile(fpath) and fname.endswith((".json", ".md", ".txt")):
            try:
                with open(fpath) as f:
                    content = f.read()[:500]
                summaries.append(f"### {fname}\n{content}")
            except Exception:
                continue
    if summaries:
        return "\n\n".join(summaries[:5])  # cap at 5 files
    return None


# ---------------------------------------------------------------------------
# Branch creation for experiments
# ---------------------------------------------------------------------------

def create_experiment_branches(
    tree_state: TreeSearchState,
    workspace_dir: str,
    research_question: str,
    *,
    model: str = "claude-sonnet-4-6",
    failure_memory: Optional[Any] = None,
) -> list[TreeNode]:
    """Generate experiment design branches and add them to the tree.

    Unlike proof branches (one per claim), experiment branches are generated
    once for the overall research question. Each branch represents a different
    experimental design / methodology.
    """
    config = tree_state.config
    experiment_claim_id = "experiment_main"

    # Skip if active branches already exist
    existing = [
        n for n in tree_state.nodes.values()
        if n.claim_id == experiment_claim_id
        and n.node_type == NodeType.EXPERIMENT_DESIGN
        and n.status not in (NodeStatus.FAILED, NodeStatus.PRUNED)
    ]
    if existing:
        return []

    theory_predictions = _load_theory_predictions(workspace_dir)
    prior_results = _load_prior_experiment_results(workspace_dir)

    strategies = generate_experiment_strategies(
        research_question,
        n=config.max_breadth,
        prior_results=prior_results,
        theory_predictions=theory_predictions,
        failure_memory=failure_memory,
        model=model,
    )

    nodes: list[TreeNode] = []
    root = tree_state.get_root()
    parent_id = root.id if root else None

    per_branch_budget = config.budget_fraction / max(config.max_breadth, 1)

    for strategy in strategies:
        node_id = TreeSearchState.make_node_id(f"exp_{strategy.name}")
        branch_dir = fork_workspace(workspace_dir, node_id)

        node = TreeNode(
            id=node_id,
            node_type=NodeType.EXPERIMENT_DESIGN,
            parent_id=parent_id,
            claim_id=experiment_claim_id,
            strategy_description=strategy.description,
            workspace_path=branch_dir,
            depth=(tree_state.nodes[parent_id].depth + 1) if parent_id and parent_id in tree_state.nodes else 1,
            budget_cap_usd=per_branch_budget,
            metadata={
                "strategy_name": strategy.name,
                "prompt_directive": strategy.prompt_directive,
                "estimated_cost": strategy.estimated_cost,
                "information_gain": strategy.information_gain,
                "rationale": strategy.rationale,
            },
        )
        # Score using experiment-adapted weights
        node.score = _score_experiment_node(node, tree_state, strategy)
        tree_state.add_node(node)
        tree_state.budget_allocations[node_id] = per_branch_budget
        nodes.append(node)

    save_tree_state(tree_state, workspace_dir)
    return nodes


def _score_experiment_node(
    node: TreeNode,
    tree_state: TreeSearchState,
    strategy: ExperimentStrategy,
) -> float:
    """Score an experiment node based on information gain, cost, and diversity.

    Experiment scoring differs from proof scoring:
    - W_INFO_GAIN: expected information gain (does this distinguish hypotheses?)
    - W_COST_EFFICIENCY: inverse of computational cost
    - W_DIVERSITY: complement to existing experiment designs
    """
    W_INFO_GAIN = 0.40
    W_COST_EFF = 0.25
    W_DIVERSITY = 0.20
    W_DEPTH_PENALTY = 0.15

    # Information gain score
    gain_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
    info_score = gain_map.get(strategy.information_gain, 0.5)

    # Cost efficiency (lower cost = higher score)
    cost_map = {"low": 0.9, "medium": 0.6, "high": 0.3}
    cost_score = cost_map.get(strategy.estimated_cost, 0.5)

    # Diversity: penalize if similar strategy names exist
    existing_names = {
        n.metadata.get("strategy_name", "")
        for n in tree_state.nodes.values()
        if n.node_type == NodeType.EXPERIMENT_DESIGN and n.id != node.id
    }
    name_words = set(strategy.name.lower().split("_"))
    overlap = 0
    for en in existing_names:
        en_words = set(en.lower().split("_"))
        overlap = max(overlap, len(name_words & en_words) / max(len(name_words), 1))
    diversity_score = 1.0 - overlap

    # Depth penalty
    depth_penalty = max(0, 1.0 - node.depth * 0.2)

    return (
        W_INFO_GAIN * info_score
        + W_COST_EFF * cost_score
        + W_DIVERSITY * diversity_score
        + W_DEPTH_PENALTY * depth_penalty
    )


# ---------------------------------------------------------------------------
# Experiment tree search controller
# ---------------------------------------------------------------------------

def build_experiment_tree_controller(
    design_node: Callable,
    experimentation_node: Callable,
    verification_node: Callable,
    workspace_dir: str,
    tree_config: TreeSearchConfig,
    *,
    model_id: str = "claude-sonnet-4-6",
    counsel_models: Optional[List[Any]] = None,
    model_specs: Optional[List[dict]] = None,
    design_descriptor: Optional[_AgentDescriptor] = None,
    experimentation_descriptor: Optional[_AgentDescriptor] = None,
    verification_descriptor: Optional[_AgentDescriptor] = None,
    adversarial_verification: bool = False,
    adversarial_model: Optional[Any] = None,
) -> Callable:
    """Build a LangGraph node that runs tree search for the experiment track.

    Pipeline per branch: experiment_design → experimentation → verification.
    Branches represent different experimental designs / methodologies.
    The best-performing experiment branch is promoted.
    """

    def experiment_tree_search_node(state: dict) -> dict:
        ws = state.get("workspace_dir", workspace_dir)
        tree_state = ensure_tree_state(ws, tree_config)

        # Extract research question from state
        research_question = (
            state.get("agent_task", "")
            or _load_research_question(ws)
            or "Conduct experiments to validate the theoretical predictions."
        )

        max_iterations = tree_config.max_depth * tree_config.max_breadth
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Create experiment branches if none exist
            branches = create_experiment_branches(
                tree_state, ws, research_question, model=model_id,
            )

            # Select branches for execution
            pending = tree_state.get_frontier()
            # Filter to only experiment nodes
            pending = [
                n for n in pending
                if n.node_type in (
                    NodeType.EXPERIMENT_DESIGN,
                    NodeType.HYPERPARAMETER_VARIANT,
                    NodeType.ABLATION_STUDY,
                )
            ]

            if not pending:
                break

            selected = pending[:tree_config.max_parallel]

            # Execute branches in parallel
            results = _execute_experiment_branches_parallel(
                selected,
                design_node=design_node,
                experimentation_node=experimentation_node,
                verification_node=verification_node,
                state=state,
                tree_state=tree_state,
                workspace_dir=ws,
                max_parallel=tree_config.max_parallel,
                tree_config=tree_config,
                counsel_models=counsel_models,
                model_specs=model_specs,
                design_descriptor=design_descriptor,
                experimentation_descriptor=experimentation_descriptor,
                verification_descriptor=verification_descriptor,
                adversarial_verification=adversarial_verification,
                adversarial_model=adversarial_model,
            )

            # Process results — promote best, create variants for failures
            any_succeeded = False
            for node, success, failure_reason in results:
                _process_experiment_result(
                    node, success, tree_state, ws,
                    failure_reason=failure_reason, model=model_id,
                )
                if success:
                    any_succeeded = True

            if any_succeeded:
                break  # Got a good experiment — promote and move on

        save_tree_state(tree_state, ws)

        summary = tree_state.summary()
        return {
            "agent_outputs": {
                "experiment_tree_search_controller": json.dumps(summary, indent=2),
            },
            "tree_state_path": os.path.join(ws, "tree_search_state.json"),
            "agent_task": None,
        }

    experiment_tree_search_node.__name__ = "experiment_tree_search_controller"
    return experiment_tree_search_node


def _execute_experiment_branches_parallel(
    branches: list[TreeNode],
    *,
    design_node: Callable,
    experimentation_node: Callable,
    verification_node: Callable,
    state: dict,
    tree_state: TreeSearchState,
    workspace_dir: str,
    max_parallel: int = 6,
    tree_config: Optional[TreeSearchConfig] = None,
    counsel_models: Optional[List[Any]] = None,
    model_specs: Optional[List[dict]] = None,
    design_descriptor: Optional[_AgentDescriptor] = None,
    experimentation_descriptor: Optional[_AgentDescriptor] = None,
    verification_descriptor: Optional[_AgentDescriptor] = None,
    adversarial_verification: bool = False,
    adversarial_model: Optional[Any] = None,
) -> list[tuple[TreeNode, bool, str]]:
    """Execute experiment branches in parallel.

    Each branch runs: experiment_design → experimentation → verification.
    """
    results: list[tuple[TreeNode, bool, str]] = []
    _tree_config = tree_config or TreeSearchConfig()

    def _run_branch(node: TreeNode) -> tuple[TreeNode, bool, str]:
        node.mark_running()
        save_tree_state(tree_state, workspace_dir)

        strategy_directive = node.metadata.get("prompt_directive", "")
        branch_task = (
            f"[TREE SEARCH BRANCH — Experiment: {node.metadata.get('strategy_name', 'default')}]\n\n"
            f"Design and prepare the following experiment:\n\n"
            f"{strategy_directive}\n\n"
            f"Create a detailed experimental specification including:\n"
            f"- Variables to manipulate and measure\n"
            f"- Baselines to compare against\n"
            f"- Expected outcomes based on theory\n"
            f"- Evaluation metrics"
        )

        branch_state = {
            **state,
            "workspace_dir": node.workspace_path,
            "agent_task": branch_task,
            "active_branch_id": node.id,
        }

        use_counsel = counsel_models and _tree_config.should_counsel(node)
        counsel_label = " [counsel]" if use_counsel else ""
        print(f"[experiment_tree] branch {node.id}{counsel_label}: starting design")

        try:
            # Stage 1: Experiment design
            design_result = _run_agent_stage(
                design_descriptor, design_node, branch_state,
                node.workspace_path, node, _tree_config,
                counsel_models, model_specs,
            )
            branch_state = {**branch_state, **design_result}

            # Stage 2: Run experiments
            branch_state["agent_task"] = (
                f"Execute the experiment designed in the previous step. "
                f"Run all specified computations and collect results."
            )
            experiment_result = _run_agent_stage(
                experimentation_descriptor, experimentation_node, branch_state,
                node.workspace_path, node, _tree_config,
                counsel_models, model_specs,
            )
            branch_state = {**branch_state, **experiment_result}

            # Stage 3: Verify results
            branch_state["agent_task"] = (
                f"Verify the experimental results. Check statistical validity, "
                f"reproducibility, and alignment with theoretical predictions."
            )
            verify_result = _run_agent_stage(
                verification_descriptor, verification_node, branch_state,
                node.workspace_path, node, _tree_config,
                counsel_models, model_specs,
            )
            branch_state = {**branch_state, **verify_result}

            # Check verification output
            verifier_output = branch_state.get("agent_outputs", {}).get(
                "experiment_verification_agent", ""
            )

            # Heuristic: check if verification passed
            output_lower = verifier_output.lower()
            has_critical = "critical" in output_lower
            has_pass = any(w in output_lower for w in ["pass", "valid", "confirmed", "verified"])

            if has_critical:
                print(f"[experiment_tree] branch {node.id}: FAILED (critical issues found)")
                return (node, False, f"Verification found critical issues: {verifier_output[:500]}")

            # Check experiment artifacts exist
            results_exist = _check_experiment_artifacts(node.workspace_path)
            if not results_exist:
                print(f"[experiment_tree] branch {node.id}: FAILED (no result artifacts)")
                return (node, False, "No experimental result artifacts found")

            # Adversarial verification if enabled
            if adversarial_verification:
                print(f"[experiment_tree] branch {node.id}: cooperative PASSED, running adversarial check...")
                try:
                    from consortium.agents.experiment_verification_agent import build_node as build_adv_exp
                    adv_node_fn = build_adv_exp(
                        adversarial_model or design_node,
                        node.workspace_path,
                        adversarial=True,
                    )
                    adv_state = {
                        **branch_state,
                        "agent_task": (
                            f"ADVERSARIAL REVIEW: Challenge the experimental results. "
                            f"Find flaws in methodology, statistics, and conclusions. "
                            f"Rate issues as CRITICAL, MAJOR, or MINOR."
                        ),
                    }
                    adv_result = adv_node_fn(adv_state)
                    adv_output = adv_result.get("agent_outputs", {}).get(
                        "experiment_verification_agent", ""
                    )
                    node.metadata["adversarial_report"] = adv_output[:2000]
                    if "CRITICAL" in adv_output.upper():
                        print(f"[experiment_tree] branch {node.id}: ADVERSARIAL found CRITICAL issues")
                        return (node, False, f"Adversarial verification failed: {adv_output[:500]}")
                    else:
                        print(f"[experiment_tree] branch {node.id}: ADVERSARIAL passed")
                except Exception as adv_err:
                    print(f"[experiment_tree] branch {node.id}: adversarial error: {adv_err}")

            print(f"[experiment_tree] branch {node.id}: SUCCEEDED")
            return (node, True, "")

        except Exception as exc:
            print(f"[experiment_tree] branch {node.id}: ERROR ({exc})")
            return (node, False, f"Branch execution error: {exc}")

    with ThreadPoolExecutor(max_workers=min(len(branches), max_parallel)) as executor:
        futures = {executor.submit(_run_branch, node): node for node in branches}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                node = futures[future]
                results.append((node, False, f"Future error: {exc}"))

    return results


def _check_experiment_artifacts(workspace_path: str) -> bool:
    """Check that experiment branches produced result artifacts."""
    for subdir in ("paper_workspace/results", "paper_workspace/figures",
                    "experiment_workspace"):
        dirpath = os.path.join(workspace_path, subdir)
        if os.path.isdir(dirpath) and os.listdir(dirpath):
            return True
    return False


def _process_experiment_result(
    node: TreeNode,
    success: bool,
    tree_state: TreeSearchState,
    workspace_dir: str,
    *,
    failure_reason: str = "",
    model: str = "claude-sonnet-4-6",
) -> None:
    """Process a completed experiment branch.

    On success: promote workspace, prune siblings.
    On failure: mark failed, create variant branches if within depth limit.
    """
    if success:
        node.mark_succeeded()
        if node.claim_id:
            tree_state.mark_claim_resolved(node.claim_id)
        promote_branch(node.workspace_path, workspace_dir)
        # Prune siblings
        if node.claim_id:
            for n in list(tree_state.nodes.values()):
                if n.id != node.id and n.claim_id == node.claim_id and not n.is_terminal:
                    tree_state.prune_subtree(n.id)
    else:
        node.mark_failed()
        node.metadata["failure_reason"] = failure_reason
        # Don't create debugging branches for experiments by default —
        # the initial N designs are usually sufficient. Only create
        # variants if we're at depth 1 and all siblings also failed.
        all_siblings_failed = all(
            n.status == NodeStatus.FAILED
            for n in tree_state.nodes.values()
            if n.claim_id == node.claim_id and n.id != node.id
            and n.node_type == NodeType.EXPERIMENT_DESIGN
        )
        if all_siblings_failed and node.depth < tree_state.config.max_depth:
            # All initial designs failed — try ablation/variant branches
            _create_experiment_variant_branches(
                node, tree_state, workspace_dir, model=model,
            )

    save_tree_state(tree_state, workspace_dir)


def _create_experiment_variant_branches(
    failed_node: TreeNode,
    tree_state: TreeSearchState,
    workspace_dir: str,
    *,
    model: str = "claude-sonnet-4-6",
) -> list[TreeNode]:
    """Create variant experiment branches after all initial designs failed.

    Generates simplified/ablated designs based on what went wrong.
    """
    config = tree_state.config
    if failed_node.depth >= config.max_depth:
        return []

    # Gather failure context from all failed siblings
    failures = []
    for n in tree_state.nodes.values():
        if (n.claim_id == failed_node.claim_id
                and n.status == NodeStatus.FAILED
                and n.node_type == NodeType.EXPERIMENT_DESIGN):
            failures.append(
                f"- {n.metadata.get('strategy_name', '?')}: {n.metadata.get('failure_reason', 'unknown')[:200]}"
            )
    failure_context = "\n".join(failures) if failures else "No details available."

    research_question = _load_research_question(workspace_dir) or "Validate the theoretical predictions."

    variant_prompt = (
        f"Previous experiment designs all FAILED:\n{failure_context}\n\n"
        f"Generate simplified variant experiments that address these failures. "
        f"Consider ablation studies, simpler baselines, or different methodologies."
    )

    strategies = generate_experiment_strategies(
        research_question,
        n=min(config.max_breadth, 2),
        prior_results=variant_prompt,
        model=model,
    )

    nodes: list[TreeNode] = []
    per_branch_budget = failed_node.budget_cap_usd * 0.5

    for strategy in strategies:
        node_id = TreeSearchState.make_node_id(f"exp_variant_{strategy.name}")
        branch_dir = fork_workspace(workspace_dir, node_id)

        node = TreeNode(
            id=node_id,
            node_type=NodeType.ABLATION_STUDY,
            parent_id=failed_node.id,
            claim_id=failed_node.claim_id,
            strategy_description=f"[VARIANT] {strategy.description}",
            workspace_path=branch_dir,
            depth=failed_node.depth + 1,
            budget_cap_usd=per_branch_budget,
            metadata={
                "strategy_name": strategy.name,
                "prompt_directive": strategy.prompt_directive,
                "parent_failure_reason": failed_node.metadata.get("failure_reason", ""),
            },
        )
        node.score = _score_experiment_node(
            node, tree_state,
            ExperimentStrategy(
                name=strategy.name,
                description=strategy.description,
                prompt_directive=strategy.prompt_directive,
                estimated_cost=strategy.estimated_cost,
                information_gain=strategy.information_gain,
                rationale=strategy.rationale,
            ),
        )
        tree_state.add_node(node)
        nodes.append(node)

    save_tree_state(tree_state, workspace_dir)
    return nodes


# ---------------------------------------------------------------------------
# Enhanced experiment track subgraph with tree search
# ---------------------------------------------------------------------------

def build_tree_search_experiment_track(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    tree_config: Optional[TreeSearchConfig] = None,
    model_id: str = "claude-sonnet-4-6",
    adversarial_verification: bool = False,
) -> Any:
    """Build the experiment track subgraph with tree search integration.

    When tree search is enabled:
        experiment_literature → [TREE SEARCH CONTROLLER] → experiment_transcription

    The tree search controller internally runs parallel
    experiment_design → experimentation → verification chains.
    """
    from consortium.agents import (
        build_experiment_design_node,
        build_experiment_literature_node,
        build_experiment_transcription_node,
        build_experiment_verification_node,
        build_experimentation_node,
    )
    from consortium.pdf_summary import with_pdf_summary

    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models is not None else {}

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    # Literature review is unchanged
    graph.add_node(
        "experiment_literature_agent",
        _wrap(
            build_experiment_literature_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
            "experiment_literature_agent",
        ),
    )

    if tree_config and tree_config.enabled:
        # Build raw nodes without counsel — counsel managed per-branch
        design = build_experiment_design_node(model, workspace_dir, authorized_imports)
        experimentation = build_experimentation_node(model, workspace_dir, authorized_imports)
        verification = build_experiment_verification_node(model, workspace_dir, authorized_imports)

        # Build descriptors for counsel-capable branch execution
        from consortium.agents.experiment_design_agent import (
            get_tools as design_get_tools,
        )
        from consortium.agents.experimentation_agent import (
            get_tools as experimentation_get_tools,
        )
        from consortium.agents.experiment_verification_agent import (
            get_tools as verification_get_tools,
        )
        from consortium.prompts.experiment_design_instructions import (
            get_experiment_design_system_prompt,
        )
        from consortium.prompts.experimentation_instructions import (
            get_experimentation_system_prompt,
        )
        from consortium.prompts.experiment_verification_instructions import (
            get_experiment_verification_system_prompt,
        )
        from consortium.toolkits.model_utils import get_raw_model
        raw_model_id = get_raw_model(model)

        # Wrap get_tools with closures to match _AgentDescriptor's
        # expected signature: get_tools(workspace_dir) -> list
        def _design_tools(ws):
            return design_get_tools(ws, authorized_imports)

        def _experimentation_tools(ws):
            return experimentation_get_tools(ws, raw_model_id)

        def _verification_tools(ws):
            return verification_get_tools(ws, raw_model_id, authorized_imports)

        design_desc = _AgentDescriptor(
            agent_name="experiment_design_agent",
            get_system_prompt=get_experiment_design_system_prompt,
            get_tools=_design_tools,
            fallback_node=design,
            model=model,
            authorized_imports=authorized_imports,
        )
        experimentation_desc = _AgentDescriptor(
            agent_name="experimentation_agent",
            get_system_prompt=get_experimentation_system_prompt,
            get_tools=_experimentation_tools,
            fallback_node=experimentation,
            model=model,
            authorized_imports=authorized_imports,
        )
        verification_desc = _AgentDescriptor(
            agent_name="experiment_verification_agent",
            get_system_prompt=get_experiment_verification_system_prompt,
            get_tools=_verification_tools,
            fallback_node=verification,
            model=model,
            authorized_imports=authorized_imports,
        )

        tree_controller = build_experiment_tree_controller(
            design_node=design,
            experimentation_node=experimentation,
            verification_node=verification,
            workspace_dir=workspace_dir,
            tree_config=tree_config,
            model_id=model_id,
            counsel_models=counsel_models,
            design_descriptor=design_desc,
            experimentation_descriptor=experimentation_desc,
            verification_descriptor=verification_desc,
            adversarial_verification=adversarial_verification,
            adversarial_model=model,
        )

        graph.add_node("experiment_tree_search_controller", tree_controller)
    else:
        # Standard linear pipeline
        graph.add_node(
            "experiment_design_agent",
            _wrap(
                build_experiment_design_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
                "experiment_design_agent",
            ),
        )
        graph.add_node(
            "experimentation_agent",
            _wrap(
                build_experimentation_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
                "experimentation_agent",
            ),
        )
        graph.add_node(
            "experiment_verification_agent",
            _wrap(
                build_experiment_verification_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
                "experiment_verification_agent",
            ),
        )

    graph.add_node(
        "experiment_transcription_agent",
        _wrap(
            build_experiment_transcription_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
            "experiment_transcription_agent",
        ),
    )

    # Wire edges
    graph.set_entry_point("experiment_literature_agent")

    if tree_config and tree_config.enabled:
        graph.add_edge("experiment_literature_agent", "experiment_tree_search_controller")
        graph.add_edge("experiment_tree_search_controller", "experiment_transcription_agent")
    else:
        graph.add_edge("experiment_literature_agent", "experiment_design_agent")
        graph.add_edge("experiment_design_agent", "experimentation_agent")
        graph.add_edge("experimentation_agent", "experiment_verification_agent")
        graph.add_edge("experiment_verification_agent", "experiment_transcription_agent")

    graph.add_edge("experiment_transcription_agent", END)
    return graph.compile()
