"""LLM-based node scoring for tree search branch prioritisation.

Combines multiple signals — LLM promise assessment, claim-graph impact,
cost efficiency, depth, and diversity — into a single [0, 1] score used
by the tree manager for best-first expansion.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from consortium.cli_completion import cli_completion
from consortium.tree_search.tree_state import NodeStatus, TreeNode, TreeSearchState
from consortium.utils import infer_cli_backend


# ---------------------------------------------------------------------------
# Scoring weights  (sum to 1.0)
# ---------------------------------------------------------------------------

W_LLM_PROMISE = 0.40
W_IMPACT = 0.25
W_EFFICIENCY = 0.15
W_DEPTH = 0.10
W_DIVERSITY = 0.10


# ---------------------------------------------------------------------------
# Individual signal functions
# ---------------------------------------------------------------------------

def _llm_promise_score(
    node: TreeNode,
    *,
    model: str = "claude-sonnet-4-6",
) -> float:
    """Ask an LLM to rate the mathematical promise of a proof strategy.

    Returns a float in [0, 1].
    """
    system = (
        "You are a mathematical research evaluator. "
        "Given a proof strategy description, rate its likelihood of producing "
        "a correct, complete proof on a scale from 0.0 to 1.0. "
        "Consider mathematical rigour, feasibility, and the quality of the "
        "proposed approach. Respond with a single JSON object: "
        '{"score": <float>, "reasoning": "<brief explanation>"}.'
    )
    user = (
        f"Strategy: {node.strategy_description}\n\n"
        f"Node type: {node.node_type.value}\n"
        f"Claim ID: {node.claim_id or 'N/A'}\n"
        f"Depth in tree: {node.depth}"
    )
    if node.metadata.get("prior_gaps"):
        user += f"\n\nPrior verification gaps:\n{json.dumps(node.metadata['prior_gaps'], indent=2)}"

    try:
        raw = cli_completion(
            user,
            system_prompt=system,
            backend=infer_cli_backend(model),
            model=model,
        ).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
        data = json.loads(raw)
        return max(0.0, min(1.0, float(data.get("score", 0.5))))
    except Exception:
        return 0.5  # safe fallback


def _claim_graph_impact(
    node: TreeNode,
    claim_graph: dict[str, Any],
) -> float:
    """Fraction of total claims that are transitively downstream of *node.claim_id*.

    Higher impact = resolving this claim unblocks more downstream work.
    """
    if not node.claim_id:
        return 0.0

    claims = claim_graph.get("claims", [])
    if not claims:
        return 0.0

    total = len(claims)

    # Build reverse adjacency: claim -> set of claims that depend on it
    dependents: dict[str, set[str]] = {}
    for c in claims:
        for dep in c.get("depends_on", []):
            dependents.setdefault(dep, set()).add(c["id"])

    # BFS from claim_id to count transitive dependents
    visited: set[str] = set()
    stack = [node.claim_id]
    while stack:
        cid = stack.pop()
        if cid in visited:
            continue
        visited.add(cid)
        stack.extend(dependents.get(cid, set()))

    # Exclude the claim itself from the count
    downstream_count = len(visited) - 1
    return downstream_count / max(total, 1)


def _cost_efficiency(node: TreeNode) -> float:
    """How much budget headroom this branch has.  1.0 = full budget, 0.0 = exhausted."""
    if node.budget_cap_usd <= 0:
        return 0.5
    return max(0.0, 1.0 - (node.cost_usd / node.budget_cap_usd))


def _depth_penalty(node: TreeNode) -> float:
    """Prefer shallower solutions.  Returns a value in (0, 1]."""
    return 0.95 ** node.depth


def _sibling_diversity(node: TreeNode, tree_state: TreeSearchState) -> float:
    """Bonus for approaches that are different from siblings.

    Uses a simple heuristic: if the node's strategy_description is short or
    absent, return 0.5 (neutral).  Otherwise, count how many siblings have
    the same node_type — fewer means more diverse.
    """
    if not node.parent_id:
        return 1.0

    siblings = tree_state.get_children(node.parent_id)
    if len(siblings) <= 1:
        return 1.0

    same_type = sum(1 for s in siblings if s.node_type == node.node_type and s.id != node.id)
    # All-different = 1.0, all-same = 0.5
    return max(0.5, 1.0 - 0.5 * (same_type / max(len(siblings) - 1, 1)))


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def score_node(
    node: TreeNode,
    tree_state: TreeSearchState,
    claim_graph: dict[str, Any],
    *,
    model: str = "claude-sonnet-4-6",
    skip_llm: bool = False,
) -> float:
    """Compute the composite score for *node*.

    Parameters
    ----------
    skip_llm : bool
        If True, use a neutral 0.5 for the LLM promise component (useful
        for batch rescoring where LLM calls are too expensive).
    """
    llm = 0.5 if skip_llm else _llm_promise_score(node, model=model)
    impact = _claim_graph_impact(node, claim_graph)
    efficiency = _cost_efficiency(node)
    depth = _depth_penalty(node)
    diversity = _sibling_diversity(node, tree_state)

    score = (
        W_LLM_PROMISE * llm
        + W_IMPACT * impact
        + W_EFFICIENCY * efficiency
        + W_DEPTH * depth
        + W_DIVERSITY * diversity
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# Score calibration — feedback loop on LLM promise accuracy
# ---------------------------------------------------------------------------

class ScoreCalibrator:
    """Track predicted scores vs actual outcomes to calibrate LLM promise scoring.

    Records (predicted_score, actual_success) pairs and computes a correction
    factor.  When the LLM systematically overestimates, the correction factor
    will be < 1.0; when it underestimates, > 1.0.

    Persisted to ``{workspace_dir}/score_calibration.json``.
    """

    def __init__(self) -> None:
        self.outcomes: list[tuple[float, bool]] = []

    def record_outcome(self, predicted: float, success: bool) -> None:
        self.outcomes.append((predicted, success))

    @property
    def correction_factor(self) -> float:
        """Ratio of actual success rate to average predicted score.

        Returns 1.0 when no data or when predictions are well-calibrated.
        """
        if len(self.outcomes) < 3:
            return 1.0
        avg_predicted = sum(p for p, _ in self.outcomes) / len(self.outcomes)
        actual_rate = sum(1 for _, s in self.outcomes if s) / len(self.outcomes)
        if avg_predicted < 0.01:
            return 1.0
        return actual_rate / avg_predicted

    def adjusted_weights(self) -> dict[str, float]:
        """Return adjusted scoring weights based on calibration.

        If the LLM is poorly calibrated (correction far from 1.0), reduce
        W_LLM_PROMISE and redistribute to W_IMPACT and W_EFFICIENCY.
        """
        cf = self.correction_factor
        calibration_quality = 1.0 - min(abs(cf - 1.0), 0.5)  # 1.0 = perfect, 0.5 = terrible

        # Scale LLM weight by calibration quality
        llm_w = W_LLM_PROMISE * calibration_quality
        redistribution = W_LLM_PROMISE - llm_w
        return {
            "W_LLM_PROMISE": round(llm_w, 4),
            "W_IMPACT": round(W_IMPACT + redistribution * 0.6, 4),
            "W_EFFICIENCY": round(W_EFFICIENCY + redistribution * 0.4, 4),
            "W_DEPTH": W_DEPTH,
            "W_DIVERSITY": W_DIVERSITY,
        }

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"outcomes": [{"predicted": p, "success": s} for p, s in self.outcomes]},
                f, indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "ScoreCalibrator":
        import os
        cal = cls()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                cal.outcomes = [(o["predicted"], o["success"]) for o in data.get("outcomes", [])]
            except Exception:
                pass
        return cal


def rescore_all(
    tree_state: TreeSearchState,
    claim_graph: dict[str, Any],
    *,
    model: str = "claude-sonnet-4-6",
    skip_llm: bool = True,
) -> None:
    """Re-score all pending nodes in *tree_state* (in-place).

    Uses ``skip_llm=True`` by default to avoid expensive LLM calls during
    batch rescoring.  The initial LLM promise score is computed once at node
    creation time.
    """
    for node in tree_state.get_pending_nodes():
        node.score = score_node(
            node,
            tree_state,
            claim_graph,
            model=model,
            skip_llm=skip_llm,
        )
